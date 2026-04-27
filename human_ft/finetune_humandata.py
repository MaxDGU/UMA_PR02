#!/usr/bin/env python3
"""
Fine-tune an already-trained checkpoint on human fraction traces.

Pipeline:
1) Resolve a checkpoint directory (default: <model_path>/best).
2) Load model + tokenizer.
3) Fine-tune on train CSV (default: data/human_ft/data_train_nlp.csv).
4) Evaluate validation NLL on explicit val CSV each epoch.
5) Save best/last/final checkpoints and metric artifacts.
6) Optionally run SP2013 final-answer evaluation during training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_REPLICATION_DIR = ROOT / "results" / "transformer_replication"
if str(TRANSFORMER_REPLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_REPLICATION_DIR))

DEFAULT_SP2013_CSV = str(TRANSFORMER_REPLICATION_DIR / ".." / "UMA_replication" / "sp2013.csv")
DEFAULT_SP2013_GRID_G = "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10"
DEFAULT_SP2013_GRID_D = "0.1,0.3,0.5,0.7,0.9"
DEFAULT_SP2013_GRID_RT = "3,4,5,6"
DEFAULT_SP2013_GRID_ICE = "0,25,50,75,100"


_RUNTIME_IMPORT_ERROR: Exception | None = None

try:
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    from train_transformer_hf import (
        DEFAULT_SP2013_CSV,
        DEFAULT_SP2013_GRID_D,
        DEFAULT_SP2013_GRID_G,
        DEFAULT_SP2013_GRID_ICE,
        DEFAULT_SP2013_GRID_RT,
        NLPtracesDataset,
        build_masked_student_prompt,
        build_plain_prompt,
        build_student_id_prompt,
        build_student_prompt,
        build_collate_fn,
        build_training_state,
        evaluate,
        evaluate_sp2013_final_answer,
        move_batch,
        parse_float_list_arg,
        pick_amp_dtype,
        save_checkpoint,
        set_seed,
    )
except (ImportError, SyntaxError) as exc:
    _RUNTIME_IMPORT_ERROR = exc


def _ensure_runtime_dependencies() -> None:
    if _RUNTIME_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Human fine-tuning requires the training dependencies used by "
        "results/transformer_replication/train_transformer_hf.py "
        "(including pandas, torch, and transformers)."
    ) from _RUNTIME_IMPORT_ERROR


def parse_positive_int_list_arg(value: str, *, arg_name: str) -> List[int]:
    text = str(value or "").strip()
    if text == "":
        return []

    values: List[int] = []
    for item in text.split(","):
        token = item.strip()
        if token == "":
            raise ValueError(f"Invalid --{arg_name} value: {value!r}")
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid value in --{arg_name}: {token!r}") from exc
        if parsed <= 0:
            raise ValueError(f"--{arg_name} expects positive integers, got {parsed}")
        values.append(parsed)
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained checkpoint on human NLP traces.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help=(
            "Checkpoint/run directory used when --init_strategy=checkpoint. "
            "If this directory does not directly contain model weights, the script tries "
            "<model_path>/<checkpoint_subdir>."
        ),
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Base model/config source used for tokenizer loading and non-checkpoint initialization.",
    )
    parser.add_argument(
        "--init_strategy",
        choices=["checkpoint", "pretrained", "random"],
        default="checkpoint",
        help="Initialization source for human-data finetuning.",
    )
    parser.add_argument(
        "--checkpoint_subdir",
        type=str,
        default="best",
        help="Subdirectory to use when --model_path points to a run directory (default: best).",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join("data", "human_ft", "data_train_nlp.csv"),
        help="Train CSV (NLP columns preferred).",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=os.path.join("data", "human_ft", "data_val_nlp.csv"),
        help="Validation CSV used for NLL tracking (NLP columns preferred).",
    )
    parser.add_argument("--prompt_col", type=str, default="instruction_nl")
    parser.add_argument("--response_col", type=str, default="response_nl")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--instruction_style",
        choices=["existing", "plain", "masked_student", "student_id", "student_values"],
        default="plain",
        help="How to construct training prompts after CSV normalization.",
    )
    parser.add_argument(
        "--panel_csv",
        type=str,
        default="",
        help=(
            "Optional parameter panel CSV used to expand human rows over a fixed tuple set "
            "when --instruction_style=student_values."
        ),
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--train_on_prompt", action="store_true")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional optimizer-step cap for smoke tests. Use 0 to train for all epochs.",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.0,
        help="Weight for response-token KL(policy || frozen reference). Use 0 to disable.",
    )
    parser.add_argument(
        "--kl_reference_model_path",
        type=str,
        default="",
        help=(
            "Optional run/checkpoint directory for the frozen KL reference. "
            "Defaults to the resolved source checkpoint when --kl_beta > 0."
        ),
    )
    parser.add_argument(
        "--kl_reference_checkpoint_subdir",
        type=str,
        default="",
        help=(
            "Subdirectory to use when --kl_reference_model_path points to a run directory. "
            "Defaults to --checkpoint_subdir."
        ),
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed-precision mode on CUDA.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load from local HF cache only.",
    )
    parser.add_argument(
        "--best_by",
        choices=["val_nll", "sp2013_acc"],
        default="val_nll",
        help="Best-checkpoint selection criterion.",
    )

    parser.add_argument("--sp2013_csv", type=str, default=DEFAULT_SP2013_CSV)
    parser.add_argument("--sp2013_max_new_tokens", type=int, default=192)
    parser.add_argument(
        "--sp2013_num_rollouts",
        type=int,
        default=1000,
        help="Number of sampled SP2013 generations per prompt for averaged accuracy.",
    )
    parser.add_argument(
        "--sp2013_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic decoding for SP2013 eval.",
    )
    parser.add_argument("--sp2013_temperature", type=float, default=0.7)
    parser.add_argument("--sp2013_top_p", type=float, default=0.95)
    parser.add_argument("--sp2013_top_k", type=int, default=50)
    parser.add_argument("--sp2013_sample_seed", type=int, default=123)
    parser.add_argument(
        "--sp2013_rollout_batch_size",
        type=int,
        default=8,
        help="Number of sampled return sequences to request per SP2013 generation call.",
    )
    parser.add_argument(
        "--sp2013_progress_every",
        type=int,
        default=1000,
        help="Print SP2013 rollout progress every N sampled generations (0 disables).",
    )
    parser.add_argument(
        "--eval_sp2013",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate SP2013 final-answer accuracy after each epoch.",
    )
    parser.add_argument(
        "--eval_loaded_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate the loaded checkpoint before any finetuning.",
    )
    parser.add_argument(
        "--sp2013_use_param_grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use UMA parameter grid during SP2013 eval.",
    )
    parser.add_argument(
        "--save_epoch_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save a standalone checkpoint directory after each epoch (epoch_1, epoch_2, ...).",
    )
    parser.add_argument(
        "--save_epoch_list",
        type=str,
        default="",
        help="Optional comma-separated epoch list to save as standalone checkpoints, e.g. 1,5,25,50,100.",
    )
    parser.add_argument(
        "--save_update_list",
        type=str,
        default="",
        help="Optional comma-separated optimizer-update list to save as standalone checkpoints, e.g. 1,2,3,4,5,6.",
    )
    parser.add_argument("--sp2013_grid_g", type=str, default=DEFAULT_SP2013_GRID_G)
    parser.add_argument("--sp2013_grid_d", type=str, default=DEFAULT_SP2013_GRID_D)
    parser.add_argument("--sp2013_grid_rt", type=str, default=DEFAULT_SP2013_GRID_RT)
    parser.add_argument("--sp2013_grid_ice", type=str, default=DEFAULT_SP2013_GRID_ICE)

    args = parser.parse_args()
    args.save_epoch_list = parse_positive_int_list_arg(args.save_epoch_list, arg_name="save_epoch_list")
    args.save_update_list = parse_positive_int_list_arg(args.save_update_list, arg_name="save_update_list")
    if args.save_epoch_list and max(args.save_epoch_list) > int(args.epochs):
        raise ValueError(
            f"--save_epoch_list has epoch {max(args.save_epoch_list)} but --epochs={args.epochs}"
        )
    return args


def ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def has_model_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_config.json",
        "config.json",
    ]
    return any(os.path.exists(os.path.join(path, name)) for name in candidates)


def resolve_checkpoint_dir(model_path: str, checkpoint_subdir: str) -> str:
    model_path = os.path.abspath(model_path)
    if has_model_files(model_path):
        return model_path

    candidate = os.path.join(model_path, checkpoint_subdir)
    if has_model_files(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not resolve checkpoint directory from model_path={model_path}. "
        f"Tried '{model_path}' and '{candidate}'."
    )


def read_csv_flexible(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def pick_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in columns:
            return c
    return ""


def clean_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.strip()
    )


def normalize_subjid_token(value: object) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def normalize_to_nlp_frame(df: pd.DataFrame, prompt_col: str, response_col: str) -> pd.DataFrame:
    # Already in NLP format.
    if prompt_col in df.columns and response_col in df.columns:
        out = df.copy()
        out[prompt_col] = clean_series(out[prompt_col])
        out[response_col] = clean_series(out[response_col])
        if "prob" in out.columns:
            out["prob"] = clean_series(out["prob"])
        else:
            out["prob"] = ""
        if "subjid" in out.columns:
            out["subjid"] = out["subjid"].map(normalize_subjid_token)
        out = out[(out[prompt_col] != "") & (out[response_col] != "")].reset_index(drop=True)
        return out

    # Raw human format fallback.
    prob_col = pick_first_existing(df.columns, ["prob", "prob.1"])
    resp_col = pick_first_existing(df.columns, ["resp", "resp.1"])
    reasoning_col = pick_first_existing(df.columns, ["strategy", "Exp", "Comments"])

    missing = []
    if not prob_col:
        missing.append("prob/prob.1")
    if not resp_col:
        missing.append("resp/resp.1")
    if not reasoning_col:
        missing.append("strategy/Exp/Comments")
    if missing:
        raise ValueError(
            "Could not build NLP format from CSV. Missing columns: "
            + ", ".join(missing)
        )

    prob = clean_series(df[prob_col])
    resp = clean_series(df[resp_col])
    reasoning = clean_series(df[reasoning_col])
    keep = (prob != "") & (resp != "") & (reasoning != "")
    prob = prob[keep]
    resp = resp[keep]
    reasoning = reasoning[keep]
    prompt = "Solve this fraction problem: " + prob + "=?"
    response = reasoning + "\n### answer: " + resp

    out = df.loc[keep].copy()
    out["prob"] = prob
    out[prompt_col] = prompt
    out[response_col] = response
    out = out[(out[prompt_col] != "") & (out[response_col] != "")].reset_index(drop=True)
    return out


def rewrite_instruction_style(
    frame: pd.DataFrame,
    *,
    prompt_col: str,
    instruction_style: str,
    panel_csv: str = "",
) -> pd.DataFrame:
    if instruction_style == "existing":
        return frame

    if "prob" not in frame.columns:
        raise ValueError(
            f"--instruction_style {instruction_style} requires a 'prob' column after normalization."
        )

    out = frame.copy()
    out["prob"] = clean_series(out["prob"])
    if (out["prob"] == "").any():
        missing_rows = int((out["prob"] == "").sum())
        raise ValueError(
            f"--instruction_style {instruction_style} requires non-empty 'prob' values; "
            f"found {missing_rows} empty rows."
        )

    if instruction_style == "plain":
        out[prompt_col] = out["prob"].map(build_plain_prompt)
        return out
    if instruction_style == "masked_student":
        out[prompt_col] = out["prob"].map(build_masked_student_prompt)
        return out
    if instruction_style == "student_id":
        if "subjid" not in out.columns:
            raise ValueError("--instruction_style student_id requires a 'subjid' column after normalization.")
        out["subjid"] = out["subjid"].map(normalize_subjid_token)
        if (out["subjid"] == "").any():
            missing_rows = int((out["subjid"] == "").sum())
            raise ValueError(
                "--instruction_style student_id requires non-empty 'subjid' values; "
                f"found {missing_rows} empty rows."
            )
        out[prompt_col] = [
            build_student_id_prompt(prob=prob, subjid=subjid)
            for prob, subjid in zip(out["prob"], out["subjid"])
        ]
        return out
    if instruction_style == "student_values":
        required_cols = ["g", "d", "rt_mu", "ice"]
        missing_cols = [col for col in required_cols if col not in out.columns]
        if missing_cols:
            if not panel_csv:
                raise ValueError(
                    "--instruction_style student_values requires columns g,d,rt_mu,ice "
                    "after normalization, or an explicit --panel_csv for panel expansion."
                )
            panel_df = pd.read_csv(panel_csv)
            required_panel_cols = {"g", "d", "rt_mu", "ice"}
            panel_missing = required_panel_cols.difference(panel_df.columns)
            if panel_missing:
                raise ValueError(
                    f"Panel CSV missing required columns for student_values: {sorted(panel_missing)}"
                )
            panel_keep = ["g", "d", "rt_mu", "ice"]
            if "subjid" in panel_df.columns:
                panel_keep = ["subjid"] + panel_keep
            panel_df = panel_df.loc[:, panel_keep].drop_duplicates(subset=["g", "d", "rt_mu", "ice"]).copy()
            if "subjid" in panel_df.columns:
                panel_df = panel_df.rename(columns={"subjid": "panel_subjid"})
            out = (
                out.assign(_cross_key=1)
                .merge(panel_df.assign(_cross_key=1), on="_cross_key", how="inner")
                .drop(columns="_cross_key")
            )
        for col in required_cols:
            if out[col].isna().any():
                missing_rows = int(out[col].isna().sum())
                raise ValueError(
                    f"--instruction_style student_values requires non-empty '{col}' values; "
                    f"found {missing_rows} empty rows."
                )
        out[prompt_col] = [
            build_student_prompt(
                prob=prob,
                g=float(g),
                d=float(d),
                rt_mu=float(rt_mu),
                ice=float(ice),
            )
            for prob, g, d, rt_mu, ice in zip(out["prob"], out["g"], out["d"], out["rt_mu"], out["ice"])
        ]
        return out
    raise ValueError(f"Unknown --instruction_style: {instruction_style}")


def default_output_dir(model_path: str, base_model_name: str, init_strategy: str) -> str:
    if init_strategy == "checkpoint" and model_path.strip() != "":
        tag = os.path.basename(os.path.abspath(model_path.rstrip("/")))
    else:
        tag = os.path.basename(str(base_model_name).rstrip("/")) or "smollm2_135m"
        tag = f"{tag}_{init_strategy}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("human_ft", "runs", f"{tag}_finetune_human_{ts}")


def safe_exp(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    if math.isinf(value):
        return float("inf") if value > 0 else 0.0
    try:
        return float(math.exp(value))
    except OverflowError:
        return float("inf")


def load_checkpoint_model(checkpoint_dir: str, *, local_files_only: bool, trainable_adapter: bool = True):
    adapter_cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "LoRA adapter checkpoint detected, but `peft` is not installed."
            ) from exc
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_name_or_path = adapter_cfg.get("base_model_name_or_path", "")
        if not base_model_name_or_path:
            raise ValueError(f"adapter_config.json missing base_model_name_or_path: {adapter_cfg_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            local_files_only=local_files_only,
        )
        return PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=trainable_adapter)

    return AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        local_files_only=local_files_only,
    )


def freeze_reference_model(model) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def compute_response_sft_kl_loss(
    policy_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ref_logits: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return response-token SFT + beta * KL(policy || reference)."""
    shift_logits = policy_logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    response_mask = shift_labels.ne(-100)
    token_count_t = response_mask.sum()
    denom = token_count_t.clamp_min(1).to(dtype=shift_logits.dtype)

    nll_tokens = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    nll_sum = nll_tokens.masked_select(response_mask).sum()

    beta = float(kl_beta)
    if beta > 0:
        if ref_logits is None:
            raise ValueError("ref_logits is required when kl_beta > 0.")
        ref_shift_logits = ref_logits[:, :-1, :].contiguous().float()
        policy_log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
        kl_by_pos = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        kl_sum = kl_by_pos.masked_select(response_mask).sum()
    else:
        kl_sum = shift_logits.new_zeros(())

    sft_nll = nll_sum / denom
    token_kl = kl_sum / denom
    total_loss = sft_nll + (beta * token_kl)
    return total_loss, {
        "sft_nll": sft_nll.detach(),
        "kl": token_kl.detach(),
        "objective": total_loss.detach(),
        "nll_sum": nll_sum.detach(),
        "kl_sum": kl_sum.detach(),
        "token_count": int(token_count_t.detach().cpu().item()),
    }


def evaluate_sft_kl_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_dtype,
    *,
    ref_model=None,
    kl_beta: float = 0.0,
) -> Dict[str, Any]:
    if len(loader) == 0:
        return {
            "n_examples": 0,
            "n_tokens": 0,
            "nll": float("nan"),
            "ppl": float("nan"),
            "kl": float("nan"),
            "objective": float("nan"),
        }

    model.eval()
    if ref_model is not None:
        ref_model.eval()
    use_amp = amp_dtype is not None
    nll_sum = 0.0
    kl_sum = 0.0
    token_count = 0
    example_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch["labels"]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                policy_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                ref_logits = None
                if ref_model is not None and float(kl_beta) > 0:
                    ref_logits = ref_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                _, metrics = compute_response_sft_kl_loss(
                    policy_logits,
                    labels,
                    ref_logits=ref_logits,
                    kl_beta=kl_beta,
                )
            nll_sum += float(metrics["nll_sum"].item())
            kl_sum += float(metrics["kl_sum"].item())
            token_count += int(metrics["token_count"])
            example_count += int(labels.shape[0])

    denom = max(token_count, 1)
    nll = nll_sum / denom
    kl = kl_sum / denom
    objective = nll + (float(kl_beta) * kl)
    return {
        "n_examples": int(example_count),
        "n_tokens": int(token_count),
        "nll": float(nll),
        "ppl": safe_exp(nll),
        "kl": float(kl),
        "objective": float(objective),
    }


def build_sft_kl_history_record(
    *,
    epoch: int,
    phase: str,
    updates_done: int,
    lr: float,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    kl_beta: float,
    elapsed_sec: float,
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "phase": str(phase),
        "updates_done": int(updates_done),
        "lr": float(lr),
        "kl_beta": float(kl_beta),
        "train_n_examples": int(train_metrics["n_examples"]),
        "train_n_tokens": int(train_metrics["n_tokens"]),
        "train_nll": float(train_metrics["nll"]),
        "train_ppl": float(train_metrics["ppl"]),
        "train_kl": float(train_metrics["kl"]),
        "train_objective": float(train_metrics["objective"]),
        "val_n_examples": int(val_metrics["n_examples"]),
        "val_n_tokens": int(val_metrics["n_tokens"]),
        "val_nll": float(val_metrics["nll"]),
        "val_ppl": float(val_metrics["ppl"]),
        "val_kl": float(val_metrics["kl"]),
        "val_objective": float(val_metrics["objective"]),
        "elapsed_sec": float(elapsed_sec),
    }


def write_sft_kl_history(out_dir: str, records: Sequence[Dict[str, Any]]) -> None:
    json_path = os.path.join(out_dir, "sft_kl_history.json")
    csv_path = os.path.join(out_dir, "sft_kl_history.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, indent=2)
    pd.DataFrame(list(records)).to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    _ensure_runtime_dependencies()
    set_seed(args.seed)

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1.")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be >= 0.")
    if args.kl_beta < 0:
        raise ValueError("--kl_beta must be >= 0.")
    if args.sp2013_num_rollouts < 1:
        raise ValueError("--sp2013_num_rollouts must be >= 1.")
    if args.sp2013_rollout_batch_size < 1:
        raise ValueError("--sp2013_rollout_batch_size must be >= 1.")
    if args.best_by == "sp2013_acc" and not args.eval_sp2013:
        raise ValueError("--best_by sp2013_acc requires --eval_sp2013.")
    if args.init_strategy == "checkpoint" and args.model_path.strip() == "":
        raise ValueError("--model_path is required when --init_strategy=checkpoint.")
    if args.eval_sp2013 and args.instruction_style == "student_id":
        raise ValueError(
            "--instruction_style student_id is not supported by the in-training SP2013 evaluator; "
            "use the standalone distribution evaluator instead."
        )
    if args.eval_sp2013 and args.instruction_style == "student_values" and not args.sp2013_use_param_grid:
        raise ValueError(
            "--instruction_style student_values requires --sp2013_use_param_grid for the in-training "
            "SP2013 evaluator; use the standalone panel-grid evaluator otherwise."
        )
    if args.panel_csv and args.instruction_style != "student_values":
        raise ValueError("--panel_csv is only supported with --instruction_style student_values.")

    ensure_exists(args.train_csv, "Train CSV")
    ensure_exists(args.val_csv, "Validation CSV")
    if args.panel_csv:
        ensure_exists(args.panel_csv, "Panel CSV")

    checkpoint_dir = ""
    if args.init_strategy == "checkpoint":
        checkpoint_dir = resolve_checkpoint_dir(args.model_path, args.checkpoint_subdir)
    out_dir = args.output_dir.strip() or default_output_dir(
        args.model_path,
        args.base_model_name,
        args.init_strategy,
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    args.model_name = args.base_model_name

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    amp_dtype = pick_amp_dtype(args.amp_dtype, device)
    use_amp = amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    print("=" * 70, flush=True)
    print("HUMAN-DATA FINETUNE PIPELINE", flush=True)
    print("=" * 70, flush=True)
    print(f"init_strategy: {args.init_strategy}", flush=True)
    if args.init_strategy == "checkpoint":
        print(f"checkpoint_dir: {checkpoint_dir}", flush=True)
    else:
        print(f"base_model_name: {args.base_model_name}", flush=True)
    print(f"train_csv: {args.train_csv}", flush=True)
    print(f"val_csv:   {args.val_csv}", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"amp_dtype: {str(amp_dtype) if amp_dtype is not None else 'none'}", flush=True)

    train_df_raw = read_csv_flexible(args.train_csv)
    val_df_raw = read_csv_flexible(args.val_csv)
    train_df = normalize_to_nlp_frame(train_df_raw, args.prompt_col, args.response_col)
    val_df = normalize_to_nlp_frame(val_df_raw, args.prompt_col, args.response_col)
    train_df = rewrite_instruction_style(
        train_df,
        prompt_col=args.prompt_col,
        instruction_style=args.instruction_style,
        panel_csv=args.panel_csv,
    )
    val_df = rewrite_instruction_style(
        val_df,
        prompt_col=args.prompt_col,
        instruction_style=args.instruction_style,
        panel_csv=args.panel_csv,
    )

    print(f"rows train: {len(train_df):,}", flush=True)
    print(f"rows val:   {len(val_df):,}", flush=True)
    print(f"instruction_style: {args.instruction_style}", flush=True)
    print("", flush=True)

    tokenizer_source = checkpoint_dir if args.init_strategy == "checkpoint" else args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # SmolLM2 is decoder-only; left padding is the safe choice for generation-time batching.
    tokenizer.padding_side = "left"

    # Support both full checkpoints and LoRA adapter checkpoints.
    if args.init_strategy == "random":
        config = AutoConfig.from_pretrained(
            args.base_model_name,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForCausalLM.from_config(config)
    elif args.init_strategy == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            local_files_only=args.local_files_only,
        )
    else:
        model = load_checkpoint_model(
            checkpoint_dir,
            local_files_only=args.local_files_only,
            trainable_adapter=True,
        )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    kl_ref_model = None
    kl_reference_source = ""
    if args.kl_beta > 0:
        if args.kl_reference_model_path.strip():
            ref_subdir = args.kl_reference_checkpoint_subdir.strip() or args.checkpoint_subdir
            kl_reference_source = resolve_checkpoint_dir(args.kl_reference_model_path, ref_subdir)
            kl_ref_model = load_checkpoint_model(
                kl_reference_source,
                local_files_only=args.local_files_only,
                trainable_adapter=False,
            )
        elif args.init_strategy == "checkpoint":
            kl_reference_source = checkpoint_dir
            kl_ref_model = load_checkpoint_model(
                checkpoint_dir,
                local_files_only=args.local_files_only,
                trainable_adapter=False,
            )
        else:
            kl_reference_source = args.base_model_name
            kl_ref_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name,
                local_files_only=args.local_files_only,
            )
        if len(tokenizer) > kl_ref_model.get_input_embeddings().num_embeddings:
            kl_ref_model.resize_token_embeddings(len(tokenizer))
        kl_ref_model.config.pad_token_id = tokenizer.pad_token_id
        kl_ref_model.config.use_cache = False
        freeze_reference_model(kl_ref_model)
        kl_ref_model.to(device)
        print(f"kl_beta: {args.kl_beta}", flush=True)
        print(f"kl_reference: {kl_reference_source}", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params (total/trainable): {total_params:,}/{trainable_params:,}", flush=True)

    collate = build_collate_fn(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_on_prompt=args.train_on_prompt,
    )

    train_ds = NLPtracesDataset(
        train_df[args.prompt_col].tolist(),
        train_df[args.response_col].tolist(),
    )
    val_ds = NLPtracesDataset(
        val_df[args.prompt_col].tolist(),
        val_df[args.response_col].tolist(),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    planned_updates = updates_per_epoch * args.epochs
    total_updates = min(planned_updates, args.max_steps) if args.max_steps > 0 else planned_updates
    if args.save_update_list and max(args.save_update_list) > int(total_updates):
        raise ValueError(
            f"--save_update_list has update {max(args.save_update_list)} but the run will perform only {total_updates} updates."
        )
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_updates),
    )

    sp2013_grid_g: List[float] = []
    sp2013_grid_d: List[float] = []
    sp2013_grid_rt: List[float] = []
    sp2013_grid_ice: List[float] = []
    if args.sp2013_use_param_grid:
        sp2013_grid_g = parse_float_list_arg(args.sp2013_grid_g, "sp2013_grid_g")
        sp2013_grid_d = parse_float_list_arg(args.sp2013_grid_d, "sp2013_grid_d")
        sp2013_grid_rt = parse_float_list_arg(args.sp2013_grid_rt, "sp2013_grid_rt")
        sp2013_grid_ice = parse_float_list_arg(args.sp2013_grid_ice, "sp2013_grid_ice")
    sp2013_prompt_style = (
        "student_values"
        if args.sp2013_use_param_grid
        else ("masked_student" if args.instruction_style == "masked_student" else "plain")
    )

    history: List[Dict[str, Any]] = []
    sft_kl_history: List[Dict[str, Any]] = []
    update_checkpoint_set = set(args.save_update_list)
    global_update = 0
    best_dir = os.path.join(out_dir, "best")
    last_dir = os.path.join(out_dir, "last")
    final_dir = os.path.join(out_dir, "final")
    best_val_nll = float("inf")
    best_sp2013 = float("-inf")
    best_epoch = None
    t0 = time.time()

    if args.kl_beta > 0:
        train_metrics = evaluate_sft_kl_metrics(
            model,
            train_eval_loader,
            device,
            amp_dtype,
            ref_model=kl_ref_model,
            kl_beta=args.kl_beta,
        )
        val_metrics = evaluate_sft_kl_metrics(
            model,
            val_loader,
            device,
            amp_dtype,
            ref_model=kl_ref_model,
            kl_beta=args.kl_beta,
        )
        sft_kl_record = build_sft_kl_history_record(
            epoch=0,
            phase="loaded_model",
            updates_done=0,
            lr=scheduler.get_last_lr()[0],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            kl_beta=args.kl_beta,
            elapsed_sec=time.time() - t0,
        )
        sft_kl_history.append(sft_kl_record)
        write_sft_kl_history(out_dir, sft_kl_history)
        if not args.eval_loaded_model:
            history.append(
                {
                    "epoch": 0,
                    "phase": "loaded_model",
                    "updates_done": 0,
                    "train_nll": sft_kl_record["train_nll"],
                    "val_nll": sft_kl_record["val_nll"],
                    "val_ppl": sft_kl_record["val_ppl"],
                    "train_kl": sft_kl_record["train_kl"],
                    "val_kl": sft_kl_record["val_kl"],
                    "kl_beta": float(args.kl_beta),
                    "sp2013_acc": float("nan"),
                    "sp2013_by_op": {},
                    "elapsed_sec": sft_kl_record["elapsed_sec"],
                }
            )
        print(
            f"[sft+kl epoch 00] train_nll={sft_kl_record['train_nll']:.4f} "
            f"val_nll={sft_kl_record['val_nll']:.4f} "
            f"train_kl={sft_kl_record['train_kl']:.6f} val_kl={sft_kl_record['val_kl']:.6f}",
            flush=True,
        )

    if args.eval_loaded_model:
        loaded_val_nll = evaluate(model, val_loader, device, amp_dtype, distributed=False) if len(val_ds) > 0 else float("nan")
        if math.isfinite(loaded_val_nll):
            try:
                loaded_val_ppl = float(math.exp(loaded_val_nll))
            except OverflowError:
                loaded_val_ppl = float("inf")
        else:
            loaded_val_ppl = float("nan")

        loaded_sp2013_acc = float("nan")
        loaded_sp2013_by_op: Dict[str, float] = {}
        if args.eval_sp2013:
            loaded_sp2013_metrics, loaded_sp2013_df = evaluate_sp2013_final_answer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                sp2013_csv=args.sp2013_csv,
                max_new_tokens=args.sp2013_max_new_tokens,
                use_param_grid=args.sp2013_use_param_grid,
                grid_g=sp2013_grid_g,
                grid_d=sp2013_grid_d,
                grid_rt=sp2013_grid_rt,
                grid_ice=sp2013_grid_ice,
                num_rollouts=args.sp2013_num_rollouts,
                do_sample=args.sp2013_do_sample,
                temperature=args.sp2013_temperature,
                top_p=args.sp2013_top_p,
                top_k=args.sp2013_top_k,
                sample_seed=args.sp2013_sample_seed,
                rollout_batch_size=args.sp2013_rollout_batch_size,
                progress_every=args.sp2013_progress_every,
                progress_label="sp2013_loaded_model",
                prompt_style=sp2013_prompt_style,
            )
            loaded_sp2013_acc = loaded_sp2013_metrics.get("overall_acc", float("nan"))
            loaded_sp2013_by_op = loaded_sp2013_metrics.get("by_op", {})
            loaded_sp2013_df.to_csv(os.path.join(out_dir, "sp2013_loaded_model.csv"), index=False)
            with open(os.path.join(out_dir, "sp2013_metrics_loaded_model.json"), "w", encoding="utf-8") as f:
                json.dump(loaded_sp2013_metrics, f, indent=2)

        loaded_record = {
            "epoch": 0,
            "phase": "loaded_model",
            "updates_done": 0,
            "train_nll": float("nan"),
            "val_nll": loaded_val_nll,
            "val_ppl": loaded_val_ppl,
            "sp2013_acc": loaded_sp2013_acc,
            "sp2013_by_op": loaded_sp2013_by_op,
            "elapsed_sec": time.time() - t0,
        }
        history.append(loaded_record)
        with open(os.path.join(out_dir, "loaded_model_eval.json"), "w", encoding="utf-8") as f:
            json.dump(loaded_record, f, indent=2)
        print(
            f"[loaded model] val_nll={loaded_val_nll:.4f} val_ppl={loaded_val_ppl:.2f} "
            f"sp2013_acc={loaded_sp2013_acc:.4f}",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_sft_nll_sum = 0.0
        train_kl_sum = 0.0
        train_steps = 0
        accum_counter = 0
        stop_training = False

        for step, batch in enumerate(train_loader, start=1):
            accum_counter += 1
            batch = move_batch(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if args.kl_beta > 0:
                    policy_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    with torch.no_grad():
                        ref_logits = kl_ref_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                    loss, loss_metrics = compute_response_sft_kl_loss(
                        policy_logits,
                        batch["labels"],
                        ref_logits=ref_logits,
                        kl_beta=args.kl_beta,
                    )
                else:
                    loss = model(**batch).loss
                    loss_metrics = {
                        "sft_nll": loss.detach(),
                        "kl": torch.zeros((), device=loss.device),
                        "objective": loss.detach(),
                    }
            loss_for_backward = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            train_loss_sum += loss.item()
            train_sft_nll_sum += float(loss_metrics["sft_nll"].item())
            train_kl_sum += float(loss_metrics["kl"].item())
            train_steps += 1

            take_step = (accum_counter >= args.grad_accum_steps) or (step == len(train_loader))
            if not take_step:
                continue

            if args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_counter = 0
            global_update += 1

            if global_update in update_checkpoint_set:
                update_in_epoch = global_update - ((epoch - 1) * updates_per_epoch)
                training_state = build_training_state(
                    epoch=epoch,
                    updates_in_epoch=update_in_epoch,
                    global_update=global_update,
                    best_val=best_val_nll,
                    best_sp2013=best_sp2013,
                    elapsed_sec=time.time() - t0,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )
                update_dir = os.path.join(out_dir, f"update_{global_update}")
                save_checkpoint(update_dir, model, tokenizer, vars(args), history, training_state=training_state)
                print(f"  saved update checkpoint -> {update_dir}", flush=True)

            if args.log_every > 0 and global_update % args.log_every == 0:
                avg_so_far = train_loss_sum / max(train_steps, 1)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch}/{args.epochs} | update {global_update}/{total_updates} "
                    f"| train_loss {avg_so_far:.4f} | lr {lr_now:.2e}",
                    flush=True,
                )
            if args.max_steps > 0 and global_update >= args.max_steps:
                stop_training = True
                break

        train_batch_objective = train_loss_sum / max(train_steps, 1)
        train_batch_sft_nll = train_sft_nll_sum / max(train_steps, 1)
        train_batch_kl = train_kl_sum / max(train_steps, 1)
        if args.kl_beta > 0:
            train_metrics = evaluate_sft_kl_metrics(
                model,
                train_eval_loader,
                device,
                amp_dtype,
                ref_model=kl_ref_model,
                kl_beta=args.kl_beta,
            )
            val_metrics = evaluate_sft_kl_metrics(
                model,
                val_loader,
                device,
                amp_dtype,
                ref_model=kl_ref_model,
                kl_beta=args.kl_beta,
            )
            sft_kl_record = build_sft_kl_history_record(
                epoch=epoch,
                phase="epoch_end",
                updates_done=global_update,
                lr=scheduler.get_last_lr()[0],
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                kl_beta=args.kl_beta,
                elapsed_sec=time.time() - t0,
            )
            sft_kl_history.append(sft_kl_record)
            write_sft_kl_history(out_dir, sft_kl_history)
            train_nll = float(train_metrics["nll"])
            val_nll = float(val_metrics["nll"])
            val_ppl = float(val_metrics["ppl"])
        else:
            train_nll = train_batch_objective
            val_nll = evaluate(model, val_loader, device, amp_dtype, distributed=False) if len(val_ds) > 0 else float("nan")
            val_ppl = safe_exp(val_nll)

        sp2013_acc = float("nan")
        sp2013_by_op: Dict[str, float] = {}
        if args.eval_sp2013:
            sp2013_metrics, sp2013_df = evaluate_sp2013_final_answer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                sp2013_csv=args.sp2013_csv,
                max_new_tokens=args.sp2013_max_new_tokens,
                use_param_grid=args.sp2013_use_param_grid,
                grid_g=sp2013_grid_g,
                grid_d=sp2013_grid_d,
                grid_rt=sp2013_grid_rt,
                grid_ice=sp2013_grid_ice,
                num_rollouts=args.sp2013_num_rollouts,
                do_sample=args.sp2013_do_sample,
                temperature=args.sp2013_temperature,
                top_p=args.sp2013_top_p,
                top_k=args.sp2013_top_k,
                sample_seed=args.sp2013_sample_seed + (epoch * 1000003),
                rollout_batch_size=args.sp2013_rollout_batch_size,
                progress_every=args.sp2013_progress_every,
                progress_label=f"sp2013_epoch_{epoch:02d}",
                prompt_style=sp2013_prompt_style,
            )
            sp2013_acc = sp2013_metrics.get("overall_acc", float("nan"))
            sp2013_by_op = sp2013_metrics.get("by_op", {})
            sp2013_df.to_csv(os.path.join(out_dir, f"sp2013_epoch_{epoch:02d}.csv"), index=False)
            with open(os.path.join(out_dir, f"sp2013_metrics_epoch_{epoch:02d}.json"), "w", encoding="utf-8") as f:
                json.dump(sp2013_metrics, f, indent=2)

        eval_record = {
            "epoch": epoch,
            "updates_done": global_update,
            "train_nll": train_nll,
            "val_nll": val_nll,
            "val_ppl": val_ppl,
            "train_batch_objective": train_batch_objective,
            "train_batch_sft_nll": train_batch_sft_nll,
            "train_batch_kl": train_batch_kl,
            "kl_beta": float(args.kl_beta),
            "sp2013_acc": sp2013_acc,
            "sp2013_by_op": sp2013_by_op,
            "elapsed_sec": time.time() - t0,
        }
        history.append(eval_record)

        print(
            f"[epoch {epoch:02d}] train_nll={train_nll:.4f} "
            f"val_nll={val_nll:.4f} val_ppl={val_ppl:.2f} "
            f"train_batch_kl={train_batch_kl:.6f} sp2013_acc={sp2013_acc:.4f}",
            flush=True,
        )

        if args.best_by == "sp2013_acc":
            improved = math.isfinite(sp2013_acc) and (sp2013_acc > best_sp2013)
        else:
            improved = math.isfinite(val_nll) and (val_nll < best_val_nll)

        if improved:
            if math.isfinite(val_nll):
                best_val_nll = val_nll
            if math.isfinite(sp2013_acc):
                best_sp2013 = sp2013_acc
            best_epoch = epoch
            save_checkpoint(best_dir, model, tokenizer, vars(args), history)
            print(f"  saved best checkpoint -> {best_dir}", flush=True)

        training_state = build_training_state(
            epoch=epoch,
            updates_in_epoch=updates_per_epoch,
            global_update=global_update,
            best_val=best_val_nll,
            best_sp2013=best_sp2013,
            elapsed_sec=time.time() - t0,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        save_checkpoint(last_dir, model, tokenizer, vars(args), history, training_state=training_state)
        print(f"  saved last checkpoint -> {last_dir}", flush=True)
        should_save_epoch_checkpoint = args.save_epoch_checkpoints or epoch in set(args.save_epoch_list)
        if should_save_epoch_checkpoint:
            epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
            save_checkpoint(epoch_dir, model, tokenizer, vars(args), history, training_state=training_state)
            print(f"  saved epoch checkpoint -> {epoch_dir}", flush=True)
        if stop_training:
            print(f"stopping early after max_steps={args.max_steps}", flush=True)
            break

    completed_epoch = int(history[-1]["epoch"]) if history else 0
    final_state = build_training_state(
        epoch=completed_epoch,
        updates_in_epoch=updates_per_epoch,
        global_update=global_update,
        best_val=best_val_nll,
        best_sp2013=best_sp2013,
        elapsed_sec=time.time() - t0,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    save_checkpoint(final_dir, model, tokenizer, vars(args), history, training_state=final_state)
    print(f"saved final checkpoint -> {final_dir}", flush=True)

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        best_sp2013_summary = best_sp2013 if math.isfinite(best_sp2013) else None
        final_sft_kl = sft_kl_history[-1] if sft_kl_history else {}
        json.dump(
            {
                "best_by": args.best_by,
                "best_val_nll": best_val_nll,
                "best_sp2013_acc": best_sp2013_summary,
                "best_epoch": best_epoch,
                "epochs": args.epochs,
                "completed_epoch": completed_epoch,
                "updates_done": global_update,
                "max_steps": int(args.max_steps),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "instruction_style": args.instruction_style,
                "panel_csv": args.panel_csv or None,
                "kl_beta": float(args.kl_beta),
                "kl_reference_source": kl_reference_source or None,
                "save_epoch_list": list(args.save_epoch_list),
                "save_update_list": list(args.save_update_list),
                "final_train_nll": final_sft_kl.get("train_nll"),
                "final_val_nll": final_sft_kl.get("val_nll"),
                "final_train_kl": final_sft_kl.get("train_kl"),
                "final_val_kl": final_sft_kl.get("val_kl"),
                "panel_tuple_count": int(train_df[["g", "d", "rt_mu", "ice"]].drop_duplicates().shape[0])
                if {"g", "d", "rt_mu", "ice"}.issubset(train_df.columns)
                else None,
                "init_strategy": args.init_strategy,
                "base_model_name": args.base_model_name,
                "checkpoint_source": checkpoint_dir if args.init_strategy == "checkpoint" else args.base_model_name,
            },
            f,
            indent=2,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
