#!/usr/bin/env python3
"""
Fine-tune a pretrained causal LM on NLP-translated UMA traces.

Default backbone:
  HuggingFaceTB/SmolLM2-135M

Supports:
  - single GPU / CPU
  - multi-GPU single node
  - multi-node DDP via torchrun

Expected input data:
  results/transformer_replication/uma_traces_all_nlp.csv.gz
with at least:
  - instruction_nl
  - response_nl

Example (single node, 8 GPUs):
  torchrun --nproc_per_node=8 results/transformer_replication/train_transformer_hf.py \
    --epochs 1 --batch_size 64 --grad_accum_steps 1

Example (multi-node, 8 nodes x 8 GPUs):
  torchrun --nnodes=8 --nproc_per_node=8 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    results/transformer_replication/train_transformer_hf.py --epochs 1
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import random
import re
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_CSV = os.path.join(SCRIPT_DIR, "uma_traces_all_nlp.csv.gz")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "smollm2_135m_nlp_finetune")
DEFAULT_SP2013_CSV = os.path.join(SCRIPT_DIR, "..", "UMA_replication", "sp2013.csv")
DEFAULT_SP2013_GRID_G = "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10"
DEFAULT_SP2013_GRID_D = "0.1,0.3,0.5,0.7,0.9"
DEFAULT_SP2013_GRID_RT = "3,4,5,6"
DEFAULT_SP2013_GRID_ICE = "0,25,50,75,100"
TRAINER_STATE_FILE = "trainer_state.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM2-135M on NLP UMA traces.")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--init_from_scratch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize model weights from architecture config instead of pretrained weights. "
            "Tokenizer still loads from --model_name."
        ),
    )
    parser.add_argument("--data_csv", type=str, default=DEFAULT_DATA_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt_col", type=str, default="instruction_nl")
    parser.add_argument("--response_col", type=str, default="response_nl")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-device batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Per-device eval batch size.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument(
        "--split_by_problem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split train/val/test by unique problem string to prevent cross-split problem overlap.",
    )
    parser.add_argument(
        "--problem_col",
        type=str,
        default="prob",
        help="Problem column used for contamination checks and optional problem-level split.",
    )
    parser.add_argument(
        "--drop_train_eval_problem_overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After row-level split, drop train rows whose problem appears in val/test.",
    )
    parser.add_argument(
        "--move_sp2013_rows_to_val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the initial split, move any rows whose problem appears in --sp2013_csv "
            "out of train/test and into val."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="Legacy flag: when per-eval checkpoint saving is disabled, suppress non-best epoch-end checkpoints.",
    )
    parser.add_argument(
        "--save_eval_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a full checkpoint at every scheduled evaluation point.",
    )
    parser.add_argument(
        "--save_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save/update <output_dir>/best when the selected metric improves.",
    )
    parser.add_argument(
        "--save_final_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save <output_dir>/final after training completes.",
    )
    parser.add_argument(
        "--verify_saved_checkpoint_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, reload one saved checkpoint and write a loaded-model summary artifact.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--train_on_prompt", action="store_true", help="If set, include prompt tokens in loss.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on CUDA.")
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed-precision mode on CUDA.",
    )
    parser.add_argument("--preview_samples", type=int, default=3, help="Generate N preview samples after training.")
    parser.add_argument("--preview_max_new_tokens", type=int, default=96)
    parser.add_argument("--sp2013_csv", type=str, default=DEFAULT_SP2013_CSV)
    parser.add_argument("--sp2013_max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_sp2013", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sp2013_every", type=int, default=1, help="Evaluate SP2013 every N epochs.")
    parser.add_argument(
        "--sp2013_use_param_grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate each SP2013 problem across a UMA parameter grid and average results.",
    )
    parser.add_argument(
        "--sp2013_grid_g",
        type=str,
        default=DEFAULT_SP2013_GRID_G,
        help="Comma-separated g values for SP2013 grid eval.",
    )
    parser.add_argument(
        "--sp2013_grid_d",
        type=str,
        default=DEFAULT_SP2013_GRID_D,
        help="Comma-separated d values for SP2013 grid eval.",
    )
    parser.add_argument(
        "--sp2013_grid_rt",
        type=str,
        default=DEFAULT_SP2013_GRID_RT,
        help="Comma-separated rt_mu values for SP2013 grid eval.",
    )
    parser.add_argument(
        "--sp2013_grid_ice",
        type=str,
        default=DEFAULT_SP2013_GRID_ICE,
        help="Comma-separated ice values for SP2013 grid eval.",
    )
    parser.add_argument(
        "--evals_per_epoch",
        type=int,
        default=1,
        help="Run evaluation this many times per epoch (>=1).",
    )
    parser.add_argument(
        "--best_by",
        choices=["val_loss", "sp2013_acc", "id_val_acc"],
        default="val_loss",
        help="Checkpoint selection criterion for the best model.",
    )
    parser.add_argument(
        "--eval_id_final_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate in-distribution generated final-answer accuracy on held-out NLP split.",
    )
    parser.add_argument(
        "--id_eval_split",
        choices=["val", "test", "both"],
        default="val",
        help="Which held-out split(s) to use for in-distribution final-answer eval.",
    )
    parser.add_argument(
        "--id_eval_size",
        type=int,
        default=64,
        help="Sample size per split for in-distribution final-answer eval (<=0 means full split).",
    )
    parser.add_argument("--id_eval_max_new_tokens", type=int, default=64)
    parser.add_argument("--id_eval_every", type=int, default=1, help="Evaluate in-distribution answer metrics every N epochs.")
    parser.add_argument("--id_eval_seed", type=int, default=123)
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Checkpoint dir to resume from; use 'auto' for <output_dir>/last.",
    )
    parser.add_argument(
        "--save_last_every_updates",
        type=int,
        default=0,
        help="If >0, save resumable checkpoint every N optimizer updates.",
    )
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LoRA fine-tuning (required for larger backbones like 7B).",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument(
        "--id_eval_target",
        choices=["response", "true"],
        default="response",
        help="ID final-answer target: response_nl answer token (response) or mathematically true answer from prob (true).",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load HF model/tokenizer from local cache only (avoids network calls on compute nodes).",
    )
    parser.add_argument(
        "--check_format_consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate prompt format consistency across train/val/test and ensure SP2013 eval "
            "uses the same prompt style as training."
        ),
    )
    parser.add_argument(
        "--train_prompt_student_mode",
        choices=["none", "always", "dropout"],
        default="none",
        help=(
            "Optionally rewrite training/validation/test prompts at load time to include a "
            "<student> ... </student> prefix derived from g/d/rt_mu/ice."
        ),
    )
    parser.add_argument(
        "--train_prompt_student_dropout_prob",
        type=float,
        default=0.5,
        help=(
            "When --train_prompt_student_mode=dropout, keep plain no-prefix prompts with this "
            "probability and add the student prefix to the remaining rows."
        ),
    )
    parser.add_argument(
        "--train_prompt_student_dropout_seed",
        type=int,
        default=0,
        help="Seed mixed into deterministic per-row hashing for prompt-prefix dropout.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> Tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        pg_device = local_rank if backend == "nccl" else None
        dist.init_process_group(backend=backend, init_method="env://", device_id=pg_device)

    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size, local_rank, distributed


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        # Avoid a teardown barrier here. Rank 0 can spend substantial time in
        # post-training single-rank verification, and nonzero ranks should not
        # sit in a collective waiting for it.
        dist.destroy_process_group()


def rank0_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


class NLPtracesDataset(Dataset):
    def __init__(self, prompts: Sequence[str], responses: Sequence[str]):
        self.prompts = list(prompts)
        self.responses = list(responses)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.prompts[idx], self.responses[idx]


def build_collate_fn(tokenizer: AutoTokenizer, max_length: int, train_on_prompt: bool):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    def collate(batch: Sequence[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        attention_list: List[List[int]] = []

        for prompt, response in batch:
            prompt = str(prompt).strip()
            response = str(response).strip()

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            if eos_id is not None:
                response_ids = response_ids + [eos_id]

            max_resp = max_length - len(prompt_ids)
            if max_resp <= 0:
                keep_prompt = max(1, max_length // 2)
                prompt_ids = prompt_ids[-keep_prompt:]
                max_resp = max_length - len(prompt_ids)
            response_ids = response_ids[: max(1, max_resp)]

            ids = prompt_ids + response_ids
            if train_on_prompt:
                labels = ids.copy()
            else:
                labels = ([-100] * len(prompt_ids)) + response_ids.copy()

            if len(ids) > max_length:
                ids = ids[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(ids)
            labels_list.append(labels)
            attention_list.append([1] * len(ids))

        max_len_batch = max(len(ids) for ids in input_ids_list)
        bsz = len(input_ids_list)

        input_ids = torch.full((bsz, max_len_batch), fill_value=pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len_batch), fill_value=-100, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len_batch), dtype=torch.long)

        for i, (ids, lbs, attn) in enumerate(zip(input_ids_list, labels_list, attention_list)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(lbs, dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(attn, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def pick_amp_dtype(mode: str, device: torch.device):
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def distributed_scalar_mean(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return float(value)
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def distributed_barrier(distributed: bool, device: Optional[torch.device] = None) -> None:
    if not distributed:
        return
    if device is not None and device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
        return
    dist.barrier()


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_dtype,
    distributed: bool,
) -> float:
    if len(loader) == 0:
        return float("nan")

    model.eval()
    loss_sum = 0.0
    count = 0
    use_amp = amp_dtype is not None
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(**batch).loss
            loss_sum += loss.item()
            count += 1

    local = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    denom = max(float(local[1].item()), 1.0)
    return float(local[0].item() / denom)


def save_checkpoint(
    out_dir: str,
    model,
    tokenizer: AutoTokenizer,
    train_args: Dict,
    history: List[Dict],
    training_state: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    to_save = model.module if hasattr(model, "module") else model
    to_save.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(train_args, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if training_state is not None:
        torch.save(training_state, os.path.join(out_dir, TRAINER_STATE_FILE))


def parse_csv_list(text: str) -> List[str]:
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def build_eval_checkpoint_dir(output_dir: str, epoch: int, update_in_epoch: int) -> str:
    return os.path.join(output_dir, f"eval_e{epoch:02d}_u{update_in_epoch:07d}")


def list_eval_checkpoint_dirs(output_dir: str, require_training_state: bool = False) -> List[str]:
    out: List[Tuple[int, int, str]] = []
    if not os.path.isdir(output_dir):
        return []

    pattern = re.compile(r"^eval_e(\d+)_u(\d+)$")
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is None:
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        if require_training_state and not os.path.isfile(os.path.join(path, TRAINER_STATE_FILE)):
            continue
        out.append((int(match.group(1)), int(match.group(2)), path))

    out.sort(key=lambda item: (item[0], item[1], item[2]))
    return [path for _, _, path in out]


def load_training_state_if_exists(ckpt_dir: str) -> Optional[Dict[str, Any]]:
    state_path = os.path.join(ckpt_dir, TRAINER_STATE_FILE)
    if not os.path.isfile(state_path):
        return None
    state = torch.load(state_path, map_location="cpu")
    return state if isinstance(state, dict) else None


def checkpoint_progress_key(ckpt_dir: str) -> Optional[Tuple[int, int, int, str]]:
    state = load_training_state_if_exists(ckpt_dir)
    if state is None:
        return None
    return (
        int(state.get("global_update", -1)),
        int(state.get("epoch", -1)),
        int(state.get("updates_in_epoch", -1)),
        ckpt_dir,
    )


def get_latest_resumable_checkpoint(output_dir: str) -> str:
    candidates: List[Tuple[int, int, int, str]] = []

    last_dir = os.path.join(output_dir, "last")
    key = checkpoint_progress_key(last_dir)
    if key is not None:
        candidates.append(key)

    for ckpt_dir in list_eval_checkpoint_dirs(output_dir, require_training_state=True):
        key = checkpoint_progress_key(ckpt_dir)
        if key is not None:
            candidates.append(key)

    if not candidates:
        return ""

    candidates.sort()
    return candidates[-1][3]


def get_resume_dir(output_dir: str, resume_from: str) -> str:
    raw = str(resume_from).strip()
    if raw == "":
        return ""
    if raw.lower() == "auto":
        return get_latest_resumable_checkpoint(output_dir)
    candidate = raw
    state_path = os.path.join(candidate, TRAINER_STATE_FILE)
    if os.path.isdir(candidate) and os.path.isfile(state_path):
        return candidate
    return ""


def pick_saved_checkpoint_for_reload(output_dir: str) -> str:
    best_dir = os.path.join(output_dir, "best")
    if os.path.isdir(best_dir):
        return best_dir

    eval_dirs = list_eval_checkpoint_dirs(output_dir, require_training_state=False)
    if eval_dirs:
        return eval_dirs[-1]

    final_dir = os.path.join(output_dir, "final")
    if os.path.isdir(final_dir):
        return final_dir

    last_dir = os.path.join(output_dir, "last")
    if os.path.isdir(last_dir):
        return last_dir

    return ""


def load_history_if_exists(ckpt_dir: str) -> List[Dict]:
    path = os.path.join(ckpt_dir, "history.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def build_training_state(
    epoch: int,
    updates_in_epoch: int,
    global_update: int,
    best_val: float,
    best_sp2013: float,
    elapsed_sec: float,
    optimizer,
    scheduler,
    scaler,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "updates_in_epoch": int(updates_in_epoch),
        "global_update": int(global_update),
        "best_val": float(best_val),
        "best_sp2013": float(best_sp2013),
        "elapsed_sec": float(elapsed_sec),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if scaler.is_enabled():
        state["scaler"] = scaler.state_dict()
    return state


def resolve_model_source(model_name_or_path: str, local_files_only: bool) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    if not local_files_only:
        return model_name_or_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "local_files_only=True requires `huggingface_hub` to resolve cached snapshots."
        ) from exc
    try:
        return snapshot_download(repo_id=model_name_or_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Model '{model_name_or_path}' not found in local HF cache. "
            "Predownload it on a login node or run once with --no-local_files_only."
        ) from exc


def load_eval_model(
    checkpoint_dir: str,
    base_model_name_or_path: str,
    device: torch.device,
    local_files_only: bool = True,
):
    if os.path.isfile(os.path.join(checkpoint_dir, "adapter_config.json")):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                f"LoRA checkpoint at {checkpoint_dir} requires `peft`, but it is unavailable."
            ) from exc
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            local_files_only=local_files_only,
        )
    return model.to(device)


def generate_previews(
    model,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    core = model.module if hasattr(model, "module") else model
    core.eval()
    previews = []
    for prompt in prompts:
        p = str(prompt).strip()
        encoded = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = core.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        previews.append({"prompt": p, "generation": text.strip()})
    return previews


def op_from_prob(prob: str) -> str:
    p = str(prob)
    if "+" in p:
        return "+"
    if "-" in p:
        return "-"
    if "*" in p:
        return "*"
    if ":" in p:
        return ":"
    return "?"


NUMERIC_TOKEN_RE = r"[+\-]?(?:\d+(?:/\d+)?|\d*\.\d+)"
PROB_RE = re.compile(rf"\s*({NUMERIC_TOKEN_RE})\s*([+\-*:])\s*({NUMERIC_TOKEN_RE})\s*")
PROMPT_PROB_RE = re.compile(r"Solve this fraction problem:\s*(.*?)\s*=\?\s*$", flags=re.IGNORECASE)


def parse_numeric_token(token: str) -> Fraction:
    t = str(token).strip()
    if "/" in t:
        num_s, den_s = t.split("/", 1)
        return Fraction(int(num_s), int(den_s))
    return Fraction(t)


def compute_correct_answer(prob: str) -> str:
    m = PROB_RE.fullmatch(str(prob))
    if not m:
        return "?"
    left_s, op, right_s = m.group(1), m.group(2), m.group(3)
    try:
        left = parse_numeric_token(left_s)
        right = parse_numeric_token(right_s)
        if op == "+":
            ans = left + right
        elif op == "-":
            ans = left - right
        elif op == "*":
            ans = left * right
        elif op == ":":
            if right == 0:
                return "?"
            ans = left / right
        else:
            return "?"
        return str(ans.numerator) if ans.denominator == 1 else f"{ans.numerator}/{ans.denominator}"
    except Exception:
        return "?"


def extract_prob_from_prompt(prompt: str) -> str:
    m = PROMPT_PROB_RE.search(str(prompt).strip())
    if not m:
        return ""
    return m.group(1).strip()


def detect_prompt_style_counts(prompts: pd.Series) -> Dict[str, int]:
    text = prompts.astype(str).str.strip()
    has_student = text.str.contains(r"<student>", case=False, regex=True, na=False)
    has_solve = text.str.contains(
        r"Solve this fraction problem:\s*.+\s*=\?\s*$",
        case=False,
        regex=True,
        na=False,
    )

    style = np.where(
        has_student & has_solve,
        "student_prefixed",
        np.where(~has_student & has_solve, "plain_solve", np.where(has_student, "student_other", "other")),
    )
    counts = pd.Series(style).value_counts().to_dict()
    return {
        "student_prefixed": int(counts.get("student_prefixed", 0)),
        "plain_solve": int(counts.get("plain_solve", 0)),
        "student_other": int(counts.get("student_other", 0)),
        "other": int(counts.get("other", 0)),
    }


def dominant_prompt_style(counts: Dict[str, int]) -> str:
    style_order = ["student_prefixed", "plain_solve", "student_other", "other"]
    return max(style_order, key=lambda k: counts.get(k, 0))


def validate_prompt_format_consistency(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prompt_col: str,
    eval_sp2013: bool,
    sp2013_use_param_grid: bool,
    train_prompt_student_mode: str = "none",
) -> Dict[str, Dict[str, int]]:
    allow_mixed_plain_and_student = train_prompt_student_mode == "dropout"
    allow_sp2013_style_mismatch = train_prompt_student_mode in {"always", "dropout"}
    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    split_counts: Dict[str, Dict[str, int]] = {}
    split_styles: Dict[str, str] = {}
    for split_name, frame in split_frames.items():
        counts = detect_prompt_style_counts(frame[prompt_col]) if len(frame) > 0 else {
            "student_prefixed": 0,
            "plain_solve": 0,
            "student_other": 0,
            "other": 0,
        }
        split_counts[split_name] = counts
        split_styles[split_name] = dominant_prompt_style(counts)

        mixed = counts["student_prefixed"] > 0 and counts["plain_solve"] > 0
        noncanonical = counts["student_other"] > 0 or counts["other"] > 0
        if noncanonical or (mixed and not allow_mixed_plain_and_student):
            raise ValueError(
                f"Prompt format mismatch within {split_name} split: counts={counts}. "
                "Expected one canonical style only (all student-prefixed OR all plain)."
            )

    train_style = split_styles["train"]
    if not allow_mixed_plain_and_student and train_style not in {"student_prefixed", "plain_solve"}:
        raise ValueError(
            f"Training prompt style is non-canonical: {train_style}. "
            "Expected student-prefixed or plain prompts."
        )

    if not allow_mixed_plain_and_student:
        for split_name in ["val", "test"]:
            frame = split_frames[split_name]
            if len(frame) == 0:
                continue
            if split_styles[split_name] != train_style:
                raise ValueError(
                    f"Train/test format mismatch: train={train_style}, {split_name}={split_styles[split_name]}. "
                    f"counts_train={split_counts['train']} counts_{split_name}={split_counts[split_name]}"
                )

    if eval_sp2013 and not allow_sp2013_style_mismatch:
        expected_eval_style = "student_prefixed" if sp2013_use_param_grid else "plain_solve"
        if train_style != expected_eval_style:
            raise ValueError(
                "Train/SP2013 prompt format mismatch: "
                f"train style={train_style}, SP2013 eval style={expected_eval_style}. "
                "Use --sp2013_use_param_grid when training prompts are student-prefixed, "
                "or --no-sp2013_use_param_grid when training prompts are plain."
            )

    return split_counts


def normalize_answer(ans: str) -> str:
    t = str(ans).strip().replace(" ", "")
    t = t.rstrip(".,;:!?")
    return t


def answer_to_float(ans: str):
    t = normalize_answer(ans)
    if t == "" or t in {"?", "nan", "None"}:
        return None
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            return float(int(a) / int(b))
        return float(t)
    except Exception:
        try:
            return float(Fraction(t))
        except Exception:
            return None


def answers_match(pred: str, correct: str, tol: float = 1e-6) -> bool:
    a = answer_to_float(pred)
    b = answer_to_float(correct)
    if a is None or b is None:
        return False
    return abs(a - b) < tol


@contextmanager
def seeded_torch_rng(device: torch.device, seed: int):
    devices = []
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [device_index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def is_finite_number(value: float) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def parse_float_list_arg(text: str, name: str) -> List[float]:
    raw = str(text).strip()
    if raw == "":
        raise ValueError(f"--{name} must not be empty.")
    values: List[float] = []
    for token in raw.split(","):
        tok = token.strip()
        if tok == "":
            continue
        try:
            values.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"--{name} has non-numeric token: {tok}") from exc
    if not values:
        raise ValueError(f"--{name} has no valid numeric values.")
    return values


def format_numeric_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def build_student_prompt(prob: str, g: float, d: float, rt_mu: float, ice: float) -> str:
    return (
        f"<student> g {format_numeric_token(g)} d {format_numeric_token(d)} "
        f"rt {format_numeric_token(rt_mu)} ice {format_numeric_token(ice)} </student>\n"
        f"Solve this fraction problem: {prob}=?"
    )


def stable_hash_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def build_dropout_prompt_keys(
    frame: pd.DataFrame,
    problem_col: str,
    response_col: str,
) -> pd.Series:
    def text_series(col: str) -> pd.Series:
        if col not in frame.columns:
            return pd.Series("", index=frame.index, dtype="object")
        return frame[col].fillna("").astype(str).str.strip()

    fallback = text_series(problem_col)
    for col in [response_col, "g", "d", "rt_mu", "ice"]:
        fallback = fallback + "::" + text_series(col)

    source_uid = text_series("source_uid")
    return source_uid.where(source_uid != "", fallback)


def apply_train_prompt_student_mode(
    frame: pd.DataFrame,
    prompt_col: str,
    response_col: str,
    problem_col: str,
    mode: str,
    dropout_prob: float,
    dropout_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if mode == "none" or len(frame) == 0:
        return frame, {
            "rows": int(len(frame)),
            "student_visible": 0,
            "plain_retained": int(len(frame)),
        }

    required_cols = [problem_col, "g", "d", "rt_mu", "ice"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(
            "train_prompt_student_mode requires standard translated columns in the input CSV: "
            f"missing {missing}"
        )

    if mode == "always":
        visible_mask = np.ones(len(frame), dtype=bool)
    elif mode == "dropout":
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("--train_prompt_student_dropout_prob must be in [0, 1].")
        keys = build_dropout_prompt_keys(frame=frame, problem_col=problem_col, response_col=response_col)
        visible_mask = keys.map(
            lambda key: stable_hash_fraction(f"{int(dropout_seed)}:{key}") >= float(dropout_prob)
        ).to_numpy(dtype=bool)
    else:
        raise ValueError(f"Unknown --train_prompt_student_mode: {mode}")

    visible_count = int(np.count_nonzero(visible_mask))
    if visible_count == 0:
        return frame, {
            "rows": int(len(frame)),
            "student_visible": 0,
            "plain_retained": int(len(frame)),
        }

    frame = frame.copy()
    visible_rows = frame.loc[visible_mask]
    frame.loc[visible_mask, prompt_col] = [
        build_student_prompt(prob=prob, g=g, d=d, rt_mu=rt_mu, ice=ice)
        for prob, g, d, rt_mu, ice in zip(
            visible_rows[problem_col].astype(str).str.strip().tolist(),
            visible_rows["g"].tolist(),
            visible_rows["d"].tolist(),
            visible_rows["rt_mu"].tolist(),
            visible_rows["ice"].tolist(),
        )
    ]
    return frame, {
        "rows": int(len(frame)),
        "student_visible": visible_count,
        "plain_retained": int(len(frame) - visible_count),
    }


def extract_final_answer(generation: str) -> str:
    txt = str(generation)
    m = re.search(r"###\s*answer\s*:\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if m:
        answer_text = m.group(1).strip()
        if answer_text:
            candidate = normalize_answer(answer_text.split()[0])
            if candidate != "" and answer_to_float(candidate) is not None:
                return candidate

    # Prefer valid fraction tokens from the generation body, then integers.
    frac_matches = re.findall(r"-?\d+/-?\d+", txt)
    for tok in reversed(frac_matches):
        candidate = normalize_answer(tok)
        if answer_to_float(candidate) is not None:
            return candidate

    int_matches = re.findall(r"-?\d+", txt)
    for tok in reversed(int_matches):
        candidate = normalize_answer(tok)
        if answer_to_float(candidate) is not None:
            return candidate
    return "?"


def build_sp2013_param_grid(
    use_param_grid: bool,
    grid_g: Sequence[float],
    grid_d: Sequence[float],
    grid_rt: Sequence[float],
    grid_ice: Sequence[float],
    param_grid_override: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
    if not use_param_grid:
        return [(None, None, None, None)]

    if param_grid_override is not None:
        param_grid = list(param_grid_override)
        if len(param_grid) == 0:
            raise ValueError("SP2013 parameter-grid eval received empty --param_grid_override.")
        return param_grid

    if not (grid_g and grid_d and grid_rt and grid_ice):
        raise ValueError("SP2013 parameter-grid eval enabled but one or more grid lists are empty.")
    return [
        (g, d, rt, ice)
        for g in grid_g
        for d in grid_d
        for rt in grid_rt
        for ice in grid_ice
    ]


def compute_rollout_accuracy_metrics(
    out_df: pd.DataFrame,
    use_param_grid: bool,
) -> Dict[str, Any]:
    if len(out_df) == 0:
        return {
            "overall_acc": float("nan"),
            "by_op": {},
            "sample_overall_acc": float("nan"),
            "sample_by_op": {},
            "rollout_any_acc": float("nan"),
            "rollout_by_op_any_acc": {},
            "n": 0,
            "n_prompt_rollout_groups": 0,
            "n_problems": 0,
            "use_param_grid": bool(use_param_grid),
        }

    sample_overall = float(out_df["is_correct"].mean())
    sample_by_op: Dict[str, float] = {}
    for op in ["+", "-", "*", ":"]:
        op_df = out_df[out_df["op"] == op]
        sample_by_op[op] = float(op_df["is_correct"].mean()) if len(op_df) > 0 else float("nan")

    group_cols = ["prob", "op"] + (["g", "d", "rt_mu", "ice"] if use_param_grid else [])
    rollout_df = out_df.groupby(group_cols, as_index=False)["is_correct"].agg(
        rollout_mean_acc="mean",
        rollout_any_correct="max",
        num_rollouts="size",
    )

    rollout_overall = float(rollout_df["rollout_mean_acc"].mean()) if len(rollout_df) > 0 else float("nan")
    rollout_any_acc = float(rollout_df["rollout_any_correct"].mean()) if len(rollout_df) > 0 else float("nan")

    rollout_by_op: Dict[str, float] = {}
    rollout_by_op_any: Dict[str, float] = {}
    for op in ["+", "-", "*", ":"]:
        op_rollout_df = rollout_df[rollout_df["op"] == op]
        rollout_by_op[op] = (
            float(op_rollout_df["rollout_mean_acc"].mean()) if len(op_rollout_df) > 0 else float("nan")
        )
        rollout_by_op_any[op] = (
            float(op_rollout_df["rollout_any_correct"].mean()) if len(op_rollout_df) > 0 else float("nan")
        )

    metrics: Dict[str, Any] = {
        "overall_acc": rollout_overall,
        "by_op": rollout_by_op,
        "sample_overall_acc": sample_overall,
        "sample_by_op": sample_by_op,
        "rollout_mean_acc": rollout_overall,
        "rollout_by_op_mean_acc": rollout_by_op,
        "rollout_any_acc": rollout_any_acc,
        "rollout_by_op_any_acc": rollout_by_op_any,
        "n": int(len(out_df)),
        "n_prompt_rollout_groups": int(len(rollout_df)),
        "n_problems": int(out_df["prob"].nunique()),
        "use_param_grid": bool(use_param_grid),
    }
    if len(rollout_df) > 0:
        per_problem = rollout_df.groupby("prob", as_index=False)["rollout_mean_acc"].mean()
        metrics["problem_acc_mean"] = float(per_problem["rollout_mean_acc"].mean())
        metrics["problem_acc_std"] = float(per_problem["rollout_mean_acc"].std(ddof=0))
        metrics["rollout_count_min"] = int(rollout_df["num_rollouts"].min())
        metrics["rollout_count_max"] = int(rollout_df["num_rollouts"].max())
    return metrics


def evaluate_sp2013_final_answer(
    model,
    tokenizer: AutoTokenizer,
    device: torch.device,
    sp2013_csv: str,
    max_new_tokens: int,
    use_param_grid: bool = False,
    grid_g: Sequence[float] = (),
    grid_d: Sequence[float] = (),
    grid_rt: Sequence[float] = (),
    grid_ice: Sequence[float] = (),
    param_grid_override: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    num_rollouts: int = 1,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    sample_seed: int = 123,
    rollout_batch_size: int = 1,
    progress_every: int = 0,
    progress_label: str = "sp2013_eval",
):
    if not os.path.exists(sp2013_csv):
        return {"overall_acc": float("nan"), "by_op": {}, "n": 0}, pd.DataFrame()

    core = model.module if hasattr(model, "module") else model
    core.eval()
    df = pd.read_csv(sp2013_csv)
    if "prob" not in df.columns:
        raise ValueError(f"SP2013 file missing 'prob' column: {sp2013_csv}")

    num_rollouts = int(num_rollouts)
    rollout_batch_size = int(rollout_batch_size)
    if num_rollouts < 1:
        raise ValueError("SP2013 eval requires num_rollouts >= 1.")
    if rollout_batch_size < 1:
        raise ValueError("SP2013 eval requires rollout_batch_size >= 1.")
    if num_rollouts > 1 and not do_sample:
        raise ValueError("SP2013 eval with num_rollouts > 1 requires do_sample=True.")

    param_grid = build_sp2013_param_grid(
        use_param_grid=use_param_grid,
        grid_g=grid_g,
        grid_d=grid_d,
        grid_rt=grid_rt,
        grid_ice=grid_ice,
        param_grid_override=param_grid_override,
    )

    probs = df["prob"].astype(str).tolist()
    total_items = len(probs) * len(param_grid) * num_rollouts
    progress_every = int(progress_every or 0)
    if progress_every < 0:
        progress_every = 0
    start_time = time.time()

    if progress_every > 0 and total_items > 0:
        print(
            f"[{progress_label}] starting eval: problems={len(probs)}, "
            f"grid={len(param_grid)}, rollouts={num_rollouts}, total_generations={total_items}",
            flush=True,
        )

    rows = []
    done = 0
    prompt_eval_idx = 0
    for prob_idx, prob in enumerate(probs, start=1):
        correct = compute_correct_answer(prob)
        op = op_from_prob(prob)
        for g, d, rt, ice in param_grid:
            if use_param_grid:
                prompt = build_student_prompt(prob=prob, g=g, d=d, rt_mu=rt, ice=ice)
            else:
                prompt = f"Solve this fraction problem: {prob}=?"

            prompt_eval_idx += 1
            if do_sample:
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                prompt_len = encoded["input_ids"].shape[1]
                prompt_seed_base = int(sample_seed + (prompt_eval_idx * 1000003))
                rollout_idx = 0
                while rollout_idx < num_rollouts:
                    chunk_size = min(rollout_batch_size, num_rollouts - rollout_idx)
                    with torch.no_grad():
                        with seeded_torch_rng(device, prompt_seed_base + rollout_idx):
                            out = core.generate(
                                **encoded,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                num_return_sequences=chunk_size,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                    for local_idx in range(chunk_size):
                        sample_idx = rollout_idx + local_idx
                        gen_ids = out[local_idx, prompt_len:]
                        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        pred = extract_final_answer(gen_text)
                        is_correct = answers_match(pred, correct)
                        rows.append(
                            {
                                "prob": prob,
                                "op": op,
                                "g": g if use_param_grid else np.nan,
                                "d": d if use_param_grid else np.nan,
                                "rt_mu": rt if use_param_grid else np.nan,
                                "ice": ice if use_param_grid else np.nan,
                                "sample_idx": sample_idx,
                                "prompt_text": prompt,
                                "pred_answer": pred,
                                "correct_answer": correct,
                                "is_correct": int(is_correct),
                                "generation_text": gen_text.strip(),
                            }
                        )
                        done += 1
                        if progress_every > 0 and (done % progress_every == 0 or done == total_items):
                            elapsed = max(time.time() - start_time, 1e-9)
                            rate = done / elapsed
                            remaining = max(0.0, (total_items - done) / rate) if rate > 0 else float("inf")
                            pct = (100.0 * done / total_items) if total_items > 0 else 100.0
                            print(
                                f"[{progress_label}] {done}/{total_items} ({pct:.1f}%) "
                                f"prob={prob_idx}/{len(probs)} rate={rate:.2f} gen/s "
                                f"eta={remaining/60.0:.1f}m",
                                flush=True,
                            )
                    rollout_idx += chunk_size
            else:
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                with torch.no_grad():
                    out = core.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen_ids = out[0, encoded["input_ids"].shape[1] :]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred = extract_final_answer(gen_text)
                is_correct = answers_match(pred, correct)
                rows.append(
                    {
                        "prob": prob,
                        "op": op,
                        "g": g if use_param_grid else np.nan,
                        "d": d if use_param_grid else np.nan,
                        "rt_mu": rt if use_param_grid else np.nan,
                        "ice": ice if use_param_grid else np.nan,
                        "sample_idx": 0,
                        "prompt_text": prompt,
                        "pred_answer": pred,
                        "correct_answer": correct,
                        "is_correct": int(is_correct),
                        "generation_text": gen_text.strip(),
                    }
                )
                done += 1
                if progress_every > 0 and (done % progress_every == 0 or done == total_items):
                    elapsed = max(time.time() - start_time, 1e-9)
                    rate = done / elapsed
                    remaining = max(0.0, (total_items - done) / rate) if rate > 0 else float("inf")
                    pct = (100.0 * done / total_items) if total_items > 0 else 100.0
                    print(
                        f"[{progress_label}] {done}/{total_items} ({pct:.1f}%) "
                        f"prob={prob_idx}/{len(probs)} rate={rate:.2f} gen/s "
                        f"eta={remaining/60.0:.1f}m",
                        flush=True,
                    )

    out_df = pd.DataFrame(rows)
    metrics = compute_rollout_accuracy_metrics(out_df, use_param_grid=use_param_grid)
    metrics["grid_size"] = int(len(param_grid))
    metrics["num_rollouts"] = int(num_rollouts)
    metrics["do_sample"] = bool(do_sample)
    metrics["temperature"] = float(temperature)
    metrics["top_p"] = float(top_p)
    metrics["top_k"] = int(top_k)
    metrics["sample_seed"] = int(sample_seed)
    metrics["rollout_batch_size"] = int(rollout_batch_size)
    return metrics, out_df


def evaluate_in_distribution_final_answer(
    model,
    tokenizer: AutoTokenizer,
    device: torch.device,
    eval_df: pd.DataFrame,
    prompt_col: str,
    response_col: str,
    max_new_tokens: int,
    sample_size: int,
    seed: int,
    target_mode: str = "response",
    prob_col: str = "prob",
):
    if eval_df is None or len(eval_df) == 0:
        return {
            "target_mode": target_mode,
            "acc_scored": float("nan"),
            "acc_all": float("nan"),
            "coverage": 0.0,
            "n_total": 0,
            "n_scored": 0,
            "n_correct": 0,
        }, pd.DataFrame()

    if sample_size is not None and sample_size > 0 and sample_size < len(eval_df):
        work_df = eval_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        work_df = eval_df.reset_index(drop=True)

    core = model.module if hasattr(model, "module") else model
    core.eval()

    rows = []
    for _, row in work_df.iterrows():
        p = str(row[prompt_col]).strip()
        target_response = str(row[response_col])
        prob = str(row.get(prob_col, "")).strip() if prob_col in work_df.columns else ""
        if prob == "":
            prob = extract_prob_from_prompt(p)

        if target_mode == "true":
            gold_answer = compute_correct_answer(prob)
        else:
            gold_answer = extract_final_answer(target_response)
        gold_parseable = answer_to_float(gold_answer) is not None

        encoded = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = core.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, encoded["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_answer = extract_final_answer(gen_text)
        is_correct = answers_match(pred_answer, gold_answer) if gold_parseable else False

        rows.append(
            {
                "prob": prob,
                "prompt": p,
                "target_answer": gold_answer,
                "target_parseable": int(gold_parseable),
                "pred_answer": pred_answer,
                "is_correct": int(is_correct),
                "generation_text": gen_text.strip(),
            }
        )

    out_df = pd.DataFrame(rows)
    n_total = int(len(out_df))
    n_scored = int(out_df["target_parseable"].sum()) if n_total > 0 else 0
    n_correct = int(out_df[(out_df["target_parseable"] == 1) & (out_df["is_correct"] == 1)].shape[0]) if n_total > 0 else 0
    coverage = float(n_scored / n_total) if n_total > 0 else 0.0
    acc_scored = float(n_correct / n_scored) if n_scored > 0 else float("nan")
    acc_all = float(n_correct / n_total) if n_total > 0 else float("nan")

    metrics = {
        "target_mode": target_mode,
        "acc_scored": acc_scored,
        "acc_all": acc_all,
        "coverage": coverage,
        "n_total": n_total,
        "n_scored": n_scored,
        "n_correct": n_correct,
    }
    return metrics, out_df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rank, world_size, local_rank, distributed = setup_distributed()

    try:
        if args.tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if not os.path.exists(args.data_csv):
            raise FileNotFoundError(f"Data CSV not found: {args.data_csv}")

        os.makedirs(args.output_dir, exist_ok=True)

        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank if distributed else 0)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        amp_dtype = pick_amp_dtype(args.amp_dtype, device)
        use_amp = amp_dtype is not None
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

        rank0_print(rank, "=" * 70)
        rank0_print(rank, "HF BACKBONE FINETUNING ON NLP UMA TRACES")
        rank0_print(rank, "=" * 70)
        rank0_print(rank, f"model: {args.model_name}")
        rank0_print(rank, f"init_from_scratch: {args.init_from_scratch}")
        rank0_print(rank, f"data: {args.data_csv}")
        rank0_print(rank, f"output_dir: {args.output_dir}")
        rank0_print(rank, f"distributed: {distributed} (world_size={world_size})")
        rank0_print(rank, f"sp2013_eval: {args.eval_sp2013} ({args.sp2013_csv})")
        rank0_print(rank, f"sp2013_param_grid_eval: {args.sp2013_use_param_grid}")
        rank0_print(rank, f"evals_per_epoch: {args.evals_per_epoch}")
        rank0_print(rank, f"local_files_only: {args.local_files_only}")
        rank0_print(rank, f"resume_from: {args.resume_from if str(args.resume_from).strip() else '(disabled)'}")
        rank0_print(rank, f"save_last_every_updates: {args.save_last_every_updates}")
        rank0_print(rank, f"save_eval_checkpoints: {args.save_eval_checkpoints}")
        rank0_print(rank, f"save_best_checkpoint: {args.save_best_checkpoint}")
        rank0_print(rank, f"save_final_checkpoint: {args.save_final_checkpoint}")
        rank0_print(rank, f"verify_saved_checkpoint_load: {args.verify_saved_checkpoint_load}")
        rank0_print(rank, f"best checkpoint criterion: {args.best_by}")
        rank0_print(rank, f"check_format_consistency: {args.check_format_consistency}")
        rank0_print(
            rank,
            f"id_eval: {args.eval_id_final_answer} "
            f"(split={args.id_eval_split}, sample_size={args.id_eval_size}, target={args.id_eval_target})",
        )
        rank0_print(rank, "")

        if args.evals_per_epoch < 1:
            raise ValueError("--evals_per_epoch must be >= 1.")
        if args.best_by == "sp2013_acc" and (not args.eval_sp2013 or args.sp2013_every <= 0):
            raise ValueError("--best_by sp2013_acc requires SP2013 eval enabled (set --eval_sp2013 and --sp2013_every > 0).")
        if args.best_by == "id_val_acc":
            if (not args.eval_id_final_answer) or (args.id_eval_every <= 0) or (args.id_eval_split not in {"val", "both"}):
                raise ValueError(
                    "--best_by id_val_acc requires ID eval enabled with val split "
                    "(set --eval_id_final_answer, --id_eval_every > 0, and --id_eval_split val|both)."
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
            grid_size = len(sp2013_grid_g) * len(sp2013_grid_d) * len(sp2013_grid_rt) * len(sp2013_grid_ice)
            rank0_print(
                rank,
                f"sp2013 grid sizes: g={len(sp2013_grid_g)} d={len(sp2013_grid_d)} "
                f"rt={len(sp2013_grid_rt)} ice={len(sp2013_grid_ice)} -> total={grid_size}",
            )

        if not 0.0 <= args.train_prompt_student_dropout_prob <= 1.0:
            raise ValueError("--train_prompt_student_dropout_prob must be in [0, 1].")

        prompt_student_mode_active = args.train_prompt_student_mode != "none"
        needs_problem_col = (
            args.split_by_problem
            or args.drop_train_eval_problem_overlap
            or args.move_sp2013_rows_to_val
            or prompt_student_mode_active
        )
        usecols = [args.prompt_col, args.response_col]
        if needs_problem_col and args.problem_col not in usecols:
            usecols.append(args.problem_col)
        if prompt_student_mode_active:
            for col in ["g", "d", "rt_mu", "ice", "source_uid"]:
                if col not in usecols:
                    usecols.append(col)

        df = pd.read_csv(args.data_csv, usecols=usecols)
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {args.data_csv}: {missing}")

        df = df.dropna(subset=usecols).copy()
        df[args.prompt_col] = df[args.prompt_col].astype(str).str.strip()
        df[args.response_col] = df[args.response_col].astype(str).str.strip()
        if args.problem_col in df.columns:
            df[args.problem_col] = df[args.problem_col].astype(str).str.strip()
        df = df[(df[args.prompt_col] != "") & (df[args.response_col] != "")].reset_index(drop=True)
        if args.problem_col in df.columns:
            df = df[df[args.problem_col] != ""].reset_index(drop=True)

        if args.max_samples is not None:
            n = min(args.max_samples, len(df))
            df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

        if not 0.0 <= args.val_frac < 1.0:
            raise ValueError("--val_frac must be in [0, 1).")
        if not 0.0 <= args.test_frac < 1.0:
            raise ValueError("--test_frac must be in [0, 1).")
        if args.val_frac + args.test_frac >= 1.0:
            raise ValueError("--val_frac + --test_frac must be < 1.")

        if args.split_by_problem:
            if args.problem_col not in df.columns:
                raise ValueError(
                    f"--split_by_problem requires problem column '{args.problem_col}' in {args.data_csv}."
                )

            uniq_probs = df[args.problem_col].dropna().astype(str).unique()
            perm_probs = np.random.RandomState(args.seed).permutation(uniq_probs)

            n_test_probs = int(len(perm_probs) * args.test_frac)
            n_val_probs = int(len(perm_probs) * args.val_frac)

            test_probs = set(perm_probs[:n_test_probs].tolist())
            val_probs = set(perm_probs[n_test_probs : n_test_probs + n_val_probs].tolist())
            train_probs = set(perm_probs[n_test_probs + n_val_probs :].tolist())

            test_df = df[df[args.problem_col].isin(test_probs)].reset_index(drop=True)
            val_df = df[df[args.problem_col].isin(val_probs)].reset_index(drop=True)
            train_df = df[df[args.problem_col].isin(train_probs)].reset_index(drop=True)

            rank0_print(rank, f"split mode: problem-level ({args.problem_col})")
            rank0_print(rank, f"unique problems total: {len(perm_probs):,}")
            rank0_print(rank, f"unique problems train: {len(train_probs):,}")
            rank0_print(rank, f"unique problems val:   {len(val_probs):,}")
            rank0_print(rank, f"unique problems test:  {len(test_probs):,}")
        else:
            perm = np.random.RandomState(args.seed).permutation(len(df))
            n_test = int(len(df) * args.test_frac)
            n_val = int(len(df) * args.val_frac)
            test_idx = perm[:n_test]
            val_idx = perm[n_test : n_test + n_val]
            train_idx = perm[n_test + n_val :]

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            rank0_print(rank, "split mode: row-level")

            if args.drop_train_eval_problem_overlap:
                if args.problem_col not in df.columns:
                    raise ValueError(
                        "--drop_train_eval_problem_overlap requires problem column "
                        f"'{args.problem_col}' in {args.data_csv}."
                    )
                blocked_probs = set(val_df[args.problem_col].tolist()) | set(test_df[args.problem_col].tolist())
                before_rows = len(train_df)
                train_df = train_df[~train_df[args.problem_col].isin(blocked_probs)].reset_index(drop=True)
                removed_rows = before_rows - len(train_df)
                rank0_print(
                    rank,
                    f"drop_train_eval_problem_overlap: removed {removed_rows:,} train rows "
                    f"(remaining {len(train_df):,})",
                )

        if args.move_sp2013_rows_to_val:
            if args.problem_col not in df.columns:
                raise ValueError(
                    "--move_sp2013_rows_to_val requires problem column "
                    f"'{args.problem_col}' in {args.data_csv}."
                )
            sp2013_ref = pd.read_csv(args.sp2013_csv)
            sp2013_problem_col = args.problem_col if args.problem_col in sp2013_ref.columns else "prob"
            if sp2013_problem_col not in sp2013_ref.columns:
                raise ValueError(
                    "--move_sp2013_rows_to_val requires a problem column in "
                    f"{args.sp2013_csv}; tried '{args.problem_col}' and 'prob'."
                )
            sp2013_probs = set(sp2013_ref[sp2013_problem_col].dropna().astype(str).str.strip().tolist())

            train_sp_mask = train_df[args.problem_col].isin(sp2013_probs)
            test_sp_mask = test_df[args.problem_col].isin(sp2013_probs)
            val_sp_mask = val_df[args.problem_col].isin(sp2013_probs)

            moved_train_rows = int(train_sp_mask.sum())
            moved_test_rows = int(test_sp_mask.sum())
            val_sp_rows_before = int(val_sp_mask.sum())
            val_sp_probs_before = int(val_df.loc[val_sp_mask, args.problem_col].nunique())
            train_sp_probs_before = int(train_df.loc[train_sp_mask, args.problem_col].nunique())
            test_sp_probs_before = int(test_df.loc[test_sp_mask, args.problem_col].nunique())

            moved_frames = []
            if moved_train_rows > 0:
                moved_frames.append(train_df.loc[train_sp_mask].copy())
            if moved_test_rows > 0:
                moved_frames.append(test_df.loc[test_sp_mask].copy())

            train_df = train_df.loc[~train_sp_mask].reset_index(drop=True)
            test_df = test_df.loc[~test_sp_mask].reset_index(drop=True)
            if moved_frames:
                val_df = pd.concat([val_df] + moved_frames, ignore_index=True)

            final_val_sp_mask = val_df[args.problem_col].isin(sp2013_probs)
            final_val_sp_rows = int(final_val_sp_mask.sum())
            final_val_sp_probs = int(val_df.loc[final_val_sp_mask, args.problem_col].nunique())
            rank0_print(
                rank,
                (
                    "move_sp2013_rows_to_val: "
                    f"sp2013_probs={len(sp2013_probs):,}, "
                    f"moved_train_rows={moved_train_rows:,} ({train_sp_probs_before:,} probs), "
                    f"moved_test_rows={moved_test_rows:,} ({test_sp_probs_before:,} probs), "
                    f"val_sp_rows_before={val_sp_rows_before:,} ({val_sp_probs_before:,} probs), "
                    f"val_sp_rows_after={final_val_sp_rows:,} ({final_val_sp_probs:,} probs)"
                ),
            )

        if len(train_df) == 0:
            raise ValueError("No training rows after split.")

        rank0_print(rank, f"rows total: {len(df):,}")
        rank0_print(rank, f"rows train: {len(train_df):,}")
        rank0_print(rank, f"rows val:   {len(val_df):,}")
        rank0_print(rank, f"rows test:  {len(test_df):,}")

        if prompt_student_mode_active:
            rank0_print(
                rank,
                "train prompt student mode: "
                f"{args.train_prompt_student_mode} (dropout_prob={args.train_prompt_student_dropout_prob}, "
                f"dropout_seed={args.train_prompt_student_dropout_seed})",
            )
            split_prompt_rewrite_stats: Dict[str, Dict[str, int]] = {}
            rewritten_splits: Dict[str, pd.DataFrame] = {}
            for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
                rewritten_frame, rewrite_stats = apply_train_prompt_student_mode(
                    frame=frame,
                    prompt_col=args.prompt_col,
                    response_col=args.response_col,
                    problem_col=args.problem_col,
                    mode=args.train_prompt_student_mode,
                    dropout_prob=args.train_prompt_student_dropout_prob,
                    dropout_seed=args.train_prompt_student_dropout_seed,
                )
                rewritten_splits[split_name] = rewritten_frame
                split_prompt_rewrite_stats[split_name] = rewrite_stats
            train_df = rewritten_splits["train"]
            val_df = rewritten_splits["val"]
            test_df = rewritten_splits["test"]

            for split_name in ["train", "val", "test"]:
                rewrite_stats = split_prompt_rewrite_stats[split_name]
                rank0_print(
                    rank,
                    (
                        f"prompt rewrite {split_name}: rows={rewrite_stats['rows']:,}, "
                        f"student_visible={rewrite_stats['student_visible']:,}, "
                        f"plain_retained={rewrite_stats['plain_retained']:,}"
                    ),
                )

        if args.problem_col in train_df.columns:
            train_probs = set(train_df[args.problem_col].tolist())
            val_probs = set(val_df[args.problem_col].tolist())
            test_probs = set(test_df[args.problem_col].tolist())
            overlap_train_val = len(train_probs & val_probs)
            overlap_train_test = len(train_probs & test_probs)
            overlap_val_test = len(val_probs & test_probs)
            rank0_print(
                rank,
                "problem overlap counts: "
                f"train∩val={overlap_train_val:,}, "
                f"train∩test={overlap_train_test:,}, "
                f"val∩test={overlap_val_test:,}",
            )

        if args.check_format_consistency:
            format_counts = validate_prompt_format_consistency(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                prompt_col=args.prompt_col,
                eval_sp2013=args.eval_sp2013,
                sp2013_use_param_grid=args.sp2013_use_param_grid,
                train_prompt_student_mode=args.train_prompt_student_mode,
            )
            rank0_print(rank, "prompt format counts by split:")
            for split_name in ["train", "val", "test"]:
                counts = format_counts[split_name]
                rank0_print(
                    rank,
                    (
                        f"  {split_name}: student_prefixed={counts['student_prefixed']:,}, "
                        f"plain_solve={counts['plain_solve']:,}, "
                        f"student_other={counts['student_other']:,}, other={counts['other']:,}"
                    ),
                )

        rank0_print(rank, "")

        if args.grad_accum_steps < 1:
            raise ValueError("--grad_accum_steps must be >= 1.")
        if args.save_last_every_updates < 0:
            raise ValueError("--save_last_every_updates must be >= 0.")

        resume_dir = get_resume_dir(args.output_dir, args.resume_from)
        if str(args.resume_from).strip() and not resume_dir:
            rank0_print(
                rank,
                f"resume requested ({args.resume_from}) but no valid {TRAINER_STATE_FILE} found; starting fresh.",
            )
        elif resume_dir:
            rank0_print(rank, f"resume checkpoint: {resume_dir}")

        if args.init_from_scratch and resume_dir:
            raise ValueError("--init_from_scratch cannot be used with --resume_from.")

        model_source = resolve_model_source(args.model_name, args.local_files_only)
        tokenizer_source = model_source
        if resume_dir and os.path.isfile(os.path.join(resume_dir, "tokenizer_config.json")):
            tokenizer_source = resume_dir
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

        is_lora_resume = bool(resume_dir) and os.path.isfile(os.path.join(resume_dir, "adapter_config.json"))
        use_lora = bool(args.use_lora or is_lora_resume)
        lora_target_modules = parse_csv_list(args.lora_target_modules)

        if args.use_lora and resume_dir and not is_lora_resume:
            raise ValueError(
                "--use_lora is enabled, but the resume checkpoint is not a LoRA adapter checkpoint."
            )

        if is_lora_resume:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "LoRA checkpoint detected but `peft` is not installed in the active environment."
                ) from exc
            base_model = AutoModelForCausalLM.from_pretrained(
                model_source,
                local_files_only=args.local_files_only,
            )
            model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
        elif resume_dir:
            model = AutoModelForCausalLM.from_pretrained(
                resume_dir,
                local_files_only=args.local_files_only,
            )
        else:
            if args.init_from_scratch:
                config = AutoConfig.from_pretrained(
                    model_source,
                    local_files_only=args.local_files_only,
                )
                model = AutoModelForCausalLM.from_config(config)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    local_files_only=args.local_files_only,
                )
            if use_lora:
                if not lora_target_modules:
                    raise ValueError("--lora_target_modules must include at least one module name.")
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                except ImportError as exc:
                    raise ImportError("LoRA requested but `peft` is not installed in the active environment.") from exc
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=lora_target_modules,
                    bias="none",
                )
                model = get_peft_model(model, lora_cfg)

        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to(device)

        if distributed:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

        core_model = model.module if hasattr(model, "module") else model
        total_params = sum(p.numel() for p in core_model.parameters())
        trainable_params = sum(p.numel() for p in core_model.parameters() if p.requires_grad)

        rank0_print(rank, f"device: {device}")
        rank0_print(rank, f"amp_dtype: {str(amp_dtype) if amp_dtype is not None else 'none'}")
        rank0_print(rank, f"params (total/trainable): {total_params:,}/{trainable_params:,}")
        rank0_print(rank, f"use_lora: {use_lora}")
        if use_lora:
            rank0_print(rank, f"lora target modules: {','.join(lora_target_modules)}")
            if rank == 0 and hasattr(core_model, "print_trainable_parameters"):
                core_model.print_trainable_parameters()
        global_batch = args.batch_size * world_size * max(1, args.grad_accum_steps)
        rank0_print(rank, f"global batch (samples/update): {global_batch}")
        rank0_print(rank, "")

        collate = build_collate_fn(tokenizer, max_length=args.max_length, train_on_prompt=args.train_on_prompt)
        train_ds = NLPtracesDataset(train_df[args.prompt_col].tolist(), train_df[args.response_col].tolist())
        val_ds = NLPtracesDataset(val_df[args.prompt_col].tolist(), val_df[args.response_col].tolist())
        test_ds = NLPtracesDataset(test_df[args.prompt_col].tolist(), test_df[args.response_col].tolist())

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin,
            collate_fn=collate,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=pin,
            collate_fn=collate,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=pin,
            collate_fn=collate,
            drop_last=False,
        )

        optim_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
        updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
        total_updates = updates_per_epoch * args.epochs
        warmup_steps = int(total_updates * args.warmup_ratio)
        rank0_print(rank, f"optimizer updates/epoch: {updates_per_epoch}")
        rank0_print(rank, f"total optimizer updates: {total_updates}")
        rank0_print(rank, f"warmup updates: {warmup_steps}")
        rank0_print(rank, "")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_updates),
        )

        history: List[Dict] = []
        best_val = float("inf")
        best_sp2013 = float("-inf")
        best_dir = os.path.join(args.output_dir, "best")
        last_dir = os.path.join(args.output_dir, "last")
        final_dir = os.path.join(args.output_dir, "final")
        global_update = 0
        start_epoch = 1
        resume_updates_in_epoch = 0
        elapsed_before_resume = 0.0

        if resume_dir:
            history = load_history_if_exists(resume_dir)
            state = torch.load(os.path.join(resume_dir, TRAINER_STATE_FILE), map_location="cpu")
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if scaler.is_enabled() and "scaler" in state:
                scaler.load_state_dict(state["scaler"])

            global_update = int(state.get("global_update", 0))
            start_epoch = int(state.get("epoch", 1))
            resume_updates_in_epoch = int(state.get("updates_in_epoch", 0))
            best_val = float(state.get("best_val", best_val))
            best_sp2013 = float(state.get("best_sp2013", best_sp2013))
            elapsed_before_resume = float(state.get("elapsed_sec", 0.0))

            if resume_updates_in_epoch >= updates_per_epoch:
                start_epoch += resume_updates_in_epoch // updates_per_epoch
                resume_updates_in_epoch = resume_updates_in_epoch % updates_per_epoch

            rank0_print(
                rank,
                f"resuming from epoch={start_epoch}, update_in_epoch={resume_updates_in_epoch}, global_update={global_update}",
            )
            rank0_print(rank, f"loaded history records: {len(history):,}")

        t0 = time.time() - elapsed_before_resume

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum = 0.0
            train_steps = 0
            accum_counter = 0
            updates_in_epoch = resume_updates_in_epoch if epoch == start_epoch else 0
            updates_to_skip = resume_updates_in_epoch if (epoch == start_epoch and resume_updates_in_epoch > 0) else 0
            skipped_updates = 0
            eval_targets = sorted(
                {
                    max(1, min(updates_per_epoch, math.ceil((i * updates_per_epoch) / args.evals_per_epoch)))
                    for i in range(1, args.evals_per_epoch + 1)
                }
            )
            if updates_to_skip > 0:
                rank0_print(
                    rank,
                    f"epoch {epoch}: skipping {updates_to_skip} already-completed optimizer updates from checkpoint.",
                )

            for step, batch in enumerate(train_loader, start=1):
                accum_counter += 1

                take_step = (accum_counter >= args.grad_accum_steps) or (step == len(train_loader))
                if skipped_updates < updates_to_skip:
                    if take_step:
                        skipped_updates += 1
                        accum_counter = 0
                    continue

                batch = move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss = model(**batch).loss
                loss_for_backward = loss / args.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                train_loss_sum += loss.item()
                train_steps += 1

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
                updates_in_epoch += 1

                if rank == 0 and args.save_last_every_updates > 0 and (global_update % args.save_last_every_updates == 0):
                    state = build_training_state(
                        epoch=epoch,
                        updates_in_epoch=updates_in_epoch,
                        global_update=global_update,
                        best_val=best_val,
                        best_sp2013=best_sp2013,
                        elapsed_sec=time.time() - t0,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    save_checkpoint(last_dir, model, tokenizer, vars(args), history, training_state=state)
                    print(
                        f"  saved resumable checkpoint -> {last_dir} "
                        f"(global_update={global_update})",
                        flush=True,
                    )

                if rank == 0 and global_update % args.log_every == 0:
                    avg_so_far = train_loss_sum / max(1, train_steps)
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"epoch {epoch}/{args.epochs} | update {global_update}/{total_updates} "
                        f"| train_loss {avg_so_far:.4f} | lr {lr_now:.2e}",
                        flush=True,
                    )

                if updates_in_epoch not in eval_targets:
                    continue

                local = torch.tensor([train_loss_sum, train_steps], dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(local, op=dist.ReduceOp.SUM)
                train_loss = float(local[0].item() / max(local[1].item(), 1.0))
                val_loss = evaluate(model, val_loader, device, amp_dtype, distributed) if len(val_ds) > 0 else float("nan")
                test_loss = evaluate(model, test_loader, device, amp_dtype, distributed) if len(test_ds) > 0 else float("nan")

                sp2013_acc = float("nan")
                sp2013_by_op: Dict[str, float] = {}
                id_val_acc = float("nan")
                id_val_coverage = float("nan")
                id_test_acc = float("nan")
                id_test_coverage = float("nan")
                eval_tag = f"e{epoch:02d}_u{updates_in_epoch:04d}"
                at_epoch_end = updates_in_epoch >= updates_per_epoch

                distributed_barrier(distributed, device)

                if rank == 0 and args.eval_sp2013 and args.sp2013_every > 0 and (epoch % args.sp2013_every == 0):
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
                    )
                    sp2013_acc = sp2013_metrics["overall_acc"]
                    sp2013_by_op = sp2013_metrics["by_op"]
                    sp2013_df.to_csv(os.path.join(args.output_dir, f"sp2013_{eval_tag}.csv"), index=False)
                    with open(os.path.join(args.output_dir, f"sp2013_metrics_{eval_tag}.json"), "w", encoding="utf-8") as f:
                        json.dump(sp2013_metrics, f, indent=2)
                    if at_epoch_end:
                        sp2013_df.to_csv(os.path.join(args.output_dir, f"sp2013_epoch_{epoch:02d}.csv"), index=False)
                        with open(os.path.join(args.output_dir, f"sp2013_metrics_epoch_{epoch:02d}.json"), "w", encoding="utf-8") as f:
                            json.dump(sp2013_metrics, f, indent=2)

                if rank == 0 and args.eval_id_final_answer and args.id_eval_every > 0 and (epoch % args.id_eval_every == 0):
                    if args.id_eval_split in {"val", "both"} and len(val_df) > 0:
                        id_val_metrics, id_val_df = evaluate_in_distribution_final_answer(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            eval_df=val_df,
                            prompt_col=args.prompt_col,
                            response_col=args.response_col,
                            max_new_tokens=args.id_eval_max_new_tokens,
                            sample_size=args.id_eval_size,
                            seed=args.id_eval_seed,
                            target_mode=args.id_eval_target,
                        )
                        id_val_acc = id_val_metrics["acc_scored"]
                        id_val_coverage = id_val_metrics["coverage"]
                        id_val_df.to_csv(os.path.join(args.output_dir, f"id_val_{eval_tag}.csv"), index=False)
                        with open(os.path.join(args.output_dir, f"id_val_metrics_{eval_tag}.json"), "w", encoding="utf-8") as f:
                            json.dump(id_val_metrics, f, indent=2)
                        if at_epoch_end:
                            id_val_df.to_csv(os.path.join(args.output_dir, f"id_val_epoch_{epoch:02d}.csv"), index=False)
                            with open(os.path.join(args.output_dir, f"id_val_metrics_epoch_{epoch:02d}.json"), "w", encoding="utf-8") as f:
                                json.dump(id_val_metrics, f, indent=2)

                    if args.id_eval_split in {"test", "both"} and len(test_df) > 0:
                        id_test_metrics, id_test_df = evaluate_in_distribution_final_answer(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            eval_df=test_df,
                            prompt_col=args.prompt_col,
                            response_col=args.response_col,
                            max_new_tokens=args.id_eval_max_new_tokens,
                            sample_size=args.id_eval_size,
                            seed=args.id_eval_seed,
                            target_mode=args.id_eval_target,
                        )
                        id_test_acc = id_test_metrics["acc_scored"]
                        id_test_coverage = id_test_metrics["coverage"]
                        id_test_df.to_csv(os.path.join(args.output_dir, f"id_test_{eval_tag}.csv"), index=False)
                        with open(os.path.join(args.output_dir, f"id_test_metrics_{eval_tag}.json"), "w", encoding="utf-8") as f:
                            json.dump(id_test_metrics, f, indent=2)
                        if at_epoch_end:
                            id_test_df.to_csv(os.path.join(args.output_dir, f"id_test_epoch_{epoch:02d}.csv"), index=False)
                            with open(os.path.join(args.output_dir, f"id_test_metrics_epoch_{epoch:02d}.json"), "w", encoding="utf-8") as f:
                                json.dump(id_test_metrics, f, indent=2)

                eval_record = {
                    "epoch": epoch,
                    "eval_tag": eval_tag,
                    "update_in_epoch": updates_in_epoch,
                    "updates_done": global_update,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "sp2013_acc": sp2013_acc,
                    "sp2013_by_op": sp2013_by_op,
                    "id_val_acc": id_val_acc,
                    "id_val_coverage": id_val_coverage,
                    "id_test_acc": id_test_acc,
                    "id_test_coverage": id_test_coverage,
                    "elapsed_sec": time.time() - t0,
                }

                if rank == 0:
                    history.append(eval_record)
                    print(
                        f"[eval {eval_tag}] train={train_loss:.4f} val={val_loss:.4f} test={test_loss:.4f} "
                        f"sp2013={sp2013_acc:.4f} id_val={id_val_acc:.4f}",
                        flush=True,
                    )

                    if args.best_by == "sp2013_acc":
                        improved = is_finite_number(sp2013_acc) and (
                            (sp2013_acc > best_sp2013)
                            or (
                                sp2013_acc == best_sp2013
                                and is_finite_number(val_loss)
                                and val_loss < best_val
                            )
                        )
                    elif args.best_by == "id_val_acc":
                        improved = is_finite_number(id_val_acc) and (
                            (id_val_acc > best_sp2013)
                            or (
                                id_val_acc == best_sp2013
                                and is_finite_number(val_loss)
                                and val_loss < best_val
                            )
                        )
                    else:
                        improved = is_finite_number(val_loss) and (val_loss < best_val)

                    if improved:
                        if is_finite_number(val_loss):
                            best_val = val_loss
                        if args.best_by == "id_val_acc":
                            if is_finite_number(id_val_acc):
                                best_sp2013 = id_val_acc
                        elif is_finite_number(sp2013_acc):
                            best_sp2013 = sp2013_acc

                    eval_state = build_training_state(
                        epoch=epoch,
                        updates_in_epoch=updates_in_epoch,
                        global_update=global_update,
                        best_val=best_val,
                        best_sp2013=best_sp2013,
                        elapsed_sec=time.time() - t0,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )

                    if args.save_eval_checkpoints:
                        eval_dir = build_eval_checkpoint_dir(args.output_dir, epoch, updates_in_epoch)
                        save_checkpoint(eval_dir, model, tokenizer, vars(args), history, training_state=eval_state)
                        print(f"  saved eval checkpoint -> {eval_dir}", flush=True)

                    if improved:
                        if args.save_best_checkpoint:
                            save_checkpoint(best_dir, model, tokenizer, vars(args), history)
                            if args.best_by == "sp2013_acc":
                                print(
                                    f"  saved best checkpoint -> {best_dir} "
                                    f"(sp2013={sp2013_acc:.4f}, val={val_loss:.4f})",
                                    flush=True,
                                )
                            elif args.best_by == "id_val_acc":
                                print(
                                    f"  saved best checkpoint -> {best_dir} "
                                    f"(id_val={id_val_acc:.4f}, val={val_loss:.4f})",
                                    flush=True,
                                )
                            else:
                                print(f"  saved best checkpoint -> {best_dir} (val={val_loss:.4f})", flush=True)
                    elif (not args.save_best_only) and at_epoch_end and (not args.save_eval_checkpoints):
                        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch:02d}")
                        save_checkpoint(epoch_dir, model, tokenizer, vars(args), history)
                        print(f"  saved checkpoint -> {epoch_dir}", flush=True)

                    if at_epoch_end:
                        save_checkpoint(last_dir, model, tokenizer, vars(args), history, training_state=eval_state)
                        print(
                            f"  saved resumable checkpoint -> {last_dir} "
                            f"(epoch={epoch}, update_in_epoch={updates_in_epoch})",
                            flush=True,
                        )

                distributed_barrier(distributed, device)

                if not at_epoch_end:
                    model.train()

        # Preview generation and reload verification are rank-0-only inference/I/O
        # passes. Let nonzero ranks leave instead of idling in NCCL collectives.
        if distributed and rank != 0:
            return

        if rank == 0 and args.save_final_checkpoint:
            final_state = build_training_state(
                epoch=max(1, args.epochs),
                updates_in_epoch=updates_per_epoch,
                global_update=global_update,
                best_val=best_val,
                best_sp2013=best_sp2013,
                elapsed_sec=time.time() - t0,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            save_checkpoint(final_dir, model, tokenizer, vars(args), history, training_state=final_state)
            print(f"\nSaved final checkpoint -> {final_dir}", flush=True)

        if rank == 0 and args.preview_samples > 0:
            preview_pool = val_df[args.prompt_col].head(args.preview_samples).tolist()
            if not preview_pool:
                preview_pool = train_df[args.prompt_col].head(args.preview_samples).tolist()
            if preview_pool:
                previews = generate_previews(
                    model,
                    tokenizer,
                    preview_pool[: args.preview_samples],
                    device=device,
                    max_new_tokens=args.preview_max_new_tokens,
                )
                preview_path = os.path.join(args.output_dir, "preview_generations.json")
                with open(preview_path, "w", encoding="utf-8") as f:
                    json.dump(previews, f, indent=2)
                print(f"Saved previews -> {preview_path}", flush=True)

        if rank == 0 and args.verify_saved_checkpoint_load:
            print("\nVerifying saved checkpoint load...", flush=True)
            reload_dir = pick_saved_checkpoint_for_reload(args.output_dir)
            reloaded_model = None
            if reload_dir:
                print(f"Loading checkpoint for verification: {reload_dir}", flush=True)
                reloaded_model = load_eval_model(
                    reload_dir,
                    model_source,
                    device,
                    local_files_only=args.local_files_only,
                )
                reloaded_model.eval()
                print("Loaded checkpoint for verification.", flush=True)
            else:
                print("No saved checkpoint found for verification.", flush=True)

            if reloaded_model is not None:
                loaded_summary: Dict[str, Any] = {
                    "checkpoint_source": reload_dir,
                    "phase": "loaded_model",
                    "elapsed_sec": time.time() - t0,
                }

                if len(test_ds) > 0:
                    print("Running reloaded-checkpoint test-loss eval...", flush=True)
                    test_loader_full = DataLoader(
                        test_ds,
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        sampler=None,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        collate_fn=collate,
                        drop_last=False,
                    )
                    loaded_test = evaluate(reloaded_model, test_loader_full, device, amp_dtype, distributed=False)
                    loaded_summary["test_loss"] = loaded_test
                    with open(os.path.join(args.output_dir, "best_test_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump({"checkpoint_source": reload_dir, "best_test_loss": loaded_test}, f, indent=2)
                    print(f"Reloaded-checkpoint test loss: {loaded_test:.4f}", flush=True)

                if args.eval_sp2013:
                    print("Running reloaded-checkpoint SP2013 eval...", flush=True)
                    sp_metrics, sp_df = evaluate_sp2013_final_answer(
                        model=reloaded_model,
                        tokenizer=tokenizer,
                        device=device,
                        sp2013_csv=args.sp2013_csv,
                        max_new_tokens=args.sp2013_max_new_tokens,
                        use_param_grid=args.sp2013_use_param_grid,
                        grid_g=sp2013_grid_g,
                        grid_d=sp2013_grid_d,
                        grid_rt=sp2013_grid_rt,
                        grid_ice=sp2013_grid_ice,
                    )
                    sp_df.to_csv(os.path.join(args.output_dir, "best_sp2013.csv"), index=False)
                    with open(os.path.join(args.output_dir, "best_sp2013_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(sp_metrics, f, indent=2)
                    loaded_summary["sp2013_acc"] = sp_metrics.get("overall_acc", float("nan"))
                    print(
                        f"Reloaded-checkpoint SP2013 acc: {sp_metrics.get('overall_acc', float('nan')):.4f}",
                        flush=True,
                    )

                if args.eval_id_final_answer:
                    if args.id_eval_split in {"val", "both"} and len(val_df) > 0:
                        print("Running reloaded-checkpoint ID-val eval...", flush=True)
                        id_val_metrics, id_val_df = evaluate_in_distribution_final_answer(
                            model=reloaded_model,
                            tokenizer=tokenizer,
                            device=device,
                            eval_df=val_df,
                            prompt_col=args.prompt_col,
                            response_col=args.response_col,
                            max_new_tokens=args.id_eval_max_new_tokens,
                            sample_size=args.id_eval_size,
                            seed=args.id_eval_seed,
                            target_mode=args.id_eval_target,
                        )
                        id_val_df.to_csv(os.path.join(args.output_dir, "best_id_val.csv"), index=False)
                        with open(os.path.join(args.output_dir, "best_id_val_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(id_val_metrics, f, indent=2)
                        loaded_summary["id_val_acc"] = id_val_metrics.get("acc_scored", float("nan"))
                        loaded_summary["id_val_coverage"] = id_val_metrics.get("coverage", float("nan"))
                        print(
                            f"Reloaded-checkpoint ID-val answer_acc: {id_val_metrics.get('acc_scored', float('nan')):.4f} "
                            f"(coverage={id_val_metrics.get('coverage', float('nan')):.4f})",
                            flush=True,
                        )

                    if args.id_eval_split in {"test", "both"} and len(test_df) > 0:
                        print("Running reloaded-checkpoint ID-test eval...", flush=True)
                        id_test_metrics, id_test_df = evaluate_in_distribution_final_answer(
                            model=reloaded_model,
                            tokenizer=tokenizer,
                            device=device,
                            eval_df=test_df,
                            prompt_col=args.prompt_col,
                            response_col=args.response_col,
                            max_new_tokens=args.id_eval_max_new_tokens,
                            sample_size=args.id_eval_size,
                            seed=args.id_eval_seed,
                            target_mode=args.id_eval_target,
                        )
                        id_test_df.to_csv(os.path.join(args.output_dir, "best_id_test.csv"), index=False)
                        with open(os.path.join(args.output_dir, "best_id_test_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(id_test_metrics, f, indent=2)
                        loaded_summary["id_test_acc"] = id_test_metrics.get("acc_scored", float("nan"))
                        loaded_summary["id_test_coverage"] = id_test_metrics.get("coverage", float("nan"))
                        print(
                            f"Reloaded-checkpoint ID-test answer_acc: {id_test_metrics.get('acc_scored', float('nan')):.4f} "
                            f"(coverage={id_test_metrics.get('coverage', float('nan')):.4f})",
                            flush=True,
                        )

                with open(os.path.join(args.output_dir, "loaded_model_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(loaded_summary, f, indent=2)

                del reloaded_model

        if rank == 0:
            print("\nDone.", flush=True)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
