#!/usr/bin/env python3
"""
Fine-tune an already-trained checkpoint on human fraction traces.

Pipeline:
1) Resolve a checkpoint directory (default: <model_path>/best).
2) Load model + tokenizer.
3) Fine-tune on train CSV (default: results/human/data_train_nlp.csv).
4) Evaluate validation NLL on explicit val CSV each epoch.
5) Save best/last/final checkpoints and metric artifacts.
6) Optionally run SP2013 final-answer evaluation during training.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from train_transformer_hf import (
    DEFAULT_SP2013_CSV,
    DEFAULT_SP2013_GRID_D,
    DEFAULT_SP2013_GRID_G,
    DEFAULT_SP2013_GRID_ICE,
    DEFAULT_SP2013_GRID_RT,
    NLPtracesDataset,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained checkpoint on human NLP traces.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Model/run directory. If this directory does not directly contain model weights, "
            "the script tries <model_path>/<checkpoint_subdir>."
        ),
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
        default=os.path.join("results", "human", "data_train_nlp.csv"),
        help="Train CSV (NLP columns preferred).",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=os.path.join("results", "human", "data_val_nlp.csv"),
        help="Validation CSV used for NLL tracking (NLP columns preferred).",
    )
    parser.add_argument("--prompt_col", type=str, default="instruction_nl")
    parser.add_argument("--response_col", type=str, default="response_nl")
    parser.add_argument("--output_dir", type=str, default="")

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
    parser.add_argument("--sp2013_grid_g", type=str, default=DEFAULT_SP2013_GRID_G)
    parser.add_argument("--sp2013_grid_d", type=str, default=DEFAULT_SP2013_GRID_D)
    parser.add_argument("--sp2013_grid_rt", type=str, default=DEFAULT_SP2013_GRID_RT)
    parser.add_argument("--sp2013_grid_ice", type=str, default=DEFAULT_SP2013_GRID_ICE)

    return parser.parse_args()


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


def default_output_dir(model_path: str) -> str:
    tag = os.path.basename(os.path.abspath(model_path.rstrip("/")))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", "transformer_replication", f"{tag}_finetune_human_{ts}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1.")
    if args.sp2013_num_rollouts < 1:
        raise ValueError("--sp2013_num_rollouts must be >= 1.")
    if args.sp2013_rollout_batch_size < 1:
        raise ValueError("--sp2013_rollout_batch_size must be >= 1.")
    if args.best_by == "sp2013_acc" and not args.eval_sp2013:
        raise ValueError("--best_by sp2013_acc requires --eval_sp2013.")

    ensure_exists(args.train_csv, "Train CSV")
    ensure_exists(args.val_csv, "Validation CSV")

    checkpoint_dir = resolve_checkpoint_dir(args.model_path, args.checkpoint_subdir)
    out_dir = args.output_dir.strip() or default_output_dir(args.model_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

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
    print(f"checkpoint_dir: {checkpoint_dir}", flush=True)
    print(f"train_csv: {args.train_csv}", flush=True)
    print(f"val_csv:   {args.val_csv}", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"amp_dtype: {str(amp_dtype) if amp_dtype is not None else 'none'}", flush=True)

    train_df_raw = read_csv_flexible(args.train_csv)
    val_df_raw = read_csv_flexible(args.val_csv)
    train_df = normalize_to_nlp_frame(train_df_raw, args.prompt_col, args.response_col)
    val_df = normalize_to_nlp_frame(val_df_raw, args.prompt_col, args.response_col)

    print(f"rows train: {len(train_df):,}", flush=True)
    print(f"rows val:   {len(val_df):,}", flush=True)
    print("", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
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
            local_files_only=args.local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            local_files_only=args.local_files_only,
        )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

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

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_updates = updates_per_epoch * args.epochs
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

    history: List[Dict[str, Any]] = []
    global_update = 0
    best_dir = os.path.join(out_dir, "best")
    last_dir = os.path.join(out_dir, "last")
    final_dir = os.path.join(out_dir, "final")
    best_val_nll = float("inf")
    best_sp2013 = float("-inf")
    t0 = time.time()

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
        train_steps = 0
        accum_counter = 0

        for step, batch in enumerate(train_loader, start=1):
            accum_counter += 1
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

            if args.log_every > 0 and global_update % args.log_every == 0:
                avg_so_far = train_loss_sum / max(train_steps, 1)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch}/{args.epochs} | update {global_update}/{total_updates} "
                    f"| train_loss {avg_so_far:.4f} | lr {lr_now:.2e}",
                    flush=True,
                )

        train_nll = train_loss_sum / max(train_steps, 1)
        val_nll = evaluate(model, val_loader, device, amp_dtype, distributed=False) if len(val_ds) > 0 else float("nan")
        if math.isfinite(val_nll):
            try:
                val_ppl = float(math.exp(val_nll))
            except OverflowError:
                val_ppl = float("inf")
        else:
            val_ppl = float("nan")

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
            "sp2013_acc": sp2013_acc,
            "sp2013_by_op": sp2013_by_op,
            "elapsed_sec": time.time() - t0,
        }
        history.append(eval_record)

        print(
            f"[epoch {epoch:02d}] train_nll={train_nll:.4f} "
            f"val_nll={val_nll:.4f} val_ppl={val_ppl:.2f} sp2013_acc={sp2013_acc:.4f}",
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
        if args.save_epoch_checkpoints:
            epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
            save_checkpoint(epoch_dir, model, tokenizer, vars(args), history, training_state=training_state)
            print(f"  saved epoch checkpoint -> {epoch_dir}", flush=True)

    final_state = build_training_state(
        epoch=args.epochs,
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
        json.dump(
            {
                "best_by": args.best_by,
                "best_val_nll": best_val_nll,
                "best_sp2013_acc": best_sp2013_summary,
                "epochs": args.epochs,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "checkpoint_source": checkpoint_dir,
            },
            f,
            indent=2,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
