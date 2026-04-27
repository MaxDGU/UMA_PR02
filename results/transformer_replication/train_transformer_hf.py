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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_CSV = os.path.join(SCRIPT_DIR, "uma_traces_all_nlp.csv.gz")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "smollm2_135m_nlp_finetune")
DEFAULT_SP2013_CSV = os.path.join(SCRIPT_DIR, "..", "UMA_replication", "sp2013.csv")
DEFAULT_SP2013_TARGET_CSV = os.path.join(
    SCRIPT_DIR,
    "..",
    "UMA_replication",
    "sp2013_eval",
    "seed_1",
    "sp2013_seed1_all_models.csv",
)
DEFAULT_SP2013_GRID_G = "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10"
DEFAULT_SP2013_GRID_D = "0.1,0.3,0.5,0.7,0.9"
DEFAULT_SP2013_GRID_RT = "3,4,5,6"
DEFAULT_SP2013_GRID_ICE = "0,25,50,75,100"
TRAINER_STATE_FILE = "trainer_state.pt"


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float,
    min_lr: float,
) -> LambdaLR:
    total_steps = max(1, int(num_training_steps))
    warmup_steps = max(0, int(num_warmup_steps))
    if base_lr <= 0.0:
        if min_lr != 0.0:
            raise ValueError("--min_lr must be 0 when --lr is 0.")
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    if min_lr < 0.0:
        raise ValueError("--min_lr must be non-negative.")
    if min_lr > base_lr:
        raise ValueError("--min_lr cannot exceed --lr.")

    min_ratio = min_lr / base_lr

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            decay_ratio = 1.0 - progress
        elif scheduler_type == "cosine":
            decay_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported lr scheduler type: {scheduler_type}")
        return min_ratio + (1.0 - min_ratio) * decay_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minimum learning-rate floor reached at the end of decay.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        choices=["cosine", "linear"],
        default="cosine",
        help="Learning-rate scheduler family.",
    )
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
    parser.add_argument(
        "--strict_sp2013_only_validation_layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --move_sp2013_rows_to_val is active with val/test fractions at 0, require the "
            "original full-layout SP2013 validation pool (16,000 rows, 16 problems, and 1,000 "
            "student tuples when parameter columns are present). Disable this for filtered "
            "datasets that intentionally keep only a subset of SP2013 rows."
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
    parser.add_argument("--sp2013_target_csv", type=str, default=DEFAULT_SP2013_TARGET_CSV)
    parser.add_argument(
        "--sp2013_val_source_csv",
        type=str,
        default="",
        help=(
            "Optional alternate CSV used only to build the SP2013 validation pool when "
            "--move_sp2013_rows_to_val is active. This is useful when training data is filtered "
            "but the held-out SP2013 validation set should remain the full unfiltered pool."
        ),
    )
    parser.add_argument("--sp2013_max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_sp2013", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sp2013_every", type=int, default=1, help="Evaluate SP2013 every N epochs.")
    parser.add_argument(
        "--sp2013_target_mode",
        choices=["true", "uma_tuple", "uma_distribution"],
        default="true",
        help="SP2013 primary target: mathematically true answer, tuple-conditioned UMA answers, or no-param UMA answer distribution.",
    )
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
        "--loss_evals_per_epoch",
        type=int,
        default=0,
        help=(
            "Run lightweight validation-loss evaluation this many times per epoch. "
            "Set to 0 to reuse --evals_per_epoch."
        ),
    )
    parser.add_argument(
        "--best_by",
        choices=["val_loss", "sp2013_acc", "sp2013_primary", "id_val_acc"],
        default="val_loss",
        help="Checkpoint selection criterion for the best model.",
    )
    parser.add_argument("--sp2013_num_rollouts", type=int, default=1)
    parser.add_argument("--sp2013_do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sp2013_temperature", type=float, default=0.7)
    parser.add_argument("--sp2013_top_p", type=float, default=0.95)
    parser.add_argument("--sp2013_top_k", type=int, default=50)
    parser.add_argument("--sp2013_sample_seed", type=int, default=123)
    parser.add_argument("--sp2013_rollout_batch_size", type=int, default=1)
    parser.add_argument("--sp2013_progress_every", type=int, default=0)
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
    parser.add_argument("--id_eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--id_eval_every", type=int, default=1, help="Evaluate in-distribution answer metrics every N epochs.")
    parser.add_argument("--id_eval_seed", type=int, default=123)
    parser.add_argument(
        "--id_eval_batch_size",
        type=int,
        default=64,
        help="Generation batch size for in-distribution final-answer eval.",
    )
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
        "--id_eval_report_true_target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also report true-answer accuracy from the same ID-eval generations.",
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
        "--domain",
        choices=["fraction", "whole_number"],
        default="fraction",
        help=(
            "Arithmetic domain. Controls the prompt noun ('fraction problem' vs "
            "'whole-number arithmetic problem') used when the trainer rewrites or builds prompts."
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


def get_legacy_random_sampler_epoch_seed(base_seed: int, epoch: int) -> int:
    if epoch < 1:
        raise ValueError("epoch must be >= 1.")
    generator = torch.Generator()
    generator.manual_seed(int(base_seed))
    seed_value = 0
    for _ in range(epoch):
        seed_value = int(torch.empty((), dtype=torch.int64).random_(generator=generator).item())
    return seed_value


class OffsetBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, start_batch: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if start_batch < 0:
            raise ValueError("start_batch must be >= 0.")
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.start_batch = int(start_batch)

    def __iter__(self):
        batch: List[int] = []
        batch_idx = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if batch_idx >= self.start_batch:
                    yield batch
                batch = []
                batch_idx += 1
        if batch and not self.drop_last and batch_idx >= self.start_batch:
            yield batch

    def __len__(self) -> int:
        base_batches = len(BatchSampler(self.sampler, self.batch_size, self.drop_last))
        return max(0, base_batches - self.start_batch)


def build_train_index_sampler(
    train_ds: Dataset,
    distributed: bool,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
) -> Sampler[int]:
    if distributed:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        sampler.set_epoch(epoch)
        return sampler

    # Match the legacy RandomSampler behavior so existing single-GPU checkpoints
    # can resume at a batch offset without replaying tens of thousands of batches.
    generator = torch.Generator()
    generator.manual_seed(get_legacy_random_sampler_epoch_seed(seed, epoch))
    return RandomSampler(train_ds, generator=generator)


def build_train_loader_for_epoch(
    train_ds: Dataset,
    collate_fn,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
    start_batch: int = 0,
) -> DataLoader:
    sampler = build_train_index_sampler(
        train_ds=train_ds,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=seed,
        epoch=epoch,
    )
    batch_sampler = OffsetBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        start_batch=start_batch,
    )
    return DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


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


def build_eval_targets(updates_per_epoch: int, num_evals_per_epoch: int) -> List[int]:
    num_evals_per_epoch = max(1, int(num_evals_per_epoch))
    return sorted(
        {
            max(1, min(updates_per_epoch, math.ceil((i * updates_per_epoch) / num_evals_per_epoch)))
            for i in range(1, num_evals_per_epoch + 1)
        }
    )


def load_saved_train_args_if_exists(ckpt_dir: str) -> Dict[str, Any]:
    path = os.path.join(ckpt_dir, "train_args.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


NUMERIC_TOKEN_RE = r"[+\-]?(?:\d+(?:/\d+)?|\d*\.\d+)"
ANSWER_TOKEN_RE = re.compile(r"-?(?:\d+(?:\.\d+)?)(?:/-?(?:\d+(?:\.\d+)?))?")
PROB_RE = re.compile(rf"\s*({NUMERIC_TOKEN_RE})\s*([+\-*/:])\s*({NUMERIC_TOKEN_RE})\s*")
DOMAIN_PROMPT_NOUN: Dict[str, str] = {
    "fraction": "fraction problem",
    "whole_number": "whole-number arithmetic problem",
}
PROMPT_NOUN_PATTERN = r"(?:fraction problem|whole-number arithmetic problem)"
PROMPT_PROB_RE = re.compile(
    rf"Solve this {PROMPT_NOUN_PATTERN}:\s*(.*?)\s*=\?\s*$", flags=re.IGNORECASE
)


def domain_prompt_template(domain: str) -> str:
    noun = DOMAIN_PROMPT_NOUN.get(domain, "fraction problem")
    return f"Solve this {noun}:"


def canonicalize_division_op(op: str) -> str:
    return ":" if str(op).strip() == "/" else str(op).strip()


def display_operation_symbol(op: str) -> str:
    return "/" if canonicalize_division_op(op) == ":" else canonicalize_division_op(op)


def parse_binary_problem(prob: str) -> Optional[Tuple[str, str, str]]:
    m = PROB_RE.fullmatch(str(prob).strip())
    if not m:
        return None
    left_s, op, right_s = m.group(1), canonicalize_division_op(m.group(2)), m.group(3)
    return left_s, op, right_s


def render_problem_for_prompt(prob: str) -> str:
    parsed = parse_binary_problem(prob)
    if parsed is None:
        return str(prob).strip()
    left_s, op, right_s = parsed
    surface_op = " / " if op == ":" else f" {op} "
    return f"{left_s}{surface_op}{right_s}"


def op_from_prob(prob: str) -> str:
    parsed = parse_binary_problem(prob)
    if parsed is None:
        return "?"
    return display_operation_symbol(parsed[1])


def parse_numeric_token(token: str) -> Fraction:
    t = str(token).strip()
    if "/" in t:
        num_s, den_s = t.split("/", 1)
        return Fraction(int(num_s), int(den_s))
    return Fraction(t)


def compute_correct_answer(prob: str) -> str:
    parsed = parse_binary_problem(prob)
    if parsed is None:
        return "?"
    left_s, op, right_s = parsed
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
        rf"Solve this {PROMPT_NOUN_PATTERN}:\s*.+\s*=\?\s*$",
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


def answer_to_fraction(ans: str) -> Optional[Fraction]:
    t = normalize_answer(ans)
    if t == "" or t in {"?", "nan", "None"}:
        return None
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            den = Fraction(b)
            if den == 0:
                return None
            return Fraction(a) / den
        return Fraction(t)
    except Exception:
        return None


def answer_to_float(ans: str):
    frac = answer_to_fraction(ans)
    if frac is None:
        return None
    return float(frac)


def canonicalize_answer_label(ans: str) -> str:
    frac = answer_to_fraction(ans)
    if frac is None:
        token = normalize_answer(ans)
        return token if token != "" else "?"
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


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


def build_student_prompt(
    prob: str, g: float, d: float, rt_mu: float, ice: float, domain: str = "fraction"
) -> str:
    prompt_prob = render_problem_for_prompt(prob)
    return (
        f"<student> g {format_numeric_token(g)} d {format_numeric_token(d)} "
        f"rt {format_numeric_token(rt_mu)} ice {format_numeric_token(ice)} </student>\n"
        f"{domain_prompt_template(domain)} {prompt_prob}=?"
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
    domain: str = "fraction",
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
        build_student_prompt(prob=prob, g=g, d=d, rt_mu=rt_mu, ice=ice, domain=domain)
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


def count_unique_student_param_tuples(frame: pd.DataFrame) -> Optional[int]:
    required_cols = ["g", "d", "rt_mu", "ice"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        return None
    if len(frame) == 0:
        return 0
    return int(frame.loc[:, required_cols].drop_duplicates().shape[0])


def load_and_clean_trace_frame(
    csv_path: str,
    usecols: Sequence[str],
    prompt_col: str,
    response_col: str,
    problem_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=list(usecols))
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.dropna(subset=list(usecols)).copy()
    df[prompt_col] = df[prompt_col].astype(str).str.strip()
    df[response_col] = df[response_col].astype(str).str.strip()
    if problem_col in df.columns:
        df[problem_col] = df[problem_col].astype(str).str.strip()
    df = df[(df[prompt_col] != "") & (df[response_col] != "")].reset_index(drop=True)
    if problem_col in df.columns:
        df = df[df[problem_col] != ""].reset_index(drop=True)
    return df


def assert_sp2013_only_validation_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    problem_col: str,
    sp2013_probs: Sequence[str],
    strict_layout: bool = True,
) -> Dict[str, Any]:
    expected_val_rows = 16000
    expected_prob_count = 16
    expected_param_tuple_count = 1000

    sp2013_prob_set = set(str(prob).strip() for prob in sp2013_probs)
    train_sp_rows = int(train_df[problem_col].isin(sp2013_prob_set).sum())
    test_sp_rows = int(test_df[problem_col].isin(sp2013_prob_set).sum())
    val_sp_rows = int(val_df[problem_col].isin(sp2013_prob_set).sum())
    val_prob_count = int(val_df[problem_col].nunique())
    val_param_tuple_count = count_unique_student_param_tuples(val_df)

    if train_sp_rows != 0 or test_sp_rows != 0:
        raise ValueError(
            "SP2013-only validation split failed: "
            f"train_sp_rows={train_sp_rows}, test_sp_rows={test_sp_rows}."
        )
    if val_sp_rows != len(val_df):
        raise ValueError(
            "SP2013-only validation split failed: "
            f"val_sp_rows={val_sp_rows}, val_rows={len(val_df)}."
        )
    if len(val_df) == 0:
        raise ValueError(
            "SP2013-only validation split failed: "
            "validation split is empty after moving SP2013 rows."
        )
    if strict_layout and len(val_df) != expected_val_rows:
        raise ValueError(
            "SP2013-only validation split failed: "
            f"expected val_rows={expected_val_rows}, got {len(val_df)}."
        )
    if strict_layout and (val_prob_count != expected_prob_count or len(sp2013_prob_set) != expected_prob_count):
        raise ValueError(
            "SP2013-only validation split failed: "
            f"expected {expected_prob_count} unique problems, got val={val_prob_count}, ref={len(sp2013_prob_set)}."
        )
    if strict_layout:
        if val_param_tuple_count is None:
            raise ValueError(
                "SP2013-only validation split failed: "
                "strict layout requires student parameter columns g/d/rt_mu/ice in validation rows."
            )
        if val_param_tuple_count != expected_param_tuple_count:
            raise ValueError(
                "SP2013-only validation split failed: "
                f"expected {expected_param_tuple_count} unique student tuples, got {val_param_tuple_count}."
            )

    return {
        "val_rows": int(len(val_df)),
        "val_prob_count": val_prob_count,
        "ref_prob_count": int(len(sp2013_prob_set)),
        "val_param_tuple_count": (
            int(val_param_tuple_count) if val_param_tuple_count is not None else None
        ),
        "train_sp_rows": train_sp_rows,
        "test_sp_rows": test_sp_rows,
        "strict_layout": bool(strict_layout),
    }


def summarize_id_eval_target(
    out_df: pd.DataFrame,
    target_prefix: str,
) -> Dict[str, Any]:
    parseable_col = f"{target_prefix}_target_parseable"
    correct_col = f"is_correct_{target_prefix}"
    n_total = int(len(out_df))
    n_scored = int(out_df[parseable_col].sum()) if n_total > 0 else 0
    n_correct = int(out_df[(out_df[parseable_col] == 1) & (out_df[correct_col] == 1)].shape[0]) if n_total > 0 else 0
    coverage = float(n_scored / n_total) if n_total > 0 else 0.0
    acc_scored = float(n_correct / n_scored) if n_scored > 0 else float("nan")
    acc_all = float(n_correct / n_total) if n_total > 0 else float("nan")
    return {
        "acc_scored": acc_scored,
        "acc_all": acc_all,
        "coverage": coverage,
        "n_total": n_total,
        "n_scored": n_scored,
        "n_correct": n_correct,
    }


def write_best_eval_artifacts(
    output_dir: str,
    eval_record: Dict[str, Any],
    sp2013_metrics: Optional[Dict[str, Any]] = None,
    sp2013_df: Optional[pd.DataFrame] = None,
    id_val_metrics: Optional[Dict[str, Any]] = None,
    id_val_df: Optional[pd.DataFrame] = None,
    id_test_metrics: Optional[Dict[str, Any]] = None,
    id_test_df: Optional[pd.DataFrame] = None,
) -> None:
    with open(os.path.join(output_dir, "best_checkpoint_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)
    if sp2013_metrics is not None and sp2013_df is not None:
        sp2013_df.to_csv(os.path.join(output_dir, "best_sp2013.csv"), index=False)
        with open(os.path.join(output_dir, "best_sp2013_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(sp2013_metrics, f, indent=2)
    if id_val_metrics is not None and id_val_df is not None:
        id_val_df.to_csv(os.path.join(output_dir, "best_id_val.csv"), index=False)
        with open(os.path.join(output_dir, "best_id_val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(id_val_metrics, f, indent=2)
    if id_test_metrics is not None and id_test_df is not None:
        id_test_df.to_csv(os.path.join(output_dir, "best_id_test.csv"), index=False)
        with open(os.path.join(output_dir, "best_id_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(id_test_metrics, f, indent=2)


def extract_answer_token(generation: str) -> str:
    txt = str(generation)
    m = re.search(r"###\s*answer\s*:\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if m:
        answer_text = m.group(1).strip()
        if answer_text:
            candidate = normalize_answer(answer_text.split()[0])
            if candidate != "":
                return candidate

    for tok in reversed(ANSWER_TOKEN_RE.findall(txt)):
        candidate = normalize_answer(tok)
        if candidate != "":
            return candidate
    return "?"


def extract_final_answer(generation: str) -> str:
    candidate = extract_answer_token(generation)
    if candidate != "?" and answer_to_float(candidate) is not None:
        return canonicalize_answer_label(candidate)
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


def normalize_problem_key(prob: str) -> str:
    parsed = parse_binary_problem(prob)
    if parsed is None:
        return str(prob).strip()
    left_s, op, right_s = parsed
    return f"{left_s}{op}{right_s}"


def build_sp2013_tuple_key(prob: str, g: Any, d: Any, rt_mu: Any, ice: Any) -> Tuple[str, str, str, str, str]:
    return (
        normalize_problem_key(prob),
        format_numeric_token(float(g)),
        format_numeric_token(float(d)),
        format_numeric_token(float(rt_mu)),
        format_numeric_token(float(ice)),
    )


def summarize_parseable_match_metrics(frame: pd.DataFrame, parseable_col: str, correct_col: str) -> Dict[str, Any]:
    n_total = int(len(frame))
    n_scored = int(frame[parseable_col].sum()) if n_total > 0 else 0
    n_correct = int(frame[(frame[parseable_col] == 1) & (frame[correct_col] == 1)].shape[0]) if n_total > 0 else 0
    coverage = float(n_scored / n_total) if n_total > 0 else 0.0
    acc_scored = float(n_correct / n_scored) if n_scored > 0 else float("nan")
    acc_all = float(n_correct / n_total) if n_total > 0 else float("nan")
    return {
        "acc_scored": acc_scored,
        "acc_all": acc_all,
        "coverage": coverage,
        "n_total": n_total,
        "n_scored": n_scored,
        "n_correct": n_correct,
    }


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    support = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(key, 0.0)) - float(q.get(key, 0.0))) for key in support)


def build_answer_distribution(series: pd.Series) -> Dict[str, float]:
    if len(series) == 0:
        return {}
    counts = series.fillna("?").astype(str).value_counts(dropna=False)
    total = float(counts.sum())
    return {str(key): float(val / total) for key, val in counts.items()}


def load_sp2013_target_frame(target_csv: str) -> pd.DataFrame:
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"SP2013 target CSV not found: {target_csv}")
    target_df = pd.read_csv(target_csv)
    required_cols = {"prob", "ans"}
    if not required_cols.issubset(target_df.columns):
        raise ValueError(f"SP2013 target CSV missing columns {sorted(required_cols - set(target_df.columns))}: {target_csv}")
    target_df = target_df.copy()
    target_df["prob_key"] = target_df["prob"].astype(str).map(normalize_problem_key)
    target_df["target_answer"] = target_df["ans"].astype(str).map(normalize_answer)
    target_df["target_answer_label"] = target_df["target_answer"].map(canonicalize_answer_label)
    target_df["target_answer_parseable"] = target_df["target_answer"].map(lambda x: int(answer_to_float(x) is not None))
    target_df["op"] = target_df["prob"].astype(str).map(op_from_prob)
    if {"g", "d", "rt_mu", "ice"}.issubset(target_df.columns):
        target_df["tuple_key"] = [
            build_sp2013_tuple_key(prob, g, d, rt_mu, ice)
            for prob, g, d, rt_mu, ice in zip(
                target_df["prob"].tolist(),
                target_df["g"].tolist(),
                target_df["d"].tolist(),
                target_df["rt_mu"].tolist(),
                target_df["ice"].tolist(),
            )
        ]
    return target_df


def compute_sp2013_uma_tuple_metrics(out_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    work_df = out_df.copy()
    work_df["prob_key"] = work_df["prob"].astype(str).map(normalize_problem_key)
    work_df["tuple_key"] = [
        build_sp2013_tuple_key(prob, g, d, rt_mu, ice)
        for prob, g, d, rt_mu, ice in zip(
            work_df["prob"].tolist(),
            work_df["g"].tolist(),
            work_df["d"].tolist(),
            work_df["rt_mu"].tolist(),
            work_df["ice"].tolist(),
        )
    ]
    target_slice = target_df.loc[:, ["tuple_key", "target_answer", "target_answer_label", "target_answer_parseable"]].drop_duplicates("tuple_key")
    merged = work_df.merge(target_slice, on="tuple_key", how="left", validate="many_to_one")
    if merged["target_answer"].isna().any():
        missing = int(merged["target_answer"].isna().sum())
        raise ValueError(f"Missing UMA tuple targets for {missing} generated SP2013 rows.")

    merged["is_correct_target"] = [
        int(answers_match(pred, target)) if int(parseable) == 1 else 0
        for pred, target, parseable in zip(
            merged["pred_answer"].tolist(),
            merged["target_answer"].tolist(),
            merged["target_answer_parseable"].tolist(),
        )
    ]
    merged["true_target_parseable"] = 1

    teacher = summarize_parseable_match_metrics(merged, "target_answer_parseable", "is_correct_target")
    true_metrics = summarize_parseable_match_metrics(merged, "true_target_parseable", "is_correct_true")

    teacher_by_op: Dict[str, float] = {}
    true_by_op: Dict[str, float] = {}
    for op in ["+", "-", "*", "/"]:
        op_df = merged[merged["op"] == op]
        teacher_by_op[op] = float(op_df["is_correct_target"].mean()) if len(op_df) > 0 else float("nan")
        true_by_op[op] = float(op_df["is_correct_true"].mean()) if len(op_df) > 0 else float("nan")

    metrics = {
        "mode": "uma_tuple",
        "primary_name": "teacher_tuple_acc_all",
        "primary_value": teacher["acc_all"],
        "primary_higher_is_better": True,
        "overall_acc": teacher["acc_all"],
        "teacher_acc_scored": teacher["acc_scored"],
        "teacher_acc_all": teacher["acc_all"],
        "teacher_coverage": teacher["coverage"],
        "teacher_n_total": teacher["n_total"],
        "teacher_n_scored": teacher["n_scored"],
        "teacher_n_correct": teacher["n_correct"],
        "teacher_by_op": teacher_by_op,
        "true_acc_scored": true_metrics["acc_scored"],
        "true_acc_all": true_metrics["acc_all"],
        "true_coverage": true_metrics["coverage"],
        "true_n_total": true_metrics["n_total"],
        "true_n_scored": true_metrics["n_scored"],
        "true_n_correct": true_metrics["n_correct"],
        "true_by_op": true_by_op,
        "n": int(len(merged)),
        "n_problems": int(merged["prob"].nunique()),
        "n_target_rows": int(len(target_df)),
    }
    return metrics, merged


def compute_sp2013_uma_distribution_metrics(out_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, Any]:
    work_df = out_df.copy()
    work_df["prob_key"] = work_df["prob"].astype(str).map(normalize_problem_key)

    target_rows = target_df.copy()
    target_dist_rows = []
    pred_dist_rows = []
    tv_rows = []
    for prob_key, group in work_df.groupby("prob_key"):
        target_group = target_rows[target_rows["prob_key"] == prob_key]
        if len(target_group) == 0:
            raise ValueError(f"Missing UMA distribution target for SP2013 problem: {prob_key}")
        target_dist = build_answer_distribution(target_group["target_answer_label"])
        pred_dist = build_answer_distribution(group["pred_answer_label"])
        tv = total_variation_distance(pred_dist, target_dist)
        op = str(group["op"].iloc[0])
        target_dist_rows.append({"prob": prob_key, "op": op, "target_distribution": target_dist})
        pred_dist_rows.append({"prob": prob_key, "op": op, "pred_distribution": pred_dist})
        tv_rows.append(
            {
                "prob": prob_key,
                "op": op,
                "tv_distance": float(tv),
                "n_rollouts": int(len(group)),
            }
        )

    tv_df = pd.DataFrame(tv_rows)
    by_op_tv: Dict[str, float] = {}
    for op in ["+", "-", "*", "/"]:
        op_df = tv_df[tv_df["op"] == op]
        by_op_tv[op] = float(op_df["tv_distance"].mean()) if len(op_df) > 0 else float("nan")

    parseable_cov = float(work_df["pred_answer_parseable"].mean()) if len(work_df) > 0 else float("nan")
    true_acc = float(work_df["is_correct_true"].mean()) if len(work_df) > 0 else float("nan")
    metrics = {
        "mode": "uma_distribution",
        "primary_name": "distribution_tv_mean",
        "primary_value": float(tv_df["tv_distance"].mean()) if len(tv_df) > 0 else float("nan"),
        "primary_higher_is_better": False,
        "overall_acc": float("nan"),
        "distribution_tv_mean": float(tv_df["tv_distance"].mean()) if len(tv_df) > 0 else float("nan"),
        "distribution_tv_by_op": by_op_tv,
        "parseable_coverage": parseable_cov,
        "true_acc_all": true_acc,
        "sample_parseable_n": int(work_df["pred_answer_parseable"].sum()) if len(work_df) > 0 else 0,
        "sample_total_n": int(len(work_df)),
        "n": int(len(work_df)),
        "n_problems": int(tv_df["prob"].nunique()) if len(tv_df) > 0 else 0,
        "num_rollouts": int(work_df.groupby("prob_key").size().iloc[0]) if len(work_df) > 0 else 0,
        "per_problem_tv": tv_df.to_dict(orient="records"),
    }
    return metrics


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
    for op in ["+", "-", "*", "/"]:
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
    for op in ["+", "-", "*", "/"]:
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
        "primary_name": "overall_acc",
        "primary_value": rollout_overall,
        "primary_higher_is_better": True,
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
    target_csv: str,
    max_new_tokens: int,
    target_mode: str = "true",
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
    fixed_student_prompt_tuple: Optional[Tuple[float, float, float, float]] = None,
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
    if fixed_student_prompt_tuple is not None and use_param_grid:
        raise ValueError("Fixed student-prompt tuple cannot be combined with --sp2013_use_param_grid.")
    if target_mode == "uma_tuple" and not use_param_grid:
        raise ValueError("SP2013 UMA tuple eval requires --sp2013_use_param_grid.")
    if target_mode == "uma_distribution" and use_param_grid:
        raise ValueError("SP2013 UMA distribution eval requires --no-sp2013_use_param_grid.")

    prompt_tuple: Optional[Tuple[float, float, float, float]] = None
    if fixed_student_prompt_tuple is not None:
        if len(fixed_student_prompt_tuple) != 4:
            raise ValueError("fixed_student_prompt_tuple must be a 4-tuple: (g, d, rt_mu, ice).")
        prompt_tuple = tuple(float(v) for v in fixed_student_prompt_tuple)

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
    target_df: Optional[pd.DataFrame] = None
    if target_mode in {"uma_tuple", "uma_distribution"}:
        target_df = load_sp2013_target_frame(target_csv)

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
            prompt_g, prompt_d, prompt_rt, prompt_ice = g, d, rt, ice
            if prompt_tuple is not None:
                prompt_g, prompt_d, prompt_rt, prompt_ice = prompt_tuple

            if use_param_grid or prompt_tuple is not None:
                prompt = build_student_prompt(
                    prob=prob, g=prompt_g, d=prompt_d, rt_mu=prompt_rt, ice=prompt_ice,
                    domain=getattr(args, "domain", "fraction"),
                )
            else:
                prompt = f"{domain_prompt_template(getattr(args, 'domain', 'fraction'))} {render_problem_for_prompt(prob)}=?"

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
                        pred_token = extract_answer_token(gen_text)
                        pred = extract_final_answer(gen_text)
                        is_correct = answers_match(pred, correct)
                        rows.append(
                            {
                                "prob": prob,
                                "op": op,
                                "g": prompt_g if (use_param_grid or prompt_tuple is not None) else np.nan,
                                "d": prompt_d if (use_param_grid or prompt_tuple is not None) else np.nan,
                                "rt_mu": prompt_rt if (use_param_grid or prompt_tuple is not None) else np.nan,
                                "ice": prompt_ice if (use_param_grid or prompt_tuple is not None) else np.nan,
                                "sample_idx": sample_idx,
                                "prompt_text": prompt,
                                "pred_answer_token": pred_token,
                                "pred_answer_label": canonicalize_answer_label(pred_token),
                                "pred_answer_parseable": int(answer_to_float(pred_token) is not None),
                                "pred_answer": pred,
                                "correct_answer": correct,
                                "is_correct_true": int(is_correct),
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
                pred_token = extract_answer_token(gen_text)
                pred = extract_final_answer(gen_text)
                is_correct = answers_match(pred, correct)
                rows.append(
                    {
                        "prob": prob,
                        "op": op,
                        "g": prompt_g if (use_param_grid or prompt_tuple is not None) else np.nan,
                        "d": prompt_d if (use_param_grid or prompt_tuple is not None) else np.nan,
                        "rt_mu": prompt_rt if (use_param_grid or prompt_tuple is not None) else np.nan,
                        "ice": prompt_ice if (use_param_grid or prompt_tuple is not None) else np.nan,
                        "sample_idx": 0,
                        "prompt_text": prompt,
                        "pred_answer_token": pred_token,
                        "pred_answer_label": canonicalize_answer_label(pred_token),
                        "pred_answer_parseable": int(answer_to_float(pred_token) is not None),
                        "pred_answer": pred,
                        "correct_answer": correct,
                        "is_correct_true": int(is_correct),
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
    if target_mode == "uma_tuple":
        assert target_df is not None
        metrics, out_df = compute_sp2013_uma_tuple_metrics(out_df, target_df)
    elif target_mode == "uma_distribution":
        assert target_df is not None
        metrics = compute_sp2013_uma_distribution_metrics(out_df, target_df)
    else:
        metrics = compute_rollout_accuracy_metrics(out_df, use_param_grid=use_param_grid)
    metrics["grid_size"] = int(len(param_grid))
    metrics["num_rollouts"] = int(num_rollouts)
    metrics["do_sample"] = bool(do_sample)
    metrics["temperature"] = float(temperature)
    metrics["top_p"] = float(top_p)
    metrics["top_k"] = int(top_k)
    metrics["sample_seed"] = int(sample_seed)
    metrics["rollout_batch_size"] = int(rollout_batch_size)
    metrics["target_mode"] = str(target_mode)
    metrics["sp2013_target_csv"] = target_csv if target_mode in {"uma_tuple", "uma_distribution"} else None
    metrics["fixed_student_prompt_tuple"] = list(prompt_tuple) if prompt_tuple is not None else None
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
    report_true_target: bool = False,
    batch_size: int = 64,
    prob_col: str = "prob",
):
    if eval_df is None or len(eval_df) == 0:
        empty_metrics = {
            "target_mode": target_mode,
            "acc_scored": float("nan"),
            "acc_all": float("nan"),
            "coverage": 0.0,
            "n_total": 0,
            "n_scored": 0,
            "n_correct": 0,
            "batch_size": int(batch_size),
            "report_true_target": bool(report_true_target),
        }
        if report_true_target:
            empty_metrics.update(
                {
                    "true_acc_scored": float("nan"),
                    "true_acc_all": float("nan"),
                    "true_coverage": 0.0,
                    "true_n_total": 0,
                    "true_n_scored": 0,
                    "true_n_correct": 0,
                }
            )
        return empty_metrics, pd.DataFrame()

    if sample_size is not None and sample_size > 0 and sample_size < len(eval_df):
        work_df = eval_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        work_df = eval_df.reset_index(drop=True)

    core = model.module if hasattr(model, "module") else model
    core.eval()
    batch_size = max(1, int(batch_size))

    rows = []
    for start in range(0, len(work_df), batch_size):
        batch_df = work_df.iloc[start : start + batch_size].reset_index(drop=True)
        prompts = batch_df[prompt_col].astype(str).str.strip().tolist()
        target_responses = batch_df[response_col].astype(str).tolist()
        probs = (
            batch_df[prob_col].astype(str).str.strip().tolist()
            if prob_col in batch_df.columns
            else [""] * len(batch_df)
        )
        probs = [prob if prob != "" else extract_prob_from_prompt(prompt) for prob, prompt in zip(probs, prompts)]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = core.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        gen_texts = tokenizer.batch_decode(out[:, prompt_width:], skip_special_tokens=True)

        for prompt, target_response, prob, gen_text in zip(prompts, target_responses, probs, gen_texts):
            response_gold_answer = extract_final_answer(target_response)
            true_gold_answer = compute_correct_answer(prob)
            response_parseable = answer_to_float(response_gold_answer) is not None
            true_parseable = answer_to_float(true_gold_answer) is not None
            pred_answer = extract_final_answer(gen_text)

            rows.append(
                {
                    "prob": prob,
                    "prompt": prompt,
                    "response_target_answer": response_gold_answer,
                    "response_target_parseable": int(response_parseable),
                    "true_target_answer": true_gold_answer,
                    "true_target_parseable": int(true_parseable),
                    "pred_answer": pred_answer,
                    "is_correct_response": int(answers_match(pred_answer, response_gold_answer)) if response_parseable else 0,
                    "is_correct_true": int(answers_match(pred_answer, true_gold_answer)) if true_parseable else 0,
                    "generation_text": gen_text.strip(),
                }
            )

    out_df = pd.DataFrame(rows)
    primary_prefix = "true" if target_mode == "true" else "response"
    primary_metrics = summarize_id_eval_target(out_df, primary_prefix)
    metrics = {
        "target_mode": target_mode,
        "batch_size": int(batch_size),
        "report_true_target": bool(report_true_target),
        **primary_metrics,
    }
    if report_true_target:
        true_metrics = summarize_id_eval_target(out_df, "true")
        metrics.update(
            {
                "true_acc_scored": true_metrics["acc_scored"],
                "true_acc_all": true_metrics["acc_all"],
                "true_coverage": true_metrics["coverage"],
                "true_n_total": true_metrics["n_total"],
                "true_n_scored": true_metrics["n_scored"],
                "true_n_correct": true_metrics["n_correct"],
            }
        )
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
        rank0_print(rank, f"sp2013_target_mode: {args.sp2013_target_mode}")
        rank0_print(rank, f"sp2013_target_csv: {args.sp2013_target_csv}")
        rank0_print(rank, f"sp2013_param_grid_eval: {args.sp2013_use_param_grid}")
        rank0_print(
            rank,
            "sp2013_sampling: "
            f"rollouts={args.sp2013_num_rollouts} do_sample={args.sp2013_do_sample} "
            f"temperature={args.sp2013_temperature} top_p={args.sp2013_top_p} "
            f"top_k={args.sp2013_top_k} rollout_batch={args.sp2013_rollout_batch_size}",
        )
        rank0_print(rank, f"evals_per_epoch: {args.evals_per_epoch}")
        resolved_loss_evals_per_epoch = args.loss_evals_per_epoch if args.loss_evals_per_epoch > 0 else args.evals_per_epoch
        rank0_print(rank, f"loss_evals_per_epoch: {resolved_loss_evals_per_epoch}")
        rank0_print(rank, f"lr: {args.lr}")
        rank0_print(rank, f"min_lr: {args.min_lr}")
        rank0_print(rank, f"lr_scheduler_type: {args.lr_scheduler_type}")
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
            f"(split={args.id_eval_split}, sample_size={args.id_eval_size}, target={args.id_eval_target}, "
            f"report_true_target={args.id_eval_report_true_target}, batch_size={args.id_eval_batch_size})",
        )
        rank0_print(rank, "")

        if args.evals_per_epoch < 1:
            raise ValueError("--evals_per_epoch must be >= 1.")
        if args.loss_evals_per_epoch < 0:
            raise ValueError("--loss_evals_per_epoch must be >= 0.")
        if args.id_eval_batch_size < 1:
            raise ValueError("--id_eval_batch_size must be >= 1.")
        if args.best_by in {"sp2013_acc", "sp2013_primary"} and (not args.eval_sp2013 or args.sp2013_every <= 0):
            raise ValueError("--best_by sp2013_acc/sp2013_primary requires SP2013 eval enabled.")
        if args.best_by == "id_val_acc":
            if (not args.eval_id_final_answer) or (args.id_eval_every <= 0) or (args.id_eval_split not in {"val", "both"}):
                raise ValueError(
                    "--best_by id_val_acc requires ID eval enabled with val split "
                    "(set --eval_id_final_answer, --id_eval_every > 0, and --id_eval_split val|both)."
                )
        if args.sp2013_num_rollouts < 1:
            raise ValueError("--sp2013_num_rollouts must be >= 1.")
        if args.sp2013_rollout_batch_size < 1:
            raise ValueError("--sp2013_rollout_batch_size must be >= 1.")
        if args.sp2013_target_mode == "uma_tuple" and not args.sp2013_use_param_grid:
            raise ValueError("--sp2013_target_mode uma_tuple requires --sp2013_use_param_grid.")
        if args.sp2013_target_mode == "uma_distribution" and args.sp2013_use_param_grid:
            raise ValueError("--sp2013_target_mode uma_distribution requires --no-sp2013_use_param_grid.")
        if args.sp2013_num_rollouts > 1 and not args.sp2013_do_sample:
            raise ValueError("--sp2013_num_rollouts > 1 requires --sp2013_do_sample.")
        if args.sp2013_target_mode in {"uma_tuple", "uma_distribution"} and not os.path.exists(args.sp2013_target_csv):
            raise FileNotFoundError(f"SP2013 target CSV not found: {args.sp2013_target_csv}")

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

        df = load_and_clean_trace_frame(
            csv_path=args.data_csv,
            usecols=usecols,
            prompt_col=args.prompt_col,
            response_col=args.response_col,
            problem_col=args.problem_col,
        )

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

        sp2013_probs: List[str] = []
        sp2013_val_source_df: Optional[pd.DataFrame] = None
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
            sp2013_probs = sp2013_ref[sp2013_problem_col].dropna().astype(str).str.strip().tolist()
            sp2013_prob_set = set(sp2013_probs)

            if args.sp2013_val_source_csv:
                sp2013_val_source_df = load_and_clean_trace_frame(
                    csv_path=args.sp2013_val_source_csv,
                    usecols=usecols,
                    prompt_col=args.prompt_col,
                    response_col=args.response_col,
                    problem_col=args.problem_col,
                )
                sp2013_val_source_df = sp2013_val_source_df[
                    sp2013_val_source_df[args.problem_col].isin(sp2013_prob_set)
                ].reset_index(drop=True)
                if len(sp2013_val_source_df) == 0:
                    raise ValueError(
                        "--sp2013_val_source_csv did not contain any SP2013 rows after cleaning: "
                        f"{args.sp2013_val_source_csv}"
                    )

            train_sp_mask = train_df[args.problem_col].isin(sp2013_prob_set)
            test_sp_mask = test_df[args.problem_col].isin(sp2013_prob_set)
            val_sp_mask = val_df[args.problem_col].isin(sp2013_prob_set)

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
            if sp2013_val_source_df is not None:
                val_df = val_df.loc[~val_sp_mask].reset_index(drop=True)
                val_df = pd.concat([val_df, sp2013_val_source_df], ignore_index=True)
            elif moved_frames:
                val_df = pd.concat([val_df] + moved_frames, ignore_index=True)

            final_val_sp_mask = val_df[args.problem_col].isin(sp2013_prob_set)
            final_val_sp_rows = int(final_val_sp_mask.sum())
            final_val_sp_probs = int(val_df.loc[final_val_sp_mask, args.problem_col].nunique())
            source_label = (
                f", val_source_rows_added={len(sp2013_val_source_df):,}"
                if sp2013_val_source_df is not None
                else ""
            )
            rank0_print(
                rank,
                (
                    "move_sp2013_rows_to_val: "
                    f"sp2013_probs={len(sp2013_prob_set):,}, "
                    f"moved_train_rows={moved_train_rows:,} ({train_sp_probs_before:,} probs), "
                    f"moved_test_rows={moved_test_rows:,} ({test_sp_probs_before:,} probs), "
                    f"val_sp_rows_before={val_sp_rows_before:,} ({val_sp_probs_before:,} probs), "
                    f"val_sp_rows_after={final_val_sp_rows:,} ({final_val_sp_probs:,} probs)"
                    f"{source_label}"
                ),
            )

        if len(train_df) == 0:
            raise ValueError("No training rows after split.")

        rank0_print(rank, f"rows total: {len(df):,}")
        rank0_print(rank, f"rows train: {len(train_df):,}")
        rank0_print(rank, f"rows val:   {len(val_df):,}")
        rank0_print(rank, f"rows test:  {len(test_df):,}")

        if args.move_sp2013_rows_to_val and args.val_frac == 0.0 and args.test_frac == 0.0:
            sp2013_validation_stats = assert_sp2013_only_validation_split(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                problem_col=args.problem_col,
                sp2013_probs=sp2013_probs,
                strict_layout=args.strict_sp2013_only_validation_layout,
            )
            val_param_tuple_label = (
                "n/a"
                if sp2013_validation_stats["val_param_tuple_count"] is None
                else f"{sp2013_validation_stats['val_param_tuple_count']:,}"
            )
            if args.strict_sp2013_only_validation_layout:
                rank0_print(
                    rank,
                    "sp2013_only_validation: confirmed val_rows=16,000, val_probs=16, "
                    "val_student_param_tuples=1,000, train_sp_rows=0, test_sp_rows=0",
                )
            else:
                rank0_print(
                    rank,
                    "sp2013_only_validation: relaxed layout accepted "
                    f"(val_rows={sp2013_validation_stats['val_rows']:,}, "
                    f"val_probs={sp2013_validation_stats['val_prob_count']:,}/"
                    f"{sp2013_validation_stats['ref_prob_count']:,}, "
                    f"val_student_param_tuples={val_param_tuple_label}, "
                    f"train_sp_rows={sp2013_validation_stats['train_sp_rows']:,}, "
                    f"test_sp_rows={sp2013_validation_stats['test_sp_rows']:,})",
                )

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
                    domain=args.domain,
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
        if resume_dir:
            saved_train_args = load_saved_train_args_if_exists(resume_dir)
            saved_scheduler_type = str(saved_train_args.get("lr_scheduler_type", "cosine"))
            if saved_scheduler_type != args.lr_scheduler_type:
                raise ValueError(
                    "Resume checkpoint scheduler mismatch: "
                    f"checkpoint uses {saved_scheduler_type}, requested {args.lr_scheduler_type}."
                )
            saved_min_lr = float(saved_train_args.get("min_lr", 0.0))
            if not math.isclose(saved_min_lr, args.min_lr, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "Resume checkpoint min_lr mismatch: "
                    f"checkpoint uses {saved_min_lr}, requested {args.min_lr}."
                )

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

        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

        pin = device.type == "cuda"
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
        train_batches_per_epoch = len(
            OffsetBatchSampler(
                sampler=build_train_index_sampler(
                    train_ds=train_ds,
                    distributed=distributed,
                    world_size=world_size,
                    rank=rank,
                    seed=args.seed,
                    epoch=1,
                ),
                batch_size=args.batch_size,
                drop_last=False,
                start_batch=0,
            )
        )
        updates_per_epoch = max(1, math.ceil(train_batches_per_epoch / args.grad_accum_steps))
        total_updates = updates_per_epoch * args.epochs
        warmup_steps = int(total_updates * args.warmup_ratio)
        rank0_print(rank, f"train batches/epoch: {train_batches_per_epoch}")
        rank0_print(rank, f"optimizer updates/epoch: {updates_per_epoch}")
        rank0_print(rank, f"total optimizer updates: {total_updates}")
        rank0_print(rank, f"warmup updates: {warmup_steps}")
        rank0_print(rank, "")
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=args.lr_scheduler_type,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_updates),
            base_lr=args.lr,
            min_lr=args.min_lr,
        )

        history: List[Dict] = []
        best_val = float("inf")
        if args.best_by == "sp2013_primary" and args.sp2013_target_mode == "uma_distribution":
            best_sp2013 = float("inf")
        else:
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
            resume_batches_in_epoch = resume_updates_in_epoch * args.grad_accum_steps if epoch == start_epoch else 0
            train_loader = build_train_loader_for_epoch(
                train_ds=train_ds,
                collate_fn=collate,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin,
                distributed=distributed,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                start_batch=resume_batches_in_epoch,
            )
            if resume_batches_in_epoch > 0:
                rank0_print(
                    rank,
                    (
                        f"epoch {epoch}: resuming from batch {resume_batches_in_epoch:,} "
                        "without replaying skipped dataloader work."
                    ),
                )
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum = 0.0
            train_steps = 0
            accum_counter = 0
            updates_in_epoch = resume_updates_in_epoch if epoch == start_epoch else 0
            loss_eval_targets = build_eval_targets(updates_per_epoch, resolved_loss_evals_per_epoch)
            full_eval_targets = build_eval_targets(updates_per_epoch, args.evals_per_epoch)

            for step, batch in enumerate(train_loader, start=1):
                accum_counter += 1

                take_step = (accum_counter >= args.grad_accum_steps) or (step == len(train_loader))

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

                run_full_eval = updates_in_epoch in full_eval_targets
                run_loss_eval = run_full_eval or (updates_in_epoch in loss_eval_targets)
                if not run_loss_eval:
                    continue

                local = torch.tensor([train_loss_sum, train_steps], dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(local, op=dist.ReduceOp.SUM)
                train_loss = float(local[0].item() / max(local[1].item(), 1.0))
                val_loss = evaluate(model, val_loader, device, amp_dtype, distributed) if len(val_ds) > 0 else float("nan")
                test_loss = (
                    evaluate(model, test_loader, device, amp_dtype, distributed)
                    if run_full_eval and len(test_ds) > 0
                    else float("nan")
                )

                sp2013_acc = float("nan")
                sp2013_by_op: Dict[str, float] = {}
                sp2013_primary = float("nan")
                sp2013_primary_name = "overall_acc"
                sp2013_primary_higher_is_better = True
                sp2013_metrics: Optional[Dict[str, Any]] = None
                sp2013_df: Optional[pd.DataFrame] = None
                id_val_acc = float("nan")
                id_val_coverage = float("nan")
                id_val_true_acc = float("nan")
                id_val_true_coverage = float("nan")
                id_val_metrics: Optional[Dict[str, Any]] = None
                id_val_df: Optional[pd.DataFrame] = None
                id_test_acc = float("nan")
                id_test_coverage = float("nan")
                id_test_true_acc = float("nan")
                id_test_true_coverage = float("nan")
                id_test_metrics: Optional[Dict[str, Any]] = None
                id_test_df: Optional[pd.DataFrame] = None
                eval_tag = f"e{epoch:02d}_u{updates_in_epoch:04d}"
                at_epoch_end = updates_in_epoch >= updates_per_epoch

                if run_full_eval:
                    distributed_barrier(distributed, device)

                if run_full_eval and rank == 0 and args.eval_sp2013 and args.sp2013_every > 0 and (epoch % args.sp2013_every == 0):
                    sp2013_metrics, sp2013_df = evaluate_sp2013_final_answer(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        sp2013_csv=args.sp2013_csv,
                        target_csv=args.sp2013_target_csv,
                        max_new_tokens=args.sp2013_max_new_tokens,
                        target_mode=args.sp2013_target_mode,
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
                    )
                    sp2013_acc = sp2013_metrics["overall_acc"]
                    sp2013_by_op = sp2013_metrics.get("by_op", sp2013_metrics.get("teacher_by_op", {}))
                    sp2013_primary = sp2013_metrics.get("primary_value", sp2013_acc)
                    sp2013_primary_name = sp2013_metrics.get("primary_name", "overall_acc")
                    sp2013_primary_higher_is_better = bool(sp2013_metrics.get("primary_higher_is_better", True))
                    sp2013_df.to_csv(os.path.join(args.output_dir, f"sp2013_{eval_tag}.csv"), index=False)
                    with open(os.path.join(args.output_dir, f"sp2013_metrics_{eval_tag}.json"), "w", encoding="utf-8") as f:
                        json.dump(sp2013_metrics, f, indent=2)
                    if at_epoch_end:
                        sp2013_df.to_csv(os.path.join(args.output_dir, f"sp2013_epoch_{epoch:02d}.csv"), index=False)
                        with open(os.path.join(args.output_dir, f"sp2013_metrics_epoch_{epoch:02d}.json"), "w", encoding="utf-8") as f:
                            json.dump(sp2013_metrics, f, indent=2)

                if run_full_eval and rank == 0 and args.eval_id_final_answer and args.id_eval_every > 0 and (epoch % args.id_eval_every == 0):
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
                            report_true_target=args.id_eval_report_true_target,
                            batch_size=args.id_eval_batch_size,
                        )
                        id_val_acc = id_val_metrics["acc_scored"]
                        id_val_coverage = id_val_metrics["coverage"]
                        id_val_true_acc = id_val_metrics.get("true_acc_scored", float("nan"))
                        id_val_true_coverage = id_val_metrics.get("true_coverage", float("nan"))
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
                            report_true_target=args.id_eval_report_true_target,
                            batch_size=args.id_eval_batch_size,
                        )
                        id_test_acc = id_test_metrics["acc_scored"]
                        id_test_coverage = id_test_metrics["coverage"]
                        id_test_true_acc = id_test_metrics.get("true_acc_scored", float("nan"))
                        id_test_true_coverage = id_test_metrics.get("true_coverage", float("nan"))
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
                    "eval_scope": "full" if run_full_eval else "loss_only",
                    "update_in_epoch": updates_in_epoch,
                    "updates_done": global_update,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "sp2013_acc": sp2013_acc,
                    "sp2013_primary": sp2013_primary,
                    "sp2013_primary_name": sp2013_primary_name,
                    "sp2013_primary_higher_is_better": sp2013_primary_higher_is_better,
                    "sp2013_by_op": sp2013_by_op,
                    "id_val_acc": id_val_acc,
                    "id_val_coverage": id_val_coverage,
                    "id_val_true_acc": id_val_true_acc,
                    "id_val_true_coverage": id_val_true_coverage,
                    "id_test_acc": id_test_acc,
                    "id_test_coverage": id_test_coverage,
                    "id_test_true_acc": id_test_true_acc,
                    "id_test_true_coverage": id_test_true_coverage,
                    "elapsed_sec": time.time() - t0,
                }

                if rank == 0:
                    history.append(eval_record)
                    msg = f"[eval {eval_record['eval_scope']} {eval_tag}] train={train_loss:.4f} val={val_loss:.4f}"
                    if is_finite_number(test_loss):
                        msg += f" test={test_loss:.4f}"
                    if run_full_eval:
                        msg += (
                            f" sp2013={sp2013_acc:.4f} sp2013_primary={sp2013_primary:.4f} "
                            f"({sp2013_primary_name}) id_val={id_val_acc:.4f}"
                        )
                        if is_finite_number(id_val_true_acc):
                            msg += f" id_val_true={id_val_true_acc:.4f}"
                    print(msg, flush=True)

                    if args.best_by in {"sp2013_acc", "sp2013_primary"}:
                        compare_value = sp2013_acc if args.best_by == "sp2013_acc" else sp2013_primary
                        higher_is_better = True if args.best_by == "sp2013_acc" else sp2013_primary_higher_is_better
                        if higher_is_better:
                            improved = is_finite_number(compare_value) and (
                                (compare_value > best_sp2013)
                                or (
                                    compare_value == best_sp2013
                                    and is_finite_number(val_loss)
                                    and val_loss < best_val
                                )
                            )
                        else:
                            improved = is_finite_number(compare_value) and (
                                (compare_value < best_sp2013)
                                or (
                                    compare_value == best_sp2013
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
                        elif args.best_by in {"sp2013_acc", "sp2013_primary"}:
                            compare_value = sp2013_acc if args.best_by == "sp2013_acc" else sp2013_primary
                            if is_finite_number(compare_value):
                                best_sp2013 = compare_value

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
                            write_best_eval_artifacts(
                                output_dir=args.output_dir,
                                eval_record=eval_record,
                                sp2013_metrics=sp2013_metrics,
                                sp2013_df=sp2013_df,
                                id_val_metrics=id_val_metrics,
                                id_val_df=id_val_df,
                                id_test_metrics=id_test_metrics,
                                id_test_df=id_test_df,
                            )
                            if args.best_by == "sp2013_acc":
                                print(
                                    f"  saved best checkpoint -> {best_dir} "
                                    f"(sp2013={sp2013_acc:.4f}, val={val_loss:.4f})",
                                    flush=True,
                                )
                            elif args.best_by == "sp2013_primary":
                                print(
                                    f"  saved best checkpoint -> {best_dir} "
                                    f"({sp2013_primary_name}={sp2013_primary:.4f}, val={val_loss:.4f})",
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
                        target_csv=args.sp2013_target_csv,
                        max_new_tokens=args.sp2013_max_new_tokens,
                        target_mode=args.sp2013_target_mode,
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
                    )
                    sp_df.to_csv(os.path.join(args.output_dir, "best_sp2013.csv"), index=False)
                    with open(os.path.join(args.output_dir, "best_sp2013_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(sp_metrics, f, indent=2)
                    loaded_summary["sp2013_acc"] = sp_metrics.get("overall_acc", float("nan"))
                    loaded_summary["sp2013_primary"] = sp_metrics.get("primary_value", float("nan"))
                    loaded_summary["sp2013_primary_name"] = sp_metrics.get("primary_name", "overall_acc")
                    print(
                        f"Reloaded-checkpoint SP2013 primary: {sp_metrics.get('primary_value', float('nan')):.4f} "
                        f"({sp_metrics.get('primary_name', 'overall_acc')})",
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
                            report_true_target=args.id_eval_report_true_target,
                            batch_size=args.id_eval_batch_size,
                        )
                        id_val_df.to_csv(os.path.join(args.output_dir, "best_id_val.csv"), index=False)
                        with open(os.path.join(args.output_dir, "best_id_val_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(id_val_metrics, f, indent=2)
                        loaded_summary["id_val_acc"] = id_val_metrics.get("acc_scored", float("nan"))
                        loaded_summary["id_val_coverage"] = id_val_metrics.get("coverage", float("nan"))
                        loaded_summary["id_val_true_acc"] = id_val_metrics.get("true_acc_scored", float("nan"))
                        loaded_summary["id_val_true_coverage"] = id_val_metrics.get("true_coverage", float("nan"))
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
                            report_true_target=args.id_eval_report_true_target,
                            batch_size=args.id_eval_batch_size,
                        )
                        id_test_df.to_csv(os.path.join(args.output_dir, "best_id_test.csv"), index=False)
                        with open(os.path.join(args.output_dir, "best_id_test_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(id_test_metrics, f, indent=2)
                        loaded_summary["id_test_acc"] = id_test_metrics.get("acc_scored", float("nan"))
                        loaded_summary["id_test_coverage"] = id_test_metrics.get("coverage", float("nan"))
                        loaded_summary["id_test_true_acc"] = id_test_metrics.get("true_acc_scored", float("nan"))
                        loaded_summary["id_test_true_coverage"] = id_test_metrics.get("true_coverage", float("nan"))
                        print(
                            f"Reloaded-checkpoint ID-test answer_acc: {id_test_metrics.get('acc_scored', float('nan')):.4f} "
                            f"(coverage={id_test_metrics.get('coverage', float('nan')):.4f})",
                            flush=True,
                        )

                with open(os.path.join(args.output_dir, "loaded_model_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(loaded_summary, f, indent=2)

                del reloaded_model

        if rank == 0:
            best_record: Optional[Dict[str, Any]] = None
            if history:
                if args.best_by == "val_loss":
                    candidates = [rec for rec in history if is_finite_number(rec.get("val_loss", float("nan")))]
                    if candidates:
                        best_record = min(candidates, key=lambda rec: (rec["val_loss"], -rec.get("updates_done", 0)))
                elif args.best_by == "sp2013_primary":
                    candidates = [rec for rec in history if is_finite_number(rec.get("sp2013_primary", float("nan")))]
                    if candidates:
                        higher_is_better = bool(candidates[0].get("sp2013_primary_higher_is_better", True))
                        if higher_is_better:
                            best_record = max(
                                candidates,
                                key=lambda rec: (
                                    rec["sp2013_primary"],
                                    -(rec["val_loss"] if is_finite_number(rec.get("val_loss", float("nan"))) else float("inf")),
                                    rec.get("updates_done", 0),
                                ),
                            )
                        else:
                            best_record = min(
                                candidates,
                                key=lambda rec: (
                                    rec["sp2013_primary"],
                                    rec["val_loss"] if is_finite_number(rec.get("val_loss", float("nan"))) else float("inf"),
                                    -rec.get("updates_done", 0),
                                ),
                            )
                else:
                    metric_key = "sp2013_acc" if args.best_by == "sp2013_acc" else "id_val_acc"
                    candidates = [rec for rec in history if is_finite_number(rec.get(metric_key, float("nan")))]
                    if candidates:
                        best_record = max(
                            candidates,
                            key=lambda rec: (
                                rec[metric_key],
                                -(rec["val_loss"] if is_finite_number(rec.get("val_loss", float("nan"))) else float("inf")),
                                rec.get("updates_done", 0),
                            ),
                        )

            def finite_or_none(value: Any) -> Optional[float]:
                try:
                    value_f = float(value)
                except Exception:
                    return None
                return value_f if is_finite_number(value_f) else None

            summary_metrics = {
                "best_by": args.best_by,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "lr_scheduler_type": args.lr_scheduler_type,
                "epochs": int(args.epochs),
                "loss_evals_per_epoch": int(resolved_loss_evals_per_epoch),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "checkpoint_source": resume_dir or model_source,
                "move_sp2013_rows_to_val": bool(args.move_sp2013_rows_to_val),
                "train_prompt_student_mode": args.train_prompt_student_mode,
                "sp2013_target_mode": args.sp2013_target_mode,
                "id_eval_target": args.id_eval_target,
                "id_eval_report_true_target": bool(args.id_eval_report_true_target),
                "best_checkpoint_dir": best_dir if os.path.isdir(best_dir) else None,
                "best_epoch": int(best_record["epoch"]) if best_record is not None else None,
                "best_eval_tag": best_record.get("eval_tag") if best_record is not None else None,
                "best_updates_done": int(best_record["updates_done"]) if best_record is not None else None,
                "best_val_loss": finite_or_none(best_record.get("val_loss")) if best_record is not None else None,
                "best_sp2013_acc": finite_or_none(best_record.get("sp2013_acc")) if best_record is not None else None,
                "best_sp2013_primary": finite_or_none(best_record.get("sp2013_primary")) if best_record is not None else None,
                "best_sp2013_primary_name": best_record.get("sp2013_primary_name") if best_record is not None else None,
                "best_id_val_acc": finite_or_none(best_record.get("id_val_acc")) if best_record is not None else None,
                "best_id_val_true_acc": finite_or_none(best_record.get("id_val_true_acc")) if best_record is not None else None,
                "best_id_test_acc": finite_or_none(best_record.get("id_test_acc")) if best_record is not None else None,
                "best_id_test_true_acc": finite_or_none(best_record.get("id_test_true_acc")) if best_record is not None else None,
            }
            with open(os.path.join(args.output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(summary_metrics, f, indent=2)

        if rank == 0:
            print("\nDone.", flush=True)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
