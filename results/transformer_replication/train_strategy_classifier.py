#!/usr/bin/env python3
"""
Train a BERT-style sequence classifier for UMA strategy prediction.

The default task is coarse family classification:
    KDON_AS, KDON_OG -> KDON
    CDON_AS, CDON_OG -> CDON
    ONOD_M, ONOD_OG -> ONOD
    ICDM_D, ICDM_OG -> ICDM
    CROP_M -> CROP

The script is designed for the translated UMA NLP CSV and keeps preprocessing
streaming/chunked so large compressed inputs can be prepared without loading the
full table into RAM at once.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


DEFAULT_INPUT_CSV = os.path.join(
    "results",
    "transformer_replication",
    "synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz",
)

FAMILY_CODES = {"KDON", "CDON", "ONOD", "ICDM", "CROP", "OTHER"}


@dataclass(frozen=True)
class PreparedPaths:
    train_jsonl_gz: Path
    val_jsonl_gz: Path
    test_jsonl_gz: Path
    manifest_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a strategy classifier on translated UMA text.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--prepared_manifest",
        type=str,
        default="",
        help="Optional prepared_manifest.json from a previous run; skips raw CSV preprocessing.",
    )
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument(
        "--text_mode",
        choices=["response", "prob_response", "instruction_response"],
        default="prob_response",
        help="How to build classifier input text.",
    )
    parser.add_argument(
        "--label_mode",
        choices=["family", "raw"],
        default="family",
        help="Predict coarse families or raw strategy codes.",
    )
    parser.add_argument(
        "--include_other",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep OTHER rows instead of dropping them.",
    )
    parser.add_argument(
        "--drop_answer_line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove lines starting with ### answer:/### correctness: from response text.",
    )
    parser.add_argument(
        "--split_mode",
        choices=["problem", "row"],
        default="problem",
        help="Deterministic split by problem string or by row/source id.",
    )
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Deterministic row-level subsample fraction applied before splitting.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional hard cap on kept rows after filtering/sampling.",
    )
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument(
        "--train_subsample_count",
        type=int,
        default=None,
        help="Optional exact cap on train examples after loading a prepared split; val/test stay fixed.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--scheduler",
        choices=["linear", "cosine"],
        default="linear",
        help="Learning-rate schedule.",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tokenize_num_proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--best_by",
        choices=["macro_f1", "accuracy", "balanced_accuracy", "val_loss"],
        default="macro_f1",
    )
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights from the train split.",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed precision mode on CUDA.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load tokenizer/model from local cache only.",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only create split JSONL files and manifest, then exit.",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_hash_01(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    out = str(value).strip()
    if out.lower() == "nan":
        return default
    return out if out != "" else default


def collapse_strategy(strategy_code: str, label_mode: str, include_other: bool) -> Optional[str]:
    code = safe_text(strategy_code, default="OTHER")
    if label_mode == "raw":
        if code == "OTHER" and not include_other:
            return None
        return code

    prefix = code.split("_", 1)[0]
    if prefix in FAMILY_CODES:
        label = prefix
    else:
        label = "OTHER"

    if label == "OTHER" and not include_other:
        return None
    return label


def strip_answer_lines(text: str) -> str:
    kept: List[str] = []
    for line in text.splitlines():
        lower = line.strip().lower()
        if lower.startswith("### answer:") or lower.startswith("### correctness:"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def build_classifier_text(
    prob: str,
    instruction_nl: str,
    response_nl: str,
    text_mode: str,
    drop_answer_line: bool,
) -> str:
    response = strip_answer_lines(response_nl) if drop_answer_line else response_nl.strip()
    problem_line = f"Solve this fraction problem: {prob}=?"

    if text_mode == "response":
        return response
    if text_mode == "prob_response":
        return f"Problem: {prob}=?\nResponse:\n{response}".strip()
    if text_mode == "instruction_response":
        prompt = instruction_nl.strip() if instruction_nl.strip() else problem_line
        return f"Prompt:\n{prompt}\nResponse:\n{response}".strip()
    raise ValueError(f"Unsupported text_mode: {text_mode}")


def resolve_output_dir(raw: str) -> Path:
    if raw.strip():
        return Path(raw).resolve()
    return Path(
        "results",
        "transformer_replication",
        f"strategy_classifier_{now_ts()}",
    ).resolve()


def required_columns_for_mode(text_mode: str) -> List[str]:
    cols = ["prob", "strategy", "response_nl"]
    if text_mode == "instruction_response":
        cols.append("instruction_nl")
    return cols


def choose_split(key: str, val_frac: float, test_frac: float) -> str:
    frac = stable_hash_01(key)
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "validation"
    return "train"


def resolve_row_uid(record: Dict[str, Any], global_row_idx: int) -> str:
    source_uid = safe_text(record.get("source_uid"), default="")
    if source_uid != "":
        return source_uid
    prob = safe_text(record.get("prob"), default="?")
    strategy = safe_text(record.get("strategy"), default="?")
    return f"row::{global_row_idx}::{prob}::{strategy}"


def prepared_paths(output_dir: Path) -> PreparedPaths:
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    return PreparedPaths(
        train_jsonl_gz=prepared_dir / "train.jsonl.gz",
        val_jsonl_gz=prepared_dir / "validation.jsonl.gz",
        test_jsonl_gz=prepared_dir / "test.jsonl.gz",
        manifest_json=prepared_dir / "prepared_manifest.json",
    )


def read_prepared_paths_from_manifest(manifest_path: Path) -> PreparedPaths:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    files = payload["prepared_files"]
    return PreparedPaths(
        train_jsonl_gz=Path(files["train"]).resolve(),
        val_jsonl_gz=Path(files["validation"]).resolve(),
        test_jsonl_gz=Path(files["test"]).resolve(),
        manifest_json=manifest_path.resolve(),
    )


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.val_frac < 1.0:
        raise ValueError(f"--val_frac must be in [0, 1), got {args.val_frac}")
    if not 0.0 <= args.test_frac < 1.0:
        raise ValueError(f"--test_frac must be in [0, 1), got {args.test_frac}")
    if args.val_frac + args.test_frac >= 1.0:
        raise ValueError("--val_frac + --test_frac must be < 1.0")
    if not 0.0 < args.sample_frac <= 1.0:
        raise ValueError(f"--sample_frac must be in (0, 1], got {args.sample_frac}")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.epochs < 1 and not args.prepare_only:
        raise ValueError("--epochs must be >= 1 unless --prepare_only is set")
    if args.train_subsample_count is not None and args.train_subsample_count < 1:
        raise ValueError("--train_subsample_count must be >= 1")


def preprocess_to_jsonl(args: argparse.Namespace, output_dir: Path) -> PreparedPaths:
    paths = prepared_paths(output_dir)
    input_path = Path(args.input_csv).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    header_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
    needed = required_columns_for_mode(args.text_mode)
    for col in needed:
        if col not in header_cols:
            raise ValueError(f"Required column '{col}' not found in {input_path}")

    optional_cols = [col for col in ["instruction_nl", "source_uid"] if col in header_cols and col not in needed]
    usecols = needed + optional_cols

    split_file_map = {
        "train": paths.train_jsonl_gz,
        "validation": paths.val_jsonl_gz,
        "test": paths.test_jsonl_gz,
    }
    writers: Dict[str, Any] = {}
    handles: Dict[str, Any] = {}
    for split_name, split_path in split_file_map.items():
        handles[split_name] = gzip.open(split_path, "wt", encoding="utf-8")
        writers[split_name] = handles[split_name]

    total_read = 0
    total_kept = 0
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    global_row_idx = 0

    try:
        for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=args.chunksize):
            records = chunk.to_dict(orient="records")
            for record in records:
                total_read += 1
                row_uid = resolve_row_uid(record, global_row_idx)
                global_row_idx += 1

                if args.sample_frac < 1.0 and stable_hash_01(f"sample::{row_uid}") >= args.sample_frac:
                    continue

                label_text = collapse_strategy(
                    safe_text(record.get("strategy"), default="OTHER"),
                    label_mode=args.label_mode,
                    include_other=bool(args.include_other),
                )
                if label_text is None:
                    continue

                prob = safe_text(record.get("prob"), default="?")
                instruction_nl = safe_text(record.get("instruction_nl"), default="")
                response_nl = safe_text(record.get("response_nl"), default="")
                text = build_classifier_text(
                    prob=prob,
                    instruction_nl=instruction_nl,
                    response_nl=response_nl,
                    text_mode=args.text_mode,
                    drop_answer_line=bool(args.drop_answer_line),
                )
                if text == "":
                    continue

                split_key = prob if args.split_mode == "problem" else row_uid
                split_name = choose_split(split_key, val_frac=args.val_frac, test_frac=args.test_frac)
                payload = {
                    "text": text,
                    "label_text": label_text,
                    "prob": prob,
                    "orig_strategy": safe_text(record.get("strategy"), default="OTHER"),
                    "row_uid": row_uid,
                }
                writers[split_name].write(json.dumps(payload, ensure_ascii=True) + "\n")

                total_kept += 1
                split_counts[split_name] += 1
                label_counts[label_text] += 1
                split_label_counts[split_name][label_text] += 1

                if args.max_rows is not None and total_kept >= args.max_rows:
                    break

            if args.max_rows is not None and total_kept >= args.max_rows:
                break
    finally:
        for handle in handles.values():
            handle.close()

    if total_kept == 0:
        raise ValueError("No rows were kept after filtering/sampling. Relax the settings and try again.")

    label_list = sorted(label_counts.keys())
    missing_train = [label for label in label_list if split_label_counts["train"].get(label, 0) == 0]
    if missing_train:
        raise ValueError(
            "The train split is missing labels after filtering/splitting: "
            + ", ".join(missing_train)
        )

    manifest = {
        "input_csv": str(input_path),
        "text_mode": args.text_mode,
        "label_mode": args.label_mode,
        "include_other": bool(args.include_other),
        "drop_answer_line": bool(args.drop_answer_line),
        "split_mode": args.split_mode,
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "sample_frac": float(args.sample_frac),
        "max_rows": args.max_rows,
        "chunksize": int(args.chunksize),
        "total_rows_read": int(total_read),
        "total_rows_kept": int(total_kept),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "split_label_counts": {split: dict(counter) for split, counter in split_label_counts.items()},
        "label_list": label_list,
        "prepared_files": {
            "train": str(paths.train_jsonl_gz),
            "validation": str(paths.val_jsonl_gz),
            "test": str(paths.test_jsonl_gz),
        },
    }
    with paths.manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return paths


def load_prepared_dataset(paths: PreparedPaths, label_list: List[str], tokenizer, args: argparse.Namespace) -> DatasetDict:
    data_files = {
        "train": str(paths.train_jsonl_gz),
        "validation": str(paths.val_jsonl_gz),
        "test": str(paths.test_jsonl_gz),
    }
    ds = load_dataset("json", data_files=data_files)
    if args.train_subsample_count is not None:
        keep_n = min(int(args.train_subsample_count), len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(keep_n))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    def encode_label(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"labels": label_to_id[example["label_text"]]}

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    ds = ds.map(encode_label, desc="encoding_labels")
    ds = ds.map(
        tokenize_batch,
        batched=True,
        num_proc=max(1, int(args.tokenize_num_proc)),
        desc="tokenizing",
    )
    remove_cols = [col for col in ["text", "label_text", "prob", "orig_strategy", "row_uid"] if col in ds["train"].column_names]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
    return ds


def compute_class_weights(label_list: List[str], train_label_counts: Dict[str, int]) -> torch.Tensor:
    counts = np.array([max(1, int(train_label_counts.get(label, 0))) for label in label_list], dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def label_counts_from_dataset(dataset_split, label_list: List[str]) -> Dict[str, int]:
    counts = Counter(dataset_split["labels"])
    return {label: int(counts.get(idx, 0)) for idx, label in enumerate(label_list)}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_amp_dtype(mode: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_dataloader(dataset_split, collator, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def per_label_metrics(labels: np.ndarray, preds: np.ndarray, label_list: List[str]) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(label_list))),
        zero_division=0,
    )
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(label_list):
        metrics[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return metrics


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
    label_list: List[str],
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            with build_autocast_context(device, amp_dtype):
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    labels_np = np.asarray(all_labels, dtype=np.int64)
    preds_np = np.asarray(all_preds, dtype=np.int64)
    conf = confusion_matrix(labels_np, preds_np, labels=list(range(len(label_list))))
    metrics = {
        "n_examples": int(total_examples),
        "val_loss": float(total_loss / max(total_examples, 1)),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "per_label": per_label_metrics(labels_np, preds_np, label_list),
        "confusion_matrix": conf.tolist(),
        "label_order": label_list,
    }
    return metrics


def save_confusion_csv(path: Path, confusion_rows: List[List[int]], label_list: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold_pred"] + label_list)
        for label, row in zip(label_list, confusion_rows):
            writer.writerow([label] + row)


def metric_value(metrics: Dict[str, Any], best_by: str) -> float:
    if best_by == "val_loss":
        return -float(metrics["val_loss"])
    return float(metrics[best_by])


def save_checkpoint(
    path: Path,
    model,
    tokenizer,
    label_list: List[str],
    args: argparse.Namespace,
    metadata: Dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    save_json(
        path / "classifier_metadata.json",
        {
            "label_list": label_list,
            "label_to_id": {label: idx for idx, label in enumerate(label_list)},
            "args": vars(args),
            "metadata": metadata,
        },
    )


def train_classifier(args: argparse.Namespace, output_dir: Path, prepared: PreparedPaths) -> None:
    with prepared.manifest_json.open("r", encoding="utf-8") as f:
        prep_manifest = json.load(f)

    label_list = list(prep_manifest["label_list"])
    tokenizer_name = args.tokenizer_name.strip() or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=args.local_files_only)
    dataset = load_prepared_dataset(prepared, label_list=label_list, tokenizer=tokenizer, args=args)
    train_label_counts = label_counts_from_dataset(dataset["train"], label_list)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = build_dataloader(
        dataset["train"],
        collator=collator,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        dataset["validation"],
        collator=collator,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_dataloader(
        dataset["test"],
        collator=collator,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label={idx: label for idx, label in enumerate(label_list)},
        label2id={label: idx for idx, label in enumerate(label_list)},
        local_files_only=args.local_files_only,
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    if args.class_weighting:
        class_weights = compute_class_weights(label_list, train_label_counts).to(device)
    else:
        class_weights = torch.ones(len(label_list), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    save_json(output_dir / "train_args.json", vars(args))
    save_json(
        output_dir / "label_space.json",
        {
            "label_list": label_list,
            "train_label_counts": train_label_counts,
            "class_weights": class_weights.detach().cpu().tolist(),
            "prepared_manifest": str(prepared.manifest_json),
            "prepared_split_counts": prep_manifest.get("split_counts", {}),
            "train_subsample_count": args.train_subsample_count,
            "effective_train_examples": int(len(dataset["train"])),
            "effective_validation_examples": int(len(dataset["validation"])),
            "effective_test_examples": int(len(dataset["test"])),
        },
    )

    history: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            batch_size = int(labels.shape[0])

            with build_autocast_context(device, amp_dtype):
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss_for_backward = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            running_loss += float(loss.item()) * batch_size
            running_examples += batch_size

            should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if args.log_every > 0 and step % args.log_every == 0:
                avg_loss = running_loss / max(running_examples, 1)
                print(
                    f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )

        train_loss = running_loss / max(running_examples, 1)
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
            label_list=label_list,
            amp_dtype=amp_dtype,
        )

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["val_loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "macro_f1": float(val_metrics["macro_f1"]),
            "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "elapsed_sec": float(time.time() - t0),
        }
        history.append(epoch_metrics)
        print(
            f"[eval] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} bal_acc={val_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

        score = metric_value(val_metrics, args.best_by)
        is_best = best_score is None or score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best",
                model=model,
                tokenizer=tokenizer,
                label_list=label_list,
                args=args,
                metadata={"epoch": epoch, "val_metrics": val_metrics},
            )
            save_json(output_dir / "best_val_metrics.json", val_metrics)
            save_confusion_csv(
                output_dir / "best_val_confusion.csv",
                val_metrics["confusion_matrix"],
                label_list,
            )

        save_json(output_dir / "history.json", {"history": history, "best_epoch": best_epoch})

    save_checkpoint(
        output_dir / "last",
        model=model,
        tokenizer=tokenizer,
        label_list=label_list,
        args=args,
        metadata={"epoch": args.epochs},
    )

    best_dir = output_dir / "best"
    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir, local_files_only=True).to(device)
    best_model.eval()
    test_metrics = evaluate_model(
        model=best_model,
        dataloader=test_loader,
        device=device,
        loss_fn=loss_fn,
        label_list=label_list,
        amp_dtype=amp_dtype,
    )
    save_json(
        output_dir / "test_metrics.json",
        {
            **test_metrics,
            "best_epoch": best_epoch,
            "best_by": args.best_by,
        },
    )
    save_confusion_csv(
        output_dir / "test_confusion.csv",
        test_metrics["confusion_matrix"],
        label_list,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepared_manifest.strip():
        prepared = read_prepared_paths_from_manifest(Path(args.prepared_manifest).resolve())
    else:
        prepared = preprocess_to_jsonl(args, output_dir)
    print("=" * 70, flush=True)
    print("STRATEGY CLASSIFIER", flush=True)
    print("=" * 70, flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"prepared_manifest: {prepared.manifest_json}", flush=True)
    print(f"model_name: {args.model_name}", flush=True)
    print(f"text_mode: {args.text_mode}", flush=True)
    print(f"label_mode: {args.label_mode}", flush=True)
    print(f"split_mode: {args.split_mode}", flush=True)
    print("", flush=True)

    if args.prepare_only:
        print("prepare_only=true; exiting after preprocessing.", flush=True)
        return

    train_classifier(args, output_dir=output_dir, prepared=prepared)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
