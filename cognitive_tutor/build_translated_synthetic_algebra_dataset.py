#!/usr/bin/env python3
"""Package translated synthetic algebra traces into train-ready CSV splits."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, TextIO


DEFAULT_INPUT_CSV = Path(
    "results/UMA_replication/algebra_uma_k100_v3_ct_flow/"
    "algebra_uma_synth_matched_master_10k_k100_v3_ct_flow_nlp_human_mixed.csv.gz"
)
DEFAULT_OUT_DIR = Path(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "translated_synthetic_algebra_v3_human_mixed"
)
DEFAULT_PREFIX = "singlevar_algebra_synth_v3_human_mixed"

REQUIRED_COLUMNS = [
    "subjid",
    "prob",
    "equation_type",
    "variable",
    "answer",
    "correct",
    "instruction_nl",
    "response_nl",
]

OUTPUT_COLUMNS = [
    "source_uid",
    "split",
    "subjid",
    "prob",
    "equation_type",
    "variable",
    "answer",
    "correct",
    "g",
    "d",
    "c",
    "rt_mu",
    "ice",
    "instruction_nl",
    "response_nl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--test-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--preview-rows", type=int, default=24)
    return parser.parse_args()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def safe_text(value: object) -> str:
    return str(value or "").strip()


def validate_fracs(val_frac: float, test_frac: float) -> None:
    if val_frac < 0 or test_frac < 0:
        raise ValueError("val/test fractions must be non-negative")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")


def hash_unit(text: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_for_problem(prob: str, seed: int, val_frac: float, test_frac: float) -> str:
    value = hash_unit(prob, seed)
    if value < val_frac:
        return "val"
    if value < val_frac + test_frac:
        return "test"
    return "train"


def source_uid(row: dict[str, str]) -> str:
    key = "|".join(
        [
            safe_text(row.get("subjid")),
            safe_text(row.get("prob")),
            safe_text(row.get("answer")),
            safe_text(row.get("correct")),
            safe_text(row.get("response_nl")),
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def normalize_row(row: dict[str, str], split: str) -> dict[str, str]:
    out = {column: safe_text(row.get(column)) for column in OUTPUT_COLUMNS}
    out["source_uid"] = source_uid(row)
    out["split"] = split
    return out


def init_stats() -> dict[str, object]:
    return {
        "rows": 0,
        "correct_rows": 0,
        "unique_probs": set(),
        "unique_subjids": set(),
        "equation_type_counts": Counter(),
        "correct_by_equation_type": Counter(),
        "response_missing_answer_marker": 0,
        "instruction_empty": 0,
        "response_empty": 0,
        "response_lengths": [],
    }


def update_stats(stats: dict[str, object], row: dict[str, str]) -> None:
    stats["rows"] = int(stats["rows"]) + 1
    correct = 1 if safe_text(row.get("correct")) == "1" else 0
    stats["correct_rows"] = int(stats["correct_rows"]) + correct
    stats["unique_probs"].add(safe_text(row.get("prob")))  # type: ignore[union-attr]
    stats["unique_subjids"].add(safe_text(row.get("subjid")))  # type: ignore[union-attr]
    equation_type = safe_text(row.get("equation_type"))
    stats["equation_type_counts"][equation_type] += 1  # type: ignore[index]
    if correct:
        stats["correct_by_equation_type"][equation_type] += 1  # type: ignore[index]
    if "### answer:" not in safe_text(row.get("response_nl")):
        stats["response_missing_answer_marker"] = int(stats["response_missing_answer_marker"]) + 1
    if not safe_text(row.get("instruction_nl")):
        stats["instruction_empty"] = int(stats["instruction_empty"]) + 1
    response = safe_text(row.get("response_nl"))
    if not response:
        stats["response_empty"] = int(stats["response_empty"]) + 1
    stats["response_lengths"].append(len(response))  # type: ignore[union-attr]


def finalize_stats(stats: dict[str, object]) -> dict[str, object]:
    rows = int(stats["rows"])
    response_lengths = list(stats["response_lengths"])  # type: ignore[arg-type]
    equation_type_counts: Counter[str] = stats["equation_type_counts"]  # type: ignore[assignment]
    correct_by_equation_type: Counter[str] = stats["correct_by_equation_type"]  # type: ignore[assignment]
    return {
        "rows": rows,
        "accuracy_rate": int(stats["correct_rows"]) / rows if rows else None,
        "unique_probs": len(stats["unique_probs"]),  # type: ignore[arg-type]
        "unique_subjids": len(stats["unique_subjids"]),  # type: ignore[arg-type]
        "equation_type_counts": dict(equation_type_counts),
        "accuracy_by_equation_type": {
            equation_type: correct_by_equation_type[equation_type] / count
            for equation_type, count in equation_type_counts.items()
            if count
        },
        "response_missing_answer_marker": int(stats["response_missing_answer_marker"]),
        "instruction_empty": int(stats["instruction_empty"]),
        "response_empty": int(stats["response_empty"]),
        "response_length_chars": summarize_lengths(response_lengths),
    }


def summarize_lengths(values: list[int]) -> dict[str, object]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "mean": statistics.fmean(sorted_values),
        "p50": percentile(sorted_values, 0.50),
        "p95": percentile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def validate_columns(fieldnames: Iterable[str] | None, input_csv: Path) -> None:
    columns = set(fieldnames or [])
    missing = sorted(set(REQUIRED_COLUMNS).difference(columns))
    if missing:
        raise ValueError(f"{input_csv} missing required columns: {missing}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_dataset(
    input_csv: Path = DEFAULT_INPUT_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    prefix: str = DEFAULT_PREFIX,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    seed: int = 20260428,
    max_rows: int | None = None,
    preview_rows: int = 24,
) -> dict[str, object]:
    validate_fracs(val_frac, test_frac)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        split: out_dir / f"{prefix}_{split}.csv.gz"
        for split in ["train", "val", "test"]
    }
    preview_path = out_dir / f"{prefix}_preview.csv"
    audit_path = out_dir / f"{prefix}_audit.json"
    manifest_path = out_dir / "manifest.json"

    stats = {split: init_stats() for split in ["train", "val", "test"]}
    global_stats = init_stats()
    preview: list[dict[str, str]] = []
    problem_splits: dict[str, str] = {}
    rows_read = 0

    handles = {split: open_text(path, "wt") for split, path in split_paths.items()}
    try:
        writers = {split: csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS) for split, handle in handles.items()}
        for writer in writers.values():
            writer.writeheader()

        with open_text(input_csv, "rt") as in_handle:
            reader = csv.DictReader(in_handle)
            validate_columns(reader.fieldnames, input_csv)
            for row in reader:
                prob = safe_text(row.get("prob"))
                split = split_for_problem(prob, seed=seed, val_frac=val_frac, test_frac=test_frac)
                problem_splits[prob] = split
                output_row = normalize_row(row, split)
                writers[split].writerow(output_row)
                update_stats(stats[split], output_row)
                update_stats(global_stats, output_row)
                if len(preview) < preview_rows:
                    preview.append(output_row)
                rows_read += 1
                if max_rows is not None and rows_read >= max_rows:
                    break
    finally:
        for handle in handles.values():
            handle.close()

    with preview_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(preview)

    audit = {
        "source_csv": str(input_csv),
        "out_dir": str(out_dir),
        "prefix": prefix,
        "definition": (
            "Translated synthetic single-variable algebra distillation dataset. "
            "Rows come from v3 CT-flow synthetic traces translated to human_mixed NLP. "
            "Splits are deterministic by problem string to avoid equation overlap."
        ),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "max_rows": max_rows,
        "columns": OUTPUT_COLUMNS,
        "splits": {
            split: {
                "path": str(split_paths[split]),
                **finalize_stats(split_stats),
            }
            for split, split_stats in stats.items()
        },
        "global": finalize_stats(global_stats),
        "problem_split_counts": dict(Counter(problem_splits.values())),
        "preview_csv": str(preview_path),
    }
    write_json(audit_path, audit)

    manifest = {
        "name": prefix,
        "type": "translated_synthetic_algebra_distillation",
        "primary_train_csv": str(split_paths["train"]),
        "validation_csv": str(split_paths["val"]),
        "test_csv": str(split_paths["test"]),
        "audit_json": str(audit_path),
        "preview_csv": str(preview_path),
        "prompt_col": "instruction_nl",
        "response_col": "response_nl",
        "problem_col": "prob",
        "split_policy": "deterministic_hash_by_problem",
        "source_csv": str(input_csv),
    }
    write_json(manifest_path, manifest)
    audit["manifest_json"] = str(manifest_path)
    return audit


def main() -> None:
    args = parse_args()
    audit = build_dataset(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        max_rows=args.max_rows,
        preview_rows=args.preview_rows,
    )
    print(f"out_dir={audit['out_dir']}", flush=True)
    for split, split_stats in audit["splits"].items():
        print(
            f"{split}: rows={split_stats['rows']} "
            f"unique_probs={split_stats['unique_probs']} "
            f"acc={split_stats['accuracy_rate']:.4f}",
            flush=True,
        )
    print(f"manifest_json={audit['manifest_json']}", flush=True)


if __name__ == "__main__":
    main()
