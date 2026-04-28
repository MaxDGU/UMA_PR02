#!/usr/bin/env python3
"""Build single-variable algebra human traces in a UMA-compatible schema."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cognitive_tutor.algebra_uma import (
    UMA_TRACE_COLUMNS,
    build_correct_trace,
    dedupe_preserve,
    format_fraction,
    get_equation_metadata,
)
from cognitive_tutor.build_trace_firstshot_dataset import safe_int


DEFAULT_DATASET = "algebra_2006_2007"
DEFAULT_INPUT_ROOT = Path("cognitive_tutor/data/extracted")
DEFAULT_PROCESSED_DIR = Path("cognitive_tutor/data/processed/algebra_2006_2007")
DEFAULT_OUT_DIR = Path("cognitive_tutor/data/processed/algebra_2006_2007")
DEFAULT_SPLITS = ("train", "master")
DEFAULT_INPUT_PREFIX = "trace_firstshot_math_only"

EXTRA_HUMAN_COLUMNS = [
    "problem_name",
    "problem_hierarchy",
    "problem_view",
    "target_kind",
    "final_answer",
    "first_wrong_step_index",
    "source_rows",
    "raw_kc_labels",
    "instruction_nl",
    "response_nl",
]

HUMAN_UMA_COLUMNS = UMA_TRACE_COLUMNS + EXTRA_HUMAN_COLUMNS

PROBLEM_SET_COLUMNS = [
    "prob",
    "equation_text",
    "equation_type",
    "variable",
    "solution",
    "source_split",
    "n_attempts",
    "accuracy_rate",
]


@dataclass(frozen=True)
class RawStep:
    row_id: int
    step_name: str
    correct_first_attempt: int
    incorrects: int
    hints: int
    kc_text: str
    kc_labels: list[str]


def parse_kc_labels(kc_text: str) -> list[str]:
    labels: list[str] = []
    for piece in str(kc_text or "").split("~~"):
        for marker in ("SkillRule:", "Rule:"):
            if marker not in piece:
                continue
            after = piece.split(marker, 1)[1]
            label = after.split(";", 1)[0].split("(", 1)[0].split("]", 1)[0].strip()
            if label:
                labels.append(label)
            break
    return dedupe_preserve(labels)


def read_raw_steps(path: Path) -> dict[int, RawStep]:
    raw_steps: dict[int, RawStep] = {}
    with path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for fallback_row_id, row in enumerate(reader, start=1):
            row_id = safe_int(row.get("Row", ""), fallback_row_id)
            kc_text = str(row.get("KC(Default)", "") or "")
            raw_steps[row_id] = RawStep(
                row_id=row_id,
                step_name=str(row.get("Step Name", "") or "").strip(),
                correct_first_attempt=safe_int(row.get("Correct First Attempt", "")),
                incorrects=safe_int(row.get("Incorrects", "")),
                hints=safe_int(row.get("Hints", "")),
                kc_text=kc_text,
                kc_labels=parse_kc_labels(kc_text),
            )
    return raw_steps


def load_json_list(text: object) -> list[object]:
    if text is None:
        return []
    value = str(text).strip()
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def build_trace_steps_json(
    trace_steps: Sequence[str],
    source_rows: Sequence[object],
    raw_steps: dict[int, RawStep],
) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    all_labels: list[str] = []
    for idx, step in enumerate(trace_steps, start=1):
        row_id = safe_int(source_rows[idx - 1], 0) if idx - 1 < len(source_rows) else 0
        raw = raw_steps.get(row_id)
        labels = raw.kc_labels if raw else []
        all_labels.extend(labels)
        records.append(
            {
                "step_index": idx,
                "state": step,
                "source_row": row_id,
                "kc_labels": labels,
                "kc_text": raw.kc_text if raw else "",
                "correct_first_attempt": raw.correct_first_attempt if raw else "",
                "incorrects": raw.incorrects if raw else "",
                "hints": raw.hints if raw else "",
            }
        )
    return records, dedupe_preserve(all_labels)


def human_row_to_uma(
    row: dict[str, str],
    raw_steps: dict[int, RawStep],
    split: str,
    seed: int,
) -> dict[str, object] | None:
    meta = get_equation_metadata(row.get("prob", ""))
    if meta is None:
        return None

    trace_steps = [str(step) for step in load_json_list(row.get("trace", ""))]
    source_rows = load_json_list(row.get("source_rows", ""))
    trace_records, kc_labels = build_trace_steps_json(trace_steps, source_rows, raw_steps)
    correct_trace = build_correct_trace(meta)
    exec_rules = kc_labels or correct_trace.exec_rules
    acc = safe_int(row.get("acc", ""))
    first_wrong_step = "" if acc else row.get("first_wrong_step", "") or row.get("resp", "")
    solution = format_fraction(meta.solution)
    answer = solution if acc else first_wrong_step
    raw_kc_labels = " | ".join(kc_labels)
    strategy = row.get("strategy", "") or correct_trace.strategy
    response_nl = f"{strategy}\n### answer: {answer}" if strategy else f"### answer: {answer}"

    output: dict[str, object] = {
        "subjid": row.get("subjid", ""),
        "seed": seed,
        "prob": meta.equation_text,
        "operation": meta.equation_type,
        "equation_type": meta.equation_type,
        "variable": meta.variable,
        "solution": solution,
        "strategy": strategy,
        "goals": " | ".join(correct_trace.goals),
        "exec": " | ".join(exec_rules),
        "answer": answer,
        "correct": acc,
        "is_correct": acc,
        "g": "",
        "d": "",
        "c": "",
        "rt_mu": "",
        "ice": "",
        "first_wrong_step": first_wrong_step,
        "trace_step_count": len(trace_records),
        "trace_rules_full": raw_kc_labels or " | ".join(correct_trace.exec_rules),
        "trace_steps_json": json.dumps(trace_records, separators=(",", ":")),
        "source": f"human_{DEFAULT_DATASET}_{split}",
        "problem_name": row.get("problem_name", row.get("prob", "")),
        "problem_hierarchy": row.get("problem_hierarchy", ""),
        "problem_view": row.get("problem_view", ""),
        "target_kind": row.get("target_kind", ""),
        "final_answer": solution,
        "first_wrong_step_index": row.get("first_wrong_step_index", ""),
        "source_rows": json.dumps(source_rows, separators=(",", ":")),
        "raw_kc_labels": raw_kc_labels,
        "instruction_nl": f"Solve this algebra problem: {meta.equation_text}.",
        "response_nl": response_nl,
    }
    return output


def read_processed_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(input_rows: int, rows: Sequence[dict[str, object]]) -> dict[str, object]:
    acc_values = [safe_int(row["correct"]) for row in rows]
    return {
        "input_rows": input_rows,
        "retained_rows": len(rows),
        "retained_rate": len(rows) / input_rows if input_rows else None,
        "accuracy_rate": statistics.fmean(acc_values) if acc_values else None,
        "equation_type_counts": dict(Counter(str(row["equation_type"]) for row in rows)),
        "target_counts": dict(Counter("correct" if safe_int(row["correct"]) else "stuck" for row in rows)),
    }


def build_problem_set(rows: Sequence[dict[str, object]], split: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prob"])].append(dict(row))

    problem_rows: list[dict[str, object]] = []
    for prob, prob_rows in sorted(grouped.items()):
        acc_values = [safe_int(row["correct"]) for row in prob_rows]
        first = prob_rows[0]
        problem_rows.append(
            {
                "prob": prob,
                "equation_text": prob,
                "equation_type": first["equation_type"],
                "variable": first["variable"],
                "solution": first["solution"],
                "source_split": split,
                "n_attempts": len(prob_rows),
                "accuracy_rate": statistics.fmean(acc_values) if acc_values else "",
            }
        )
    return problem_rows


def build_split(
    dataset: str,
    split: str,
    input_root: Path,
    processed_dir: Path,
    out_dir: Path,
    input_prefix: str,
) -> dict[str, object]:
    processed_path = processed_dir / f"{input_prefix}_{split}.csv"
    raw_path = input_root / dataset / f"{dataset}_{split}.txt"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed split: {processed_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw split: {raw_path}")

    processed_rows = read_processed_rows(processed_path)
    raw_steps = read_raw_steps(raw_path)
    output_rows = [
        converted
        for idx, row in enumerate(processed_rows, start=1)
        if (converted := human_row_to_uma(row, raw_steps=raw_steps, split=split, seed=idx)) is not None
    ]
    problem_rows = build_problem_set(output_rows, split)

    output_path = out_dir / f"singlevar_equation_human_uma_{split}.csv"
    problem_path = out_dir / f"singlevar_equation_problem_set_{split}.csv"
    write_rows(output_path, output_rows, HUMAN_UMA_COLUMNS)
    write_rows(problem_path, problem_rows, PROBLEM_SET_COLUMNS)

    return {
        "processed_input": str(processed_path),
        "raw_input": str(raw_path),
        "output": str(output_path),
        "problem_set_output": str(problem_path),
        "summary": summarize_rows(len(processed_rows), output_rows),
        "problem_set": {"unique_problems": len(problem_rows)},
    }


def build_single_variable_algebra_uma_outputs(
    dataset: str = DEFAULT_DATASET,
    splits: Sequence[str] = DEFAULT_SPLITS,
    input_root: Path = DEFAULT_INPUT_ROOT,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    input_prefix: str = DEFAULT_INPUT_PREFIX,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {
        split: build_split(
            dataset=dataset,
            split=split,
            input_root=input_root,
            processed_dir=processed_dir,
            out_dir=out_dir,
            input_prefix=input_prefix,
        )
        for split in splits
    }
    audit = {
        "dataset": dataset,
        "definition": (
            "Conservative single-variable equation subset for UMA-style algebra evaluation. "
            "Rows are retained only when the problem is a solvable linear or simple denominator "
            "equation and not an arithmetic-only assignment such as x = 23*30."
        ),
        "input_prefix": input_prefix,
        "splits": split_summaries,
        "columns": HUMAN_UMA_COLUMNS,
        "problem_set_columns": PROBLEM_SET_COLUMNS,
    }
    audit_path = out_dir / "singlevar_equation_human_uma_audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    audit["audit_path"] = str(audit_path)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build UMA-compatible human traces for single-variable Algebra 2006-2007 equations."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_single_variable_algebra_uma_outputs(
        dataset=args.dataset,
        splits=args.splits,
        input_root=args.input_root,
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        input_prefix=args.input_prefix,
    )
    for split, summary in audit["splits"].items():
        split_summary = summary["summary"]
        print(
            f"{split}: retained {split_summary['retained_rows']} / "
            f"{split_summary['input_rows']} rows, "
            f"acc={split_summary['accuracy_rate']:.4f}",
            flush=True,
        )
    print(f"audit_json: {audit['audit_path']}", flush=True)


if __name__ == "__main__":
    main()
