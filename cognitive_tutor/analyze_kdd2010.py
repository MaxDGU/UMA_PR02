#!/usr/bin/env python3
"""Summarize local KDD2010 Cognitive Tutor datasets.

Question. Which observable signals in the released KDD2010 student-step logs are
available for cognitive-model distillation in this repository?

Setup. Let
    D = {(algebra_2005_2006, train/test/master),
         (algebra_2006_2007, train/test/master),
         (bridge_to_algebra_2006_2007, train/test/master)}
be the extracted tab-delimited files under ``cognitive_tutor/data/extracted``.
Each row is a student-step event with student identity, problem context, step
identity, tutor timestamps, success metadata, and optional KC annotations.

Objective. We compute stable dataset-level summaries that help decide whether to
use these logs for:
1. student-conditioned imitation or prediction,
2. distillation of latent cognitive state,
3. transfer into the current UMA-to-language training pipeline.

Method. We stream each file row by row, so the analysis is robust to the large
train files. We summarize row counts, student coverage, per-student trajectory
lengths, label availability, hint/error statistics, and KC coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator


DATASETS = (
    "algebra_2005_2006",
    "algebra_2006_2007",
    "bridge_to_algebra_2006_2007",
)
SPLITS = ("train", "test", "master")
DEFAULT_ROOT = Path("cognitive_tutor/data/extracted")
DEFAULT_OUTPUT = Path("cognitive_tutor/analysis/kdd2010_summary.json")


def safe_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    try:
        if "." in text:
            return int(float(text))
        return int(text)
    except ValueError:
        return None


def quantiles(values: Iterable[int]) -> Dict[str, float]:
    seq = sorted(int(v) for v in values)
    if not seq:
        return {}
    n = len(seq)

    def pick(p: float) -> float:
        if n == 1:
            return float(seq[0])
        idx = p * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return float(seq[lo])
        weight = idx - lo
        return float(seq[lo] * (1.0 - weight) + seq[hi] * weight)

    return {
        "min": float(seq[0]),
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": float(seq[-1]),
        "mean": float(statistics.fmean(seq)),
    }


def split_kcs(value: str) -> Iterator[str]:
    text = value.strip()
    if not text:
        return
    for piece in text.split("~~"):
        cleaned = piece.strip()
        if cleaned:
            yield cleaned


def find_column(columns: list[str], prefix: str) -> str | None:
    wanted = prefix.lower()
    for column in columns:
        if column.lower().startswith(wanted):
            return column
    return None


def summarize_file(path: Path) -> dict:
    with path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No header found in {path}")

        kc_column = find_column(fieldnames, "KC(")
        opp_column = find_column(fieldnames, "Opportunity(")

        rows = 0
        students = set()
        problem_hierarchies = set()
        problem_names = set()
        step_names = set()
        rows_with_any_time = 0
        rows_with_kc = 0
        rows_with_multi_kc = 0
        rows_with_label = 0
        correct_first_attempt_ones = 0
        incorrects_values: list[int] = []
        hints_values: list[int] = []
        duration_values: list[float] = []
        student_step_counts: Counter[str] = Counter()
        kc_counter: Counter[str] = Counter()
        opp_counter: Counter[str] = Counter()
        sample_rows: list[dict] = []

        for row in reader:
            rows += 1
            student = row.get("Anon Student Id", "").strip()
            if student:
                students.add(student)
                student_step_counts[student] += 1

            hierarchy = row.get("Problem Hierarchy", "").strip()
            if hierarchy:
                problem_hierarchies.add(hierarchy)

            problem_name = row.get("Problem Name", "").strip()
            if problem_name:
                problem_names.add(problem_name)

            step_name = row.get("Step Name", "").strip()
            if step_name:
                step_names.add(step_name)

            if any(
                row.get(col, "").strip()
                for col in (
                    "Step Start Time",
                    "First Transaction Time",
                    "Correct Transaction Time",
                    "Step End Time",
                )
            ):
                rows_with_any_time += 1

            cfa = row.get("Correct First Attempt", "").strip()
            if cfa:
                rows_with_label += 1
                if cfa == "1":
                    correct_first_attempt_ones += 1

            incorrects = safe_int(row.get("Incorrects", ""))
            if incorrects is not None:
                incorrects_values.append(incorrects)

            hints = safe_int(row.get("Hints", ""))
            if hints is not None:
                hints_values.append(hints)

            duration = safe_float(row.get("Step Duration (sec)", ""))
            if duration is not None:
                duration_values.append(duration)

            if kc_column:
                kc_value = row.get(kc_column, "")
                kc_items = list(split_kcs(kc_value))
                if kc_items:
                    rows_with_kc += 1
                    if len(kc_items) > 1:
                        rows_with_multi_kc += 1
                    kc_counter.update(kc_items)

            if opp_column:
                opp_value = row.get(opp_column, "")
                opp_items = [piece.strip() for piece in opp_value.split("~~") if piece.strip()]
                opp_counter.update(opp_items)

            if len(sample_rows) < 3:
                sample_rows.append(
                    {
                        "student": student,
                        "problem_hierarchy": hierarchy,
                        "problem_name": problem_name,
                        "step_name": step_name,
                        "correct_first_attempt": cfa,
                        "incorrects": row.get("Incorrects", "").strip(),
                        "hints": row.get("Hints", "").strip(),
                        "kc": row.get(kc_column, "").strip() if kc_column else "",
                    }
                )

    student_length_stats = quantiles(student_step_counts.values())
    duration_stats = quantiles(int(v) for v in duration_values) if duration_values else {}

    result = {
        "path": str(path),
        "columns": fieldnames,
        "row_count": rows,
        "unique_students": len(students),
        "unique_problem_hierarchies": len(problem_hierarchies),
        "unique_problem_names": len(problem_names),
        "unique_step_names": len(step_names),
        "rows_with_any_timestamp": rows_with_any_time,
        "rows_with_any_timestamp_rate": rows_with_any_time / rows if rows else None,
        "rows_with_label": rows_with_label,
        "rows_with_label_rate": rows_with_label / rows if rows else None,
        "correct_first_attempt_rate": (
            correct_first_attempt_ones / rows_with_label if rows_with_label else None
        ),
        "rows_with_kc": rows_with_kc,
        "rows_with_kc_rate": rows_with_kc / rows if rows else None,
        "rows_with_multi_kc": rows_with_multi_kc,
        "rows_with_multi_kc_rate": rows_with_multi_kc / rows if rows else None,
        "unique_kcs": len(kc_counter),
        "top_kcs": kc_counter.most_common(15),
        "top_opportunities": opp_counter.most_common(15),
        "student_step_count_stats": student_length_stats,
        "incorrects_mean": float(statistics.fmean(incorrects_values)) if incorrects_values else None,
        "hints_mean": float(statistics.fmean(hints_values)) if hints_values else None,
        "duration_seconds_mean": float(statistics.fmean(duration_values)) if duration_values else None,
        "duration_seconds_quantiles_int_rounded": duration_stats,
        "sample_rows": sample_rows,
    }
    return result


def build_summary(root: Path) -> dict:
    summary: dict = {
        "root": str(root),
        "datasets": {},
    }
    for dataset in DATASETS:
        dataset_dir = root / dataset
        dataset_summary = {"splits": {}}
        for split in SPLITS:
            path = dataset_dir / f"{dataset}_{split}.txt"
            dataset_summary["splits"][split] = summarize_file(path)
        summary["datasets"][dataset] = dataset_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
