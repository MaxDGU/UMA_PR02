#!/usr/bin/env python3
"""Compare algebra UMA traces against UMA-formatted human traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cognitive_tutor.build_trace_firstshot_dataset import safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate algebra UMA distribution match to human traces.")
    parser.add_argument(
        "--human-csv",
        type=Path,
        default=Path("cognitive_tutor/data/processed/algebra_2006_2007/singlevar_equation_human_uma_master.csv"),
    )
    parser.add_argument("--uma-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def distribution(rows: Iterable[dict[str, str]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "") or "") for row in rows)


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def total_variation(left: Counter[str], right: Counter[str]) -> float | None:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return None
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left[key] / left_total - right[key] / right_total) for key in keys)


def stuck_key(row: dict[str, str]) -> str:
    if safe_int(row.get("correct", row.get("is_correct", "0"))):
        return "CORRECT"
    return str(row.get("first_wrong_step", "") or row.get("answer", "") or "UNKNOWN")


def add_stuck_keys(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["_stuck_key"] = stuck_key(row)
        out.append(copied)
    return out


def accuracy(rows: list[dict[str, str]]) -> float | None:
    values = [safe_int(row.get("correct", row.get("is_correct", ""))) for row in rows]
    return statistics.fmean(values) if values else None


def answer_exact_rate(rows: list[dict[str, str]]) -> float | None:
    values = []
    for row in rows:
        solution = str(row.get("solution", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if solution:
            values.append(int(answer == solution))
    return statistics.fmean(values) if values else None


def mean_step_count(rows: list[dict[str, str]]) -> float | None:
    values = [safe_int(row.get("trace_step_count", "")) for row in rows]
    return statistics.fmean(values) if values else None


def by_equation_type_tv(human_rows: list[dict[str, str]], uma_rows: list[dict[str, str]]) -> dict[str, float | None]:
    human_by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    uma_by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in human_rows:
        human_by_type[str(row.get("equation_type", ""))].append(row)
    for row in uma_rows:
        uma_by_type[str(row.get("equation_type", ""))].append(row)

    metrics: dict[str, float | None] = {}
    for equation_type in sorted(set(human_by_type) | set(uma_by_type)):
        metrics[equation_type] = total_variation(
            distribution(human_by_type[equation_type], "_stuck_key"),
            distribution(uma_by_type[equation_type], "_stuck_key"),
        )
    return metrics


def evaluate(human_csv: Path, uma_csv: Path) -> dict[str, object]:
    human_rows = add_stuck_keys(read_rows(human_csv))
    uma_rows = add_stuck_keys(read_rows(uma_csv))
    human_stuck = distribution(human_rows, "_stuck_key")
    uma_stuck = distribution(uma_rows, "_stuck_key")

    return {
        "human_csv": str(human_csv),
        "uma_csv": str(uma_csv),
        "row_counts": {"human": len(human_rows), "uma": len(uma_rows)},
        "accuracy": {"human": accuracy(human_rows), "uma": accuracy(uma_rows)},
        "answer_exact_rate": {"human": answer_exact_rate(human_rows), "uma": answer_exact_rate(uma_rows)},
        "mean_step_count": {"human": mean_step_count(human_rows), "uma": mean_step_count(uma_rows)},
        "stuck_distribution_tv": total_variation(human_stuck, uma_stuck),
        "stuck_distribution_tv_by_equation_type": by_equation_type_tv(human_rows, uma_rows),
        "human_stuck_distribution": normalize_counter(human_stuck),
        "uma_stuck_distribution": normalize_counter(uma_stuck),
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.human_csv, args.uma_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"out_json={args.out_json}", flush=True)
    print(f"stuck_distribution_tv={metrics['stuck_distribution_tv']}", flush=True)


if __name__ == "__main__":
    main()
