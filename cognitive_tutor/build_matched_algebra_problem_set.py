#!/usr/bin/env python3
"""Build a synthetic algebra problem set matched to a reference distribution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cognitive_tutor.algebra_uma import (
    EquationMetadata,
    format_fraction,
    generate_synthetic_equations_matched,
    read_problem_csv,
)


DEFAULT_REFERENCE_CSV = Path(
    "cognitive_tutor/data/processed/algebra_2006_2007/singlevar_equation_problem_set_master.csv"
)
DEFAULT_OUT_CSV = Path(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "singlevar_equation_synthetic_matched_master_10000.csv"
)

OUTPUT_COLUMNS = [
    "prob",
    "equation_text",
    "equation_type",
    "variable",
    "solution",
    "source_split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic equations matched to a reference split.")
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--reference-col", default="prob")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def meta_to_row(meta: EquationMetadata, source_split: str) -> dict[str, object]:
    return {
        "prob": meta.equation_text,
        "equation_text": meta.equation_text,
        "equation_type": meta.equation_type,
        "variable": meta.variable,
        "solution": format_fraction(meta.solution),
        "source_split": source_split,
    }


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def proportions(values: Sequence[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def main() -> None:
    args = parse_args()
    references = read_problem_csv(args.reference_csv, problem_col=args.reference_col)
    metas = generate_synthetic_equations_matched(references, count=args.count, seed=args.seed)
    rows = [meta_to_row(meta, source_split=f"synthetic_matched:{args.reference_csv.name}") for meta in metas]
    write_rows(args.out_csv, rows)

    audit = {
        "reference_csv": str(args.reference_csv),
        "out_csv": str(args.out_csv),
        "count": len(rows),
        "seed": args.seed,
        "reference_equation_type_proportions": proportions([meta.equation_type for meta in references]),
        "synthetic_equation_type_proportions": proportions([meta.equation_type for meta in metas]),
        "reference_variable_proportions": proportions([meta.variable for meta in references]),
        "synthetic_variable_proportions": proportions([meta.variable for meta in metas]),
    }
    audit_path = args.out_csv.with_suffix(".audit.json")
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote_rows={len(rows)}", flush=True)
    print(f"out_csv={args.out_csv}", flush=True)
    print(f"audit_json={audit_path}", flush=True)


if __name__ == "__main__":
    main()
