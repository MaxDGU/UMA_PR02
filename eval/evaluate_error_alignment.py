#!/usr/bin/env python3
"""Compute answer-error alignment metrics for saved arithmetic outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from eval import eval_utils as eu
except ImportError:  # pragma: no cover - supports direct execution from eval/
    import eval_utils as eu


def accuracy_by_problem(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["problem"], as_index=False)
        .agg(
            model_acc=("is_correct", "mean"),
            n_samples=("is_correct", "size"),
        )
        .sort_values("problem")
    )


def accuracy_by_cell(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    group_cols = ["operation", "denom_type"] if domain == "fraction" else ["operation", "operands"]
    return (
        frame.groupby(group_cols, as_index=False)
        .agg(
            model_acc=("is_correct", "mean"),
            n_samples=("is_correct", "size"),
            n_problems=("problem", "nunique"),
        )
        .sort_values(group_cols)
    )


def concat_without_all_na_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    return pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame()


def summarize_source(
    source: eu.SourceFrame,
    human_problem: pd.DataFrame,
    human_cell: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = source.frame.copy()
    summary: dict[str, Any] = {
        "source_key": source.source_key,
        "source_family": source.source_family,
        "source_model": source.source_model,
        "domain": source.domain,
        "source_path": str(source.path),
        "row_count": int(len(frame)),
        "problem_count": int(frame["problem"].nunique()),
        "accuracy": float(frame["is_correct"].mean()) if len(frame) else None,
        "mae_h_pp": None,
        "mag_h_pp": None,
        "MAE-H": None,
        "MAG-H": None,
        "human_problem_overlap": 0,
        "human_cell_overlap": 0,
    }

    per_problem = accuracy_by_problem(frame)
    per_problem.insert(0, "source_model", source.source_model)
    per_problem.insert(0, "source_family", source.source_family)
    per_problem.insert(0, "source_key", source.source_key)
    per_problem.insert(0, "domain", source.domain)

    per_cell = accuracy_by_cell(frame, source.domain)
    per_cell.insert(0, "source_model", source.source_model)
    per_cell.insert(0, "source_family", source.source_family)
    per_cell.insert(0, "source_key", source.source_key)
    per_cell.insert(0, "domain", source.domain)

    if source.domain == "fraction":
        merged_problem = per_problem.merge(human_problem, on="problem", how="left")
        merged_problem["abs_err_pp"] = (merged_problem["model_acc"] - merged_problem["human_acc"]).abs() * 100.0
        per_problem = merged_problem

        merged_cell = per_cell.merge(human_cell, on=["operation", "denom_type"], how="left")
        merged_cell["abs_gap_h_pp"] = (merged_cell["model_acc"] - merged_cell["human_acc"]).abs() * 100.0
        per_cell = merged_cell

        valid_problem = per_problem["human_acc"].notna()
        valid_cell = per_cell["human_acc"].notna()
        mae_h = float(per_problem.loc[valid_problem, "abs_err_pp"].mean()) if valid_problem.any() else None
        mag_h = float(per_cell.loc[valid_cell, "abs_gap_h_pp"].mean()) if valid_cell.any() else None
        summary.update(
            {
                "mae_h_pp": mae_h,
                "mag_h_pp": mag_h,
                "MAE-H": mae_h,
                "MAG-H": mag_h,
                "human_problem_overlap": int(valid_problem.sum()),
                "human_cell_overlap": int(valid_cell.sum()),
            }
        )
    else:
        per_problem["human_acc"] = pd.NA
        per_problem["human_rows"] = pd.NA
        per_problem["abs_err_pp"] = pd.NA
        per_cell["human_acc"] = pd.NA
        per_cell["human_rows"] = pd.NA
        per_cell["abs_gap_h_pp"] = pd.NA

    return summary, per_problem, per_cell


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["fraction", "decimal", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=eu.EVAL_DIR / "results" / "error_alignment")
    parser.add_argument("--frontier-outputs-dir", type=Path, default=eu.DEFAULT_FRONTIER_OUTPUTS_DIR)
    parser.add_argument("--centaur-fraction-csv", type=Path, default=eu.DEFAULT_CENTAUR_FRACTION_CSV)
    parser.add_argument("--centaur-decimal-csv", type=Path, default=eu.DEFAULT_CENTAUR_DECIMAL_CSV)
    parser.add_argument("--uma-csv", type=Path, default=eu.DEFAULT_UMA_CSV)
    parser.add_argument("--human-fraction-csv", type=Path, default=eu.DEFAULT_FRACTION_HUMAN_CSV)
    parser.add_argument("--include-frontier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-centaur", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-uma", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    domains = eu.domains_from_arg(args.domain)
    out_dir = args.out_dir if args.out_dir.is_absolute() else eu.ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = eu.load_all_sources(
        domains=domains,
        frontier_outputs_dir=eu.resolve_path(args.frontier_outputs_dir),
        centaur_fraction_csv=eu.resolve_path(args.centaur_fraction_csv),
        centaur_decimal_csv=eu.resolve_path(args.centaur_decimal_csv),
        uma_csv=eu.resolve_path(args.uma_csv),
        include_frontier=args.include_frontier,
        include_centaur=args.include_centaur,
        include_uma=args.include_uma,
    )
    if not sources:
        raise ValueError("No saved output sources were found for the requested configuration.")

    human_problem, human_cell = eu.load_sp2013_human_accuracy(eu.resolve_path(args.human_fraction_csv))
    summary_rows: list[dict[str, Any]] = []
    per_problem_frames: list[pd.DataFrame] = []
    per_cell_frames: list[pd.DataFrame] = []
    for source in sources:
        summary, per_problem, per_cell = summarize_source(source, human_problem, human_cell)
        summary_rows.append(summary)
        per_problem_frames.append(per_problem)
        per_cell_frames.append(per_cell)

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain", "source_family", "source_key"])
    per_problem_df = concat_without_all_na_frames(per_problem_frames).sort_values(
        ["domain", "source_family", "source_key", "problem"]
    )
    per_cell_df = concat_without_all_na_frames(per_cell_frames).sort_values(
        ["domain", "source_family", "source_key", "operation"]
    )

    summary_df.to_csv(out_dir / "per_model_summary.csv", index=False)
    per_problem_df.to_csv(out_dir / "per_problem.csv", index=False)
    per_cell_df.to_csv(out_dir / "per_cell.csv", index=False)
    eu.write_json(
        out_dir / "manifest.json",
        {
            "created_at": eu.timestamp_utc(),
            "domains": list(domains),
            "human_fraction_csv": str(eu.resolve_path(args.human_fraction_csv)),
            "source_count": int(len(sources)),
            "sources": [
                {
                    "source_key": source.source_key,
                    "source_family": source.source_family,
                    "source_model": source.source_model,
                    "domain": source.domain,
                    "path": str(source.path),
                    "rows": int(len(source.frame)),
                }
                for source in sources
            ],
            "metric_definitions": {
                "MAE-H": "Mean absolute per-problem accuracy gap vs SP2013 human accuracy, in percentage points.",
                "MAG-H": "Mean absolute operation x denominator-type cell accuracy gap vs SP2013 human accuracy, in percentage points.",
            },
        },
    )
    print(f"Wrote error-alignment metrics for {len(sources)} sources to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
