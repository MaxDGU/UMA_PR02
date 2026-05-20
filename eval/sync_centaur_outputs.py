#!/usr/bin/env python3
"""Copy Centaur arithmetic runs into stable eval output CSVs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval import eval_utils as eu  # noqa: E402


@dataclass(frozen=True)
class CentaurRun:
    domain: str
    source: Path
    target_subdir: str
    target_name: str = "centaur_70b.csv"


DEFAULT_RUNS = (
    CentaurRun(
        domain="fraction",
        source=ROOT_DIR / "Centaur" / "runs" / "centaur_vllm_fraction_s100" / "outputs.csv",
        target_subdir="fraction",
    ),
    CentaurRun(
        domain="decimal",
        source=ROOT_DIR / "Centaur" / "runs" / "centaur_vllm_decimal_s100" / "outputs.csv",
        target_subdir="decimal",
    ),
    CentaurRun(
        domain="fraction",
        source=ROOT_DIR / "Centaur" / "runs" / "centaur_vllm_fraction_humanft8_s100" / "outputs.csv",
        target_subdir="fraction_humanft8",
    ),
)


def stable_output_frame(raw: pd.DataFrame, run: CentaurRun) -> pd.DataFrame:
    source = eu.normalize_output_frame(
        raw,
        domain=run.domain,
        source_key=eu.CENTAUR_SOURCE_KEY,
        source_family="centaur",
        source_model="Centaur-70B",
        source_path=run.source,
        problem_col="problem",
        response_col="model_response",
        answer_col="parsed_answer",
        correct_col="is_correct",
        sample_col="sample_idx",
        source_row_col="row_index",
    )
    frame = source.frame
    if run.domain == "fraction":
        return pd.DataFrame(
            {
                "problem": frame["problem"],
                "operation": frame["operation"],
                "denom_type": frame["denom_type"],
                "correct_answer": frame["correct_answer"],
                "model": "Centaur-70B",
                "sample_idx": frame["sample_idx"],
                "parsed_answer": frame["parsed_answer"],
                "is_correct": frame["is_correct"],
                "model_response": frame["model_response"],
            }
        )
    return pd.DataFrame(
        {
            "problem": frame["problem"],
            "operation": frame["operation"],
            "operands": frame["operands"],
            "correct_answer": frame["correct_answer"],
            "model": "Centaur-70B",
            "sample_idx": frame["sample_idx"],
            "parsed_answer": frame["parsed_answer"],
            "is_correct": frame["is_correct"],
            "model_response": frame["model_response"],
        }
    )


def sync_run(run: CentaurRun, outputs_dir: Path, force: bool) -> Path | None:
    if not run.source.exists():
        print(f"Missing {run.source}", flush=True)
        return None
    target = outputs_dir / run.target_subdir / run.target_name
    if target.exists() and not force:
        print(f"Exists {target}; use --force to overwrite", flush=True)
        return target

    raw = eu.read_csv_auto(run.source)
    stable = stable_output_frame(raw, run)
    target.parent.mkdir(parents=True, exist_ok=True)
    stable.to_csv(target, index=False)
    n_problems = stable["problem"].nunique()
    print(f"Wrote {target} ({len(stable)} rows, {n_problems} problems)", flush=True)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=eu.DEFAULT_FRONTIER_OUTPUTS_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing stable Centaur outputs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else ROOT_DIR / args.outputs_dir
    for run in DEFAULT_RUNS:
        sync_run(run, outputs_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
