#!/usr/bin/env python3
"""Train/run lightweight algebra UMA students over a parameter panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cognitive_tutor.algebra_uma import (
    DEFAULT_PANEL_CSV,
    UMA_TRACE_COLUMNS,
    generate_synthetic_equations,
    generate_synthetic_equations_matched,
    load_human_calibration,
    read_panel,
    read_problem_csv,
    simulate_attempt,
    simulate_attempt_v2_calibrated,
    simulate_attempt_v3_ct_flow,
    train_student,
    write_trace_rows,
)


DEFAULT_OUT_CSV = Path("results/UMA_replication/algebra_uma_k100/algebra_uma_traces.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train lightweight algebra UMA students on synthetic equations and "
            "emit UMA-compatible traces for synthetic or human-matched problems."
        )
    )
    parser.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL_CSV)
    parser.add_argument("--problem-csv", type=Path, default=None)
    parser.add_argument("--problem-col", default="prob")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--synthetic-train-size", type=int, default=2000)
    parser.add_argument("--synthetic-eval-size", type=int, default=1000)
    parser.add_argument(
        "--synthetic-match-csv",
        type=Path,
        default=None,
        help="When no --problem-csv is given, generate synthetic eval problems matched to this problem CSV.",
    )
    parser.add_argument(
        "--synthetic-match-col",
        default="prob",
        help="Problem column for --synthetic-match-csv.",
    )
    parser.add_argument(
        "--synthetic-train-match-csv",
        type=Path,
        default=None,
        help="Generate synthetic training curriculum matched to this problem CSV.",
    )
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument(
        "--sim-version",
        choices=["v1", "v2_calibrated", "v3_ct_flow"],
        default="v1",
        help="Simulator behavior to use for emitted evaluation traces.",
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path("cognitive_tutor/data/processed/algebra_2006_2007/singlevar_equation_human_uma_master.csv"),
        help="Human UMA CSV used by --sim-version v2_calibrated.",
    )
    parser.add_argument(
        "--emit-train-traces",
        action="store_true",
        help="Also emit attempts on the synthetic training curriculum.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_panel_rows(panel_rows: list[dict[str, object]], start_index: int, end_index: int | None) -> list[dict[str, object]]:
    if end_index is None:
        end_index = len(panel_rows) - 1
    if start_index < 0 or end_index < start_index:
        raise ValueError(f"Invalid panel range: {start_index}..{end_index}")
    return panel_rows[start_index : end_index + 1]


def load_eval_problems(args: argparse.Namespace):
    if args.problem_csv is not None:
        return read_problem_csv(
            args.problem_csv,
            problem_col=args.problem_col,
            max_problems=args.max_problems,
        ), "human_matched_problem_set"
    if args.synthetic_match_csv is not None:
        references = read_problem_csv(args.synthetic_match_csv, problem_col=args.synthetic_match_col)
        problems = generate_synthetic_equations_matched(
            references,
            args.synthetic_eval_size,
            seed=args.seed + 10_000,
        )
        if args.max_problems is not None:
            problems = problems[: args.max_problems]
        return problems, "synthetic_eval_matched"
    problems = generate_synthetic_equations(args.synthetic_eval_size, seed=args.seed + 10_000)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]
    return problems, "synthetic_eval"


def run_simulation(args: argparse.Namespace) -> list[dict[str, object]]:
    panel_rows = select_panel_rows(read_panel(args.panel_csv), args.start_index, args.end_index)
    eval_problems, eval_source = load_eval_problems(args)
    calibration = (
        load_human_calibration(args.calibration_csv)
        if args.sim_version in {"v2_calibrated", "v3_ct_flow"}
        else None
    )
    rows: list[dict[str, object]] = []
    train_match_references = (
        read_problem_csv(args.synthetic_train_match_csv, problem_col=args.synthetic_match_col)
        if args.synthetic_train_match_csv is not None
        else None
    )

    for panel_offset, params in enumerate(panel_rows, start=args.start_index):
        if train_match_references is not None:
            train_curriculum = generate_synthetic_equations_matched(
                train_match_references,
                args.synthetic_train_size,
                seed=args.seed + panel_offset * 17,
            )
        else:
            train_curriculum = generate_synthetic_equations(
                args.synthetic_train_size,
                seed=args.seed + panel_offset * 17,
            )
        state = train_student(params, train_curriculum, seed=args.seed)

        if args.emit_train_traces:
            for problem_idx, meta in enumerate(train_curriculum, start=1):
                rows.append(
                    simulate_attempt(
                        state,
                        meta,
                        seed=args.seed + problem_idx,
                        source="synthetic_train",
                    ).to_row()
                )

        for problem_idx, meta in enumerate(eval_problems, start=1):
            if args.sim_version == "v2_calibrated":
                if calibration is None:
                    raise RuntimeError("v2_calibrated requires calibration")
                attempt = simulate_attempt_v2_calibrated(
                    state,
                    meta,
                    calibration=calibration,
                    seed=args.seed + 100_000 + problem_idx,
                    source=f"{eval_source}_v2_calibrated",
                )
            elif args.sim_version == "v3_ct_flow":
                if calibration is None:
                    raise RuntimeError("v3_ct_flow requires calibration")
                attempt = simulate_attempt_v3_ct_flow(
                    state,
                    meta,
                    calibration=calibration,
                    seed=args.seed + 100_000 + problem_idx,
                    source=f"{eval_source}_v3_ct_flow",
                )
            else:
                attempt = simulate_attempt(
                    state,
                    meta,
                    seed=args.seed + 100_000 + problem_idx,
                    source=eval_source,
                )
            rows.append(attempt.to_row())
    return rows


def main() -> None:
    args = parse_args()
    rows = run_simulation(args)
    write_trace_rows(args.out_csv, rows, overwrite=args.overwrite)
    print(f"wrote_rows={len(rows)}", flush=True)
    print(f"out_csv={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
