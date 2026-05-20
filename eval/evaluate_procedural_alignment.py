#!/usr/bin/env python3
"""Run judge-based procedural-alignment extraction for saved reasoning traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from procedural_alignment import fraction_strategy_extraction as fse
from procedural_alignment.strategy_mapping import map_human_fraction_code_to_family

try:
    from eval import eval_utils as eu
except ImportError:  # pragma: no cover - supports direct execution from eval/
    import eval_utils as eu


FRACTION_FAMILIES = ("AS", "M", "D", "OTHER")
DECIMAL_CODES = ("AS", "M", "BOTH", "O")


def stage_source_inputs(source: eu.SourceFrame, out_dir: Path) -> Path:
    frame = source.frame.copy()
    frame = frame[frame["model_response"].map(eu.clean_text).ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{source.source_key} {source.domain} has no non-empty reasoning traces.")
    staged = pd.DataFrame(
        {
            "sample_id": [
                f"{source.source_key}:{source.domain}:{idx:06d}"
                for idx in range(len(frame))
            ],
            "source_key": source.source_key,
            "source_family": source.source_family,
            "source_model": source.source_model,
            "source_path": str(source.path),
            "prob": frame["problem"],
            "operation": frame["operation"],
            "op1": frame["op1"],
            "op2": frame["op2"],
            "parsed_answer": frame["parsed_answer"],
            "is_correct": frame["is_correct"].astype(int),
            "model_response": frame["model_response"],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source.domain}_{source.source_key}.csv"
    staged.to_csv(path, index=False)
    return path


def build_fse_args(
    *,
    domain: str,
    staged_csv: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    fse_args = [
        "--domain",
        domain,
        "--evidence",
        "trace",
        "--input_csv",
        str(staged_csv),
        "--input_encoding",
        "utf-8",
        "--problem_col",
        "prob",
        "--trace_col",
        "model_response",
        "--answer_col",
        "parsed_answer",
        "--id_col",
        "sample_id",
        "--gold_code_col",
        "",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--api_version",
        args.api_version,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_output_tokens",
        str(args.max_output_tokens),
        "--thinking_budget",
        str(args.thinking_budget),
        "--max_retries",
        str(args.max_retries),
        "--retry_base_seconds",
        str(args.retry_base_seconds),
        "--sleep_seconds",
        str(args.sleep_seconds),
        "--max_rows",
        str(args.max_rows),
        "--out_dir",
        str(run_dir),
        "--few_shot_per_code",
        str(args.few_shot_per_code),
        "--few_shot_preset",
        args.few_shot_preset,
        "--max_trace_chars",
        str(args.max_trace_chars),
    ]
    if domain == "decimal":
        fse_args.extend(
            [
                "--operation_col",
                "operation",
                "--op1_col",
                "op1",
                "--op2_col",
                "op2",
            ]
        )
        if args.decimal_human_csv:
            fse_args.extend(["--reference_human_csv", str(eu.resolve_path(args.decimal_human_csv))])
    if args.thinking_level:
        fse_args.extend(["--thinking_level", args.thinking_level])
    if args.force:
        fse_args.append("--force")
    return fse_args


def count_distribution(values: pd.Series, labels: Sequence[str]) -> pd.DataFrame:
    counts = values.value_counts(dropna=False).to_dict()
    total = sum(int(counts.get(label, 0)) for label in labels)
    return pd.DataFrame(
        {
            "label": list(labels),
            "pred_count": [int(counts.get(label, 0)) for label in labels],
            "pred_prop": [float(counts.get(label, 0) / total) if total else 0.0 for label in labels],
        }
    )


def summarize_fraction_run(source: eu.SourceFrame, run_dir: Path, reference: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        return (
            {
                "source_key": source.source_key,
                "source_family": source.source_family,
                "source_model": source.source_model,
                "domain": source.domain,
                "status": "requests_written",
            },
            pd.DataFrame(),
        )
    pred = pd.read_csv(predictions_path)
    pred["strategy_family"] = pred["pred_code"].map(map_human_fraction_code_to_family)
    dist = count_distribution(pred["strategy_family"], FRACTION_FAMILIES).rename(columns={"label": "strategy_family"})
    dist = reference.merge(dist, on="strategy_family", how="left")
    dist[["pred_count", "pred_prop"]] = dist[["pred_count", "pred_prop"]].fillna(0)
    dist["abs_prop_diff"] = (dist["reference_prop"] - dist["pred_prop"]).abs()
    tvd = 0.5 * float(dist["abs_prop_diff"].sum())
    dist.insert(0, "source_model", source.source_model)
    dist.insert(0, "source_family", source.source_family)
    dist.insert(0, "source_key", source.source_key)
    dist.insert(0, "domain", source.domain)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").exists() else {}
    return (
        {
            "source_key": source.source_key,
            "source_family": source.source_family,
            "source_model": source.source_model,
            "domain": source.domain,
            "status": "completed",
            "row_count": int(len(pred)),
            "parse_success_rate": metrics.get("parse_success_rate"),
            "strategy_family_tvd_to_sp2013_human": tvd,
        },
        dist,
    )


def summarize_decimal_run(source: eu.SourceFrame, run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        return (
            {
                "source_key": source.source_key,
                "source_family": source.source_family,
                "source_model": source.source_model,
                "domain": source.domain,
                "status": "requests_written",
            },
            pd.DataFrame(),
        )
    pred = pd.read_csv(predictions_path)
    dist = count_distribution(pred["pred_code"], DECIMAL_CODES).rename(columns={"label": "strategy_code"})
    dist.insert(0, "source_model", source.source_model)
    dist.insert(0, "source_family", source.source_family)
    dist.insert(0, "source_key", source.source_key)
    dist.insert(0, "domain", source.domain)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").exists() else {}
    return (
        {
            "source_key": source.source_key,
            "source_family": source.source_family,
            "source_model": source.source_model,
            "domain": source.domain,
            "status": "completed",
            "row_count": int(len(pred)),
            "parse_success_rate": metrics.get("parse_success_rate"),
            "distribution_tvd_to_reference": metrics.get("distribution_tvd_to_reference"),
            "primary_accuracy": metrics.get("primary_accuracy"),
        },
        dist,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["fraction", "decimal", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=eu.EVAL_DIR / "results" / "procedural_alignment")
    parser.add_argument("--frontier-outputs-dir", type=Path, default=eu.DEFAULT_FRONTIER_OUTPUTS_DIR)
    parser.add_argument("--centaur-fraction-csv", type=Path, default=eu.DEFAULT_CENTAUR_FRACTION_CSV)
    parser.add_argument("--centaur-decimal-csv", type=Path, default=eu.DEFAULT_CENTAUR_DECIMAL_CSV)
    parser.add_argument("--human-fraction-csv", type=Path, default=eu.DEFAULT_FRACTION_HUMAN_CSV)
    parser.add_argument("--decimal-human-csv", type=Path, default=None)
    parser.add_argument("--include-frontier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-centaur", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=["requests", "gemini", "gemini_batch"], default="gemini_batch")
    parser.add_argument("--model", default=fse.DEFAULT_MODEL)
    parser.add_argument("--api-version", default="v1beta")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--thinking-level", default="")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--few-shot-per-code", type=int, default=0)
    parser.add_argument("--few-shot-preset", choices=["human_shortest", "canonical"], default="human_shortest")
    parser.add_argument("--max-trace-chars", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    domains = eu.domains_from_arg(args.domain)
    out_dir = args.out_dir if args.out_dir.is_absolute() else eu.ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "inputs"
    runs_dir = out_dir / "runs"

    sources = eu.load_all_sources(
        domains=domains,
        frontier_outputs_dir=eu.resolve_path(args.frontier_outputs_dir),
        centaur_fraction_csv=eu.resolve_path(args.centaur_fraction_csv),
        centaur_decimal_csv=eu.resolve_path(args.centaur_decimal_csv),
        include_frontier=args.include_frontier,
        include_centaur=args.include_centaur,
        include_uma=False,
    )
    sources = [source for source in sources if source.frame["model_response"].map(eu.clean_text).ne("").any()]
    if not sources:
        raise ValueError("No reasoning-trace sources were found for procedural extraction.")

    sp2013_reference = eu.load_sp2013_strategy_family_distribution(eu.resolve_path(args.human_fraction_csv))
    summary_rows: list[dict[str, Any]] = []
    distribution_frames: list[pd.DataFrame] = []
    for source in sources:
        staged_csv = stage_source_inputs(source, input_dir)
        run_dir = runs_dir / f"{source.domain}_{source.source_key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        fse_args = build_fse_args(domain=source.domain, staged_csv=staged_csv, run_dir=run_dir, args=args)
        print(f"Running procedural extraction for {source.domain} {source.source_key} -> {run_dir}", flush=True)
        status = fse.main(fse_args)
        if status != 0:
            raise RuntimeError(f"fraction_strategy_extraction failed for {source.domain} {source.source_key}: {status}")
        if source.domain == "fraction":
            summary, dist = summarize_fraction_run(source, run_dir, sp2013_reference)
        else:
            summary, dist = summarize_decimal_run(source, run_dir)
        summary_rows.append(summary)
        if not dist.empty:
            distribution_frames.append(dist)

    pd.DataFrame(summary_rows).sort_values(["domain", "source_family", "source_key"]).to_csv(
        out_dir / "summary.csv",
        index=False,
    )
    if distribution_frames:
        pd.concat(distribution_frames, ignore_index=True).to_csv(out_dir / "strategy_family_distribution.csv", index=False)
    eu.write_json(
        out_dir / "manifest.json",
        {
            "created_at": eu.timestamp_utc(),
            "domains": list(domains),
            "backend": args.backend,
            "model": args.model,
            "source_count": int(len(sources)),
            "uma_excluded": "UMA answer trials do not include reasoning traces for strategy extraction.",
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
        },
    )
    print(f"Wrote procedural-alignment outputs for {len(sources)} sources to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
