#!/usr/bin/env python3
"""Run human-likeness preference judging for distilled-vs-frontier sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval import human_likeness_preference as hlp

try:
    from eval import eval_utils as eu
except ImportError:  # pragma: no cover - supports direct execution from eval/
    import eval_utils as eu


DEFAULT_BASELINE_CSV = (
    eu.ROOT_DIR
    / "results"
    / "procedural_alignment"
    / "human_ft_fraction_eval_20260428_200211"
    / "panel25_distilled"
    / "panel25_distilled_human_ft_rollouts.csv.gz"
)
DEFAULT_GEMINI_CSV = (
    eu.ROOT_DIR
    / "results"
    / "procedural_alignment"
    / "frontier_human_ft_max_prompt"
    / "gemini_2p5_flash_2000_repaired"
    / "parsed_samples.csv"
)
DEFAULT_GPT_CSV = (
    eu.ROOT_DIR
    / "results"
    / "procedural_alignment"
    / "frontier_human_ft_max_prompt"
    / "gpt_4p1_mini_2000_repaired"
    / "parsed_samples.csv"
)


def source_spec(name: str, path: Path) -> str:
    return (
        f"name={name},path={path},trace_col=model_response,"
        "answer_col=parsed_answer,id_col=request_id,problem_col=prob"
    )


def default_comparisons() -> list[str]:
    return [
        source_spec("gemini_2p5_flash", DEFAULT_GEMINI_CSV),
        source_spec("gpt_4p1_mini", DEFAULT_GPT_CSV),
    ]


def missing_sources(baseline_csv: Path, comparison_specs: Sequence[str]) -> list[str]:
    missing = []
    if not baseline_csv.exists():
        missing.append(f"baseline_csv: {baseline_csv}")
    for text in comparison_specs:
        spec = hlp.parse_source_spec(text, "model_response", "parsed_answer", "request_id")
        path = spec.path if spec.path.is_absolute() else eu.ROOT_DIR / spec.path
        if not path.exists():
            missing.append(f"comparison {spec.name}: {path}")
    return missing


def preflight_error_message(missing: Sequence[str]) -> str:
    lines = [
        "Missing humanlike-preference source files:",
        *[f"- {item}" for item in missing],
        "",
        "Provide the distilled baseline with --baseline-csv. Expected baseline columns:",
        "  prob, generation_text, pred_answer",
        "",
        "Provide each frontier comparison with --comparison. Expected comparison columns:",
        "  prob, model_response, parsed_answer, request_id",
        "",
        "The standalone preference judge can run once those CSVs are present.",
    ]
    return "\n".join(lines)


def build_hlp_args(args: argparse.Namespace, baseline_csv: Path, comparisons: Sequence[str], out_dir: Path) -> list[str]:
    hlp_args = [
        "--backend",
        args.backend,
        "--baseline_csv",
        str(baseline_csv),
        "--baseline_name",
        args.baseline_name,
        "--baseline_trace_col",
        args.baseline_trace_col,
        "--baseline_answer_col",
        args.baseline_answer_col,
        "--baseline_id_col",
        args.baseline_id_col,
        "--samples_per_problem",
        str(args.samples_per_problem),
        "--max_pairs_per_comparison",
        str(args.max_pairs_per_comparison),
        "--seed",
        str(args.seed),
        "--judge_model",
        args.judge_model,
        "--api_style",
        args.api_style,
        "--sandbox_endpoint",
        args.sandbox_endpoint,
        "--portkey_base_url",
        args.portkey_base_url,
        "--api_version",
        args.api_version,
        "--api_key_env",
        args.api_key_env,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_tokens",
        str(args.max_tokens),
        "--max_retries",
        str(args.max_retries),
        "--retry_base_seconds",
        str(args.retry_base_seconds),
        "--request_timeout_seconds",
        str(args.request_timeout_seconds),
        "--sleep_seconds",
        str(args.sleep_seconds),
        "--progress_every",
        str(args.progress_every),
        "--out_dir",
        str(out_dir),
    ]
    for comparison in comparisons:
        hlp_args.extend(["--comparison", comparison])
    if args.force:
        hlp_args.append("--force")
    return hlp_args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=eu.EVAL_DIR / "results" / "humanlike_preference" / "distilled_vs_frontier")
    parser.add_argument("--backend", choices=["requests", "sandbox"], default="sandbox")
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--baseline-name", default="distilled_panel25")
    parser.add_argument("--baseline-trace-col", default="generation_text")
    parser.add_argument("--baseline-answer-col", default="pred_answer")
    parser.add_argument("--baseline-id-col", default="")
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--samples-per-problem", type=int, default=25)
    parser.add_argument("--max-pairs-per-comparison", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--api-style", choices=["azure", "portkey"], default="portkey")
    parser.add_argument("--sandbox-endpoint", default=hlp.DEFAULT_SANDBOX_ENDPOINT)
    parser.add_argument("--portkey-base-url", default=hlp.DEFAULT_PORTKEY_BASE_URL)
    parser.add_argument("--api-version", default=hlp.DEFAULT_SANDBOX_API_VERSION)
    parser.add_argument("--api-key-env", default="AI_SANDBOX_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else eu.ROOT_DIR / args.out_dir
    baseline_csv = eu.resolve_path(args.baseline_csv)
    comparisons = args.comparison or default_comparisons()

    if not args.skip_preflight:
        missing = missing_sources(baseline_csv, comparisons)
        if missing:
            raise SystemExit(preflight_error_message(missing))

    hlp_args = build_hlp_args(args, baseline_csv, comparisons, out_dir)
    status = hlp.main(hlp_args)
    if status != 0:
        raise RuntimeError(f"human_likeness_preference failed with status {status}")
    print(f"Wrote humanlike-preference outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
