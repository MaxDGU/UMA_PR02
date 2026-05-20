#!/usr/bin/env python3
"""Run eval inference for the shared frontier arithmetic baselines."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
OUTPUTS_DIR = EVAL_DIR / "outputs"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval import frontier_arithmetic_baseline as fab  # noqa: E402


@dataclass(frozen=True)
class ModelPreset:
    backend: str
    model: str
    label: str
    temperature: str
    top_p: str
    reasoning_effort: str = ""
    gemini_thinking_level: str = "minimal"


MODEL_PRESETS: dict[str, ModelPreset] = {
    "gpt_4_1_mini": ModelPreset(
        backend="sandbox",
        model="gpt-4.1-mini",
        label="GPT-4.1 mini",
        temperature="1",
        top_p="0.95",
    ),
    "gpt_5_5_low": ModelPreset(
        backend="sandbox",
        model="gpt-5.5",
        label="GPT-5.5 low",
        temperature="none",
        top_p="none",
        reasoning_effort="low",
    ),
    "gemini_2_5_flash": ModelPreset(
        backend="gemini",
        model="gemini-2.5-flash",
        label="Gemini 2.5 Flash",
        temperature="1",
        top_p="0.95",
    ),
    "gemini_3_flash": ModelPreset(
        backend="gemini",
        model="gemini-3-flash-preview",
        label="Gemini 3 Flash",
        temperature="1",
        top_p="0.95",
    ),
    "claude_sonnet_4_6": ModelPreset(
        backend="sandbox",
        model="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        temperature="1",
        top_p="0.95",
    ),
}


def resolve_path(path_text: str | None, default: Path) -> Path:
    path = Path(path_text) if path_text else default
    return path if path.is_absolute() else ROOT_DIR / path


def selected_domains(value: str) -> list[str]:
    if value == "both":
        return ["fraction", "decimal"]
    return [value]


def selected_model_slugs(value: str) -> list[str]:
    if value == "all":
        return list(MODEL_PRESETS)
    slugs = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [slug for slug in slugs if slug not in MODEL_PRESETS]
    if unknown:
        known = ", ".join(MODEL_PRESETS)
        raise SystemExit(f"Unknown model slug(s): {', '.join(unknown)}. Known slugs: {known}")
    return slugs


def row_text(row: pd.Series, column: str) -> str:
    if column not in row:
        return ""
    return fab.clean_text(row[column])


def build_eval_loader(fraction_problem_column: str, decimal_problem_column: str):
    def load_problems(domain: str) -> list[fab.ProblemRecord]:
        if domain == "fraction":
            path = DATA_DIR / "fraction_problems.csv"
            problem_column = fraction_problem_column
            metadata_builder = fab.fraction_metadata
            operand_column = "denom_type"
        else:
            path = DATA_DIR / "decimal_problems.csv"
            problem_column = decimal_problem_column
            metadata_builder = fab.decimal_metadata
            operand_column = "operands"

        if not path.exists():
            raise FileNotFoundError(f"Missing eval problem file: {path}")

        df = pd.read_csv(path)
        if "prob" not in df.columns:
            raise ValueError(f"{path} must include a canonical 'prob' column.")
        if problem_column not in df.columns:
            raise ValueError(f"{path} does not have requested problem column {problem_column!r}.")

        problems: list[fab.ProblemRecord] = []
        for _, row in df.iterrows():
            canonical = row_text(row, "prob")
            prompted = row_text(row, problem_column) or canonical
            meta = metadata_builder(canonical)
            problems.append(
                fab.ProblemRecord(
                    problem=prompted,
                    operation=row_text(row, "operation") or meta.operation,
                    operands=row_text(row, operand_column) or meta.operands,
                    correct_answer=row_text(row, "correct_answer") or meta.correct_answer,
                )
            )
        return problems

    return load_problems


def compat_filename(domain: str) -> str:
    if domain == "fraction":
        return "llm_outputs_sp2013_fractions.csv"
    return "llm_outputs_bss2021_decimals.csv"


def build_frontier_args(domain: str, preset: ModelPreset, args: argparse.Namespace, out_dir: Path) -> list[str]:
    samples = args.fraction_samples_per_problem if domain == "fraction" else args.decimal_samples_per_problem
    frontier_args = [
        "--domain",
        domain,
        "--backend",
        preset.backend,
        "--model",
        preset.model,
        "--model_label",
        preset.label,
        "--samples_per_problem",
        str(samples),
        "--max_rows",
        str(args.max_rows),
        "--out_dir",
        str(out_dir),
        "--temperature",
        preset.temperature,
        "--top_p",
        preset.top_p,
        "--max_tokens",
        str(args.max_tokens),
        "--token_limit_param",
        args.token_limit_param,
        "--gemini_thinking_level",
        preset.gemini_thinking_level,
        "--api_key_env",
        args.api_key_env,
        "--gemini_api_key_env",
        args.gemini_api_key_env,
        "--gemini_api_version",
        args.gemini_api_version,
        "--portkey_base_url",
        args.portkey_base_url,
        "--request_timeout_seconds",
        str(args.request_timeout_seconds),
        "--max_retries",
        str(args.max_retries),
        "--retry_base_seconds",
        str(args.retry_base_seconds),
        "--sleep_seconds",
        str(args.sleep_seconds),
        "--progress_every",
        str(args.progress_every),
    ]
    if preset.reasoning_effort:
        frontier_args.extend(["--reasoning_effort", preset.reasoning_effort])
    if args.force:
        frontier_args.append("--force")
    return frontier_args


def copy_stable_output(domain: str, model_slug: str, out_dir: Path) -> Path:
    source = out_dir / compat_filename(domain)
    if not source.exists():
        raise FileNotFoundError(f"Expected inference output was not written: {source}")
    target_dir = OUTPUTS_DIR / domain
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{model_slug}.csv"
    shutil.copy2(source, target)
    return target


def run_one(domain: str, model_slug: str, args: argparse.Namespace, run_root: Path) -> None:
    preset = MODEL_PRESETS[model_slug]
    out_dir = run_root / f"{domain}_{model_slug}"
    frontier_args = build_frontier_args(domain, preset, args, out_dir)
    print(f"Running {domain} {model_slug} ({preset.label}) -> {out_dir}", flush=True)

    if args.dry_run:
        print("  " + " ".join(["frontier_arithmetic_baseline.py", *frontier_args]), flush=True)
        return

    status = fab.main(frontier_args)
    if status != 0:
        raise RuntimeError(f"{domain} {model_slug} failed with status {status}")
    if args.update_outputs:
        target = copy_stable_output(domain, model_slug, out_dir)
        print(f"Updated {target}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    known_models = ", ".join(MODEL_PRESETS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["fraction", "decimal", "both"], default="both")
    parser.add_argument("--models", default="all", help=f"Comma-separated slugs or 'all'. Known slugs: {known_models}")
    parser.add_argument("--run-root", default=None, help="Defaults to eval/runs/<timestamp>.")
    parser.add_argument("--fraction-samples-per-problem", type=int, default=5)
    parser.add_argument("--decimal-samples-per-problem", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all requested rows.")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--token-limit-param", choices=["auto", "max_tokens", "max_completion_tokens"], default="auto")
    parser.add_argument("--fraction-problem-column", default="prob")
    parser.add_argument("--decimal-problem-column", default="prob")
    parser.add_argument("--api-key-env", default="AI_SANDBOX_KEY")
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--gemini-api-version", default="v1alpha")
    parser.add_argument(
        "--portkey-base-url",
        default=os.environ.get("AI_SANDBOX_PORTKEY_BASE_URL", fab.DEFAULT_PORTKEY_BASE_URL),
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-outputs", dest="update_outputs", action="store_true", default=True)
    parser.add_argument("--no-update-outputs", dest="update_outputs", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = resolve_path(args.run_root, EVAL_DIR / "runs" / timestamp)
    domains = selected_domains(args.domain)
    model_slugs = selected_model_slugs(args.models)

    fab.load_problems = build_eval_loader(args.fraction_problem_column, args.decimal_problem_column)

    print(f"Run root: {run_root}", flush=True)
    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        for model_slug in model_slugs:
            run_one(domain, model_slug, args, run_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
