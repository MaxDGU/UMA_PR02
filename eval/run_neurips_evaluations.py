#!/usr/bin/env python3
"""Run the reproducible NeurIPS evaluation entrypoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from eval import eval_utils as eu
except ImportError:  # pragma: no cover - supports direct execution from eval/
    import eval_utils as eu


VALID_STAGES = ("error", "procedural", "humanlike")


def parse_stages(text: str) -> list[str]:
    stages = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(stages).difference(VALID_STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stages: {', '.join(unknown)}")
    return stages or list(VALID_STAGES)


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=eu.ROOT_DIR, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=parse_stages, default=list(VALID_STAGES), help="Comma-separated subset of error,procedural,humanlike.")
    parser.add_argument("--out-root", type=Path, default=eu.EVAL_DIR / "results")
    parser.add_argument("--domain", choices=["fraction", "decimal", "both"], default="both")
    parser.add_argument("--requests-only", action="store_true", help="Use request-generation backends for judge stages.")
    parser.add_argument("--procedural-max-rows", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_root = args.out_root if args.out_root.is_absolute() else eu.ROOT_DIR / args.out_root

    if "error" in args.stages:
        command = [
            sys.executable,
            "eval/evaluate_error_alignment.py",
            "--domain",
            args.domain,
            "--out-dir",
            str(out_root / "error_alignment"),
        ]
        run_command(command)

    if "procedural" in args.stages:
        command = [
            sys.executable,
            "eval/evaluate_procedural_alignment.py",
            "--domain",
            args.domain,
            "--backend",
            "requests" if args.requests_only else "gemini_batch",
            "--max-rows",
            str(args.procedural_max_rows),
            "--out-dir",
            str(out_root / "procedural_alignment"),
        ]
        if args.force:
            command.append("--force")
        run_command(command)

    if "humanlike" in args.stages:
        command = [
            sys.executable,
            "eval/evaluate_humanlike_preference.py",
            "--backend",
            "requests" if args.requests_only else "sandbox",
            "--out-dir",
            str(out_root / "humanlike_preference" / "distilled_vs_frontier"),
        ]
        if args.force:
            command.append("--force")
        run_command(command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
