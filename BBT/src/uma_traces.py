#!/usr/bin/env python3
"""Subpipeline: generate UMA traces from saved UMA models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.common import (  # type: ignore
        SubpipelineResult,
        add_bool_optional,
        add_store_true,
        build_manifest,
        ensure_dict,
        load_config,
        maybe_write_manifest,
        now_utc_iso,
        resolve_output_dir,
        resolve_path,
        run_command,
    )
else:
    from .common import (
        SubpipelineResult,
        add_bool_optional,
        add_store_true,
        build_manifest,
        ensure_dict,
        load_config,
        maybe_write_manifest,
        now_utc_iso,
        resolve_output_dir,
        resolve_path,
        run_command,
    )


SUBPIPELINE_NAME = "uma_traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UMA trace-generation subpipeline.")
    parser.add_argument("--config", required=True, help="Path to resolved pipeline config JSON.")
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--python", default="python", help="Python executable for wrapped scripts.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def run_subpipeline(
    config: Dict[str, Any],
    run_dir: Path,
    python_exec: str,
    dry_run: bool,
    upstream_manifest: Optional[Path] = None,
    upstream_primary_output: Optional[Path] = None,
) -> SubpipelineResult:
    del upstream_manifest, upstream_primary_output

    started = now_utc_iso()
    sub_cfg = ensure_dict(ensure_dict(config, "subpipelines"), SUBPIPELINE_NAME)
    paths_cfg = ensure_dict(config, "paths")

    output_dir = resolve_output_dir(config, SUBPIPELINE_NAME, run_dir)
    manifest_path = output_dir / "manifest.json"
    default_out_csv = str(output_dir / "uma_traces.csv")
    out_csv = resolve_path(sub_cfg.get("out_csv", default_out_csv), run_dir)

    script = resolve_path(
        sub_cfg.get("script", "results/UMA_replication/run_saved_models_on_problem_set.py"),
        run_dir,
    )
    models_dir = resolve_path(
        sub_cfg.get("models_dir") or paths_cfg.get("uma_models_dir", "results/UMA_replication/trained_models"),
        run_dir,
    )
    problem_csv = resolve_path(
        sub_cfg.get("problem_csv") or paths_cfg.get("uma_problem_csv", "1. Model/Problem Sets Training/go_math.csv"),
        run_dir,
    )

    cmd = [
        python_exec,
        str(script),
        "--models-dir",
        str(models_dir),
        "--problem-csv",
        str(problem_csv),
        "--problem-col",
        str(sub_cfg.get("problem_col", "prob")),
        "--out-csv",
        str(out_csv),
        "--progress-every",
        str(sub_cfg.get("progress_every", 10)),
        "--trace-max-subproblems-per-step",
        str(sub_cfg.get("trace_max_subproblems_per_step", 6)),
    ]

    start_subjid = sub_cfg.get("start_subjid")
    end_subjid = sub_cfg.get("end_subjid")
    if start_subjid is not None:
        cmd.extend(["--start-subjid", str(start_subjid)])
    if end_subjid is not None:
        cmd.extend(["--end-subjid", str(end_subjid)])

    add_store_true(cmd, "--dedup-problems", bool(sub_cfg.get("dedup_problems", False)))
    add_store_true(cmd, "--resume", bool(sub_cfg.get("resume", True)))
    add_store_true(cmd, "--overwrite", bool(sub_cfg.get("overwrite", False)))
    add_bool_optional(cmd, "--include-trace-steps", bool(sub_cfg.get("include_trace_steps", True)))

    run_command(cmd, dry_run=dry_run)

    finished = now_utc_iso()
    status = "dry_run" if dry_run else "success"
    metrics_summary = {
        "output_exists": out_csv.exists(),
        "include_trace_steps": bool(sub_cfg.get("include_trace_steps", True)),
    }

    manifest = build_manifest(
        subpipeline_name=SUBPIPELINE_NAME,
        status=status,
        started_at_utc=started,
        finished_at_utc=finished,
        config_snapshot=sub_cfg,
        inputs={
            "models_dir": str(models_dir.resolve()),
            "problem_csv": str(problem_csv.resolve()),
        },
        primary_output=out_csv,
        secondary_outputs=[],
        metrics_summary=metrics_summary,
        upstream_manifest=None,
    )
    maybe_write_manifest(manifest, manifest_path, dry_run=dry_run)
    return SubpipelineResult(manifest=manifest, manifest_path=manifest_path, primary_output=out_csv)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve(), overrides=args.set)
    run_subpipeline(
        config=cfg,
        run_dir=Path(args.run_dir).resolve(),
        python_exec=args.python,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
