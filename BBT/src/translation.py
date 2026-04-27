#!/usr/bin/env python3
"""Subpipeline: translate UMA traces into NLP format."""

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
        resolve_upstream_input,
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
        resolve_upstream_input,
        run_command,
    )


SUBPIPELINE_NAME = "translation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trace translation subpipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--upstream-manifest", default="")
    parser.add_argument("--upstream-primary-output", default="")
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
    started = now_utc_iso()
    sub_cfg = ensure_dict(ensure_dict(config, "subpipelines"), SUBPIPELINE_NAME)

    output_dir = resolve_output_dir(config, SUBPIPELINE_NAME, run_dir)
    manifest_path = output_dir / "manifest.json"
    default_out_csv = str(output_dir / "uma_traces_nlp.csv.gz")
    out_csv = resolve_path(sub_cfg.get("out_csv", default_out_csv), run_dir)

    source_csv = resolve_upstream_input(
        config=config,
        subpipeline_name=SUBPIPELINE_NAME,
        run_dir=run_dir,
        upstream_manifest=upstream_manifest,
        upstream_primary_output=upstream_primary_output,
    )
    if source_csv is None:
        raise ValueError("translation subpipeline requires a trace CSV input.")

    script = resolve_path(
        sub_cfg.get("script", "results/transformer_replication/translate_uma_traces_to_nlp.py"),
        run_dir,
    )

    cmd = [
        python_exec,
        str(script),
        "--output-csv",
        str(out_csv),
        "--chunksize",
        str(sub_cfg.get("chunksize", 100000)),
        "--input-csv",
        str(source_csv),
        "--sp2013-eval-dir",
        str(resolve_path(sub_cfg.get("sp2013_eval_dir", "results/UMA_replication/sp2013_eval"), run_dir)),
        "--sp2013-seeds",
        str(sub_cfg.get("sp2013_seeds", "1-20")),
    ]

    add_bool_optional(cmd, "--include-default-uma-traces", bool(sub_cfg.get("include_default_uma_traces", False)))
    add_bool_optional(cmd, "--include-sp2013-seeds", bool(sub_cfg.get("include_sp2013_seeds", False)))

    student_prompt_mode = str(sub_cfg.get("student_prompt_mode", "")).strip()
    if student_prompt_mode:
        if "include_student_params" in sub_cfg:
            legacy_mode = "always" if bool(sub_cfg.get("include_student_params")) else "none"
            if student_prompt_mode != legacy_mode:
                raise ValueError(
                    "translation config conflict: student_prompt_mode="
                    f"{student_prompt_mode} disagrees with include_student_params={sub_cfg.get('include_student_params')}"
                )
        cmd.extend(["--student-prompt-mode", student_prompt_mode])

    student_params_drop_prob = sub_cfg.get("student_params_drop_prob")
    if student_params_drop_prob is not None:
        cmd.extend(["--student-params-drop-prob", str(student_params_drop_prob)])

    student_params_drop_seed = sub_cfg.get("student_params_drop_seed")
    if student_params_drop_seed is not None:
        cmd.extend(["--student-params-drop-seed", str(student_params_drop_seed)])

    if not student_prompt_mode and "include_student_params" in sub_cfg:
        add_bool_optional(cmd, "--include-student-params", bool(sub_cfg.get("include_student_params")))

    max_rows = sub_cfg.get("max_rows")
    if max_rows is not None:
        cmd.extend(["--max-rows", str(max_rows)])

    add_store_true(cmd, "--minimal-columns", bool(sub_cfg.get("minimal_columns", False)))
    add_store_true(cmd, "--include-outcome-text", bool(sub_cfg.get("include_outcome_text", False)))

    reasoning_mode = str(sub_cfg.get("reasoning_mode", "")).strip()
    if reasoning_mode:
        cmd.extend(["--reasoning-mode", reasoning_mode])

    row_filter = str(sub_cfg.get("row_filter", "")).strip()
    if row_filter:
        cmd.extend(["--row-filter", row_filter])

    add_bool_optional(
        cmd,
        "--surface-hidden-trace-steps",
        bool(sub_cfg.get("surface_hidden_trace_steps", False)),
    )

    extra_inputs = sub_cfg.get("input_csv_extra", [])
    if extra_inputs is None:
        extra_inputs = []
    if not isinstance(extra_inputs, list):
        raise ValueError("subpipelines.translation.input_csv_extra must be a list.")
    for raw in extra_inputs:
        text = str(raw).strip()
        if text:
            cmd.extend(["--input-csv", str(resolve_path(text, run_dir))])

    run_command(cmd, dry_run=dry_run)

    finished = now_utc_iso()
    status = "dry_run" if dry_run else "success"
    manifest = build_manifest(
        subpipeline_name=SUBPIPELINE_NAME,
        status=status,
        started_at_utc=started,
        finished_at_utc=finished,
        config_snapshot=sub_cfg,
        inputs={"source_trace_csv": str(source_csv.resolve())},
        primary_output=out_csv,
        secondary_outputs=[],
        metrics_summary={"output_exists": out_csv.exists()},
        upstream_manifest=upstream_manifest,
    )
    maybe_write_manifest(manifest, manifest_path, dry_run=dry_run)
    return SubpipelineResult(manifest=manifest, manifest_path=manifest_path, primary_output=out_csv)


def main() -> None:
    args = parse_args()
    upstream_manifest = Path(args.upstream_manifest).resolve() if str(args.upstream_manifest).strip() else None
    upstream_primary_output = Path(args.upstream_primary_output).resolve() if str(args.upstream_primary_output).strip() else None
    cfg = load_config(Path(args.config).resolve(), overrides=args.set)
    run_subpipeline(
        config=cfg,
        run_dir=Path(args.run_dir).resolve(),
        python_exec=args.python,
        dry_run=args.dry_run,
        upstream_manifest=upstream_manifest,
        upstream_primary_output=upstream_primary_output,
    )


if __name__ == "__main__":
    main()
