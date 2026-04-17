#!/usr/bin/env python3
"""Subpipeline: sampling-based evaluation of distilled model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.common import (  # type: ignore
        SubpipelineResult,
        add_bool_optional,
        build_manifest,
        ensure_dict,
        load_config,
        maybe_write_manifest,
        now_utc_iso,
        read_json,
        resolve_output_dir,
        resolve_path,
        resolve_upstream_input,
        run_command,
    )
else:
    from .common import (
        SubpipelineResult,
        add_bool_optional,
        build_manifest,
        ensure_dict,
        load_config,
        maybe_write_manifest,
        now_utc_iso,
        read_json,
        resolve_output_dir,
        resolve_path,
        resolve_upstream_input,
        run_command,
    )


SUBPIPELINE_NAME = "distill_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distill-eval subpipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--upstream-manifest", default="")
    parser.add_argument("--upstream-primary-output", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def _resolve_datasets(config: Dict[str, Any], run_dir: Path, stage_cfg: Dict[str, Any]) -> list[dict[str, str]]:
    paths_cfg = ensure_dict(config, "paths")
    raw = stage_cfg.get("datasets")
    if raw is None:
        raw = [
            {
                "name": "sp2013",
                "csv": paths_cfg.get("sp2013_csv", "results/UMA_replication/sp2013.csv"),
                "problem_col": "prob",
            }
        ]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("subpipelines.distill_eval.datasets must be a non-empty list.")

    out: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{idx}] must be an object.")
        name = str(item.get("name", "")).strip()
        csv_path = str(item.get("csv", "")).strip()
        problem_col = str(item.get("problem_col", "prob")).strip() or "prob"
        if name == "" or csv_path == "":
            raise ValueError(f"datasets[{idx}] requires non-empty name and csv.")
        out.append({
            "name": name,
            "csv": str(resolve_path(csv_path, run_dir)),
            "problem_col": problem_col,
        })
    return out


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

    model_run_dir = resolve_upstream_input(
        config=config,
        subpipeline_name=SUBPIPELINE_NAME,
        run_dir=run_dir,
        upstream_manifest=upstream_manifest,
        upstream_primary_output=upstream_primary_output,
    )
    if model_run_dir is None:
        raise ValueError("distill_eval subpipeline requires distill run directory input.")

    eval_script = resolve_path(sub_cfg.get("script", "BBT/eval_sampling.py"), run_dir)
    out_csv = resolve_path(sub_cfg.get("out_csv", str(output_dir / "eval_samples.csv")), run_dir)
    out_json = resolve_path(sub_cfg.get("out_json", str(output_dir / "eval_metrics.json")), run_dir)
    datasets = _resolve_datasets(config, run_dir, sub_cfg)

    cmd = [
        python_exec,
        str(eval_script),
        "--run_dir",
        str(model_run_dir),
        "--checkpoint_subdir",
        str(sub_cfg.get("checkpoint_subdir", "best")),
        "--datasets_json",
        json.dumps(datasets, separators=(",", ":")),
        "--out_csv",
        str(out_csv),
        "--out_json",
        str(out_json),
        "--sp2013_grid_g",
        str(sub_cfg.get("sp2013_grid_g", "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")),
        "--sp2013_grid_d",
        str(sub_cfg.get("sp2013_grid_d", "0.1,0.3,0.5,0.7,0.9")),
        "--sp2013_grid_rt",
        str(sub_cfg.get("sp2013_grid_rt", "3,4,5,6")),
        "--sp2013_grid_ice",
        str(sub_cfg.get("sp2013_grid_ice", "0,25,50,75,100")),
        "--max_new_tokens",
        str(sub_cfg.get("max_new_tokens", 128)),
        "--num_samples_per_prompt",
        str(sub_cfg.get("num_samples_per_prompt", 5)),
        "--temperature",
        str(sub_cfg.get("temperature", 0.7)),
        "--top_p",
        str(sub_cfg.get("top_p", 0.95)),
        "--top_k",
        str(sub_cfg.get("top_k", 50)),
        "--sample_seed",
        str(sub_cfg.get("sample_seed", 1001)),
        "--progress_every",
        str(sub_cfg.get("progress_every", 100)),
        "--local_files_only",
        str(sub_cfg.get("local_files_only", "auto")),
    ]

    model_source = str(sub_cfg.get("model_source", "")).strip()
    if model_source:
        cmd.extend(["--model_source", model_source])

    add_bool_optional(cmd, "--use_param_grid", bool(sub_cfg.get("use_param_grid", True)))

    run_command(cmd, dry_run=dry_run)

    metrics_summary: Dict[str, Any] = {"output_exists": out_json.exists()}
    if out_json.exists() and not dry_run:
        try:
            payload = read_json(out_json)
            metrics_summary = payload.get("overall", payload)
        except Exception:
            pass

    finished = now_utc_iso()
    status = "dry_run" if dry_run else "success"
    manifest = build_manifest(
        subpipeline_name=SUBPIPELINE_NAME,
        status=status,
        started_at_utc=started,
        finished_at_utc=finished,
        config_snapshot=sub_cfg,
        inputs={
            "model_run_dir": str(model_run_dir.resolve()),
            "datasets": json.dumps(datasets, separators=(",", ":")),
        },
        primary_output=out_csv,
        secondary_outputs=[str(out_json.resolve())],
        metrics_summary=metrics_summary,
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
