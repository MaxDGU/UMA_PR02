#!/usr/bin/env python3
"""Subpipeline: evaluate human-finetuned model (held-out NLL + one-shot sampling eval)."""

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
        write_json,
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
        write_json,
    )


SUBPIPELINE_NAME = "human_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run human-eval subpipeline.")
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
        raise ValueError("subpipelines.human_eval.datasets must be a non-empty list.")

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


def _extract_human_nll(human_run_dir: Path) -> Dict[str, Any]:
    summary_path = human_run_dir / "summary_metrics.json"
    history_path = human_run_dir / "history.json"

    summary = read_json(summary_path) if summary_path.exists() else {}
    history = read_json(history_path) if history_path.exists() else []

    final_val_nll = None
    if isinstance(history, list):
        # Prefer the last epoch row (skip possible loaded_model row).
        epoch_rows = [row for row in history if isinstance(row, dict) and "epoch" in row]
        if epoch_rows:
            final_val_nll = epoch_rows[-1].get("val_nll")

    return {
        "best_val_nll": summary.get("best_val_nll"),
        "best_sp2013_acc": summary.get("best_sp2013_acc"),
        "final_val_nll": final_val_nll,
        "summary_metrics_path": str(summary_path.resolve()) if summary_path.exists() else None,
        "history_path": str(history_path.resolve()) if history_path.exists() else None,
    }


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

    human_run_dir = resolve_upstream_input(
        config=config,
        subpipeline_name=SUBPIPELINE_NAME,
        run_dir=run_dir,
        upstream_manifest=upstream_manifest,
        upstream_primary_output=upstream_primary_output,
    )
    if human_run_dir is None:
        raise ValueError("human_eval subpipeline requires human-finetune run directory input.")

    eval_script = resolve_path(sub_cfg.get("script", "BBT/eval_sampling.py"), run_dir)
    sampling_csv = resolve_path(sub_cfg.get("sampling_csv", str(output_dir / "sampling_eval_samples.csv")), run_dir)
    sampling_json = resolve_path(sub_cfg.get("sampling_json", str(output_dir / "sampling_eval_metrics.json")), run_dir)
    final_metrics_path = resolve_path(sub_cfg.get("out_json", str(output_dir / "human_eval_metrics.json")), run_dir)

    datasets = _resolve_datasets(config, run_dir, sub_cfg)

    cmd = [
        python_exec,
        str(eval_script),
        "--run_dir",
        str(human_run_dir),
        "--checkpoint_subdir",
        str(sub_cfg.get("checkpoint_subdir", "best")),
        "--datasets_json",
        json.dumps(datasets, separators=(",", ":")),
        "--out_csv",
        str(sampling_csv),
        "--out_json",
        str(sampling_json),
        "--sp2013_grid_g",
        str(sub_cfg.get("sp2013_grid_g", "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")),
        "--sp2013_grid_d",
        str(sub_cfg.get("sp2013_grid_d", "0.1,0.3,0.5,0.7,0.9")),
        "--sp2013_grid_rt",
        str(sub_cfg.get("sp2013_grid_rt", "3,4,5,6")),
        "--sp2013_grid_ice",
        str(sub_cfg.get("sp2013_grid_ice", "0,25,50,75,100")),
        "--max_new_tokens",
        str(sub_cfg.get("max_new_tokens", 192)),
        "--num_samples_per_prompt",
        str(sub_cfg.get("num_samples_per_prompt", 1000)),
        "--temperature",
        str(sub_cfg.get("temperature", 0.7)),
        "--top_p",
        str(sub_cfg.get("top_p", 0.95)),
        "--top_k",
        str(sub_cfg.get("top_k", 50)),
        "--sample_seed",
        str(sub_cfg.get("sample_seed", 2001)),
        "--progress_every",
        str(sub_cfg.get("progress_every", 1000)),
        "--local_files_only",
        str(sub_cfg.get("local_files_only", "auto")),
    ]
    model_source = str(sub_cfg.get("model_source", "")).strip()
    if model_source:
        cmd.extend(["--model_source", model_source])

    add_bool_optional(cmd, "--use_param_grid", bool(sub_cfg.get("use_param_grid", False)))

    run_command(cmd, dry_run=dry_run)

    nll_metrics = _extract_human_nll(human_run_dir)
    sampling_metrics: Dict[str, Any] = {}
    if sampling_json.exists() and not dry_run:
        try:
            sampling_metrics = read_json(sampling_json)
        except Exception:
            sampling_metrics = {}

    combined = {
        "human_run_dir": str(human_run_dir.resolve()),
        "heldout_human_nll": nll_metrics,
        "sampling_eval": sampling_metrics,
    }
    if not dry_run:
        write_json(final_metrics_path, combined)

    finished = now_utc_iso()
    status = "dry_run" if dry_run else "success"
    manifest = build_manifest(
        subpipeline_name=SUBPIPELINE_NAME,
        status=status,
        started_at_utc=started,
        finished_at_utc=finished,
        config_snapshot=sub_cfg,
        inputs={
            "human_finetune_run_dir": str(human_run_dir.resolve()),
            "datasets": json.dumps(datasets, separators=(",", ":")),
        },
        primary_output=final_metrics_path,
        secondary_outputs=[str(sampling_csv.resolve()), str(sampling_json.resolve())],
        metrics_summary={
            "best_val_nll": nll_metrics.get("best_val_nll"),
            "sampling_overall": sampling_metrics.get("overall", {}).get("sample_overall_acc")
            if isinstance(sampling_metrics, dict)
            else None,
        },
        upstream_manifest=upstream_manifest,
    )
    maybe_write_manifest(manifest, manifest_path, dry_run=dry_run)
    return SubpipelineResult(manifest=manifest, manifest_path=manifest_path, primary_output=final_metrics_path)


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
