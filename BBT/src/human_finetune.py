#!/usr/bin/env python3
"""Subpipeline: human-data finetuning on top of distilled model."""

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


SUBPIPELINE_NAME = "human_finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run human-finetune subpipeline.")
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
    paths_cfg = ensure_dict(config, "paths")
    exp_cfg = ensure_dict(config, "experiment")

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
        raise ValueError("human_finetune subpipeline requires distill run directory input.")

    script = resolve_path(sub_cfg.get("script", "human_ft/finetune_humandata.py"), run_dir)
    train_csv = resolve_path(
        sub_cfg.get("train_csv") or paths_cfg.get("human_train_csv", "data/human_ft/data_train_nlp.csv"),
        run_dir,
    )
    val_csv = resolve_path(
        sub_cfg.get("val_csv") or paths_cfg.get("human_val_csv", "data/human_ft/data_val_nlp.csv"),
        run_dir,
    )
    sp2013_csv = resolve_path(
        sub_cfg.get("sp2013_csv") or paths_cfg.get("sp2013_csv", "results/UMA_replication/sp2013.csv"),
        run_dir,
    )

    cmd = [
        python_exec,
        str(script),
        "--model_path",
        str(model_run_dir),
        "--checkpoint_subdir",
        str(sub_cfg.get("checkpoint_subdir", "best")),
        "--train_csv",
        str(train_csv),
        "--val_csv",
        str(val_csv),
        "--prompt_col",
        str(sub_cfg.get("prompt_col", "instruction_nl")),
        "--response_col",
        str(sub_cfg.get("response_col", "response_nl")),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(sub_cfg.get("epochs", 3)),
        "--batch_size",
        str(sub_cfg.get("batch_size", 2)),
        "--eval_batch_size",
        str(sub_cfg.get("eval_batch_size", 2)),
        "--grad_accum_steps",
        str(sub_cfg.get("grad_accum_steps", 16)),
        "--lr",
        str(sub_cfg.get("lr", 1.5e-5)),
        "--weight_decay",
        str(sub_cfg.get("weight_decay", 0.01)),
        "--warmup_ratio",
        str(sub_cfg.get("warmup_ratio", 0.03)),
        "--max_grad_norm",
        str(sub_cfg.get("max_grad_norm", 1.0)),
        "--max_length",
        str(sub_cfg.get("max_length", 192)),
        "--num_workers",
        str(sub_cfg.get("num_workers", 2)),
        "--seed",
        str(sub_cfg.get("seed", exp_cfg.get("seed", 42))),
        "--log_every",
        str(sub_cfg.get("log_every", 20)),
        "--amp_dtype",
        str(sub_cfg.get("amp_dtype", "auto")),
        "--best_by",
        str(sub_cfg.get("best_by", "val_nll")),
        "--sp2013_csv",
        str(sp2013_csv),
        "--sp2013_max_new_tokens",
        str(sub_cfg.get("sp2013_max_new_tokens", 192)),
        "--sp2013_num_rollouts",
        str(sub_cfg.get("sp2013_num_rollouts", 1000)),
        "--sp2013_temperature",
        str(sub_cfg.get("sp2013_temperature", 0.7)),
        "--sp2013_top_p",
        str(sub_cfg.get("sp2013_top_p", 0.95)),
        "--sp2013_top_k",
        str(sub_cfg.get("sp2013_top_k", 50)),
        "--sp2013_sample_seed",
        str(sub_cfg.get("sp2013_sample_seed", 123)),
        "--sp2013_rollout_batch_size",
        str(sub_cfg.get("sp2013_rollout_batch_size", 8)),
        "--sp2013_progress_every",
        str(sub_cfg.get("sp2013_progress_every", 1000)),
        "--sp2013_grid_g",
        str(sub_cfg.get("sp2013_grid_g", "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")),
        "--sp2013_grid_d",
        str(sub_cfg.get("sp2013_grid_d", "0.1,0.3,0.5,0.7,0.9")),
        "--sp2013_grid_rt",
        str(sub_cfg.get("sp2013_grid_rt", "3,4,5,6")),
        "--sp2013_grid_ice",
        str(sub_cfg.get("sp2013_grid_ice", "0,25,50,75,100")),
    ]

    add_store_true(cmd, "--gradient_checkpointing", bool(sub_cfg.get("gradient_checkpointing", False)))
    add_store_true(cmd, "--train_on_prompt", bool(sub_cfg.get("train_on_prompt", False)))
    add_bool_optional(cmd, "--local_files_only", bool(sub_cfg.get("local_files_only", True)))
    add_bool_optional(cmd, "--eval_sp2013", bool(sub_cfg.get("eval_sp2013", False)))
    add_bool_optional(cmd, "--eval_loaded_model", bool(sub_cfg.get("eval_loaded_model", False)))
    add_bool_optional(cmd, "--sp2013_do_sample", bool(sub_cfg.get("sp2013_do_sample", True)))
    add_bool_optional(cmd, "--sp2013_use_param_grid", bool(sub_cfg.get("sp2013_use_param_grid", False)))

    run_command(cmd, dry_run=dry_run)

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
            "train_csv": str(train_csv.resolve()),
            "val_csv": str(val_csv.resolve()),
            "sp2013_csv": str(sp2013_csv.resolve()),
        },
        primary_output=output_dir,
        secondary_outputs=[
            str((output_dir / "summary_metrics.json").resolve()),
            str((output_dir / "history.json").resolve()),
        ],
        metrics_summary={"output_exists": output_dir.exists()},
        upstream_manifest=upstream_manifest,
    )
    maybe_write_manifest(manifest, manifest_path, dry_run=dry_run)
    return SubpipelineResult(manifest=manifest, manifest_path=manifest_path, primary_output=output_dir)


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
