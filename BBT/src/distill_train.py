#!/usr/bin/env python3
"""Subpipeline: train distilled LM on translated UMA traces."""

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


SUBPIPELINE_NAME = "distill_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distill-training subpipeline.")
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

    data_csv = resolve_upstream_input(
        config=config,
        subpipeline_name=SUBPIPELINE_NAME,
        run_dir=run_dir,
        upstream_manifest=upstream_manifest,
        upstream_primary_output=upstream_primary_output,
    )
    if data_csv is None:
        raise ValueError("distill_train subpipeline requires translated NLP CSV input.")

    script = resolve_path(
        sub_cfg.get("script", "results/transformer_replication/train_transformer_hf.py"),
        run_dir,
    )
    sp2013_csv = resolve_path(
        sub_cfg.get("sp2013_csv") or paths_cfg.get("sp2013_csv", "results/UMA_replication/sp2013.csv"),
        run_dir,
    )

    cmd = [
        python_exec,
        str(script),
        "--model_name",
        str(sub_cfg.get("model_name", "HuggingFaceTB/SmolLM2-1.7B")),
        "--data_csv",
        str(data_csv),
        "--output_dir",
        str(output_dir),
        "--prompt_col",
        str(sub_cfg.get("prompt_col", "instruction_nl")),
        "--response_col",
        str(sub_cfg.get("response_col", "response_nl")),
        "--epochs",
        str(sub_cfg.get("epochs", 1)),
        "--batch_size",
        str(sub_cfg.get("batch_size", 4)),
        "--eval_batch_size",
        str(sub_cfg.get("eval_batch_size", 4)),
        "--grad_accum_steps",
        str(sub_cfg.get("grad_accum_steps", 8)),
        "--lr",
        str(sub_cfg.get("lr", 2e-5)),
        "--weight_decay",
        str(sub_cfg.get("weight_decay", 0.01)),
        "--warmup_ratio",
        str(sub_cfg.get("warmup_ratio", 0.03)),
        "--max_grad_norm",
        str(sub_cfg.get("max_grad_norm", 1.0)),
        "--max_length",
        str(sub_cfg.get("max_length", 256)),
        "--val_frac",
        str(sub_cfg.get("val_frac", 0.05)),
        "--test_frac",
        str(sub_cfg.get("test_frac", 0.05)),
        "--problem_col",
        str(sub_cfg.get("problem_col", "prob")),
        "--num_workers",
        str(sub_cfg.get("num_workers", 4)),
        "--seed",
        str(sub_cfg.get("seed", exp_cfg.get("seed", 42))),
        "--log_every",
        str(sub_cfg.get("log_every", 20)),
        "--amp_dtype",
        str(sub_cfg.get("amp_dtype", "auto")),
        "--preview_samples",
        str(sub_cfg.get("preview_samples", 3)),
        "--preview_max_new_tokens",
        str(sub_cfg.get("preview_max_new_tokens", 96)),
        "--sp2013_csv",
        str(sp2013_csv),
        "--sp2013_max_new_tokens",
        str(sub_cfg.get("sp2013_max_new_tokens", 128)),
        "--sp2013_every",
        str(sub_cfg.get("sp2013_every", 1)),
        "--sp2013_grid_g",
        str(sub_cfg.get("sp2013_grid_g", "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")),
        "--sp2013_grid_d",
        str(sub_cfg.get("sp2013_grid_d", "0.1,0.3,0.5,0.7,0.9")),
        "--sp2013_grid_rt",
        str(sub_cfg.get("sp2013_grid_rt", "3,4,5,6")),
        "--sp2013_grid_ice",
        str(sub_cfg.get("sp2013_grid_ice", "0,25,50,75,100")),
        "--evals_per_epoch",
        str(sub_cfg.get("evals_per_epoch", 1)),
        "--best_by",
        str(sub_cfg.get("best_by", "val_loss")),
        "--id_eval_split",
        str(sub_cfg.get("id_eval_split", "val")),
        "--id_eval_size",
        str(sub_cfg.get("id_eval_size", 64)),
        "--id_eval_max_new_tokens",
        str(sub_cfg.get("id_eval_max_new_tokens", 64)),
        "--id_eval_every",
        str(sub_cfg.get("id_eval_every", 1)),
        "--id_eval_seed",
        str(sub_cfg.get("id_eval_seed", 123)),
        "--id_eval_target",
        str(sub_cfg.get("id_eval_target", "response")),
        "--resume_from",
        str(sub_cfg.get("resume_from", "")),
        "--save_last_every_updates",
        str(sub_cfg.get("save_last_every_updates", 0)),
        "--lora_r",
        str(sub_cfg.get("lora_r", 16)),
        "--lora_alpha",
        str(sub_cfg.get("lora_alpha", 32)),
        "--lora_dropout",
        str(sub_cfg.get("lora_dropout", 0.05)),
        "--lora_target_modules",
        str(sub_cfg.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")),
    ]

    max_samples = sub_cfg.get("max_samples")
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])

    domain = str(sub_cfg.get("domain", "")).strip()
    if domain:
        cmd.extend(["--domain", domain])

    train_prompt_student_mode = str(sub_cfg.get("train_prompt_student_mode", "")).strip()
    if train_prompt_student_mode:
        cmd.extend(["--train_prompt_student_mode", train_prompt_student_mode])

    add_store_true(cmd, "--save_best_only", bool(sub_cfg.get("save_best_only", False)))
    add_store_true(cmd, "--gradient_checkpointing", bool(sub_cfg.get("gradient_checkpointing", False)))
    add_store_true(cmd, "--train_on_prompt", bool(sub_cfg.get("train_on_prompt", False)))
    add_store_true(cmd, "--tf32", bool(sub_cfg.get("tf32", False)))

    add_bool_optional(cmd, "--split_by_problem", bool(sub_cfg.get("split_by_problem", False)))
    add_bool_optional(cmd, "--drop_train_eval_problem_overlap", bool(sub_cfg.get("drop_train_eval_problem_overlap", False)))
    add_bool_optional(cmd, "--move_sp2013_rows_to_val", bool(sub_cfg.get("move_sp2013_rows_to_val", True)))
    add_bool_optional(cmd, "--eval_sp2013", bool(sub_cfg.get("eval_sp2013", False)))
    add_bool_optional(cmd, "--sp2013_use_param_grid", bool(sub_cfg.get("sp2013_use_param_grid", True)))
    add_bool_optional(cmd, "--eval_id_final_answer", bool(sub_cfg.get("eval_id_final_answer", False)))
    add_bool_optional(cmd, "--use_lora", bool(sub_cfg.get("use_lora", False)))
    add_bool_optional(cmd, "--local_files_only", bool(sub_cfg.get("local_files_only", True)))
    add_bool_optional(cmd, "--check_format_consistency", bool(sub_cfg.get("check_format_consistency", True)))
    add_bool_optional(cmd, "--init_from_scratch", bool(sub_cfg.get("init_from_scratch", False)))

    run_command(cmd, dry_run=dry_run)

    finished = now_utc_iso()
    status = "dry_run" if dry_run else "success"
    secondary = [
        str((output_dir / "best").resolve()),
        str((output_dir / "last").resolve()),
        str((output_dir / "final").resolve()),
        str((output_dir / "history.json").resolve()),
    ]
    manifest = build_manifest(
        subpipeline_name=SUBPIPELINE_NAME,
        status=status,
        started_at_utc=started,
        finished_at_utc=finished,
        config_snapshot=sub_cfg,
        inputs={
            "translated_csv": str(data_csv.resolve()),
            "sp2013_csv": str(sp2013_csv.resolve()),
        },
        primary_output=output_dir,
        secondary_outputs=secondary,
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
