#!/usr/bin/env python3
"""BBT experiment orchestrator from an experiment-subpipeline perspective."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src import distill_eval, distill_train, human_eval, human_finetune, translation, uma_traces  # type: ignore
    from src.common import (  # type: ignore
        REPO_ROOT,
        SUBPIPELINE_ORDER,
        SubpipelineResult,
        ensure_dict,
        load_config,
        now_utc_iso,
        resolve_path,
        write_json,
    )
else:
    from .src import distill_eval, distill_train, human_eval, human_finetune, translation, uma_traces
    from .src.common import (
        REPO_ROOT,
        SUBPIPELINE_ORDER,
        SubpipelineResult,
        ensure_dict,
        load_config,
        now_utc_iso,
        resolve_path,
        write_json,
    )


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default_pipeline.json"

SUBPIPELINE_RUNNERS = {
    "uma_traces": uma_traces.run_subpipeline,
    "translation": translation.run_subpipeline,
    "distill_train": distill_train.run_subpipeline,
    "distill_eval": distill_eval.run_subpipeline,
    "human_finetune": human_finetune.run_subpipeline,
    "human_eval": human_eval.run_subpipeline,
}

UPSTREAM_SOURCE = {
    "uma_traces": None,
    "translation": "uma_traces",
    "distill_train": "translation",
    "distill_eval": "distill_train",
    "human_finetune": "distill_train",
    "human_eval": "human_finetune",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BBT experiment subpipelines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Pipeline config JSON path.")
    parser.add_argument("--run-dir", default="", help="Explicit run dir (optional).")
    parser.add_argument(
        "--subpipelines",
        default="all",
        help=f"Comma list or 'all'. Known: {', '.join(SUBPIPELINE_ORDER)}",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Config override(s).")
    parser.add_argument("--python", default="python", help="Python executable for wrapped stage scripts.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_subpipeline_spec(spec: str) -> List[str]:
    text = str(spec).strip().lower()
    if text == "all":
        return list(SUBPIPELINE_ORDER)
    selected = [token.strip() for token in str(spec).split(",") if token.strip()]
    if not selected:
        raise ValueError("--subpipelines resolved to an empty list.")
    unknown = [name for name in selected if name not in SUBPIPELINE_ORDER]
    if unknown:
        raise ValueError(f"Unknown subpipeline(s): {unknown}. Known: {SUBPIPELINE_ORDER}")
    return selected


def resolve_run_dir(cfg: Dict[str, Any], explicit_run_dir: str) -> Path:
    if str(explicit_run_dir).strip():
        candidate = Path(str(explicit_run_dir).strip())
        if candidate.is_absolute():
            return candidate.resolve()
        return (REPO_ROOT / candidate).resolve()

    exp_cfg = ensure_dict(cfg, "experiment")
    run_root = resolve_path(exp_cfg.get("run_root", "results/bbt"), run_dir=REPO_ROOT)
    run_name = str(exp_cfg.get("name", "bbt_experiment")).strip() or "bbt_experiment"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (run_root / f"{run_name}_{ts}").resolve()


def stage_enabled(cfg: Dict[str, Any], subpipeline_name: str) -> bool:
    stage_cfg = ensure_dict(ensure_dict(cfg, "subpipelines"), subpipeline_name)
    return bool(stage_cfg.get("enabled", True))


def build_pipeline_summary(
    config_path: Path,
    run_dir: Path,
    selected: List[str],
    executed: List[str],
    skipped: List[str],
    failed: Optional[str],
    results: Dict[str, SubpipelineResult],
) -> Dict[str, Any]:
    manifests: Dict[str, Any] = {}
    for name, result in results.items():
        manifests[name] = {
            "manifest_path": str(result.manifest_path.resolve()),
            "primary_output": str(result.primary_output.resolve()),
            "status": result.manifest.get("status", "unknown"),
        }

    return {
        "generated_at_utc": now_utc_iso(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "selected_subpipelines": selected,
        "executed_subpipelines": executed,
        "skipped_subpipelines": skipped,
        "failed_subpipeline": failed,
        "subpipeline_results": manifests,
    }


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path, overrides=args.set)
    selected = parse_subpipeline_spec(args.subpipelines)
    run_dir = resolve_run_dir(cfg, args.run_dir)

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "resolved_config.json", cfg)

    print("=" * 80, flush=True)
    print("BBT EXPERIMENT PIPELINE", flush=True)
    print("=" * 80, flush=True)
    print(f"config: {cfg_path}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"selected: {selected}", flush=True)
    print(f"dry_run: {args.dry_run}", flush=True)
    print("", flush=True)

    results: Dict[str, SubpipelineResult] = {}
    executed: List[str] = []
    skipped: List[str] = []
    failed: Optional[str] = None

    for name in SUBPIPELINE_ORDER:
        if name not in selected:
            continue
        if not stage_enabled(cfg, name):
            print(f"[skip] {name} (enabled=false)", flush=True)
            skipped.append(name)
            continue

        prev = UPSTREAM_SOURCE[name]
        upstream_manifest: Optional[Path] = None
        upstream_primary_output: Optional[Path] = None
        if prev and prev in results:
            upstream_manifest = results[prev].manifest_path
            upstream_primary_output = results[prev].primary_output

        print(f"\n[subpipeline] {name}", flush=True)
        runner = SUBPIPELINE_RUNNERS[name]
        try:
            result = runner(
                config=cfg,
                run_dir=run_dir,
                python_exec=args.python,
                dry_run=args.dry_run,
                upstream_manifest=upstream_manifest,
                upstream_primary_output=upstream_primary_output,
            )
            results[name] = result
            executed.append(name)
            print(f"manifest: {result.manifest_path}", flush=True)
            print(f"primary_output: {result.primary_output}", flush=True)
        except Exception:
            failed = name
            raise

    summary = build_pipeline_summary(
        config_path=cfg_path,
        run_dir=run_dir,
        selected=selected,
        executed=executed,
        skipped=skipped,
        failed=failed,
        results=results,
    )

    print("", flush=True)
    print("Done.", flush=True)
    print(f"executed: {executed}", flush=True)
    print(f"skipped: {skipped}", flush=True)
    if failed is not None:
        print(f"failed: {failed}", flush=True)

    summary_path = run_dir / "pipeline_summary.json"
    if not args.dry_run:
        write_json(summary_path, summary)
        print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
