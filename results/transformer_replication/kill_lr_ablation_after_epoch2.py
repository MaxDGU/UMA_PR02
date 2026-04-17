#!/usr/bin/env python3
"""
Watch a learning-rate ablation manifest and cancel remaining jobs after epoch 2.

Default behavior:
  - resolve the latest jobs_135m_paramsalways_lr_scheduler_ablation_*.tsv manifest
  - watch stage-1 runs
  - wait until each stage-1 out_dir has id_val_metrics_epoch_02.json
  - cancel any still-active jobs from that manifest
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_GLOB = "jobs_135m_paramsalways_lr_scheduler_ablation_*.tsv"
DEFAULT_SENTINEL = "id_val_metrics_epoch_02.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cancel LR-ablation jobs after epoch-2 eval is written.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to a jobs_135m_paramsalways_lr_scheduler_ablation_*.tsv manifest. Defaults to the latest one.",
    )
    parser.add_argument(
        "--manifest-glob",
        type=str,
        default=DEFAULT_MANIFEST_GLOB,
        help="Glob used to resolve the latest manifest when --manifest is omitted.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Stage whose out_dir should be checked for epoch completion.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2,
        help="Epoch number whose completion artifact should trigger cancellation.",
    )
    parser.add_argument(
        "--sentinel-name",
        type=str,
        default=DEFAULT_SENTINEL,
        help="Filename expected inside each stage out_dir when the target epoch has finished.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=120,
        help="Polling interval when not using --once.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit instead of polling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which job ids would be canceled without calling scancel.",
    )
    return parser.parse_args()


def now_string() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_manifest(manifest_arg: str, manifest_glob: str) -> Path:
    if manifest_arg.strip():
        path = Path(manifest_arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    candidates = sorted(
        Path(p).resolve()
        for p in glob.glob(str(SCRIPT_DIR / manifest_glob))
        if Path(p).is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No manifest matched {manifest_glob!r} under {SCRIPT_DIR}"
        )
    return candidates[-1]


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def filter_stage_rows(rows: Iterable[Dict[str, str]], stage: int) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        try:
            row_stage = int(str(row.get("stage", "")).strip())
        except ValueError:
            continue
        if row_stage == stage:
            out.append(row)
    return out


def sentinel_path_for_row(row: Dict[str, str], sentinel_name: str) -> Path:
    out_dir = Path(str(row["out_dir"]).strip())
    return out_dir / sentinel_name


def row_is_complete(row: Dict[str, str], sentinel_name: str) -> bool:
    return sentinel_path_for_row(row, sentinel_name).is_file()


def active_job_ids(job_ids: List[str]) -> List[str]:
    clean_ids = [job_id for job_id in job_ids if str(job_id).strip()]
    if not clean_ids:
        return []
    proc = subprocess.run(
        ["squeue", "-h", "-o", "%A", "-j", ",".join(clean_ids)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 and proc.stdout.strip() == "":
        return []
    active = []
    for line in proc.stdout.splitlines():
        job_id = line.strip()
        if job_id:
            active.append(job_id)
    return active


def cancel_jobs(job_ids: List[str], dry_run: bool) -> None:
    if not job_ids:
        return
    if dry_run:
        print(f"{now_string()} dry-run: would cancel {','.join(job_ids)}", flush=True)
        return
    proc = subprocess.run(["scancel", *job_ids], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(f"scancel failed for {job_ids}: {stderr or 'unknown error'}")
    print(f"{now_string()} canceled jobs: {','.join(job_ids)}", flush=True)


def print_status(stage_rows: List[Dict[str, str]], sentinel_name: str, active_ids: List[str]) -> None:
    completed = [row for row in stage_rows if row_is_complete(row, sentinel_name)]
    pending = [row for row in stage_rows if not row_is_complete(row, sentinel_name)]
    print(
        f"{now_string()} epoch-2 monitor: "
        f"completed_stage_runs={len(completed)}/{len(stage_rows)} "
        f"active_manifest_jobs={len(active_ids)}",
        flush=True,
    )
    for row in pending:
        sentinel = sentinel_path_for_row(row, sentinel_name)
        print(
            f"  waiting: job_id={row['job_id']} job_name={row['job_name']} sentinel={sentinel}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be >= 1.")

    manifest_path = resolve_manifest(args.manifest, args.manifest_glob)
    sentinel_name = args.sentinel_name
    if sentinel_name == DEFAULT_SENTINEL and args.epoch != 2:
        sentinel_name = f"id_val_metrics_epoch_{args.epoch:02d}.json"

    rows = load_manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    stage_rows = filter_stage_rows(rows, args.stage)
    if not stage_rows:
        raise ValueError(f"No rows with stage={args.stage} found in {manifest_path}")

    all_job_ids = [str(row["job_id"]).strip() for row in rows if str(row.get("job_id", "")).strip()]
    print(f"{now_string()} monitor manifest: {manifest_path}", flush=True)
    print(
        f"{now_string()} monitor config: stage={args.stage} epoch={args.epoch} "
        f"sentinel={sentinel_name} once={args.once} dry_run={args.dry_run} poll_seconds={args.poll_seconds}",
        flush=True,
    )

    while True:
        active_ids = active_job_ids(all_job_ids)
        print_status(stage_rows, sentinel_name, active_ids)

        all_complete = all(row_is_complete(row, sentinel_name) for row in stage_rows)
        if all_complete:
            remaining = active_job_ids(all_job_ids)
            if remaining:
                cancel_jobs(remaining, dry_run=args.dry_run)
            else:
                print(f"{now_string()} no active manifest jobs remain; nothing to cancel.", flush=True)
            return 0

        if args.once:
            print(f"{now_string()} target epoch not finished yet; exiting because --once was set.", flush=True)
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"{now_string()} interrupted.", flush=True)
        raise SystemExit(130)
