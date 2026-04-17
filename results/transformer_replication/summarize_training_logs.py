#!/usr/bin/env python3
"""
Summarize transformer training logs when checkpoint directories are missing.

This is useful for resumed runs that span multiple Slurm jobs: pass all related
`.out` files and the script will merge them by `output_dir`, extract the eval
records, and print a compact summary. It can also write a combined eval CSV and
summary JSON for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OUTPUT_DIR_RE = re.compile(r"^output_dir(?:=|:\s+)(?P<value>.+)$")
MODEL_RE = re.compile(r"^model:\s+(?P<value>.+)$")
DATA_RE = re.compile(r"^data:\s+(?P<value>.+)$")
ROWS_RE = re.compile(r"^rows (?P<split>train|val|test):\s+(?P<value>[0-9,]+)$")
INT_FIELD_PATTERNS = {
    "global_batch": re.compile(r"^global batch \(samples/update\):\s+(?P<value>[0-9,]+)$"),
    "updates_per_epoch": re.compile(r"^optimizer updates/epoch:\s+(?P<value>[0-9,]+)$"),
    "total_updates": re.compile(r"^total optimizer updates:\s+(?P<value>[0-9,]+)$"),
    "warmup_updates": re.compile(r"^warmup updates:\s+(?P<value>[0-9,]+)$"),
}
PROMPT_MODE_RE = re.compile(r"^train prompt student mode:\s+(?P<value>.+)$")
MOVE_SP2013_RE = re.compile(r"^move_sp2013_rows_to_val:\s+(?P<value>.+)$")
RESUME_RE = re.compile(
    r"^resuming from epoch=(?P<epoch>\d+), update_in_epoch=(?P<update_in_epoch>\d+), "
    r"global_update=(?P<global_update>\d+)$"
)
TRAIN_RE = re.compile(
    r"^epoch (?P<epoch>\d+)/(?P<epochs>\d+) \| update (?P<global_update>\d+)/(?P<total_updates>\d+) "
    r"\| train_loss (?P<train_loss>[-+0-9.eE]+) \| lr (?P<lr>[-+0-9.eE]+)$"
)
EVAL_RE = re.compile(
    r"^\[eval e(?P<epoch>\d+)_u(?P<update_in_epoch>\d+)\] "
    r"train=(?P<train_loss>[-+0-9.eE]+) "
    r"val=(?P<val_loss>[-+0-9.eE]+) "
    r"test=(?P<test_loss>[-+0-9.eE]+) "
    r"sp2013=(?P<sp2013_acc>[-+0-9.eE]+) "
    r"id_val=(?P<id_val_acc>[-+0-9.eE]+)$"
)


@dataclass
class ParsedLog:
    path: Path
    output_dir: str
    meta: dict[str, Any] = field(default_factory=dict)
    resumes: list[dict[str, int]] = field(default_factory=list)
    train_points: list[dict[str, Any]] = field(default_factory=list)
    eval_points: list[dict[str, Any]] = field(default_factory=list)


def parse_int(text: str) -> int:
    return int(text.replace(",", ""))


def parse_float(text: str) -> float:
    return float(text)


def parse_log(path: Path) -> ParsedLog:
    meta: dict[str, Any] = {
        "log_path": str(path),
        "rows_train": None,
        "rows_val": None,
        "rows_test": None,
        "global_batch": None,
        "updates_per_epoch": None,
        "total_updates": None,
        "warmup_updates": None,
        "model": None,
        "data": None,
        "prompt_mode": None,
        "move_sp2013_rows_to_val": None,
    }
    output_dir = ""
    resumes: list[dict[str, int]] = []
    train_points: list[dict[str, Any]] = []
    eval_points: list[dict[str, Any]] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = OUTPUT_DIR_RE.match(line)
        if m:
            output_dir = m.group("value")
            continue

        m = MODEL_RE.match(line)
        if m:
            meta["model"] = m.group("value")
            continue

        m = DATA_RE.match(line)
        if m:
            meta["data"] = m.group("value")
            continue

        m = ROWS_RE.match(line)
        if m:
            meta[f"rows_{m.group('split')}"] = parse_int(m.group("value"))
            continue

        m = PROMPT_MODE_RE.match(line)
        if m:
            meta["prompt_mode"] = m.group("value")
            continue

        m = MOVE_SP2013_RE.match(line)
        if m:
            meta["move_sp2013_rows_to_val"] = m.group("value")
            continue

        matched_int_field = False
        for key, pattern in INT_FIELD_PATTERNS.items():
            m = pattern.match(line)
            if m:
                meta[key] = parse_int(m.group("value"))
                matched_int_field = True
                break
        if matched_int_field:
            continue

        m = RESUME_RE.match(line)
        if m:
            resumes.append({k: int(v) for k, v in m.groupdict().items()})
            continue

        m = TRAIN_RE.match(line)
        if m:
            train_points.append(
                {
                    "epoch": int(m.group("epoch")),
                    "epochs_total": int(m.group("epochs")),
                    "global_update": int(m.group("global_update")),
                    "total_updates": int(m.group("total_updates")),
                    "train_loss": parse_float(m.group("train_loss")),
                    "lr": parse_float(m.group("lr")),
                    "log_path": str(path),
                }
            )
            continue

        m = EVAL_RE.match(line)
        if m:
            eval_points.append(
                {
                    "epoch": int(m.group("epoch")),
                    "update_in_epoch": int(m.group("update_in_epoch")),
                    "train_loss": parse_float(m.group("train_loss")),
                    "val_loss": parse_float(m.group("val_loss")),
                    "test_loss": parse_float(m.group("test_loss")),
                    "sp2013_acc": parse_float(m.group("sp2013_acc")),
                    "id_val_acc": parse_float(m.group("id_val_acc")),
                    "log_path": str(path),
                }
            )

    if not output_dir:
        output_dir = str(path)

    updates_per_epoch = meta.get("updates_per_epoch")
    for row in eval_points:
        if updates_per_epoch:
            row["global_update"] = (row["epoch"] - 1) * updates_per_epoch + row["update_in_epoch"]
        else:
            row["global_update"] = None

    return ParsedLog(
        path=path,
        output_dir=output_dir,
        meta=meta,
        resumes=resumes,
        train_points=train_points,
        eval_points=eval_points,
    )


def first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def dedupe_by_global_update(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[int, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.get("global_update") or -1, item.get("log_path", ""))):
        global_update = row.get("global_update")
        if global_update is None:
            ordered.append(row)
            continue
        deduped[global_update] = row
    for global_update in sorted(deduped):
        ordered.append(deduped[global_update])
    return ordered


def combine_group(parsed_logs: list[ParsedLog]) -> dict[str, Any]:
    sorted_logs = sorted(
        parsed_logs,
        key=lambda item: min(
            [row["global_update"] for row in item.train_points if row.get("global_update") is not None]
            + [row["global_update"] for row in item.eval_points if row.get("global_update") is not None]
            + [resume["global_update"] for resume in item.resumes]
            + [0]
        ),
    )

    meta_keys = [
        "model",
        "data",
        "rows_train",
        "rows_val",
        "rows_test",
        "global_batch",
        "updates_per_epoch",
        "total_updates",
        "warmup_updates",
        "prompt_mode",
        "move_sp2013_rows_to_val",
    ]
    meta = {key: first_non_null([item.meta.get(key) for item in sorted_logs]) for key in meta_keys}
    train_points = dedupe_by_global_update([row for item in sorted_logs for row in item.train_points])
    eval_points = dedupe_by_global_update([row for item in sorted_logs for row in item.eval_points])
    resumes = [resume for item in sorted_logs for resume in item.resumes]

    last_eval = eval_points[-1] if eval_points else None
    final_eval_train = last_eval["train_loss"] if last_eval else math.nan
    plateau_update = None
    plateau_epoch = None
    if eval_points and math.isfinite(final_eval_train):
        for row in eval_points:
            if abs(row["train_loss"] - final_eval_train) <= 1e-4:
                plateau_update = row.get("global_update")
                plateau_epoch = row.get("epoch")
                break

    best_sp = None
    best_val = None
    if eval_points:
        best_sp = max(eval_points, key=lambda row: (row["sp2013_acc"], -(row["val_loss"])))
        best_val = min(eval_points, key=lambda row: row["val_loss"])

    return {
        "output_dir": sorted_logs[0].output_dir,
        "log_paths": [str(item.path) for item in sorted_logs],
        "meta": meta,
        "resumes": resumes,
        "train_points": train_points,
        "eval_points": eval_points,
        "summary": {
            "n_logs": len(sorted_logs),
            "n_train_points": len(train_points),
            "n_eval_points": len(eval_points),
            "first_train_point": train_points[0] if train_points else None,
            "last_train_point": train_points[-1] if train_points else None,
            "first_eval_point": eval_points[0] if eval_points else None,
            "last_eval_point": last_eval,
            "best_sp2013_point": best_sp,
            "best_val_point": best_val,
            "first_near_final_eval_update": plateau_update,
            "first_near_final_eval_epoch": plateau_epoch,
        },
    }


def write_eval_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        output_dir = run["output_dir"]
        meta = run["meta"]
        for row in run["eval_points"]:
            rows.append(
                {
                    "output_dir": output_dir,
                    "model": meta.get("model"),
                    "prompt_mode": meta.get("prompt_mode"),
                    "rows_train": meta.get("rows_train"),
                    "rows_val": meta.get("rows_val"),
                    "rows_test": meta.get("rows_test"),
                    "global_batch": meta.get("global_batch"),
                    "updates_per_epoch": meta.get("updates_per_epoch"),
                    **row,
                }
            )

    fieldnames = [
        "output_dir",
        "model",
        "prompt_mode",
        "rows_train",
        "rows_val",
        "rows_test",
        "global_batch",
        "updates_per_epoch",
        "epoch",
        "update_in_epoch",
        "global_update",
        "train_loss",
        "val_loss",
        "test_loss",
        "sp2013_acc",
        "id_val_acc",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_run(run: dict[str, Any]) -> None:
    summary = run["summary"]
    meta = run["meta"]
    print(f"Run: {run['output_dir']}")
    print(f"  logs: {len(run['log_paths'])}")
    for path in run["log_paths"]:
        print(f"    - {path}")
    print(f"  model: {meta.get('model')}")
    print(f"  prompt_mode: {meta.get('prompt_mode')}")
    print(f"  rows train/val/test: {meta.get('rows_train')} / {meta.get('rows_val')} / {meta.get('rows_test')}")
    print(
        "  global_batch / updates_per_epoch / total_updates: "
        f"{meta.get('global_batch')} / {meta.get('updates_per_epoch')} / {meta.get('total_updates')}"
    )
    if meta.get("move_sp2013_rows_to_val"):
        print(f"  move_sp2013_rows_to_val: {meta['move_sp2013_rows_to_val']}")
    if summary["first_train_point"] and summary["last_train_point"]:
        first = summary["first_train_point"]
        last = summary["last_train_point"]
        print(
            "  train log span: "
            f"update {first['global_update']} loss={first['train_loss']:.4f} -> "
            f"update {last['global_update']} loss={last['train_loss']:.4f}"
        )
    if summary["first_eval_point"] and summary["last_eval_point"]:
        first = summary["first_eval_point"]
        last = summary["last_eval_point"]
        print(
            "  eval span: "
            f"update {first['global_update']} train={first['train_loss']:.4f} val={first['val_loss']:.4f} "
            f"sp2013={first['sp2013_acc']:.4f}"
        )
        print(
            "             "
            f"update {last['global_update']} train={last['train_loss']:.4f} val={last['val_loss']:.4f} "
            f"sp2013={last['sp2013_acc']:.4f}"
        )
    if summary["best_sp2013_point"]:
        best_sp = summary["best_sp2013_point"]
        print(
            "  best SP2013: "
            f"{best_sp['sp2013_acc']:.4f} at update {best_sp['global_update']} "
            f"(epoch {best_sp['epoch']} u{best_sp['update_in_epoch']})"
        )
    if summary["best_val_point"]:
        best_val = summary["best_val_point"]
        print(
            "  best val loss: "
            f"{best_val['val_loss']:.4f} at update {best_val['global_update']} "
            f"(epoch {best_val['epoch']} u{best_val['update_in_epoch']})"
        )
    if summary["first_near_final_eval_update"] is not None:
        print(
            "  first eval within 1e-4 of final train loss: "
            f"update {summary['first_near_final_eval_update']} "
            f"(epoch {summary['first_near_final_eval_epoch']})"
        )
    if run["resumes"]:
        print("  resume points:")
        for resume in run["resumes"]:
            print(
                "    - "
                f"epoch={resume['epoch']} update_in_epoch={resume['update_in_epoch']} "
                f"global_update={resume['global_update']}"
            )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Trainer .out logs to parse")
    parser.add_argument("--eval-csv", type=Path, default=None, help="Optional path for combined eval CSV output")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path for combined summary JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_logs = [parse_log(Path(log_path)) for log_path in args.logs]

    groups: dict[str, list[ParsedLog]] = defaultdict(list)
    for item in parsed_logs:
        groups[item.output_dir].append(item)

    runs = [combine_group(items) for _, items in sorted(groups.items(), key=lambda kv: kv[0])]
    for run in runs:
        print_run(run)

    if args.eval_csv is not None:
        args.eval_csv.parent.mkdir(parents=True, exist_ok=True)
        write_eval_csv(args.eval_csv, runs)
        print(f"Wrote eval CSV -> {args.eval_csv}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "output_dir": run["output_dir"],
                "log_paths": run["log_paths"],
                "meta": run["meta"],
                "resumes": run["resumes"],
                "summary": run["summary"],
            }
            for run in runs
        ]
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote summary JSON -> {args.summary_json}")


if __name__ == "__main__":
    main()
