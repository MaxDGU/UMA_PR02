#!/usr/bin/env python3
"""Shared helpers for BBT experiment subpipelines."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBPIPELINE_ORDER = [
    "uma_traces",
    "translation",
    "distill_train",
    "distill_eval",
    "human_finetune",
    "human_eval",
]

SUBPIPELINE_DIRS = {
    "uma_traces": "01_uma_traces",
    "translation": "02_translation",
    "distill_train": "03_distill_train",
    "distill_eval": "04_distill_eval",
    "human_finetune": "05_human_finetune",
    "human_eval": "06_human_eval",
}

EXTERNAL_INPUT_KEYS = {
    "translation": "uma_trace_csv",
    "distill_train": "translated_csv",
    "distill_eval": "distill_run_dir",
    "human_finetune": "distill_run_dir",
    "human_eval": "human_finetune_run_dir",
}


@dataclass
class SubpipelineResult:
    manifest: Dict[str, Any]
    manifest_path: Path
    primary_output: Path


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_dict(obj: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        value = {}
        obj[key] = value
    return value


def parse_override_value(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError("Empty override key.")
    cursor: Dict[str, Any] = cfg
    for key in keys[:-1]:
        node = cursor.get(key)
        if not isinstance(node, dict):
            node = {}
            cursor[key] = node
        cursor = node
    cursor[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        set_nested(out, key.strip(), parse_override_value(value))
    return out


def resolve_path(path_value: Any, run_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    rendered = str(path_value).replace("{run_dir}", str(run_dir))
    candidate = Path(rendered)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def resolve_output_dir(config: Dict[str, Any], subpipeline_name: str, run_dir: Path) -> Path:
    sub_cfg = ensure_dict(ensure_dict(config, "subpipelines"), subpipeline_name)
    default = f"{{run_dir}}/{SUBPIPELINE_DIRS[subpipeline_name]}"
    return resolve_path(sub_cfg.get("output_dir", default), run_dir)


def load_config(config_path: Path, overrides: Optional[list[str]] = None) -> Dict[str, Any]:
    cfg = read_json(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_command(cmd: list[str], dry_run: bool, cwd: Path = REPO_ROOT) -> None:
    print(command_text(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def add_bool_optional(cmd: list[str], flag: str, enabled: bool) -> None:
    if not flag.startswith("--"):
        raise ValueError(f"Flag must start with '--': {flag}")
    cmd.append(flag if enabled else f"--no-{flag[2:]}")


def add_store_true(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def load_upstream_primary_from_manifest(manifest_path: Path) -> Path:
    payload = read_json(manifest_path)
    primary = str(payload.get("primary_output", "")).strip()
    if primary == "":
        raise ValueError(f"Upstream manifest missing primary_output: {manifest_path}")
    return Path(primary).resolve()


def resolve_upstream_input(
    config: Dict[str, Any],
    subpipeline_name: str,
    run_dir: Path,
    upstream_manifest: Optional[Path],
    upstream_primary_output: Optional[Path],
) -> Optional[Path]:
    if upstream_primary_output is not None:
        return upstream_primary_output.resolve()

    if upstream_manifest is not None:
        return load_upstream_primary_from_manifest(upstream_manifest.resolve())

    key = EXTERNAL_INPUT_KEYS.get(subpipeline_name)
    if key is None:
        return None

    ext_cfg = ensure_dict(config, "external_inputs")
    raw = str(ext_cfg.get(key, "")).strip()
    if raw == "":
        raise ValueError(
            f"Subpipeline '{subpipeline_name}' requires upstream input but none was provided. "
            f"Set external_inputs.{key} or provide upstream manifest."
        )
    return resolve_path(raw, run_dir)


def build_manifest(
    *,
    subpipeline_name: str,
    status: str,
    started_at_utc: str,
    finished_at_utc: str,
    config_snapshot: Dict[str, Any],
    inputs: Dict[str, str],
    primary_output: Path,
    secondary_outputs: list[str],
    metrics_summary: Optional[Dict[str, Any]],
    upstream_manifest: Optional[Path],
) -> Dict[str, Any]:
    return {
        "subpipeline_name": subpipeline_name,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "config_snapshot": config_snapshot,
        "inputs": inputs,
        "primary_output": str(primary_output.resolve()),
        "secondary_outputs": secondary_outputs,
        "metrics_summary": metrics_summary or {},
        "upstream_manifest": str(upstream_manifest.resolve()) if upstream_manifest is not None else None,
    }


def maybe_write_manifest(manifest: Dict[str, Any], manifest_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    write_json(manifest_path, manifest)
