#!/usr/bin/env python3
"""
Build paired translated datasets for a panel of UMA parameter sets from one saved seed corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from build_fixed_tuple_trace_surface_datasets import (
    ensure_parent,
    filter_out_sp2013_rows,
    load_normalized_problem_set,
    run_translate,
    summarize_raw_trace_health,
    verify_matching_row_sets,
)
from train_transformer_hf import normalize_problem_key


TUPLE_COLUMNS = ["g", "d", "rt_mu", "ice"]
PANEL_COLUMNS = ["subjid", *TUPLE_COLUMNS]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    uma_dir = results_dir / "UMA_replication"
    tr_dir = results_dir / "transformer_replication"

    parser = argparse.ArgumentParser(description="Build translated datasets for a panel of UMA parameter sets from one seed.")
    parser.add_argument(
        "--panel_csv",
        type=Path,
        default=uma_dir / "representative_25_params" / "even_g_rt5_ice50.csv",
        help="CSV containing one row per selected subjid / parameter tuple.",
    )
    parser.add_argument(
        "--raw_seed_csv",
        type=Path,
        default=uma_dir / "synth_unique_seed1_all1000" / "uma_traces_synth_unique_seed1_all1000.csv",
        help="Raw UMA trace CSV for one seed across all 1000 models.",
    )
    parser.add_argument(
        "--subject_accuracy_csv",
        type=Path,
        default=uma_dir / "synth_unique_seed1_all1000" / "accuracy_by_subject.csv",
        help="Per-subjid summary used for fast fixed-block extraction when available.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expected_panel_size", type=int, default=25)
    parser.add_argument("--expected_g_values", type=str, default="0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--expected_d_values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--expected_rt_mu", type=float, default=5.0)
    parser.add_argument("--expected_ice", type=float, default=50.0)
    parser.add_argument(
        "--merged_csv",
        type=Path,
        default=uma_dir / "panel_seed_traces" / "synth_unique_panel25_even_g_rt5_ice50_seed1_merged.csv.gz",
        help="Output pooled raw trace CSV.gz for the selected panel.",
    )
    parser.add_argument(
        "--default_csv",
        type=Path,
        default=tr_dir / "synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_default_mincols.csv.gz",
        help="Default humanlike translated CSV.",
    )
    parser.add_argument(
        "--surface_csv",
        type=Path,
        default=tr_dir / "synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_surface_hidden_mincols.csv.gz",
        help="Surface-hidden translated CSV.",
    )
    parser.add_argument(
        "--default_train_csv",
        type=Path,
        default=tr_dir / "synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_default_nosp2013_mincols.csv.gz",
        help="Default translated CSV with SP2013 rows and literal ? answers removed.",
    )
    parser.add_argument(
        "--surface_train_csv",
        type=Path,
        default=tr_dir / "synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_surface_hidden_nosp2013_mincols.csv.gz",
        help="Surface-hidden translated CSV with SP2013 rows and literal ? answers removed.",
    )
    parser.add_argument(
        "--sp2013_csv",
        type=Path,
        default=uma_dir / "sp2013.csv",
        help="SP2013 problem CSV used for exclusion.",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=tr_dir / "panel25_even_g_rt5_ice50_seed1_trace_surface_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="trace_or_child",
        help="Translator reasoning mode.",
    )
    parser.add_argument(
        "--extract_mode",
        type=str,
        choices=["auto", "fast", "grep", "chunk"],
        default="auto",
        help="How to subset the large raw seed CSV. 'auto' tries fast block extraction first, then falls back to grep.",
    )
    return parser.parse_args()


def parse_float_set(text: str) -> set[float]:
    tokens = [tok.strip() for tok in str(text).split(",")]
    values = {float(tok) for tok in tokens if tok != ""}
    if not values:
        raise ValueError(f"Expected at least one numeric value in: {text!r}")
    return values


def normalize_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in PANEL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["subjid"] = out["subjid"].astype(int)
    return out.sort_values("subjid").reset_index(drop=True)


def load_panel(panel_csv: Path, expected_panel_size: int) -> pd.DataFrame:
    panel = pd.read_csv(panel_csv, usecols=PANEL_COLUMNS).drop_duplicates()
    if len(panel) != expected_panel_size:
        raise ValueError(f"Expected {expected_panel_size} panel rows in {panel_csv}, found {len(panel)}.")
    panel = normalize_numeric_frame(panel)
    if panel["subjid"].duplicated().any():
        dupes = panel.loc[panel["subjid"].duplicated(), "subjid"].tolist()
        raise ValueError(f"Panel CSV contains duplicate subjid entries: {dupes[:10]}")
    return panel


def load_panel_subset(raw_seed_csv: Path, subjids: Sequence[int], seed: int) -> pd.DataFrame:
    subjids_sorted = sorted(int(x) for x in subjids)
    subjids_set = set(int(x) for x in subjids)
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_seed_csv, chunksize=100_000):
        keep = chunk["subjid"].astype(int).isin(subjids_set)
        if keep.any():
            out = chunk.loc[keep].copy()
            out["seed"] = int(seed)
            frames.append(out)
    if not frames:
        raise ValueError(f"No rows matched panel subjids in raw seed CSV: {raw_seed_csv}")
    subset = pd.concat(frames, ignore_index=True)
    return subset.sort_values(["subjid", "prob"]).reset_index(drop=True)


def load_panel_subset_grep(raw_seed_csv: Path, subjids: Sequence[int], seed: int) -> pd.DataFrame:
    subjids_sorted = sorted(int(x) for x in subjids)
    pattern = "^(" + "|".join(str(x) for x in subjids_sorted) + "),"

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="panel_seed_subset_", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with tmp_path.open("w", encoding="utf-8") as out:
                subprocess.run(
                    ["head", "-n", "1", str(raw_seed_csv)],
                    stdout=out,
                    check=True,
                    env={**os.environ, "LC_ALL": "C"},
                )
                subprocess.run(
                    ["grep", "-E", pattern, str(raw_seed_csv)],
                    stdout=out,
                    check=True,
                    env={**os.environ, "LC_ALL": "C"},
                )
        subset = pd.read_csv(tmp_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    if subset.empty:
        raise ValueError(f"No rows matched panel subjids in raw seed CSV via grep: {raw_seed_csv}")

    subset["seed"] = int(seed)
    return subset.sort_values(["subjid", "prob"]).reset_index(drop=True)


def load_panel_subset_fast(
    raw_seed_csv: Path,
    subject_accuracy_csv: Path,
    subjids: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    counts = pd.read_csv(subject_accuracy_csv, usecols=["subjid", "n"])
    counts["subjid"] = pd.to_numeric(counts["subjid"], errors="raise").astype(int)
    counts["n"] = pd.to_numeric(counts["n"], errors="raise").astype(int)
    if counts["n"].nunique() != 1:
        raise ValueError(f"{subject_accuracy_csv} does not describe a fixed-block layout.")

    block_rows = int(counts["n"].iloc[0])
    missing = sorted(set(int(x) for x in subjids) - set(counts["subjid"].tolist()))
    if missing:
        raise ValueError(f"Missing selected subjids in {subject_accuracy_csv}: {missing[:10]}")

    mapping = build_lexicographic_two_subjid_mapping(len(counts))
    missing_from_mapping = sorted(set(int(x) for x in subjids) - set(mapping.keys()))
    if missing_from_mapping:
        raise ValueError(f"Could not locate selected subjids in raw block mapping: {missing_from_mapping[:10]}")

    sed_cmd = ["sed", "-n", "-e", "1p"]
    for subjid in sorted(int(x) for x in subjids):
        block_idx = mapping[subjid]
        start = 2 + block_idx * block_rows
        end = 1 + (block_idx + 1) * block_rows
        sed_cmd.extend(["-e", f"{start},{end}p"])
    sed_cmd.append(str(raw_seed_csv))

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="panel_seed_subset_", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            subprocess.run(sed_cmd, stdout=tmp, check=True)
        subset = pd.read_csv(tmp_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    subset["seed"] = int(seed)
    return subset.sort_values(["subjid", "prob"]).reset_index(drop=True)


def build_lexicographic_two_subjid_mapping(n_blocks: int) -> Dict[int, int]:
    if n_blocks % 2 != 0:
        raise ValueError(f"Expected an even number of subject blocks, found {n_blocks}.")

    shard_starts = list(range(0, n_blocks, 2))
    shard_starts = sorted(shard_starts, key=lambda start: f"synth_unique_seed1_{start}_to_{start + 1}.csv")

    mapping: Dict[int, int] = {}
    for shard_idx, start in enumerate(shard_starts):
        mapping[start] = 2 * shard_idx
        mapping[start + 1] = 2 * shard_idx + 1
    return mapping


def verify_panel_subset(
    subset: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    expected_g_values: set[float],
    expected_d_values: set[float],
    expected_rt_mu: float,
    expected_ice: float,
    seed: int,
) -> Dict[str, object]:
    actual_subjids = sorted(pd.to_numeric(subset["subjid"], errors="raise").astype(int).unique().tolist())
    expected_subjids = panel["subjid"].astype(int).tolist()
    if actual_subjids != expected_subjids:
        raise ValueError(
            "Raw subset subjid mismatch. "
            f"expected={expected_subjids[:10]}... actual={actual_subjids[:10]}..."
        )

    if "seed" not in subset.columns:
        raise ValueError("Raw subset must include a seed column.")
    actual_seed_values = sorted(pd.to_numeric(subset["seed"], errors="raise").astype(int).unique().tolist())
    if actual_seed_values != [int(seed)]:
        raise ValueError(f"Expected exactly seed={seed} in subset, found {actual_seed_values}.")

    actual_panel = normalize_numeric_frame(subset.loc[:, PANEL_COLUMNS].drop_duplicates())
    if not actual_panel.equals(panel):
        mismatch = pd.concat(
            [
                panel.assign(_side="expected"),
                actual_panel.assign(_side="actual"),
            ],
            ignore_index=True,
        ).drop_duplicates(keep=False)
        raise ValueError(f"Raw subset parameter tuples do not match panel CSV:\n{mismatch.head(20)}")

    g_values = {float(x) for x in actual_panel["g"].tolist()}
    d_values = {float(x) for x in actual_panel["d"].tolist()}
    rt_mu_values = {float(x) for x in actual_panel["rt_mu"].tolist()}
    ice_values = {float(x) for x in actual_panel["ice"].tolist()}

    if g_values != expected_g_values:
        raise ValueError(f"Expected g values {sorted(expected_g_values)}, found {sorted(g_values)}")
    if d_values != expected_d_values:
        raise ValueError(f"Expected d values {sorted(expected_d_values)}, found {sorted(d_values)}")
    if rt_mu_values != {expected_rt_mu}:
        raise ValueError(f"Expected rt_mu={expected_rt_mu}, found {sorted(rt_mu_values)}")
    if ice_values != {expected_ice}:
        raise ValueError(f"Expected ice={expected_ice}, found {sorted(ice_values)}")

    return {
        "rows": int(len(subset)),
        "unique_subjids": int(len(actual_subjids)),
        "unique_parameter_tuples": int(len(actual_panel)),
        "seed_values": actual_seed_values,
        "g_values": sorted(g_values),
        "d_values": sorted(d_values),
        "rt_mu_values": sorted(rt_mu_values),
        "ice_values": sorted(ice_values),
        "unique_probs": int(subset["prob"].astype(str).nunique()),
    }


def summarize_clean_output(csv_path: Path, sp2013_keys: set[str]) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    missing_columns = [col for col in ["subjid", "g", "d", "rt_mu", "ice", "response_nl"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")

    prob_keys = df["prob"].astype(str).map(normalize_problem_key)
    literal_question_answer = df["answer"].astype(str).str.strip().eq("?")
    response_question_answer = df["response_nl"].astype(str).str.contains(r"### answer:\s*\?", regex=True)

    return {
        "rows": int(len(df)),
        "unique_probs": int(df["prob"].astype(str).nunique()),
        "sp2013_overlap_rows": int(prob_keys.isin(sp2013_keys).sum()),
        "literal_question_answer_rows": int(literal_question_answer.sum()),
        "response_question_answer_rows": int(response_question_answer.sum()),
        "parameter_tuple_count": int(df.loc[:, TUPLE_COLUMNS].drop_duplicates().shape[0]),
    }


def main() -> None:
    args = parse_args()

    expected_g_values = parse_float_set(args.expected_g_values)
    expected_d_values = parse_float_set(args.expected_d_values)
    panel = load_panel(args.panel_csv, args.expected_panel_size)

    expected_kwargs = dict(
        expected_g_values=expected_g_values,
        expected_d_values=expected_d_values,
        expected_rt_mu=float(args.expected_rt_mu),
        expected_ice=float(args.expected_ice),
        seed=args.seed,
    )

    subjids = panel["subjid"].tolist()
    extract_error: Exception | None = None

    if args.extract_mode == "fast":
        merged = load_panel_subset_fast(args.raw_seed_csv, args.subject_accuracy_csv, subjids, args.seed)
    elif args.extract_mode == "grep":
        merged = load_panel_subset_grep(args.raw_seed_csv, subjids, args.seed)
    elif args.extract_mode == "chunk":
        merged = load_panel_subset(args.raw_seed_csv, subjids, args.seed)
    else:
        merged = None
        if args.subject_accuracy_csv.exists():
            try:
                candidate = load_panel_subset_fast(args.raw_seed_csv, args.subject_accuracy_csv, subjids, args.seed)
                raw_panel_stats = verify_panel_subset(candidate, panel, **expected_kwargs)
                merged = candidate
            except Exception as exc:  # pragma: no cover - fallback path for corrupted block-order assumptions
                extract_error = exc
        if merged is None:
            if extract_error is not None:
                print(
                    "Fast panel extraction failed verification; falling back to grep scan of the raw seed CSV.\n"
                    f"Reason: {extract_error}"
                )
            merged = load_panel_subset_grep(args.raw_seed_csv, subjids, args.seed)
            raw_panel_stats = verify_panel_subset(merged, panel, **expected_kwargs)

    if args.extract_mode != "auto":
        raw_panel_stats = verify_panel_subset(merged, panel, **expected_kwargs)
    raw_trace_health = summarize_raw_trace_health(merged)

    ensure_parent(args.merged_csv)
    merged.to_csv(args.merged_csv, index=False, compression="gzip")

    run_translate(args.merged_csv, args.default_csv, surface_hidden_trace_steps=False, reasoning_mode=args.reasoning_mode)
    run_translate(args.merged_csv, args.surface_csv, surface_hidden_trace_steps=True, reasoning_mode=args.reasoning_mode)

    translation_rowset_stats = verify_matching_row_sets(args.default_csv, args.surface_csv)

    sp2013_keys = load_normalized_problem_set(args.sp2013_csv)
    default_filter_stats = filter_out_sp2013_rows(args.default_csv, args.default_train_csv, sp2013_keys)
    surface_filter_stats = filter_out_sp2013_rows(args.surface_csv, args.surface_train_csv, sp2013_keys)
    clean_rowset_stats = verify_matching_row_sets(args.default_train_csv, args.surface_train_csv)

    default_clean_checks = summarize_clean_output(args.default_train_csv, sp2013_keys)
    surface_clean_checks = summarize_clean_output(args.surface_train_csv, sp2013_keys)

    summary = {
        "seed": int(args.seed),
        "panel_csv": str(args.panel_csv),
        "raw_seed_csv": str(args.raw_seed_csv),
        "subject_accuracy_csv": str(args.subject_accuracy_csv),
        "expected_panel_size": int(args.expected_panel_size),
        "expected_g_values": sorted(expected_g_values),
        "expected_d_values": sorted(expected_d_values),
        "expected_rt_mu": float(args.expected_rt_mu),
        "expected_ice": float(args.expected_ice),
        "panel_subjids": panel["subjid"].astype(int).tolist(),
        "raw_panel_verification": raw_panel_stats,
        "raw_trace_health": raw_trace_health,
        "translation_rowset_verification": translation_rowset_stats,
        "clean_rowset_verification": clean_rowset_stats,
        "default_filter": default_filter_stats,
        "surface_filter": surface_filter_stats,
        "default_clean_checks": default_clean_checks,
        "surface_clean_checks": surface_clean_checks,
        "paths": {
            "merged_csv": str(args.merged_csv),
            "default_csv": str(args.default_csv),
            "surface_csv": str(args.surface_csv),
            "default_train_csv": str(args.default_train_csv),
            "surface_train_csv": str(args.surface_train_csv),
            "summary_json": str(args.summary_json),
        },
    }

    ensure_parent(args.summary_json)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
