#!/usr/bin/env python3
"""
Build the fixed-tuple trace-surface experiment datasets for subjid 472.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from train_transformer_hf import build_student_prompt, normalize_problem_key, resolve_model_source


TRACE_REQUIRED_COLUMNS = ["trace_rules_full", "trace_step_count", "trace_steps_json"]
TUPLE_COLUMNS = ["g", "d", "rt_mu", "ice"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    uma_dir = results_dir / "UMA_replication"
    tr_dir = results_dir / "transformer_replication"
    parser = argparse.ArgumentParser(description="Build fixed-tuple trace-surface datasets.")
    parser.add_argument("--subjid", type=int, default=472)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=20)
    parser.add_argument(
        "--trace_dir",
        type=Path,
        default=uma_dir / "synth_unique_seed_traces" / "subjid_0472",
        help="Directory containing one raw UMA trace CSV per seed.",
    )
    parser.add_argument(
        "--merged_csv",
        type=Path,
        default=uma_dir / "synth_unique_seed_traces" / "subjid_0472" / "synth_unique_subjid_0472_seeds1_20_merged.csv.gz",
        help="Merged raw trace CSV path.",
    )
    parser.add_argument(
        "--default_csv",
        type=Path,
        default=tr_dir / "synth_unique_subjid_0472_seeds1_20_tracev10_default_mincols.csv.gz",
        help="Default humanlike translated CSV.",
    )
    parser.add_argument(
        "--surface_csv",
        type=Path,
        default=tr_dir / "synth_unique_subjid_0472_seeds1_20_tracev10_surface_hidden_mincols.csv.gz",
        help="Surface-hidden-steps translated CSV.",
    )
    parser.add_argument(
        "--default_train_csv",
        type=Path,
        default=tr_dir / "synth_unique_subjid_0472_seeds1_20_tracev10_default_nosp2013_mincols.csv.gz",
        help="Default translated CSV with SP2013 problems removed.",
    )
    parser.add_argument(
        "--surface_train_csv",
        type=Path,
        default=tr_dir / "synth_unique_subjid_0472_seeds1_20_tracev10_surface_hidden_nosp2013_mincols.csv.gz",
        help="Surface-hidden translated CSV with SP2013 problems removed.",
    )
    parser.add_argument(
        "--sp2013_csv",
        type=Path,
        default=uma_dir / "sp2013.csv",
        help="SP2013 problem CSV used for exclusion.",
    )
    parser.add_argument(
        "--context_json",
        type=Path,
        default=tr_dir / "fixed_tuple_trace_surface_context_subjid0472.json",
        help="Output JSON with shared max_length and max_new_tokens.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Tokenizer source used for context-length preflight.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve model/tokenizer from local HF cache only.",
    )
    parser.add_argument("--prompt_col", type=str, default="instruction_nl")
    parser.add_argument("--response_col", type=str, default="response_nl")
    parser.add_argument("--max_length_min", type=int, default=256)
    parser.add_argument("--max_length_cap", type=int, default=384)
    parser.add_argument("--max_new_tokens_min", type=int, default=192)
    parser.add_argument("--max_new_tokens_cap", type=int, default=320)
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="trace_or_child",
        help="Translator reasoning mode.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def seed_paths(trace_dir: Path, subjid: int, seed_start: int, seed_end: int) -> List[Tuple[int, Path]]:
    subjid_padded = f"{subjid:04d}"
    out: List[Tuple[int, Path]] = []
    for seed in range(seed_start, seed_end + 1):
        path = trace_dir / f"synth_unique_subjid_{subjid_padded}_seed_{seed}.csv"
        out.append((seed, path))
    return out


def load_seed_frame(seed: int, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed trace CSV: {path}")
    header = pd.read_csv(path, nrows=0)
    missing = [col for col in TRACE_REQUIRED_COLUMNS if col not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing trace columns: {missing}")
    df = pd.read_csv(path)
    df["seed"] = int(seed)
    return df


def verify_fixed_tuple(frame: pd.DataFrame) -> Dict[str, float]:
    tuple_df = frame.loc[:, TUPLE_COLUMNS].drop_duplicates().reset_index(drop=True)
    if len(tuple_df) != 1:
        raise ValueError(f"Expected exactly one parameter tuple, found {len(tuple_df)}: {tuple_df.to_dict(orient='records')[:5]}")
    return {col: float(tuple_df.iloc[0][col]) for col in TUPLE_COLUMNS}


def merge_seed_traces(paths: Sequence[Tuple[int, Path]], merged_csv: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frames = [load_seed_frame(seed, path) for seed, path in paths]
    merged = pd.concat(frames, ignore_index=True)
    tuple_values = verify_fixed_tuple(merged)
    ensure_parent(merged_csv)
    merged.to_csv(merged_csv, index=False, compression="gzip")
    return merged, tuple_values


def summarize_raw_trace_health(frame: pd.DataFrame) -> Dict[str, int]:
    answer_series = frame.get("answer", pd.Series(["?"] * len(frame), index=frame.index)).astype(str).str.strip()
    strategy_series = frame.get("strategy", pd.Series(["OTHER"] * len(frame), index=frame.index)).astype(str).str.strip()
    if "trace_step_count" in frame.columns:
        step_counts = pd.to_numeric(frame["trace_step_count"], errors="coerce").fillna(0).astype(int)
    else:
        step_counts = pd.Series(0, index=frame.index, dtype=int)

    unknown_answer = answer_series.isin({"", "?", "None", "nan", "NaN"})
    other_strategy = strategy_series.eq("OTHER")
    single_step = step_counts.le(1)
    degenerate = unknown_answer & other_strategy & single_step

    summary = {
        "rows": int(len(frame)),
        "unknown_answer_rows": int(unknown_answer.sum()),
        "other_strategy_rows": int(other_strategy.sum()),
        "single_step_rows": int(single_step.sum()),
        "degenerate_rows": int(degenerate.sum()),
    }
    if summary["rows"] > 0 and summary["degenerate_rows"] == summary["rows"]:
        raise ValueError(
            "Merged raw UMA traces are fully degenerate "
            f"(all rows are single-step OTHER / answer=?). Summary: {summary}"
        )
    return summary


def run_translate(input_csv: Path, output_csv: Path, *, surface_hidden_trace_steps: bool, reasoning_mode: str) -> None:
    ensure_parent(output_csv)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "translate_uma_traces_to_nlp.py"),
        "--no-include-default-uma-traces",
        "--no-include-sp2013-seeds",
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--minimal-columns",
        "--student-prompt-mode",
        "none",
        "--reasoning-mode",
        reasoning_mode,
        "--row-filter",
        "none",
    ]
    if surface_hidden_trace_steps:
        cmd.append("--surface-hidden-trace-steps")
    subprocess.run(cmd, check=True)


def load_normalized_problem_set(csv_path: Path, problem_col: str = "prob") -> set[str]:
    df = pd.read_csv(csv_path, usecols=[problem_col])
    return {normalize_problem_key(prob) for prob in df[problem_col].astype(str).tolist()}


def filter_out_sp2013_rows(input_csv: Path, output_csv: Path, sp2013_keys: set[str]) -> Dict[str, int]:
    df = pd.read_csv(input_csv)
    keys = df["prob"].astype(str).map(normalize_problem_key)
    sp2013_mask = keys.isin(sp2013_keys)
    keep = ~sp2013_mask
    literal_question_answer = df["answer"].astype(str).str.strip().eq("?")
    keep &= ~literal_question_answer
    filtered = df.loc[keep].reset_index(drop=True)
    ensure_parent(output_csv)
    filtered.to_csv(output_csv, index=False, compression="gzip")
    return {
        "rows_in": int(len(df)),
        "rows_out": int(len(filtered)),
        "rows_dropped": int(len(df) - len(filtered)),
        "sp2013_rows_dropped": int(sp2013_mask.sum()),
        "literal_question_answer_rows_dropped": int(literal_question_answer.sum()),
        "unique_probs_out": int(filtered["prob"].astype(str).nunique()),
    }


def verify_matching_row_sets(default_csv: Path, surface_csv: Path) -> Dict[str, int]:
    usecols = ["source_uid", "prob", "g", "d", "rt_mu", "ice"]
    left = pd.read_csv(default_csv, usecols=usecols)
    right = pd.read_csv(surface_csv, usecols=usecols)
    if len(left) != len(right):
        raise ValueError(f"Translated row count mismatch: {len(left)} vs {len(right)}")
    if list(left.columns) != list(right.columns):
        raise ValueError("Translated schema mismatch between default and surfaced datasets.")
    if not left.equals(right):
        mismatch = int((left != right).any(axis=1).sum())
        raise ValueError(f"Translated datasets differ on non-text columns for {mismatch} rows.")
    return {
        "rows": int(len(left)),
        "unique_probs": int(left["prob"].astype(str).nunique()),
        "unique_source_uids": int(left["source_uid"].astype(str).nunique()),
    }


def round_up(value: float, step: int) -> int:
    return int(step * math.ceil(float(value) / float(step)))


def choose_length_bound(p999: float, observed_max: int, *, step: int, minimum: int, cap: int) -> int:
    chosen = max(minimum, round_up(p999 + 16.0, step))
    chosen = min(chosen, cap)
    if observed_max > chosen:
        chosen = round_up(observed_max, step)
    return int(chosen)


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collect_token_lengths(csv_paths: Sequence[Path], tokenizer: AutoTokenizer, response_col: str) -> Dict[str, object]:
    combined_lengths: List[int] = []
    response_lengths: List[int] = []
    per_file: Dict[str, Dict[str, float]] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        prompts = [
            build_student_prompt(prob=str(prob), g=float(g), d=float(d), rt_mu=float(rt), ice=float(ice))
            for prob, g, d, rt, ice in zip(
                df["prob"].tolist(),
                df["g"].tolist(),
                df["d"].tolist(),
                df["rt_mu"].tolist(),
                df["ice"].tolist(),
            )
        ]
        responses = df[response_col].astype(str).tolist()

        local_combined: List[int] = []
        local_response: List[int] = []
        for prompt_batch, response_batch in zip(batched(prompts, 256), batched(responses, 256)):
            prompt_tokens = tokenizer(list(prompt_batch), add_special_tokens=False)["input_ids"]
            response_tokens = tokenizer(list(response_batch), add_special_tokens=False)["input_ids"]
            for prompt_ids, response_ids in zip(prompt_tokens, response_tokens):
                response_len = int(len(response_ids))
                combined_len = int(len(prompt_ids) + response_len)
                local_response.append(response_len)
                local_combined.append(combined_len)

        combined_lengths.extend(local_combined)
        response_lengths.extend(local_response)
        per_file[csv_path.name] = {
            "rows": int(len(df)),
            "combined_max": int(max(local_combined)) if local_combined else 0,
            "response_max": int(max(local_response)) if local_response else 0,
            "combined_p99_9": float(np.quantile(local_combined, 0.999)) if local_combined else float("nan"),
            "response_p99_9": float(np.quantile(local_response, 0.999)) if local_response else float("nan"),
        }

    if not combined_lengths or not response_lengths:
        raise ValueError("No token lengths collected; check the filtered dataset inputs.")

    return {
        "per_file": per_file,
        "combined_lengths": combined_lengths,
        "response_lengths": response_lengths,
    }


def build_context_summary(
    csv_paths: Sequence[Path],
    tokenizer: AutoTokenizer,
    response_col: str,
    *,
    max_length_min: int,
    max_length_cap: int,
    max_new_tokens_min: int,
    max_new_tokens_cap: int,
) -> Dict[str, object]:
    raw = collect_token_lengths(csv_paths, tokenizer, response_col=response_col)
    combined = raw["combined_lengths"]
    response = raw["response_lengths"]
    combined_p999 = float(np.quantile(combined, 0.999))
    response_p999 = float(np.quantile(response, 0.999))
    combined_max = int(max(combined))
    response_max = int(max(response))

    shared_max_length = choose_length_bound(
        combined_p999,
        combined_max,
        step=32,
        minimum=max_length_min,
        cap=max_length_cap,
    )
    shared_max_new_tokens = choose_length_bound(
        response_p999,
        response_max,
        step=16,
        minimum=max_new_tokens_min,
        cap=max_new_tokens_cap,
    )
    return {
        "per_file": raw["per_file"],
        "shared_settings": {
            "max_length": int(shared_max_length),
            "max_new_tokens": int(shared_max_new_tokens),
        },
        "aggregate": {
            "rows_total": int(len(combined)),
            "prompt_plus_response_p99_9": combined_p999,
            "prompt_plus_response_max": combined_max,
            "response_p99_9": response_p999,
            "response_max": response_max,
        },
    }


def main() -> None:
    args = parse_args()

    paths = seed_paths(args.trace_dir, args.subjid, args.seed_start, args.seed_end)
    merged_df, tuple_values = merge_seed_traces(paths, args.merged_csv)
    raw_trace_health = summarize_raw_trace_health(merged_df)

    run_translate(args.merged_csv, args.default_csv, surface_hidden_trace_steps=False, reasoning_mode=args.reasoning_mode)
    run_translate(args.merged_csv, args.surface_csv, surface_hidden_trace_steps=True, reasoning_mode=args.reasoning_mode)

    rowset_stats = verify_matching_row_sets(args.default_csv, args.surface_csv)

    sp2013_keys = load_normalized_problem_set(args.sp2013_csv)
    default_filter_stats = filter_out_sp2013_rows(args.default_csv, args.default_train_csv, sp2013_keys)
    surface_filter_stats = filter_out_sp2013_rows(args.surface_csv, args.surface_train_csv, sp2013_keys)
    if default_filter_stats["rows_out"] == 0 or surface_filter_stats["rows_out"] == 0:
        raise ValueError(
            "SP2013 exclusion removed all rows from at least one translated dataset. "
            f"default_rows_out={default_filter_stats['rows_out']} "
            f"surface_rows_out={surface_filter_stats['rows_out']}"
        )
    train_rowset_stats = verify_matching_row_sets(args.default_train_csv, args.surface_train_csv)

    model_source = resolve_model_source(args.model_name, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    context = build_context_summary(
        [args.default_train_csv, args.surface_train_csv],
        tokenizer,
        response_col=args.response_col,
        max_length_min=args.max_length_min,
        max_length_cap=args.max_length_cap,
        max_new_tokens_min=args.max_new_tokens_min,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )

    summary = {
        "subjid": int(args.subjid),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_end),
        "n_seed_files": int(len(paths)),
        "tuple": tuple_values,
        "merged_rows": int(len(merged_df)),
        "merged_unique_probs": int(merged_df["prob"].astype(str).nunique()),
        "raw_trace_health": raw_trace_health,
        "rowset_verification": rowset_stats,
        "train_rowset_verification": train_rowset_stats,
        "default_filter": default_filter_stats,
        "surface_filter": surface_filter_stats,
        "tokenizer_source": str(model_source),
        "context": context,
        "paths": {
            "merged_csv": str(args.merged_csv),
            "default_csv": str(args.default_csv),
            "surface_csv": str(args.surface_csv),
            "default_train_csv": str(args.default_train_csv),
            "surface_train_csv": str(args.surface_train_csv),
            "sp2013_csv": str(args.sp2013_csv),
        },
    }

    ensure_parent(args.context_json)
    with args.context_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
