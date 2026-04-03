#!/usr/bin/env python3
"""Sampling-based evaluation for SP2013 and compatible fraction-problem datasets."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch


_TTHF = None


def _get_tthf():
    global _TTHF
    if _TTHF is not None:
        return _TTHF
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "results" / "transformer_replication"))
    import train_transformer_hf as tthf  # type: ignore

    _TTHF = tthf
    return _TTHF


def parse_args() -> argparse.Namespace:
    tthf = _get_tthf()
    parser = argparse.ArgumentParser(description="Sampling-based model evaluation on fraction datasets.")
    parser.add_argument("--run_dir", required=True, help="Training/fine-tune run directory containing checkpoint subdir.")
    parser.add_argument("--checkpoint_subdir", default="best")
    parser.add_argument(
        "--datasets_json",
        required=True,
        help=(
            "JSON list of dataset specs: "
            "[{\"name\":\"sp2013\",\"csv\":\"/path/sp2013.csv\",\"problem_col\":\"prob\"}]"
        ),
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)

    parser.add_argument("--use_param_grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sp2013_grid_g", default=tthf.DEFAULT_SP2013_GRID_G)
    parser.add_argument("--sp2013_grid_d", default=tthf.DEFAULT_SP2013_GRID_D)
    parser.add_argument("--sp2013_grid_rt", default=tthf.DEFAULT_SP2013_GRID_RT)
    parser.add_argument("--sp2013_grid_ice", default=tthf.DEFAULT_SP2013_GRID_ICE)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples_per_prompt", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=123)
    parser.add_argument("--progress_every", type=int, default=100)

    parser.add_argument("--local_files_only", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--model_source", default="")
    return parser.parse_args()


def parse_local_files_only(mode: str, train_args: Dict[str, Any]) -> bool:
    text = str(mode).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return bool(train_args.get("local_files_only", True))


def build_param_grid(
    use_param_grid: bool,
    grid_g: Sequence[float],
    grid_d: Sequence[float],
    grid_rt: Sequence[float],
    grid_ice: Sequence[float],
) -> List[Tuple[float, float, float, float]]:
    if not use_param_grid:
        return [(float("nan"), float("nan"), float("nan"), float("nan"))]
    return [(g, d, rt, ice) for g in grid_g for d in grid_d for rt in grid_rt for ice in grid_ice]


@contextmanager
def seeded_torch_rng(device, seed: int):
    devices = []
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [device_index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def generate_one(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> str:
    core = model.module if hasattr(model, "module") else model
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        with seeded_torch_rng(device, seed):
            out = core.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    gen_ids = out[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def parse_datasets_json(raw: str) -> List[Dict[str, str]]:
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("--datasets_json must be a non-empty list.")
    out: List[Dict[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset spec at index {idx} must be an object.")
        name = str(item.get("name", "")).strip()
        csv_path = str(item.get("csv", "")).strip()
        problem_col = str(item.get("problem_col", "prob")).strip() or "prob"
        if name == "" or csv_path == "":
            raise ValueError(f"Dataset spec at index {idx} requires non-empty name and csv.")
        out.append({"name": name, "csv": csv_path, "problem_col": problem_col})
    return out


def compute_dataset_metrics(df: pd.DataFrame, use_param_grid: bool) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "n_rows": 0,
            "sample_overall_acc": float("nan"),
            "sample_by_op": {},
            "pair_mean_acc": float("nan"),
            "pair_any_acc": float("nan"),
            "pair_by_op_mean_acc": {},
            "pair_by_op_any_acc": {},
            "n_pairs": 0,
            "n_problems": 0,
            "use_param_grid": bool(use_param_grid),
        }

    sample_overall = float(df["is_correct"].mean())
    sample_by_op: Dict[str, float] = {}
    for op in ["+", "-", "*", ":"]:
        op_df = df[df["op"] == op]
        sample_by_op[op] = float(op_df["is_correct"].mean()) if len(op_df) > 0 else float("nan")

    pair_cols = ["dataset", "prob", "op"] + (["g", "d", "rt_mu", "ice"] if use_param_grid else [])
    pair_df = df.groupby(pair_cols, as_index=False)["is_correct"].agg(pair_mean_acc="mean", pair_any_correct="max")
    pair_mean_acc = float(pair_df["pair_mean_acc"].mean()) if len(pair_df) > 0 else float("nan")
    pair_any_acc = float(pair_df["pair_any_correct"].mean()) if len(pair_df) > 0 else float("nan")

    pair_by_op_mean_acc: Dict[str, float] = {}
    pair_by_op_any_acc: Dict[str, float] = {}
    for op in ["+", "-", "*", ":"]:
        op_pairs = pair_df[pair_df["op"] == op]
        pair_by_op_mean_acc[op] = float(op_pairs["pair_mean_acc"].mean()) if len(op_pairs) > 0 else float("nan")
        pair_by_op_any_acc[op] = float(op_pairs["pair_any_correct"].mean()) if len(op_pairs) > 0 else float("nan")

    return {
        "n_rows": int(len(df)),
        "sample_overall_acc": sample_overall,
        "sample_by_op": sample_by_op,
        "pair_mean_acc": pair_mean_acc,
        "pair_any_acc": pair_any_acc,
        "pair_by_op_mean_acc": pair_by_op_mean_acc,
        "pair_by_op_any_acc": pair_by_op_any_acc,
        "n_pairs": int(len(pair_df)),
        "n_problems": int(df["prob"].nunique()),
        "use_param_grid": bool(use_param_grid),
    }


def evaluate(
    model,
    tokenizer,
    device,
    datasets: List[Dict[str, str]],
    param_grid: List[Tuple[float, float, float, float]],
    use_param_grid: bool,
    num_samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    sample_seed: int,
    progress_every: int,
) -> pd.DataFrame:
    tthf = _get_tthf()
    rows: List[Dict[str, Any]] = []

    total_prompts = 0
    dataset_frames: List[Tuple[str, str, pd.DataFrame]] = []
    for spec in datasets:
        dataset_name = spec["name"]
        csv_path = spec["csv"]
        problem_col = spec["problem_col"]
        df = pd.read_csv(csv_path)
        if problem_col not in df.columns:
            raise ValueError(f"Dataset '{dataset_name}' missing problem column '{problem_col}': {csv_path}")
        dataset_frames.append((dataset_name, problem_col, df))
        total_prompts += len(df)

    total_generations = total_prompts * len(param_grid) * num_samples_per_prompt
    done = 0
    t0 = time.time()

    for dataset_name, problem_col, df in dataset_frames:
        problems = df[problem_col].astype(str).tolist()
        for prob_idx, prob in enumerate(problems, start=1):
            correct = tthf.compute_correct_answer(prob)
            op = tthf.op_from_prob(prob)
            for grid_idx, (g, d, rt_mu, ice) in enumerate(param_grid):
                if use_param_grid:
                    prompt = tthf.build_student_prompt(prob=prob, g=g, d=d, rt_mu=rt_mu, ice=ice)
                else:
                    prompt = f"Solve this fraction problem: {prob}=?"

                for sample_idx in range(num_samples_per_prompt):
                    seed = int(sample_seed + done)
                    gen_text = generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                    )
                    pred = tthf.extract_final_answer(gen_text)
                    is_correct = int(tthf.answers_match(pred, correct))
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "prob": prob,
                            "op": op,
                            "g": g if use_param_grid else float("nan"),
                            "d": d if use_param_grid else float("nan"),
                            "rt_mu": rt_mu if use_param_grid else float("nan"),
                            "ice": ice if use_param_grid else float("nan"),
                            "grid_index": grid_idx if use_param_grid else 0,
                            "sample_idx": sample_idx,
                            "sample_seed": seed,
                            "pred_answer": pred,
                            "correct_answer": correct,
                            "is_correct": is_correct,
                            "generation_text": gen_text,
                        }
                    )
                    done += 1
                    if progress_every > 0 and (done % progress_every == 0 or done == total_generations):
                        elapsed = max(time.time() - t0, 1e-9)
                        rate = done / elapsed
                        eta_sec = (total_generations - done) / max(rate, 1e-12)
                        pct = 100.0 * done / max(total_generations, 1)
                        print(
                            f"[progress] {done}/{total_generations} ({pct:.1f}%) "
                            f"dataset={dataset_name} prob={prob_idx}/{len(problems)} rate={rate:.2f} gen/s eta={eta_sec/60.0:.1f}m",
                            flush=True,
                        )

    return pd.DataFrame(rows)


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    tthf = _get_tthf()
    args = parse_args()

    if args.num_samples_per_prompt < 1:
        raise ValueError("--num_samples_per_prompt must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if not (0 < args.top_p <= 1.0):
        raise ValueError("--top_p must be in (0, 1].")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = (run_dir / args.checkpoint_subdir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    out_csv = Path(args.out_csv).resolve()
    out_json = Path(args.out_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    datasets = parse_datasets_json(args.datasets_json)
    datasets_resolved = [
        {
            "name": spec["name"],
            "csv": str(Path(spec["csv"]).resolve()),
            "problem_col": spec["problem_col"],
        }
        for spec in datasets
    ]

    train_args_path = checkpoint_dir / "train_args.json"
    train_args: Dict[str, Any] = {}
    if train_args_path.exists():
        with train_args_path.open("r", encoding="utf-8") as f:
            train_args = json.load(f)

    model_source = str(args.model_source).strip() or str(train_args.get("model_name", "HuggingFaceTB/SmolLM2-135M"))
    local_files_only = parse_local_files_only(args.local_files_only, train_args)

    grid_g = tthf.parse_float_list_arg(args.sp2013_grid_g, "sp2013_grid_g")
    grid_d = tthf.parse_float_list_arg(args.sp2013_grid_d, "sp2013_grid_d")
    grid_rt = tthf.parse_float_list_arg(args.sp2013_grid_rt, "sp2013_grid_rt")
    grid_ice = tthf.parse_float_list_arg(args.sp2013_grid_ice, "sp2013_grid_ice")
    param_grid = build_param_grid(
        use_param_grid=bool(args.use_param_grid),
        grid_g=grid_g,
        grid_d=grid_d,
        grid_rt=grid_rt,
        grid_ice=grid_ice,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70, flush=True)
    print("SAMPLING EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"checkpoint_dir: {checkpoint_dir}", flush=True)
    print(f"datasets: {[d['name'] for d in datasets_resolved]}", flush=True)
    print(f"use_param_grid: {args.use_param_grid} (size={len(param_grid)})", flush=True)
    print(f"num_samples_per_prompt: {args.num_samples_per_prompt}", flush=True)
    print(f"sampling: temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}", flush=True)
    print(f"local_files_only: {local_files_only}", flush=True)
    print(f"device: {device}", flush=True)
    print("", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # SmolLM2 is decoder-only; left padding is the safe choice for generation-time batching.
    tokenizer.padding_side = "left"

    model = tthf.load_eval_model(
        checkpoint_dir=str(checkpoint_dir),
        base_model_name_or_path=model_source,
        device=device,
        local_files_only=local_files_only,
    )
    model.eval()

    out_df = evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        datasets=datasets_resolved,
        param_grid=param_grid,
        use_param_grid=bool(args.use_param_grid),
        num_samples_per_prompt=int(args.num_samples_per_prompt),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        sample_seed=int(args.sample_seed),
        progress_every=max(0, int(args.progress_every)),
    )

    dataset_metrics: Dict[str, Any] = {}
    for dataset_name in out_df["dataset"].unique().tolist() if len(out_df) > 0 else []:
        dataset_metrics[dataset_name] = compute_dataset_metrics(
            out_df[out_df["dataset"] == dataset_name].reset_index(drop=True),
            use_param_grid=bool(args.use_param_grid),
        )

    overall_metrics = compute_dataset_metrics(out_df, use_param_grid=bool(args.use_param_grid))
    metrics = {
        "overall": overall_metrics,
        "by_dataset": dataset_metrics,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "datasets": datasets_resolved,
        "grid_size": int(len(param_grid)),
        "num_samples_per_prompt": int(args.num_samples_per_prompt),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "sample_seed": int(args.sample_seed),
        "model_source": model_source,
        "local_files_only": bool(local_files_only),
    }

    out_df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("", flush=True)
    print(f"wrote_csv={out_csv}", flush=True)
    print(f"wrote_metrics={out_json}", flush=True)


if __name__ == "__main__":
    main()
