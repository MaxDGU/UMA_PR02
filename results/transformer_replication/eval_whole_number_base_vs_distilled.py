#!/usr/bin/env python3
"""Evaluate base SmolLM2-135M vs whole-number-distilled checkpoint.

Runs both models on the held-out single-digit test sets (sd_add: 81 problems,
sd_mul: 64 problems = 145 total). All test problems are excluded from
training (per generate_traces_parallel.py holdout logic). Reports per-cell
accuracy and dumps full generations for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "Solve this whole-number arithmetic problem: {prob}=?"
ANSWER_RE = re.compile(r"###\s*answer\s*:\s*(-?\d+)", flags=re.IGNORECASE)
FALLBACK_NUM_RE = re.compile(r"(-?\d+)\s*\.?\s*$")  # last number in text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M",
                   help="HF id or path of the base/untuned model.")
    p.add_argument("--distilled-checkpoint", required=True,
                   help="Path to distilled checkpoint dir (typically '<run>/best').")
    p.add_argument("--sd-add-csv", default="UMA_PR02_fork/1. Model/Problem Sets Testing/sd_add.csv",
                   help="Path (relative to project root) to sd_add.csv.")
    p.add_argument("--sd-mul-csv", default="UMA_PR02_fork/1. Model/Problem Sets Testing/sd_mul.csv",
                   help="Path (relative to project root) to sd_mul.csv.")
    p.add_argument("--project-root", default="/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions",
                   help="Where to resolve test-set CSV paths.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def compute_correct(prob: str) -> Optional[int]:
    m = re.fullmatch(r"\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*", prob)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def parse_response_answer(text: str) -> Optional[int]:
    m = ANSWER_RE.search(text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    # Fallback: last integer in the text (handles base-model that doesn't follow the contract)
    nums = re.findall(r"-?\d+", text)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


def cell_label(prob: str) -> str:
    """Cell label for whole-number test problems: op + 'sd' (single-digit)."""
    if "+" in prob:
        return "Add_sd"
    if "-" in prob:
        return "Sub_sd"
    if "*" in prob:
        return "Mul_sd"
    return "Other"


def load_test_problems(project_root: str, sd_add_csv: str, sd_mul_csv: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, rel in [("sd_add", sd_add_csv), ("sd_mul", sd_mul_csv)]:
        path = Path(project_root) / rel
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            prob = str(r["prob"]).strip()
            rows.append({"prob": prob, "set": label, "cell": cell_label(prob), "key": compute_correct(prob)})
    return rows


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int, device: str) -> List[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Strip prompt tokens; decode only the new tokens
    prompt_lens = enc["input_ids"].shape[1]
    completions = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_lens:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return completions


def evaluate_model(model_name_or_path: str, label: str, problems: List[Dict[str, object]],
                   max_new_tokens: int, batch_size: int, device: str) -> List[Dict[str, object]]:
    print(f"\n{'='*60}\n[eval] {label}: loading {model_name_or_path}\n{'='*60}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched causal generation
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    results: List[Dict[str, object]] = []
    prompts_full = [PROMPT_TEMPLATE.format(prob=r["prob"]) for r in problems]
    for i in range(0, len(problems), batch_size):
        batch_probs = problems[i:i+batch_size]
        batch_prompts = prompts_full[i:i+batch_size]
        completions = generate_batch(model, tokenizer, batch_prompts, max_new_tokens, device)
        for r, prompt, comp in zip(batch_probs, batch_prompts, completions):
            parsed = parse_response_answer(comp)
            correct = (parsed is not None and parsed == r["key"])
            results.append({
                "model": label,
                "set": r["set"], "cell": r["cell"],
                "prob": r["prob"], "key": r["key"],
                "prompt": prompt,
                "generation": comp,
                "parsed_answer": parsed,
                "is_correct": bool(correct),
            })
    # Free GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def summarize(results: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    by_cell = df.groupby(["model", "cell"])["is_correct"].agg(["mean", "count"]).reset_index()
    overall = df.groupby("model")["is_correct"].agg(["mean", "count"]).reset_index()
    overall["cell"] = "ALL"
    out = pd.concat([by_cell, overall], ignore_index=True)
    out["mean"] = (out["mean"] * 100).round(1)
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_test_problems(args.project_root, args.sd_add_csv, args.sd_mul_csv)
    print(f"Loaded {len(problems)} test problems "
          f"({sum(1 for p in problems if p['set']=='sd_add')} sd_add, "
          f"{sum(1 for p in problems if p['set']=='sd_mul')} sd_mul)")

    all_rows: List[Dict[str, object]] = []
    for label, ckpt in [("base", args.base_model), ("distilled", args.distilled_checkpoint)]:
        rows = evaluate_model(ckpt, label, problems, args.max_new_tokens, args.batch_size, args.device)
        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    raw_path = out_dir / "raw_generations.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw generations -> {raw_path}")

    summary = summarize(all_rows)
    summary_path = out_dir / "accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}\nACCURACY (%)\n{'='*60}")
    pivot = summary.pivot_table(index="cell", columns="model", values="mean").round(1)
    print(pivot.to_string())

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_problems": len(problems),
            "models": {
                lbl: {
                    "overall_accuracy_pct": float(raw_df[raw_df["model"]==lbl]["is_correct"].mean() * 100),
                    "n": int((raw_df["model"]==lbl).sum()),
                }
                for lbl in ["base", "distilled"]
            },
            "by_cell": pivot.to_dict(orient="index"),
        }, f, indent=2)
    print(f"\nSaved summary -> {summary_path}, {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
