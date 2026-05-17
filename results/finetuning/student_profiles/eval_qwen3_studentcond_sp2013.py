#!/usr/bin/env python3
"""
Evaluate a Qwen3 (base / distill-LoRA / distill+humanFT-LoRA) checkpoint on
the 16 SP2013 fraction problems and compute MAE pp vs the human accuracy
distribution from siegler_fraction_human.csv.

Supports multi-temperature sweep in a single job (model is loaded once and
reused across all temperatures) and optional per-rollout output for later
re-grading.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SP2013_PROBLEMS = [
    "3/5+1/5", "4/5+3/5", "3/5+1/4", "2/3+3/5",
    "3/5-1/5", "4/5-3/5", "3/5-1/4", "2/3-3/5",
    "3/5*1/5", "4/5*3/5", "3/5*1/4", "2/3*3/5",
    "3/5:1/5", "4/5:3/5", "3/5:1/4", "2/3:3/5",
]
OP_CHARS = {"+", "-", "*", ":"}


def _format_num(v: float) -> str:
    """Match train_transformer_hf.format_numeric_token."""
    fv = float(v)
    if fv.is_integer():
        return str(int(fv))
    return f"{fv:.10f}".rstrip("0").rstrip(".")


def _space_problem(prob: str) -> str:
    for ch in ("+", "-", "*", ":"):
        if ch in prob:
            left, right = prob.split(ch, 1)
            return f"{left.strip()} {ch} {right.strip()}"
    return prob


def build_studentcond_prompt(prob: str, g: float, d: float, rt: float, ice: float) -> str:
    return (
        f"<student> g {_format_num(g)} d {_format_num(d)} "
        f"rt {_format_num(rt)} ice {_format_num(ice)} </student>\n"
        f"Solve this fraction problem: {_space_problem(prob)}=?"
    )


def gt_answer(prob: str) -> Fraction:
    m = re.match(r"(\d+)/(\d+)\s*([+\-*:])\s*(\d+)/(\d+)", prob)
    if m is None:
        raise ValueError(f"unparseable problem: {prob}")
    a, b, op, c, d = int(m[1]), int(m[2]), m[3], int(m[4]), int(m[5])
    f1, f2 = Fraction(a, b), Fraction(c, d)
    return {"+": f1 + f2, "-": f1 - f2, "*": f1 * f2, ":": f1 / f2}[op]


def op_of(prob: str) -> str:
    for ch, name in (("+", "add"), ("-", "sub"), ("*", "mul"), (":", "div")):
        if ch in prob:
            return name
    raise ValueError(prob)


_ANS_TAG_RE = re.compile(r"###\s*answer\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)
# Permissive fallback patterns for "answer-like" tokens in free text.
_MIXED_RE = re.compile(r"(-?)(\d+)\s+(\d+)\s*/\s*(\d+)")
_FRAC_RE = re.compile(r"-?\d+\s*/\s*-?\d+")
_DEC_RE = re.compile(r"-?\d+\.\d+")


def parse_answer_token(s: str) -> Optional[Fraction]:
    """Parse a single candidate answer string into a Fraction.

    Handles: mixed number ('2 2/5'), plain fraction ('3/5'), integer ('5'),
    decimal ('2.4'). Tolerates trailing punctuation and extra text after the
    answer token.
    """
    if not s:
        return None
    s = s.strip()
    # Mixed number: "2 2/5"
    m = re.match(r"^\s*(-?)(\d+)\s+(\d+)\s*/\s*(\d+)\b", s)
    if m:
        sign = -1 if m[1] == "-" else 1
        whole, num, den = int(m[2]), int(m[3]), int(m[4])
        if den == 0:
            return None
        return sign * (Fraction(whole) + Fraction(num, den))
    # Plain fraction: "3/5"
    m = re.match(r"^\s*(-?\d+)\s*/\s*(-?\d+)\b", s)
    if m:
        try:
            return Fraction(int(m[1]), int(m[2]))
        except ZeroDivisionError:
            return None
    # Decimal: "2.4"
    m = re.match(r"^\s*(-?\d+\.\d+)", s)
    if m:
        try:
            return Fraction(Decimal(m[1]))
        except (InvalidOperation, ValueError):
            return None
    # Integer: "5"
    m = re.match(r"^\s*(-?\d+)", s)
    if m:
        try:
            return Fraction(int(m[1]))
        except ValueError:
            return None
    return None


def extract_answer(text: str) -> Optional[Fraction]:
    """Pull the final answer out of a model rollout. Prefers `### answer: X`."""
    m = _ANS_TAG_RE.search(text)
    if m is not None:
        candidate = m.group(1).strip()
        f = parse_answer_token(candidate)
        if f is not None:
            return f
    # Fallback: scan from the end for the latest mixed-number, then fraction,
    # then decimal, then integer match.
    for pattern in (_MIXED_RE, _FRAC_RE, _DEC_RE):
        matches = list(pattern.finditer(text))
        if matches:
            f = parse_answer_token(matches[-1].group(0))
            if f is not None:
                return f
    return None


def is_correct(pred: Optional[Fraction], gt: Fraction) -> bool:
    if pred is None:
        return False
    return pred == gt


def load_model(
    base_model: str,
    adapter_dir: Optional[Path],
    device: torch.device,
    local_files_only: bool,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    tok_src = str(adapter_dir) if adapter_dir is not None else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def rollout_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> list[str]:
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=128,
    ).to(device)
    torch.manual_seed(seed)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded


def evaluate_one_temperature(
    model,
    tokenizer,
    device: torch.device,
    args,
    T: float,
    human_acc: dict[str, float],
    save_rollouts: bool,
) -> tuple[dict, pd.DataFrame, Optional[pd.DataFrame]]:
    """Run num_rollouts per problem at the given T. Returns (summary, per-problem df,
    optional rollouts df)."""
    records = []
    rollout_rows = []
    seed_base = args.sample_seed
    bsz = args.rollout_batch_size
    for prob in SP2013_PROBLEMS:
        gt = gt_answer(prob)
        if args.student_g is not None:
            prompt = build_studentcond_prompt(
                prob, args.student_g, args.student_d, args.student_rt, args.student_ice
            )
        else:
            prompt = args.prompt_template.format(prob=prob)
        n_correct = 0
        n_total = 0
        t0 = time.time()
        for batch_idx in range(0, args.num_rollouts, bsz):
            sub = min(bsz, args.num_rollouts - batch_idx)
            decoded = rollout_batch(
                model=model, tokenizer=tokenizer,
                prompts=[prompt] * sub, device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=T, top_p=args.top_p, top_k=args.top_k,
                seed=seed_base + batch_idx,
            )
            for k, txt in enumerate(decoded):
                pred = extract_answer(txt)
                ok = is_correct(pred, gt)
                n_correct += int(ok)
                n_total += 1
                if save_rollouts:
                    rollout_rows.append({
                        "T": T, "prob": prob, "op": op_of(prob), "gt": str(gt),
                        "rollout_idx": batch_idx + k,
                        "raw_text": txt[:500],
                        "pred": "" if pred is None else str(pred),
                        "correct": int(ok),
                    })
        acc = n_correct / max(n_total, 1)
        records.append({
            "T": T, "prob": prob, "op": op_of(prob), "gt": str(gt),
            "n_rollouts": n_total, "n_correct": n_correct, "acc_model": acc,
            "elapsed_sec": round(time.time() - t0, 2),
        })
        print(f"[eval T={T}] {prob:<12s} acc={acc:.3f} ({n_correct}/{n_total}) "
              f"in {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame.from_records(records)
    df["acc_human"] = df["prob"].map(human_acc).astype(float)
    df["abs_err_pp"] = (df["acc_model"] - df["acc_human"]).abs() * 100.0
    summary = {
        "temperature": T,
        "mae_pp_overall": float(df["abs_err_pp"].mean()),
        "mae_pp_by_op": {op: float(v) for op, v in df.groupby("op")["abs_err_pp"].mean().to_dict().items()},
        "acc_overall_model": float(df["acc_model"].mean()),
        "acc_overall_human": float(df["acc_human"].mean()),
        "num_rollouts": args.num_rollouts,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    rollouts_df = pd.DataFrame.from_records(rollout_rows) if save_rollouts else None
    return summary, df, rollouts_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", type=Path, default=None)
    p.add_argument("--condition_label", required=True,
                   help="Base label; per-temperature suffix _t{TT} is auto-appended for multi-T runs.")
    p.add_argument("--human_csv", type=Path, required=True,
                   help="Path to siegler_fraction_human.csv (per-problem human accuracies).")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_rollouts", type=int, default=256)
    p.add_argument("--rollout_batch_size", type=int, default=64,
                   help="Per-batch rollouts. Default raised from 16; H100/H200 can fit far more.")
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=None,
                   help="Single-temperature mode (backwards compatible).")
    p.add_argument("--temperatures", type=str, default=None,
                   help="Comma-separated list of temperatures; overrides --temperature.")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--sample_seed", type=int, default=123)
    p.add_argument("--prompt_template", type=str, default="Solve this fraction problem: {prob}=?")
    p.add_argument("--student_g", type=float, default=None,
                   help="If set, use student-conditioned prompt with these g/d/rt/ice values.")
    p.add_argument("--student_d", type=float, default=None)
    p.add_argument("--student_rt", type=float, default=None)
    p.add_argument("--student_ice", type=float, default=None)
    p.add_argument("--no_local_files_only", action="store_true")
    p.add_argument("--save_rollouts", action="store_true",
                   help="Persist per-rollout raw text + parsed answer for later re-grading.")
    args = p.parse_args()

    if args.temperatures:
        Ts = [float(t) for t in args.temperatures.split(",") if t.strip()]
    elif args.temperature is not None:
        Ts = [float(args.temperature)]
    else:
        Ts = [0.7]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sc_params = [args.student_g, args.student_d, args.student_rt, args.student_ice]
    sc_set = [p is not None for p in sc_params]
    if any(sc_set) and not all(sc_set):
        raise SystemExit("--student_g/d/rt/ice must all be set together, or none.")
    if all(sc_set):
        demo = build_studentcond_prompt(
            SP2013_PROBLEMS[0], args.student_g, args.student_d, args.student_rt, args.student_ice
        )
        print(f"[eval] student-conditioned prompt prefix in use:\n{demo}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] condition={args.condition_label} device={device} Ts={Ts}", flush=True)
    print(f"[eval] base={args.base_model}", flush=True)
    print(f"[eval] adapter={args.adapter_dir}", flush=True)
    print(f"[eval] batch_size={args.rollout_batch_size} num_rollouts={args.num_rollouts}", flush=True)

    t_load = time.time()
    model, tokenizer = load_model(
        args.base_model, args.adapter_dir, device,
        local_files_only=not args.no_local_files_only,
    )
    print(f"[eval] model loaded in {time.time()-t_load:.1f}s", flush=True)

    # Human accuracies (used for MAE computation; same across temperatures).
    human = pd.read_csv(args.human_csv, low_memory=False)
    human = human.dropna(subset=["acc"]).copy()
    human["prob_uma"] = human["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    human_acc = human.groupby("prob_uma")["acc"].mean().to_dict()

    all_rollouts = []
    for T in Ts:
        t_start = time.time()
        summary, per_problem, rollouts_df = evaluate_one_temperature(
            model, tokenizer, device, args, T, human_acc, args.save_rollouts,
        )
        elapsed = time.time() - t_start

        t_label = f"{T:.1f}".replace(".", "").rstrip("0") or "0"
        # Match historical labeling: t07, t10, t11, t13, t15, t17, t19, t09
        # Two-digit form from "0.7"→"07", "1.0"→"10", "1.3"→"13"
        t_tag = f"{int(round(T*10)):02d}"
        cond_label = args.condition_label if len(Ts) == 1 else f"{args.condition_label}_t{t_tag}"

        out_csv = args.output_dir / f"sp2013_per_problem_{cond_label}.csv"
        per_problem.to_csv(out_csv, index=False)
        summary.update({
            "condition": cond_label,
            "base_model": args.base_model,
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir is not None else None,
            "elapsed_sec": round(elapsed, 2),
        })
        out_json = args.output_dir / f"sp2013_summary_{cond_label}.json"
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"[eval] T={T} -> MAE pp={summary['mae_pp_overall']:.2f} "
              f"acc={summary['acc_overall_model']:.3f} "
              f"({elapsed:.1f}s); wrote {out_json.name}", flush=True)

        if rollouts_df is not None:
            all_rollouts.append(rollouts_df.assign(condition=cond_label))

    if all_rollouts:
        rollouts_path = args.output_dir / f"rollouts_{args.condition_label}.csv.gz"
        pd.concat(all_rollouts, ignore_index=True).to_csv(rollouts_path, index=False, compression="gzip")
        print(f"[eval] wrote {rollouts_path}", flush=True)


if __name__ == "__main__":
    main()
