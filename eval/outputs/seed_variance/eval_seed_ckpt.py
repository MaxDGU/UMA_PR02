#!/usr/bin/env python3
"""Roll out one humanFT seed checkpoint on the paper problem set (fractions or
decimals) at the paper protocol (T=1.0, top_p=0.95) and write a per-sample CSV.
Scoring stored in the CSV uses the ### answer marker parser for quick logging;
the plotting stage re-scores with the fixed final-answer parsers.
"""
from __future__ import annotations
import argparse
import re
from fractions import Fraction
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FRAC_PROBS = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/fraction_4b/qwen3_4b_base.csv"
DEC_PROBS = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/decimal_4b/qwen3_4b_base.csv"


def marker_answer(resp):
    m = re.search(r"###\s*answer:\s*([^\n]+?)(?:\n|$)", resp or "")
    return m.group(1).strip().rstrip(".").strip() if m else ""


def to_num(s):
    s = str(s).strip()
    if not s:
        return None
    if "/" in s:
        try:
            n, d = s.split("/", 1)
            return Fraction(int(float(n)), int(float(d)))
        except Exception:
            return None
    try:
        return Fraction(float(s)).limit_denominator(100000)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--domain", choices=["fraction", "decimal"], required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=192)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, args.ckpt)
    model = model.merge_and_unload().to(device).eval()

    src = FRAC_PROBS if args.domain == "fraction" else DEC_PROBS
    meta_col = "denom_type" if args.domain == "fraction" else "operands"
    probs = (pd.read_csv(src)[["problem", "correct_answer", "operation", meta_col]]
             .drop_duplicates(subset=["problem"]).reset_index(drop=True))
    prompts = [f"Solve this {args.domain} problem: {p}=?" for p in probs["problem"]]
    print(f"{len(probs)} problems x {args.n_samples} samples", flush=True)

    outs = [[] for _ in prompts]
    for r in range(args.n_samples):
        if r % 20 == 0:
            print(f"  round {r}/{args.n_samples}", flush=True)
        for start in range(0, len(prompts), args.batch_size):
            b = prompts[start:start + args.batch_size]
            enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                                     max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in enumerate(texts):
                outs[start + i].append(t)

    rows = []
    for i in range(len(probs)):
        true_s = str(probs.loc[i, "correct_answer"]).strip()
        true_n = to_num(true_s)
        for s, resp in enumerate(outs[i]):
            pred = marker_answer(resp)
            pn = to_num(pred)
            rows.append({
                "problem": probs.loc[i, "problem"],
                "operation": probs.loc[i, "operation"],
                meta_col: probs.loc[i, meta_col],
                "correct_answer": true_s,
                "sample_idx": s,
                "parsed_answer": pred,
                "is_correct": int(pn is not None and true_n is not None and pn == true_n),
                "model_response": resp,
            })
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  acc={df['is_correct'].mean()*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
