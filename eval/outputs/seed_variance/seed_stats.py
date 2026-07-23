#!/usr/bin/env python3
"""Per-cell and overall accuracy for every (config, seed) rollout set, scored
with the fixed final-answer parsers (fractions: mixed-number-aware region
parser; decimals: final-answer-region Decimal comparison). Seed 42 = the
canonical rollouts behind Table 1. Writes seed_cell_acc.csv (long format).
"""
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp")
from rescore_frac_dir import extract_final_answer, to_frac

JOB = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc")
SV = JOB / "tmp/seed_variance"
EV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs")
NEW = JOB / "tmp/newckpt_rollouts_4bi"

CANONICAL = {
    "base_frac": EV / "fraction_4b/qwen3_4b_distill_humanft.csv",
    "inst_frac": NEW / "rollouts_4bi_fractions_distill_humanFT.csv",
    "base_dec": EV / "decimal_4b/qwen3_4b_distill_humanft.csv",
    "inst_dec": NEW / "rollouts_4bi_decimals_distill_humanFT.csv",
}

DEC_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def dec_candidates(text):
    if not isinstance(text, str):
        return set()
    m = list(re.finditer(r"(?:###\s*answer|final\s+answer|answer)[:\s]", text, re.I))
    region = text[m[-1].start():] if m else text[-100:]
    out = set()
    for g in DEC_RE.finditer(region):
        try:
            out.add(Decimal(g.group(1)).normalize())
        except InvalidOperation:
            pass
    return out


def score(df, domain):
    if domain == "fraction":
        correct = df["correct_answer"].apply(to_frac)
        return np.array([int(c is not None and c in extract_final_answer(r))
                         for r, c in zip(df["model_response"], correct)])
    ok = []
    for r, c in zip(df["model_response"], df["correct_answer"]):
        try:
            t = Decimal(str(c)).normalize()
        except InvalidOperation:
            ok.append(0)
            continue
        ok.append(int(t in dec_candidates(r)))
    return np.array(ok)


def cell_col(df, domain):
    if domain == "fraction":
        # canonical files use denom_type; seed evals too; inst_frac newckpt uses 'operands' for denom
        col = "denom_type" if "denom_type" in df.columns else "operands"
        return df["operation"].str.capitalize().str[:3] + " " + df[col].str.upper()
    return df["operation"].str.capitalize() + " " + df["operands"].str.upper()


def main():
    rows = []
    for config in ["base_frac", "inst_frac", "base_dec", "inst_dec"]:
        domain = "fraction" if config.endswith("frac") else "decimal"
        paths = {42: CANONICAL[config]}
        for s in (43, 44, 45, 46):
            paths[s] = SV / f"rollouts/{config}_seed{s}.csv"
        for seed, p in paths.items():
            df = pd.read_csv(p)
            df["ok"] = score(df, domain)
            df["cell"] = cell_col(df, domain)
            for cell, g in df.groupby("cell"):
                rows.append({"config": config, "seed": seed, "cell": cell,
                             "acc": g["ok"].mean() * 100, "n": len(g)})
            rows.append({"config": config, "seed": seed, "cell": "__overall__",
                         "acc": df["ok"].mean() * 100, "n": len(df)})
            print(f"{config} seed={seed}: overall={df['ok'].mean()*100:.1f}% ({p.name})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(SV / "seed_cell_acc.csv", index=False)
    print("\n=== overall acc: mean +/- sd across 5 seeds ===")
    ov = out[out.cell == "__overall__"]
    for config, g in ov.groupby("config"):
        print(f"{config}: {g['acc'].mean():.1f} +/- {g['acc'].std(ddof=1):.1f} "
              f"(seed42={g[g.seed == 42]['acc'].iloc[0]:.1f})")
    print("\n=== per-cell sd range across seeds ===")
    pc = out[out.cell != "__overall__"]
    for config, g in pc.groupby("config"):
        sd = g.groupby("cell")["acc"].std(ddof=1)
        print(f"{config}: cell-sd min={sd.min():.1f} median={sd.median():.1f} max={sd.max():.1f}")


if __name__ == "__main__":
    main()
