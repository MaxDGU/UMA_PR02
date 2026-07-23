#!/usr/bin/env python3
"""Per-cell |gap to UMA| on single-digit whole-number arithmetic: Qwen3-4B base
vs distill. There is no public human whole-number dataset at this grade level,
so the reference is UMA's 1,000-student rollout (sd_add: 0.581, sd_mul: 0.255).

Cells: Add-NoCarry (a+b<10), Add-Carry (a+b>=10), Mul-Small (a*b<=20),
Mul-Large (a*b>20). Bars are MAG-U in percentage points; lower is closer to UMA.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


UMA_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_whole_ref")
BASE_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/qwen_base_whole_eval_20260521_155355")
DISTILL_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_whole_3shards_subset1m_base_e1_20260520_143547_lora/whole_eval_20260521_101502")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")


def parse(prob: str):
    m = re.match(r"\s*(\d+)\s*([+*])\s*(\d+)\s*", prob)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return a, op, b


def assign_cell(prob: str):
    a, op, b = parse(prob)
    if op == "+":
        return "Add\nNoCarry" if a + b < 10 else "Add\nCarry"
    return "Mul\nSmall" if a * b <= 20 else "Mul\nLarge"


def acc_per_cell_from_uma():
    add = pd.read_csv(UMA_DIR / "sd_add_per_problem.csv")
    mul = pd.read_csv(UMA_DIR / "sd_mul_per_problem.csv")
    df = pd.concat([add, mul], ignore_index=True)
    df["cell"] = df["prob"].apply(assign_cell)
    out = (df.groupby("cell")
             .apply(lambda g: (g["uma_acc"] * g["n_trials"]).sum() / g["n_trials"].sum())
             .rename("acc").reset_index())
    return out


def acc_per_cell_from_samples(per_sample_path: Path):
    df = pd.read_csv(per_sample_path, low_memory=False)
    df["prob"] = df["problem"].astype(str).str.replace("×", "*").str.replace(" ", "")
    df["cell"] = df["prob"].apply(assign_cell)
    out = df.groupby("cell")["is_correct"].mean().rename("acc").reset_index()
    return out


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    cell_order = ["Add\nNoCarry", "Add\nCarry", "Mul\nSmall", "Mul\nLarge"]
    uma = acc_per_cell_from_uma().set_index("cell").loc[cell_order]
    base = acc_per_cell_from_samples(BASE_DIR / "per_sample.csv").set_index("cell").loc[cell_order]
    distill = acc_per_cell_from_samples(DISTILL_DIR / "per_sample.csv").set_index("cell").loc[cell_order]

    print("\nPer-cell accuracies:")
    print(pd.DataFrame({"UMA": uma["acc"], "Base": base["acc"], "Distill": distill["acc"]}))

    base_gap = (base["acc"] - uma["acc"]).abs().values * 100
    distill_gap = (distill["acc"] - uma["acc"]).abs().values * 100
    mag_base = float(base_gap.mean())
    mag_distill = float(distill_gap.mean())

    viridis = plt.cm.viridis(np.linspace(0.15, 0.8, 3))
    c_base, c_distill = viridis[0], viridis[1]

    fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=300)
    x = np.arange(len(cell_order))
    w = 0.36
    ax.bar(x - w/2, base_gap, w, color=c_base,
           label=f"Base  (MAE {mag_base:.1f})",
           edgecolor="white", linewidth=0.4)
    ax.bar(x + w/2, distill_gap, w, color=c_distill,
           label=f"+ distill  (MAE {mag_distill:.1f})",
           edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_order, fontsize=8)
    ax.set_ylabel("MAE model vs UMA", fontsize=9)
    y_max = max(50, max(base_gap.max(), distill_gap.max()) + 5)
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              bbox_to_anchor=(1.0, 1.05))
    sns.despine(ax=ax)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "qwen_whole_base_distill_uma_per_cell.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    print(f"base     MAG-U = {mag_base:.2f} pp")
    print(f"distill  MAG-U = {mag_distill:.2f} pp")


if __name__ == "__main__":
    main()
