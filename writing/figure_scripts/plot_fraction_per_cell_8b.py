#!/usr/bin/env python3
"""Per-cell |gap to human| on SP2013 fractions for Qwen3-8B across 4 variants.

Bars per cell: Base / + humanFT (from base) / + distill / + distill + humanFT.
8 cells = operation (add/sub/mul/div) x denom-type (ED/UD).
Mirrors plot_decimal_trio_bss2021.py for parity with the decimal per-cell figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


HUMAN_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
DISTILL_FT_ROLLOUTS = Path("/tmp/uma_pr02_max2/results/finetuning/trajectory_at_T10/rollouts/rollouts_8B.csv.gz")
FROMBASE_ROLLOUTS = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_frombase_trajectory_t10_fixed/rollouts_fbtraj_8b_hftep1.csv.gz")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

# 8 cells in canonical order: matches §4 (add/sub/mul/div x ED/UD)
CELL_ORDER = [
    ("add", "ED"), ("add", "UD"),
    ("sub", "ED"), ("sub", "UD"),
    ("mul", "ED"), ("mul", "UD"),
    ("div", "ED"), ("div", "UD"),
]
OP_DISPLAY = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}


def load_human_cells():
    h = pd.read_csv(HUMAN_CSV, low_memory=False).dropna(subset=["acc"])
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    cell_acc = h.groupby(["operation", "operands"])["acc"].mean()
    prob_to_cell = h.groupby("prob")[["operation", "operands"]].first()
    return cell_acc, prob_to_cell.reset_index()


def model_cell_acc(rollouts, prob_to_cell, stage=None):
    df = rollouts if stage is None else rollouts[rollouts["stage"] == stage]
    df = df.merge(prob_to_cell, on="prob", how="left")
    return df.groupby(["operation", "operands"])["correct"].mean()


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    human_cell, prob_to_cell = load_human_cells()
    dft = pd.read_csv(DISTILL_FT_ROLLOUTS, compression="gzip")
    fb = pd.read_csv(FROMBASE_ROLLOUTS, compression="gzip")

    base = model_cell_acc(dft, prob_to_cell, "base")
    hft = model_cell_acc(fb, prob_to_cell)  # file is already hft_ep1 only
    distill = model_cell_acc(dft, prob_to_cell, "distill_ep1")
    dhft = model_cell_acc(dft, prob_to_cell, "hft_ep1")

    def gap(m):
        return (m.reindex(CELL_ORDER) - human_cell.reindex(CELL_ORDER)).abs().values * 100

    gaps = {
        "Base": gap(base),
        "+ humanFT": gap(hft),
        "+ distill": gap(distill),
        "+ distill\n+ humanFT": gap(dhft),
    }
    maes = {k: float(np.mean(v)) for k, v in gaps.items()}

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(gaps)))
    labels = [f"{OP_DISPLAY[op]}\n{od}" for op, od in CELL_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 2.8), dpi=300)
    x = np.arange(len(CELL_ORDER))
    n = len(gaps)
    w = 0.80 / n
    offsets = (np.arange(n) - (n - 1) / 2) * w
    for off, (lbl, g), color in zip(offsets, gaps.items(), colors):
        ax.bar(x + off, g, w, color=color,
               label=f"{lbl.replace(chr(10), ' ')}  (MAE {maes[lbl]:.1f})",
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE model vs human", fontsize=9)
    y_max = max(70, max(g.max() for g in gaps.values()) + 5)
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              bbox_to_anchor=(1.0, 1.05))
    sns.despine(ax=ax)
    plt.tight_layout()

    out_png = OUT_DIR / "qwen3_8b_fraction_per_cell.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}\n")
    print("per-cell MAE (pp):")
    df_print = pd.DataFrame({k: v for k, v in gaps.items()}, index=labels).round(1)
    print(df_print.to_string())
    print()
    for lbl, v in maes.items():
        print(f"  {lbl.replace(chr(10), ' '):24s}  overall MAE = {v:.2f} pp")


if __name__ == "__main__":
    main()
