#!/usr/bin/env python3
"""Single bar plot for Qwen3-8B on SP2013 fractions across four variants:
Base, + humanFT (from base), + distill, + distill + humanFT.

MAE is per-cell |Acc_model - Acc_human| averaged over the 8 SP2013 cells.
Numbers come from the trajectory CSVs already produced by the magh plot scripts.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
DISTILL_FT_CSV = FIG_DIR / "qwen_trajectory_at_T10_magh.csv"
FROMBASE_CSV = FIG_DIR / "qwen_frombase_trajectory_at_T10_magh.csv"


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df_dft = pd.read_csv(DISTILL_FT_CSV)
    df_fb = pd.read_csv(FROMBASE_CSV)

    base = float(df_dft.query("size=='8B' and stage=='base'")["MAG_H_pp"].iloc[0])
    hft_only = float(df_fb.query("size=='8B' and stage=='hft_ep1'")["MAG_H_pp"].iloc[0])
    distill_only = float(df_dft.query("size=='8B' and stage=='distill_ep1'")["MAG_H_pp"].iloc[0])
    distill_hft = float(df_dft.query("size=='8B' and stage=='hft_ep1'")["MAG_H_pp"].iloc[0])

    labels = ["Base", "+ humanFT", "+ distill", "+ distill\n+ humanFT"]
    values = [base, hft_only, distill_only, distill_hft]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))

    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=300)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.4, width=0.65)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    fontsize=8, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE model vs human", fontsize=9)
    ax.set_ylim(0, max(values) * 1.18)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_png = FIG_DIR / "qwen3_8b_fraction_variants.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")
    for lbl, v in zip(labels, values):
        print(f"  {lbl.replace(chr(10),' '):24s}  MAE = {v:.2f} pp")


if __name__ == "__main__":
    main()
