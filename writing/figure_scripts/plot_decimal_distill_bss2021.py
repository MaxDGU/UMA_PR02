#!/usr/bin/env python3
"""Per-cell BSS2021 accuracy: Qwen3-4B decimal distill vs human reference.

Reads per_cell.csv from the eval run and produces a publication-quality grouped
bar chart matching the project's plot style (viridis, sans-serif, paper context).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EVAL_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260518_134624_lora/bss2021_eval_20260520_124043")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = pd.read_csv(EVAL_DIR / "per_cell.csv")
    cell_order = [("Add", "EDD"), ("Add", "UDD"), ("Add", "D-W"),
                  ("Mul", "EDD"), ("Mul", "UDD"), ("Mul", "D-W")]
    labels = [f"{op}\n{od}" for op, od in cell_order]
    df = df.set_index(["operation", "operands"]).loc[cell_order].reset_index()

    model = df["acc_model"].values * 100
    human = df["acc_human"].values * 100
    mag_h = float((df["abs_gap"] * 100).mean())

    viridis = plt.cm.viridis(np.linspace(0.15, 0.7, 2))
    model_color, human_color = viridis[0], viridis[1]

    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    x = np.arange(len(cell_order))
    w = 0.38
    bars_m = ax.bar(x - w/2, model, w, color=model_color,
                    label="Qwen3-4B distill", edgecolor="white", linewidth=0.5)
    bars_h = ax.bar(x + w/2, human, w, color=human_color,
                    label="BSS2021 human", edgecolor="white", linewidth=0.5)

    for bars in (bars_m, bars_h):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              bbox_to_anchor=(1.0, 1.05))
    ax.text(0.98, 0.02, f"MAG-H = {mag_h:.1f} pp", transform=ax.transAxes,
            fontsize=8, color="black", va="bottom", ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()

    out_png = OUT_DIR / "qwen_decimal_distill_bss2021_per_cell.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")

    print("\n=== Per-cell breakdown ===")
    print(df[["operation", "operands", "acc_model", "acc_human", "abs_gap"]].to_string(index=False))
    print(f"\nMAG-H = {mag_h:.2f} pp")


if __name__ == "__main__":
    main()
