#!/usr/bin/env python3
"""Clean mirror of qwen_trajectory_at_T10.png using the new (correct) parser.
MAE pp from the rollouts at /tmp/uma_pr02_max2/results/finetuning/trajectory_at_T10/rollouts/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SRC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures/qwen_trajectory_parser_comparison.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

STAGES = ["base", "distill_ep1", "hft_ep1", "hft_ep2", "hft_ep3", "hft_ep5"]
STAGE_LABELS = ["pretrained\nbase", "+ distill\nep1", "+ humanFT\nep1",
                "+ humanFT\nep2", "+ humanFT\nep3", "+ humanFT\nep5"]


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = pd.read_csv(SRC)[["size", "stage", "MAE_new"]].rename(columns={"MAE_new": "MAE_pp"})
    pivot = df.pivot_table(index="stage", columns="size", values="MAE_pp").reindex(STAGES)
    print(pivot.to_string())

    sizes = ["0.6B", "1.7B", "4B", "8B"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    color_for = dict(zip(sizes, colors))

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    x = np.arange(len(STAGES))
    for size in sizes:
        sub = df[df["size"] == size].set_index("stage").reindex(STAGES)
        y = sub["MAE_pp"].values.astype(float)
        valid = ~np.isnan(y)
        ax.plot(x[valid], y[valid], marker="o", markersize=6, linewidth=2.0,
                color=color_for[size], label=f"Qwen3-{size}")
        for xi, v in zip(x[valid], y[valid]):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("MAE pp vs human SP2013 (T=1.0, fixed parser)", fontsize=9)
    ax.set_title("Trajectory: pretrained → distill → humanFT, at fixed T=1.0 [fixed parser]",
                 fontsize=10, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    out_png = OUT_DIR / "qwen_trajectory_at_T10_fixed_parser.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
