#!/usr/bin/env python3
"""MAE and NLL vs human, Qwen3-4B-Base vs Qwen3-4B-Instruct-2507 across the four
training stages, on SP2013 fractions and BSS2021 decimals. Values are the same
as Table 1 in the paper (post parser-fix on Instruct fractions)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

VARIANTS = ["Base", "+ humanFT", "+ distill", "+ distill + humanFT"]

# Table 1 numbers (Instruct fractions MAEs updated post parser-fix)
DATA = {
    "fractions": {
        "MAE": {"Base": [27.8, 26.2], "+ humanFT": [23.0, 22.5],
                "+ distill": [14.4, 13.0], "+ distill + humanFT": [5.2, 11.0]},
        "NLL": {"Base": [2.15, 2.61], "+ humanFT": [1.23, 1.11],
                "+ distill": [3.15, 3.69], "+ distill + humanFT": [1.37, 1.32]},
    },
    "decimals": {
        "MAE": {"Base": [34.3, 9.7], "+ humanFT": [6.2, 7.3],
                "+ distill": [18.3, 25.0], "+ distill + humanFT": [6.5, 5.7]},
        "NLL": {"Base": [2.44, 3.60], "+ humanFT": [0.33, 0.36],
                "+ distill": [2.29, 2.69], "+ distill + humanFT": [0.17, 0.17]},
    },
}

METRIC_YLABEL = {"MAE": "MAE (pp) $\\downarrow$",
                 "NLL": "NLL $\\downarrow$"}


def main():
    sns.set_style("ticks"); sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.6), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, 2))
    x = np.arange(len(VARIANTS))
    w = 0.38

    for i, dom in enumerate(["fractions", "decimals"]):
        for j, metric in enumerate(["MAE", "NLL"]):
            ax = axes[i, j]
            base_vals = [DATA[dom][metric][v][0] for v in VARIANTS]
            inst_vals = [DATA[dom][metric][v][1] for v in VARIANTS]
            b1 = ax.bar(x - w/2, base_vals, w, color=colors[0],
                        label="Qwen3-4B-Base", edgecolor="white", linewidth=0.4)
            b2 = ax.bar(x + w/2, inst_vals, w, color=colors[1],
                        label="Qwen3-4B-Instruct-2507", edgecolor="white", linewidth=0.4)
            for bar, val in list(zip(b1, base_vals)) + list(zip(b2, inst_vals)):
                ax.text(bar.get_x() + bar.get_width()/2, val + max(base_vals+inst_vals)*0.015,
                        f"{val:.2f}" if metric == "NLL" else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=6.5)
            ax.set_xticks(x)
            ax.set_xticklabels(VARIANTS, fontsize=7.5, rotation=15, ha="right")
            ax.set_ylabel(METRIC_YLABEL[metric], fontsize=8.5)
            ax.set_ylim(0, max(base_vals + inst_vals) * 1.20)
            sns.despine(ax=ax)
            if i == 0 and j == 0:
                ax.set_title(f"Fractions (SP2013)", fontsize=9.5, loc="left")
            if i == 0 and j == 1:
                ax.set_title(f"Fractions (SP2013)", fontsize=9.5, loc="left")
            if i == 1 and j == 0:
                ax.set_title(f"Decimals (BSS2021)", fontsize=9.5, loc="left")
            if i == 1 and j == 1:
                ax.set_title(f"Decimals (BSS2021)", fontsize=9.5, loc="left")

    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7.5)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR/"mae_nll_base_vs_instruct.png"
    plt.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
