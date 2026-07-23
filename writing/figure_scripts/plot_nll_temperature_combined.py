#!/usr/bin/env python3
"""Combined NLL-vs-T figure: 2x2 grid, rows = base model, columns = domain."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV_BASE = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/nll_tsweep_results.csv")
CSV_INSTRUCT = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/nll_tsweep_instruct_results.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
PAPER_FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/emnlp-paper/figures")

VARIANT_ORDER = ["Base", "+ humanFT", "+ distill", "+ distill + humanFT"]


def main():
    sns.set_style("ticks"); sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df_base = pd.read_csv(CSV_BASE).assign(base="Qwen3-4B-Base")
    df_inst = pd.read_csv(CSV_INSTRUCT).assign(base="Qwen3-4B-Instruct-2507")
    df = pd.concat([df_base, df_inst], ignore_index=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(VARIANT_ORDER)))

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), dpi=300, sharex=True)
    base_models = ["Qwen3-4B-Base", "Qwen3-4B-Instruct-2507"]
    domains = [("fractions", "Fractions (SP2013)"), ("decimals", "Decimals (BSS2021)")]

    for i, base in enumerate(base_models):
        for j, (dom, title) in enumerate(domains):
            ax = axes[i, j]
            sub = df[(df["base"] == base) & (df["domain"] == dom)]
            for v, c in zip(VARIANT_ORDER, colors):
                vsub = sub[sub["variant"] == v].sort_values("T")
                ax.plot(vsub["T"].values, vsub["mean_nll_per_token"].values,
                        marker="o", markersize=3.0, linewidth=1.1, color=c, label=v)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if i == 1:
                ax.set_xlabel("Decoding temperature $T$", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{base.split('-')[-1] if 'Instruct' not in base else 'Instruct'}\nNLL (nats / token)",
                              fontsize=9)
            sns.despine(ax=ax)

    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=7, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qwen3_4b_nll_temperature_combined.png"
    plt.savefig(out, bbox_inches="tight")
    if PAPER_FIG_DIR.exists():
        plt.savefig(PAPER_FIG_DIR / "qwen3_4b_nll_temperature_combined.png", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
