#!/usr/bin/env python3
"""NLL vs decoding temperature for Qwen3-4B across the four training variants
and both domains. Two-panel figure; viridis palette."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/nll_tsweep_results.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
PAPER_FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/emnlp-paper/figures")

VARIANT_ORDER = ["Base", "+ humanFT", "+ distill", "+ distill + humanFT"]

def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = pd.read_csv(CSV)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(VARIANT_ORDER)))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=300, sharey=False)

    for ax, dom, title in [(axes[0], "fractions", "Fractions (SP2013)"),
                           (axes[1], "decimals", "Decimals (BSS2021)")]:
        sub = df[df["domain"] == dom]
        for v, c in zip(VARIANT_ORDER, colors):
            vsub = sub[sub["variant"] == v].sort_values("T")
            ax.plot(vsub["T"].values, vsub["mean_nll_per_token"].values,
                    marker="o", markersize=3.5, linewidth=1.2,
                    color=c, label=v)
        ax.set_xlabel("Decoding temperature $T$", fontsize=9)
        ax.set_title(title, fontsize=9)
        sns.despine(ax=ax)
    axes[0].set_ylabel("NLL of held-out human responses\n(nats / token, lower = better)", fontsize=9)
    axes[1].legend(loc="upper left", frameon=False, fontsize=7)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qwen3_4b_nll_temperature_curves.png"
    plt.savefig(out, bbox_inches="tight")
    if PAPER_FIG_DIR.exists():
        plt.savefig(PAPER_FIG_DIR / "qwen3_4b_nll_temperature_curves.png", bbox_inches="tight")
    print("wrote", out)
    for dom in ("fractions", "decimals"):
        print(f"\n[{dom}] argmin T per variant:")
        for v in VARIANT_ORDER:
            s = df[(df.domain == dom) & (df.variant == v)].sort_values("T")
            i = s["mean_nll_per_token"].idxmin()
            print(f"  {v:22s}  T*={s.loc[i,'T']:.1f}  NLL*={s.loc[i,'mean_nll_per_token']:.3f}")


if __name__ == "__main__":
    main()
