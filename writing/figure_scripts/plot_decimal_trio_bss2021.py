#!/usr/bin/env python3
"""Per-cell accuracy on BSS2021 decimals: Human, UMA, and Qwen3-4B +distill+humanFT.

Six operation-by-operand-type cells (Add/Mul x EDD/UDD/D-W). Three bars per
cell. Y axis is raw per-cell accuracy.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DISTILL_HFT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_humanft_bss2021_from_distill_20260520_143350/bss2021_eval_20260520_144339")
UMA_SUMMARY = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/uma_bss2021_summary.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
PAPER_FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/emnlp-paper/figures")

CELLS = [("Add", "EDD"), ("Add", "UDD"), ("Add", "D-W"),
         ("Mul", "EDD"), ("Mul", "UDD"), ("Mul", "D-W")]
SEED_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/seed_cell_acc.csv")


def seed_sd(config):
    """Per-cell accuracy SD (pp) across the 5 humanFT seeds, ordered as CELLS."""
    s = pd.read_csv(SEED_CSV)
    s = s[(s.config == config) & (s.cell != "__overall__")]
    sd = s.groupby("cell")["acc"].std(ddof=1)
    keys = [f"{op} {od.upper()}" for op, od in CELLS]
    return sd.reindex(keys).values


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    per_cell = pd.read_csv(DISTILL_HFT_DIR / "per_cell.csv")
    per_cell = per_cell.set_index(["operation", "operands"]).loc[CELLS].reset_index()
    h_acc = per_cell["acc_human"].values
    m_acc = per_cell["acc_model"].values

    u = pd.read_csv(UMA_SUMMARY)
    uma_per_cell = u.groupby(["operation", "operands"]).apply(
        lambda x: (x["acc"] * x["n"]).sum() / x["n"].sum()
    )
    u_acc = uma_per_cell.reindex(CELLS).values

    series = [("Human (BSS2021)", h_acc), ("UMA", u_acc), ("Cognitive-LLM (ours)", m_acc)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(series)))
    labels = [f"{op}\n{od}" for op, od in CELLS]

    fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=300)
    x = np.arange(len(CELLS))
    n = len(series)
    w = 0.78 / n
    offsets = (np.arange(n) - (n - 1) / 2) * w
    m_sd = seed_sd("base_dec")
    for off, (lbl, v), c in zip(offsets, series, colors):
        yerr = m_sd if lbl.startswith("Cognitive-LLM") else None
        ax.bar(x + off, v * 100, w, color=c, label=lbl,
               edgecolor="white", linewidth=0.4,
               yerr=yerr, error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="0.25"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              bbox_to_anchor=(1.0, 1.05))
    sns.despine(ax=ax)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qwen_decimal_base_distill_humanft_bss2021_per_cell.png"
    plt.savefig(out, bbox_inches="tight")
    if PAPER_FIG_DIR.exists():
        plt.savefig(PAPER_FIG_DIR / "qwen_decimal_base_distill_humanft_bss2021_per_cell.png", bbox_inches="tight")
    print("wrote", out)
    df = pd.DataFrame({lbl: v for lbl, v in series}, index=labels) * 100
    print("Per-cell accuracy (%):")
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()
