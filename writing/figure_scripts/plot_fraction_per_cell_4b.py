#!/usr/bin/env python3
"""Per-cell accuracy on SP2013 fractions: Human, UMA, and Qwen3-4B +distill+humanFT.

Eight operation-by-denominator cells (add/sub/mul/div x ED/UD). Three bars per
cell. Y axis is raw per-cell accuracy (proportion of correct responses).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HUMAN_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
UMA_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/verification_full_results.csv")
MODEL_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/max_uma_pr02/results/finetuning/trajectory_at_T10/rollouts/rollouts_4B.csv.gz")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
PAPER_FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/emnlp-paper/figures")

CELLS = [("add","ED"),("add","UD"),("sub","ED"),("sub","UD"),
         ("mul","ED"),("mul","UD"),("div","ED"),("div","UD")]
OPLAB = {"add":"Add","sub":"Sub","mul":"Mul","div":"Div"}
SEED_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/seed_cell_acc.csv")


def seed_sd(config):
    """Per-cell accuracy SD (pp) across the 5 humanFT seeds, ordered as CELLS."""
    s = pd.read_csv(SEED_CSV)
    s = s[(s.config == config) & (s.cell != "__overall__")]
    sd = s.groupby("cell")["acc"].std(ddof=1)
    keys = [f"{OPLAB[o]} {d}" for o, d in CELLS]
    return sd.reindex(keys).values


def human_acc():
    h = pd.read_csv(HUMAN_CSV, low_memory=False).dropna(subset=["acc"])
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    per_cell = h.groupby(["operation", "operands"])["acc"].mean()
    prob_cells = h.groupby("prob")[["operation", "operands"]].first().reset_index()
    return per_cell.reindex(CELLS).values, prob_cells


def uma_acc():
    u = pd.read_csv(UMA_CSV, low_memory=False)
    return u.groupby(["operation", "denoms"])["acc"].mean().reindex(CELLS).values


def model_acc(prob_cells, stage="hft_ep1"):
    r = pd.read_csv(MODEL_CSV, compression="gzip")
    r = r[r["stage"] == stage].copy()
    r["prob"] = r["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    r = r.merge(prob_cells, on="prob", how="left")
    return r.groupby(["operation", "operands"])["correct"].mean().reindex(CELLS).values


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    h_acc, prob_cells = human_acc()
    u_acc = uma_acc()
    m_acc = model_acc(prob_cells)
    series = [("Human (SP2013)", h_acc), ("UMA", u_acc), ("Cognitive-LLM (ours)", m_acc)]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(series)))
    labels = [f"{OPLAB[o]}\n{d}" for o, d in CELLS]

    fig, ax = plt.subplots(figsize=(6.8, 2.8), dpi=300)
    x = np.arange(len(CELLS))
    n = len(series)
    w = 0.78 / n
    offsets = (np.arange(n) - (n - 1) / 2) * w
    m_sd = seed_sd("base_frac")
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
    out = OUT_DIR / "qwen3_4b_fraction_per_cell.png"
    plt.savefig(out, bbox_inches="tight")
    if PAPER_FIG_DIR.exists():
        plt.savefig(PAPER_FIG_DIR / "qwen3_4b_fraction_per_cell.png", bbox_inches="tight")
    print("wrote", out)
    df = pd.DataFrame({lbl: v for lbl, v in series}, index=labels).round(3) * 100
    print("Per-cell accuracy (%):")
    print(df.round(1).to_string())


main()
