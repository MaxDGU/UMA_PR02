#!/usr/bin/env python3
"""Figure 12 (redone): persona descriptions vs UMA-parameter prefix, as sorted
curves (Figure 11 style, no shaded tier bands).

Two panels (Base, Instruct). In each: the UMA-parameter prefix produces a rising
curve across profiles (grey); the natural-language persona descriptions stay
nearly flat (coloured), sorted the same way. x-axis is percentile rank so the
~1000 UMA profiles and the ~15 personas share one axis.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# UMA-parameter per-profile curves (model accuracy under UMA prefix)
UMA_BASE = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/base_4b_per_profile_with_uma.csv")
UMA_INST = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/profile_ctrl_instruct_lr5e5ep3/rollouts_per_profile.csv")

BASE_LABEL = {
    "base_4b": "Qwen3-4B-Base + distill + humanFT",
    "instruct_4bi": "Qwen3-4B-Instruct-2507 + distill + humanFT",
}


def uma_sorted(path):
    df = pd.read_csv(path)
    v = np.sort(df["model_acc"].dropna().values) * 100
    x = np.linspace(0, 1, len(v))
    return x, v


def persona_sorted(summary_df, base):
    d = summary_df[summary_df["base"] == base].sort_values("acc")
    v = d["acc"].values * 100
    x = np.linspace(0, 1, len(v))
    return x, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-summary", required=True,
                    help="wide_persona_summary.csv (or plain_persona_summary.csv interim)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper-out", default=None)
    args = ap.parse_args()

    sns.set_style("ticks"); sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    persona = pd.read_csv(args.persona_summary)
    uma_paths = {"base_4b": UMA_BASE, "instruct_4bi": UMA_INST}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.0), dpi=300, sharey=True)
    persona_color = plt.cm.viridis(0.35)

    for ax, base in [(axL, "base_4b"), (axR, "instruct_4bi")]:
        xu, vu = uma_sorted(uma_paths[base])
        ax.plot(xu, vu, color="#888888", linewidth=1.4, label="UMA parameter prefix", zorder=1)
        xp, vp = persona_sorted(persona, base)
        ax.plot(xp, vp, "o-", color=persona_color, markersize=4, linewidth=1.2,
                label="Persona description", zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Learner rank (sorted by model accuracy)", fontsize=8)
        ax.set_title(BASE_LABEL[base], fontsize=9, loc="left")
        sns.despine(ax=ax)

    axL.set_ylabel("Accuracy on SP2013 (%)", fontsize=9)
    axL.legend(loc="upper left", frameon=False, fontsize=7.5)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print("wrote", out)
    if args.paper_out:
        plt.savefig(args.paper_out, bbox_inches="tight")
        print("wrote", args.paper_out)


if __name__ == "__main__":
    main()
