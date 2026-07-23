#!/usr/bin/env python3
"""Two figures showing that varying UMA learner parameters at prompt time
controls the accuracy of our distilled LLM across the low/medium/high
performance bands defined by UMA's calibration on SP2013.

Design A: 4-panel marginal sweep (one panel per parameter).
Design B: Single sorted-profile diagonal plot.

Uses the per_sample.csv from the distill checkpoint profile-spread run.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


import sys
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/profile_spread_distill_full1000_20260521_115255")
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# UMA tier accuracies on SP2013 (from per-tier figure)
UMA_LOW, UMA_MID, UMA_HIGH = 0.34, 0.54, 0.71

# Tier bands centered on UMA references
LOW_HI = (UMA_LOW + UMA_MID) / 2     # 0.44
MID_HI = (UMA_MID + UMA_HIGH) / 2    # 0.625

PARAM_LABELS = {
    "g":   ("guess rate", "how often the learner gives up and guesses"),
    "d":   ("memory", "how reliably they remember intermediate steps"),
    "rt":  ("retrieval speed", "how quickly they reach for a memorized answer"),
    "ice": ("misreading", "how often they misread the problem"),
}


def load_per_profile():
    df = pd.read_csv(SRC / "per_sample.csv", low_memory=False)
    per_profile = (df.groupby(["g", "d", "rt", "ice"])
                     .agg(acc=("is_correct", "mean"), n=("is_correct", "size"))
                     .reset_index())
    return per_profile


def add_tier_bands(ax, x_extent):
    # Single-hue Blues gradient: low = light, mid = medium, high = dark.
    # Same family throughout for an easy-on-eyes gradient.
    cmap = plt.cm.Blues
    ax.axhspan(0.0, LOW_HI, facecolor=cmap(0.25), alpha=0.45, lw=0, zorder=0)
    ax.axhspan(LOW_HI, MID_HI, facecolor=cmap(0.55), alpha=0.45, lw=0, zorder=0)
    ax.axhspan(MID_HI, 1.0, facecolor=cmap(0.85), alpha=0.45, lw=0, zorder=0)
    for y in (UMA_LOW, UMA_MID, UMA_HIGH):
        ax.axhline(y, color="#333333", lw=0.7, alpha=0.55, linestyle="--", zorder=1)


def annotate_tiers(ax, x_pos, ha="left"):
    cmap = plt.cm.Blues
    ax.text(x_pos, UMA_HIGH + 0.05, "high-performing", fontsize=8, ha=ha, va="bottom", color=cmap(0.95))
    ax.text(x_pos, UMA_MID + 0.05, "medium-performing", fontsize=8, ha=ha, va="bottom", color=cmap(0.75))
    ax.text(x_pos, UMA_LOW + 0.05, "low-performing", fontsize=8, ha=ha, va="bottom", color=cmap(0.55))


# --- Design A: 4-panel marginal sweep ----------------------------------

def design_a(per_profile: pd.DataFrame, out_png: Path):
    sns.set_style("ticks"); sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.4), dpi=300, sharey=True)
    for ax, param in zip(axes, ["g", "d", "rt", "ice"]):
        marginal = (per_profile.groupby(param)["acc"]
                    .agg(["mean", "std", "count"]).reset_index())
        x = marginal[param].astype(float).values
        y = marginal["mean"].values
        # bootstrap-style err: use std across other params at this value
        err = (marginal["std"] / np.sqrt(marginal["count"])).values
        add_tier_bands(ax, x)
        # Color points by parameter value using viridis
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(x)))
        for i in range(len(x)):
            ax.errorbar(x[i], y[i], yerr=err[i], fmt="o", color=colors[i],
                        markersize=5, ecolor="black", elinewidth=0.8, capsize=2, zorder=3)
        ax.plot(x, y, color="black", lw=0.8, alpha=0.5, zorder=2)
        ax.set_xlabel(PARAM_LABELS[param][0] + f" ($\\!{param}\\!$)", fontsize=8)
        ax.set_xticks(x)
        if param == "ice":
            ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=7)
        elif param == "rt":
            ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=7)
        else:
            ax.set_xticklabels([f"{v:.2g}" for v in x], fontsize=7)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)
    axes[0].set_ylabel("model accuracy", fontsize=9)
    # Tier labels only on right side of last panel
    annotate_tiers(axes[-1], axes[-1].get_xlim()[1] * 1.02, ha="left")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


# --- Design B: single sorted-diagonal plot -----------------------------

def design_b(per_profile: pd.DataFrame, out_png: Path):
    sns.set_style("ticks"); sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df_full = per_profile.sort_values("acc").reset_index(drop=True)
    n = len(df_full)
    # Subsample for visual cleanliness: many profiles share the same accuracy
    # (since acc = correct / 16 is granular), which makes overlapping dots look
    # like solid bars. ~90 dots with vertical jitter reads as individual
    # profiles while preserving the sorted-diagonal shape.
    n_show = min(90, n)
    idx = np.linspace(0, n - 1, n_show).round().astype(int)
    df = df_full.iloc[idx].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=300)
    add_tier_bands(ax, (0, n_show))

    ax.scatter(np.arange(n_show), df["acc"].values,
               color="#d95f02", s=22,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xlim(0, n_show - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("UMA learner profiles, sorted by model accuracy", fontsize=9)
    ax.set_ylabel("model accuracy (SP2013)", fontsize=9)

    # Tier labels on the right edge
    cmap = plt.cm.Blues
    ax.text(n_show * 1.005, UMA_HIGH + 0.04, "high-performing", fontsize=8, ha="left", color=cmap(0.95))
    ax.text(n_show * 1.005, UMA_MID + 0.04, "medium-performing", fontsize=8, ha="left", color=cmap(0.75))
    ax.text(n_show * 1.005, UMA_LOW + 0.04, "low-performing", fontsize=8, ha="left", color=cmap(0.55))

    # Annotate the two extremes with their g, d, rt parameter values (from full data)
    lo, hi = df_full.iloc[0], df_full.iloc[-1]
    ax.annotate(f"$g$={lo['g']:.2g}, $d$={lo['d']:.1g}, $rt$={int(lo['rt'])}",
                xy=(0, lo["acc"]), xytext=(8, -22),
                textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6))
    ax.annotate(f"$g$={hi['g']:.2g}, $d$={hi['d']:.1g}, $rt$={int(hi['rt'])}",
                xy=(n_show - 1, hi["acc"]), xytext=(-80, 14),
                textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6))

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    per_profile = load_per_profile()
    print(f"n_profiles={len(per_profile)}, "
          f"acc min={per_profile['acc'].min():.3f} "
          f"median={per_profile['acc'].median():.3f} "
          f"max={per_profile['acc'].max():.3f}")
    suffix = f"_{TAG}" if TAG else ""
    design_a(per_profile, OUT_DIR / f"profile_controllability_a_marginals{suffix}.png")
    design_b(per_profile, OUT_DIR / f"profile_controllability_b_diagonal{suffix}.png")


if __name__ == "__main__":
    main()
