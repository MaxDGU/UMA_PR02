#!/usr/bin/env python3
"""Mean SP2013 accuracy vs sampling temperature for Qwen3-4B student-conditioned
distill, broken out by profile (low/mid/high). UMA per-profile targets drawn
as horizontal dashed lines.

Reveals where the conditioning signal lives in T-space: spread peaks at
T~1.5-1.7, collapses at low T (model picks correct mode regardless of profile)
and degrades at very high T (sampling noise dominates).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SP2013 = ['3/5+1/5','4/5+3/5','3/5+1/4','2/3+3/5',
          '3/5-1/5','4/5-3/5','3/5-1/4','2/3-3/5',
          '3/5*1/5','4/5*3/5','3/5*1/4','2/3*3/5',
          '3/5:1/5','4/5:3/5','3/5:1/4','2/3:3/5']
EVAL_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_4b_studentcond_profile_evals_20260517_133445_tsweep")
TARGETS = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/student_profiles_4b_targets.csv")
OUT = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures/qwen4b_studentcond_t_sweep.png")

TS = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0]
PROFILES = ["low", "mid", "high"]


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    targets = pd.read_csv(TARGETS)
    target_mean = {p: targets[targets["profile"] == p]["acc"].mean() for p in PROFILES}

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(PROFILES)))
    color_for = dict(zip(PROFILES, colors))

    rows = []
    for prof in PROFILES:
        for T in TS:
            t_tag = f"{int(round(T*10)):02d}"
            f = EVAL_DIR / f"sp2013_per_problem_q3_4b_sc_{prof}_t{t_tag}.csv"
            m = pd.read_csv(f).set_index("prob").reindex(SP2013)["acc_model"].values
            rows.append({"profile": prof, "T": T, "model_mean_acc": float(m.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    print(df.pivot(index="T", columns="profile", values="model_mean_acc").to_string())

    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=300)
    for prof in PROFILES:
        sub = df[df["profile"] == prof]
        ax.plot(sub["T"], sub["model_mean_acc"], marker="o", markersize=5,
                linewidth=2.0, color=color_for[prof], label=f"{prof}")
        ax.axhline(target_mean[prof], linestyle="--", linewidth=1.0,
                   color=color_for[prof], alpha=0.6)
        ax.annotate(f"UMA {prof}={target_mean[prof]:.2f}",
                    xy=(2.0, target_mean[prof]), xytext=(4, 1),
                    textcoords="offset points",
                    fontsize=7, color=color_for[prof], alpha=0.8)
    ax.set_xlabel("sampling temperature T", fontsize=9)
    ax.set_ylabel("SP2013 mean accuracy", fontsize=9)
    ax.set_xticks(TS)
    ax.set_xticklabels([f"{t:.1f}" for t in TS], fontsize=8)
    ax.set_ylim(0.30, 0.80)
    ax.legend(loc="lower left", frameon=False, fontsize=9, title="profile",
              title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
