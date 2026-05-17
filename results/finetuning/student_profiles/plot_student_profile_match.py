#!/usr/bin/env python3
"""Plot Qwen3-4B student-conditioned outputs vs UMA per-profile targets on SP2013.

3 panels: low / mid / high. Each panel shows per-problem accuracy: UMA profile
target (light viridis bar) vs Qwen3-4B conditioned output (dark viridis bar).
Reports per-profile MAE pp in the panel title.

Inputs:
  --eval_dir   /scratch/.../qwen3_4b_studentcond_profile_evals_<RUN_TS>/
               (contains sp2013_per_problem_q3_4b_studentcond_<profile>.csv)
  --targets    data/student_profiles_4b_targets.csv  (per-profile per-problem
               UMA acc, from build_student_profiles.py)
Output: figures/qwen4b_student_profile_match.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SP2013_ORDER = [
    "3/5+1/5", "4/5+3/5", "3/5+1/4", "2/3+3/5",
    "3/5-1/5", "4/5-3/5", "3/5-1/4", "2/3-3/5",
    "3/5*1/5", "4/5*3/5", "3/5*1/4", "2/3*3/5",
    "3/5:1/5", "4/5:3/5", "3/5:1/4", "2/3:3/5",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=Path, required=True)
    p.add_argument("--targets", type=Path,
                   default=Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/"
                                "fractions/data/student_profiles_4b_targets.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/"
                                "fractions/figures/qwen4b_student_profile_match.png"))
    args = p.parse_args()

    targets = pd.read_csv(args.targets)

    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    profiles = ["low", "mid", "high"]
    light_c, dark_c = plt.cm.viridis([0.30, 0.75])
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), dpi=300, sharey=True)

    x = np.arange(len(SP2013_ORDER))
    width = 0.4
    for ax, prof in zip(axes, profiles):
        tgt = (targets[targets["profile"] == prof]
               .set_index("prob")
               .reindex(SP2013_ORDER)["acc"]
               .values.astype(float))
        model_csv = args.eval_dir / f"sp2013_per_problem_q3_4b_studentcond_{prof}.csv"
        model = (pd.read_csv(model_csv)
                 .set_index("prob")
                 .reindex(SP2013_ORDER)["acc_model"]
                 .values.astype(float))
        mae_pp = float(np.mean(np.abs(model - tgt))) * 100.0

        ax.bar(x - width/2, tgt, width=width, color=light_c, label="UMA profile target")
        ax.bar(x + width/2, model, width=width, color=dark_c, label="Qwen3-4B (cond.)")
        ax.set_xticks(x)
        ax.set_xticklabels(SP2013_ORDER, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{prof}-performer  (MAE = {mae_pp:.1f} pp)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax)

    axes[0].set_ylabel("SP2013 accuracy", fontsize=9)
    axes[-1].legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
