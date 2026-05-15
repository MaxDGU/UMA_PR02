#!/usr/bin/env python3
"""Characterize the parser drift by re-grading the staged rollouts with both
the OLD (buggy) parser and the NEW (mixed-number/decimal-aware) parser.

Output:
  - figures/qwen_trajectory_parser_comparison.csv
  - figures/qwen_trajectory_parser_comparison.png  (mirror of qwen_trajectory_at_T10.png
    with two lines per size: new parser solid, old parser dashed)
  - figures/parser_drift_breakdown.csv (per-format diagnosis)
"""
from __future__ import annotations

import gzip
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from eval_qwen3_chained_sp2013 import extract_answer as new_extract_answer
from eval_qwen3_chained_sp2013 import gt_answer

ROLLOUTS_DIR = Path("/tmp/uma_pr02_max2/results/finetuning/trajectory_at_T10/rollouts")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

SIZE_FILES = [("0.6B", "rollouts_0p6B.csv.gz"),
              ("1.7B", "rollouts_1p7B.csv.gz"),
              ("4B",   "rollouts_4B.csv.gz"),
              ("8B",   "rollouts_8B.csv.gz")]
STAGES = ["base", "distill_ep1", "hft_ep1", "hft_ep2", "hft_ep3", "hft_ep5"]
STAGE_LABELS = ["pretrained\nbase", "+ distill\nep1", "+ humanFT\nep1",
                "+ humanFT\nep2", "+ humanFT\nep3", "+ humanFT\nep5"]


# ---- Verbatim copy of the OLD parser (pre-refactor) for re-grading ----
_OLD_ANS_TAG_RE = re.compile(r"###\s*answer\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)
_OLD_FRAC_RE = re.compile(r"-?\d+\s*/\s*-?\d+")


def old_parse_frac(s: str) -> Optional[Fraction]:
    s = s.strip().split()[0] if s.strip() else ""
    s = s.rstrip(".,")
    try:
        if "/" in s:
            n, d = s.split("/", 1)
            return Fraction(int(n.strip()), int(d.strip()))
        return Fraction(int(s))
    except (ValueError, ZeroDivisionError):
        return None


def old_extract_answer(text: str) -> Optional[Fraction]:
    m = _OLD_ANS_TAG_RE.search(text)
    if m is not None:
        candidate = m.group(1).strip()
        f = old_parse_frac(candidate)
        if f is not None:
            return f
    fracs = _OLD_FRAC_RE.findall(text)
    if fracs:
        return old_parse_frac(fracs[-1])
    return None


def classify_answer_format(text: str) -> str:
    """Look at the substring after '### answer:' (or first 80 chars) and tag the
    answer format the model emitted."""
    m = _OLD_ANS_TAG_RE.search(text)
    cand = m.group(1).strip() if m else text[:80]
    cand = cand.strip().split()[0:3]
    cand_str = " ".join(cand)
    if re.match(r"^-?\d+\s+\d+\s*/\s*\d+", cand_str):
        return "mixed_number"
    if re.match(r"^-?\d+\.\d+", cand_str):
        return "decimal"
    if re.match(r"^-?\d+\s*/\s*-?\d+", cand_str):
        return "plain_fraction"
    if re.match(r"^-?\d+\b", cand_str):
        return "integer"
    return "other"


def load_human() -> dict:
    p = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
    h = pd.read_csv(p, low_memory=False).dropna(subset=["acc"])
    h["prob_uma"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    return h.groupby("prob_uma")["acc"].mean().to_dict()


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    human_acc = load_human()
    all_rows = []
    drift_rows = []
    fmt_rows = []
    for size, fname in SIZE_FILES:
        p = ROLLOUTS_DIR / fname
        df = pd.read_csv(p, compression="gzip")
        gts = {pr: gt_answer(pr) for pr in df["prob"].unique()}
        # Re-grade with both parsers (rollouts identical, only parser differs).
        new_correct, old_correct, fmts = [], [], []
        for _, row in df.iterrows():
            t = str(row["raw_text"])
            gt = gts[row["prob"]]
            new_p = new_extract_answer(t)
            old_p = old_extract_answer(t)
            new_correct.append(int(new_p is not None and new_p == gt))
            old_correct.append(int(old_p is not None and old_p == gt))
            fmts.append(classify_answer_format(t))
        df["correct_new"] = new_correct
        df["correct_old"] = old_correct
        df["ans_fmt"] = fmts

        # Per-(stage, prob) accuracy under each parser.
        for stage in STAGES:
            sub = df[df["stage"] == stage]
            if sub.empty:
                continue
            per_prob = (sub.groupby(["prob", "op"], as_index=False)
                          .agg(n=("correct_new", "size"),
                               c_new=("correct_new", "sum"),
                               c_old=("correct_old", "sum")))
            per_prob["acc_new"] = per_prob["c_new"] / per_prob["n"]
            per_prob["acc_old"] = per_prob["c_old"] / per_prob["n"]
            per_prob["acc_h"] = per_prob["prob"].map(human_acc)
            per_prob["err_new"] = (per_prob["acc_new"] - per_prob["acc_h"]).abs() * 100
            per_prob["err_old"] = (per_prob["acc_old"] - per_prob["acc_h"]).abs() * 100
            mae_new = per_prob["err_new"].mean()
            mae_old = per_prob["err_old"].mean()
            all_rows.append({"size": size, "stage": stage,
                             "MAE_new": mae_new, "MAE_old": mae_old,
                             "delta": mae_new - mae_old,
                             "acc_new": per_prob["acc_new"].mean(),
                             "acc_old": per_prob["acc_old"].mean()})
            # Drift breakdown: where do old and new disagree?
            disagree = sub[sub["correct_new"] != sub["correct_old"]]
            drift_rows.append({"size": size, "stage": stage,
                               "n_rollouts": len(sub),
                               "n_disagree": len(disagree),
                               "n_new_only": ((sub["correct_new"] == 1) & (sub["correct_old"] == 0)).sum(),
                               "n_old_only": ((sub["correct_new"] == 0) & (sub["correct_old"] == 1)).sum(),
                               "pct_disagree": 100 * len(disagree) / max(len(sub), 1)})
            # Format breakdown
            for fmt, fsub in sub.groupby("ans_fmt"):
                fmt_rows.append({"size": size, "stage": stage, "fmt": fmt,
                                 "n": len(fsub),
                                 "frac_new_correct": fsub["correct_new"].mean(),
                                 "frac_old_correct": fsub["correct_old"].mean()})

    main_df = pd.DataFrame(all_rows)
    drift_df = pd.DataFrame(drift_rows)
    fmt_df = pd.DataFrame(fmt_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(OUT_DIR / "qwen_trajectory_parser_comparison.csv", index=False)
    drift_df.to_csv(OUT_DIR / "parser_drift_breakdown.csv", index=False)
    fmt_df.to_csv(OUT_DIR / "parser_drift_format_breakdown.csv", index=False)

    print("\n=== MAE pp under each parser (same rollouts) ===")
    pivot = main_df.pivot_table(index="stage", columns="size", values=["MAE_new", "MAE_old"]).reindex(STAGES)
    print(pivot.to_string())
    print("\n=== Drift breakdown (rollout-level disagreement under the two parsers) ===")
    pivot2 = drift_df.pivot_table(index="stage", columns="size", values="pct_disagree").reindex(STAGES)
    print(pivot2.round(2).to_string())
    print("\n=== Where the new parser catches answers the old parser missed (per format) ===")
    print(fmt_df.groupby("fmt").agg(n=("n","sum"),
                                    pct_new_right=("frac_new_correct", "mean"),
                                    pct_old_right=("frac_old_correct", "mean")).round(3).to_string())

    # ---- Plot mirror of qwen_trajectory_at_T10.png ----
    sizes = ["0.6B", "1.7B", "4B", "8B"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    color_for = dict(zip(sizes, colors))

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=300)
    x = np.arange(len(STAGES))
    for size in sizes:
        sub = main_df[main_df["size"] == size].set_index("stage").reindex(STAGES)
        y_new = sub["MAE_new"].values.astype(float)
        y_old = sub["MAE_old"].values.astype(float)
        valid_n = ~np.isnan(y_new)
        valid_o = ~np.isnan(y_old)
        ax.plot(x[valid_n], y_new[valid_n], marker="o", markersize=6, linewidth=2.0,
                color=color_for[size], label=f"Qwen3-{size} (fixed parser)")
        ax.plot(x[valid_o], y_old[valid_o], marker="s", markersize=4, linewidth=1.2,
                color=color_for[size], linestyle="--", alpha=0.7,
                label=f"Qwen3-{size} (old parser)")
        for xi, vn, vo in zip(x[valid_n], y_new[valid_n], y_old[valid_o]):
            if not np.isnan(vn):
                ax.annotate(f"{vn:.1f}", (xi, vn), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=7, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("MAE pp vs human SP2013 (T=1.0)", fontsize=9)
    ax.set_title("Parser-fix drift in trajectory MAE (same rollouts, only parser differs)",
                 fontsize=10, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    out_png = OUT_DIR / "qwen_trajectory_parser_comparison.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
