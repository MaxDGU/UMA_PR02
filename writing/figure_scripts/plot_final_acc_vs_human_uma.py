#!/usr/bin/env python3
"""Overall raw accuracy of the two post-trained (distill+humanFT) Cognitive-LLMs
against the Human and UMA references, decimals (left) and fractions (right).
Placed beside Table 1 in the paper.

Fraction rollouts are (re-)scored with the fixed mixed-number-aware
final-answer parser (tmp/rescore_frac_dir.py); decimal rollouts with an
analogous final-answer-region decimal parser using exact Decimal string
comparison (avoiding the float-comparison bug found in the ICL runner).
Human/UMA references use the same sources as Figure 1
(plot_offshelf_accuracy.py).
"""
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp")
from rescore_frac_dir import extract_final_answer, to_frac

JOB = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc")
EV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs")
NEW = JOB / "tmp/newckpt_rollouts_4bi"

COG_BASE_FRAC = EV / "fraction_4b/qwen3_4b_distill_humanft.csv"   # already fixed-parser scored
COG_INST_FRAC = NEW / "rollouts_4bi_fractions_distill_humanFT.csv"
COG_BASE_DEC = EV / "decimal_4b/qwen3_4b_distill_humanft.csv"
COG_INST_DEC = NEW / "rollouts_4bi_decimals_distill_humanFT.csv"

HUMAN_FRAC_CSV = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv"
UMA_FRAC_CSV = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/verification_full_results.csv"
DEC_HUMAN_PC = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_humanft_bss2021_from_distill_20260520_143350/bss2021_eval_20260520_144339/per_cell.csv"
DEC_UMA_SUMMARY = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/uma_bss2021_summary.csv"

OUT = JOB / "emnlp-paper/figures/final_acc_vs_human_uma.png"


def frac_acc(csv_path):
    df = pd.read_csv(csv_path)
    correct = df["correct_answer"].apply(to_frac)
    ok = [int(c is not None and c in extract_final_answer(r))
          for r, c in zip(df["model_response"], correct)]
    return float(np.mean(ok))


DEC_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def dec_candidates(text):
    if not isinstance(text, str):
        return set()
    m_mk = list(re.finditer(r"(?:###\s*answer|final\s+answer|answer)[:\s]", text, re.I))
    region = text[m_mk[-1].start():] if m_mk else text[-100:]
    out = set()
    for m in DEC_RE.finditer(region):
        try:
            out.add(Decimal(m.group(1)).normalize())
        except InvalidOperation:
            pass
    return out


def dec_acc(csv_path):
    df = pd.read_csv(csv_path)
    ok = []
    for r, c in zip(df["model_response"], df["correct_answer"]):
        try:
            target = Decimal(str(c)).normalize()
        except InvalidOperation:
            ok.append(0)
            continue
        ok.append(int(target in dec_candidates(r)))
    return float(np.mean(ok))


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    h = pd.read_csv(HUMAN_FRAC_CSV, low_memory=False).dropna(subset=["acc"])
    human_frac = float(h["acc"].mean())
    u = pd.read_csv(UMA_FRAC_CSV, low_memory=False)
    uma_frac = float(u["acc"].mean())
    pc = pd.read_csv(DEC_HUMAN_PC).set_index(["operation", "operands"])
    human_dec = float((pc["acc_human"] * pc["n_human"]).sum() / pc["n_human"].sum())
    ud = pd.read_csv(DEC_UMA_SUMMARY)
    uma_dec = float((ud["acc"] * ud["n"]).sum() / ud["n"].sum())

    base_frac, inst_frac = frac_acc(COG_BASE_FRAC), frac_acc(COG_INST_FRAC)
    base_dec, inst_dec = dec_acc(COG_BASE_DEC), dec_acc(COG_INST_DEC)

    labels = ["Human", "UMA", "Ours (Base)", "Ours (Inst.)"]
    dec_vals = [human_dec, uma_dec, base_dec, inst_dec]
    frac_vals = [human_frac, uma_frac, base_frac, inst_frac]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    # Overall-accuracy SD across the 5 humanFT seeds (whiskers on the Ours bars).
    sv = pd.read_csv(JOB / "tmp/seed_variance/seed_cell_acc.csv")
    sv = sv[sv.cell == "__overall__"]

    def sd(config):
        return float(sv[sv.config == config]["acc"].std(ddof=1))

    dec_err = [np.nan, np.nan, sd("base_dec"), sd("inst_dec")]
    frac_err = [np.nan, np.nan, sd("base_frac"), sd("inst_frac")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(3.4, 2.35), dpi=300, sharey=True)
    x = np.arange(len(labels))
    for ax, vals, err, title in [(axL, dec_vals, dec_err, "Decimals"),
                                 (axR, frac_vals, frac_err, "Fractions")]:
        bars = ax.bar(x, [v * 100 for v in vals], color=colors,
                      edgecolor="white", linewidth=0.4,
                      yerr=err, error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="0.25"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=38, ha="right")
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.axhline(vals[0] * 100, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        for bar, v, e in zip(bars, vals, err):
            lift = 0.0 if np.isnan(e) else e
            ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + lift + 1.5,
                    f"{v*100:.0f}", ha="center", va="bottom", fontsize=6.5)
        sns.despine(ax=ax)
    axL.set_ylabel("Overall accuracy (%)", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures/final_acc_vs_human_uma.png",
                bbox_inches="tight")
    print("wrote", OUT)
    for lab, d, f in zip(labels, dec_vals, frac_vals):
        print(f"{lab:14s} dec={d*100:5.1f}%  frac={f*100:5.1f}%")


if __name__ == "__main__":
    main()
