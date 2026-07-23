#!/usr/bin/env python3
"""Overall raw-accuracy comparison: Human vs UMA vs Cognitive-LLM (ours) vs three
off-the-shelf frontier models, on SP2013 fractions and BSS2021 decimals.

Off-the-shelf frontier rollouts use the existing paper persona prompt
("You are a 7th grader solving {domain} arithmetic problems...")."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
PAPER_FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/emnlp-paper/figures")

# Cognitive-LLM (ours, Base 4B +distill+humanFT) rollouts
COG_FRAC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/fraction_4b/qwen3_4b_distill_humanft.csv"
COG_DEC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/decimal_4b/qwen3_4b_distill_humanft.csv"

# Off-the-shelf Qwen3-4B-Base (no LoRA)
QWEN_BASE_FRAC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/fraction_4b/qwen3_4b_base.csv"
QWEN_BASE_DEC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/decimal_4b/qwen3_4b_base.csv"

# Off-the-shelf Qwen3-4B-Instruct (no LoRA) rolled out with the model's native chat template
# — the bare-prompt version at memo_rollouts_4bi/ falls back to page reproduction and undercounts
# real Instruct competence.
QWEN_INSTRUCT_FRAC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/instruct_offshelf_chat_20260706_153630/instruct_4b_chat_fractions.csv"
QWEN_INSTRUCT_DEC  = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/instruct_offshelf_chat_20260706_153630/instruct_4b_chat_decimals.csv"

# Off-the-shelf frontier rollouts (paper persona)
FRONT_FRAC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/fraction_baselines_100samples"
FRONT_DEC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/persona_sweep_baselines/decimal"

# Human and UMA aggregate references
HUMAN_FRAC_CSV = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv"
UMA_FRAC_CSV = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/verification_full_results.csv"
DEC_HUMAN_PC = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_humanft_bss2021_from_distill_20260520_143350/bss2021_eval_20260520_144339/per_cell.csv"
DEC_UMA_SUMMARY = "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/uma_bss2021_summary.csv"

FRONTIER_MODELS = [("claude_sonnet_4_6", "Claude Sonnet 4.6"),
                   ("gemini_3_flash",    "Gemini 3 Flash"),
                   ("gpt_5_5_low",       "GPT-5.5 (low)")]


def acc(csv_path):
    df = pd.read_csv(csv_path)
    return float(df["is_correct"].mean())


def main():
    sns.set_style("ticks"); sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial","Helvetica","DejaVu Sans"]

    # Fractions overall acc references
    h = pd.read_csv(HUMAN_FRAC_CSV, low_memory=False).dropna(subset=["acc"])
    hf_overall = float(h["acc"].mean())
    u = pd.read_csv(UMA_FRAC_CSV, low_memory=False)
    uf_overall = float(u["acc"].mean())
    cog_frac = acc(COG_FRAC)
    qwen_base_frac = acc(QWEN_BASE_FRAC)
    qwen_instruct_frac = acc(QWEN_INSTRUCT_FRAC)
    front_frac = {m: acc(f"{FRONT_FRAC}/{slug}.csv") for slug, m in FRONTIER_MODELS}

    # Decimal references
    pc = pd.read_csv(DEC_HUMAN_PC).set_index(["operation","operands"])
    hd_overall = float((pc["acc_human"] * pc["n_human"]).sum() / pc["n_human"].sum())
    udec = pd.read_csv(DEC_UMA_SUMMARY)
    ud_overall = float((udec["acc"] * udec["n"]).sum() / udec["n"].sum())
    cog_dec = acc(COG_DEC)
    qwen_base_dec = acc(QWEN_BASE_DEC)
    qwen_instruct_dec = acc(QWEN_INSTRUCT_DEC)
    front_dec = {m: acc(f"{FRONT_DEC}/{slug}_paper_baseline.csv") for slug, m in FRONTIER_MODELS}

    # Build figure: 2 panels side by side
    bars_order = ["Human", "UMA", "Cognitive-LLM (ours)",
                  "Qwen3-4B-Base", "Qwen3-4B-Instruct",
                  "Claude Sonnet 4.6", "Gemini 3 Flash", "GPT-5.5 (low)"]
    frac_vals = [hf_overall, uf_overall, cog_frac,
                 qwen_base_frac, qwen_instruct_frac,
                 front_frac["Claude Sonnet 4.6"], front_frac["Gemini 3 Flash"], front_frac["GPT-5.5 (low)"]]
    dec_vals = [hd_overall, ud_overall, cog_dec,
                qwen_base_dec, qwen_instruct_dec,
                front_dec["Claude Sonnet 4.6"], front_dec["Gemini 3 Flash"], front_dec["GPT-5.5 (low)"]]

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(bars_order)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 2.8), dpi=300, sharey=True)
    x = np.arange(len(bars_order))
    for ax, vals, title in [(axL, frac_vals, "Fractions (SP2013)"),
                            (axR, dec_vals, "Decimals (BSS2021)")]:
        bars = ax.bar(x, [v*100 for v in vals], color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(bars_order, fontsize=7, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 105)
        ax.axhline(vals[0]*100, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v*100 + 1.5, f"{v*100:.0f}",
                    ha="center", va="bottom", fontsize=7)
        sns.despine(ax=ax)
    axL.set_ylabel("Overall accuracy (%)", fontsize=9)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR/"raw_accuracy_offshelf_vs_ours.png"
    plt.savefig(out, bbox_inches="tight")
    if PAPER_FIG_DIR.exists():
        plt.savefig(PAPER_FIG_DIR/"raw_accuracy_offshelf_vs_ours.png", bbox_inches="tight")

    print("wrote", out)
    print("\n=== Fractions overall accuracy ===")
    for n, v in zip(bars_order, frac_vals):
        print(f"  {n:25s}  {v*100:5.1f}%")
    print("\n=== Decimals overall accuracy ===")
    for n, v in zip(bars_order, dec_vals):
        print(f"  {n:25s}  {v*100:5.1f}%")


if __name__ == "__main__":
    main()
