#!/usr/bin/env python3
"""Per-cell accuracy comparison: 4B Base vs 4B Instruct, both as Cognitive-LLM.

Side-by-side panels for fractions and decimals, four bars per cell:
Human reference, UMA, Cognitive-LLM (Base 4B), Cognitive-LLM (Instruct 4B)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INSTRUCT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/newckpt_rollouts_4bi")
BASE_FRAC_ROLLOUTS = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/fraction_4b/qwen3_4b_distill_humanft.csv")
BASE_DEC_ROLLOUTS = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/decimal_4b/qwen3_4b_distill_humanft.csv")
HUMAN_FRAC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
UMA_FRAC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/verification_full_results.csv")
DEC_BSS_SUMMARY = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/uma_bss2021_summary.csv")
DEC_HUMAN_PER_CELL = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_humanft_bss2021_from_distill_20260520_143350/bss2021_eval_20260520_144339/per_cell.csv")

OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")
PAPER_FIG_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/emnlp-paper/figures")

FRAC_CELLS = [("add","ED"),("add","UD"),("sub","ED"),("sub","UD"),
              ("mul","ED"),("mul","UD"),("div","ED"),("div","UD")]
FRAC_LABELS = {"add":"Add","sub":"Sub","mul":"Mul","div":"Div"}
DEC_CELLS = [("Add","EDD"),("Add","UDD"),("Add","D-W"),
             ("Mul","EDD"),("Mul","UDD"),("Mul","D-W")]
SEED_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/seed_cell_acc.csv")


def seed_sd(config, cells):
    """Per-cell accuracy SD (pp) across the 5 humanFT seeds."""
    s = pd.read_csv(SEED_CSV)
    s = s[(s.config == config) & (s.cell != "__overall__")]
    sd = s.groupby("cell")["acc"].std(ddof=1)
    keys = [f"{str(op).capitalize()[:3]} {od.upper()}" if config.endswith("frac")
            else f"{op} {od.upper()}" for op, od in cells]
    return sd.reindex(keys).values


def frac_human_uma():
    h = pd.read_csv(HUMAN_FRAC, low_memory=False).dropna(subset=["acc"])
    human = h.groupby(["operation","operands"])["acc"].mean()
    u = pd.read_csv(UMA_FRAC, low_memory=False)
    uma = u.groupby(["operation","denoms"])["acc"].mean()
    return human.reindex(FRAC_CELLS).values, uma.reindex(FRAC_CELLS).values


def dec_human_uma():
    pc = pd.read_csv(DEC_HUMAN_PER_CELL).set_index(["operation","operands"]).loc[DEC_CELLS]
    human = pc["acc_human"].values
    u = pd.read_csv(DEC_BSS_SUMMARY)
    uma = u.groupby(["operation","operands"]).apply(
        lambda x: (x["acc"]*x["n"]).sum()/x["n"].sum()
    ).reindex(DEC_CELLS).values
    return human, uma


def cell_acc(csv_path, cells, op_col="operation", od_col="operands", lowercase=False):
    df = pd.read_csv(csv_path)
    if lowercase:
        df[op_col] = df[op_col].str.lower()
    g = df.groupby([op_col, od_col])["is_correct"].mean()
    return g.reindex(cells).values


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial","Helvetica","DejaVu Sans"]

    # Fractions
    h_f, u_f = frac_human_uma()
    base_frac = cell_acc(BASE_FRAC_ROLLOUTS, [(o.capitalize(), d) for o,d in FRAC_CELLS],
                          op_col="operation", od_col="denom_type")
    instruct_frac = cell_acc(INSTRUCT_DIR/"rollouts_4bi_fractions_distill_humanFT.csv",
                              [(o.capitalize(), d) for o,d in FRAC_CELLS],
                              op_col="operation", od_col="operands")

    # Decimals
    h_d, u_d = dec_human_uma()
    base_dec = cell_acc(BASE_DEC_ROLLOUTS, DEC_CELLS,
                         op_col="operation", od_col="operands")
    instruct_dec = cell_acc(INSTRUCT_DIR/"rollouts_4bi_decimals_distill_humanFT.csv",
                             DEC_CELLS, op_col="operation", od_col="operands")

    series_frac = [
        ("Human (SP2013)", h_f),
        ("UMA", u_f),
        ("Cognitive-LLM (4B Base)", base_frac),
        ("Cognitive-LLM (4B Instruct)", instruct_frac),
    ]
    series_dec = [
        ("Human (BSS2021)", h_d),
        ("UMA", u_d),
        ("Cognitive-LLM (4B Base)", base_dec),
        ("Cognitive-LLM (4B Instruct)", instruct_dec),
    ]

    colors = plt.cm.viridis(np.linspace(0.10, 0.90, 4))
    frac_xlabels = [f"{FRAC_LABELS[o]}\n{d}" for o,d in FRAC_CELLS]
    dec_xlabels = [f"{op}\n{od}" for op, od in DEC_CELLS]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.0), dpi=300,
                                    gridspec_kw={"width_ratios":[8,6]})

    sd_map = {
        "Fractions (SP2013)": {"Cognitive-LLM (4B Base)": seed_sd("base_frac", FRAC_CELLS),
                               "Cognitive-LLM (4B Instruct)": seed_sd("inst_frac", FRAC_CELLS)},
        "Decimals (BSS2021)": {"Cognitive-LLM (4B Base)": seed_sd("base_dec", DEC_CELLS),
                               "Cognitive-LLM (4B Instruct)": seed_sd("inst_dec", DEC_CELLS)},
    }
    for ax, series, xlabels, title in [
        (axL, series_frac, frac_xlabels, "Fractions (SP2013)"),
        (axR, series_dec, dec_xlabels, "Decimals (BSS2021)"),
    ]:
        x = np.arange(len(xlabels))
        n = len(series); w = 0.78/n
        offs = (np.arange(n) - (n-1)/2)*w
        for off, (lbl, v), c in zip(offs, series, colors):
            yerr = sd_map[title].get(lbl)
            ax.bar(x+off, v*100, w, color=c, label=lbl,
                   edgecolor="white", linewidth=0.4,
                   yerr=yerr, error_kw=dict(elinewidth=0.6, capsize=1.2, ecolor="0.25"))
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=9)
        sns.despine(ax=ax)
    axL.set_ylabel("Accuracy (%)", fontsize=9)
    axR.legend(loc="upper right", frameon=False, fontsize=7, bbox_to_anchor=(1.02,1.08))
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR/"qwen3_4b_vs_4bi_per_cell_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    if PAPER_FIG_DIR.exists():
        plt.savefig(PAPER_FIG_DIR/"qwen3_4b_vs_4bi_per_cell_comparison.png", bbox_inches="tight")
    print("wrote", out)

    # Per-cell summary CSV for all 4 Instruct variants
    rows = []
    for var, slug in [("Base","Base"),("+ humanFT","humanFT"),
                      ("+ distill","distill"),("+ distill + humanFT","distill_humanFT")]:
        for dom, cells, op_col, od_col in [
            ("fractions", [(o.capitalize(), d) for o,d in FRAC_CELLS], "operation", "operands"),
            ("decimals", DEC_CELLS, "operation", "operands"),
        ]:
            p = INSTRUCT_DIR/f"rollouts_4bi_{dom}_{slug}.csv"
            df = pd.read_csv(p)
            g = df.groupby([op_col, od_col])["is_correct"].mean().reindex(cells)
            for (op,od), acc in g.items():
                rows.append({"variant":var,"domain":dom,"operation":op,
                              "operands":od,"acc_instruct":acc})
    pd.DataFrame(rows).to_csv(OUT_DIR/"per_cell_acc_instruct.csv", index=False)
    print("wrote", OUT_DIR/"per_cell_acc_instruct.csv")

    print("\n=== Cognitive-LLM (4B Instruct) per-cell accuracy ===")
    print("\nFractions:")
    for (op,od), acc_base, acc_inst in zip(FRAC_CELLS, base_frac, instruct_frac):
        print(f"  {FRAC_LABELS[op]:5s}{od}  Base={acc_base*100:5.1f}%  Instruct={acc_inst*100:5.1f}%")
    print("\nDecimals:")
    for (op,od), acc_base, acc_inst in zip(DEC_CELLS, base_dec, instruct_dec):
        print(f"  {op:5s}{od:5s}  Base={acc_base*100:5.1f}%  Instruct={acc_inst*100:5.1f}%")


if __name__ == "__main__":
    main()
