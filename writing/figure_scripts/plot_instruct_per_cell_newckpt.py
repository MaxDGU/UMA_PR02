#!/usr/bin/env python3
"""Per-cell accuracy comparison: 4B Base vs 4B Instruct, both as CPT-LLM.

Side-by-side panels for fractions and decimals, four bars per cell:
student reference, UMA, CPT-LLM (Base 4B), CPT-LLM (Instruct 4B)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PAPER_ROOT = REPO_ROOT / "writing" / "emnlp202026-humanlike-math-reasoning"
OUTPUTS = REPO_ROOT / "eval" / "outputs"
INSTRUCT_FRAC_DIR = OUTPUTS / "fraction_4bi"
INSTRUCT_DEC_DIR = OUTPUTS / "decimal_4bi"
BASE_FRAC_ROLLOUTS = OUTPUTS / "fraction_4b" / "qwen3_4b_distill_humanft.csv"
BASE_DEC_ROLLOUTS = OUTPUTS / "decimal_4b" / "qwen3_4b_distill_humanft.csv"
HUMAN_FRAC = REPO_ROOT / "eval" / "data" / "siegler_fraction_human.csv"
UMA_FRAC = OUTPUTS / "uma_reference" / "verification_full_results.csv.gz"
DEC_BSS_SUMMARY = OUTPUTS / "uma_reference" / "uma_bss2021_summary.csv"
DEC_HUMAN_PER_CELL = OUTPUTS / "decimals_new" / "per_cell.csv"
SEED_CSV = OUTPUTS / "seed_variance" / "seed_cell_acc.csv"
PAPER_FIG_DIR = PAPER_ROOT / "figures"
SYSNAME = "CPT-LLM"

FRAC_CELLS = [("add","ED"),("add","UD"),("sub","ED"),("sub","UD"),
              ("mul","ED"),("mul","UD"),("div","ED"),("div","UD")]
FRAC_LABELS = {"add":"Add","sub":"Sub","mul":"Mul","div":"Div"}
DEC_CELLS = [("Add","EDD"),("Add","UDD"),("Add","D-W"),
             ("Mul","EDD"),("Mul","UDD"),("Mul","D-W")]
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
    instruct_frac = cell_acc(INSTRUCT_FRAC_DIR/"rollouts_4bi_fractions_distill_humanFT.csv",
                              [(o.capitalize(), d) for o,d in FRAC_CELLS],
                              op_col="operation", od_col="operands")

    # Decimals
    h_d, u_d = dec_human_uma()
    base_dec = cell_acc(BASE_DEC_ROLLOUTS, DEC_CELLS,
                         op_col="operation", od_col="operands")
    instruct_dec = cell_acc(INSTRUCT_DEC_DIR/"rollouts_4bi_decimals_distill_humanFT.csv",
                             DEC_CELLS, op_col="operation", od_col="operands")

    series_frac = [
        ("Students", h_f),
        ("UMA", u_f),
        (f"{SYSNAME} (4B Base)", base_frac),
        (f"{SYSNAME} (4B Instruct)", instruct_frac),
    ]
    series_dec = [
        ("Students", h_d),
        ("UMA", u_d),
        (f"{SYSNAME} (4B Base)", base_dec),
        (f"{SYSNAME} (4B Instruct)", instruct_dec),
    ]

    colors = plt.cm.viridis(np.linspace(0.10, 0.90, 4))
    frac_xlabels = [f"{FRAC_LABELS[o]}\n{d}" for o,d in FRAC_CELLS]
    dec_xlabels = [f"{op}\n{od}" for op, od in DEC_CELLS]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.0), dpi=300,
                                    gridspec_kw={"width_ratios":[8,6]})

    sd_map = {
        "Fractions": {f"{SYSNAME} (4B Base)": seed_sd("base_frac", FRAC_CELLS),
                      f"{SYSNAME} (4B Instruct)": seed_sd("inst_frac", FRAC_CELLS)},
        "Decimals": {f"{SYSNAME} (4B Base)": seed_sd("base_dec", DEC_CELLS),
                     f"{SYSNAME} (4B Instruct)": seed_sd("inst_dec", DEC_CELLS)},
    }
    for ax, series, xlabels, title in [
        (axL, series_frac, frac_xlabels, "Fractions"),
        (axR, series_dec, dec_xlabels, "Decimals"),
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

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = PAPER_FIG_DIR / f"qwen3_4b_vs_4bi_per_cell_comparison.{suffix}"
        plt.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print(f"\n=== {SYSNAME} (4B Instruct) per-cell accuracy ===")
    print("\nFractions:")
    for (op,od), acc_base, acc_inst in zip(FRAC_CELLS, base_frac, instruct_frac):
        print(f"  {FRAC_LABELS[op]:5s}{od}  Base={acc_base*100:5.1f}%  Instruct={acc_inst*100:5.1f}%")
    print("\nDecimals:")
    for (op,od), acc_base, acc_inst in zip(DEC_CELLS, base_dec, instruct_dec):
        print(f"  {op:5s}{od:5s}  Base={acc_base*100:5.1f}%  Instruct={acc_inst*100:5.1f}%")


if __name__ == "__main__":
    main()
