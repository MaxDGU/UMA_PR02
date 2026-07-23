#!/usr/bin/env python3
"""Per-model MAG-H with bootstrap-over-cells 95% CI, viridis style.

Two panels side by side (fractions, decimals). Bars are sorted within each
panel by MAG-H ascending. Includes: 5 frontier LLMs, Centaur-70B, Direct UMA,
and our Qwen3-4B at base / +distill / +humanFT stages.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---- data sources --------------------------------------------------------

FRONTIER_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/max_uma_pr02/eval/outputs")
FRAC_UPSAMPLED_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/fraction_baselines_100samples")
QWEN_BASE_DEC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_base_bss2021_eval_20260520_145110/per_sample.csv")
QWEN_DISTILL_DEC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260518_134624_lora/bss2021_eval_20260520_124043/per_sample.csv")
QWEN_HFT_DEC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_humanft_bss2021_from_distill_20260520_143350/bss2021_eval_20260520_144339/per_sample.csv")

QWEN_FRAC_ROLLOUTS = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_pr02_push/results/finetuning/trajectory_at_T10/rollouts/rollouts_4B.csv.gz")

UMA_FRAC_REF = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/uma_vs_human_results.csv")
UMA_DEC_REF = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/uma_bss2021_summary.csv")
HUMAN_FRAC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
HUMAN_DEC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/bss2021_human_summary.csv")

OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

FRONTIER_MODELS = {
    "gpt_4_1_mini": "GPT-4.1 mini",
    "gpt_5_5_low": "GPT-5.5 low",
    "claude_sonnet_4_6": "Claude Sonnet 4.6",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "gemini_3_flash": "Gemini 3 Flash",
    "centaur_70b": "Centaur-70B",
}
FRONTIER_MODELS_DEC = FRONTIER_MODELS
FRONTIER_MODELS_FRAC = FRONTIER_MODELS


# ---- helpers --------------------------------------------------------------

def normalize_op(s: str) -> str:
    return {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div",
            "Add": "Add", "Sub": "Sub", "Mul": "Mul", "Div": "Div"}.get(s, s)


def per_problem_acc(df: pd.DataFrame, prob_col: str = "problem") -> pd.DataFrame:
    return df.groupby(prob_col, as_index=False)["is_correct"].mean().rename(
        columns={"is_correct": "model_acc", prob_col: "prob"})


def load_frontier(model_key: str, domain: str) -> pd.DataFrame:
    if domain == "fraction" and model_key != "centaur_70b":
        path = FRAC_UPSAMPLED_DIR / f"{model_key}.csv"
    else:
        path = FRONTIER_DIR / domain / f"{model_key}.csv"
    df = pd.read_csv(path, low_memory=False)
    df["is_correct"] = df["is_correct"].astype(bool).astype(int)
    return per_problem_acc(df, "problem")


def load_qwen_decimal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["is_correct"] = df["is_correct"].astype(int)
    return per_problem_acc(df, "problem")


def load_qwen_frac_stage(stage: str) -> pd.DataFrame:
    df = pd.read_csv(QWEN_FRAC_ROLLOUTS, compression="gzip", low_memory=False)
    sub = df[df["stage"] == stage].copy()
    sub = sub.rename(columns={"prob": "problem", "correct": "is_correct"})
    sub["is_correct"] = sub["is_correct"].astype(int)
    return per_problem_acc(sub, "problem")


def load_uma_frac() -> pd.DataFrame:
    h = pd.read_csv(UMA_FRAC_REF, low_memory=False)
    return h.rename(columns={"uma_acc": "model_acc"})[["prob", "model_acc"]]


def load_uma_dec() -> pd.DataFrame:
    u = pd.read_csv(UMA_DEC_REF, low_memory=False)
    return u.rename(columns={"acc": "model_acc"})[["prob", "model_acc"]]


def load_human_frac_per_problem() -> pd.DataFrame:
    h = pd.read_csv(HUMAN_FRAC, low_memory=False).dropna(subset=["acc"])
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    return h.groupby("prob", as_index=False)["acc"].mean().rename(columns={"acc": "acc_human"})


def load_human_dec_per_problem() -> pd.DataFrame:
    h = pd.read_csv(HUMAN_DEC, low_memory=False)
    return h.rename(columns={"acc_mean": "acc_human"})[["prob", "acc_human"]]


# Fractions cell partition: 8 cells = {add,sub,mul,div} × {ED,UD}
FRAC_PROBS_ED = {"3/5+1/5", "3/5-1/5", "3/5*1/5", "3/5:1/5",
                 "4/5+3/5", "4/5-3/5", "4/5*3/5", "4/5:3/5"}
FRAC_PROBS_UD = {"3/5+1/4", "3/5-1/4", "3/5*1/4", "3/5:1/4",
                 "2/3+3/5", "2/3-3/5", "2/3*3/5", "2/3:3/5"}


def annotate_frac_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prob"] = df["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    df["denom_type"] = df["prob"].map(
        lambda p: "ED" if p in FRAC_PROBS_ED else ("UD" if p in FRAC_PROBS_UD else None))
    df["operation"] = df["prob"].map(
        lambda p: "Add" if "+" in p else ("Sub" if "-" in p else ("Mul" if "*" in p else "Div")))
    return df


# Decimals cell partition: 6 cells = {Add,Mul} × {EDD,UDD,D-W}
DEC_OPERANDS = {
    "12.3+5.6": "EDD", "24.45+0.34": "EDD",
    "2.46+4.1": "UDD", "0.826+0.12": "UDD",
    "5.61+23": "D-W", "0.415+52": "D-W",
    "2.4*1.2": "EDD", "0.41*0.31": "EDD",
    "2.3*0.13": "UDD", "0.32*2.1": "UDD",
    "31*3.2": "D-W", "14*0.21": "D-W",
}


def annotate_dec_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["operands"] = df["prob"].map(DEC_OPERANDS)
    df["operation"] = df["prob"].map(lambda p: "Add" if "+" in p else "Mul")
    return df


# ---- MAG-H per model ------------------------------------------------------

def magh_with_bootstrap(model_per_cell: np.ndarray, human_per_cell: np.ndarray,
                        n_boot: int = 5000, seed: int = 12345) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    gaps = np.abs(model_per_cell - human_per_cell)
    n = len(gaps)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = gaps[idx].mean()
    point = gaps.mean()
    return float(point * 100), float(np.quantile(boots, 0.025) * 100), float(np.quantile(boots, 0.975) * 100)


def compute_panel(domain: str) -> pd.DataFrame:
    rows = []
    if domain == "fraction":
        human = load_human_frac_per_problem()
        cell_keys = ["operation", "denom_type"]
        annotate = annotate_frac_cells
    else:
        human = load_human_dec_per_problem()
        cell_keys = ["operation", "operands"]
        annotate = annotate_dec_cells

    # Build human cell-level accuracy (model-independent)
    human_cell = annotate(human.rename(columns={"acc_human": "model_acc"}))
    human_cell = human_cell.groupby(cell_keys, as_index=False)["model_acc"].mean().rename(
        columns={"model_acc": "acc_human"})

    # Source models
    sources = []
    frontier_map = FRONTIER_MODELS_FRAC if domain == "fraction" else FRONTIER_MODELS_DEC
    for key, label in frontier_map.items():
        sources.append((label, load_frontier(key, domain)))

    if domain == "fraction":
        sources.append(("Direct UMA", load_uma_frac()))
        sources.append(("Qwen3-4B (base)", load_qwen_frac_stage("base")))
        sources.append(("Cognitive Distillation + FT (ours)", load_qwen_frac_stage("hft_ep1")))
    else:
        sources.append(("Direct UMA", load_uma_dec()))
        sources.append(("Qwen3-4B (base)", load_qwen_decimal(QWEN_BASE_DEC)))
        sources.append(("Cognitive Distillation + FT (ours)", load_qwen_decimal(QWEN_HFT_DEC)))

    for label, per_prob in sources:
        annotated = annotate(per_prob)
        cell_acc = annotated.groupby(cell_keys, as_index=False)["model_acc"].mean()
        merged = cell_acc.merge(human_cell, on=cell_keys, how="inner")
        if merged.isna().any().any() or len(merged) == 0:
            print(f"skip {label} on {domain} (missing cells)")
            continue
        point, lo, hi = magh_with_bootstrap(merged["model_acc"].values,
                                            merged["acc_human"].values)
        rows.append({"model": label, "domain": domain,
                     "mag_h": point, "ci_lo": lo, "ci_hi": hi,
                     "n_cells": len(merged)})
    return pd.DataFrame(rows)


# ---- plotting -------------------------------------------------------------

OURS_LABELS = {"Qwen3-4B (base)", "+ cognitive distillation", "+ human fine-tuning"}


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    frac = compute_panel("fraction").sort_values("mag_h", ascending=True).reset_index(drop=True)
    dec = compute_panel("decimal").sort_values("mag_h", ascending=True).reset_index(drop=True)

    print("\n=== FRACTIONS ===")
    print(frac.to_string(index=False))
    print("\n=== DECIMALS ===")
    print(dec.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), dpi=300, sharex=False)
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(frac), len(dec))))

    for ax, df_panel, title in zip(axes, [frac, dec], ["Fractions (SP2013)", "Decimals (BSS2021)"]):
        y = np.arange(len(df_panel))[::-1]
        colors = [palette[i] for i in range(len(df_panel))]
        err_lo = (df_panel["mag_h"] - df_panel["ci_lo"]).values
        err_hi = (df_panel["ci_hi"] - df_panel["mag_h"]).values
        ax.barh(y, df_panel["mag_h"].values, color=colors,
                xerr=np.vstack([err_lo, err_hi]), error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2),
                edgecolor="white", linewidth=0.4, height=0.7)
        # Bold "ours" labels
        labels = []
        for m in df_panel["model"]:
            if m in OURS_LABELS:
                labels.append(r"$\bf{" + m.replace(" ", "\\ ").replace("+", "+") + r"}$")
            else:
                labels.append(m)
        ax.set_yticks(y)
        ax.set_yticklabels(df_panel["model"].values, fontsize=8)
        ax.set_xlabel("MAE model vs human", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(df_panel["ci_hi"].max() * 1.10, 5))
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax, left=False)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "magh_baselines_viridis.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
