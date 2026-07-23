#!/usr/bin/env python3
"""Per-model problem-macro TVD vs human strategy distribution, with bootstrap CIs.

Fractions: 4-family comparison (AS, M, D, OTHER). Model canonical codes
KDON/CDON->AS, ONOD->M, ICDM/CROP->D, OTHER->OTHER. Human strat column collapses
to the same families via SP2013_STRAT_TO_STRATEGY_FAMILY.

Decimals: 4 codes (AS, M, BOTH, O) — already at family level from the classifier.
Human distribution from BSS2021 strat_add/strat_mul binary cues.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RUNS = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/strategy_runs")
HUMAN_FRAC = Path("/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/max_uma_pr02/eval/data/siegler_fraction_human.csv")
HUMAN_DEC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/bss2021_human_responses.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

MODEL_ORDER = [
    ("ours_humanft",      "Cognitive Distillation + FT (ours)"),
    ("centaur_70b",       "Centaur-70B"),
    ("gpt_4_1_mini",      "GPT-4.1 mini"),
    ("gpt_5_5_low",       "GPT-5.5 low"),
    ("claude_sonnet_4_6", "Claude Sonnet 4.6"),
    ("gemini_2_5_flash",  "Gemini 2.5 Flash"),
    ("gemini_3_flash",    "Gemini 3 Flash"),
]

# Fractions: canonical->family
CANON_TO_FAMILY = {"KDON": "AS", "CDON": "AS", "ONOD": "M",
                   "ICDM": "D", "CROP": "D", "OTHER": "OTHER"}
FRAC_FAMILIES = ["AS", "M", "D", "OTHER"]
# SP2013 raw strat -> family
SP_TO_FAMILY = {"OpNumKeepDen": "AS", "IndepComp": "M",
                "InvertOper": "D", "Other/None": "OTHER"}

DEC_FAMILIES = ["AS", "M", "BOTH", "O"]


# ---- Fractions ----

def fraction_human_distribution() -> pd.DataFrame:
    h = pd.read_csv(HUMAN_FRAC, low_memory=False)
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    h["family"] = h["strat"].map(SP_TO_FAMILY).fillna("OTHER")
    dist = (h.groupby(["prob", "family"]).size()
            .groupby(level=0).apply(lambda s: s / s.sum())
            .unstack(fill_value=0.0))
    for fam in FRAC_FAMILIES:
        if fam not in dist.columns:
            dist[fam] = 0.0
    return dist[FRAC_FAMILIES]


def fraction_model_distribution(slug: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / f"fraction_{slug}" / "predictions.csv", low_memory=False)
    df["problem"] = df["problem"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    df["family"] = df["pred_code"].map(CANON_TO_FAMILY).fillna("OTHER")
    dist = (df.groupby(["problem", "family"]).size()
            .groupby(level=0).apply(lambda s: s / s.sum())
            .unstack(fill_value=0.0))
    for fam in FRAC_FAMILIES:
        if fam not in dist.columns:
            dist[fam] = 0.0
    return dist[FRAC_FAMILIES]


# ---- Decimals ----

def decimal_human_distribution() -> pd.DataFrame:
    h = pd.read_csv(HUMAN_DEC, low_memory=False)
    def assign_fam(r):
        a = int(r["strat_add"] > 0)
        m = int(r["strat_mul"] > 0)
        if a and m: return "BOTH"
        if a: return "AS"
        if m: return "M"
        return "O"
    h["family"] = h.apply(assign_fam, axis=1)
    dist = (h.groupby(["prob", "family"]).size()
            .groupby(level=0).apply(lambda s: s / s.sum())
            .unstack(fill_value=0.0))
    for fam in DEC_FAMILIES:
        if fam not in dist.columns:
            dist[fam] = 0.0
    return dist[DEC_FAMILIES]


def decimal_model_distribution(slug: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / f"decimal_{slug}" / "predictions.csv", low_memory=False)
    df["family"] = df["pred_code"].astype(str).str.upper().where(
        df["pred_code"].astype(str).str.upper().isin(DEC_FAMILIES), "O")
    dist = (df.groupby(["problem", "family"]).size()
            .groupby(level=0).apply(lambda s: s / s.sum())
            .unstack(fill_value=0.0))
    for fam in DEC_FAMILIES:
        if fam not in dist.columns:
            dist[fam] = 0.0
    return dist[DEC_FAMILIES]


# ---- TVD & bootstrap ----

def tvd_with_bootstrap(model: pd.DataFrame, human: pd.DataFrame,
                       n_boot: int = 5000, seed: int = 12345):
    common = sorted(set(model.index) & set(human.index))
    if len(common) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = model.loc[common].values
    h = human.loc[common].values
    per_prob = 0.5 * np.abs(m - h).sum(axis=1)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(per_prob)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = per_prob[idx].mean()
    point = per_prob.mean()
    return float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), n


def compute_panel(domain: str) -> pd.DataFrame:
    if domain == "fraction":
        human = fraction_human_distribution()
        model_fn = fraction_model_distribution
    else:
        human = decimal_human_distribution()
        model_fn = decimal_model_distribution
    rows = []
    for slug, label in MODEL_ORDER:
        if not (RUNS / f"{domain}_{slug}" / "predictions.csv").exists():
            continue
        model = model_fn(slug)
        point, lo, hi, n = tvd_with_bootstrap(model, human)
        rows.append({"model": label, "domain": domain,
                     "tvd": point, "ci_lo": lo, "ci_hi": hi, "n_problems": n})
    return pd.DataFrame(rows)


# ---- Plot ----

def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    frac = compute_panel("fraction").sort_values("tvd").reset_index(drop=True)

    print("\n=== FRACTIONS ===")
    print(frac.to_string(index=False))

    fig, axes = plt.subplots(1, 1, figsize=(4.0, 3.4), dpi=300)
    axes = [axes]
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(frac)))

    for ax, df_panel, title in zip(axes, [frac], ["Fractions (SP2013)"]):
        y = np.arange(len(df_panel))[::-1]
        colors = [palette[i] for i in range(len(df_panel))]
        err_lo = (df_panel["tvd"] - df_panel["ci_lo"]).values
        err_hi = (df_panel["ci_hi"] - df_panel["tvd"]).values
        ax.barh(y, df_panel["tvd"].values, color=colors,
                xerr=np.vstack([err_lo, err_hi]),
                error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2),
                edgecolor="white", linewidth=0.4, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(df_panel["model"].values, fontsize=8)
        ax.set_xlabel("TVD vs human strategy distribution", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(df_panel["ci_hi"].max() * 1.1, 0.6))
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "tvd_baselines_viridis.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
