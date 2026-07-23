#!/usr/bin/env python3
"""Restricted-TVD: compare model and human strategy distributions per problem
restricted to OFF-TEXTBOOK codes. For each problem the expected textbook
strategy is dropped, so the metric measures only the procedural variability
that doesn't trivially reduce to "model used the right operation".

Fractions: expected per problem
  Add/Sub × ED: KDON | Add/Sub × UD: CDON | Mul: ONOD | Div: ICDM

Decimals: expected per problem
  Add: AS | Mul: M
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

FRAC_CANON = ["KDON", "CDON", "ONOD", "ICDM", "CROP", "OTHER"]
SP_TO_CANON_HUMAN = {"OpNumKeepDen": ["KDON", "CDON"],  # the human label is family-level
                     "IndepComp": ["ONOD"],
                     "InvertOper": ["ICDM", "CROP"],
                     "Other/None": ["OTHER"]}
# For per-problem expected procedure, use op + denom family
CANON_TO_FAMILY = {"KDON": "AS", "CDON": "AS", "ONOD": "M",
                   "ICDM": "D", "CROP": "D", "OTHER": "OTHER"}
SP_TO_FAMILY = {"OpNumKeepDen": "AS", "IndepComp": "M",
                "InvertOper": "D", "Other/None": "OTHER"}
FRAC_FAMILIES = ["AS", "M", "D", "OTHER"]
DEC_FAMILIES = ["AS", "M", "BOTH", "O"]


def fraction_expected_family(problem: str) -> str:
    """Map a fraction problem to its textbook family."""
    p = problem.replace("÷", ":")
    if "+" in p or "-" in p:
        # Add/Sub: AS family (KDON for ED, CDON for UD; both family AS)
        return "AS"
    if "*" in p:
        return "M"
    if ":" in p or "/" in p[1:]:
        return "D"
    return "OTHER"


def decimal_expected_family(problem: str) -> str:
    return "AS" if "+" in problem else "M"


def restricted_tvd(model_dist: dict, human_dist: dict, expected: str) -> float:
    """Per-problem TVD restricted to families != expected. No renorm."""
    families = set(model_dist) | set(human_dist)
    families.discard(expected)
    return 0.5 * sum(abs(model_dist.get(f, 0.0) - human_dist.get(f, 0.0)) for f in families)


# ---- Build per-problem distributions ----

def _normalize_rows(pivot: pd.DataFrame, families: list) -> pd.DataFrame:
    out = pivot.reindex(columns=families, fill_value=0.0).astype(float)
    sums = out.sum(axis=1).replace(0, 1.0)
    return out.div(sums, axis=0)


def fraction_distributions():
    h = pd.read_csv(HUMAN_FRAC, low_memory=False)
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    h["family"] = h["strat"].map(SP_TO_FAMILY).fillna("OTHER")
    pivot = h.pivot_table(index="prob", columns="family", values="strat",
                          aggfunc="count", fill_value=0)
    return _normalize_rows(pivot, FRAC_FAMILIES)


def fraction_model_dist(slug: str):
    df = pd.read_csv(RUNS / f"fraction_{slug}" / "predictions.csv", low_memory=False)
    df["problem"] = df["problem"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    df["family"] = df["pred_code"].map(CANON_TO_FAMILY).fillna("OTHER")
    pivot = df.pivot_table(index="problem", columns="family", values="pred_code",
                           aggfunc="count", fill_value=0)
    return _normalize_rows(pivot, FRAC_FAMILIES)


def decimal_human_distributions():
    h = pd.read_csv(HUMAN_DEC, low_memory=False)
    def fam(r):
        a, m = int(r["strat_add"] > 0), int(r["strat_mul"] > 0)
        if a and m: return "BOTH"
        if a: return "AS"
        if m: return "M"
        return "O"
    h["family"] = h.apply(fam, axis=1)
    pivot = h.pivot_table(index="prob", columns="family", values="strat_add",
                          aggfunc="count", fill_value=0)
    return _normalize_rows(pivot, DEC_FAMILIES)


def decimal_model_dist(slug: str):
    df = pd.read_csv(RUNS / f"decimal_{slug}" / "predictions.csv", low_memory=False)
    fam = df["pred_code"].astype(str).str.upper()
    df["family"] = fam.where(fam.isin(DEC_FAMILIES), "O")
    pivot = df.pivot_table(index="problem", columns="family", values="pred_code",
                           aggfunc="count", fill_value=0)
    return _normalize_rows(pivot, DEC_FAMILIES)


# ---- Compute restricted TVDs per panel ----

def compute_panel(domain: str):
    if domain == "fraction":
        human = fraction_distributions()
        expected_fn = fraction_expected_family
        model_fn = fraction_model_dist
    else:
        human = decimal_human_distributions()
        expected_fn = decimal_expected_family
        model_fn = decimal_model_dist
    rows = []
    for slug, label in MODEL_ORDER:
        if not (RUNS / f"{domain}_{slug}" / "predictions.csv").exists():
            continue
        model = model_fn(slug)
        common = sorted(set(model.index) & set(human.index))
        if not common:
            continue
        per_prob = np.array([
            restricted_tvd(model.loc[p].to_dict(), human.loc[p].to_dict(), expected_fn(p))
            for p in common
        ])
        rng = np.random.default_rng(12345)
        n = len(per_prob)
        boots = np.array([per_prob[rng.integers(0, n, size=n)].mean() for _ in range(5000)])
        rows.append({"model": label, "domain": domain,
                     "tvd_restricted": float(per_prob.mean()),
                     "ci_lo": float(np.quantile(boots, 0.025)),
                     "ci_hi": float(np.quantile(boots, 0.975)),
                     "n_problems": n})
    return pd.DataFrame(rows)


def main():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    frac = compute_panel("fraction").sort_values("tvd_restricted").reset_index(drop=True)
    dec = compute_panel("decimal").sort_values("tvd_restricted").reset_index(drop=True)

    print("\n=== FRACTIONS (off-textbook TVD) ===")
    print(frac.to_string(index=False))
    print("\n=== DECIMALS (off-textbook TVD) ===")
    print(dec.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), dpi=300, sharex=False)
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(frac), len(dec))))
    for ax, df_panel, title in zip(axes, [frac, dec], ["Fractions (SP2013)", "Decimals (BSS2021)"]):
        y = np.arange(len(df_panel))[::-1]
        colors = [palette[i] for i in range(len(df_panel))]
        err_lo = (df_panel["tvd_restricted"] - df_panel["ci_lo"]).values
        err_hi = (df_panel["ci_hi"] - df_panel["tvd_restricted"]).values
        ax.barh(y, df_panel["tvd_restricted"].values, color=colors,
                xerr=np.vstack([err_lo, err_hi]),
                error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2),
                edgecolor="white", linewidth=0.4, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(df_panel["model"].values, fontsize=8)
        ax.set_xlabel("Off-textbook TVD vs humans", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(df_panel["ci_hi"].max() * 1.1, 0.4))
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "tvd_restricted_viridis.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
