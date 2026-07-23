#!/usr/bin/env python3
"""From-base humanFT trajectory at T=1.0 with per-cell MAG-H (paper metric).

Mirror of plot_frombase_trajectory_at_T10_fixed_parser.py, but aggregates per
operation x operands cell (8 cells for SP2013 fractions) instead of per-problem.
Reads the same fbtraj rollouts produced by the fixed-parser eval driver, plus
the original base-stage rollouts from the distill trajectory experiment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FROMBASE_FIXED = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_frombase_trajectory_t10_fixed")
ROLLOUTS_DIR = Path("/tmp/uma_pr02_max2/results/finetuning/trajectory_at_T10/rollouts")
HUMAN_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

STAGES = ["base", "hft_ep1", "hft_ep2", "hft_ep3", "hft_ep5"]
STAGE_LABELS = ["pretrained\nbase", "+ humanFT\nep1", "+ humanFT\nep2", "+ humanFT\nep3", "+ humanFT\nep5"]
SIZE_MAP = {"0p6b": "0.6B", "1p7b": "1.7B", "4b": "4B", "8b": "8B"}
SIZE_TO_BASE_FILE = {"0.6B": "0p6B", "1.7B": "1p7B", "4B": "4B", "8B": "8B"}


def load_human() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(HUMAN_CSV, low_memory=False).dropna(subset=["acc"])
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    cell = h.groupby(["operation", "operands"])["acc"].mean().rename("acc_human").reset_index()
    prob_to_cell = h.groupby("prob")[["operation", "operands"]].first().reset_index()
    return cell, prob_to_cell


def magh_from_rollouts(df: pd.DataFrame,
                       human_cell: pd.DataFrame,
                       prob_to_cell: pd.DataFrame) -> float:
    sub = df.merge(prob_to_cell, on="prob", how="left")
    cell_acc = sub.groupby(["operation", "operands"])["correct"].mean().rename("acc_model").reset_index()
    merged = cell_acc.merge(human_cell, on=["operation", "operands"], how="inner")
    return float((merged["acc_model"] - merged["acc_human"]).abs().mean() * 100.0)


def base_magh(size: str, human_cell, prob_to_cell) -> float:
    f = ROLLOUTS_DIR / f"rollouts_{SIZE_TO_BASE_FILE[size]}.csv.gz"
    df = pd.read_csv(f, compression="gzip")
    return magh_from_rollouts(df[df["stage"] == "base"], human_cell, prob_to_cell)


def hft_magh(size_raw: str, ep: int, human_cell, prob_to_cell) -> float | None:
    f = FROMBASE_FIXED / f"rollouts_fbtraj_{size_raw}_hftep{ep}.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f, compression="gzip")
    return magh_from_rollouts(df, human_cell, prob_to_cell)


def collect() -> pd.DataFrame:
    human_cell, prob_to_cell = load_human()
    rows = []
    for size_raw, size in SIZE_MAP.items():
        rows.append({"size": size, "stage": "base",
                     "MAG_H_pp": base_magh(size, human_cell, prob_to_cell)})
        for ep, stage in [(1, "hft_ep1"), (2, "hft_ep2"), (3, "hft_ep3"), (5, "hft_ep5")]:
            rows.append({"size": size, "stage": stage,
                         "MAG_H_pp": hft_magh(size_raw, ep, human_cell, prob_to_cell)})
    return pd.DataFrame(rows)


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = collect()
    print("\n=== From-base humanFT trajectory @ T=1.0 [MAG-H per-cell] ===")
    pivot = df.pivot_table(index="stage", columns="size", values="MAG_H_pp").reindex(STAGES)
    print(pivot.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "qwen_frombase_trajectory_at_T10_magh.csv", index=False)

    sizes = ["0.6B", "1.7B", "4B", "8B"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    color_for = dict(zip(sizes, colors))

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    x = np.arange(len(STAGES))
    for size in sizes:
        sub = df[df["size"] == size].set_index("stage").reindex(STAGES)
        y = sub["MAG_H_pp"].values.astype(float)
        valid = ~np.isnan(y)
        ax.plot(x[valid], y[valid], marker="o", markersize=6, linewidth=2.0,
                color=color_for[size], label=f"Qwen3-{size}")
        for xi, v in zip(x[valid], y[valid]):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("MAE model vs human", fontsize=9)
    ax.set_ylim(5, 42)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    out_png = OUT_DIR / "qwen_frombase_trajectory_at_T10_magh.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
