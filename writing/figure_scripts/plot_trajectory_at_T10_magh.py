#!/usr/bin/env python3
"""Trajectory at T=1.0 with per-cell MAG-H (matches the paper definition).

Same rollouts as plot_trajectory_at_T10_fixed_parser.py, but aggregates per
operation x operands cell (8 cells for SP2013 fractions) instead of per-problem.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROLLOUTS_DIR = Path("/tmp/uma_pr02_max2/results/finetuning/trajectory_at_T10/rollouts")
HUMAN_CSV = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data/siegler_fraction_human.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")

STAGES = ["base", "distill_ep1", "hft_ep1", "hft_ep2", "hft_ep3", "hft_ep5"]
STAGE_LABELS = ["pretrained\nbase", "+ distill\nep1", "+ humanFT\nep1",
                "+ humanFT\nep2", "+ humanFT\nep3", "+ humanFT\nep5"]
SIZE_MAP = {"0p6B": "0.6B", "1p7B": "1.7B", "4B": "4B", "8B": "8B"}


def load_human() -> pd.DataFrame:
    h = pd.read_csv(HUMAN_CSV, low_memory=False).dropna(subset=["acc"])
    h["prob"] = h["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    cell = h.groupby(["operation", "operands"])["acc"].mean().rename("acc_human").reset_index()
    prob_to_cell = h.groupby("prob")[["operation", "operands"]].first().reset_index()
    return cell, prob_to_cell


def magh_for(df_size: pd.DataFrame, stage: str,
             human_cell: pd.DataFrame, prob_to_cell: pd.DataFrame) -> float | None:
    sub = df_size[df_size["stage"] == stage]
    if sub.empty:
        return None
    sub = sub.merge(prob_to_cell, on="prob", how="left")
    cell_acc = sub.groupby(["operation", "operands"])["correct"].mean().rename("acc_model").reset_index()
    merged = cell_acc.merge(human_cell, on=["operation", "operands"], how="inner")
    return float((merged["acc_model"] - merged["acc_human"]).abs().mean() * 100.0)


def mae_for_perprob(df_size: pd.DataFrame, stage: str, human_acc: dict[str, float]) -> float | None:
    sub = df_size[df_size["stage"] == stage]
    if sub.empty:
        return None
    per_prob = sub.groupby("prob")["correct"].mean().rename("acc_model").reset_index()
    per_prob["acc_human"] = per_prob["prob"].map(human_acc).astype(float)
    return float((per_prob["acc_model"] - per_prob["acc_human"]).abs().mean() * 100.0)


def collect() -> pd.DataFrame:
    human_cell, prob_to_cell = load_human()
    human_acc_perprob = pd.read_csv(HUMAN_CSV, low_memory=False).dropna(subset=["acc"])
    human_acc_perprob["prob"] = human_acc_perprob["prob"].astype(str).str.replace("÷", ":", regex=False).str.strip()
    perprob_dict = human_acc_perprob.groupby("prob")["acc"].mean().to_dict()
    rows = []
    for size_file, size in SIZE_MAP.items():
        df = pd.read_csv(ROLLOUTS_DIR / f"rollouts_{size_file}.csv.gz", compression="gzip")
        for stage in STAGES:
            rows.append({
                "size": size,
                "stage": stage,
                "MAG_H_pp": magh_for(df, stage, human_cell, prob_to_cell),
                "MAE_pp_perprob": mae_for_perprob(df, stage, perprob_dict),
            })
    return pd.DataFrame(rows)


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = collect()
    print("\n=== Trajectory @ T=1.0 [MAG-H per-cell] ===")
    pivot_magh = df.pivot_table(index="stage", columns="size", values="MAG_H_pp").reindex(STAGES)
    print(pivot_magh.to_string())
    print("\n=== Side-by-side: MAG-H (per-cell) vs MAE (per-problem) ===")
    side = df.assign(delta=df["MAG_H_pp"] - df["MAE_pp_perprob"])
    print(side.pivot_table(index="stage", columns="size", values="delta").reindex(STAGES).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "qwen_trajectory_at_T10_magh.csv", index=False)

    sizes = ["0.6B", "1.7B", "4B", "8B"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    color_for = dict(zip(sizes, colors))

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
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
    out_png = OUT_DIR / "qwen_trajectory_at_T10_magh.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
