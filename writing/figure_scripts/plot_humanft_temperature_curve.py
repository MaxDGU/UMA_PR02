#!/usr/bin/env python3
"""Plot cheap-fix humanFT MAE pp vs human SP2013 as a function of sampling
temperature, for all four Qwen3 sizes. Mirrors plot_distill_temperature_curve.py
so the two figures are directly comparable side-by-side.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SWEEP = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_humanft_cheapfix_tempsweep")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")


SIZE_MAP = {"0p6b": "0.6B", "1p7b": "1.7B", "4b": "4B", "8b": "8B"}


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted(SWEEP.glob("sp2013_summary_hftc_*_t*.json")):
        d = json.loads(p.read_text())
        stem = p.stem  # sp2013_summary_hftc_<size>_t<TT>
        parts = stem.split("_")
        size_raw = parts[3]  # hftc_<size>
        rows.append({
            "size": SIZE_MAP.get(size_raw, size_raw),
            "T": float(d["temperature"]),
            "MAE_pp": float(d["mae_pp_overall"]),
            "acc": float(d["acc_overall_model"]),
            "MAE_add": d["mae_pp_by_op"]["add"],
            "MAE_sub": d["mae_pp_by_op"]["sub"],
            "MAE_mul": d["mae_pp_by_op"]["mul"],
            "MAE_div": d["mae_pp_by_op"]["div"],
        })
    return pd.DataFrame(rows).sort_values(["size", "T"])


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = collect()
    print("\n=== HumanFT (cheap-fix) temperature curve ===")
    pivot = df.pivot_table(index="T", columns="size", values="MAE_pp")
    print(pivot.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "qwen_humanft_temperature_curve.csv", index=False)

    sizes = ["0.6B", "1.7B", "4B", "8B"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    color_for = dict(zip(sizes, colors))

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=300)
    for size in sizes:
        sub = df[df["size"] == size]
        if sub.empty:
            continue
        ax.plot(sub["T"], sub["MAE_pp"], marker="o", markersize=5, linewidth=1.6,
                color=color_for[size], linestyle="-",
                label=f"Qwen3-{size}")
        imin = sub["MAE_pp"].idxmin()
        ax.scatter(sub.loc[imin, "T"], sub.loc[imin, "MAE_pp"],
                   s=80, marker="*", color=color_for[size],
                   edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel("Decoding temperature $T$", fontsize=9)
    ax.set_ylabel("MAE model vs human", fontsize=9)
    ax.set_title("Distill + humanFT (SP2013)", fontsize=10, pad=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    out_png = OUT_DIR / "qwen_humanft_temperature_curve.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
