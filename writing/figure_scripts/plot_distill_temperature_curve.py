#!/usr/bin/env python3
"""Plot distill-only MAE pp vs human SP2013 distribution as a function of
sampling temperature, for all four Qwen3 sizes.

0.6B/1.7B numbers come from the collaborator's vllm panel25 sweep
(`/tmp/uma_pr02_max/results/finetuning/qwen_{0p6B,1p7B}/all_conditions_comparison.csv`,
MAG-H pp). 4B/8B numbers come from our chained-eval T-sweep in
`finetune/output/qwen3_distill_tempsweep/` plus T=0.7 from chained_eval and
T=1.0 from qwen3_trajectory_eval/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLLAB_ROOT = Path("/tmp/uma_pr02_max/results/finetuning")
LOCAL_TEMPSWEEP = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_distill_tempsweep")
LOCAL_CHAINED = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_chained_eval_20260503_110636")
LOCAL_TRAJ = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune/output/qwen3_trajectory_eval")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/figures")


def collect_small_collab() -> pd.DataFrame:
    """Collaborator's vllm panel25 pipeline (different prompt, 1000 rollouts/problem)."""
    rows = []
    for size, dirname in [("0.6B", "qwen_0p6B"), ("1.7B", "qwen_1p7B")]:
        df = pd.read_csv(COLLAB_ROOT / dirname / "all_conditions_comparison.csv")
        for _, r in df.iterrows():
            rows.append({"size": size, "T": float(r["temperature"]),
                         "MAE_pp": float(r["MAG_H_pp"]),
                         "pipeline": "vllm_panel25"})
    return pd.DataFrame(rows)


def collect_large() -> pd.DataFrame:
    rows = []
    # T=0.7 from chained_eval comparison (already-evaluated)
    df_07 = pd.read_csv(LOCAL_CHAINED / "qwen3_chained_eval_comparison.csv")
    for size, cond in [("4B", "q3_4b_distill"), ("8B", "q3_8b_distill")]:
        r = df_07[df_07["condition"] == cond].iloc[0]
        rows.append({"size": size, "T": 0.7,
                     "MAE_pp": float(r["MAE_pp"]),
                     "pipeline": "hf_plain"})
    # T=1.0 from trajectory eval
    for size in ["4b", "8b"]:
        p = LOCAL_TRAJ / f"sp2013_summary_traj_{size}_distill.json"
        d = json.loads(p.read_text())
        rows.append({"size": size.upper(), "T": float(d["temperature"]),
                     "MAE_pp": float(d["mae_pp_overall"]),
                     "pipeline": "hf_plain"})
    # T-sweep
    for p in sorted(LOCAL_TEMPSWEEP.glob("sp2013_summary_dsw_*_distill_t*.json")):
        d = json.loads(p.read_text())
        raw = p.stem.split("_")[3]  # ..._dsw_4b_distill_t09 -> "4b"
        size_map = {"0p6b": "0.6B", "1p7b": "1.7B", "4b": "4B", "8b": "8B"}
        size = size_map.get(raw, raw.upper())
        rows.append({"size": size, "T": float(d["temperature"]),
                     "MAE_pp": float(d["mae_pp_overall"]),
                     "pipeline": "hf_plain"})
    return pd.DataFrame(rows)


def main() -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    small = collect_small_collab()
    large = collect_large()
    df = pd.concat([small, large], ignore_index=True).sort_values(["size", "pipeline", "T"])
    print("\n=== Distill temperature curve (MAE pp vs human) ===")
    pivot = df.pivot_table(index="T", columns=["size", "pipeline"], values="MAE_pp")
    print(pivot.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "qwen_distill_temperature_curve.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    sizes = ["0.6B", "1.7B", "4B", "8B"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    color_for = dict(zip(sizes, colors))

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=300)
    for size in sizes:
        # Prefer our hf_plain T-sweep where available; fall back to vllm_panel25 (only collab has 0.6B/1.7B in their pipeline).
        sub_hf = df[(df["size"] == size) & (df["pipeline"] == "hf_plain")].sort_values("T")
        sub_vllm = df[(df["size"] == size) & (df["pipeline"] == "vllm_panel25")].sort_values("T")
        for sub, ls, lbl_suffix in [(sub_hf, "-", "hf256"), (sub_vllm, ":", "vllm1000")]:
            if sub.empty:
                continue
            ax.plot(sub["T"], sub["MAE_pp"], marker="o", markersize=5, linewidth=1.6,
                    color=color_for[size], linestyle=ls,
                    label=f"Qwen3-{size} ({lbl_suffix})")
            imin = sub["MAE_pp"].idxmin()
            ax.scatter(sub.loc[imin, "T"], sub.loc[imin, "MAE_pp"],
                       s=80, marker="*", color=color_for[size],
                       edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel("Decoding temperature $T$", fontsize=9)
    ax.set_ylabel("MAE model vs human", fontsize=9)
    ax.set_title("Distill only (SP2013)", fontsize=10, pad=8)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    out_png = OUT_DIR / "qwen_distill_temperature_curve.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
