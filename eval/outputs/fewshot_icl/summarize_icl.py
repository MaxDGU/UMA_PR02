#!/usr/bin/env python3
"""Aggregate few-shot ICL baseline rollouts into a single summary CSV.

For each (model, k_shot, domain) cell, compute per-cell accuracy (using the
same `operation` + `denom_type`/`operands` partitions as the rest of the paper)
then take the mean absolute gap (MAE in percentage points) to the human
profile.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_ROOT = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/fewshot_icl_baselines"
)
SIEGLER_FRACTION_HUMAN = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_pr02_push/data/siegler_fraction_human.csv"
)
BSS2021_HUMAN_PER_CELL = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_humanft_bss2021_from_distill_20260520_143350/bss2021_eval_20260520_144339/per_cell.csv"
)

# Slugs we ran, in canonical order
MODEL_SLUGS = ["claude_sonnet_4_6", "gemini_3_flash", "gpt_5_5_low"]
K_VALUES = [0, 2, 5, 10]


def fraction_human_per_cell() -> pd.DataFrame:
    """Aggregate Siegler SP2013 to per-(operation, operands) human accuracy.

    The CSV's `operation` column uses canonical mapping {add, sub, mult, div}.
    We map to Add/Sub/Mul/Div to match the model outputs.
    """
    df = pd.read_csv(SIEGLER_FRACTION_HUMAN)
    op_map = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}
    df = df.copy()
    df["operation"] = df["operation"].map(op_map)
    df = df[df["operation"].notna() & df["operands"].isin(["ED", "UD"])]
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
    agg = (
        df.groupby(["operation", "operands"], as_index=False)["acc"].mean()
        .rename(columns={"acc": "acc_human", "operands": "denom_type"})
    )
    return agg


def decimal_human_per_cell() -> pd.DataFrame:
    """Use the per-cell BSS2021 human profile already computed for the paper."""
    df = pd.read_csv(BSS2021_HUMAN_PER_CELL)
    return df[["operation", "operands", "acc_human"]].copy()


def cell_accuracy_for_run(csv_path: Path, domain: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    group_col = "denom_type" if domain == "fraction" else "operands"
    # Accuracy at the (operation, group_col) level: mean over all rollouts.
    cell = df.groupby(["operation", group_col], as_index=False)["is_correct"].mean()
    cell = cell.rename(columns={"is_correct": "acc"})
    return cell


def mae_pp(cells: pd.DataFrame, human: pd.DataFrame, group_col: str) -> float:
    merged = cells.merge(human, on=["operation", group_col], how="inner")
    if merged.empty:
        return float("nan")
    return float((merged["acc"] - merged["acc_human"]).abs().mean() * 100.0)


def main() -> int:
    rows = []
    frac_human = fraction_human_per_cell()
    dec_human = decimal_human_per_cell()

    for domain in ["fraction", "decimal"]:
        group_col = "denom_type" if domain == "fraction" else "operands"
        human = frac_human if domain == "fraction" else dec_human
        for slug in MODEL_SLUGS:
            for k in K_VALUES:
                csv = OUT_ROOT / domain / f"{slug}_k{k}.csv"
                if not csv.exists():
                    rows.append({"model": slug, "k_shot": k, "domain": domain, "n_rollouts": 0, "acc": None, "mae_pp_vs_human": None})
                    continue
                df = pd.read_csv(csv)
                acc = float(df["is_correct"].mean())
                cells = cell_accuracy_for_run(csv, domain)
                mae = mae_pp(cells, human, group_col)
                rows.append(
                    {
                        "model": slug,
                        "k_shot": k,
                        "domain": domain,
                        "n_rollouts": len(df),
                        "acc": acc,
                        "mae_pp_vs_human": mae,
                    }
                )

    summary = pd.DataFrame(rows)
    out = OUT_ROOT / "summary.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
