#!/usr/bin/env python3
"""Build low/mid/high student profiles from sp2013_traces.csv.

Rank the ~996 UMA students by SP2013 accuracy, split into tertiles, and pick
the parameter combo (g, d, rt_mu, ice) closest to each tertile's median
accuracy. Profile params are snapped to the UMA grid.

Outputs:
  data/student_profiles_4b.csv          per-student profile assignment
  data/student_profiles_4b_summary.csv  per-profile rep params + targets
  data/student_profiles_4b_targets.csv  per-profile per-problem acc target
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/verification_full_results.csv")
OUT_DIR = Path("/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/data")


def main() -> None:
    df = pd.read_csv(SRC)
    df = df.rename(columns={"acc": "is_correct"})
    df["is_correct"] = df["is_correct"].astype(int)
    print(f"loaded {len(df):,} rows, {df['subjid'].nunique()} students, {df['prob'].nunique()} problems")

    per_student = (
        df.groupby("subjid")
        .agg(acc=("is_correct", "mean"),
             g=("g", "first"), d=("d", "first"),
             rt_mu=("rt_mu", "first"), ice=("ice", "first"))
        .reset_index()
        .sort_values("acc")
        .reset_index(drop=True)
    )
    n = len(per_student)
    print(f"per-student acc: min={per_student['acc'].min():.3f} med={per_student['acc'].median():.3f} max={per_student['acc'].max():.3f}")

    cut_lo = n // 3
    cut_hi = (2 * n) // 3
    per_student["profile"] = "mid"
    per_student.loc[:cut_lo - 1, "profile"] = "low"
    per_student.loc[cut_hi:, "profile"] = "high"

    out_per_student = OUT_DIR / "student_profiles_4b.csv"
    per_student.to_csv(out_per_student, index=False)
    print(f"wrote {out_per_student}")

    summary_rows = []
    per_problem_rows = []
    for profile in ("low", "mid", "high"):
        sub = per_student[per_student["profile"] == profile]
        target_acc = sub["acc"].median()
        idx = (sub["acc"] - target_acc).abs().idxmin()
        rep = sub.loc[idx]
        summary_rows.append({
            "profile": profile,
            "n_students": len(sub),
            "acc_min": sub["acc"].min(),
            "acc_median": target_acc,
            "acc_max": sub["acc"].max(),
            "rep_subjid": int(rep["subjid"]),
            "rep_acc": rep["acc"],
            "rep_g": rep["g"], "rep_d": rep["d"],
            "rep_rt_mu": rep["rt_mu"], "rep_ice": rep["ice"],
        })

        sub_traces = df[df["subjid"].isin(sub["subjid"])]
        prob_acc = (
            sub_traces.groupby("prob")
            .agg(acc=("is_correct", "mean"))
            .reset_index()
        )
        prob_acc["op"] = prob_acc["prob"].apply(
            lambda p: next(c for c in "+-*:" if c in p)
        )
        prob_acc.insert(0, "profile", profile)
        per_problem_rows.append(prob_acc)

    summary = pd.DataFrame(summary_rows)
    out_summary = OUT_DIR / "student_profiles_4b_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"wrote {out_summary}")
    print(summary.to_string(index=False))

    targets = pd.concat(per_problem_rows, ignore_index=True)
    out_targets = OUT_DIR / "student_profiles_4b_targets.csv"
    targets.to_csv(out_targets, index=False)
    print(f"wrote {out_targets} ({len(targets)} rows)")


if __name__ == "__main__":
    main()
