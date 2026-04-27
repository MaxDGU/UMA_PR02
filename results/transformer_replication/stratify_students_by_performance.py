"""Pick a stratified subsample of students that mirrors the full UMA accuracy distribution.

Input: per-trial UMA eval CSV from eval_uma_on_test_problems.py (with subjid column).
Output:
  - subjid_subset.txt with N subjids, one per line
  - stratification_summary.csv showing accuracy per chosen student + bin index
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--uma-csv', required=True,
                   help='Per-trial UMA eval csv with columns subjid, prob, is_correct')
    p.add_argument('--n-bins', type=int, default=100,
                   help='Number of equal-mass bins to draw 1 student from each')
    p.add_argument('--out-subjids', required=True,
                   help='Path to write 1 subjid per line')
    p.add_argument('--out-summary', required=True,
                   help='Path to write stratification summary CSV')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    df = pd.read_csv(a.uma_csv)
    per_student = (df.groupby('subjid')['is_correct'].mean()
                     .reset_index().rename(columns={'is_correct': 'accuracy'})
                     .sort_values(['accuracy', 'subjid']).reset_index(drop=True))
    n = len(per_student)
    print(f"Loaded {n} students, accuracy range {per_student['accuracy'].min():.3f}–"
          f"{per_student['accuracy'].max():.3f}")

    rng = np.random.default_rng(a.seed)
    bin_size = n / a.n_bins
    chosen = []
    for b in range(a.n_bins):
        lo = int(np.floor(b * bin_size))
        hi = int(np.floor((b + 1) * bin_size))
        if hi <= lo:
            hi = lo + 1
        bin_slice = per_student.iloc[lo:hi]
        # pick the median-accuracy student (deterministic given the sort)
        mid = len(bin_slice) // 2
        sel = bin_slice.iloc[mid]
        chosen.append({
            'bin': b,
            'subjid': int(sel['subjid']),
            'accuracy': float(sel['accuracy']),
            'bin_lo_acc': float(bin_slice['accuracy'].min()),
            'bin_hi_acc': float(bin_slice['accuracy'].max()),
            'bin_size': len(bin_slice),
        })
    chosen_df = pd.DataFrame(chosen)

    os.makedirs(os.path.dirname(a.out_subjids) or '.', exist_ok=True)
    with open(a.out_subjids, 'w') as f:
        for r in chosen:
            f.write(f"{r['subjid']}\n")
    chosen_df.to_csv(a.out_summary, index=False)

    print(f"\nChose {len(chosen_df)} students")
    print(f"  subjid range: {chosen_df.subjid.min()}–{chosen_df.subjid.max()}")
    print(f"  accuracy range: {chosen_df.accuracy.min():.3f}–{chosen_df.accuracy.max():.3f}")
    print(f"  full-cohort accuracy mean: {per_student.accuracy.mean():.3f}")
    print(f"  subset accuracy mean:      {chosen_df.accuracy.mean():.3f}")
    print(f"\nSaved subjids -> {a.out_subjids}")
    print(f"Saved summary -> {a.out_summary}")


if __name__ == '__main__':
    main()
