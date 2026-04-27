"""Plot 4-way + scaling comparison: base / distilled-60 / distilled-100 / UMA-60 / UMA-1000.

Inputs:
  --eval-60       raw_generations.csv from base+distilled-60 eval (provides 'base' and 'distilled-60')
  --eval-100      raw_generations.csv from distilled-100 eval (provides 'distilled-100' rows)
  --uma-60-csv    per-trial UMA eval on subjids 0..59
  --uma-1000-csv  per-trial UMA eval on subjids 0..999
  --out-dir       where to write outputs
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-60', required=True)
    p.add_argument('--eval-100', required=True)
    p.add_argument('--uma-60-csv', required=True)
    p.add_argument('--uma-1000-csv', required=True)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    e60 = pd.read_csv(a.eval_60)
    e100 = pd.read_csv(a.eval_100)
    u60 = pd.read_csv(a.uma_60_csv)
    u1000 = pd.read_csv(a.uma_1000_csv)

    # Build per-problem accuracy frames with consistent (model, set, prob, accuracy) shape
    frames = []
    # base + distilled-60 from eval_60
    e60_acc = (e60.groupby(['model', 'set', 'prob'])['is_correct'].mean()
                  .reset_index().rename(columns={'is_correct': 'accuracy'}))
    e60_acc['model'] = e60_acc['model'].map({'base': 'base', 'distilled': 'distilled-60'})
    frames.append(e60_acc)

    # distilled-100 from eval_100 (drop the base copy)
    e100_acc = (e100[e100['model'] == 'distilled']
                .groupby(['model', 'set', 'prob'])['is_correct'].mean()
                .reset_index().rename(columns={'is_correct': 'accuracy'}))
    e100_acc['model'] = 'distilled-100'
    frames.append(e100_acc)

    # UMA-60 + UMA-1000 from per-trial UMA evals
    u60_acc = (u60.groupby(['set', 'prob'])['is_correct'].mean()
                  .reset_index().rename(columns={'is_correct': 'accuracy'}))
    u60_acc['model'] = 'uma-60'
    u1000_acc = (u1000.groupby(['set', 'prob'])['is_correct'].mean()
                     .reset_index().rename(columns={'is_correct': 'accuracy'}))
    u1000_acc['model'] = 'uma-1000'
    for f in (u60_acc, u1000_acc):
        frames.append(f[['model', 'set', 'prob', 'accuracy']])

    combined = pd.concat(frames, ignore_index=True)

    # Per-operation summary
    rows = []
    for m in ['base', 'distilled-60', 'distilled-100', 'uma-60', 'uma-1000']:
        for s in ['sd_add', 'sd_mul']:
            sub = combined[(combined['model'] == m) & (combined['set'] == s)]
            rows.append({'model': m, 'set': s, 'mean_pct': sub['accuracy'].mean() * 100,
                         'n_problems': len(sub)})
        all_sub = combined[combined['model'] == m]
        rows.append({'model': m, 'set': 'ALL', 'mean_pct': all_sub['accuracy'].mean() * 100,
                     'n_problems': len(all_sub)})
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(a.out_dir, 'comparison_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(summary.pivot_table(index='set', columns='model', values='mean_pct').round(1).to_string())
    print(f"\nSaved -> {summary_path}")

    raw_path = os.path.join(a.out_dir, 'per_problem_accuracy.csv')
    combined.to_csv(raw_path, index=False)
    print(f"Saved per-problem accuracy -> {raw_path}")

    # ---------------- Bar chart ----------------
    sns.set_style('ticks')
    sns.set_context('paper')
    plt.rcParams['font.family'] = 'sans-serif'

    model_order = ['base', 'distilled-60', 'distilled-100', 'uma-60', 'uma-1000']
    model_labels = {
        'base': 'SmolLM2 (base)',
        'distilled-60': 'Distilled, 60 students\n(low-quality cohort)',
        'distilled-100': 'Distilled, 100 students\n(stratified)',
        'uma-60': 'UMA, 60 students\n(low-quality cohort)',
        'uma-1000': 'UMA, full 1000\nstudents',
    }
    set_order = ['sd_add', 'sd_mul']
    set_labels = {'sd_add': 'Single-digit\naddition (n=81)',
                  'sd_mul': 'Single-digit\nmultiplication (n=64)'}
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, 5))

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)
    width = 0.16
    x = np.arange(len(set_order))
    for i, m in enumerate(model_order):
        vals = []
        errs = []
        for s in set_order:
            sub = combined[(combined['model'] == m) & (combined['set'] == s)]['accuracy']
            vals.append(sub.mean() * 100)
            errs.append(sub.std() / np.sqrt(max(len(sub), 1)) * 100)
        ax.bar(x + (i - 2) * width, vals, width, yerr=errs, capsize=2,
               color=colors[i], label=model_labels[m], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([set_labels[s] for s in set_order])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=7.5, frameon=False)
    sns.despine()
    plt.tight_layout()
    bar_path = os.path.join(a.out_dir, 'fig_per_operation_accuracy_4way.pdf')
    plt.savefig(bar_path, bbox_inches='tight')
    plt.savefig(bar_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bar plot -> {bar_path}")

    # ---------------- Per-problem scatter ----------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=300, sharey=True)
    for j, s in enumerate(set_order):
        ax = axes[j]
        sub = combined[combined['set'] == s].copy()
        order = (sub[sub['model'] == 'uma-1000']
                 .sort_values('accuracy')['prob'].tolist())
        prob_to_x = {p: i for i, p in enumerate(order)}
        sub['x'] = sub['prob'].map(prob_to_x)
        for i, m in enumerate(model_order):
            d = sub[sub['model'] == m].sort_values('x')
            ax.scatter(d['x'], d['accuracy'] * 100, s=12, alpha=0.7,
                       color=colors[i], label=model_labels[m].replace('\n', ' ') if j == 0 else None,
                       edgecolor='none')
        ax.set_title(set_labels[s].replace('\n', ' '), fontsize=9)
        ax.set_xlabel('Problem (sorted by UMA-1000 accuracy)')
        ax.set_ylim(-3, 110)
        if j == 0:
            ax.set_ylabel('Accuracy (%)')
            ax.legend(loc='lower right', fontsize=6.5, frameon=False)
    sns.despine()
    plt.tight_layout()
    scatter_path = os.path.join(a.out_dir, 'fig_per_problem_accuracy_4way.pdf')
    plt.savefig(scatter_path)
    plt.savefig(scatter_path.replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"Saved per-problem plot -> {scatter_path}")


if __name__ == '__main__':
    main()
