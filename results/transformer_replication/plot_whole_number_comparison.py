"""Plot per-operation accuracy: base SmolLM2 vs distilled vs UMA on 145 test problems.

Inputs:
  - LLM eval (base + distilled): raw_generations.csv from eval_whole_number_base_vs_distilled.py
  - UMA eval: uma_eval_60students.csv from eval_uma_on_test_problems.py
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
    p.add_argument('--llm-csv', required=True,
                   help='raw_generations.csv from base-vs-distilled eval')
    p.add_argument('--uma-csv', required=True,
                   help='uma_eval_60students.csv from UMA eval')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    llm = pd.read_csv(args.llm_csv)
    uma = pd.read_csv(args.uma_csv)

    # Per-problem accuracy for LLM models
    llm_per_prob = (llm.groupby(['model', 'set', 'prob'])['is_correct']
                       .mean().reset_index().rename(columns={'is_correct': 'accuracy'}))
    # UMA: average over students
    uma_per_prob = (uma.groupby(['set', 'prob'])['is_correct']
                       .mean().reset_index().rename(columns={'is_correct': 'accuracy'}))
    uma_per_prob['model'] = 'uma'
    uma_per_prob = uma_per_prob[['model', 'set', 'prob', 'accuracy']]

    combined = pd.concat([llm_per_prob, uma_per_prob], ignore_index=True)

    # Per-operation summary
    summary = combined.groupby(['model', 'set'])['accuracy'].agg(['mean', 'count']).reset_index()
    summary['mean_pct'] = (summary['mean'] * 100).round(1)
    overall = combined.groupby('model')['accuracy'].mean().reset_index()
    overall['set'] = 'ALL'
    overall['count'] = combined.groupby('model').size().values
    overall['mean_pct'] = (overall['accuracy'] * 100).round(1)
    overall = overall.rename(columns={'accuracy': 'mean'})
    summary = pd.concat([summary, overall[['model', 'set', 'mean', 'count', 'mean_pct']]],
                        ignore_index=True)
    summary_path = os.path.join(args.out_dir, 'comparison_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(summary.pivot_table(index='set', columns='model', values='mean_pct').to_string())
    print(f"\nSaved summary -> {summary_path}")

    # Per-problem combined CSV
    raw_path = os.path.join(args.out_dir, 'per_problem_accuracy.csv')
    combined.to_csv(raw_path, index=False)
    print(f"Saved per-problem accuracy -> {raw_path}")

    # ---------------- Plot 1: bar chart by operation ----------------
    sns.set_style('ticks')
    sns.set_context('paper')
    plt.rcParams['font.family'] = 'sans-serif'

    model_order = ['base', 'distilled', 'uma']
    model_labels = {'base': 'SmolLM2-135M (base)',
                    'distilled': 'SmolLM2-135M (distilled)',
                    'uma': 'UMA (60 students)'}
    set_order = ['sd_add', 'sd_mul']
    set_labels = {'sd_add': 'Single-digit\naddition (n=81)',
                  'sd_mul': 'Single-digit\nmultiplication (n=64)'}
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, 3))

    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    width = 0.25
    x = np.arange(len(set_order))
    for i, m in enumerate(model_order):
        vals = []
        errs = []
        for s in set_order:
            sub = combined[(combined['model'] == m) & (combined['set'] == s)]['accuracy']
            vals.append(sub.mean() * 100)
            errs.append(sub.std() / np.sqrt(len(sub)) * 100)
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2,
               color=colors[i], label=model_labels[m], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([set_labels[s] for s in set_order])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    sns.despine()
    plt.tight_layout()
    bar_path = os.path.join(args.out_dir, 'fig_per_operation_accuracy.pdf')
    plt.savefig(bar_path)
    plt.savefig(bar_path.replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"Saved bar plot -> {bar_path}")

    # ---------------- Plot 2: per-problem scatter / heatmap ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=300, sharey=True)
    for j, s in enumerate(set_order):
        ax = axes[j]
        # sort problems by UMA accuracy on this set, plot all 3 models as lines/dots
        sub = combined[combined['set'] == s].copy()
        order = (sub[sub['model'] == 'uma']
                 .sort_values('accuracy')['prob'].tolist())
        prob_to_x = {p: i for i, p in enumerate(order)}
        sub['x'] = sub['prob'].map(prob_to_x)
        for i, m in enumerate(model_order):
            d = sub[sub['model'] == m].sort_values('x')
            ax.scatter(d['x'], d['accuracy'] * 100, s=14, alpha=0.7,
                       color=colors[i], label=model_labels[m] if j == 0 else None,
                       edgecolor='none')
        ax.set_title(set_labels[s].replace('\n', ' '), fontsize=9)
        ax.set_xlabel('Problem (sorted by UMA accuracy)')
        ax.set_ylim(-3, 105)
        if j == 0:
            ax.set_ylabel('Accuracy (%)')
            ax.legend(loc='upper left', fontsize=7, frameon=False)
    sns.despine()
    plt.tight_layout()
    scatter_path = os.path.join(args.out_dir, 'fig_per_problem_accuracy.pdf')
    plt.savefig(scatter_path)
    plt.savefig(scatter_path.replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"Saved per-problem plot -> {scatter_path}")


if __name__ == '__main__':
    main()
