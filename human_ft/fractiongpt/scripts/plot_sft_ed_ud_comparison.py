#!/usr/bin/env python3
"""SP2013 ED/UD accuracy: Human vs UMA vs pretrained FractionGPT vs SFT+KL (sft_norm_beta10).

Recreates the style of figures/figure_ed_ud_comparison.png, adding the SFT+KL bar.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
})

PROJECT = '/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions'

# Per-problem (model %, human %) from fair eval logs at T=0.5
PRETRAINED = {
    '2/3*3/5': 54.7, '2/3+3/5': 53.2, '2/3-3/5': 57.1, '2/3:3/5':  3.9,
    '3/5*1/4': 56.1, '3/5*1/5': 50.0, '3/5+1/4': 50.0, '3/5+1/5': 66.4,
    '3/5-1/4': 61.5, '3/5-1/5': 88.3, '3/5:1/4':  7.4, '3/5:1/5': 45.5,
    '4/5*3/5': 56.9, '4/5+3/5': 65.6, '4/5-3/5': 85.9, '4/5:3/5': 12.2,
}
SFT_KL = {
    '2/3*3/5': 57.7, '2/3+3/5': 53.2, '2/3-3/5': 59.7, '2/3:3/5': 13.0,
    '3/5*1/4': 56.8, '3/5*1/5': 50.0, '3/5+1/4': 56.0, '3/5+1/5': 65.2,
    '3/5-1/4': 53.8, '3/5-1/5': 86.7, '3/5:1/4':  9.7, '3/5:1/5': 40.7,
    '4/5*3/5': 50.5, '4/5+3/5': 68.7, '4/5-3/5': 83.4, '4/5:3/5': 16.7,
}
HUMAN = {
    '2/3*3/5': 58.3, '2/3+3/5': 50.8, '2/3-3/5': 61.7, '2/3:3/5': 15.0,
    '3/5*1/4': 55.0, '3/5*1/5': 31.7, '3/5+1/4': 50.8, '3/5+1/5': 78.3,
    '3/5-1/4': 54.2, '3/5-1/5': 82.5, '3/5:1/4': 20.8, '3/5:1/5': 35.8,
    '4/5*3/5': 40.8, '4/5+3/5': 74.2, '4/5-3/5': 83.3, '4/5:3/5': 20.0,
}


def classify(prob):
    fracs = re.findall(r'(\d+)/(\d+)', prob)
    dt = 'ED' if fracs[0][1] == fracs[1][1] else 'UD'
    for op_char, op in (('+', 'Add'), ('-', 'Sub'), ('*', 'Mul'), (':', 'Div')):
        if op_char in prob:
            return op, dt
    return None, dt


def avg_by_category(pct_dict):
    bucket = {}
    for prob, pct in pct_dict.items():
        op, dt = classify(prob)
        bucket.setdefault((op, dt), []).append(pct)
    return {k: np.mean(v) for k, v in bucket.items()}


# UMA: use 15,936 traces on SP2013-style problems (same ED/UD structure)
uma_df = pd.read_csv(os.path.join(PROJECT, 'data', 'sp2013_traces.csv'))
uma_df['op'], uma_df['dt'] = zip(*uma_df['prob'].apply(classify))
uma_by_cat = (uma_df.groupby(['op', 'dt'])['is_correct'].mean() * 100).to_dict()

human_by_cat = avg_by_category(HUMAN)
pre_by_cat = avg_by_category(PRETRAINED)
sft_by_cat = avg_by_category(SFT_KL)

op_order = ['Add', 'Sub', 'Mul', 'Div']
dt_order = ['ED', 'UD']
categories = [(op, dt) for op in op_order for dt in dt_order]

human_vals = [human_by_cat[c] for c in categories]
uma_vals = [uma_by_cat[c] for c in categories]
pre_vals = [pre_by_cat[c] for c in categories]
sft_vals = [sft_by_cat[c] for c in categories]

colors = plt.cm.viridis(np.linspace(0.0, 0.9, 4))

fig, ax = plt.subplots(figsize=(8, 3.6), dpi=300)
x = np.arange(len(categories))
w = 0.2

ax.bar(x - 1.5 * w, human_vals, w, label='Human', color=colors[0])
ax.bar(x - 0.5 * w, uma_vals, w, label='UMA', color=colors[1])
ax.bar(x + 0.5 * w, pre_vals, w, label='FractionGPT (pretrained, t=0.5)', color=colors[2])
ax.bar(x + 1.5 * w, sft_vals, w, label='FractionGPT + SFT/KL (t=0.5)', color=colors[3])

for i in range(1, len(categories)):
    if categories[i][0] != categories[i - 1][0]:
        ax.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'{op}\n{dt}' for op, dt in categories])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 95)
ax.set_title('SP2013', fontweight='bold')
ax.legend(loc='upper right', frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()

out = os.path.join(PROJECT, 'figures', 'figure_ed_ud_comparison_sft.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('Saved', out)

# Print table
print('\n               Human    UMA   Pre   SFT+KL')
for c, h, u, p, s in zip(categories, human_vals, uma_vals, pre_vals, sft_vals):
    print(f'  {c[0]:3s} {c[1]:2s}: {h:5.1f}  {u:5.1f}  {p:5.1f}  {s:5.1f}')
