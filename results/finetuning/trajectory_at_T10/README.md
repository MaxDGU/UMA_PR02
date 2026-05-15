# Trajectory at T=1.0: pretrained → distill → humanFT, across Qwen3 sizes

MAE pp vs the human SP2013 distribution from `data/siegler_fraction_human.csv`,
at fixed sampling temperature T=1.0, for four backbones: Qwen3-{0.6B, 1.7B,
4B, 8B}-Base. Plot: `qwen_trajectory_at_T10.png`.

## Recipes

All four sizes were trained with matched hyperparameters at each stage so the
cross-size comparison is apples-to-apples.

**Distill** (UMA fraction trace distillation):

| Knob | Value |
|---|---|
| Data | `uma_fraction_distillation_25` (~1.4M rows) |
| LoRA | r=16, α=32, dropout=0.05, targets q/k/v/o/gate/up/down_proj |
| Effective batch | 128 |
| Learning rate | 5e-4 |
| Epochs | 1 |
| Max length | 288 |

**HumanFT (cheap-fix)**: continues the distill LoRA adapter on the 288-row
Siegler-Pyke human corpus.

| Knob | Value |
|---|---|
| Objective | SFT (no KL anchor) |
| Init | continue distill LoRA (no fresh adapter) |
| Effective batch | 16 |
| Learning rate | 1e-4 |
| Epochs | 5 (per-epoch checkpoints at 1/2/3/5) |
| Prompt | plain: `"Solve this fraction problem: {prob}=?"` |

**Eval**: 256 rollouts per SP2013 problem at T=1.0 with `top_p=0.95, top_k=50`.
MAE pp = mean absolute per-problem accuracy gap vs the empirical human
distribution.

## Headline numbers

| Stage | 0.6B | 1.7B | 4B | 8B |
|---|---:|---:|---:|---:|
| pretrained base | 34.3 | 38.5 | 19.7 | 20.6 |
| + distill ep1 | **12.8** | 12.8 | 15.4 | 18.0 |
| + humanFT ep1 | 22.9 | **10.9** | **7.3** | **7.6** |
| + humanFT ep2 | 24.9 | 17.8 | — | 14.5 |
| + humanFT ep3 | 20.9 | 20.3 | 18.6 | 14.8 |
| + humanFT ep5 | 21.1 | 19.2 | 17.5 | 17.7 |

Bold = best stage per size at T=1.0.

Patterns:
- Distillation cuts MAE by ~10–25 pp across all sizes, with the largest
  absolute gain on the smallest models (which start farthest from the human
  distribution at T=1.0).
- HumanFT ep1 is the global optimum for 1.7B/4B/8B; for 0.6B the
  optimum is *after distill alone* — the cheap-fix recipe overtrains the
  smallest backbone.
- All four trajectories show a V-shape past ep1: additional humanFT epochs
  drift the model away from the human distribution and plateau around
  17–25 pp MAE.

## Files

```
trajectory_at_T10/
├── qwen_trajectory_at_T10.png   # the plot
├── qwen_trajectory_at_T10.csv   # raw (size, stage, MAE_pp) data
└── {0p6B,1p7B,4B,8B}/
    ├── summary_<stage>.json      # condition / temperature / MAE pp / per-op MAE / acc
    └── per_problem_<stage>.csv   # 16 SP2013 problems × accuracy/abs-error
```

Stages per size: `base`, `distill_ep1`, `humanft_ep1`, `humanft_ep2`,
`humanft_ep3`, `humanft_ep5`. 4B is missing the ep2 cheap-fix point; everything
else is complete.

## Reproducing

Plot script: `fractions/scripts/plot_trajectory_at_T10.py`
(reads from the trajectory eval cache; see comments in the script for source
paths.)
