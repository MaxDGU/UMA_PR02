# Panel scaling: distilling on all 996 UMA learners vs the 25-learner panel (fractions)

Controlled comparison of the fraction pipeline when the distillation corpus is drawn from the paper's 25-learner
panel versus the full population of 996 saved UMA learners (1,000-profile grid; subjids 92, 93, 97, 98 never
finished training). Every stage follows the paper recipe; only the learner set changes. Study design and run log:
`DESIGN.md`. All rollouts: 16 SP2013 problems x 256 samples, T=1.0, top_p=0.95, scored with the
mixed-number-aware parser in `scripts/rescore_frac_dir.py`. MAE = mean absolute error vs. children over the 8
operation x denominator cells.

## Arms
| arm | learners | corpus rows | status |
|---|---|---|---|
| A | paper's 25 (g in .02-.10 even, all d, rt_mu 5, ice 50) | 518K | done |
| B | all 996, 520 problems per learner | 518K | done |
| D | 25 drawn at random from the 996 (seeds 42, 43) | 518K | done |
| balanced 98 | 10 per g value | 518K sub / 2.03M full | done |
| C | all 996, all problems | 20.6M | **failed: saved checkpoint is corrupted (see diag_C/); not a result** |

Arms A and B are matched on rows (466,200 vs 466,128 trained), updates (3,643 vs 3,642), effective batch 128,
lr 5e-4 cosine, 1 epoch; human-FT is 288 Siegler (2011) responses, 1 epoch (18 updates), lr 1e-4. Neither stage
uses early stopping: each evaluates once at the end and keeps the final checkpoint.

## Main result (arm A vs arm B, human-FT mean +- SD over seeds 42-45)
| | A: paper's 25 | B: all 996 |
|---|---|---|
| distill MAE / accuracy | 10.3 / 60.0% | 9.5 / 53.9% |
| distill val NLL, held-out children | 3.07 | 3.28 |
| distill + human-FT MAE | 7.6 +- 2.1 | 18.7 +- 3.0 (Welch p = 0.001) |
| distill + human-FT accuracy | 47.4 +- 1.5% | 32.9 +- 2.9% |
| distill + human-FT val NLL | 1.371 +- 0.013 | 1.364 +- 0.013 (p = 0.47) |
Children: 51.6% accuracy. Per-seed: `results/A_vs_B_per_seed.csv`; per-cell: `results/A_vs_B_per_cell.csv`.
The arms are nearly identical after distillation; the gap opens during human-FT and is concentrated in
unequal-denominator addition and subtraction (B: 17% and 25% vs children 51% and 58%).

## Why (kl_test/)
* Strategy consistency (`kl_test/test1_strategy_consistency.txt`): B's training data uses the common-denominator
  strategy slightly less often (67% vs 74%). After human-FT, 48% of B's unequal-denominator answers match no
  recognizable procedure (A: 25%).
* Human-FT anchored to the distilled model with a KL penalty (`kl_test/test2_results.txt`, seeds 42-43):
  | arm | kl_beta | MAE | unequal-denom +/- acc | response words | val NLL |
  |---|---|---|---|---|---|
  | A | 0 | 6.7 | 59% | 20 | 1.37 |
  | A | 1.0 | 7.2 | 69% | 46 | 1.55 |
  | B | 0 | 16.9 | 24% | 20 | 1.36 |
  | B | 1.0 | 5.6 | 51% | 43 | 1.59 |
  Anchoring removes the A-B gap at a cost in NLL. Interpretation: human-FT on short child responses erases the
  step-by-step reasoning that B's common-denominator procedure depends on.

## Files
* `results/` rollout CSVs per arm and stage (`*_distill.csv`, `*_dh*.csv`), `summary.csv`, A-vs-B tables.
  Arm A = `arm_v10_p25regen_*`; `control_published_dh.csv` = the paper's checkpoint through this harness.
* `fig/` figures and plotting scripts (`armB_vs_A_mae_nll`, `armB_vs_A_valloss_acc`), `distill_nll.csv`.
* `kl_test/`, `diag_C/` tests above. `scripts/` generation, translation, arm-building, distillation, human-FT and
  scoring scripts (paths point at della). `learner_lists/` subjid lists per arm.
* Not included (size): per-learner raw traces (~265 GB), translated corpora (1.3 GB full, 31-51 MB per arm), checkpoints.
