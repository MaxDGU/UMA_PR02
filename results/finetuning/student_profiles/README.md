# Student-conditioned Qwen3-4B: per-profile SP2013 matching

Can a single Qwen3-4B model simulate the full **distribution** of UMA students
on SP2013 — not just the population mean? This experiment adds explicit
`<student> g X d Y rt Z ice W </student>` conditioning at distillation and
checks whether the trained model reproduces low / mid / high performer
accuracy curves when prompted with the corresponding profile.

Headline (`qwen4b_student_profile_match.png`, SP2013 @ T=1.0, 256 rollouts/problem):

| Profile | UMA target acc | Qwen3-4B (cond.) acc | MAE pp vs UMA target |
|---|---:|---:|---:|
| low  | 0.34 | 0.56 | 22.0 |
| mid  | 0.54 | 0.59 | 21.0 |
| high | 0.71 | 0.63 | 16.1 |

The model's mean accuracy moves **monotonically** with profile
(0.56 → 0.59 → 0.63), in the same direction as UMA's spread
(0.34 → 0.54 → 0.71). Conditioning works directionally; it does
**not** fully recover the UMA spread (Δ=0.07 for the model vs Δ=0.37 for
UMA). The model under-suppresses its baseline capability when conditioned
on a low-performer profile — green bars sit above blue across the entire
low panel. The high-performer panel matches UMA most closely.

## Recipe

**Profile definition** (`build_student_profiles.py`):
1. Compute per-student SP2013 accuracy across all 996 UMA students from
   `UMA_replication/verification_full_results.csv` (15,936 rows, 996 × 16
   problems).
2. Sort by accuracy and split into low/mid/high tertiles
   (332/332/332 students).
3. Median-accuracy student per tertile is the "rep". Reps used:
   - low:  `g=0.01, d=0.1, rt_mu=3, ice=100`  (acc 0.375)
   - mid:  `g=0.07, d=0.1, rt_mu=5, ice=25`   (acc 0.5625)
   - high: `g=0.10, d=0.7, rt_mu=4, ice=25`   (acc 0.6875)
4. Per-profile per-problem **UMA target** = mean of 332 students'
   accuracy on each SP2013 problem (`student_profiles_targets.csv`).

**Student-conditioned distillation**: re-distilled Qwen3-4B with the same
hyperparameters as the trajectory experiment but with
`--train_prompt_student_mode=always`. Every training row's prompt becomes:

```
<student> g <g> d <d> rt <rt_mu> ice <ice> </student>
Solve this fraction problem: <prob>=?
```

derived from each row's UMA parameters. LoRA r=16 α=32, eff_bs=128,
LR=5e-4, 1 epoch, max_len=288. 460,185 training rows. Final val_loss
**0.0183** (cf. trajectory unconditioned distill val_loss ≈ 0.2) —
student conditioning makes the task highly determined.

**Per-profile inference**: prompt the trained model with each profile's
rep parameters; 256 rollouts per SP2013 problem at T=1.0, top_p=0.95,
top_k=50. Score with the fixed-parser `extract_answer`.

## Files

```
student_profiles/
├── qwen4b_student_profile_match.png      # 3-panel low/mid/high figure
├── student_profiles.csv                  # 996 students × {subjid, params, acc, profile}
├── student_profiles_summary.csv          # per-profile reps + acc stats
├── student_profiles_targets.csv          # per-profile per-problem UMA accuracy target
├── per_problem/                          # one row per SP2013 problem × profile
│   ├── sp2013_per_problem_q3_4b_studentcond_{low,mid,high}.csv
│   └── sp2013_summary_q3_4b_studentcond_{low,mid,high}.json
├── rollouts/                             # raw model outputs (3 × 4,096 rollouts each)
│   └── rollouts_q3_4b_studentcond_{low,mid,high}.csv.gz
├── build_student_profiles.py             # profile assignment + UMA targets
├── eval_qwen3_studentcond_sp2013.py      # eval driver with --student_g/d/rt/ice
├── run_eval_studentcond.slurm            # sbatch wrapper around the eval
├── submit_studentcond_evals.sh           # submit 3 profile evals at once
└── plot_student_profile_match.py         # 3-panel figure
```

## Reproducing

1. Build profiles from `verification_full_results.csv`:
   ```
   python build_student_profiles.py
   ```
2. Re-distill Qwen3-4B with `TRAIN_PROMPT_STUDENT_MODE=always` (use the
   trajectory `slurm_train_qwen3_distill_1epoch.sbatch` template; export
   the env var explicitly).
3. Update `submit_studentcond_evals.sh` `ADAPTER_DIR` to point at your
   trained LoRA (`best/` or `final/`) and run:
   ```
   bash submit_studentcond_evals.sh
   ```
4. Plot:
   ```
   python plot_student_profile_match.py \
     --eval_dir <path to v2 eval dir>
   ```

## Caveats

- The model was distilled on `uma_fraction_distillation_25` (panel25
  synthetic UMA traces, ~460k rows). The held-out target distribution
  comes from `verification_full_results.csv` (the canonical UMA-on-SP2013
  per-student responses). These are independent datasets so per-problem
  alignment is not by construction.
- The cheap-fix humanFT was **not** applied here — the comparison is
  against the conditioned distill output. HumanFT targets a single human
  distribution and would collapse the per-profile differences.
- Profile reps are *medians* of accuracy-stratified groups, not
  parameter-space cluster centers. The rep parameters happen to sit at
  different (g, d, rt_mu, ice) corners but the population groups are
  defined by performance, not parameters.
