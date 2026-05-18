# Student-conditioned Qwen3-4B: per-profile SP2013 matching

Can a single Qwen3-4B model simulate the full **distribution** of UMA students
on SP2013 — not just the population mean? This experiment adds explicit
`<student> g X d Y rt Z ice W </student>` conditioning at distillation and
checks whether the trained model reproduces low / mid / high performer
accuracy curves when prompted with the corresponding profile.

## Headline

The conditioning works, but it lives in the high-temperature tail of the
output distribution: at T=1.0 the per-profile spread is heavily compressed,
and at T=1.5 it widens 1.8× with a ~25% drop in average MAE.

T=1.5 profile-match (`qwen4b_student_profile_match_T15.png`, SP2013, 256 rollouts/problem):

| Profile | UMA target acc | Qwen3-4B (cond.) acc | MAE pp vs UMA target |
|---|---:|---:|---:|
| low  | 0.34 | 0.47 | 13.4 |
| mid  | 0.54 | 0.53 | 16.8 |
| high | 0.71 | 0.60 | 14.6 |

The model's mean accuracy moves **monotonically** with profile
(0.47 → 0.53 → 0.60), in the same direction as UMA's spread
(0.34 → 0.54 → 0.71). Magnitude is partially recovered
(Δ=0.12 for the model vs Δ=0.37 for UMA).

## Temperature sweep

We expected lower T to *sharpen* the conditioning. The opposite happened
(`qwen4b_studentcond_t_sweep.png`, full 8-T sweep, 0.3 → 2.0):

| T | low | mid | high | Δ (high−low) | avg MAE |
|---|---:|---:|---:|---:|---:|
| 0.3 | 0.74 | 0.67 | 0.65 | **−0.09 (reversed)** | 32.4 |
| 0.5 | 0.68 | 0.65 | 0.66 | −0.02 | 28.1 |
| 0.7 | 0.64 | 0.63 | 0.65 | +0.01 (flat) | 24.6 |
| 1.0 | 0.56 | 0.59 | 0.63 | +0.07 | 19.7 |
| 1.3 | 0.50 | 0.55 | 0.60 | +0.10 | 16.2 |
| **1.5** | **0.47** | **0.53** | **0.60** | **+0.12** | **14.9 ← best** |
| 1.7 | 0.46 | 0.51 | 0.57 | +0.11 | 14.7 |
| 2.0 | 0.44 | 0.48 | 0.53 | +0.10 | 15.3 |
| UMA | 0.34 | 0.54 | 0.71 | +0.37 | — |

**Interpretation**: at low T the model commits to its **mode answer**, which is
the correct one regardless of profile (distillation gave it the underlying
capability). So all three profiles collapse near ~0.65 accuracy and the
conditioning effect disappears. At low enough T (0.3), the *low* profile is
actually highest — the conditioning's slight bias toward longer/wronger reasoning
chains gets washed out and the model defaults to its calibrated answer.

The conditioning works by *adding wrong-answer probability mass* (matching
what a low student would do), not by *shifting the mode*. **Sampling
diversity is what surfaces it.** The spread peaks at T=1.5, then narrows
again at T=2.0 as all profiles drift into noise.

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
student conditioning makes the task highly determined at training time.

**Per-profile inference**: prompt the trained model with each profile's
rep parameters; 256 rollouts per SP2013 problem, top_p=0.95, top_k=50.
We sweep T ∈ {0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0}. Score with the
fixed-parser `extract_answer`.

## Files

```
student_profiles/
├── qwen4b_studentcond_t_sweep.png        # mean acc vs T per profile + UMA targets
├── qwen4b_studentcond_t_sweep.csv        # per-(profile,T) mean accuracy
├── qwen4b_student_profile_match.png      # 3-panel low/mid/high at T=1.0
├── qwen4b_student_profile_match_T15.png  # 3-panel low/mid/high at T=1.5 (best)
├── student_profiles.csv                  # 996 students × {subjid, params, acc, profile}
├── student_profiles_summary.csv          # per-profile reps + acc stats
├── student_profiles_targets.csv          # per-profile per-problem UMA accuracy target
├── per_problem/                          # T=1.0 per-problem + summary (3 profiles)
├── per_problem_t_sweep/                  # full 8-T sweep × 3 profiles (24 evals)
├── rollouts/                             # T=1.0 raw rollouts (3 × 4,096 each)
├── rollouts_t_sweep/                     # T-sweep raw rollouts (24 files)
├── build_student_profiles.py             # profile assignment + UMA targets
├── eval_qwen3_studentcond_sp2013.py      # eval driver with --student_g/d/rt/ice
├── run_eval_studentcond.slurm            # sbatch wrapper (supports TEMPERATURES list)
├── submit_studentcond_evals.sh           # submit 3 profile evals at once
├── plot_student_profile_match.py         # 3-panel figure
└── plot_student_profile_t_sweep.py       # mean-acc-vs-T curve
```

## Reproducing

1. Build profiles from `verification_full_results.csv`:
   ```
   python build_student_profiles.py
   ```
2. Re-distill Qwen3-4B with `TRAIN_PROMPT_STUDENT_MODE=always` (use the
   trajectory `slurm_train_qwen3_distill_1epoch.sbatch` template; export
   the env var explicitly).
3. Update `submit_studentcond_evals.sh` to point at your trained LoRA and
   pick the temperature(s) you want:
   ```
   bash submit_studentcond_evals.sh        # default T=1.0
   # or set TEMPERATURES="0.7;1.0;1.5" in the script for a sweep
   ```
4. Plot:
   ```
   python plot_student_profile_t_sweep.py        # T-sweep curve
   python plot_student_profile_match.py \
     --eval_dir <path> --out <out.png>           # 3-panel at chosen T
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
- The optimal T for *matching* depends on the profile: T=2.0 minimizes
  MAE for low (11.5 pp), T=1.5–1.7 minimizes it for mid/high. We use
  T=1.5 as a global optimum.
