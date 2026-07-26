# Paper figure/table → raw data map (EMNLP submission)

Plotting scripts for every figure live in `writing/figure_scripts/` (paths inside
them point at della; swap the input paths for the copies in this directory).

Every figure and table in the paper repo (ch-shin/emnlp202026-humanlike-math-reasoning)
maps to per-sample inference outputs in this directory. All rollouts: T=1.0,
top_p=0.95 unless noted. "Fixed parser" = mixed-number-aware final-answer parser
(see `seed_variance/seed_stats.py` for the scoring functions used in the final pass).

## Table 1 (MAE/NLL, both base models) + accuracy bar chart
- Base 4B, all 4 stages: `fraction_4b/` and `decimal_4b/` (256 samples/problem).
  `*_v1.csv` = original parser backups; the non-suffixed files are re-scored.
- Instruct 4B (canonical lr=5e-5 ep3 checkpoint): `fraction_4bi/`, `decimal_4bi/`.
- Human refs: SP2013 `…/fractions/data/siegler_fraction_human.csv`; BSS2021 per-cell
  from the bss2021_eval under the 4b humanFT run (see paper repo tables/).
- UMA refs: `uma_reference/verification_full_results.csv.gz` (996 trained learners ×
  16 SP2013 problems, the same panel used for distillation) and
  `uma_reference/uma_bss2021_summary.csv` (decimals).

## Figures 3–4 (per-cell fractions/decimals) and appendix Base-vs-Instruct per-cell
Same rollouts as Table 1 (`fraction_4b`, `decimal_4b`, `fraction_4bi`, `decimal_4bi`)
plus the same human/UMA references.

## Seed-variance error bars (all fine-tuning figures)
`seed_variance/`: 4 configs (base/inst × frac/dec) × seeds 43–46, 100 samples/problem,
canonical checkpoint per config (best-by-val-NLL; epoch_3 for inst_frac). Seed 42 =
the canonical Table-1 rollouts above. `seed_cell_acc.csv` = per-cell accuracy for all
5 seeds with the fixed parsers; `seed_humanft.sbatch` + `eval_seed_ckpt.py` reproduce.

## Table 3 (15-persona sweep, frontier + off-shelf Qwen)
- Frontier (Claude Sonnet 4.6, Gemini 3 Flash, GPT-5.5-low), 100 samples/problem:
  `persona_sweep/fraction/*_wide_*.csv.gz` (15 personas) and the original 10-persona
  files (`*_paper_baseline` … `*_cohort_anchored`); decimals under
  `persona_sweep/decimal/`. Runner: `persona_sweep/persona_sweep_runner.py`.
- Off-shelf Qwen3-4B-Base / Instruct-2507, 60 samples/problem:
  `persona_offshelf/*.csv.gz` (`*_tail` = last 2 base personas after a walltime kill).
  Runner: `persona_offshelf/offshelf_wide_persona.py`.

## ICL table (k ∈ {0,2,5,10})
`fewshot_icl/{fraction,decimal}/<model>_k<k>.csv.gz`, 100 samples/problem;
`summary.csv` has per-(model,k,domain) accuracy and MAE. k=0 uses identical prompt
scaffolding with no examples. Note: decimal k=0 files were re-scored after a
float-vs-string comparison bug in the runner (fixed in `run_fewshot_icl.py`).

## Figure 1 (raw off-the-shelf accuracy)
Frontier fraction rollouts: `fraction/` (paper persona, 100 samples/problem);
frontier decimals: `persona_sweep/decimal/*_paper_baseline.csv.gz`. Off-shelf
Qwen3-4B-Base: `fraction_4b/qwen3_4b_base.csv`, `decimal_4b/…`. Off-shelf
Instruct-2507 with its native chat template: `offshelf_chat/`.
Script: `writing/figure_scripts/plot_offshelf_accuracy.py`.

## MAE-bars figure (magh_baselines) + TVD figure
MAE bars: same frontier/Centaur/ours rollouts as above
(`writing/figure_scripts/plot_magh_baselines.py`). TVD: strategy-labeled
predictions per model/domain in `strategy_tvd/<domain>_<model>_predictions.csv.gz`
(`plot_tvd_baselines.py`). The Figure-3-consistent fraction comparison adds
`strategy_tvd/fraction_direct_human_ft_predictions.csv.gz`; its sampling and
Gemini classification configuration are recorded in
`strategy_tvd/fraction_direct_human_ft_classification.json`. The combined
fraction/decimal TVD table also uses
`strategy_tvd/decimal_direct_human_ft_predictions.csv.gz`; its all-row
Gemini Batch configuration is recorded in
`strategy_tvd/decimal_direct_human_ft_classification.json`.

## NLL-vs-temperature figure + Table 1 NLL column
`nll_temperature/`: `nll_tsweep_results.csv` (Base 4B), `nll_tsweep_instruct_results.csv`
(Instruct, T=1.0 row spliced from the lr=5e-5 ep3 checkpoint re-run),
`nll_tsweep_dh_newckpt.csv` (that re-run's raw sweep).

## Profile-controllability figures (UMA-prefix conditioning)
`profile_controllability/`: `base_4b_per_profile_with_uma.csv` and
`instruct_lr5e5ep3_rollouts_per_profile.csv` (per-profile accuracy under the
996-profile grid, both final checkpoints); `profile_spread_distill_full1000_per_sample.csv.gz`
(distill-checkpoint per-sample run behind the main-text profile grid).

## Persona-vs-UMA-prefix curve (Fig 12)
`persona_finetuned/`: 15-persona rollouts on the fine-tuned checkpoints —
`finetuned_wide_persona_fractions.csv.gz` (both Base and Instruct) and
`wide_persona_rollouts_instruct_4bi.csv.gz` + summaries (the lr=5e-5 ep3 re-run
used in the current figure). Script: `plot_persona_vs_uma_curve.py`.

## Appendix training-dynamics + temperature-sensitivity figures
Trajectory rollouts (all 4 scales): `results/finetuning/trajectory_at_T10/rollouts/`
(already tracked). Temperature-sweep rollouts/summaries: `temperature_sweep/`.
Scripts: `plot_trajectory_at_T10_magh.py`, `plot_frombase_trajectory_at_T10_magh.py`,
`plot_humanft_temperature_curve.py`, `plot_distill_temperature_curve.py`.

## Checkpoints
Base-4B checkpoints (both domains, incl. per-epoch fine-tuning trajectories and
the decimal CV fold models) are on HuggingFace:
**huggingface.co/MaxDGUPTA/uma-cognitive-llm-checkpoints** (private — ask Max
for access; its README has the layout and a PeftModel loading snippet).
Instruct-2507 and seed-variance checkpoints remain only on della.

della originals under `/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/`:
- Base 4B distill (fractions): `qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora/best`
- Base 4B distill+humanFT (fractions): `qwen_humanft_from_distill_4b_ep1_20260503_124020/best`
- Base 4B decimals: distill `qwen3_4b_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260518_134624_lora/best`,
  +humanFT `qwen3_4b_humanft_bss2021_from_distill_20260520_143350/best`
- Instruct 4B fractions: distill `qwen3_4bi_panel25_default_base_e1_20260624_233913_lora/best`,
  +humanFT (lr 5e-5) `qwen_humanft_from_distill_4bi_lr5e-5_20260709_135108/epoch_3`
- Instruct 4B decimals: distill `qwen3_4bi_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260624_233751_lora/best`,
  +humanFT `qwen3_4bi_humanft_bss2021_from_distill_20260624_233911/best`
- Seed-variance runs (seeds 43–46): `/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/runs/`
All are LoRA adapters (~150 MB each) with `train_args.json` inside.
