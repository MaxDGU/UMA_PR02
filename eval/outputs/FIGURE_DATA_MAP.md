# Paper figure/table → raw data map (EMNLP submission)

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

## Checkpoints (not in git — della, ask Max for transfer)
Under `/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/`:
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
