# Eval Inference And Metrics

This folder keeps the shared frontier arithmetic problem sets, inference runner,
stable CSV outputs, and reproducible NeurIPS evaluation entrypoints together.
It is self-contained for the shared frontier/Centaur evaluations: the former
`procedural_alignment` helpers used by these entrypoints have been migrated
into standalone scripts in this folder.

## Data

- `data/fraction_problems.csv`: 16 SP2013 fraction problems.
- `data/decimal_problems.csv`: 12 BSS2021 decimal problems.
- `data/siegler_fraction_human.csv`: SP2013 human accuracy and coarse
  strategy-reference data for MAE-H, MAG-H, and procedural summaries.
- `data/human_ft/data_train.csv` and `data/human_ft/data_val.csv`: fraction
  human-FT references used only when the standalone strategy extractor is run
  with human few-shot examples or no `--input_csv`.

The `prob` column is the canonical prompt used by default. Fraction data also
includes:

- `distillation_problem`: spaced operators and `/` for division.
- `legacy_problem`: no-space operators and `÷` for division, matching the older
  GPT-4.1-mini and Gemini-2.5 imported outputs.

## Run Inference

Run all models for both domains:

```bash
python eval/run_inference.py
```

Run one model/domain:

```bash
python eval/run_inference.py --domain fraction --models gpt_5_5_low
```

Use the distillation-style fraction prompts:

```bash
python eval/run_inference.py \
  --domain fraction \
  --models gpt_4_1_mini,gemini_2_5_flash \
  --fraction-problem-column distillation_problem
```

By default, completed runs are copied into `eval/outputs/<domain>/<model>.csv`.
Sandbox models need `AI_SANDBOX_KEY`; Gemini models need `GEMINI_API_KEY`.

The lower-level standalone runner is also available directly:

```bash
python eval/frontier_arithmetic_baseline.py \
  --domain fraction \
  --backend gemini \
  --model gemini-2.5-flash \
  --model_label "Gemini 2.5 Flash" \
  --samples_per_problem 5
```

Sync saved Centaur runs into the same stable output layout:

```bash
python eval/sync_centaur_outputs.py --force
```

This writes `eval/outputs/fraction/centaur_70b.csv`,
`eval/outputs/decimal/centaur_70b.csv`, and the 8-problem human-FT surface at
`eval/outputs/fraction_humanft8/centaur_70b.csv`.

## Our Distilled + Human-Finetuned Qwen3 Outputs

Raw natural-language outputs from our cognitive-distillation + human-fine-tuned
Qwen3 checkpoints, in the same schema as the frontier model CSVs (one row per
problem-sample):

- `outputs/fraction/qwen3_8b_distill_humanft.csv` — Qwen3-8B distill + humanFT
  on the 16 SP2013 fraction problems (256 samples / problem, $T{=}1.0$).
- `outputs/decimal/qwen3_4b_distill_humanft.csv` — Qwen3-4B distill + humanFT
  on the 12 BSS2021 decimal problems (120 samples / problem, $T{=}1.0$).

These are the configurations reported in the paper's accuracy-profile alignment
results. Model checkpoints (LoRA adapters) are not pushed to the repo
(>100 MB per file); contact Max if you need them.

## Frontier Baseline Sample Counts

The frontier baselines used in the paper's accuracy-profile alignment figure
were sampled 100 times per problem (fractions) or 120 times per problem
(decimals) at $T{=}1.0$. Current sample counts in this folder:

| Model | `outputs/fraction/` | `outputs/decimal/` |
|---|---|---|
| Centaur-70B            | 100 | 100 |
| Claude Sonnet 4.6      | 100 | 120 |
| Gemini 3 Flash         | 100 | 120 |
| GPT-5.5 low            | 100 | 120 |
| Gemini 2.5 Flash       | 5   | 120 |
| GPT-4.1 mini           | 5   | 120 |

The two 5-sample fraction files (Gemini 2.5 Flash, GPT-4.1 mini) are from an
earlier sampling batch and have not been re-run at $100$ samples; the paper's
figure uses these models only on the decimal panel.

## NeurIPS Evaluations

Run answer-error alignment on saved outputs:

```bash
python eval/evaluate_error_alignment.py
```

This writes:

- `eval/results/error_alignment/per_model_summary.csv`
- `eval/results/error_alignment/per_problem.csv`
- `eval/results/error_alignment/per_cell.csv`

`MAE-H` is the mean absolute per-problem accuracy gap against
`data/siegler_fraction_human.csv`, in percentage points. `MAG-H` is the mean
absolute 8-cell gap over fraction `operation x denom_type`, also in percentage
points.

Run procedural-alignment extraction with the default Gemini Batch judge:

```bash
python eval/evaluate_procedural_alignment.py
```

For a prompt-only dry run:

```bash
python eval/evaluate_procedural_alignment.py --backend requests --max-rows 4
```

The lower-level standalone strategy extractor is available as:

```bash
python eval/fraction_strategy_extraction.py \
  --input_csv eval/outputs/fraction/centaur_70b.csv \
  --problem_col problem \
  --trace_col model_response \
  --answer_col parsed_answer \
  --backend requests \
  --max_rows 4
```

Run humanlike preference judging for the original distilled-vs-frontier
human-FT comparison:

```bash
python eval/evaluate_humanlike_preference.py
```

This script preflights the required 8-problem distilled/frontier source files
before calling the judge. If those files are missing, it prints the exact paths
and expected CSV schemas.

The lower-level standalone preference judge is available as:

```bash
python eval/human_likeness_preference.py --backend requests \
  --baseline_csv <baseline.csv> \
  --comparison name=frontier,path=<frontier.csv>
```

Run all stages through one entrypoint:

```bash
python eval/run_neurips_evaluations.py
```

Use `--requests-only` to generate judge requests without API calls. Procedural
extraction needs `GEMINI_API_KEY`; humanlike preference needs `AI_SANDBOX_KEY`.
