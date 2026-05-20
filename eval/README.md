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
