# NeurIPS Human-Likeness Preference Reproduction Bundle

This folder preserves raw traces, exact preference pairs, and Gemini judge
outputs for the NeurIPS-style human-likeness evaluation.

## Layout

- `raw/decimal/`: individual decimal trace CSVs for the shared baselines and
  Qwen3-4B human-FT/distillation output.
- `raw/fraction/`: individual fraction trace CSVs for the shared baselines,
  Qwen3-8B human-FT/distillation output, and the older fraction reproduction
  traces used for the NeurIPS-style preference runs.
- `pairs/fraction/<run>/`: exact `entity_samples.csv` and `pairings.csv` files
  used to build the judged fraction comparisons.
- `gemini_judge/fraction/<run>/`: Gemini judge outputs, including parsed judge
  CSVs, raw request/response JSONL, prompt, manifest, and summaries.
- `figures/`: CSV data behind the human-likeness preference bars.

## Fraction Pair Runs

- `fraction_human_ft_5way_v1_gemini31_flash_lite_judge_with_direct_uma`:
  Distilled panel25 vs Gemini 2.5 Flash, GPT-4.1 mini, Human, and Direct UMA,
  plus Human vs Gemini/GPT sanity checks.
- `fraction_human_ft_centaur_20260504`: Distilled panel25 vs Centaur-70B and
  Human vs Centaur-70B.
- `qwen_frontier3_gemini31_flash_lite`: Qwen3-8B fraction vs Claude Sonnet 4.6,
  Gemini 3 Flash, and GPT-5.5 Low, judged with Gemini 3.1 Flash Lite.
- `qwen_frontier3_requests`: request-only companion for the same Qwen/frontier
  fraction comparisons.
- `qwen_baselines_requests`: request-only Qwen3-8B fraction comparisons against
  Human, UMA, and Centaur-70B after low-count frontier baselines were skipped.
- `qwen_baselines_smoke_gemini`: one-pair smoke run covering Qwen3-8B fraction
  against all available baselines.

## Deliberate Exclusion

Raw UMA trace/source data is not included. The Direct UMA comparison is included
through the exact `pairings.csv`, `requests.jsonl`, `responses_raw.jsonl`, and
`predictions.csv` files, which preserve the judged pairs and outputs without
shipping the full raw UMA trace bank.

## Key Counts

- Main + Direct UMA judge run: 4,800 judged pairs, 8 fraction problems, 6
  preference sets.
- Centaur judge run: 1,600 judged pairs, 8 fraction problems, 2 preference
  sets.
- Judge model in the reproduced NeurIPS rows: `gemini-3.1-flash-lite-preview`,
  temperature `0.0`.

The per-run `manifest.json` and `summary_metrics.json` files record the prompt
version, seeds, backend, and source paths from the original run.
