# NeurIPS Human-Likeness Preference Reproduction Bundle

This folder preserves raw traces, exact preference pairs, and Gemini judge
outputs for the NeurIPS-style human-likeness evaluation.

## Layout

- `raw/decimal/`: pre-May-6 decimal trace sources, including BSS human
  responses, the original frontier decimal outputs, Centaur decimal outputs,
  and Qwen3-0.6B MAG-6H distillation traces.
- `raw/fraction/`: individual fraction trace CSVs for the shared baselines and
  the older fraction reproduction traces used for the NeurIPS-style preference
  runs.
- `pairs/fraction/<run>/`: exact `entity_samples.csv` and `pairings.csv` files
  used to build the judged fraction comparisons.
- `pairs/decimal/<run>/`: exact `entity_samples.csv` and `pairings.csv` files
  used to build the judged decimal comparisons.
- `gemini_judge/fraction/<run>/`: Gemini judge outputs, including parsed judge
  CSVs, raw request/response JSONL, prompt, manifest, and summaries.
- `gemini_judge/decimal/<run>/`: Gemini judge outputs for the decimal
  comparisons.
- `figures/`: CSV data behind the human-likeness preference bars.

## Fraction Pair Runs

- `fraction_human_ft_5way_v1_gemini31_flash_lite_judge_with_direct_uma`:
  Distilled panel25 vs Gemini 2.5 Flash, GPT-4.1 mini, Human, and Direct UMA,
  plus Human vs Gemini/GPT sanity checks.
- `fraction_human_ft_centaur_20260504`: Distilled panel25 vs Centaur-70B and
  Human vs Centaur-70B.

## Decimal Pair Runs

- `decimal_bss_full_gemini31lite`: Human and Qwen3-0.6B MAG-6H distillation
  comparisons against frontier baselines and UMA-NLP.
- `decimal_bss_centaur_20260504`: Human and Qwen3-0.6B MAG-6H distillation
  comparisons against Centaur-70B.

The decimal run manifests reference
`raw/decimal/decimal_bss_trace_strategy_sample100_mag6h_v12_20260501_inputs/decimal_bss_trace_sample100_per_problem.csv`
as the compact sampled trace bank used to construct the decimal entities. That
source bank includes Qwen3-1.7B rows, but the committed decimal pairings use
`qwen3_0p6b_mag6h` for the model-vs-baseline comparisons.

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
- Decimal full judge run: 10,800 judged pairs, 12 decimal problems, 9
  preference sets.
- Decimal Centaur judge run: 2,400 judged pairs, 12 decimal problems, 2
  preference sets.
- Judge model in the reproduced NeurIPS rows: `gemini-3.1-flash-lite-preview`,
  temperature `0.0`.

The per-run `manifest.json` and `summary_metrics.json` files record the prompt
version, seeds, backend, and source paths from the original run.
