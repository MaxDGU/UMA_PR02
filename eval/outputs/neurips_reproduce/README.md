# NeurIPS Human-Likeness Preference Reproduction Bundle

This folder preserves the fraction human-likeness preference artifacts used for
the NeurIPS-style evaluation.

## Contents

- `raw_non_uma/`: raw source data for all non-UMA sources used by the fraction
  preference comparisons.
  - `human_ft/data.csv`: human fraction fine-tuning responses.
  - `panel25_distilled/panel25_distilled_human_ft_rollouts.csv.gz`: distilled
    panel25 model rollouts.
  - `frontier_human_ft_max_prompt/gemini_2p5_flash_2000_repaired/parsed_samples.csv`:
    Gemini 2.5 Flash responses.
  - `frontier_human_ft_max_prompt/gpt_4p1_mini_2000_repaired/parsed_samples.csv`:
    GPT-4.1 mini responses.
  - `centaur_70b/centaur_vllm_fraction_humanft8_s100_outputs.csv`: Centaur-70B
    fraction responses.
- `judge_runs/fraction_human_ft_5way_v1_gemini31_flash_lite_judge_with_direct_uma/`:
  exact pairs, requests, raw judge responses, parsed predictions, and summaries
  for Distilled panel25 vs Gemini 2.5 Flash, GPT-4.1 mini, Human, and Direct
  UMA, plus Human vs Gemini/GPT sanity checks.
- `judge_runs/fraction_human_ft_centaur_20260504/`: exact pairs, requests, raw
  judge responses, parsed predictions, and summaries for Distilled panel25 vs
  Centaur-70B and Human vs Centaur-70B.
- `figures/`: CSV data behind the human-likeness preference bars.

## Deliberate Exclusion

Raw UMA trace/source data is not included. The Direct UMA preference run is
included through `pairings.csv`, `requests.jsonl`, `responses_raw.jsonl`, and
`predictions.csv`, which preserve the exact judged pairs and outputs without
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
