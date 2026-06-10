# Fraction (SP2013) — Qwen3-4B per-sample rollouts

These are the per-sample rollouts that back the **fractions column of Table 1**
in the EMNLP submission. Every file holds 16 SP2013 problems × 256 samples at
`T=1.0, top_p=0.95`. The schema matches `eval/outputs/fraction/qwen3_8b_distill_humanft.csv`:

```
problem, operation, denom_type, correct_answer,
model, sample_idx, parsed_answer, is_correct, model_response
```

## Files and Table-1 cross-reference

| File | Stage in paper | Acc | MAE (pp) | Reasoning quality |
|---|---|---|---|---|
| `qwen3_4b_base.csv` | Base | 24.4 | 27.8 | Brainly multiple-choice / web-text leakage |
| `qwen3_4b_humanft.csv` | + humanFT | 33.0 | 23.0 | Kid-style hedged ("I added because…") |
| `qwen3_4b_distill.csv` | + distill | 65.9 | 14.4 | UMA-style procedural narration ("I took 5 plus 5…") |
| `qwen3_4b_distill_humanft.csv` | + distill + humanFT | 47.5 | 5.2 | Kid-style terse ("I just did it i got 4/5") |

## Source checkpoints

All four stages use **Qwen3-4B-Base** + a 142 MB LoRA adapter (base for the
`Base` row, no adapter):

| Stage | Checkpoint (under `UMA_PR02_feat_humanft/results/transformer_replication/`) |
|---|---|
| + distill | `qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora/best` |
| + distill + humanFT | `qwen_humanft_from_distill_4b_ep1_20260503_124020/best` |
| + humanFT (from base) | `qwen_humanft_frombase_4b_20260512_221255/best` |

## Source rollouts

- `qwen3_4b_base.csv`, `qwen3_4b_distill.csv`, `qwen3_4b_distill_humanft.csv`
  come from a single trajectory file with stages `base`, `distill_ep1`, `hft_ep1`
  in `results/finetuning/trajectory_at_T10/rollouts/rollouts_4B.csv.gz`.
- `qwen3_4b_humanft.csv` comes from
  `qwen3_frombase_trajectory_t10_fixed/rollouts_fbtraj_4b_hftep1.csv.gz`
  (the from-base humanFT trajectory at epoch 1).

## Notes

- `denom_type` is `ED` if the two fraction operands share a denominator and
  `UD` otherwise (matches the per-cell definition used in
  `Section: Error Distribution Alignment`).
- Some `Base` rows have an empty `model_response` and `parsed_answer = NaN`:
  these are samples where the base model produced no parseable answer at all.
- `humanFT` and `distill+humanFT` rollouts inherit the SP2013 narrated style
  because the human fine-tuning targets are first-person Siegler 2011 traces.
