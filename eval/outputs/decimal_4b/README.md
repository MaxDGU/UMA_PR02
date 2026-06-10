# Decimal (BSS2021) — Qwen3-4B per-sample rollouts

These are the per-sample rollouts that back the **decimals column of Table 1**
in the EMNLP submission. Every file holds 12 BSS2021 problems × 120 samples at
`T=1.0, top_p=0.95`. The schema matches `eval/outputs/decimal/qwen3_4b_distill_humanft.csv`:

```
problem, operation, operands, correct_answer,
model, sample_idx, parsed_answer, is_correct, model_response
```

## Files and Table-1 cross-reference

| File | Stage in paper | Acc | MAE (pp) | Reasoning quality |
|---|---|---|---|---|
| `qwen3_4b_base.csv` | Base | 29.9 | 34.3 | Brainly metadata leakage |
| `qwen3_4b_humanft.csv` | + humanFT | 65.6 | 6.2 | Template (`"My answer is X."`) |
| `qwen3_4b_distill.csv` | + distill | 82.6 | 18.3 | First-person procedural ("The dots are throwing me off…") |
| `qwen3_4b_distill_humanft.csv` | + distill + humanFT | 64.3 | 6.5 | Template (`"My answer is X."`) |

## Source checkpoints

All four stages use **Qwen3-4B-Base** + a 142 MB LoRA adapter (base for the
`Base` row, no adapter):

| Stage | Checkpoint (under `UMA_PR02_feat_humanft/results/transformer_replication/`) |
|---|---|
| + distill | `qwen3_4b_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260518_134624_lora/best` |
| + distill + humanFT | `qwen3_4b_humanft_bss2021_from_distill_20260520_143350/best` |
| + humanFT (from base) | `qwen3_4b_humanft_bss2021_frombase_20260521_193439/best` |

## Source per-sample CSVs

Each file is `bss2021_eval_*/per_sample.csv` from the corresponding checkpoint's
eval folder, re-shaped to the schema above (added `sample_idx` per problem and
`model` column, renamed `response` → `model_response`).

## Why two of the four files are template-only

The BSS2021 human dataset records only **final answers**, not verbal reasoning.
The `response_nl` column we trained the humanFT objective on is therefore a
reasoning-free template (`"My answer is X.\n### answer: X"`). With an empty
reasoning slot to imitate, the model learned to emit final answers only — so
`humanFT` and `distill + humanFT` both collapse to `"My answer is X."`. Only
the `distill` row preserves real first-person reasoning, because its training
target was the natural-language rendering of UMA traces.

A remediated rerun that turns BSS2021 coded fields (`work_layout`, `dplace`,
`strat_conf`, `conc_*`) into first-person verbal traces lives in
[`../decimals_new/`](../decimals_new/) (checkpoint
`qwen3_4b_humanft_bss2021_reasoning_decimalprompt_20260524_173757/best`,
Acc 69.2, MAE 6.86 pp). That checkpoint is **not** the one Table 1 currently
reports.

## Notes

- `operands` is `EDD` (equal decimal places, e.g. `12.3+5.6`), `UDD` (unequal
  decimal places, e.g. `2.46+4.1`), or `D-W` (one decimal and one whole-number
  operand, e.g. `5.61+23`) — see `Section: Error Distribution Alignment`.
- `MAE` here is the per-cell `mag_h_pp` (mean absolute gap to the BSS2021 human
  accuracy profile in percentage points) reported in each checkpoint's
  `bss2021_eval_*/summary.json`.
