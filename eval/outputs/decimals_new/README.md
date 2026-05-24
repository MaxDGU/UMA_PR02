# Decimal reasoning-trace re-run (Qwen3-4B)

This folder holds a re-run of the Qwen3-4B decimal distill+humanFT model that
fixes the **reasoning-trace quality** of the decimal panel. The original
`eval/outputs/decimal/qwen3_4b_distill_humanft.csv` produces only the bare
template `"My answer is X"` — 100% of its responses contain no reasoning. The
model here produces human-like first-person reasoning that also reproduces the
BSS2021 decimal misconceptions.

## Why the original had no reasoning

The decimal human-FT targets (`response_nl`) were themselves the reasoning-free
`"My answer is X.\n### answer: X"` template. Unlike the SP2013 fraction data,
the BSS2021 decimal human data has **no verbal strategy column** — only coded
fields (`strat_add`, `strat_mul`, `strat_conf`, `work_layout`, `work_operand_*`,
`work_answer`, `dplace`, `conc_*`). With an empty reasoning slot to imitate, the
model learned to emit answers only.

## The fix

`decimal_verbalizer.py` deterministically turns each coded human row into a
first-person trace that narrates the human's *actual* strategy and error:

- `work_operand_*` vs `op*` → whether the student stripped the decimals
- `dplace` → how they placed the point (`Mul` = count places; `Add` on a
  multiplication = the signature overgeneralized-addition error)
- `strat_conf` / `conc_*` → conceptual-error language ("I think I mixed up the
  steps", "I had trouble with the place values")

Only the `response_nl` column was changed; the train/val split (690/230 rows),
problems, and answers are identical to the original run. Qwen3-4B was then
re-fine-tuned from the same distill checkpoint with the same hyperparameters
(epochs 3, batch 2, lr 1e-4, max_length 256, kl_beta 0), and re-evaluated with
the same protocol (120 samples/problem, T=1.0) used for the published baseline.

## Result (vs. original template humanFT)

| Metric | Original (template) | This run (reasoning) |
| --- | --- | --- |
| Genuine reasoning traces | 0% | **100%** |
| Answer-only | 100% | 0% |
| Traces with human misconception language | — | 71% |
| MAG-H (per-cell gap vs BSS2021 humans) | 6.46 pp | **5.33 pp** |
| Overall accuracy | 64.3% | 69.3% |

Reasoning went from absent to universal **and** the answer-distribution
alignment improved (MAG-H 6.46 → 5.33). Example output:

> "I ignored the decimal points and multiplied 24 and 12 as whole numbers and
> got 288. Then I counted the decimal places in both numbers and moved the point
> over that many spots." (correct multiplication strategy)
>
> "I multiplied 2.1 and 0.32 and got 6.72. I lined the decimal point up under the
> others the way I would when adding. I wasn't sure if that was the right
> method." (overgeneralized-addition misconception)

## Files

| File | Description |
| --- | --- |
| `qwen3_4b_distill_humanft_reasoning.csv` | Per-sample eval output, same schema as `eval/outputs/decimal/qwen3_4b_distill_humanft.csv` (1440 rows = 12 problems × 120 samples) |
| `per_problem.csv`, `per_cell.csv`, `summary.json` | Aggregated eval metrics (incl. `mag_h_pp` = 5.33) |
| `data_bss2021_train_nl_reasoning.csv` / `_val_` | Reasoning-target training split (690 / 230 rows) |
| `decimal_verbalizer.py` | Coded-field → first-person trace generator |
| `decimal_reasoning_before_after.{png,pdf}` | Before/after trace-quality figure |

## Caveat

A pre-existing train/eval prompt mismatch is held constant across both the
original and this run: training rewrites the prompt to "Solve this **fraction**
problem: {decimal}=?" (the `plain` instruction style), while eval uses "Solve
this **decimal** problem:". A fully prompt-consistent re-run is a pending
robustness check.
