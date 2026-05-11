# Qwen3-0.6B Fraction and Human-FT Report Export

Question. We report the strongest Qwen3-0.6B fraction result from `reports/fraction_qwen_20260428` and summarize the best available human-finetuning condition from the April 29 human-alignment reports.

Setup. MAG-H is the mean absolute 8-cell accuracy gap against the human SP2013 profile, in percentage points. The distillation rows come from `reports/fraction_qwen_20260428/tables/temperature_summary.csv`. The human-FT rows come from `reports/human_ft_qwen_compare_20260429/tables/best_magh_combined.csv` and `reports/human_ft_qwen_sftkl_dpo_20260429/tables/best_magh.csv`. No new training, model reload, or rollout re-scoring was run for this export.

## Headline result

Finding. The selected condition is `q3_0p6b_full_t15` at T=1.5: MAG-H 6.56 pp, MAG-U 6.12 pp, mean accuracy 52.86%, any-correct 98.25%, parseable 99.92%.

Finding. Among the available 0.6B human-adaptation runs, the best human-FT condition by MAG-H is SFT+KL with coefficient 0.01 at T=0.5. It reaches MAG-H 30.77 pp and mean accuracy 23.64%. This is lower than the pure-SFT prompt-sweep rows, but remains substantially farther from the human SP2013 profile than the selected panel25 distillation condition.

| condition | source | prompt/objective | T | MAG-H pp | MAG-U pp | mean acc | answer TV | parseable |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **`q3_0p6b_full_t15`** | fraction distill | panel25 | 1.5 | **6.56** | 6.12 | 52.86 | --- | 99.92 |
| `q3_0p6b_pre_humanft_plain_t11` | pre-humanFT diagnostic | plain | 1.1 | 46.13 | 51.12 | 5.43 | 88.61 | 95.41 |
| `q3_0p6b_humanft_masked_best_t05` | human-FT prompt sweep | masked SFT | 0.5 | 49.88 | 54.86 | 1.69 | 92.51 | 100.00 |
| **`q3_0p6b_humanft_sftkl001_t05`** | human-FT objective sweep | SFT+KL 0.01 | 0.5 | **30.77** | 32.91 | 23.64 | 62.69 | 99.49 |

## Prompt formats

Setup. Prompt labels describe the input template, not just a decoding setting.

| label | meaning |
|---|---|
| `plain` | No student prefix: `Solve this fraction problem: 2/3 + 3/5=?`. |
| `masked_student` | Adds `<student> g ?? d ?? rt ?? ice ?? </student>`; parameters are unknown, not loss-masked. |
| `student_id` | Adds `<student> subjid ... </student>`; eval uses 36 subject IDs x 30 rollouts per ID. |
| `student_values` / `panel25` | Adds concrete UMA parameters `(g,d,rt,ice)`; `panel25` uses 25 tuples x 40 rollouts. |

Method. `pre-humanFT` evaluates the distilled checkpoint before human-data updates. `human-FT` continues from a distilled checkpoint. `SFT+KL` adds a KL penalty to keep the update close to the source checkpoint.

## Distillation temperature sweep

| condition | T | MAG-H pp | MAG-U pp | mean acc | any correct | parseable |
|---|---:|---:|---:|---:|---:|---:|
| `q3_0p6b_full_t03` | 0.3 | 19.25 | 14.05 | 69.40 | 91.75 | 100.00 |
| `q3_0p6b_full_t05` | 0.5 | 15.56 | 10.36 | 66.59 | 94.00 | 100.00 |
| `q3_0p6b_full_t07` | 0.7 | 12.82 | 7.62 | 64.14 | 95.00 | 100.00 |
| `q3_0p6b_full_t09` | 0.9 | 10.53 | 5.06 | 61.19 | 95.75 | 100.00 |
| `q3_0p6b_full_t10` | 1.0 | 9.76 | 4.41 | 59.86 | 96.25 | 100.00 |
| `q3_0p6b_full_t11` | 1.1 | 9.04 | 4.17 | 58.27 | 96.75 | 99.99 |
| `q3_0p6b_full_t13` | 1.3 | 7.19 | 4.94 | 55.28 | 97.25 | 99.97 |
| **`q3_0p6b_full_t15`** | 1.5 | **6.56** | 6.12 | 52.86 | 98.25 | 99.92 |
| `q3_0p6b_full_t17` | 1.7 | 7.12 | 7.43 | 50.46 | 99.00 | 99.90 |
| `q3_0p6b_full_t19` | 1.9 | 7.54 | 9.07 | 47.72 | 99.50 | 99.86 |

## Best-temperature operation profile

- Add: mean accuracy 62.42%, any-correct 100.00%.
- Div: mean accuracy 22.27%, any-correct 93.00%.
- Mul: mean accuracy 53.05%, any-correct 100.00%.
- Sub: mean accuracy 73.70%, any-correct 100.00%.

## Files

- `all_conditions_comparison.csv`: report-derived temperature sweep for this model/method block.
- `human_ft_comparison.csv`: selected distillation, pre-humanFT, and human-FT comparison rows from the April 29 reports.
- `operation_temperature_summary.csv`: report-derived operation accuracies across temperatures.
- `cell_temperature_summary.csv`: report-derived 8-cell accuracies across temperatures.
- `sp2013_summary_q3_0p6b_full_t15.json`: selected-condition summary with paths back to the source metrics and rollouts.
- `sp2013_summary_q3_0p6b_humanft_sftkl001_t05.json`: best available human-FT condition by MAG-H among the exported 0.6B human-adaptation reports.
- `training_args.json`, `training_summary_metrics.json`, `training_history.json`: provenance files kept for consistency with the 4B/8B folders.

## Caveat

This export intentionally does not include per-problem MAE CSVs, because the 0.6B source reports record MAG-H/MAG-U and operation/cell summaries. Recomputing per-problem MAE would require reading the raw rollout `.csv.gz` files again and would answer a different question from the source report headline.
