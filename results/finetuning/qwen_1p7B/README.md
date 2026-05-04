# Qwen3-1.7B Fraction Report Export

Question. We report the strongest Qwen3-1.7B fraction result from `reports/fraction_qwen_20260428`, rather than the plain-prompt human-FT export. This folder is therefore keyed to MAG-H, the metric used in that report.

Setup. MAG-H is the mean absolute 8-cell accuracy gap against the human SP2013 profile, in percentage points. These values come directly from `reports/fraction_qwen_20260428/tables/temperature_summary.csv`; no new training, branch-dependent model reload, or rollout re-scoring was run for this export.

## Headline result

Finding. The selected condition is `q3_1p7b_lora_e2_t13` at T=1.3: MAG-H 7.18 pp, MAG-U 4.69 pp, mean accuracy 55.08%, any-correct 97.50%, parseable 100.00%.

| condition | T | MAG-H pp | MAG-U pp | mean acc | any correct | parseable |
|---|---:|---:|---:|---:|---:|---:|
| `q3_1p7b_lora_e2_t05` | 0.5 | 15.15 | 10.13 | 66.68 | 93.25 | 100.00 |
| `q3_1p7b_lora_e2_t07` | 0.7 | 12.13 | 7.14 | 63.69 | 94.75 | 100.00 |
| `q3_1p7b_lora_e2_t09` | 0.9 | 9.78 | 4.34 | 60.73 | 96.25 | 100.00 |
| `q3_1p7b_lora_e2_t10` | 1.0 | 9.15 | 4.11 | 59.56 | 96.25 | 100.00 |
| `q3_1p7b_lora_e2_t11` | 1.1 | 8.53 | 4.13 | 57.96 | 97.00 | 100.00 |
| **`q3_1p7b_lora_e2_t13`** | 1.3 | **7.18** | 4.69 | 55.08 | 97.50 | 100.00 |
| `q3_1p7b_lora_e2_t15` | 1.5 | 7.61 | 5.73 | 52.64 | 98.00 | 99.99 |
| `q3_1p7b_lora_e2_t17` | 1.7 | 8.61 | 6.66 | 50.28 | 98.75 | 99.96 |
| `q3_1p7b_lora_e2_t19` | 1.9 | 9.27 | 8.13 | 48.42 | 99.00 | 99.92 |

## Best-temperature operation profile

- Add: mean accuracy 65.60%, any-correct 100.00%.
- Div: mean accuracy 22.27%, any-correct 92.00%.
- Mul: mean accuracy 54.85%, any-correct 98.00%.
- Sub: mean accuracy 77.60%, any-correct 100.00%.

## Files

- `all_conditions_comparison.csv`: report-derived temperature sweep for this model/method block.
- `operation_temperature_summary.csv`: report-derived operation accuracies across temperatures.
- `cell_temperature_summary.csv`: report-derived 8-cell accuracies across temperatures.
- `sp2013_summary_q3_1p7b_lora_e2_t13.json`: selected-condition summary with paths back to the source metrics and rollouts.
- `training_args.json`, `training_summary_metrics.json`, `training_history.json`: provenance files kept for consistency with the 4B/8B folders.

## Caveat

This export intentionally does not include Max-style per-problem MAE CSVs, because the requested source report records MAG-H/MAG-U and operation/cell summaries. Recomputing per-problem MAE would require reading the raw rollout `.csv.gz` files again and would answer a different question from the fraction report headline.
