# Qwen3-0.6B Fraction Report Export

Question. We report the strongest Qwen3-0.6B fraction result from `reports/fraction_qwen_20260428`, rather than the plain-prompt human-FT export. This folder is therefore keyed to MAG-H, the metric used in that report.

Setup. MAG-H is the mean absolute 8-cell accuracy gap against the human SP2013 profile, in percentage points. These values come directly from `reports/fraction_qwen_20260428/tables/temperature_summary.csv`; no new training, branch-dependent model reload, or rollout re-scoring was run for this export.

## Headline result

Finding. The selected condition is `q3_0p6b_full_t15` at T=1.5: MAG-H 6.56 pp, MAG-U 6.12 pp, mean accuracy 52.86%, any-correct 98.25%, parseable 99.92%.

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
- `operation_temperature_summary.csv`: report-derived operation accuracies across temperatures.
- `cell_temperature_summary.csv`: report-derived 8-cell accuracies across temperatures.
- `sp2013_summary_q3_0p6b_full_t15.json`: selected-condition summary with paths back to the source metrics and rollouts.
- `training_args.json`, `training_summary_metrics.json`, `training_history.json`: provenance files kept for consistency with the 4B/8B folders.

## Caveat

This export intentionally does not include Max-style per-problem MAE CSVs, because the requested source report records MAG-H/MAG-U and operation/cell summaries. Recomputing per-problem MAE would require reading the raw rollout `.csv.gz` files again and would answer a different question from the fraction report headline.
