# Qwen3-1.7B Fraction and Human-FT Report Export

Question. We report the strongest Qwen3-1.7B fraction result from `reports/fraction_qwen_20260428` and summarize the best available human-finetuning condition from the April 29/30 human-FT reports.

Setup. MAG-H is the mean absolute 8-cell accuracy gap against the human SP2013 profile, in percentage points. The distillation rows come from `reports/fraction_qwen_20260428/tables/temperature_summary.csv`. The human-FT rows come from `reports/human_ft_qwen_compare_20260429/tables/best_magh_combined.csv` and `reports/human_ft_qwen1p7b_trainsetup_20260429/tables/temperature_summary.csv`. No new training, model reload, or rollout re-scoring was run for this export.

## Headline result

Finding. The selected condition is `q3_1p7b_lora_e2_t13` at T=1.3: MAG-H 7.18 pp, MAG-U 4.69 pp, mean accuracy 55.08%, any-correct 97.50%, parseable 100.00%.

Finding. The original high-LR human-FT prompt sweep moved the model away from the human SP2013 profile. In the later training-setup diagnostic, a much lower learning rate (`1e-6`) with effective batch 128 produced the best human-FT row: MAG-H 4.81 pp at T=1.3. That row slightly improves on the plain pre-humanFT diagnostic row by MAG-H, while preserving similar answer-distribution TV.

| condition | source | prompt/objective | T | MAG-H pp | MAG-U pp | mean acc | answer TV | parseable |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `q3_1p7b_lora_e2_t13` | fraction distill | panel25 LoRA E2 | 1.3 | 7.18 | 4.69 | 55.08 | --- | 100.00 |
| `q3_1p7b_pre_humanft_plain_t11` | pre-humanFT diagnostic | plain | 1.1 | 5.16 | 9.02 | 51.09 | 18.34 | 99.99 |
| `q3_1p7b_humanft_masked_final_t09` | human-FT prompt sweep | masked SFT, LR 5e-4 | 0.9 | 17.83 | 18.62 | 38.42 | 45.15 | 98.98 |
| **`q3_1p7b_humanft_lr1e6_b128_best_t13`** | human-FT setup diagnostic | plain SFT, LR 1e-6 | 1.3 | **4.81** | 10.03 | 49.04 | 18.19 | 99.99 |
| `q3_1p7b_humanft_lr1e6_b64_e50_final_t11` | human-FT setup diagnostic | plain SFT, LR 1e-6, 50 epochs | 1.1 | 4.90 | 9.46 | 51.15 | 17.59 | 99.99 |

## Prompt formats

Setup. Prompt labels describe the input template, not just a decoding setting.

| label | meaning |
|---|---|
| `plain` | No student prefix: `Solve this fraction problem: 2/3 + 3/5=?`. |
| `masked_student` | Adds `<student> g ?? d ?? rt ?? ice ?? </student>`; parameters are unknown, not loss-masked. |
| `student_id` | Adds `<student> subjid ... </student>`; eval uses 36 subject IDs x 30 rollouts per ID. |
| `student_values` / `panel25` | Adds concrete UMA parameters `(g,d,rt,ice)`; `panel25` uses 25 tuples x 40 rollouts. |

Method. `pre-humanFT` evaluates the distilled checkpoint before human-data updates. `human-FT` continues from a distilled checkpoint. The low-LR rows keep the same `plain` prompt while reducing the human-FT update size.

## Distillation temperature sweep

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
- `human_ft_comparison.csv`: selected distillation, pre-humanFT, and human-FT comparison rows from the April 29/30 reports.
- `operation_temperature_summary.csv`: report-derived operation accuracies across temperatures.
- `cell_temperature_summary.csv`: report-derived 8-cell accuracies across temperatures.
- `sp2013_summary_q3_1p7b_lora_e2_t13.json`: selected-condition summary with paths back to the source metrics and rollouts.
- `sp2013_summary_q3_1p7b_humanft_lr1e6_b128_t13.json`: best available human-FT condition by MAG-H among the exported 1.7B human-FT reports.
- `training_args.json`, `training_summary_metrics.json`, `training_history.json`: provenance files kept for consistency with the 4B/8B folders.

## Caveat

This export intentionally does not include per-problem MAE CSVs, because the 1.7B source reports record MAG-H/MAG-U and operation/cell summaries. Recomputing per-problem MAE would require reading the raw rollout `.csv.gz` files again and would answer a different question from the source report headline.
