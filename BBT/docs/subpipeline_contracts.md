# Subpipeline Contracts

## Manifest schema

Each subpipeline writes `<output_dir>/manifest.json` with:

1. `subpipeline_name`
2. `status` (`success` or `dry_run`)
3. `started_at_utc`
4. `finished_at_utc`
5. `config_snapshot`
6. `inputs`
7. `primary_output`
8. `secondary_outputs`
9. `metrics_summary`
10. `upstream_manifest`

## I/O by subpipeline

### `uma_traces`
- Input: UMA models dir + problem CSV.
- Primary output: trace CSV.

### `translation`
- Input: upstream trace CSV (or `external_inputs.uma_trace_csv`).
- Primary output: translated NLP CSV.

### `distill_train`
- Input: upstream NLP CSV (or `external_inputs.translated_csv`).
- Primary output: distill run directory.

### `distill_eval`
- Input: distill run directory (or `external_inputs.distill_run_dir`).
- Primary output: sampled eval CSV.
- Secondary output: sampled eval metrics JSON.

### `human_finetune`
- Input: distill run directory + human train/val CSVs.
- Primary output: human finetune run directory.

### `human_eval`
- Input: human finetune run directory (or `external_inputs.human_finetune_run_dir`).
- Primary output: combined human eval metrics JSON.
- Secondary outputs: sampling CSV + sampling metrics JSON.
