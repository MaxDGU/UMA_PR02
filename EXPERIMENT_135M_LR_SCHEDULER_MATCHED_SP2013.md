# 135M Matched-Format LR x Scheduler Ablation

## Goal

Test whether `SmolLM2-135M` can fit the UMA-distillation task when the student parameters are always included in the prompt and the checkpoint-selection signal is aligned with the UMA teacher answers on held-out SP2013 rows.

Primary hypothesis:

- `135M` may have been under-optimized in the earlier `params_always` runs.
- A different learning rate and/or scheduler may improve fit when evaluation is done on matched-format prompts instead of plain no-param prompts.

## Study Design

Model/init:

- backbone: `HuggingFaceTB/SmolLM2-135M`
- init: pretrained only

Grid:

- learning rates: `5e-4`, `1e-4`, `5e-5`, `1e-5`, `5e-6`, `1e-6`
- schedulers: `cosine`, `linear`
- total runs: `12`
- epochs: `5`

Prompt format:

- training/validation prompts use `train_prompt_student_mode=always`
- no new translated CSV is built; the existing `synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz` is rewritten at load time

Validation construction:

- `val_frac=0`
- `test_frac=0`
- `move_sp2013_rows_to_val=true`
- expected held-out validation set:
  - `16,000` rows total
  - `16` unique SP2013 problems
  - `1000` unique UMA student parameter tuples
  - zero SP2013 rows remaining in train/test

Validation metrics:

- primary checkpoint-selection metric:
  - `id_val_acc`
  - greedy final-answer accuracy against the UMA response in `response_nl`
- secondary diagnostic:
  - `id_val_true_acc`
  - greedy final-answer accuracy against the mathematically true answer
- both metrics are computed from the same matched-format generations over the full `16,000`-row validation pool

Training-time eval policy:

- `evals_per_epoch=1`
- `eval_sp2013=false`
- `eval_id_final_answer=true`
- `id_eval_split=val`
- `id_eval_size=0` (full validation split)
- `id_eval_target=response`
- `id_eval_report_true_target=true`
- `best_by=id_val_acc`

## Runtime / Submission

Launcher:

- `results/transformer_replication/submit_135m_paramsalways_lr_scheduler_ablation_ailab.sh`

Submission command:

```bash
bash results/transformer_replication/submit_135m_paramsalways_lr_scheduler_ablation_ailab.sh
```

Cluster settings:

- partition: `ailab`
- GPUs per run: `1`
- CPUs per run: `8`
- wall clock per stage: `24:00:00`
- each run is submitted as a `24h + 24h` resume chain
- resume uses `last/trainer_state.pt` and reuses optimizer + scheduler state

Key job defaults baked into the launcher:

- `BATCH_SIZE=512`
- `EVAL_BATCH_SIZE=512`
- `ID_EVAL_BATCH_SIZE=64`
- `SAVE_LAST_EVERY_UPDATES=4000`
- `VERIFY_SAVED_CHECKPOINT_LOAD=0`

Artifacts:

- one run directory per `(lr, scheduler)` pair
- root `summary_metrics.json`
- root `best_checkpoint_summary.json`
- root `best_id_val_metrics.json`
- root `best_id_val.csv`
- per-eval `id_val_*.json/csv`
- manifest TSV plus submission JSON record

## Neuronic Handoff

The current launcher already covers the 12 jobs behind the pending `s135_cos_*` and `s135_lin_*` submissions.

Preview the submission on neuronic without launching anything:

```bash
PARTITION=<neuronic_partition> \
ACCOUNT=<account_if_needed> \
QOS=<qos_if_needed> \
MEM=64G \
SKIP_MODULE_LOAD=1 \
ACTIVATE_CMD='source /path/to/venv/bin/activate' \
HF_HUB_OFFLINE=0 \
TRANSFORMERS_OFFLINE=0 \
DRY_RUN=1 \
bash results/transformer_replication/submit_135m_paramsalways_lr_scheduler_ablation_ailab.sh
```

Submit for real:

```bash
PARTITION=<neuronic_partition> \
ACCOUNT=<account_if_needed> \
QOS=<qos_if_needed> \
MEM=64G \
SKIP_MODULE_LOAD=1 \
ACTIVATE_CMD='source /path/to/venv/bin/activate' \
HF_HUB_OFFLINE=0 \
TRANSFORMERS_OFFLINE=0 \
bash results/transformer_replication/submit_135m_paramsalways_lr_scheduler_ablation_ailab.sh
```

If neuronic needs different memory pressure tuning, the main knobs are:

- `BATCH_SIZE`
- `EVAL_BATCH_SIZE`
- `ID_EVAL_BATCH_SIZE`
- `GPUS_PER_NODE`

The launcher writes both:

- `jobs_135m_paramsalways_lr_scheduler_ablation_<RUN_TS>.tsv`
- `distill_135m_paramsalways_lr_scheduler_ablation_<RUN_TS>.json`

under `results/transformer_replication/`, which makes it easier to verify that the neuronic submission matches the Princeton one.

## Expected Interpretation

If the hypothesis is right, the best runs should show:

- higher `id_val_acc` than the earlier `135M params_always` runs
- lower training/validation loss without collapsing immediately
- possibly better `id_val_true_acc`, even though that is not the checkpoint-selection metric

If the grid still fails badly, the stronger suspicion shifts away from optimizer choice and toward representation/objective mismatch.
