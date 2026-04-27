# Human Fraction Fine-Tuning Pipeline

This folder is the canonical home for human fraction supervised fine-tuning.
The old entrypoint at `results/transformer_replication/finetune_humandata.py`
is kept as a thin compatibility wrapper.

## Inputs

Default training data:

```text
data/human_ft/data_train_nlp.csv
data/human_ft/data_val_nlp.csv
```

Those CSVs contain `subjid`, `prob`, `resp`, `strategy`, `instruction_nl`, and
`response_nl`. The shared data bundle and manifest live in `data/human_ft/`.

## Local Command

```bash
python human_ft/finetune_humandata.py \
  --model_path results/transformer_replication/<distill-run> \
  --checkpoint_subdir best \
  --train_csv data/human_ft/data_train_nlp.csv \
  --val_csv data/human_ft/data_val_nlp.csv \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum_steps 16 \
  --lr 1.5e-5
```

If `--output_dir` is omitted, runs are written under `human_ft/runs/`, which is
ignored by git. You can also provide any explicit output directory.

## Training Modes

| Mode | Current status | How to run |
| --- | --- | --- |
| SFT | Migrated and canonical in `human_ft/finetune_humandata.py`. | Run with `--kl_beta 0`, or omit `--kl_beta`. |
| SFT + response-token KL | Migrated and canonical in `human_ft/finetune_humandata.py`. | Run with `--kl_beta > 0`. The reference defaults to the source checkpoint unless `--kl_reference_model_path` is set. |
| DPO | Implemented, but still in legacy transformer-replication scripts rather than canonical `human_ft/`. | Build pairs with `results/transformer_replication/build_human_uma_dpo_pairs.py`, then train with `results/transformer_replication/train_human_uma_dpo.py` or `results/transformer_replication/slurm_train_human_uma_dpo.sbatch`. |

Prompting coverage:

| Prompting variant | SFT flag | DPO pair-building flag | Notes |
| --- | --- | --- | --- |
| Plain problem prompt | `--instruction_style plain` | `--prompt_style plain` | Rebuilds prompts from `prob` as `Solve this fraction problem: ...=?`. |
| Masked student profile | `--instruction_style masked_student` | `--prompt_style masked_student` | Prefixes prompts with unknown student parameters: `<student> g ?? d ?? rt ?? ice ?? </student>`. |
| Student ID | `--instruction_style student_id` | `--prompt_style student_id` | Uses `subjid` in the prompt. In-training SP2013 eval is not supported for SFT with this style; use standalone evals. |
| Student values | `--instruction_style student_values` | `--prompt_style student_values` | Uses concrete `g`, `d`, `rt_mu`, and `ice`; can expand rows from a panel CSV. |
| Existing prompt column | `--instruction_style existing` | Not a DPO pair-builder mode | Leaves the prompt column as provided after cleanup. |

So: the experimental ingredients exist, but only SFT and SFT+KL have been
migrated into the new `human_ft` folder so far. DPO should be the next migration
if `human_ft` is meant to be the complete public human-training pipeline.

## Key Options

Model initialization:

| Option | What it does | When to use it |
| --- | --- | --- |
| `--init_strategy checkpoint` | Loads an existing distilled or fine-tuned checkpoint from `--model_path`. If `--model_path` is a run directory, `--checkpoint_subdir` selects the checkpoint inside it, usually `best`. | Normal human fine-tuning on top of a distillation run. This is the default and requires `--model_path`. |
| `--init_strategy pretrained` | Loads `--base_model_name` with pretrained Hugging Face weights, then fine-tunes directly on the human corpus. | Baselines that skip UMA distillation. |
| `--init_strategy random` | Builds a fresh model config from `--base_model_name` without pretrained weights. | Smoke tests or ablations where pretrained knowledge should be removed. |
| `--base_model_name` | Hugging Face model/config name used for tokenizer loading and for `pretrained` or `random` starts. Default: `HuggingFaceTB/SmolLM2-135M`. | Change this when comparing model sizes or architectures. |
| `--local_files_only` / `--no-local_files_only` | Controls whether model/tokenizer loads must come from the local Hugging Face cache. Default: local only. | Keep enabled on offline clusters; disable only when downloads are allowed. |

Data and prompt construction:

| Option | What it does | Notes |
| --- | --- | --- |
| `--train_csv`, `--val_csv` | Input train/validation CSVs. Defaults are `data/human_ft/data_train_nlp.csv` and `data/human_ft/data_val_nlp.csv`. | NLP CSVs should contain `instruction_nl` and `response_nl`. Raw human CSVs can also be normalized if they contain problem, response, and strategy text columns. |
| `--prompt_col`, `--response_col` | Column names used as the prompt and target response. Defaults: `instruction_nl`, `response_nl`. | The response should include the final-answer marker, for example `### answer: 3/10`. |
| `--instruction_style existing` | Leaves the prompt column as provided after basic cleanup. | Use this when the CSV already contains exactly the prompts you want to train on. |
| `--instruction_style plain` | Rebuilds prompts from `prob` as `Solve this fraction problem: ...=?`. | Default. This gives a consistent prompt surface and ignores any custom wording in `instruction_nl`. |
| `--instruction_style masked_student` | Rebuilds prompts with an unknown student-parameter prefix: `<student> g ?? d ?? rt ?? ice ?? </student>`. | Useful when the model should see the student-conditioning format without true parameters. |
| `--instruction_style student_id` | Rebuilds prompts with `<student> subjid ... </student>` using the `subjid` column. | Requires non-empty subject IDs. In-training SP2013 eval is disabled for this style; use standalone distribution evals instead. |
| `--instruction_style student_values` | Rebuilds prompts with concrete `g`, `d`, `rt_mu`, and `ice` values. | Requires those columns in the CSV, or use `--panel_csv` to cross each human row with a parameter panel. |
| `--panel_csv` | CSV containing `g`, `d`, `rt_mu`, and `ice` for `student_values` prompt expansion. | Optional `subjid` in the panel is retained as `panel_subjid`. |

Training and outputs:

| Option | What it does | Notes |
| --- | --- | --- |
| `--epochs`, `--max_steps` | `--epochs` controls full passes through the data; `--max_steps` optionally stops after a fixed number of optimizer updates. | Use `--max_steps` for smoke tests. `0` means no step cap. |
| `--batch_size`, `--eval_batch_size` | Per-device train/eval batch sizes. | Combine small `--batch_size` with larger `--grad_accum_steps` when GPU memory is tight. |
| `--grad_accum_steps` | Accumulates gradients over multiple batches before each optimizer update. | Effective batch size is roughly `batch_size * grad_accum_steps` per process. |
| `--lr`, `--weight_decay`, `--warmup_ratio`, `--max_grad_norm` | Optimizer, scheduler, and clipping settings. | Defaults are conservative for short human fine-tuning runs. |
| `--max_length` | Token length cap for prompt plus response. | Increase only if prompts/responses are being truncated. |
| `--train_on_prompt` | Includes prompt tokens in the supervised loss. | By default, prompt tokens are masked and only response tokens contribute to SFT loss. |
| `--output_dir` | Directory for checkpoints, metrics, and history files. | If omitted, the script writes to ignored `human_ft/runs/<tag>_finetune_human_<timestamp>/`. |
| `--save_epoch_checkpoints`, `--save_epoch_list`, `--save_update_list` | Save extra standalone checkpoints beyond `best`, `last`, and `final`. | Useful for checkpoint sweeps; avoid enabling casually because model files are large. |

KL regularization:

| Option | What it does | Notes |
| --- | --- | --- |
| `--kl_beta` | Adds response-token KL(policy || reference) to the SFT loss. `0` disables KL. | Use small positive values when you want human fine-tuning to stay close to the distilled model. |
| `--kl_reference_model_path` | Optional reference checkpoint/run directory for KL. | If omitted and `--kl_beta > 0`, the resolved source checkpoint is used as the frozen reference. |
| `--kl_reference_checkpoint_subdir` | Subdirectory to resolve inside `--kl_reference_model_path`. | Defaults to `--checkpoint_subdir` when omitted. |

Evaluation:

| Option | What it does | Notes |
| --- | --- | --- |
| `--eval_loaded_model` | Evaluates the loaded model before any human fine-tuning. | Useful for before/after comparisons. |
| `--eval_sp2013` | Runs SP2013 final-answer evaluation after each epoch. | Adds generation cost. Required when `--best_by sp2013_acc`. |
| `--best_by val_nll` | Selects the best checkpoint by validation NLL. | Default and usually the most stable setting for this small human validation split. |
| `--best_by sp2013_acc` | Selects the best checkpoint by SP2013 accuracy. | Requires `--eval_sp2013`. |
| `--sp2013_num_rollouts`, `--sp2013_do_sample`, `--sp2013_temperature`, `--sp2013_top_p`, `--sp2013_top_k` | Sampling settings for SP2013 generation. | `num_rollouts > 1` requires sampling. |
| `--sp2013_use_param_grid`, `--sp2013_grid_g`, `--sp2013_grid_d`, `--sp2013_grid_rt`, `--sp2013_grid_ice` | Optional UMA parameter-grid evaluation controls. | Mainly for student-parameter-conditioned evaluation modes. |

## Slurm

Single job:

```bash
sbatch --export=ALL,MODEL_PATH=results/transformer_replication/<distill-run> \
  human_ft/slurm/finetune_humandata.sbatch
```

Array job:

```bash
sbatch --array=0-<N> --export=ALL,HUMANFT_MANIFEST=path/to/manifest.tsv \
  human_ft/slurm/finetune_humandata_array.sbatch
```

The array manifest is a TSV whose column names become environment variables,
for example `DISPLAY_NAME`, `MODEL_PATH`, `CHECKPOINT_SUBDIR`, `TRAIN_CSV`,
`VAL_CSV`, `OUTPUT_DIR`, `EPOCHS`, `LR`, and `BATCH_SIZE`.

## Compatibility

Existing commands that call:

```bash
python results/transformer_replication/finetune_humandata.py
```

still work and dispatch to `human_ft.finetune_humandata.main`. BBT defaults now
prefer `human_ft/finetune_humandata.py` and `data/human_ft/*_nlp.csv`.

Local checkpoint/model outputs such as `human_ft/*_best_*/`, `human_ft/runs/`,
`human_ft/logs/`, and `*.safetensors` remain ignored.
