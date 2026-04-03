# Reproducing The UMA Distillation And Human-FT Experiments On Another Cluster

This note describes how to reproduce the current UMA transformer experiments on a different cluster.

It focuses on the experiment code currently on:

- branch: `main_changho`
- commit: `1957ae417573899c6517ed93b87486cc213d5182`

It covers:

- the processed input data we actually used
- the core Python entrypoints
- the Slurm launchers we used on Princeton
- what to change when moving to a different cluster
- the exact commands for the main distillation and human-finetuning experiments

## 1. What To Copy

### Code

Use branch `main_changho` at:

- `1957ae417573899c6517ed93b87486cc213d5182`

### Processed input data

The fastest way to reproduce the experiments is to copy this zip:

- [uma_distill_humanft_repro_inputs_20260403.zip](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/uma_distill_humanft_repro_inputs_20260403.zip)

It contains the processed inputs used by the current distillation and human-FT workflow:

- [synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz)
- [sp2013.csv](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/UMA_replication/sp2013.csv)
- [data_train_nlp.csv](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/human/data_train_nlp.csv)
- [data_val_nlp.csv](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/human/data_val_nlp.csv)

These are enough to reproduce:

- synthetic distillation on the all1000 dataset
- SP2013 evaluation
- human finetuning
- post-train one-shot human-eval sampling

## 2. Software assumptions

At minimum, you need a Python environment with:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `jupyter`

Optional but sometimes needed:

- `peft`
  Use this if you want to load or save LoRA checkpoints.

The Princeton launch scripts assume a Conda env named `torch-env`, but that is not required. On another cluster, just point the scripts at your own env or call the Python entrypoints directly.

## 3. Core entrypoints

These are the main experiment scripts:

- Distillation train:
  [train_transformer_hf.py](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/train_transformer_hf.py)
- Human finetune:
  [finetune_humandata.py](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/finetune_humandata.py)
- Sampling eval:
  [eval_sampling.py](/scratch/gpfs/BRENDEN/changho/UMA_PR02/BBT/eval_sampling.py)

Higher-level wrappers:

- BBT pipeline:
  [pipeline.py](/scratch/gpfs/BRENDEN/changho/UMA_PR02/BBT/pipeline.py)
- BBT default config:
  [default_pipeline.json](/scratch/gpfs/BRENDEN/changho/UMA_PR02/BBT/configs/default_pipeline.json)

Princeton-specific launchers:

- Main all1000 distillation grid:
  [submit_synth_unique_seed1_all1000_2x3_ailab.sh](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh)
- Prompt-format A/B wrapper:
  [submit_synth_unique_seed1_all1000_params_ab_2x3_ailab.sh](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/submit_synth_unique_seed1_all1000_params_ab_2x3_ailab.sh)
- Per-model Slurm scripts:
  - [slurm_train_fraction2400_smol135_ailab.sbatch](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/slurm_train_fraction2400_smol135_ailab.sbatch)
  - [slurm_train_fraction2400_smol360_ailab.sbatch](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/slurm_train_fraction2400_smol360_ailab.sbatch)
  - [slurm_train_fraction2400_smol1p7b_ailab.sbatch](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/slurm_train_fraction2400_smol1p7b_ailab.sbatch)

## 4. Cluster-porting checklist

If you are moving to another cluster, the cluster-specific pieces are mostly in the shell launchers.

### Things that are Princeton-specific

- `#SBATCH` headers
- `partition` names like `ailab`
- module load lines like `module load anaconda3/2024.10`
- the Conda env name `torch-env`
- local Hugging Face cache placement

### Things that should usually stay the same

- the Python experiment scripts
- the processed CSV inputs
- the model names
- the prompt-format settings
- the batch sizes, unless your GPU memory is different
- the evaluation settings

### Things to edit first on another cluster

1. In the Slurm scripts, update:
   - partition / queue
   - account / qos if your cluster requires them
   - module or env activation
   - wall time
   - GPU count if needed

2. If you do not have the models in the local Hugging Face cache:
   - either pre-download them
   - or disable offline mode / `local_files_only`

3. If your cluster does not use Slurm:
   - use the Python entrypoints directly
   - or adapt the launcher env vars into your scheduler’s job format

## 5. Important defaults in the current code

### Distillation

Current training defaults in the Python trainer are in:

- [train_transformer_hf.py](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/train_transformer_hf.py)

Important current behaviors:

- `move_sp2013_rows_to_val=true` by default
- left padding is enabled for generation-time tokenizer usage
- `params_always` and `dropout` prompt rewriting happens at trainer load time
- SP2013 eval during distillation is the cheap training-time eval, not the later 1000-sample human-FT eval

### Human finetuning

Current defaults in:

- [finetune_humandata.py](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/finetune_humandata.py)

Important defaults:

- `epochs=3`
- `lr=1.5e-5`
- `best_by=val_nll`
- `eval_sp2013=false`
- `eval_loaded_model=false`

### Human post-train eval

Current BBT human-eval flow uses:

- `checkpoint_subdir=best`
- `num_samples_per_prompt=1000`
- `use_param_grid=false`

## 6. Recommended reproduction paths

There are two good ways to reproduce the experiments:

1. Use the direct experiment scripts and Slurm launchers.
2. Use the BBT pipeline for the human-FT workflow.

For another cluster, I recommend:

- direct launchers for the large distillation grids
- BBT or direct scripts for human finetuning

## 7. Reproduce the main all1000 distillation baseline

This is the baseline no-params all1000 distillation grid.

### Princeton-style launcher

```bash
bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

Useful overrides:

```bash
RUN_TS=20260403_baseline \
EPOCHS=2 \
BEST_BY=sp2013_acc \
SP2013_USE_PARAM_GRID=0 \
MODES=pretrained,scratch \
MODELS=135m,360m,1p7b \
bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

### Direct Python form

If you do not want the launcher, the trainer can be called directly. Example for `135M pretrained`:

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --data_csv results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz \
  --sp2013_csv results/UMA_replication/sp2013.csv \
  --output_dir results/transformer_replication/my_135m_distill \
  --epochs 2 \
  --batch_size 512 \
  --eval_batch_size 512 \
  --lr 2e-5 \
  --evals_per_epoch 5 \
  --best_by sp2013_acc \
  --eval_sp2013 \
  --no-sp2013_use_param_grid \
  --eval_id_final_answer \
  --move_sp2013_rows_to_val \
  --save_best_checkpoint \
  --save_final_checkpoint
```

For `360M`, use batch size `256`. For `1.7B`, use batch size `64` and usually `2` GPUs plus gradient checkpointing.

## 8. Reproduce the prompt-format A/B distillation study

This is the March 25 experiment that compared:

- baseline `no_params` (reused old runs, not rerun)
- `params_always`
- `params_dropout50_noprefix`

### Princeton-style wrapper

```bash
RUN_TS=20260403_params_ab \
EPOCHS=2 \
bash results/transformer_replication/submit_synth_unique_seed1_all1000_params_ab_2x3_ailab.sh
```

This wrapper reuses the same base CSV and changes prompt formatting inside the trainer.

### Conditions

`params_always`:

- `TRAIN_PROMPT_STUDENT_MODE=always`
- `TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0`

`params_dropout50_noprefix`:

- `TRAIN_PROMPT_STUDENT_MODE=dropout`
- `TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.5`
- `TRAIN_PROMPT_STUDENT_DROPOUT_SEED=0`

### Direct Python form

Example for `135M pretrained params_always`:

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --data_csv results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz \
  --sp2013_csv results/UMA_replication/sp2013.csv \
  --output_dir results/transformer_replication/my_135m_paramsalways \
  --epochs 2 \
  --batch_size 512 \
  --eval_batch_size 512 \
  --lr 2e-5 \
  --evals_per_epoch 5 \
  --best_by sp2013_acc \
  --eval_sp2013 \
  --no-sp2013_use_param_grid \
  --eval_id_final_answer \
  --train_prompt_student_mode always \
  --train_prompt_student_dropout_prob 0.0 \
  --train_prompt_student_dropout_seed 0 \
  --move_sp2013_rows_to_val \
  --save_best_checkpoint \
  --save_final_checkpoint
```

Example for dropout:

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --data_csv results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz \
  --sp2013_csv results/UMA_replication/sp2013.csv \
  --output_dir results/transformer_replication/my_135m_paramsdrop50 \
  --epochs 2 \
  --batch_size 512 \
  --eval_batch_size 512 \
  --lr 2e-5 \
  --evals_per_epoch 5 \
  --best_by sp2013_acc \
  --eval_sp2013 \
  --no-sp2013_use_param_grid \
  --eval_id_final_answer \
  --train_prompt_student_mode dropout \
  --train_prompt_student_dropout_prob 0.5 \
  --train_prompt_student_dropout_seed 0 \
  --move_sp2013_rows_to_val \
  --save_best_checkpoint \
  --save_final_checkpoint
```

## 9. Reproduce the focused March 30 135M follow-up

This was the smaller-LR, longer-run `135M params_always` follow-up:

- model: `135M pretrained`
- prompt mode: `params_always`
- `lr=1e-5`
- `epochs=6`
- `move_sp2013_rows_to_val=true`

Direct command:

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --data_csv results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz \
  --sp2013_csv results/UMA_replication/sp2013.csv \
  --output_dir results/transformer_replication/my_135m_paramsalways_lr1e5_e6 \
  --epochs 6 \
  --batch_size 512 \
  --eval_batch_size 512 \
  --lr 1e-5 \
  --evals_per_epoch 5 \
  --best_by sp2013_acc \
  --eval_sp2013 \
  --no-sp2013_use_param_grid \
  --eval_id_final_answer \
  --train_prompt_student_mode always \
  --train_prompt_student_dropout_prob 0.0 \
  --train_prompt_student_dropout_seed 0 \
  --move_sp2013_rows_to_val \
  --save_best_checkpoint \
  --save_final_checkpoint
```

## 10. Reproduce the 50-epoch human-finetuning experiment

This was the “best-by-validation-NLL, evaluate once at the end” human-FT experiment.

### Recommended direct two-step form

1. Human finetune:

```bash
python results/transformer_replication/finetune_humandata.py \
  --model_path /path/to/distill_run_dir \
  --checkpoint_subdir best \
  --train_csv results/human/data_train_nlp.csv \
  --val_csv results/human/data_val_nlp.csv \
  --output_dir results/transformer_replication/my_humanft50 \
  --epochs 50 \
  --lr 1.5e-5 \
  --best_by val_nll \
  --no-eval_sp2013 \
  --no-eval_loaded_model
```

2. One-shot post-train sampling eval:

```bash
python BBT/eval_sampling.py \
  --run_dir results/transformer_replication/my_humanft50 \
  --checkpoint_subdir best \
  --datasets_json '[{"name":"sp2013","csv":"results/UMA_replication/sp2013.csv","problem_col":"prob"}]' \
  --out_csv results/transformer_replication/my_humanft50/sampling_eval_samples.csv \
  --out_json results/transformer_replication/my_humanft50/sampling_eval_metrics.json \
  --num_samples_per_prompt 1000 \
  --max_new_tokens 192 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --sample_seed 2001 \
  --progress_every 1000 \
  --no-use_param_grid
```

### BBT pipeline form

```bash
python BBT/pipeline.py \
  --subpipelines human_finetune,human_eval \
  --set external_inputs.distill_run_dir=/path/to/distill_run_dir \
  --set subpipelines.human_finetune.epochs=50
```

This keeps the default human-FT LR at `1.5e-5`.

## 11. What changed recently that matters for reproducibility

These points matter if you compare against older runs:

1. `move_sp2013_rows_to_val` is now true by default.
2. Human finetuning now defaults to:
   - `eval_sp2013=false`
   - `eval_loaded_model=false`
   - save `best` by validation NLL
3. Human sampling eval now defaults to:
   - `checkpoint_subdir=best`
   - `num_samples_per_prompt=1000`
   - `use_param_grid=false`
4. Generation-time tokenizer setup now uses left padding for SmolLM2.

## 12. Where to look when something goes wrong

### Distillation

- training history:
  `<run_dir>/last/history.json`
- best checkpoint summary:
  `<run_dir>/best/history.json`
- SP2013 per-eval files:
  `<run_dir>/sp2013_metrics_*.json`
- trainer args:
  `<run_dir>/last/train_args.json`

### Human finetuning

- training history:
  `<run_dir>/history.json`
- summary:
  `<run_dir>/summary_metrics.json`
- checkpoints:
  `<run_dir>/best`, `<run_dir>/last`, `<run_dir>/final`

### Human one-shot eval

- samples:
  `<run_dir>/sampling_eval_samples.csv`
- metrics:
  `<run_dir>/sampling_eval_metrics.json`

## 13. My recommended minimal reproduction bundle

If I were reproducing this on a new cluster with the least friction, I would do this:

1. Clone `main_changho` at commit `1957ae4`.
2. Copy [uma_distill_humanft_repro_inputs_20260403.zip](/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/uma_distill_humanft_repro_inputs_20260403.zip).
3. Unzip it into the same relative paths.
4. Pre-download the SmolLM2 checkpoints or disable offline loading.
5. Run:
   - one small `135M` direct distillation command
   - one A/B formatted `135M` direct distillation command
   - one direct human-FT + `eval_sampling.py` sequence
6. After that works, switch to your cluster scheduler’s multi-job launch method.

That path separates “cluster integration problems” from “experiment logic problems” as cleanly as possible.
