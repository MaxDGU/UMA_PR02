# Qwen3 Distillation Runbook

Question. This runbook describes how to submit one-epoch UMA distillation jobs for Qwen3 base models on the fraction panel25 default dataset and the current decimal repaired mixture dataset.

Setup. Let \(D_f\) denote the fraction panel25 default distillation CSV and \(D_d\) denote the repaired decimal mixture CSV. The training objective is next-token causal language modeling on UMA translated traces, where each example is a prompt-response pair \((x_i, y_i)\) and the loss is evaluated on response tokens only.

Setup. The fraction dataset is `results/transformer_replication/synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_default_nosp2013_mincols.csv.gz`. The decimal dataset is `results/transformer_replication/decimal_bss_mixture_add2400_mul4800_no3dpmul_repaired_v1_k100_xlsx_distill/uma_traces_decimal_bss_mix_add2400_mul4800_no3dpmul_repaired_v1_k100_xlsx_nlp_think_aloud_child_train_mincols.csv.gz`.

Method. The launcher submits eight Slurm jobs: four Qwen3 base sizes for fraction and four Qwen3 base sizes for decimal. The model IDs are `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, and `Qwen/Qwen3-8B-Base`.

Method. Full fine-tuning is used only for `0.6B`. LoRA is used for `1.7B`, `4B`, and `8B` with target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. All jobs load pretrained weights in bf16 via `--model_load_dtype bf16`; this keeps the larger frozen backbones within one L40 GPU for LoRA.

Protocol. The submitter checks that PEFT imports in the configured Python environment and that all Qwen3 Base snapshots exist in the local Hugging Face cache. It then calls `sbatch` once per dataset-model pair and writes a local manifest under `results/transformer_replication/jobs_qwen3_distill_1epoch_<timestamp>.tsv`.

Command.

```bash
bash results/transformer_replication/submit_qwen3_distill_1epoch.sh
```

Configuration. The default Python is `/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40/bin/python`, and the default Hugging Face cache is `/n/fs/cogai/.cache/hf`. Override them with `PYTHON_BIN=/path/to/python` or `HF_HOME=/path/to/cache` when using another cluster image.

Configuration. The submitter defaults to partition `all`, `gpu:l40:1`, `128G` memory, and a seven-day time limit. Override with `PARTITION`, `GRES`, `MEM`, `TIME_LIMIT`, or `EXCLUDE_NODES` if the cluster policy changes.

Evidence. A successful submit prints one line per job and then prints the manifest. The manifest records the dataset, model ID, LoRA flag, job ID, output directory, data CSV, batch size, gradient accumulation, and ID-eval batch size.

Validation. Before submitting, run `bash -n` on both shell scripts and compile the trainer:

```bash
bash -n results/transformer_replication/submit_qwen3_distill_1epoch.sh
bash -n results/transformer_replication/slurm_train_qwen3_distill_1epoch.sbatch
python -m py_compile results/transformer_replication/train_transformer_hf.py
```

Recovery. If a job fails before training starts, inspect the matching Slurm log in `results/transformer_replication/slurm_logs/`. The most common setup failures are missing Qwen snapshots, missing `peft`, or a stale `PYTHON_BIN`.

Recovery. If a job is interrupted after training starts, the Slurm script writes resumable checkpoints to `<OUT_DIR>/last`. To resume manually, resubmit the same job configuration with `RESUME_FROM=auto` and the same `OUT_DIR`.
