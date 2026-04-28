# Qwen3 Distillation Runbook

Question. This runbook describes how to submit one-epoch Qwen3 UMA distillation jobs from a fresh GitHub checkout on another server.

Setup. The runnable data live in the repository under `data/distillation/` as Parquet datasets. The Qwen launcher uses `data/distillation/uma_fraction_distillation_25` for fraction panel25 default distillation and `data/distillation/uma_decimal_bssmix_no3dpmul_repaired_v1` for decimal repaired-mixture distillation.

Setup. Let \(D_f\) be the fraction Parquet dataset and \(D_d\) be the decimal Parquet dataset. Each row is a prompt-response pair \((x_i, y_i)\), with UMA student parameters \(g,d,\mu_{rt},ice\) available as columns. Training minimizes causal language-model loss on response tokens only.

Method. The launcher submits four Qwen3 Base models for each enabled dataset: `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, and `Qwen/Qwen3-8B-Base`. Full fine-tuning is used for `0.6B`; LoRA is used for `1.7B`, `4B`, and `8B`.

Method. The LoRA target modules are `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. The jobs load pretrained weights in bf16 with `--model_load_dtype bf16`, which keeps the larger frozen backbones practical on a single modern GPU.

Setup. On a new server, clone the repo and install a Python environment with at least `torch`, `transformers`, `peft`, `pandas`, `pyarrow`, and `huggingface_hub`. Then pre-download Qwen3 Base snapshots into the cache that the Slurm jobs will use.

```bash
git clone https://github.com/MaxDGU/UMA_PR02.git
cd UMA_PR02
git checkout main_changho

python -m pip install torch transformers peft pandas pyarrow huggingface_hub

export HF_HOME="$PWD/.cache/huggingface"
python - <<'PY'
from huggingface_hub import snapshot_download

for repo_id in [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
]:
    snapshot_download(repo_id=repo_id)
PY
```

Protocol. Submit all eight jobs with the default checked-in datasets:

```bash
PYTHON_BIN="$(command -v python)" \
HF_HOME="$PWD/.cache/huggingface" \
bash results/transformer_replication/submit_qwen3_distill_1epoch.sh
```

Protocol. Submit only one side when debugging or when a cluster queue is tight:

```bash
RUN_DECIMAL=0 bash results/transformer_replication/submit_qwen3_distill_1epoch.sh
RUN_FRACTION=0 bash results/transformer_replication/submit_qwen3_distill_1epoch.sh
```

Configuration. The submitter defaults to `PARTITION=all`, `GRES=gpu:1`, `MEM=128G`, and `TIME_LIMIT=7-00:00:00`. Override these for the local scheduler, for example:

```bash
PARTITION=gpu GRES=gpu:a100:1 MEM=160G TIME_LIMIT=2-00:00:00 \
PYTHON_BIN=/path/to/env/bin/python \
HF_HOME=/path/to/hf-cache \
bash results/transformer_replication/submit_qwen3_distill_1epoch.sh
```

Configuration. Use `FRACTION_DATA=/path/to/parquet-or-csv` or `DECIMAL_DATA=/path/to/parquet-or-csv` to replace the checked-in datasets. The trainer accepts either a Parquet dataset directory, a single Parquet file, or a CSV/CSV.GZ file with the standard UMA NLP columns.

Evidence. A successful submission prints one line per job and writes `results/transformer_replication/jobs_qwen3_distill_1epoch_<timestamp>.tsv`. The manifest records dataset, model ID, LoRA flag, job ID, output directory, data path, batch size, gradient accumulation, and ID-eval batch size.

Validation. Before submitting, run:

```bash
bash -n results/transformer_replication/submit_qwen3_distill_1epoch.sh
bash -n results/transformer_replication/slurm_train_qwen3_distill_1epoch.sbatch
python -m py_compile results/transformer_replication/train_transformer_hf.py
python - <<'PY'
from results.transformer_replication.train_transformer_hf import load_and_clean_trace_frame

for path in [
    "data/distillation/uma_fraction_distillation_25",
    "data/distillation/uma_decimal_bssmix_no3dpmul_repaired_v1",
]:
    df = load_and_clean_trace_frame(
        path,
        usecols=["instruction_nl", "response_nl", "prob", "g", "d", "rt_mu", "ice"],
        prompt_col="instruction_nl",
        response_col="response_nl",
        problem_col="prob",
    )
    print(path, len(df))
PY
```

Recovery. If a job fails before training starts, inspect the matching log in `results/transformer_replication/slurm_logs/`. Common setup failures are missing Qwen snapshots, missing `peft`, missing `pyarrow`, or scheduler-specific `PARTITION`/`GRES` values.

Recovery. If a job is interrupted after training starts, the Slurm script writes resumable checkpoints to `<OUT_DIR>/last`. To resume manually, resubmit the same configuration with `RESUME_FROM=auto` and the same `OUT_DIR`.
