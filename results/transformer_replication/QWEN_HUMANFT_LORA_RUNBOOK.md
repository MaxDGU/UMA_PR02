# Qwen3 Human-FT LoRA Runbook

Question. Does fine-tuning on the 288-row human fraction corpus improve fit
to human behaviour, and is that improvement gated on base-model size? Prior
work on SmolLM2-135M did not see meaningful gains; the hypothesis is that
the model lacked the priors to absorb 288 examples without overfitting.
This runbook scans `Qwen/Qwen3-{0.6B,1.7B,4B,8B}-Base` with LoRA on the same
human corpus to test that hypothesis end-to-end.

Setup. The data are the canonical `data/human_ft/data_train_nlp.csv` (288
rows, 36 subjects, 8 problems) and `data/human_ft/data_val_nlp.csv` (96 rows,
12 held-out subjects, same 8 problems), already on this branch. The trainer
is `human_ft/finetune_humandata.py`, which builds prompts and supervises only
response tokens; LoRA wrapping has been added so it can target a fresh base
model without touching the base weights.

Method. The submitter `results/transformer_replication/submit_qwen_humanft_lora.sh`
submits one Slurm job per requested size; each job calls
`results/transformer_replication/slurm_train_qwen_humanft_lora.sbatch`, which
in turn invokes the human-FT trainer with `--use_lora` and the standard
target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
Per-size batch and grad-accum are tuned so all sizes fit on a single 80 GB
GPU with bf16 + gradient checkpointing. LR is held fixed at 1e-4 across
sizes for a clean scaling sweep.

Setup. On a fresh server, clone the repo, install the runtime deps, and
pre-fetch the Qwen3 Base snapshots (compute nodes assume HF offline).

```bash
git clone https://github.com/MaxDGU/UMA_PR02.git
cd UMA_PR02
git checkout main_max  # or the feature branch this lives on

python -m pip install torch transformers peft pandas pyarrow huggingface_hub

export HF_HOME="$PWD/.cache/huggingface"
for size in 0.6B 1.7B 4B 8B; do
  huggingface-cli download "Qwen/Qwen3-${size}-Base"
done
```

Protocol. Run the smoke test first. It uses the smallest model and caps to
20 optimizer steps so a green light means the wiring works end-to-end before
you spend GPU time on the bigger sizes.

```bash
SMOKE=1 PYTHON_BIN="$(command -v python)" \
HF_HOME="$PWD/.cache/huggingface" \
bash results/transformer_replication/submit_qwen_humanft_lora.sh
```

Protocol. Run the full sweep across all four sizes:

```bash
PYTHON_BIN="$(command -v python)" \
HF_HOME="$PWD/.cache/huggingface" \
bash results/transformer_replication/submit_qwen_humanft_lora.sh
```

Protocol. Override defaults for cluster, subset of sizes, or LR:

```bash
PARTITION=pli ACCOUNT=nam SIZES="1p7b 4b" LR=2e-4 \
PYTHON_BIN="$(command -v python)" \
HF_HOME="$PWD/.cache/huggingface" \
bash results/transformer_replication/submit_qwen_humanft_lora.sh
```

Configuration. Useful knobs (all environment variables, all overridable):

| Variable | Default | Notes |
| --- | --- | --- |
| `SIZES` | `0p6b 1p7b 4b 8b` | Subset of the size sweep. |
| `SMOKE` | `0` | `1` forces `0p6b`, `MAX_STEPS=20`, `EPOCHS=1`, 30-min wallclock. |
| `EPOCHS` | `3` | Full-run pass count. |
| `MAX_STEPS` | `0` | `>0` caps optimizer updates regardless of epochs. |
| `KL_BETA` | `0.0` | Set `>0` to anchor LoRA-finetuned model to its pre-FT distribution. |
| `INSTRUCTION_STYLE` | `plain` | `plain | masked_student | student_id | student_values | existing`. |
| `LR` / `LORA_R` / `LORA_ALPHA` | `1e-4` / `16` / `32` | Standard LoRA defaults. |
| `PARTITION` / `ACCOUNT` / `GRES` / `MEM` / `TIME_LIMIT` | `all` / unset / `gpu:1` / `64G` / `04:00:00` | Match your cluster. |
| `EVAL_SP2013` | `0` | Set `1` to run SP2013 final-answer eval after each epoch. Adds generation cost. |

Per-size batch/grad-accum are baked into the submitter (`config_for_size`):

| Size | Per-device batch | Grad accum | Effective batch |
| --- | --- | --- | --- |
| 0.6B | 4 | 4 | 16 |
| 1.7B | 4 | 4 | 16 |
| 4B   | 2 | 8 | 16 |
| 8B   | 1 | 16 | 16 |

Evidence. A successful submission writes a manifest TSV
`results/transformer_replication/jobs_qwen_humanft_lora_<ts>.tsv` listing
`size`, `model_name`, `job_id`, `out_dir`, and the per-size hyperparameters
that were sent. Each run directory ends up with `best/`, `last/`, `final/`
LoRA-adapter checkpoints and the standard history JSON/CSV used elsewhere
in the repo.

Validation. Before submitting, sanity-check the scripts:

```bash
bash -n results/transformer_replication/submit_qwen_humanft_lora.sh
bash -n results/transformer_replication/slurm_train_qwen_humanft_lora.sbatch
python -m py_compile human_ft/finetune_humandata.py
```

The 0.6B smoke takes ~1 minute on an H100. Watch for:
- `LoRA: r=16 alpha=32 dropout=0.05 target_modules=[...]`
- `trainable params: ~10M || all params: ~600M || trainable%: 1.6%`
- a smoothly decreasing `train_loss` over 20 updates and a `val_nll` line
- `saved best/last/final checkpoint -> ...`

Recovery. Compute nodes have no internet access; if the trainer logs DNS
retries on `huggingface.co`, your snapshots are not in `$HF_HOME` — re-run
the pre-fetch step on a login node and resubmit. If a run OOMs at 4B/8B,
halve `BATCH_SIZE` and double `GRAD_ACCUM_STEPS` so the effective batch is
preserved, or reduce `MAX_LENGTH` (default 256, prompts max around 60 tokens).

Hypothesis. We expect human-FT effect size to be near zero on 0.6B (matching
the SmolLM2-135M result) and to grow with model size as the prior on what
"a fraction problem" looks like becomes strong enough to absorb 288 examples
without collapsing. The cleanest comparison metric is SP2013 final-answer
accuracy MAE against the human distribution, evaluated at the best
checkpoint per size.
