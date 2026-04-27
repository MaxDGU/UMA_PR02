#!/bin/bash
# Fine-tune one panel25 run on human data from a manifest row.

#SBATCH --job-name=humft50
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=human_ft/logs/humanft50_%A_%a.out
#SBATCH --error=human_ft/logs/humanft50_%A_%a.err

set -euo pipefail
export PS1="${PS1:-}"

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/human_ft/logs"

: "${HUMANFT_MANIFEST:?Set HUMANFT_MANIFEST to the human-finetune manifest TSV.}"

eval "$(python - "${HUMANFT_MANIFEST}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import csv
import shlex
import sys

manifest_path = sys.argv[1]
task_idx = int(sys.argv[2])
with open(manifest_path, "r", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if task_idx < 0 or task_idx >= len(rows):
    raise SystemExit(f"task_idx {task_idx} out of range for {manifest_path}")
row = rows[task_idx]
for key, value in row.items():
    env_key = key.upper()
    print(f"export {env_key}={shlex.quote(str(value))}")
PY
)"

TRAIN_CSV="${TRAIN_CSV:-${ROOT_DIR}/data/human_ft/data_train_nlp.csv}"
VAL_CSV="${VAL_CSV:-${ROOT_DIR}/data/human_ft/data_val_nlp.csv}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"

export CONDA_ENV="${CONDA_ENV:-/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMP_ROOT="${TMP_ROOT:-/n/fs/cogai/cs1095/tmp}"
export TMPDIR="${TMPDIR:-${TMP_ROOT}/${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"

echo "Launching human-data finetune for ${DISPLAY_NAME}"
if [[ -n "${RUN_KEY:-}" ]]; then
  echo "run_key=${RUN_KEY}"
fi
echo "model_path=${MODEL_PATH}"
echo "base_model_name=${BASE_MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}"
echo "init_strategy=${INIT_STRATEGY:-checkpoint}"
echo "checkpoint_subdir=${CHECKPOINT_SUBDIR}"
echo "output_dir=${OUTPUT_DIR}"
if [[ -n "${PANEL_CSV:-}" ]]; then
  echo "panel_csv=${PANEL_CSV}"
fi
echo "epochs=${EPOCHS} lr=${LR} batch_size=${BATCH_SIZE} eval_batch_size=${EVAL_BATCH_SIZE}"
echo "gradient_checkpointing=${GRADIENT_CHECKPOINTING:-0}"
echo "kl_beta=${KL_BETA:-0} kl_reference_model_path=${KL_REFERENCE_MODEL_PATH:-}"

"${PYTHON_BIN}" human_ft/finetune_humandata.py \
  --model_path "${MODEL_PATH}" \
  --base_model_name "${BASE_MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}" \
  --init_strategy "${INIT_STRATEGY:-checkpoint}" \
  --checkpoint_subdir "${CHECKPOINT_SUBDIR}" \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --instruction_style "${INSTRUCTION_STYLE:-plain}" \
  $( [[ -n "${PANEL_CSV:-}" ]] && printf -- '--panel_csv %s ' "${PANEL_CSV}" ) \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --max_length "${MAX_LENGTH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --log_every "${LOG_EVERY}" \
  --max_steps "${MAX_STEPS:-0}" \
  --amp_dtype "${AMP_DTYPE}" \
  --best_by "${BEST_BY}" \
  --kl_beta "${KL_BETA:-0}" \
  $( [[ -n "${KL_REFERENCE_MODEL_PATH:-}" ]] && printf -- '--kl_reference_model_path %s ' "${KL_REFERENCE_MODEL_PATH}" ) \
  $( [[ -n "${KL_REFERENCE_CHECKPOINT_SUBDIR:-}" ]] && printf -- '--kl_reference_checkpoint_subdir %s ' "${KL_REFERENCE_CHECKPOINT_SUBDIR}" ) \
  --sp2013_csv "${SP2013_CSV}" \
  --sp2013_max_new_tokens "${SP2013_MAX_NEW_TOKENS}" \
  --sp2013_num_rollouts "${SP2013_NUM_ROLLOUTS}" \
  --sp2013_temperature "${SP2013_TEMPERATURE}" \
  --sp2013_top_p "${SP2013_TOP_P}" \
  --sp2013_top_k "${SP2013_TOP_K}" \
  --sp2013_sample_seed "${SP2013_SAMPLE_SEED}" \
  --sp2013_rollout_batch_size "${SP2013_ROLLOUT_BATCH_SIZE}" \
  --sp2013_progress_every "${SP2013_PROGRESS_EVERY}" \
  --no-eval_sp2013 \
  --no-eval_loaded_model \
  $( [[ "${GRADIENT_CHECKPOINTING:-0}" =~ ^(1|true|True|yes|YES)$ ]] && printf -- '--gradient_checkpointing ' ) \
  $( [[ "${SAVE_EPOCH_CHECKPOINTS:-0}" =~ ^(1|true|True|yes|YES)$ ]] && printf -- '--save_epoch_checkpoints ' ) \
  $( [[ -n "${SAVE_EPOCH_LIST:-}" ]] && printf -- '--save_epoch_list %s ' "${SAVE_EPOCH_LIST}" ) \
  $( [[ -n "${SAVE_UPDATE_LIST:-}" ]] && printf -- '--save_update_list %s ' "${SAVE_UPDATE_LIST}" ) \
  --local_files_only
