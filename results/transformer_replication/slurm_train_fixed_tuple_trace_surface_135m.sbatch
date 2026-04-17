#!/bin/bash
# Train one 135M fixed-tuple trace-surface run, reading context settings at runtime.

#SBATCH --job-name=ft472_135m
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/transformer_replication/logs/ft472_135m_%j.out
#SBATCH --error=results/transformer_replication/logs/ft472_135m_%j.err

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

: "${DATA_CSV:?Set DATA_CSV to the filtered translated training CSV.}"
: "${OUT_DIR:?Set OUT_DIR to the training output directory.}"
: "${CONTEXT_JSON:?Set CONTEXT_JSON to the dataset-build context JSON.}"

export CONDA_ENV="/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40"

read -r MAX_LENGTH MAX_NEW_TOKENS <<EOF
$(python - <<'PY'
import json
import os

with open(os.environ["CONTEXT_JSON"], "r", encoding="utf-8") as f:
    payload = json.load(f)
shared = payload["context"]["shared_settings"]
print(f"{int(shared['max_length'])} {int(shared['max_new_tokens'])}")
PY
)
EOF

export MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}"
export DATA_CSV
export OUT_DIR
export BATCH_SIZE="${BATCH_SIZE:-96}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-5e-4}"
export MIN_LR="${MIN_LR:-1e-6}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
export MAX_LENGTH
export VAL_FRAC="${VAL_FRAC:-0.05}"
export TEST_FRAC="${TEST_FRAC:-0.05}"
export EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
export LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH:-0}"
export BEST_BY="${BEST_BY:-val_loss}"
export EVAL_SP2013="${EVAL_SP2013:-0}"
export EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-1}"
export ID_EVAL_BATCH_SIZE="${ID_EVAL_BATCH_SIZE:-24}"
export SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
export ID_EVAL_MAX_NEW_TOKENS="${ID_EVAL_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
export PREVIEW_MAX_NEW_TOKENS="${PREVIEW_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
export SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-1}"
export SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
export SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-0}"
export VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-0}"
export MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-0}"
export STRICT_SP2013_ONLY_VALIDATION_LAYOUT="${STRICT_SP2013_ONLY_VALIDATION_LAYOUT:-0}"
export TRAIN_PROMPT_STUDENT_MODE="${TRAIN_PROMPT_STUDENT_MODE:-always}"
export PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-0}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

bash "${ROOT_DIR}/results/transformer_replication/slurm_train_fraction2400_smol135_ailab.sbatch"
