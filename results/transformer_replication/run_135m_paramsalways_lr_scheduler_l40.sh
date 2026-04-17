#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT_PREFIX="${ROOT_PREFIX:-/n/fs/cogai/cs1095}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT_PREFIX}/conda-envs/uma-transformer-l40}"
export CONDA_ENV="${CONDA_ENV:-${ENV_PREFIX}}"

mkdir -p "${ROOT_PREFIX}/tmp"
export TMPDIR="${TMPDIR:-${ROOT_PREFIX}/tmp}"
export TMP_ROOT="${TMP_ROOT:-${ROOT_PREFIX}/tmp}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-1}"
export ACTIVATE_CMD="${ACTIVATE_CMD:-source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh && conda activate ${ENV_PREFIX}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PARTITION="${PARTITION:-all}"
export MEM="${MEM:-64G}"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export CHAIN_COUNT="${CHAIN_COUNT:-4}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
export ID_EVAL_BATCH_SIZE="${ID_EVAL_BATCH_SIZE:-24}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

cd "${ROOT_DIR}"
exec bash "${ROOT_DIR}/results/transformer_replication/submit_135m_paramsalways_lr_scheduler_ablation_ailab.sh"
