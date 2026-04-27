#!/bin/bash
set -eo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/human_ft/logs"

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-torch-env}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL_PATH="${MODEL_PATH:-}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-best}"
TRAIN_CSV="${TRAIN_CSV:-${ROOT_DIR}/data/human_ft/data_train_nlp.csv}"
VAL_CSV="${VAL_CSV:-${ROOT_DIR}/data/human_ft/data_val_nlp.csv}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
OUT_DIR="${OUT_DIR:-}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1.5e-5}"
MAX_LENGTH="${MAX_LENGTH:-192}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
BEST_BY="${BEST_BY:-val_nll}"
SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-192}"
SP2013_NUM_ROLLOUTS="${SP2013_NUM_ROLLOUTS:-1000}"
SP2013_DO_SAMPLE="${SP2013_DO_SAMPLE:-1}"  # 1/0
SP2013_TEMPERATURE="${SP2013_TEMPERATURE:-0.7}"
SP2013_TOP_P="${SP2013_TOP_P:-0.95}"
SP2013_TOP_K="${SP2013_TOP_K:-50}"
SP2013_SAMPLE_SEED="${SP2013_SAMPLE_SEED:-123}"
SP2013_ROLLOUT_BATCH_SIZE="${SP2013_ROLLOUT_BATCH_SIZE:-8}"
SP2013_PROGRESS_EVERY="${SP2013_PROGRESS_EVERY:-1000}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-0}"  # 1/0
SP2013_GRID_G="${SP2013_GRID_G:-0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10}"
SP2013_GRID_D="${SP2013_GRID_D:-0.1,0.3,0.5,0.7,0.9}"
SP2013_GRID_RT="${SP2013_GRID_RT:-3,4,5,6}"
SP2013_GRID_ICE="${SP2013_GRID_ICE:-0,25,50,75,100}"
EVAL_SP2013="${EVAL_SP2013:-1}"          # 1/0
EVAL_LOADED_MODEL="${EVAL_LOADED_MODEL:-1}"  # 1/0
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"  # 1/0
SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-0}"  # 1/0
LOG_EVERY="${LOG_EVERY:-20}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required (path to prior run dir or checkpoint dir)." >&2
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  model_tag="$(basename "${MODEL_PATH}")"
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${ROOT_DIR}/human_ft/runs/${model_tag}_humanft_${ts}"
fi

echo "============================================================"
echo "Human-data finetune job"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CHECKPOINT_SUBDIR=${CHECKPOINT_SUBDIR}"
echo "TRAIN_CSV=${TRAIN_CSV}"
echo "VAL_CSV=${VAL_CSV}"
echo "SP2013_CSV=${SP2013_CSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "epochs=${EPOCHS} batch=${BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
echo "lr=${LR} max_length=${MAX_LENGTH} warmup_ratio=${WARMUP_RATIO} amp_dtype=${AMP_DTYPE}"
echo "eval_sp2013=${EVAL_SP2013} eval_loaded_model=${EVAL_LOADED_MODEL} sp2013_use_param_grid=${SP2013_USE_PARAM_GRID}"
echo "sp2013_rollouts=${SP2013_NUM_ROLLOUTS} sp2013_do_sample=${SP2013_DO_SAMPLE} temperature=${SP2013_TEMPERATURE} top_p=${SP2013_TOP_P} top_k=${SP2013_TOP_K} rollout_batch=${SP2013_ROLLOUT_BATCH_SIZE}"
echo "gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
echo "save_epoch_checkpoints=${SAVE_EPOCH_CHECKPOINTS}"
echo "============================================================"

SP2013_FLAG=(--no-eval_sp2013)
if [[ "${EVAL_SP2013}" == "1" ]]; then
  SP2013_FLAG=(--eval_sp2013)
fi

LOADED_FLAG=(--no-eval_loaded_model)
if [[ "${EVAL_LOADED_MODEL}" == "1" ]]; then
  LOADED_FLAG=(--eval_loaded_model)
fi

SP2013_SAMPLE_FLAG=(--no-sp2013_do_sample)
if [[ "${SP2013_DO_SAMPLE}" == "1" ]]; then
  SP2013_SAMPLE_FLAG=(--sp2013_do_sample)
fi

SP2013_GRID_FLAGS=(--no-sp2013_use_param_grid)
if [[ "${SP2013_USE_PARAM_GRID}" == "1" ]]; then
  SP2013_GRID_FLAGS=(
    --sp2013_use_param_grid
    --sp2013_grid_g "${SP2013_GRID_G}"
    --sp2013_grid_d "${SP2013_GRID_D}"
    --sp2013_grid_rt "${SP2013_GRID_RT}"
    --sp2013_grid_ice "${SP2013_GRID_ICE}"
  )
fi

GC_FLAG=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  GC_FLAG=(--gradient_checkpointing)
fi

EPOCH_CKPT_FLAG=(--no-save_epoch_checkpoints)
if [[ "${SAVE_EPOCH_CHECKPOINTS}" == "1" ]]; then
  EPOCH_CKPT_FLAG=(--save_epoch_checkpoints)
fi

python "${ROOT_DIR}/human_ft/finetune_humandata.py" \
  --model_path "${MODEL_PATH}" \
  --checkpoint_subdir "${CHECKPOINT_SUBDIR}" \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --sp2013_csv "${SP2013_CSV}" \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --max_length "${MAX_LENGTH}" \
  --num_workers "${NUM_WORKERS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --amp_dtype "${AMP_DTYPE}" \
  --best_by "${BEST_BY}" \
  --sp2013_max_new_tokens "${SP2013_MAX_NEW_TOKENS}" \
  --sp2013_num_rollouts "${SP2013_NUM_ROLLOUTS}" \
  --sp2013_temperature "${SP2013_TEMPERATURE}" \
  --sp2013_top_p "${SP2013_TOP_P}" \
  --sp2013_top_k "${SP2013_TOP_K}" \
  --sp2013_sample_seed "${SP2013_SAMPLE_SEED}" \
  --sp2013_rollout_batch_size "${SP2013_ROLLOUT_BATCH_SIZE}" \
  --sp2013_progress_every "${SP2013_PROGRESS_EVERY}" \
  --log_every "${LOG_EVERY}" \
  "${SP2013_FLAG[@]}" \
  "${LOADED_FLAG[@]}" \
  "${SP2013_SAMPLE_FLAG[@]}" \
  "${SP2013_GRID_FLAGS[@]}" \
  "${EPOCH_CKPT_FLAG[@]}" \
  "${GC_FLAG[@]}"
