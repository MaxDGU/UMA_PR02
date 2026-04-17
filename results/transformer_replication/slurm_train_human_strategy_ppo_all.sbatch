#!/bin/bash
#SBATCH --job-name=hppo
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/transformer_replication/logs/hppo_%j.out
#SBATCH --error=results/transformer_replication/logs/hppo_%j.err

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/results/transformer_replication/logs"

MODULE_LOAD_CMD="${MODULE_LOAD_CMD:-module load anaconda3/2024.02}"
ACTIVATE_CMD="${ACTIVATE_CMD:-}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TMP_ROOT="${TMP_ROOT:-/n/fs/cogai/cs1095/tmp}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"
export TMPDIR

ensure_conda_shell() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      source "${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidate
  for candidate in \
    /usr/local/anaconda3/2024.02 \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3"
  do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      source "${candidate}/etc/profile.d/conda.sh"
      return 0
    fi
  done
  return 1
}

if [[ "${SKIP_MODULE_LOAD}" != "1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    eval "${MODULE_LOAD_CMD}" || true
  fi
fi

if [[ -n "${ACTIVATE_CMD}" ]]; then
  eval "${ACTIVATE_CMD}"
else
  if ! ensure_conda_shell; then
    echo "conda not found; set ACTIVATE_CMD or ensure conda is on PATH" >&2
    exit 1
  fi
  conda activate "${CONDA_ENV:-base}"
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT_MODEL_DIR="${INIT_MODEL_DIR:-}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${ROOT_DIR}/results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best}"
HUMAN_TARGET_JSON="${HUMAN_TARGET_JSON:-${ROOT_DIR}/results/human/human_strategy_targets_classifier160k.json}"
SP2013_HUMAN_CSV="${SP2013_HUMAN_CSV:-${ROOT_DIR}/sp2013_human.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOSS_NAME="${LOSS_NAME:-tv}"

ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-32}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
POLICY_FORWARD_BATCH_SIZE="${POLICY_FORWARD_BATCH_SIZE:-8}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-12}"
MIN_REASONING_CHARS="${MIN_REASONING_CHARS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
TOTAL_UPDATES="${TOTAL_UPDATES:-100}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GAMMA="${GAMMA:-1.0}"
LAM="${LAM:-0.95}"
CLIPRANGE="${CLIPRANGE:-0.2}"
CLIPRANGE_VALUE="${CLIPRANGE_VALUE:-0.2}"
VF_COEF="${VF_COEF:-0.1}"
KL_COEF="${KL_COEF:-0.05}"
EMPTY_RESPONSE_PENALTY="${EMPTY_RESPONSE_PENALTY:-1.0}"
SHORT_REASONING_PENALTY="${SHORT_REASONING_PENALTY:-0.5}"
UNPARSEABLE_ANSWER_PENALTY="${UNPARSEABLE_ANSWER_PENALTY:-0.5}"
BEST_ID_INVALID_COEF="${BEST_ID_INVALID_COEF:-0.5}"
WHITEN_ADVANTAGES="${WHITEN_ADVANTAGES:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
ID_EVAL_EVERY="${ID_EVAL_EVERY:-10}"
ID_EVAL_ROLLOUTS_PER_PROBLEM="${ID_EVAL_ROLLOUTS_PER_PROBLEM:-128}"
OOD_EVAL_ROLLOUTS_PER_PROBLEM="${OOD_EVAL_ROLLOUTS_PER_PROBLEM:-128}"
SAVE_EVERY="${SAVE_EVERY:-0}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-64}"
CLASSIFIER_MAX_LENGTH="${CLASSIFIER_MAX_LENGTH:-256}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
CLASSIFIER_AMP_DTYPE="${CLASSIFIER_AMP_DTYPE:-auto}"
CLASSIFIER_DEVICE="${CLASSIFIER_DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
RUN_UNIT_CHECKS="${RUN_UNIT_CHECKS:-0}"

if [[ -z "${INIT_MODEL_DIR}" ]]; then
  echo "INIT_MODEL_DIR must be set" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR must be set" >&2
  exit 2
fi

GC_FLAG=(--no-gradient_checkpointing)
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  GC_FLAG=(--gradient_checkpointing)
fi

WHITEN_FLAG=(--no-whiten_advantages)
if [[ "${WHITEN_ADVANTAGES}" == "1" ]]; then
  WHITEN_FLAG=(--whiten_advantages)
fi

LOCAL_FILES_ONLY_FLAG=(--no-local_files_only)
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ONLY_FLAG=(--local_files_only)
fi

RUN_UNIT_FLAG=(--no-run_unit_checks)
if [[ "${RUN_UNIT_CHECKS}" == "1" ]]; then
  RUN_UNIT_FLAG=(--run_unit_checks)
fi

echo "Launching human strategy PPO"
echo "  init_model_dir=${INIT_MODEL_DIR}"
echo "  loss=${LOSS_NAME}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  rollouts_per_prompt=${ROLLOUTS_PER_PROMPT} prompt_batch_size=${PROMPT_BATCH_SIZE}"
echo "  rollout_batch_size=${ROLLOUT_BATCH_SIZE} policy_forward_batch_size=${POLICY_FORWARD_BATCH_SIZE}"
echo "  train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} mini_batch_size=${MINI_BATCH_SIZE}"
echo "  ppo_epochs=${PPO_EPOCHS} total_updates=${TOTAL_UPDATES}"
echo "  max_new_tokens=${MAX_NEW_TOKENS} min_new_tokens=${MIN_NEW_TOKENS} min_reasoning_chars=${MIN_REASONING_CHARS}"
echo "  lr=${LR} kl_coef=${KL_COEF} vf_coef=${VF_COEF}"
echo "  empty_response_penalty=${EMPTY_RESPONSE_PENALTY} short_reasoning_penalty=${SHORT_REASONING_PENALTY} unparseable_answer_penalty=${UNPARSEABLE_ANSWER_PENALTY}"
echo "  gradient_checkpointing=${GRADIENT_CHECKPOINTING} classifier_device=${CLASSIFIER_DEVICE}"
echo "  conda_env=${CONDA_ENV:-base}"

"${PYTHON_BIN}" "${ROOT_DIR}/results/transformer_replication/train_human_strategy_ppo.py" \
  --init_model_dir "${INIT_MODEL_DIR}" \
  --classifier_ckpt "${CLASSIFIER_CKPT}" \
  --human_target_json "${HUMAN_TARGET_JSON}" \
  --sp2013_human_csv "${SP2013_HUMAN_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --loss "${LOSS_NAME}" \
  --rollouts_per_prompt "${ROLLOUTS_PER_PROMPT}" \
  --prompt_batch_size "${PROMPT_BATCH_SIZE}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --policy_forward_batch_size "${POLICY_FORWARD_BATCH_SIZE}" \
  --train_micro_batch_size "${TRAIN_MICRO_BATCH_SIZE}" \
  --mini_batch_size "${MINI_BATCH_SIZE}" \
  --ppo_epochs "${PPO_EPOCHS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --min_new_tokens "${MIN_NEW_TOKENS}" \
  --min_reasoning_chars "${MIN_REASONING_CHARS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --total_updates "${TOTAL_UPDATES}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --gamma "${GAMMA}" \
  --lam "${LAM}" \
  --cliprange "${CLIPRANGE}" \
  --cliprange_value "${CLIPRANGE_VALUE}" \
  --vf_coef "${VF_COEF}" \
  --kl_coef "${KL_COEF}" \
  --empty_response_penalty "${EMPTY_RESPONSE_PENALTY}" \
  --short_reasoning_penalty "${SHORT_REASONING_PENALTY}" \
  --unparseable_answer_penalty "${UNPARSEABLE_ANSWER_PENALTY}" \
  --best_id_invalid_coef "${BEST_ID_INVALID_COEF}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --id_eval_every "${ID_EVAL_EVERY}" \
  --id_eval_rollouts_per_problem "${ID_EVAL_ROLLOUTS_PER_PROBLEM}" \
  --ood_eval_rollouts_per_problem "${OOD_EVAL_ROLLOUTS_PER_PROBLEM}" \
  --save_every "${SAVE_EVERY}" \
  --classifier_batch_size "${CLASSIFIER_BATCH_SIZE}" \
  --classifier_max_length "${CLASSIFIER_MAX_LENGTH}" \
  --amp_dtype "${AMP_DTYPE}" \
  --classifier_amp_dtype "${CLASSIFIER_AMP_DTYPE}" \
  --classifier_device "${CLASSIFIER_DEVICE}" \
  "${GC_FLAG[@]}" \
  "${WHITEN_FLAG[@]}" \
  "${LOCAL_FILES_ONLY_FLAG[@]}" \
  "${RUN_UNIT_FLAG[@]}"
