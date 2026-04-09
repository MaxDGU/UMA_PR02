#!/bin/bash
#SBATCH --job-name=f2400_s1p7b
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results/transformer_replication/logs/f2400_s1p7b_%j.out
#SBATCH --error=results/transformer_replication/logs/f2400_s1p7b_%j.err

set -eo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/results/transformer_replication/logs"

MODULE_LOAD_CMD="${MODULE_LOAD_CMD:-module load anaconda3/2024.10}"
ACTIVATE_CMD="${ACTIVATE_CMD:-}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-0}"  # 0/1
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ "${SKIP_MODULE_LOAD}" != "1" ]]; then
  eval "${MODULE_LOAD_CMD}"
fi

if [[ -n "${ACTIVATE_CMD}" ]]; then
  eval "${ACTIVATE_CMD}"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; set ACTIVATE_CMD or ensure conda is on PATH" >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-torch-env}"
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-1.7B}"
DATA_CSV="${DATA_CSV:-${ROOT_DIR}/results/transformer_replication/uma_traces_fraction2400_all_models_nlp.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/transformer_replication/smol2_1p7b_fraction2400_nlp_ailab}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
VAL_FRAC="${VAL_FRAC:-0.05}"
TEST_FRAC="${TEST_FRAC:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
GC="${GC:-1}"  # 0/1
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
BEST_BY="${BEST_BY:-val_loss}"
EVAL_SP2013="${EVAL_SP2013:-1}"  # 0/1
EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-1}"  # 0/1
SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-256}"
ID_EVAL_MAX_NEW_TOKENS="${ID_EVAL_MAX_NEW_TOKENS:-256}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-1}"  # 0/1
SP2013_GRID_G="${SP2013_GRID_G:-0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10}"
SP2013_GRID_D="${SP2013_GRID_D:-0.1,0.3,0.5,0.7,0.9}"
SP2013_GRID_RT="${SP2013_GRID_RT:-3,4,5,6}"
SP2013_GRID_ICE="${SP2013_GRID_ICE:-0,25,50,75,100}"
RESUME_FROM="${RESUME_FROM:-auto}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-200}"
INIT_FROM_SCRATCH="${INIT_FROM_SCRATCH:-0}"  # 0/1
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-1}"  # 0/1
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-0}"  # 0/1
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-0}"  # 0/1
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-1}"  # 0/1
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
REQUIRE_RESUME="${REQUIRE_RESUME:-0}"  # 0/1
MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-1}"  # 0/1
TRAIN_PROMPT_STUDENT_MODE="${TRAIN_PROMPT_STUDENT_MODE:-none}"
TRAIN_PROMPT_STUDENT_DROPOUT_PROB="${TRAIN_PROMPT_STUDENT_DROPOUT_PROB:-0.5}"
TRAIN_PROMPT_STUDENT_DROPOUT_SEED="${TRAIN_PROMPT_STUDENT_DROPOUT_SEED:-0}"

echo "Launching SmolLM2-1.7B on ${GPUS_PER_NODE}x H200"
echo "batch_size=${BATCH_SIZE} per GPU, grad_accum=1, evals_per_epoch=${EVALS_PER_EPOCH}, time_limit=${SLURM_TIMELIMIT:-24:00:00}"
echo "resume_from=${RESUME_FROM}, save_last_every_updates=${SAVE_LAST_EVERY_UPDATES}"
echo "init_from_scratch=${INIT_FROM_SCRATCH}"
echo "require_resume=${REQUIRE_RESUME}"
echo "activation: skip_module_load=${SKIP_MODULE_LOAD} conda_env=${CONDA_ENV:-torch-env} torchrun_bin=${TORCHRUN_BIN}"
if [[ -n "${ACTIVATE_CMD}" ]]; then
  echo "activation override: ACTIVATE_CMD is set"
fi
echo "move_sp2013_rows_to_val=${MOVE_SP2013_ROWS_TO_VAL}"
echo "save_eval_checkpoints=${SAVE_EVAL_CHECKPOINTS}, save_best_checkpoint=${SAVE_BEST_CHECKPOINT}, save_final_checkpoint=${SAVE_FINAL_CHECKPOINT}, verify_saved_checkpoint_load=${VERIFY_SAVED_CHECKPOINT_LOAD}"
echo "eval max_new_tokens: sp2013=${SP2013_MAX_NEW_TOKENS}, id=${ID_EVAL_MAX_NEW_TOKENS}"
echo "sp2013_use_param_grid=${SP2013_USE_PARAM_GRID} (g=${SP2013_GRID_G}, d=${SP2013_GRID_D}, rt=${SP2013_GRID_RT}, ice=${SP2013_GRID_ICE})"
echo "train_prompt_student_mode=${TRAIN_PROMPT_STUDENT_MODE} dropout_prob=${TRAIN_PROMPT_STUDENT_DROPOUT_PROB} dropout_seed=${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}"
echo "output_dir=${OUT_DIR}"

if [[ "${REQUIRE_RESUME}" == "1" && ! -f "${OUT_DIR}/last/trainer_state.pt" ]]; then
  echo "REQUIRE_RESUME=1 but no ${OUT_DIR}/last/trainer_state.pt found" >&2
  exit 1
fi

INIT_FLAG=(--no-init_from_scratch)
if [[ "${INIT_FROM_SCRATCH}" == "1" ]]; then
  INIT_FLAG=(--init_from_scratch)
fi

GC_FLAG=()
if [[ "${GC}" == "1" ]]; then
  GC_FLAG=(--gradient_checkpointing)
fi

SP2013_FLAG=(--no-eval_sp2013)
if [[ "${EVAL_SP2013}" == "1" ]]; then
  SP2013_FLAG=(--eval_sp2013)
fi

ID_EVAL_FLAG=(--no-eval_id_final_answer)
if [[ "${EVAL_ID_FINAL_ANSWER}" == "1" ]]; then
  ID_EVAL_FLAG=(--eval_id_final_answer)
fi

MOVE_SP2013_ROWS_FLAG=(--no-move_sp2013_rows_to_val)
if [[ "${MOVE_SP2013_ROWS_TO_VAL}" == "1" ]]; then
  MOVE_SP2013_ROWS_FLAG=(--move_sp2013_rows_to_val)
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

SAVE_EVAL_FLAG=(--no-save_eval_checkpoints)
if [[ "${SAVE_EVAL_CHECKPOINTS}" == "1" ]]; then
  SAVE_EVAL_FLAG=(--save_eval_checkpoints)
fi

SAVE_BEST_FLAG=(--no-save_best_checkpoint)
if [[ "${SAVE_BEST_CHECKPOINT}" == "1" ]]; then
  SAVE_BEST_FLAG=(--save_best_checkpoint)
fi

SAVE_FINAL_FLAG=(--no-save_final_checkpoint)
if [[ "${SAVE_FINAL_CHECKPOINT}" == "1" ]]; then
  SAVE_FINAL_FLAG=(--save_final_checkpoint)
fi

VERIFY_LOAD_FLAG=(--no-verify_saved_checkpoint_load)
if [[ "${VERIFY_SAVED_CHECKPOINT_LOAD}" == "1" ]]; then
  VERIFY_LOAD_FLAG=(--verify_saved_checkpoint_load)
fi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${ROOT_DIR}/results/transformer_replication/train_transformer_hf.py" \
    --model_name "${MODEL_NAME}" \
    --data_csv "${DATA_CSV}" \
    --sp2013_csv "${SP2013_CSV}" \
    --output_dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --grad_accum_steps 1 \
    --lr "${LR}" \
    --max_length "${MAX_LENGTH}" \
    --val_frac "${VAL_FRAC}" \
    --test_frac "${TEST_FRAC}" \
    --num_workers "${NUM_WORKERS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --amp_dtype "${AMP_DTYPE}" \
    --evals_per_epoch "${EVALS_PER_EPOCH}" \
    --best_by "${BEST_BY}" \
    --train_prompt_student_mode "${TRAIN_PROMPT_STUDENT_MODE}" \
    --train_prompt_student_dropout_prob "${TRAIN_PROMPT_STUDENT_DROPOUT_PROB}" \
    --train_prompt_student_dropout_seed "${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}" \
    --sp2013_max_new_tokens "${SP2013_MAX_NEW_TOKENS}" \
    --id_eval_max_new_tokens "${ID_EVAL_MAX_NEW_TOKENS}" \
    --resume_from "${RESUME_FROM}" \
    --save_last_every_updates "${SAVE_LAST_EVERY_UPDATES}" \
    "${INIT_FLAG[@]}" \
    "${SP2013_FLAG[@]}" \
    "${SP2013_GRID_FLAGS[@]}" \
    "${ID_EVAL_FLAG[@]}" \
    "${MOVE_SP2013_ROWS_FLAG[@]}" \
    "${SAVE_EVAL_FLAG[@]}" \
    "${SAVE_BEST_FLAG[@]}" \
    "${SAVE_FINAL_FLAG[@]}" \
    "${VERIFY_LOAD_FLAG[@]}" \
    "${GC_FLAG[@]}" \
    --save_best_only
