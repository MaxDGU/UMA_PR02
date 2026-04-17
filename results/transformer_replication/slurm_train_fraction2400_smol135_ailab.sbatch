#!/bin/bash
#SBATCH --job-name=f2400_s135
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/transformer_replication/logs/f2400_s135_%j.out
#SBATCH --error=results/transformer_replication/logs/f2400_s135_%j.err

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

MODULE_LOAD_CMD="${MODULE_LOAD_CMD:-module load anaconda3/2024.02}"
ACTIVATE_CMD="${ACTIVATE_CMD:-}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-0}"  # 0/1
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TMP_ROOT="${TMP_ROOT:-/n/fs/cogai/cs1095/tmp}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/${SLURM_JOB_ID:-local}}"

mkdir -p "${TMPDIR}"
export TMPDIR

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}"
DATA_CSV="${DATA_CSV:-${ROOT_DIR}/results/transformer_replication/uma_traces_fraction2400_all_models_nlp.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
SP2013_TARGET_CSV="${SP2013_TARGET_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013_eval/seed_1/sp2013_seed1_all_models.csv}"
SP2013_VAL_SOURCE_CSV="${SP2013_VAL_SOURCE_CSV:-}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/transformer_replication/smol2_135m_fraction2400_nlp_ailab}"

BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-0.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MAX_LENGTH="${MAX_LENGTH:-256}"
VAL_FRAC="${VAL_FRAC:-0.05}"
TEST_FRAC="${TEST_FRAC:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
GC="${GC:-0}"  # 0/1
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH:-0}"
BEST_BY="${BEST_BY:-val_loss}"
EVAL_SP2013="${EVAL_SP2013:-1}"  # 0/1
SP2013_TARGET_MODE="${SP2013_TARGET_MODE:-true}"
EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-1}"  # 0/1
SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-256}"
SP2013_NUM_ROLLOUTS="${SP2013_NUM_ROLLOUTS:-1}"
SP2013_DO_SAMPLE="${SP2013_DO_SAMPLE:-0}"  # 0/1
SP2013_TEMPERATURE="${SP2013_TEMPERATURE:-0.7}"
SP2013_TOP_P="${SP2013_TOP_P:-0.95}"
SP2013_TOP_K="${SP2013_TOP_K:-50}"
SP2013_SAMPLE_SEED="${SP2013_SAMPLE_SEED:-123}"
SP2013_ROLLOUT_BATCH_SIZE="${SP2013_ROLLOUT_BATCH_SIZE:-1}"
SP2013_PROGRESS_EVERY="${SP2013_PROGRESS_EVERY:-0}"
ID_EVAL_MAX_NEW_TOKENS="${ID_EVAL_MAX_NEW_TOKENS:-256}"
ID_EVAL_SPLIT="${ID_EVAL_SPLIT:-val}"
ID_EVAL_SIZE="${ID_EVAL_SIZE:-64}"
ID_EVAL_TARGET="${ID_EVAL_TARGET:-response}"
ID_EVAL_REPORT_TRUE_TARGET="${ID_EVAL_REPORT_TRUE_TARGET:-0}"  # 0/1
ID_EVAL_BATCH_SIZE="${ID_EVAL_BATCH_SIZE:-64}"
ID_EVAL_EVERY="${ID_EVAL_EVERY:-1}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-1}"  # 0/1
SP2013_GRID_G="${SP2013_GRID_G:-0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10}"
SP2013_GRID_D="${SP2013_GRID_D:-0.1,0.3,0.5,0.7,0.9}"
SP2013_GRID_RT="${SP2013_GRID_RT:-3,4,5,6}"
SP2013_GRID_ICE="${SP2013_GRID_ICE:-0,25,50,75,100}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME_FROM="${RESUME_FROM:-auto}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-500}"
INIT_FROM_SCRATCH="${INIT_FROM_SCRATCH:-0}"  # 0/1
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-1}"  # 0/1
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-0}"  # 0/1
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-0}"  # 0/1
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-1}"  # 0/1
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
REQUIRE_RESUME="${REQUIRE_RESUME:-0}"  # 0/1
MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-1}"  # 0/1
STRICT_SP2013_ONLY_VALIDATION_LAYOUT="${STRICT_SP2013_ONLY_VALIDATION_LAYOUT:-1}"  # 0/1
TRAIN_PROMPT_STUDENT_MODE="${TRAIN_PROMPT_STUDENT_MODE:-none}"
TRAIN_PROMPT_STUDENT_DROPOUT_PROB="${TRAIN_PROMPT_STUDENT_DROPOUT_PROB:-0.5}"
TRAIN_PROMPT_STUDENT_DROPOUT_SEED="${TRAIN_PROMPT_STUDENT_DROPOUT_SEED:-0}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"  # 0/1

count_csv_items() {
  local raw="${1// /}"
  if [[ -z "${raw}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "${raw}"
}

GRID_G_N="$(count_csv_items "${SP2013_GRID_G}")"
GRID_D_N="$(count_csv_items "${SP2013_GRID_D}")"
GRID_RT_N="$(count_csv_items "${SP2013_GRID_RT}")"
GRID_ICE_N="$(count_csv_items "${SP2013_GRID_ICE}")"
GRID_TOTAL="$((GRID_G_N * GRID_D_N * GRID_RT_N * GRID_ICE_N))"

echo "Launching SmolLM2-135M on ${GPUS_PER_NODE}x GPU"
echo "batch_size=${BATCH_SIZE} per GPU, grad_accum=${GRAD_ACCUM_STEPS}, evals_per_epoch=${EVALS_PER_EPOCH}, loss_evals_per_epoch=${LOSS_EVALS_PER_EPOCH}, time_limit=${SLURM_TIMELIMIT:-24:00:00}"
echo "resume_from=${RESUME_FROM}, save_last_every_updates=${SAVE_LAST_EVERY_UPDATES}"
echo "init_from_scratch=${INIT_FROM_SCRATCH}"
echo "require_resume=${REQUIRE_RESUME}"
echo "move_sp2013_rows_to_val=${MOVE_SP2013_ROWS_TO_VAL}"
echo "strict_sp2013_only_validation_layout=${STRICT_SP2013_ONLY_VALIDATION_LAYOUT}"
echo "save_eval_checkpoints=${SAVE_EVAL_CHECKPOINTS}, save_best_checkpoint=${SAVE_BEST_CHECKPOINT}, save_final_checkpoint=${SAVE_FINAL_CHECKPOINT}, verify_saved_checkpoint_load=${VERIFY_SAVED_CHECKPOINT_LOAD}"
echo "eval max_new_tokens: sp2013=${SP2013_MAX_NEW_TOKENS}, id=${ID_EVAL_MAX_NEW_TOKENS}"
echo "sp2013_target_mode=${SP2013_TARGET_MODE} target_csv=${SP2013_TARGET_CSV}"
if [[ -n "${SP2013_VAL_SOURCE_CSV}" ]]; then
  echo "sp2013_val_source_csv=${SP2013_VAL_SOURCE_CSV}"
fi
echo "sp2013_sampling: rollouts=${SP2013_NUM_ROLLOUTS} do_sample=${SP2013_DO_SAMPLE} temperature=${SP2013_TEMPERATURE} top_p=${SP2013_TOP_P} top_k=${SP2013_TOP_K} rollout_batch=${SP2013_ROLLOUT_BATCH_SIZE}"
echo "id_eval split=${ID_EVAL_SPLIT} size=${ID_EVAL_SIZE} every=${ID_EVAL_EVERY} target=${ID_EVAL_TARGET} report_true=${ID_EVAL_REPORT_TRUE_TARGET} batch_size=${ID_EVAL_BATCH_SIZE}"
echo "lr=${LR} min_lr=${MIN_LR} lr_scheduler_type=${LR_SCHEDULER_TYPE}"
echo "sp2013_use_param_grid=${SP2013_USE_PARAM_GRID} (g=${SP2013_GRID_G}, d=${SP2013_GRID_D}, rt=${SP2013_GRID_RT}, ice=${SP2013_GRID_ICE})"
echo "sp2013_grid_sizes: g=${GRID_G_N}, d=${GRID_D_N}, rt=${GRID_RT_N}, ice=${GRID_ICE_N} -> total=${GRID_TOTAL}"
echo "train_prompt_student_mode=${TRAIN_PROMPT_STUDENT_MODE} dropout_prob=${TRAIN_PROMPT_STUDENT_DROPOUT_PROB} dropout_seed=${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}"
if [[ -n "${MAX_SAMPLES}" ]]; then
  echo "max_samples=${MAX_SAMPLES}"
fi
echo "preview_samples=${PREVIEW_SAMPLES}"
echo "local_files_only=${LOCAL_FILES_ONLY}"
echo "output_dir=${OUT_DIR}"
echo "activation: skip_module_load=${SKIP_MODULE_LOAD} conda_env=${CONDA_ENV:-torch-env} torchrun_bin=${TORCHRUN_BIN}"
if [[ -n "${ACTIVATE_CMD}" ]]; then
  echo "activation override: ACTIVATE_CMD is set"
fi

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

SP2013_DO_SAMPLE_FLAG=(--no-sp2013_do_sample)
if [[ "${SP2013_DO_SAMPLE}" == "1" ]]; then
  SP2013_DO_SAMPLE_FLAG=(--sp2013_do_sample)
fi

ID_EVAL_FLAG=(--no-eval_id_final_answer)
if [[ "${EVAL_ID_FINAL_ANSWER}" == "1" ]]; then
  ID_EVAL_FLAG=(--eval_id_final_answer)
fi

ID_EVAL_REPORT_TRUE_FLAG=(--no-id_eval_report_true_target)
if [[ "${ID_EVAL_REPORT_TRUE_TARGET}" == "1" ]]; then
  ID_EVAL_REPORT_TRUE_FLAG=(--id_eval_report_true_target)
fi

MOVE_SP2013_ROWS_FLAG=(--no-move_sp2013_rows_to_val)
if [[ "${MOVE_SP2013_ROWS_TO_VAL}" == "1" ]]; then
  MOVE_SP2013_ROWS_FLAG=(--move_sp2013_rows_to_val)
fi

STRICT_SP2013_LAYOUT_FLAG=(--strict_sp2013_only_validation_layout)
if [[ "${STRICT_SP2013_ONLY_VALIDATION_LAYOUT}" != "1" ]]; then
  STRICT_SP2013_LAYOUT_FLAG=(--no-strict_sp2013_only_validation_layout)
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

LOCAL_FILES_ONLY_FLAG=(--no-local_files_only)
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ONLY_FLAG=(--local_files_only)
fi

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARGS=(--max_samples "${MAX_SAMPLES}")
fi

SP2013_VAL_SOURCE_ARGS=()
if [[ -n "${SP2013_VAL_SOURCE_CSV}" ]]; then
  SP2013_VAL_SOURCE_ARGS=(--sp2013_val_source_csv "${SP2013_VAL_SOURCE_CSV}")
fi

"${TORCHRUN_BIN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${ROOT_DIR}/results/transformer_replication/train_transformer_hf.py" \
    --model_name "${MODEL_NAME}" \
    --data_csv "${DATA_CSV}" \
    --sp2013_csv "${SP2013_CSV}" \
    --sp2013_target_csv "${SP2013_TARGET_CSV}" \
    "${SP2013_VAL_SOURCE_ARGS[@]}" \
    --output_dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --max_length "${MAX_LENGTH}" \
    --val_frac "${VAL_FRAC}" \
    --test_frac "${TEST_FRAC}" \
    --num_workers "${NUM_WORKERS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --amp_dtype "${AMP_DTYPE}" \
    --evals_per_epoch "${EVALS_PER_EPOCH}" \
    --loss_evals_per_epoch "${LOSS_EVALS_PER_EPOCH}" \
    --best_by "${BEST_BY}" \
    --sp2013_target_mode "${SP2013_TARGET_MODE}" \
    --sp2013_num_rollouts "${SP2013_NUM_ROLLOUTS}" \
    --sp2013_temperature "${SP2013_TEMPERATURE}" \
    --sp2013_top_p "${SP2013_TOP_P}" \
    --sp2013_top_k "${SP2013_TOP_K}" \
    --sp2013_sample_seed "${SP2013_SAMPLE_SEED}" \
    --sp2013_rollout_batch_size "${SP2013_ROLLOUT_BATCH_SIZE}" \
    --sp2013_progress_every "${SP2013_PROGRESS_EVERY}" \
    --id_eval_every "${ID_EVAL_EVERY}" \
    --id_eval_split "${ID_EVAL_SPLIT}" \
    --id_eval_size "${ID_EVAL_SIZE}" \
    --id_eval_target "${ID_EVAL_TARGET}" \
    --id_eval_batch_size "${ID_EVAL_BATCH_SIZE}" \
    --train_prompt_student_mode "${TRAIN_PROMPT_STUDENT_MODE}" \
    --train_prompt_student_dropout_prob "${TRAIN_PROMPT_STUDENT_DROPOUT_PROB}" \
    --train_prompt_student_dropout_seed "${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}" \
    --sp2013_max_new_tokens "${SP2013_MAX_NEW_TOKENS}" \
    --id_eval_max_new_tokens "${ID_EVAL_MAX_NEW_TOKENS}" \
    --resume_from "${RESUME_FROM}" \
    --save_last_every_updates "${SAVE_LAST_EVERY_UPDATES}" \
    --preview_samples "${PREVIEW_SAMPLES}" \
    "${MAX_SAMPLES_ARGS[@]}" \
    "${INIT_FLAG[@]}" \
    "${SP2013_FLAG[@]}" \
    "${SP2013_DO_SAMPLE_FLAG[@]}" \
    "${SP2013_GRID_FLAGS[@]}" \
    "${ID_EVAL_FLAG[@]}" \
    "${ID_EVAL_REPORT_TRUE_FLAG[@]}" \
    "${MOVE_SP2013_ROWS_FLAG[@]}" \
    "${STRICT_SP2013_LAYOUT_FLAG[@]}" \
    "${SAVE_EVAL_FLAG[@]}" \
    "${SAVE_BEST_FLAG[@]}" \
    "${SAVE_FINAL_FLAG[@]}" \
    "${VERIFY_LOAD_FLAG[@]}" \
    "${LOCAL_FILES_ONLY_FLAG[@]}" \
    "${GC_FLAG[@]}" \
    --save_best_only
