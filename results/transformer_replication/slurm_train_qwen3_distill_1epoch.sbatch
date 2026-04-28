#!/usr/bin/env bash
#SBATCH --job-name=qwen3_distill
#SBATCH --partition=all
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/transformer_replication/slurm_logs/%x_%j.out
#SBATCH --error=results/transformer_replication/slurm_logs/%x_%j.err

set -euo pipefail

ROOT="${REPO_ROOT:?REPO_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:-/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40/bin/python}"
DATA_CSV="${DATA_CSV:?DATA_CSV is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

cd "$ROOT"

export HF_HOME="${HF_HOME:-/n/fs/cogai/.cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$ROOT/results/transformer_replication/slurm_logs" "$OUT_DIR"

TASK_LABEL="${TASK_LABEL:-qwen3_distill}"
DATASET_KIND="${DATASET_KIND:-fraction}"
PROMPT_PROBLEM_KIND="${PROMPT_PROBLEM_KIND:-fraction}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-288}"
VAL_FRAC="${VAL_FRAC:-0.05}"
TEST_FRAC="${TEST_FRAC:-0.05}"
SPLIT_BY_PROBLEM="${SPLIT_BY_PROBLEM:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
MODEL_LOAD_DTYPE="${MODEL_LOAD_DTYPE:-bf16}"
GC="${GC:-1}"
BEST_BY="${BEST_BY:-val_loss}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-1}"
LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH:-1}"
EVAL_SP2013="${EVAL_SP2013:-0}"
SP2013_CSV="${SP2013_CSV:-${ROOT}/results/UMA_replication/sp2013.csv}"
SP2013_TARGET_CSV="${SP2013_TARGET_CSV:-${ROOT}/results/UMA_replication/sp2013_eval/seed_1/sp2013_seed1_all_models.csv}"
SP2013_TARGET_MODE="${SP2013_TARGET_MODE:-true}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-0}"
SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-256}"
EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-1}"
ID_EVAL_EVERY="${ID_EVAL_EVERY:-1}"
ID_EVAL_SPLIT="${ID_EVAL_SPLIT:-val}"
ID_EVAL_SIZE="${ID_EVAL_SIZE:-64}"
ID_EVAL_TARGET="${ID_EVAL_TARGET:-response}"
ID_EVAL_REPORT_TRUE_TARGET="${ID_EVAL_REPORT_TRUE_TARGET:-1}"
ID_EVAL_BATCH_SIZE="${ID_EVAL_BATCH_SIZE:-4}"
ID_EVAL_MAX_NEW_TOKENS="${ID_EVAL_MAX_NEW_TOKENS:-240}"
MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-0}"
STRICT_SP2013_ONLY_VALIDATION_LAYOUT="${STRICT_SP2013_ONLY_VALIDATION_LAYOUT:-0}"
TRAIN_PROMPT_STUDENT_MODE="${TRAIN_PROMPT_STUDENT_MODE:-always}"
TRAIN_PROMPT_STUDENT_DROPOUT_PROB="${TRAIN_PROMPT_STUDENT_DROPOUT_PROB:-0.0}"
TRAIN_PROMPT_STUDENT_DROPOUT_SEED="${TRAIN_PROMPT_STUDENT_DROPOUT_SEED:-0}"
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-0}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-0}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-4000}"
RESUME_FROM="${RESUME_FROM:-none}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-0}"

bool_flag() {
  local enabled="$1"
  local positive="$2"
  local negative="$3"
  if [[ "$enabled" == "1" ]]; then
    printf '%s' "$positive"
  else
    printf '%s' "$negative"
  fi
}

GC_FLAG=()
if [[ "$GC" == "1" ]]; then
  GC_FLAG=(--gradient_checkpointing)
fi

SPLIT_FLAG=(--no-split_by_problem)
if [[ "$SPLIT_BY_PROBLEM" == "1" ]]; then
  SPLIT_FLAG=(--split_by_problem)
fi

LORA_ARGS=(--no-use_lora)
if [[ "$USE_LORA" == "1" ]]; then
  LORA_ARGS=(
    --use_lora
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "$LORA_TARGET_MODULES"
  )
fi

EVAL_SP2013_FLAG="$(bool_flag "$EVAL_SP2013" --eval_sp2013 --no-eval_sp2013)"
SP2013_GRID_FLAG="$(bool_flag "$SP2013_USE_PARAM_GRID" --sp2013_use_param_grid --no-sp2013_use_param_grid)"
EVAL_ID_FLAG="$(bool_flag "$EVAL_ID_FINAL_ANSWER" --eval_id_final_answer --no-eval_id_final_answer)"
ID_TRUE_FLAG="$(bool_flag "$ID_EVAL_REPORT_TRUE_TARGET" --id_eval_report_true_target --no-id_eval_report_true_target)"
MOVE_SP2013_FLAG="$(bool_flag "$MOVE_SP2013_ROWS_TO_VAL" --move_sp2013_rows_to_val --no-move_sp2013_rows_to_val)"
STRICT_LAYOUT_FLAG="$(bool_flag "$STRICT_SP2013_ONLY_VALIDATION_LAYOUT" --strict_sp2013_only_validation_layout --no-strict_sp2013_only_validation_layout)"
SAVE_EVAL_FLAG="$(bool_flag "$SAVE_EVAL_CHECKPOINTS" --save_eval_checkpoints --no-save_eval_checkpoints)"
SAVE_BEST_FLAG="$(bool_flag "$SAVE_BEST_CHECKPOINT" --save_best_checkpoint --no-save_best_checkpoint)"
SAVE_FINAL_FLAG="$(bool_flag "$SAVE_FINAL_CHECKPOINT" --save_final_checkpoint --no-save_final_checkpoint)"
VERIFY_LOAD_FLAG="$(bool_flag "$VERIFY_SAVED_CHECKPOINT_LOAD" --verify_saved_checkpoint_load --no-verify_saved_checkpoint_load)"
LOCAL_FILES_FLAG="$(bool_flag "$LOCAL_FILES_ONLY" --local_files_only --no-local_files_only)"

{
  echo "[$(date -Iseconds)] starting ${TASK_LABEL}"
  echo "job_id=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-unknown} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "python_bin=${PYTHON_BIN}"
  echo "dataset_kind=${DATASET_KIND}"
  echo "data_csv=${DATA_CSV}"
  echo "model_name=${MODEL_NAME}"
  echo "model_load_dtype=${MODEL_LOAD_DTYPE}"
  echo "out_dir=${OUT_DIR}"
  echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} lr=${LR} min_lr=${MIN_LR} scheduler=${LR_SCHEDULER_TYPE}"
  echo "max_length=${MAX_LENGTH} val_frac=${VAL_FRAC} test_frac=${TEST_FRAC} split_by_problem=${SPLIT_BY_PROBLEM}"
  echo "use_lora=${USE_LORA} lora_r=${LORA_R} lora_alpha=${LORA_ALPHA} lora_dropout=${LORA_DROPOUT}"
  echo "train_prompt_student_mode=${TRAIN_PROMPT_STUDENT_MODE} prompt_problem_kind=${PROMPT_PROBLEM_KIND}"
  echo "best_by=${BEST_BY} id_eval_size=${ID_EVAL_SIZE} id_eval_batch_size=${ID_EVAL_BATCH_SIZE}"
  nvidia-smi || true

  "$PYTHON_BIN" results/transformer_replication/train_transformer_hf.py \
    --model_name "$MODEL_NAME" \
    --model_load_dtype "$MODEL_LOAD_DTYPE" \
    --data_csv "$DATA_CSV" \
    --sp2013_csv "$SP2013_CSV" \
    --sp2013_target_csv "$SP2013_TARGET_CSV" \
    --output_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --lr "$LR" \
    --min_lr "$MIN_LR" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --val_frac "$VAL_FRAC" \
    --test_frac "$TEST_FRAC" \
    --prompt_problem_kind "$PROMPT_PROBLEM_KIND" \
    --num_workers "$NUM_WORKERS" \
    --amp_dtype "$AMP_DTYPE" \
    --evals_per_epoch "$EVALS_PER_EPOCH" \
    --loss_evals_per_epoch "$LOSS_EVALS_PER_EPOCH" \
    --best_by "$BEST_BY" \
    --sp2013_target_mode "$SP2013_TARGET_MODE" \
    --sp2013_max_new_tokens "$SP2013_MAX_NEW_TOKENS" \
    --id_eval_every "$ID_EVAL_EVERY" \
    --id_eval_split "$ID_EVAL_SPLIT" \
    --id_eval_size "$ID_EVAL_SIZE" \
    --id_eval_target "$ID_EVAL_TARGET" \
    --id_eval_batch_size "$ID_EVAL_BATCH_SIZE" \
    --id_eval_max_new_tokens "$ID_EVAL_MAX_NEW_TOKENS" \
    --train_prompt_student_mode "$TRAIN_PROMPT_STUDENT_MODE" \
    --train_prompt_student_dropout_prob "$TRAIN_PROMPT_STUDENT_DROPOUT_PROB" \
    --train_prompt_student_dropout_seed "$TRAIN_PROMPT_STUDENT_DROPOUT_SEED" \
    --resume_from "$RESUME_FROM" \
    --save_last_every_updates "$SAVE_LAST_EVERY_UPDATES" \
    --preview_samples "$PREVIEW_SAMPLES" \
    "${SPLIT_FLAG[@]}" \
    "${LORA_ARGS[@]}" \
    "${GC_FLAG[@]}" \
    "$EVAL_SP2013_FLAG" \
    "$SP2013_GRID_FLAG" \
    "$EVAL_ID_FLAG" \
    "$ID_TRUE_FLAG" \
    "$MOVE_SP2013_FLAG" \
    "$STRICT_LAYOUT_FLAG" \
    "$SAVE_EVAL_FLAG" \
    "$SAVE_BEST_FLAG" \
    "$SAVE_FINAL_FLAG" \
    "$VERIFY_LOAD_FLAG" \
    "$LOCAL_FILES_FLAG" \
    --save_best_only

  echo "[$(date -Iseconds)] finished ${TASK_LABEL}"
} 2>&1 | tee "$OUT_DIR/slurm_${SLURM_JOB_ID:-local}.log"
