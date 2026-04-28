#!/usr/bin/env bash
#SBATCH --job-name=qwen_humanft_lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/transformer_replication/slurm_logs/%x_%j.out
#SBATCH --error=results/transformer_replication/slurm_logs/%x_%j.err

# Human-data fine-tune with LoRA on a Qwen3 base model.
# All knobs come from environment variables so submit_qwen_humanft_lora.sh can
# parameterize a scan over model sizes. Defaults are set so the script also
# runs standalone for one-off jobs.

set -euo pipefail

ROOT="${REPO_ROOT:?REPO_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required (e.g. Qwen/Qwen3-0.6B-Base)}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$ROOT/results/transformer_replication/slurm_logs" "$OUT_DIR"

TASK_LABEL="${TASK_LABEL:-qwen_humanft_lora}"
TRAIN_CSV="${TRAIN_CSV:-$ROOT/data/human_ft/data_train_nlp.csv}"
VAL_CSV="${VAL_CSV:-$ROOT/data/human_ft/data_val_nlp.csv}"
INSTRUCTION_STYLE="${INSTRUCTION_STYLE:-plain}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GC="${GC:-1}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
KL_BETA="${KL_BETA:-0.0}"
BEST_BY="${BEST_BY:-val_nll}"
EVAL_SP2013="${EVAL_SP2013:-0}"
EVAL_LOADED_MODEL="${EVAL_LOADED_MODEL:-0}"
SP2013_NUM_ROLLOUTS="${SP2013_NUM_ROLLOUTS:-100}"
SP2013_TEMPERATURE="${SP2013_TEMPERATURE:-0.7}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-42}"

bool_flag() {
  if [[ "$1" == "1" ]]; then printf '%s' "$2"; else printf '%s' "$3"; fi
}

GC_FLAG=()
[[ "$GC" == "1" ]] && GC_FLAG=(--gradient_checkpointing)

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
EVAL_LOADED_FLAG="$(bool_flag "$EVAL_LOADED_MODEL" --eval_loaded_model --no-eval_loaded_model)"
LOCAL_FILES_FLAG="$(bool_flag "$LOCAL_FILES_ONLY" --local_files_only --no-local_files_only)"

{
  echo "[$(date -Iseconds)] starting ${TASK_LABEL}"
  echo "job_id=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-unknown}"
  echo "model_name=${MODEL_NAME}"
  echo "use_lora=${USE_LORA} r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
  echo "epochs=${EPOCHS} max_steps=${MAX_STEPS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} lr=${LR}"
  echo "instruction_style=${INSTRUCTION_STYLE} kl_beta=${KL_BETA} best_by=${BEST_BY}"
  echo "out_dir=${OUT_DIR}"
  nvidia-smi || true

  "$PYTHON_BIN" "$ROOT/human_ft/finetune_humandata.py" \
    --init_strategy pretrained \
    --base_model_name "$MODEL_NAME" \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --output_dir "$OUT_DIR" \
    --instruction_style "$INSTRUCTION_STYLE" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --lr "$LR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --num_workers "$NUM_WORKERS" \
    --amp_dtype "$AMP_DTYPE" \
    --kl_beta "$KL_BETA" \
    --best_by "$BEST_BY" \
    --sp2013_num_rollouts "$SP2013_NUM_ROLLOUTS" \
    --sp2013_temperature "$SP2013_TEMPERATURE" \
    --log_every "$LOG_EVERY" \
    --seed "$SEED" \
    "${LORA_ARGS[@]}" \
    "${GC_FLAG[@]}" \
    "$EVAL_SP2013_FLAG" \
    "$EVAL_LOADED_FLAG" \
    "$LOCAL_FILES_FLAG"

  echo "[$(date -Iseconds)] finished ${TASK_LABEL}"
} 2>&1 | tee "$OUT_DIR/slurm_${SLURM_JOB_ID:-local}.log"
