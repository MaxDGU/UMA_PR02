#!/usr/bin/env bash
# Submit human-data fine-tune jobs across Qwen3 model sizes.
# Mirrors the pattern of submit_qwen3_distill_1epoch.sh but targets the
# human-FT entrypoint (human_ft/finetune_humandata.py) and steps through
# {0.6B, 1.7B, 4B, 8B} so we can probe the size-scaling hypothesis: small
# models can't capture human behaviour from 288 rows; bigger ones might.
#
# Usage:
#   bash results/transformer_replication/submit_qwen_humanft_lora.sh
#   SMOKE=1 bash ...                       # smallest model, max_steps=20
#   SIZES="0p6b 1p7b" bash ...             # custom subset
#   PARTITION=pli ACCOUNT=nam bash ...     # cluster overrides
#
# Pre-flight: snapshots must be in $HF_HOME (compute nodes have no internet).
#   for s in 0.6B 1.7B 4B 8B; do huggingface-cli download Qwen/Qwen3-${s}-Base; done

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT_DIR"

TR_DIR="$ROOT_DIR/results/transformer_replication"
LOG_DIR="$TR_DIR/slurm_logs"
SBATCH_SCRIPT="$TR_DIR/slurm_train_qwen_humanft_lora.sbatch"
PYTHON_BIN="${PYTHON_BIN:-python}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="$TR_DIR/jobs_qwen_humanft_lora_${RUN_TS}.tsv"

TRAIN_CSV="${TRAIN_CSV:-$ROOT_DIR/data/human_ft/data_train_nlp.csv}"
VAL_CSV="${VAL_CSV:-$ROOT_DIR/data/human_ft/data_val_nlp.csv}"
INSTRUCTION_STYLE="${INSTRUCTION_STYLE:-plain}"
EPOCHS="${EPOCHS:-3}"
SMOKE="${SMOKE:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
KL_BETA="${KL_BETA:-0.0}"

PARTITION="${PARTITION:-all}"
ACCOUNT="${ACCOUNT:-}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

# Default size sweep. Override with SIZES="0p6b 1p7b 4b 8b".
DEFAULT_SIZES="0p6b 1p7b 4b 8b"
if [[ "$SMOKE" == "1" ]]; then
  DEFAULT_SIZES="0p6b"
  MAX_STEPS="20"
  EPOCHS="1"
  TIME_LIMIT="00:30:00"
fi
SIZES="${SIZES:-$DEFAULT_SIZES}"

mkdir -p "$LOG_DIR"

if ! PYTHON_BIN_RESOLVED="$(command -v "$PYTHON_BIN" 2>/dev/null)"; then
  echo "missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi
PYTHON_BIN="$PYTHON_BIN_RESOLVED"

required_paths=("$SBATCH_SCRIPT" "$TRAIN_CSV" "$VAL_CSV")
for required in "${required_paths[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

echo "Checking PEFT in $PYTHON_BIN"
timeout 90 "$PYTHON_BIN" - <<'PY'
import peft
print(f"peft={peft.__version__}")
PY

# Per-size config.
config_for_size() {
  case "$1" in
    0p6b) echo "Qwen/Qwen3-0.6B-Base 4 4 1e-4" ;;
    1p7b) echo "Qwen/Qwen3-1.7B-Base 4 4 1e-4" ;;
    4b)   echo "Qwen/Qwen3-4B-Base 2 8 1e-4" ;;
    8b)   echo "Qwen/Qwen3-8B-Base 1 16 1e-4" ;;
    *) echo "" ;;
  esac
}

echo "Pre-flight: checking HF cache for requested sizes"
for size in $SIZES; do
  cfg="$(config_for_size "$size")"
  if [[ -z "$cfg" ]]; then
    echo "unknown size: $size" >&2; exit 2
  fi
  model_name="$(echo "$cfg" | awk '{print $1}')"
  hub_dir="$HF_HOME/hub/models--${model_name//\//--}"
  if [[ ! -d "$hub_dir" ]]; then
    echo "missing snapshot for $model_name (expected $hub_dir)" >&2
    echo "  pre-fetch with: HF_HOME=$HF_HOME huggingface-cli download $model_name" >&2
    exit 3
  fi
done

printf "size\tmodel_name\tjob_id\tjob_name\tout_dir\tbatch\tgrad_accum\tlr\n" > "$MANIFEST"

submit_one() {
  local size="$1"
  local cfg model_name batch grad_accum lr out_dir job_name log_out log_err submit_out job_id export_vars

  cfg="$(config_for_size "$size")"
  model_name="$(echo "$cfg" | awk '{print $1}')"
  batch="$(echo "$cfg" | awk '{print $2}')"
  grad_accum="$(echo "$cfg" | awk '{print $3}')"
  lr="$(echo "$cfg" | awk '{print $4}')"

  out_dir="$TR_DIR/qwen_humanft_${size}_${RUN_TS}"
  [[ "$SMOKE" == "1" ]] && out_dir="${out_dir}_smoke"
  job_name="hft${size}"
  [[ "$SMOKE" == "1" ]] && job_name="${job_name}sm"

  log_out="$LOG_DIR/${job_name}_${RUN_TS}_%j.out"
  log_err="$LOG_DIR/${job_name}_${RUN_TS}_%j.err"

  export_vars="ALL,REPO_ROOT=${ROOT_DIR},PYTHON_BIN=${PYTHON_BIN},HF_HOME=${HF_HOME},TASK_LABEL=humanft_${size},MODEL_NAME=${model_name},OUT_DIR=${out_dir},TRAIN_CSV=${TRAIN_CSV},VAL_CSV=${VAL_CSV},INSTRUCTION_STYLE=${INSTRUCTION_STYLE},EPOCHS=${EPOCHS},MAX_STEPS=${MAX_STEPS},BATCH_SIZE=${batch},EVAL_BATCH_SIZE=${batch},GRAD_ACCUM_STEPS=${grad_accum},LR=${lr},KL_BETA=${KL_BETA},USE_LORA=1,GC=1,EVAL_SP2013=0,LOCAL_FILES_ONLY=1"

  sbatch_args=(
    --parsable
    --partition="$PARTITION"
    --gres="$GRES"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$TIME_LIMIT"
    --job-name="$job_name"
    --output="$log_out"
    --error="$log_err"
    --export="$export_vars"
  )
  [[ -n "$ACCOUNT" ]] && sbatch_args+=(--account="$ACCOUNT")

  submit_out="$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT")"
  job_id="${submit_out%%;*}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$size" "$model_name" "$job_id" "$job_name" "$out_dir" "$batch" "$grad_accum" "$lr" >> "$MANIFEST"
  echo "submitted $job_id: $job_name -> $out_dir"
}

echo "Submitting Qwen3 human-FT LoRA jobs"
echo "run_ts=$RUN_TS sizes=$SIZES smoke=$SMOKE epochs=$EPOCHS max_steps=$MAX_STEPS"
echo "partition=$PARTITION gres=$GRES time=$TIME_LIMIT account=${ACCOUNT:-(default)}"
echo

for size in $SIZES; do
  submit_one "$size"
done

echo
echo "Manifest: $MANIFEST"
cat "$MANIFEST"
