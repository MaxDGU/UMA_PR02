#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT_DIR"

TR_DIR="$ROOT_DIR/results/transformer_replication"
LOG_DIR="$TR_DIR/slurm_logs"
SBATCH_SCRIPT="$TR_DIR/slurm_train_qwen3_distill_1epoch.sbatch"
PYTHON_BIN="${PYTHON_BIN:-/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40/bin/python}"
HF_HOME="${HF_HOME:-/n/fs/cogai/.cache/hf}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="$TR_DIR/jobs_qwen3_distill_1epoch_${RUN_TS}.tsv"

FRACTION_CSV="${FRACTION_CSV:-$TR_DIR/synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_default_nosp2013_mincols.csv.gz}"
DECIMAL_CSV="${DECIMAL_CSV:-$TR_DIR/decimal_bss_mixture_add2400_mul4800_no3dpmul_repaired_v1_k100_xlsx_distill/uma_traces_decimal_bss_mix_add2400_mul4800_no3dpmul_repaired_v1_k100_xlsx_nlp_think_aloud_child_train_mincols.csv.gz}"

PARTITION="${PARTITION:-all}"
GRES="${GRES:-gpu:l40:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-neu317,neu322}"

mkdir -p "$LOG_DIR"

for required in "$SBATCH_SCRIPT" "$FRACTION_CSV" "$DECIMAL_CSV" "$PYTHON_BIN"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

echo "Checking PEFT availability in $PYTHON_BIN"
timeout 90 "$PYTHON_BIN" - <<'PY'
import peft
print(f"peft={peft.__version__}")
PY

echo "Checking local Qwen3-Base snapshots in $HF_HOME"
HF_HOME="$HF_HOME" python - <<'PY'
from pathlib import Path

models = [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
]
hub = Path(__import__("os").environ["HF_HOME"]) / "hub"
errors = []
for model in models:
    repo_dir = hub / ("models--" + model.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if not ref.exists():
        errors.append(f"{model}: missing refs/main")
        continue
    snapshot = repo_dir / "snapshots" / ref.read_text(encoding="utf-8").strip()
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        errors.append(f"{model}: no safetensors in {snapshot}")
        continue
    print(f"{model}: {snapshot} ({len(weights)} safetensors)")
if errors:
    raise SystemExit("Missing model cache entries:\n- " + "\n- ".join(errors))
PY

printf "dataset\tmodel_label\tmodel_name\tuse_lora\tjob_id\tjob_name\tout_dir\tdata_csv\tbatch_size\tgrad_accum\tid_eval_batch\n" > "$MANIFEST"

submit_one() {
  local dataset="$1"
  local model_label="$2"
  local model_name="$3"
  local use_lora="$4"
  local batch_size="$5"
  local grad_accum="$6"
  local id_eval_batch="$7"

  local data_csv prompt_kind max_length val_frac test_frac split_by_problem lr_scheduler id_eval_size id_eval_tokens best_by save_last lr min_lr out_dir job_name log_out log_err export_vars submit_out job_id

  if [[ "$dataset" == "fraction_panel25_default" ]]; then
    data_csv="$FRACTION_CSV"
    prompt_kind="fraction"
    max_length="288"
    val_frac="0.05"
    test_frac="0.05"
    split_by_problem="0"
    lr_scheduler="cosine"
    id_eval_size="64"
    id_eval_tokens="240"
    best_by="val_loss"
    save_last="2000"
    lr="5e-4"
    min_lr="1e-6"
    out_dir="$TR_DIR/qwen3_${model_label}_panel25_default_base_e1_${RUN_TS}"
    job_name="q3${model_label}_p25"
  elif [[ "$dataset" == "decimal_repaired_mix" ]]; then
    data_csv="$DECIMAL_CSV"
    prompt_kind="decimal"
    max_length="256"
    val_frac="0.01"
    test_frac="0.01"
    split_by_problem="1"
    lr_scheduler="linear"
    id_eval_size="2000"
    id_eval_tokens="192"
    best_by="id_val_acc"
    save_last="4000"
    lr="5e-4"
    min_lr="1e-6"
    out_dir="$TR_DIR/qwen3_${model_label}_decimal_bssmix_no3dpmul_repaired_v1_base_e1_${RUN_TS}"
    job_name="q3${model_label}_dec"
  else
    echo "unknown dataset: $dataset" >&2
    exit 2
  fi

  if [[ "$use_lora" == "1" ]]; then
    out_dir="${out_dir}_lora"
    job_name="${job_name}l"
  fi

  log_out="$LOG_DIR/${job_name}_${RUN_TS}_%j.out"
  log_err="$LOG_DIR/${job_name}_${RUN_TS}_%j.err"

  export_vars="ALL,REPO_ROOT=${ROOT_DIR},PYTHON_BIN=${PYTHON_BIN},HF_HOME=${HF_HOME},TASK_LABEL=${dataset}_${model_label},DATASET_KIND=${dataset},MODEL_NAME=${model_name},DATA_CSV=${data_csv},OUT_DIR=${out_dir},PROMPT_PROBLEM_KIND=${prompt_kind},EPOCHS=1,BATCH_SIZE=${batch_size},EVAL_BATCH_SIZE=${batch_size},GRAD_ACCUM_STEPS=${grad_accum},LR=${lr},MIN_LR=${min_lr},LR_SCHEDULER_TYPE=${lr_scheduler},MAX_LENGTH=${max_length},VAL_FRAC=${val_frac},TEST_FRAC=${test_frac},SPLIT_BY_PROBLEM=${split_by_problem},BEST_BY=${best_by},ID_EVAL_SIZE=${id_eval_size},ID_EVAL_BATCH_SIZE=${id_eval_batch},ID_EVAL_MAX_NEW_TOKENS=${id_eval_tokens},USE_LORA=${use_lora},MODEL_LOAD_DTYPE=bf16,GC=1,EVAL_SP2013=0,SP2013_USE_PARAM_GRID=0,EVAL_ID_FINAL_ANSWER=1,ID_EVAL_REPORT_TRUE_TARGET=1,MOVE_SP2013_ROWS_TO_VAL=0,STRICT_SP2013_ONLY_VALIDATION_LAYOUT=0,TRAIN_PROMPT_STUDENT_MODE=always,TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,SAVE_EVAL_CHECKPOINTS=0,SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1,VERIFY_SAVED_CHECKPOINT_LOAD=0,LOCAL_FILES_ONLY=1,PREVIEW_SAMPLES=0,RESUME_FROM=none,SAVE_LAST_EVERY_UPDATES=${save_last}"

  submit_out="$(
    sbatch \
      --parsable \
      --partition="$PARTITION" \
      --gres="$GRES" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME_LIMIT" \
      --exclude="$EXCLUDE_NODES" \
      --job-name="$job_name" \
      --output="$log_out" \
      --error="$log_err" \
      --export="$export_vars" \
      "$SBATCH_SCRIPT"
  )"
  job_id="${submit_out%%;*}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dataset" "$model_label" "$model_name" "$use_lora" "$job_id" "$job_name" "$out_dir" "$data_csv" "$batch_size" "$grad_accum" "$id_eval_batch" >> "$MANIFEST"
  echo "submitted $job_id: $job_name -> $out_dir"
}

echo "Submitting Qwen3 1-epoch distillation jobs"
echo "run_ts=$RUN_TS"
echo "fraction_csv=$FRACTION_CSV"
echo "decimal_csv=$DECIMAL_CSV"
echo "partition=$PARTITION gres=$GRES time=$TIME_LIMIT exclude=$EXCLUDE_NODES"

submit_one "fraction_panel25_default" "0p6b" "Qwen/Qwen3-0.6B-Base" "0" "32" "4" "16"
submit_one "fraction_panel25_default" "1p7b" "Qwen/Qwen3-1.7B-Base" "1" "32" "4" "8"
submit_one "fraction_panel25_default" "4b" "Qwen/Qwen3-4B-Base" "1" "16" "8" "4"
submit_one "fraction_panel25_default" "8b" "Qwen/Qwen3-8B-Base" "1" "8" "16" "2"

submit_one "decimal_repaired_mix" "0p6b" "Qwen/Qwen3-0.6B-Base" "0" "32" "4" "16"
submit_one "decimal_repaired_mix" "1p7b" "Qwen/Qwen3-1.7B-Base" "1" "32" "4" "8"
submit_one "decimal_repaired_mix" "4b" "Qwen/Qwen3-4B-Base" "1" "16" "8" "4"
submit_one "decimal_repaired_mix" "8b" "Qwen/Qwen3-8B-Base" "1" "8" "16" "2"

echo "Manifest: $MANIFEST"
cat "$MANIFEST"
