#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_PATH="${ROOT_DIR}/results/transformer_replication/slurm_train_fraction2400_smol135_ailab.sbatch"
DATA_CSV="${DATA_CSV:-${ROOT_DIR}/results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
OUT_BASE_DIR="${OUT_BASE_DIR:-${ROOT_DIR}/results/transformer_replication}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/transformer_replication/logs}"
mkdir -p "${LOG_DIR}"

DEFAULT_SOURCE_KEYS="bestresp_5em4_cos|besttrue_1em4_lin"
DEFAULT_SOURCE_DIRS="${OUT_BASE_DIR}/smol2_135m_synth_unique_seed1_all1000_mincols_paramsalways_pretrained_matchspval_iduma_lr5em4_cos_e5_20260409_104738/best|${OUT_BASE_DIR}/smol2_135m_synth_unique_seed1_all1000_mincols_paramsalways_pretrained_matchspval_iduma_lr1em4_lin_e5_20260409_104738/best"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-${OUT_BASE_DIR}/jobs_135m_paramsalways_low_lr_continue_${RUN_TS}.tsv}"
RECORD_PATH="${RECORD_PATH:-${OUT_BASE_DIR}/distill_135m_paramsalways_low_lr_continue_${RUN_TS}.json}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_KEYS_STR="${SOURCE_KEYS:-${DEFAULT_SOURCE_KEYS}}"
SOURCE_DIRS_STR="${SOURCE_DIRS:-${DEFAULT_SOURCE_DIRS}}"

CONDA_ENV="${CONDA_ENV:-torch-env}"
PARTITION="${PARTITION:-ailab}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
ID_EVAL_BATCH_SIZE="${ID_EVAL_BATCH_SIZE:-24}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_WORKERS="${NUM_WORKERS:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-0}"
ID_EVAL_EVERY="${ID_EVAL_EVERY:-1}"
LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH:-4}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
MEM="${MEM:-}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-4000}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
MIN_LR="${MIN_LR:-1e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
TMP_ROOT="${TMP_ROOT:-/n/fs/cogai/cs1095/tmp}"

mkdir -p "${TMP_ROOT}"

IFS='|' read -r -a SOURCE_KEYS_ARR <<< "${SOURCE_KEYS_STR}"
IFS='|' read -r -a SOURCE_DIRS_ARR <<< "${SOURCE_DIRS_STR}"

if [[ "${#SOURCE_KEYS_ARR[@]}" -ne "${#SOURCE_DIRS_ARR[@]}" ]]; then
  echo "SOURCE_KEYS and SOURCE_DIRS length mismatch" >&2
  exit 2
fi

printf "source_key\tsource_dir\tjob_id\tjob_name\ttime_limit\tout_dir\tlog_out\tlog_err\n" > "${MANIFEST}"

echo "Submitting 135M params-always low-LR continuation probes"
echo "data_csv=${DATA_CSV}"
echo "sp2013_csv=${SP2013_CSV}"
echo "run_ts=${RUN_TS}"
echo "epochs=${EPOCHS}"
echo "lr=${LR}"
echo "min_lr=${MIN_LR}"
echo "lr_scheduler_type=${LR_SCHEDULER_TYPE}"
echo "batch_size=${BATCH_SIZE}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "id_eval_batch_size=${ID_EVAL_BATCH_SIZE}"
echo "manifest=${MANIFEST}"
echo

submission_tsv="$(mktemp -p "${TMP_ROOT}" "jobs_135m_paramsalways_low_lr_continue_${RUN_TS}_XXXX.tsv")"
cleanup() {
  rm -f "${submission_tsv}"
}
trap cleanup EXIT

submitted=0

for idx in "${!SOURCE_KEYS_ARR[@]}"; do
  source_key="${SOURCE_KEYS_ARR[$idx]}"
  source_dir="${SOURCE_DIRS_ARR[$idx]}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${source_dir}/model.safetensors" ]]; then
      echo "missing model checkpoint: ${source_dir}/model.safetensors" >&2
      exit 1
    fi
    if [[ ! -f "${source_dir}/train_args.json" ]]; then
      echo "missing train args: ${source_dir}/train_args.json" >&2
      exit 1
    fi
  fi

  run_name="smol2_135m_synth_unique_seed1_all1000_mincols_paramsalways_pretrained_cont_${source_key}_lr$(printf '%s' "${LR}" | sed 's/\./p/g; s/-/m/g')_${LR_SCHEDULER_TYPE}_e${EPOCHS}_${RUN_TS}"
  out_dir="${OUT_BASE_DIR}/${run_name}"
  job_name="c135_${source_key}"
  log_out="${LOG_DIR}/${job_name}_${RUN_TS}_%j.out"
  log_err="${LOG_DIR}/${job_name}_${RUN_TS}_%j.err"

  export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},DATA_CSV=${DATA_CSV},SP2013_CSV=${SP2013_CSV},OUT_DIR=${out_dir},MODEL_NAME=${source_dir},INIT_FROM_SCRATCH=0,EPOCHS=${EPOCHS},LR=${LR},MIN_LR=${MIN_LR},LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE},BEST_BY=id_val_acc,VAL_FRAC=0,TEST_FRAC=0,BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_LENGTH=${MAX_LENGTH},NUM_WORKERS=${NUM_WORKERS},WARMUP_RATIO=0.03,AMP_DTYPE=${AMP_DTYPE},GC=0,EVALS_PER_EPOCH=1,LOSS_EVALS_PER_EPOCH=${LOSS_EVALS_PER_EPOCH},EVAL_SP2013=0,EVAL_ID_FINAL_ANSWER=1,ID_EVAL_EVERY=${ID_EVAL_EVERY},ID_EVAL_SPLIT=val,ID_EVAL_SIZE=0,ID_EVAL_TARGET=response,ID_EVAL_REPORT_TRUE_TARGET=1,ID_EVAL_BATCH_SIZE=${ID_EVAL_BATCH_SIZE},ID_EVAL_MAX_NEW_TOKENS=256,MOVE_SP2013_ROWS_TO_VAL=1,TRAIN_PROMPT_STUDENT_MODE=always,TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,TRAIN_PROMPT_STUDENT_DROPOUT_SEED=0,SAVE_EVAL_CHECKPOINTS=0,SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1,VERIFY_SAVED_CHECKPOINT_LOAD=0,SAVE_LAST_EVERY_UPDATES=${SAVE_LAST_EVERY_UPDATES},LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY},PREVIEW_SAMPLES=${PREVIEW_SAMPLES},RESUME_FROM=none,REQUIRE_RESUME=0,GPUS_PER_NODE=${GPUS_PER_NODE}"

  submit_args=(
    --parsable
    --partition="${PARTITION}"
    --time="${TIME_LIMIT}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --job-name="${job_name}"
    --output="${log_out}"
    --error="${log_err}"
    --export="${export_vars}"
  )
  if [[ -n "${ACCOUNT}" ]]; then
    submit_args+=(--account="${ACCOUNT}")
  fi
  if [[ -n "${QOS}" ]]; then
    submit_args+=(--qos="${QOS}")
  fi
  if [[ -n "${MEM}" ]]; then
    submit_args+=(--mem="${MEM}")
  fi
  if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
    read -r -a extra_submit_args <<< "${SBATCH_EXTRA_ARGS}"
    submit_args+=("${extra_submit_args[@]}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[dry-run] sbatch"
    printf " %q" "${submit_args[@]}" "${SCRIPT_PATH}"
    printf "\n"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${source_key}" "${source_dir}" "DRY_RUN" "${job_name}" "${TIME_LIMIT}" "${out_dir}" "${log_out}" "${log_err}" \
      >> "${MANIFEST}"
    continue
  fi

  submit_out="$(sbatch "${submit_args[@]}" "${SCRIPT_PATH}")"
  job_id="${submit_out%%;*}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${source_key}" "${source_dir}" "${job_id}" "${job_name}" "${TIME_LIMIT}" "${out_dir}" "${log_out}" "${log_err}" \
    >> "${MANIFEST}"
  printf "%s\t%s\t%s\t%s\n" \
    "${job_id}" "${source_key}" "${source_dir}" "${out_dir}" \
    >> "${submission_tsv}"
  echo "submitted ${job_id}: source=${source_key}"
  echo "  source_dir=${source_dir}"
  echo "  out_dir=${out_dir}"
  submitted=$((submitted + 1))
done

python - "${submission_tsv}" "${RECORD_PATH}" "${RUN_TS}" "${EPOCHS}" "${LR}" "${MIN_LR}" "${LR_SCHEDULER_TYPE}" "${BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${ID_EVAL_BATCH_SIZE}" "${GRAD_ACCUM_STEPS}" "${MAX_LENGTH}" "${NUM_WORKERS}" "${AMP_DTYPE}" "${LOCAL_FILES_ONLY}" <<'PY'
import json
import sys

submission_tsv, record_path, run_ts = sys.argv[1:4]
jobs = {}
with open(submission_tsv, "r", encoding="utf-8") as f:
    for line in f:
        job_id, source_key, source_dir, out_dir = line.rstrip("\n").split("\t")
        jobs[source_key] = {
            "job_id": job_id,
            "source_dir": source_dir,
            "out_dir": out_dir,
        }

payload = {
    "run_ts": run_ts,
    "note": (
        "135M pretrained continuation probes. Each run warm-starts from a saved pretrained checkpoint directory "
        "as MODEL_NAME, but does not resume optimizer/scheduler state. This intentionally resets optimization so a "
        "new low learning-rate schedule can test whether the pretrained student is still underfit after epoch 2."
    ),
    "epochs": int(sys.argv[4]),
    "learning_rate": float(sys.argv[5]),
    "min_learning_rate": float(sys.argv[6]),
    "lr_scheduler_type": sys.argv[7],
    "batch_size": int(sys.argv[8]),
    "eval_batch_size": int(sys.argv[9]),
    "id_eval_batch_size": int(sys.argv[10]),
    "grad_accum_steps": int(sys.argv[11]),
    "max_length": int(sys.argv[12]),
    "num_workers": int(sys.argv[13]),
    "amp_dtype": sys.argv[14],
    "local_files_only": sys.argv[15] == "1",
    "jobs": jobs,
}

with open(record_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run only. No jobs were submitted."
else
  echo
  echo "Submitted ${submitted} jobs."
fi
echo "Manifest: ${MANIFEST}"
echo "Submission record: ${RECORD_PATH}"
