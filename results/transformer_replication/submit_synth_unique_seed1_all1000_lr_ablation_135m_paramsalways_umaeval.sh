#!/bin/bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

BASE_DIR="${ROOT_DIR}/results/transformer_replication"
SBATCH_SCRIPT="${BASE_DIR}/slurm_train_fraction2400_smol135_ailab.sbatch"

DATA_CSV="${DATA_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
SP2013_UMA_EVAL_CSV="${SP2013_UMA_EVAL_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013_eval/seed_1/sp2013_seed1_all_models.csv}"
OUT_BASE_DIR="${OUT_BASE_DIR:-${BASE_DIR}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
PARTITION="${PARTITION:-all}"
CONDA_ENV="${CONDA_ENV:-torch-env}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
MANIFEST="${MANIFEST:-${OUT_BASE_DIR}/jobs_synth_unique_seed1_all1000_lr_ablation_135m_paramsalways_umaeval_${RUN_TS}.tsv}"

LRS="${LRS:-5e-5,1e-5,5e-6,1e-6}"
SCHEDULERS="${SCHEDULERS:-linear,cosine}"

TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-192}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

BEST_BY="${BEST_BY:-sp2013_acc}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
EVAL_SP2013="${EVAL_SP2013:-1}"
SP2013_EVAL_TARGET="${SP2013_EVAL_TARGET:-uma}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-1}"
EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-0}"
MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-1}"

TRAIN_PROMPT_STUDENT_MODE="${TRAIN_PROMPT_STUDENT_MODE:-always}"
TRAIN_PROMPT_STUDENT_DROPOUT_PROB="${TRAIN_PROMPT_STUDENT_DROPOUT_PROB:-0.0}"
TRAIN_PROMPT_STUDENT_DROPOUT_SEED="${TRAIN_PROMPT_STUDENT_DROPOUT_SEED:-0}"

SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-0}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-0}"
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-0}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-0}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-0}"

SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-64}"
ID_EVAL_MAX_NEW_TOKENS="${ID_EVAL_MAX_NEW_TOKENS:-64}"
RESUME_FROM="${RESUME_FROM:-none}"
INIT_FROM_SCRATCH="${INIT_FROM_SCRATCH:-0}"
GC="${GC:-0}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${DATA_CSV}" && "${DRY_RUN}" != "1" ]]; then
  echo "missing data csv: ${DATA_CSV}" >&2
  exit 1
fi
if [[ ! -f "${SP2013_CSV}" && "${DRY_RUN}" != "1" ]]; then
  echo "missing sp2013 csv: ${SP2013_CSV}" >&2
  exit 1
fi
if [[ ! -f "${SP2013_UMA_EVAL_CSV}" && "${DRY_RUN}" != "1" ]]; then
  echo "missing sp2013 UMA eval csv: ${SP2013_UMA_EVAL_CSV}" >&2
  exit 1
fi

lr_tag() {
  case "$1" in
    5e-5) echo "5em5" ;;
    1e-5) echo "1em5" ;;
    5e-6) echo "5em6" ;;
    1e-6) echo "1em6" ;;
    *) echo "$1" | sed 's/e-/em/g; s/e/ep/g; s/[^A-Za-z0-9]//g' ;;
  esac
}

scheduler_tag() {
  case "$1" in
    linear) echo "lin" ;;
    cosine) echo "cos" ;;
    *) echo "$1" ;;
  esac
}

IFS=',' read -r -a LR_LIST <<< "${LRS}"
IFS=',' read -r -a SCHEDULER_LIST <<< "${SCHEDULERS}"

printf "scheduler\tlr\tjob_id\tjob_name\tpartition\ttime_limit\tgpus\tcpus\tbatch_size\teval_batch_size\tepochs\twarmup_steps\tout_dir\tlog_out\tlog_err\tdata_csv\tsp2013_uma_eval_csv\n" > "${MANIFEST}"

echo "Submitting 135M params-always LR ablation with UMA-label SP2013 selection"
echo "run_ts=${RUN_TS}"
echo "partition=${PARTITION}"
echo "schedulers=${SCHEDULERS}"
echo "lrs=${LRS}"
echo "data_csv=${DATA_CSV}"
echo "sp2013_uma_eval_csv=${SP2013_UMA_EVAL_CSV}"
echo "batch_size=${BATCH_SIZE}"
echo "epochs=${EPOCHS}"
echo "manifest=${MANIFEST}"
echo

submitted=0

for scheduler in "${SCHEDULER_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    sched_tag="$(scheduler_tag "${scheduler}")"
    lr_short="$(lr_tag "${lr}")"
    job_name="lra135_${sched_tag}_${lr_short}"
    out_dir="${OUT_BASE_DIR}/smol2_135m_s1000_paramsalways_umaeval_${sched_tag}_${lr_short}_${RUN_TS}"
    log_out="${LOG_DIR}/${job_name}_${RUN_TS}_%j.out"
    log_err="${LOG_DIR}/${job_name}_${RUN_TS}_%j.err"

    export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},DATA_CSV=${DATA_CSV},SP2013_CSV=${SP2013_CSV},SP2013_UMA_EVAL_CSV=${SP2013_UMA_EVAL_CSV},OUT_DIR=${out_dir},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},LR=${lr},LR_SCHEDULER=${scheduler},WARMUP_RATIO=${WARMUP_RATIO},WARMUP_STEPS=${WARMUP_STEPS},MAX_LENGTH=${MAX_LENGTH},NUM_WORKERS=${NUM_WORKERS},AMP_DTYPE=${AMP_DTYPE},BEST_BY=${BEST_BY},EVALS_PER_EPOCH=${EVALS_PER_EPOCH},EVAL_SP2013=${EVAL_SP2013},SP2013_EVAL_TARGET=${SP2013_EVAL_TARGET},SP2013_USE_PARAM_GRID=${SP2013_USE_PARAM_GRID},EVAL_ID_FINAL_ANSWER=${EVAL_ID_FINAL_ANSWER},MOVE_SP2013_ROWS_TO_VAL=${MOVE_SP2013_ROWS_TO_VAL},TRAIN_PROMPT_STUDENT_MODE=${TRAIN_PROMPT_STUDENT_MODE},TRAIN_PROMPT_STUDENT_DROPOUT_PROB=${TRAIN_PROMPT_STUDENT_DROPOUT_PROB},TRAIN_PROMPT_STUDENT_DROPOUT_SEED=${TRAIN_PROMPT_STUDENT_DROPOUT_SEED},SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS},SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT},SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT},VERIFY_SAVED_CHECKPOINT_LOAD=${VERIFY_SAVED_CHECKPOINT_LOAD},PREVIEW_SAMPLES=${PREVIEW_SAMPLES},SP2013_MAX_NEW_TOKENS=${SP2013_MAX_NEW_TOKENS},ID_EVAL_MAX_NEW_TOKENS=${ID_EVAL_MAX_NEW_TOKENS},SAVE_LAST_EVERY_UPDATES=${SAVE_LAST_EVERY_UPDATES},RESUME_FROM=${RESUME_FROM},INIT_FROM_SCRATCH=${INIT_FROM_SCRATCH},GC=${GC}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] sbatch --parsable --partition=${PARTITION} --time=${TIME_LIMIT} --gpus-per-node=${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} --job-name=${job_name} --output=${log_out} --error=${log_err} --export=${export_vars} ${SBATCH_SCRIPT}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${scheduler}" "${lr}" "DRY_RUN" "${job_name}" "${PARTITION}" "${TIME_LIMIT}" "${GPUS_PER_NODE}" "${CPUS_PER_TASK}" "${BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${EPOCHS}" "${WARMUP_STEPS}" "${out_dir}" "${log_out}" "${log_err}" "${DATA_CSV}" "${SP2013_UMA_EVAL_CSV}" >> "${MANIFEST}"
      continue
    fi

    submit_out="$(
      sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --job-name="${job_name}" \
        --output="${log_out}" \
        --error="${log_err}" \
        --export="${export_vars}" \
        "${SBATCH_SCRIPT}"
    )"
    job_id="${submit_out%%;*}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${scheduler}" "${lr}" "${job_id}" "${job_name}" "${PARTITION}" "${TIME_LIMIT}" "${GPUS_PER_NODE}" "${CPUS_PER_TASK}" "${BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${EPOCHS}" "${WARMUP_STEPS}" "${out_dir}" "${log_out}" "${log_err}" "${DATA_CSV}" "${SP2013_UMA_EVAL_CSV}" >> "${MANIFEST}"

    echo "submitted ${job_id}: scheduler=${scheduler} lr=${lr}"
    echo "  out_dir=${out_dir}"
    submitted=$((submitted + 1))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run only. No jobs were submitted."
else
  echo
  echo "Submitted ${submitted} jobs."
fi
echo "Manifest: ${MANIFEST}"
