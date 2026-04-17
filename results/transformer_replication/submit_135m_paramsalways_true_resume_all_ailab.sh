#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_PATH="${ROOT_DIR}/results/transformer_replication/slurm_train_fraction2400_smol135_ailab.sbatch"
DATA_CSV="${DATA_CSV:-${ROOT_DIR}/results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${ROOT_DIR}/results/transformer_replication/jobs_135m_paramsalways_lr_scheduler_ablation_20260409_104738.tsv}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/transformer_replication/logs}"
mkdir -p "${LOG_DIR}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/results/transformer_replication/jobs_135m_paramsalways_true_resume_all_${RUN_TS}.tsv}"
DRY_RUN="${DRY_RUN:-0}"

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
MEM="${MEM:-64G}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-4000}"
EPOCHS="${EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-6}"
CHAIN_COUNT="${CHAIN_COUNT:-2}"

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "missing source manifest: ${SOURCE_MANIFEST}" >&2
  exit 1
fi

printf "lr\tscheduler\tstage\tjob_id\tjob_name\tdependency\tout_dir\tlog_out\tlog_err\n" > "${MANIFEST}"

echo "Submitting 135M params-always true resumes for all LR ablation models"
echo "source_manifest=${SOURCE_MANIFEST}"
echo "run_ts=${RUN_TS}"
echo "epochs=${EPOCHS}"
echo "min_lr=${MIN_LR}"
echo "batch_size=${BATCH_SIZE}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "id_eval_batch_size=${ID_EVAL_BATCH_SIZE}"
echo "chain_count=${CHAIN_COUNT}"
echo "manifest=${MANIFEST}"
echo

while IFS=$'\t' read -r lr scheduler stage _job_id _job_name _dependency _time_limit out_dir _log_out _log_err; do
  if [[ "${lr}" == "lr" || "${stage}" != "1" ]]; then
    continue
  fi

  if [[ "${scheduler}" == "cosine" ]]; then
    sched_tag="cosine"
  elif [[ "${scheduler}" == "linear" ]]; then
    sched_tag="linear"
  else
    echo "unknown scheduler in manifest: ${scheduler}" >&2
    exit 2
  fi

  if [[ "${DRY_RUN}" != "1" && ! -f "${out_dir}/last/trainer_state.pt" ]]; then
    echo "missing resumable checkpoint: ${out_dir}/last/trainer_state.pt" >&2
    exit 1
  fi

  lr_tag="$(printf '%s' "${lr}" | sed 's/\./p/g; s/-/m/g')"
  prev_job_id=""

  for resume_stage in $(seq 1 "${CHAIN_COUNT}"); do
    job_name="r135_${sched_tag}_lr${lr_tag}_p${resume_stage}"
    log_out="${LOG_DIR}/${job_name}_${RUN_TS}_%j.out"
    log_err="${LOG_DIR}/${job_name}_${RUN_TS}_%j.err"
    dependency="none"
    dep_args=()
    if [[ "${resume_stage}" -gt 1 ]]; then
      dependency="afternotok:${prev_job_id}"
      dep_args+=(--dependency="${dependency}")
    fi

    export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},DATA_CSV=${DATA_CSV},SP2013_CSV=${SP2013_CSV},OUT_DIR=${out_dir},MODEL_NAME=HuggingFaceTB/SmolLM2-135M,INIT_FROM_SCRATCH=0,EPOCHS=${EPOCHS},LR=${lr},MIN_LR=${MIN_LR},LR_SCHEDULER_TYPE=${scheduler},BEST_BY=id_val_acc,VAL_FRAC=0,TEST_FRAC=0,BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_LENGTH=${MAX_LENGTH},NUM_WORKERS=${NUM_WORKERS},WARMUP_RATIO=0.03,AMP_DTYPE=${AMP_DTYPE},GC=0,EVALS_PER_EPOCH=1,LOSS_EVALS_PER_EPOCH=${LOSS_EVALS_PER_EPOCH},EVAL_SP2013=0,EVAL_ID_FINAL_ANSWER=1,ID_EVAL_EVERY=${ID_EVAL_EVERY},ID_EVAL_SPLIT=val,ID_EVAL_SIZE=0,ID_EVAL_TARGET=response,ID_EVAL_REPORT_TRUE_TARGET=1,ID_EVAL_BATCH_SIZE=${ID_EVAL_BATCH_SIZE},ID_EVAL_MAX_NEW_TOKENS=256,MOVE_SP2013_ROWS_TO_VAL=1,TRAIN_PROMPT_STUDENT_MODE=always,TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,TRAIN_PROMPT_STUDENT_DROPOUT_SEED=0,SAVE_EVAL_CHECKPOINTS=0,SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1,VERIFY_SAVED_CHECKPOINT_LOAD=0,SAVE_LAST_EVERY_UPDATES=${SAVE_LAST_EVERY_UPDATES},LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY},PREVIEW_SAMPLES=${PREVIEW_SAMPLES},RESUME_FROM=auto,REQUIRE_RESUME=1,GPUS_PER_NODE=${GPUS_PER_NODE}"

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
    if [[ -n "${MEM}" ]]; then
      submit_args+=(--mem="${MEM}")
    fi
    if [[ -n "${ACCOUNT}" ]]; then
      submit_args+=(--account="${ACCOUNT}")
    fi
    if [[ -n "${QOS}" ]]; then
      submit_args+=(--qos="${QOS}")
    fi
    if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
      read -r -a extra_submit_args <<< "${SBATCH_EXTRA_ARGS}"
      submit_args+=("${extra_submit_args[@]}")
    fi
    submit_args+=("${dep_args[@]}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf "[dry-run] sbatch"
      printf " %q" "${submit_args[@]}" "${SCRIPT_PATH}"
      printf "\n"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${lr}" "${scheduler}" "${resume_stage}" "DRY_RUN" "${job_name}" "${dependency}" "${out_dir}" "${log_out}" "${log_err}" \
        >> "${MANIFEST}"
      prev_job_id="DRY_RUN"
      continue
    fi

    submit_out="$(sbatch "${submit_args[@]}" "${SCRIPT_PATH}")"
    job_id="${submit_out%%;*}"
    prev_job_id="${job_id}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${lr}" "${scheduler}" "${resume_stage}" "${job_id}" "${job_name}" "${dependency}" "${out_dir}" "${log_out}" "${log_err}" \
      >> "${MANIFEST}"
    echo "submitted ${job_id}: lr=${lr} scheduler=${scheduler} stage=${resume_stage}/${CHAIN_COUNT}"
    if [[ "${dependency}" != "none" ]]; then
      echo "  dependency=${dependency}"
    fi
  done
done < "${SOURCE_MANIFEST}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run only. No jobs were submitted."
else
  echo
  echo "Submitted resume jobs."
fi
echo "Manifest: ${MANIFEST}"
