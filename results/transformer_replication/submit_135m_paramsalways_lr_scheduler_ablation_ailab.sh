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

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-${OUT_BASE_DIR}/jobs_135m_paramsalways_lr_scheduler_ablation_${RUN_TS}.tsv}"
RECORD_PATH="${RECORD_PATH:-${OUT_BASE_DIR}/distill_135m_paramsalways_lr_scheduler_ablation_${RUN_TS}.json}"
DRY_RUN="${DRY_RUN:-0}"

LRS_STR="${LRS:-5e-4 1e-4 5e-5 1e-5 5e-6 1e-6}"
SCHEDULERS_STR="${SCHEDULERS:-cosine linear}"
CONDA_ENV="${CONDA_ENV:-torch-env}"
PARTITION="${PARTITION:-ailab}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CHAIN_COUNT="${CHAIN_COUNT:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
MEM="${MEM:-}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-4000}"
MIN_LR="${MIN_LR:-1e-6}"

lr_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//+/p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

printf "lr\tscheduler\tstage\tjob_id\tjob_name\tdependency\ttime_limit\tout_dir\tlog_out\tlog_err\n" > "${MANIFEST}"

echo "Submitting 135M params-always LR x scheduler ablation"
echo "data_csv=${DATA_CSV}"
echo "sp2013_csv=${SP2013_CSV}"
echo "run_ts=${RUN_TS}"
echo "lrs=${LRS_STR}"
echo "schedulers=${SCHEDULERS_STR}"
echo "min_lr=${MIN_LR}"
echo "manifest=${MANIFEST}"
if [[ -n "${ACCOUNT}" ]]; then
  echo "account=${ACCOUNT}"
fi
if [[ -n "${QOS}" ]]; then
  echo "qos=${QOS}"
fi
if [[ -n "${MEM}" ]]; then
  echo "mem=${MEM}"
fi
if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
  echo "sbatch_extra_args=${SBATCH_EXTRA_ARGS}"
fi
echo

submission_tsv="$(mktemp)"
cleanup() {
  rm -f "${submission_tsv}"
}
trap cleanup EXIT

submitted=0

for lr in ${LRS_STR}; do
  lr_suffix="$(lr_tag "${lr}")"
  for scheduler in ${SCHEDULERS_STR}; do
    case "${scheduler}" in
      cosine) sched_suffix="cos" ;;
      linear) sched_suffix="lin" ;;
      *) echo "unknown scheduler: ${scheduler}" >&2; exit 2 ;;
    esac

    run_name="smol2_135m_synth_unique_seed1_all1000_mincols_paramsalways_pretrained_matchspval_iduma_lr${lr_suffix}_${sched_suffix}_e5_${RUN_TS}"
    out_dir="${OUT_BASE_DIR}/${run_name}"
    prev_job_id=""

    for stage in $(seq 1 "${CHAIN_COUNT}"); do
      job_name="s135_${sched_suffix}_lr${lr_suffix}_p${stage}"
      log_out="${LOG_DIR}/${job_name}_${RUN_TS}_%j.out"
      log_err="${LOG_DIR}/${job_name}_${RUN_TS}_%j.err"
      dependency=""
      resume_from="none"
      require_resume="0"
      if [[ "${stage}" -gt 1 ]]; then
        dependency="afterany:${prev_job_id}"
        resume_from="auto"
        require_resume="1"
      fi

      export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},DATA_CSV=${DATA_CSV},SP2013_CSV=${SP2013_CSV},OUT_DIR=${out_dir},MODEL_NAME=HuggingFaceTB/SmolLM2-135M,INIT_FROM_SCRATCH=0,EPOCHS=5,LR=${lr},MIN_LR=${MIN_LR},LR_SCHEDULER_TYPE=${scheduler},BEST_BY=id_val_acc,VAL_FRAC=0,TEST_FRAC=0,BATCH_SIZE=512,EVAL_BATCH_SIZE=512,MAX_LENGTH=256,WARMUP_RATIO=0.03,AMP_DTYPE=auto,GC=0,EVALS_PER_EPOCH=1,EVAL_SP2013=0,EVAL_ID_FINAL_ANSWER=1,ID_EVAL_SPLIT=val,ID_EVAL_SIZE=0,ID_EVAL_TARGET=response,ID_EVAL_REPORT_TRUE_TARGET=1,ID_EVAL_BATCH_SIZE=64,ID_EVAL_MAX_NEW_TOKENS=256,MOVE_SP2013_ROWS_TO_VAL=1,TRAIN_PROMPT_STUDENT_MODE=always,TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,TRAIN_PROMPT_STUDENT_DROPOUT_SEED=0,SAVE_EVAL_CHECKPOINTS=0,SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1,VERIFY_SAVED_CHECKPOINT_LOAD=0,SAVE_LAST_EVERY_UPDATES=${SAVE_LAST_EVERY_UPDATES},RESUME_FROM=${resume_from},REQUIRE_RESUME=${require_resume},GPUS_PER_NODE=${GPUS_PER_NODE}"

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
      if [[ -n "${dependency}" ]]; then
        submit_args+=(--dependency="${dependency}")
      fi
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
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${lr}" "${scheduler}" "${stage}" "DRY_RUN" "${job_name}" "${dependency:-none}" "${TIME_LIMIT}" "${out_dir}" "${log_out}" "${log_err}" \
          >> "${MANIFEST}"
        prev_job_id="DRY_RUN"
        continue
      fi

      submit_out="$(sbatch "${submit_args[@]}" "${SCRIPT_PATH}")"
      job_id="${submit_out%%;*}"
      prev_job_id="${job_id}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${lr}" "${scheduler}" "${stage}" "${job_id}" "${job_name}" "${dependency:-none}" "${TIME_LIMIT}" "${out_dir}" "${log_out}" "${log_err}" \
        >> "${MANIFEST}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${job_id}" "${lr}" "${scheduler}" "${stage}" "${job_name}" "${out_dir}" "${dependency:-none}" \
        >> "${submission_tsv}"
      echo "submitted ${job_id}: lr=${lr} scheduler=${scheduler} stage=${stage}/${CHAIN_COUNT}"
      if [[ -n "${dependency}" ]]; then
        echo "  dependency=${dependency}"
      fi
      echo "  out_dir=${out_dir}"
      submitted=$((submitted + 1))
    done
  done
done

python - "${submission_tsv}" "${RECORD_PATH}" "${RUN_TS}" "${LRS_STR}" "${SCHEDULERS_STR}" <<'PY'
import json
import sys

submission_tsv, record_path, run_ts, lrs_str, schedulers_str = sys.argv[1:6]
lrs = [float(x) for x in lrs_str.split()]
schedulers = schedulers_str.split()
jobs = {}
with open(submission_tsv, "r", encoding="utf-8") as f:
    for line in f:
        job_id, lr, scheduler, stage, job_name, out_dir, dependency = line.rstrip("\n").split("\t")
        key = f"lr_{lr}_scheduler_{scheduler}_stage_{stage}"
        jobs[key] = {
            "job_id": job_id,
            "lr": float(lr),
            "scheduler": scheduler,
            "stage": int(stage),
            "job_name": job_name,
            "out_dir": out_dir,
            "dependency": None if dependency == "none" else dependency,
        }

payload = {
    "run_ts": run_ts,
    "note": (
        "135M pretrained params-always distillation ablation over learning rate and scheduler. "
        "Uses SP2013-only validation (16,000 matched-format prompts), best_by=id_val_acc against UMA responses, "
        "and logs true-answer accuracy as a secondary diagnostic from the same generations. "
        "Schedulers decay to a minimum learning-rate floor instead of zero."
    ),
    "model": "HuggingFaceTB/SmolLM2-135M",
    "init": "pretrained",
    "epochs": 5,
    "min_learning_rate": 1e-6,
    "learning_rates": lrs,
    "schedulers": schedulers,
    "validation": {
        "sp2013_only": True,
        "expected_rows": 16000,
        "expected_unique_probs": 16,
        "expected_unique_student_tuples": 1000,
        "primary_metric": "id_val_acc_against_response_nl",
        "secondary_metric": "id_val_true_acc_against_true_answer",
    },
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
