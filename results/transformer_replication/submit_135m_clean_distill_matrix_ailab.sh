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
ROOT_PREFIX="${ROOT_PREFIX:-$(cd "${ROOT_DIR}/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT_PREFIX}/conda-envs/uma-transformer-l40}"

CORRECT_EXEC_CSV="${CORRECT_EXEC_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_clean_child_correct_exec_mincols.csv.gz}"
CORRECT_EXEC_AND_ANSWER_CSV="${CORRECT_EXEC_AND_ANSWER_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_clean_child_correct_exec_and_answer_mincols.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
SP2013_TARGET_CSV="${SP2013_TARGET_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013_eval/seed_1/sp2013_seed1_all_models.csv}"
SP2013_VAL_SOURCE_CSV="${SP2013_VAL_SOURCE_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"

OUT_BASE_DIR="${OUT_BASE_DIR:-${BASE_DIR}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-${OUT_BASE_DIR}/jobs_135m_clean_distill_matrix_${RUN_TS}.tsv}"
DRY_RUN="${DRY_RUN:-0}"

PARTITION="${PARTITION:-all}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
CONDA_ENV="${CONDA_ENV:-${ENV_PREFIX}}"

BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-0.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-192}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-1}"
LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH:-4}"
MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-1}"
VAL_FRAC="${VAL_FRAC:-0}"
TEST_FRAC="${TEST_FRAC:-0}"
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-0}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-0}"
SAVE_LAST_EVERY_UPDATES="${SAVE_LAST_EVERY_UPDATES:-0}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-0}"

SP2013_MAX_NEW_TOKENS="${SP2013_MAX_NEW_TOKENS:-256}"
SP2013_TEMPERATURE="${SP2013_TEMPERATURE:-0.7}"
SP2013_TOP_P="${SP2013_TOP_P:-0.95}"
SP2013_TOP_K="${SP2013_TOP_K:-50}"
SP2013_SAMPLE_SEED="${SP2013_SAMPLE_SEED:-123}"

mkdir -p "${LOG_DIR}"

for path in "${CORRECT_EXEC_CSV}" "${CORRECT_EXEC_AND_ANSWER_CSV}" "${SP2013_CSV}" "${SP2013_TARGET_CSV}" "${SP2013_VAL_SOURCE_CSV}"; do
  if [[ ! -f "${path}" && "${DRY_RUN}" != "1" ]]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
done

printf "data_condition\tprompt_condition\tjob_id\tjob_name\tpartition\ttime_limit\tlr\tlr_scheduler_type\tepochs\tloss_evals_per_epoch\tbest_by\ttrain_prompt_student_mode\tsp2013_target_mode\tsp2013_use_param_grid\tsp2013_num_rollouts\tsp2013_do_sample\tout_dir\tlog_out\tlog_err\tdata_csv\n" > "${MANIFEST}"

submit_one() {
  local data_condition="$1"
  local prompt_condition="$2"
  local data_csv="$3"
  local prompt_mode="$4"
  local sp2013_use_param_grid="$5"
  local sp2013_target_mode="$6"
  local sp2013_num_rollouts="$7"
  local sp2013_do_sample="$8"
  local sp2013_rollout_batch_size="$9"
  local sp2013_progress_every="${10}"

  local data_tag prompt_tag job_name run_name out_dir log_out log_err export_vars submit_out job_id
  data_tag="$(tr '[:upper:]' '[:lower:]' <<< "${data_condition}")"
  prompt_tag="$(tr '[:upper:]' '[:lower:]' <<< "${prompt_condition}")"
  job_name="c135_${data_tag}_${prompt_tag}"
  run_name="smol2_135m_cleanchild_${data_tag}_${prompt_tag}_lr1em4_lin_e2_${RUN_TS}"
  out_dir="${OUT_BASE_DIR}/${run_name}"
  log_out="${LOG_DIR}/${job_name}_${RUN_TS}_%j.out"
  log_err="${LOG_DIR}/${job_name}_${RUN_TS}_%j.err"

  export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},DATA_CSV=${data_csv},SP2013_CSV=${SP2013_CSV},SP2013_TARGET_CSV=${SP2013_TARGET_CSV},SP2013_VAL_SOURCE_CSV=${SP2013_VAL_SOURCE_CSV},OUT_DIR=${out_dir},MODEL_NAME=HuggingFaceTB/SmolLM2-135M,EPOCHS=${EPOCHS},LR=${LR},MIN_LR=${MIN_LR},LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE},WARMUP_RATIO=${WARMUP_RATIO},BEST_BY=sp2013_primary,VAL_FRAC=${VAL_FRAC},TEST_FRAC=${TEST_FRAC},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_LENGTH=${MAX_LENGTH},NUM_WORKERS=${NUM_WORKERS},AMP_DTYPE=${AMP_DTYPE},EVALS_PER_EPOCH=${EVALS_PER_EPOCH},LOSS_EVALS_PER_EPOCH=${LOSS_EVALS_PER_EPOCH},EVAL_SP2013=1,SP2013_TARGET_MODE=${sp2013_target_mode},SP2013_USE_PARAM_GRID=${sp2013_use_param_grid},SP2013_NUM_ROLLOUTS=${sp2013_num_rollouts},SP2013_DO_SAMPLE=${sp2013_do_sample},SP2013_TEMPERATURE=${SP2013_TEMPERATURE},SP2013_TOP_P=${SP2013_TOP_P},SP2013_TOP_K=${SP2013_TOP_K},SP2013_SAMPLE_SEED=${SP2013_SAMPLE_SEED},SP2013_ROLLOUT_BATCH_SIZE=${sp2013_rollout_batch_size},SP2013_PROGRESS_EVERY=${sp2013_progress_every},SP2013_MAX_NEW_TOKENS=${SP2013_MAX_NEW_TOKENS},EVAL_ID_FINAL_ANSWER=0,MOVE_SP2013_ROWS_TO_VAL=${MOVE_SP2013_ROWS_TO_VAL},STRICT_SP2013_ONLY_VALIDATION_LAYOUT=1,TRAIN_PROMPT_STUDENT_MODE=${prompt_mode},TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,TRAIN_PROMPT_STUDENT_DROPOUT_SEED=0,SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS},SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT},SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT},VERIFY_SAVED_CHECKPOINT_LOAD=${VERIFY_SAVED_CHECKPOINT_LOAD},SAVE_LAST_EVERY_UPDATES=${SAVE_LAST_EVERY_UPDATES},PREVIEW_SAMPLES=${PREVIEW_SAMPLES},LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${job_name} data=${data_condition} prompt=${prompt_condition}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${data_condition}" "${prompt_condition}" "DRY_RUN" "${job_name}" "${PARTITION}" "${TIME_LIMIT}" \
      "${LR}" "${LR_SCHEDULER_TYPE}" "${EPOCHS}" "${LOSS_EVALS_PER_EPOCH}" "sp2013_primary" "${prompt_mode}" "${sp2013_target_mode}" \
      "${sp2013_use_param_grid}" "${sp2013_num_rollouts}" "${sp2013_do_sample}" \
      "${out_dir}" "${log_out}" "${log_err}" "${data_csv}" >> "${MANIFEST}"
    return
  fi

  submit_out="$(
    sbatch \
      --parsable \
      --partition="${PARTITION}" \
      --time="${TIME_LIMIT}" \
      --gpus-per-node="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --job-name="${job_name}" \
      --output="${log_out}" \
      --error="${log_err}" \
      --export="${export_vars}" \
      "${SBATCH_SCRIPT}"
  )"
  job_id="${submit_out%%;*}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${data_condition}" "${prompt_condition}" "${job_id}" "${job_name}" "${PARTITION}" "${TIME_LIMIT}" \
    "${LR}" "${LR_SCHEDULER_TYPE}" "${EPOCHS}" "${LOSS_EVALS_PER_EPOCH}" "sp2013_primary" "${prompt_mode}" "${sp2013_target_mode}" \
    "${sp2013_use_param_grid}" "${sp2013_num_rollouts}" "${sp2013_do_sample}" \
    "${out_dir}" "${log_out}" "${log_err}" "${data_csv}" >> "${MANIFEST}"
  echo "submitted ${job_id}: ${job_name} data=${data_condition} prompt=${prompt_condition}"
}

echo "Submitting 135M cleaned distillation matrix"
echo "run_ts=${RUN_TS}"
echo "lr=${LR} scheduler=${LR_SCHEDULER_TYPE} epochs=${EPOCHS}"
echo "loss_evals_per_epoch=${LOSS_EVALS_PER_EPOCH} full_evals_per_epoch=${EVALS_PER_EPOCH}"
echo "batch_size=${BATCH_SIZE} eval_batch_size=${EVAL_BATCH_SIZE}"
echo "correct_exec_csv=${CORRECT_EXEC_CSV}"
echo "correct_exec_and_answer_csv=${CORRECT_EXEC_AND_ANSWER_CSV}"
echo "sp2013_val_source_csv=${SP2013_VAL_SOURCE_CSV}"
echo "manifest=${MANIFEST}"
echo

submit_one "correct_exec" "params_always" "${CORRECT_EXEC_CSV}" "always" "1" "uma_tuple" "1" "0" "1" "0"
submit_one "correct_exec" "no_params" "${CORRECT_EXEC_CSV}" "none" "0" "uma_distribution" "1000" "1" "8" "1000"
submit_one "correct_exec_and_answer" "params_always" "${CORRECT_EXEC_AND_ANSWER_CSV}" "always" "1" "uma_tuple" "1" "0" "1" "0"
submit_one "correct_exec_and_answer" "no_params" "${CORRECT_EXEC_AND_ANSWER_CSV}" "none" "0" "uma_distribution" "1000" "1" "8" "1000"

echo
echo "Manifest: ${MANIFEST}"
