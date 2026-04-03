#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_CSV="${DATA_CSV:-${ROOT_DIR}/results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"
SP2013_CSV="${SP2013_CSV:-${ROOT_DIR}/results/UMA_replication/sp2013.csv}"
OUT_BASE_DIR="${OUT_BASE_DIR:-${ROOT_DIR}/results/transformer_replication}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/transformer_replication/logs}"
PARTITION="${PARTITION:-ailab}"
CONDA_ENV="${CONDA_ENV:-torch-env}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-synth_unique_seed1_all1000_mincols}"
NAME_SUFFIX="${NAME_SUFFIX:-sp2013noparam_tracev6_bestsp}"
MODES="${MODES:-pretrained,scratch}"
MODELS="${MODELS:-135m,360m,1p7b}"
MANIFEST="${MANIFEST:-${OUT_BASE_DIR}/jobs_synth_unique_seed1_all1000_2x3_${RUN_TS}.tsv}"
DRY_RUN="${DRY_RUN:-0}"

EPOCHS="${EPOCHS:-1}"
BEST_BY="${BEST_BY:-sp2013_acc}"
RESUME_FROM="${RESUME_FROM:-none}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-0}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
EVAL_SP2013="${EVAL_SP2013:-1}"
EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-1}"
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-1}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-1}"
MOVE_SP2013_ROWS_TO_VAL="${MOVE_SP2013_ROWS_TO_VAL:-1}"
TRAIN_PROMPT_STUDENT_MODE="${TRAIN_PROMPT_STUDENT_MODE:-none}"
TRAIN_PROMPT_STUDENT_DROPOUT_PROB="${TRAIN_PROMPT_STUDENT_DROPOUT_PROB:-0.5}"
TRAIN_PROMPT_STUDENT_DROPOUT_SEED="${TRAIN_PROMPT_STUDENT_DROPOUT_SEED:-0}"

TIME_135M="${TIME_135M:-24:00:00}"
TIME_360M="${TIME_360M:-24:00:00}"
TIME_1P7B="${TIME_1P7B:-16:00:00}"

GPUS_135M="${GPUS_135M:-1}"
GPUS_360M="${GPUS_360M:-1}"
GPUS_1P7B="${GPUS_1P7B:-2}"

CHAIN_135M="${CHAIN_135M:-1}"
CHAIN_360M="${CHAIN_360M:-1}"
CHAIN_1P7B="${CHAIN_1P7B:-2}"

CPUS_135M="${CPUS_135M:-8}"
CPUS_360M="${CPUS_360M:-8}"
CPUS_1P7B="${CPUS_1P7B:-16}"

BATCH_135M="${BATCH_135M:-512}"
BATCH_360M="${BATCH_360M:-256}"
BATCH_1P7B="${BATCH_1P7B:-64}"

EVAL_BATCH_135M="${EVAL_BATCH_135M:-${BATCH_135M}}"
EVAL_BATCH_360M="${EVAL_BATCH_360M:-${BATCH_360M}}"
EVAL_BATCH_1P7B="${EVAL_BATCH_1P7B:-${BATCH_1P7B}}"

GC_135M="${GC_135M:-0}"
GC_360M="${GC_360M:-0}"
GC_1P7B="${GC_1P7B:-1}"

ROWS_TOTAL="${ROWS_TOTAL:-20736000}"
VAL_FRAC_PCT="${VAL_FRAC_PCT:-5}"
TEST_FRAC_PCT="${TEST_FRAC_PCT:-5}"

SAVE_LAST_EVERY_135M="${SAVE_LAST_EVERY_135M:-}"
SAVE_LAST_EVERY_360M="${SAVE_LAST_EVERY_360M:-}"
SAVE_LAST_EVERY_1P7B="${SAVE_LAST_EVERY_1P7B:-}"

mkdir -p "${LOG_DIR}"

IFS=',' read -r -a MODE_LIST <<< "${MODES}"
IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"

script_for_model() {
  case "$1" in
    135m) echo "results/transformer_replication/slurm_train_fraction2400_smol135_ailab.sbatch" ;;
    360m) echo "results/transformer_replication/slurm_train_fraction2400_smol360_ailab.sbatch" ;;
    1p7b) echo "results/transformer_replication/slurm_train_fraction2400_smol1p7b_ailab.sbatch" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

time_for_model() {
  case "$1" in
    135m) echo "${TIME_135M}" ;;
    360m) echo "${TIME_360M}" ;;
    1p7b) echo "${TIME_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

gpus_for_model() {
  case "$1" in
    135m) echo "${GPUS_135M}" ;;
    360m) echo "${GPUS_360M}" ;;
    1p7b) echo "${GPUS_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

chain_for_model() {
  case "$1" in
    135m) echo "${CHAIN_135M}" ;;
    360m) echo "${CHAIN_360M}" ;;
    1p7b) echo "${CHAIN_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

cpus_for_model() {
  case "$1" in
    135m) echo "${CPUS_135M}" ;;
    360m) echo "${CPUS_360M}" ;;
    1p7b) echo "${CPUS_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

batch_for_model() {
  case "$1" in
    135m) echo "${BATCH_135M}" ;;
    360m) echo "${BATCH_360M}" ;;
    1p7b) echo "${BATCH_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

eval_batch_for_model() {
  case "$1" in
    135m) echo "${EVAL_BATCH_135M}" ;;
    360m) echo "${EVAL_BATCH_360M}" ;;
    1p7b) echo "${EVAL_BATCH_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

gc_for_model() {
  case "$1" in
    135m) echo "${GC_135M}" ;;
    360m) echo "${GC_360M}" ;;
    1p7b) echo "${GC_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

save_last_every_for_model() {
  case "$1" in
    135m) echo "${SAVE_LAST_EVERY_135M}" ;;
    360m) echo "${SAVE_LAST_EVERY_360M}" ;;
    1p7b) echo "${SAVE_LAST_EVERY_1P7B}" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

default_save_last_every() {
  local batch_size="$1"
  local gpus="$2"
  local train_rows=$((ROWS_TOTAL - (ROWS_TOTAL * VAL_FRAC_PCT / 100) - (ROWS_TOTAL * TEST_FRAC_PCT / 100)))
  local global_batch=$((batch_size * gpus))
  local updates=$(((train_rows + global_batch - 1) / global_batch))
  echo $(((updates + 9) / 10))
}

init_from_scratch_for_mode() {
  case "$1" in
    pretrained) echo "0" ;;
    scratch) echo "1" ;;
    *) echo "unknown mode: $1" >&2; return 1 ;;
  esac
}

job_suffix_for() {
  local model="$1"
  local mode="$2"
  local model_tag=""
  local mode_tag=""
  case "${model}" in
    135m) model_tag="135" ;;
    360m) model_tag="360" ;;
    1p7b) model_tag="1p7b" ;;
    *) echo "unknown model: ${model}" >&2; return 1 ;;
  esac
  case "${mode}" in
    pretrained) mode_tag="pt" ;;
    scratch) mode_tag="scr" ;;
    *) echo "unknown mode: ${mode}" >&2; return 1 ;;
  esac
  echo "s1000_${model_tag}_${mode_tag}"
}

printf "mode\tmodel\tstage\tjob_id\tjob_name\tdependency\ttime_limit\tgpus\tbatch_size\teval_batch_size\tsave_last_every\tout_dir\tlog_out\tlog_err\tscript\tdata_csv\ttrain_prompt_student_mode\ttrain_prompt_student_dropout_prob\ttrain_prompt_student_dropout_seed\n" > "${MANIFEST}"

echo "Submitting synth_unique_seed1_all1000 2x3 Smol grid"
echo "data_csv=${DATA_CSV}"
echo "run_ts=${RUN_TS}"
echo "modes=${MODES}"
echo "models=${MODELS}"
echo "best_by=${BEST_BY}"
echo "resume_from=${RESUME_FROM}"
echo "sp2013_use_param_grid=${SP2013_USE_PARAM_GRID}"
echo "move_sp2013_rows_to_val=${MOVE_SP2013_ROWS_TO_VAL}"
echo "train_prompt_student_mode=${TRAIN_PROMPT_STUDENT_MODE}"
echo "train_prompt_student_dropout_prob=${TRAIN_PROMPT_STUDENT_DROPOUT_PROB}"
echo "train_prompt_student_dropout_seed=${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}"
echo "manifest=${MANIFEST}"
echo

submitted=0

for mode in "${MODE_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    script_path="$(script_for_model "${model}")"
    time_limit="$(time_for_model "${model}")"
    gpus_per_node="$(gpus_for_model "${model}")"
    chain_count="$(chain_for_model "${model}")"
    cpus_per_task="$(cpus_for_model "${model}")"
    batch_size="$(batch_for_model "${model}")"
    eval_batch_size="$(eval_batch_for_model "${model}")"
    gc_flag="$(gc_for_model "${model}")"
    save_last_every="$(save_last_every_for_model "${model}")"
    if [[ -z "${save_last_every}" ]]; then
      save_last_every="$(default_save_last_every "${batch_size}" "${gpus_per_node}")"
    fi
    init_from_scratch="$(init_from_scratch_for_mode "${mode}")"
    job_name="$(job_suffix_for "${model}" "${mode}")"
    out_dir="${OUT_BASE_DIR}/smol2_${model}_${RUN_LABEL}_${mode}_${NAME_SUFFIX}_${RUN_TS}"
    prev_job_id=""

    for stage in $(seq 1 "${chain_count}"); do
      stage_job_name="${job_name}"
      if [[ "${chain_count}" -gt 1 ]]; then
        stage_job_name="${job_name}_p${stage}"
      fi
      log_out="${LOG_DIR}/${stage_job_name}_${RUN_TS}_%j.out"
      log_err="${LOG_DIR}/${stage_job_name}_${RUN_TS}_%j.err"
      stage_resume_from="${RESUME_FROM}"
      stage_init_from_scratch="${init_from_scratch}"
      require_resume="0"
      dependency=""
      if [[ "${stage}" -gt 1 ]]; then
        stage_resume_from="auto"
        dependency="afterany:${prev_job_id}"
      fi
      if [[ "${stage_resume_from}" != "none" ]]; then
        stage_init_from_scratch="0"
        require_resume="1"
      fi
      export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},DATA_CSV=${DATA_CSV},SP2013_CSV=${SP2013_CSV},OUT_DIR=${out_dir},EPOCHS=${EPOCHS},BEST_BY=${BEST_BY},RESUME_FROM=${stage_resume_from},INIT_FROM_SCRATCH=${stage_init_from_scratch},SP2013_USE_PARAM_GRID=${SP2013_USE_PARAM_GRID},EVALS_PER_EPOCH=${EVALS_PER_EPOCH},EVAL_SP2013=${EVAL_SP2013},EVAL_ID_FINAL_ANSWER=${EVAL_ID_FINAL_ANSWER},SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS},SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT},SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT},VERIFY_SAVED_CHECKPOINT_LOAD=${VERIFY_SAVED_CHECKPOINT_LOAD},MOVE_SP2013_ROWS_TO_VAL=${MOVE_SP2013_ROWS_TO_VAL},BATCH_SIZE=${batch_size},EVAL_BATCH_SIZE=${eval_batch_size},GC=${gc_flag},SAVE_LAST_EVERY_UPDATES=${save_last_every},GPUS_PER_NODE=${gpus_per_node},REQUIRE_RESUME=${require_resume},TRAIN_PROMPT_STUDENT_MODE=${TRAIN_PROMPT_STUDENT_MODE},TRAIN_PROMPT_STUDENT_DROPOUT_PROB=${TRAIN_PROMPT_STUDENT_DROPOUT_PROB},TRAIN_PROMPT_STUDENT_DROPOUT_SEED=${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}"

      if [[ "${DRY_RUN}" == "1" ]]; then
        dep_str=""
        if [[ -n "${dependency}" ]]; then
          dep_str=" --dependency=${dependency}"
        fi
        echo "[dry-run] sbatch --parsable --partition=${PARTITION} --time=${time_limit} --gpus-per-node=${gpus_per_node} --cpus-per-task=${cpus_per_task}${dep_str} --job-name=${stage_job_name} --output=${log_out} --error=${log_err} --export=${export_vars} ${script_path}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${mode}" "${model}" "${stage}" "DRY_RUN" "${stage_job_name}" "${dependency:-none}" "${time_limit}" "${gpus_per_node}" "${batch_size}" "${eval_batch_size}" "${save_last_every}" "${out_dir}" "${log_out}" "${log_err}" "${script_path}" "${DATA_CSV}" "${TRAIN_PROMPT_STUDENT_MODE}" "${TRAIN_PROMPT_STUDENT_DROPOUT_PROB}" "${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}" >> "${MANIFEST}"
        prev_job_id="DRY_RUN"
        continue
      fi

      submit_args=(
        --parsable
        --partition="${PARTITION}"
        --time="${time_limit}"
        --gpus-per-node="${gpus_per_node}"
        --cpus-per-task="${cpus_per_task}"
        --job-name="${stage_job_name}"
        --output="${log_out}"
        --error="${log_err}"
        --export="${export_vars}"
      )
      if [[ -n "${dependency}" ]]; then
        submit_args+=(--dependency="${dependency}")
      fi

      submit_out="$(sbatch "${submit_args[@]}" "${script_path}")"
      job_id="${submit_out%%;*}"
      prev_job_id="${job_id}"

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${mode}" "${model}" "${stage}" "${job_id}" "${stage_job_name}" "${dependency:-none}" "${time_limit}" "${gpus_per_node}" "${batch_size}" "${eval_batch_size}" "${save_last_every}" "${out_dir}" "${log_out}" "${log_err}" "${script_path}" "${DATA_CSV}" "${TRAIN_PROMPT_STUDENT_MODE}" "${TRAIN_PROMPT_STUDENT_DROPOUT_PROB}" "${TRAIN_PROMPT_STUDENT_DROPOUT_SEED}" >> "${MANIFEST}"

      echo "submitted ${job_id}: model=${model} mode=${mode} stage=${stage}/${chain_count} time=${time_limit} gpus=${gpus_per_node} batch=${batch_size} eval_batch=${eval_batch_size} save_last_every=${save_last_every}"
      if [[ -n "${dependency}" ]]; then
        echo "  dependency=${dependency}"
      fi
      echo "  out_dir=${out_dir}"
      submitted=$((submitted + 1))
    done
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
