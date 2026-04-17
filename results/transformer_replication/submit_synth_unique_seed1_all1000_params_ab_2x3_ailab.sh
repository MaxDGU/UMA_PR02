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
LAUNCHER="${BASE_DIR}/submit_synth_unique_seed1_all1000_2x3_ailab.sh"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
CONDITIONS="${CONDITIONS:-params_always,params_dropout50_noprefix}"
MODES="${MODES:-pretrained,scratch}"
MODELS="${MODELS:-135m,360m,1p7b}"
PARTITION="${PARTITION:-ailab}"
OUT_BASE_DIR="${OUT_BASE_DIR:-${BASE_DIR}}"
DRY_RUN="${DRY_RUN:-0}"
BASE_DATA_CSV="${BASE_DATA_CSV:-${BASE_DIR}/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz}"

EPOCHS="${EPOCHS:-2}"
BEST_BY="${BEST_BY:-sp2013_acc}"
SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID:-0}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH:-0}"
EVAL_SP2013="${EVAL_SP2013:-1}"
EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER:-1}"
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-1}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD:-1}"
NAME_SUFFIX_BASE="${NAME_SUFFIX_BASE:-sp2013noparam_tracev7_bestsp}"

prompt_mode_for_condition() {
  case "$1" in
    params_always) echo "always" ;;
    params_dropout50_noprefix) echo "dropout" ;;
    *) echo "unknown condition: $1" >&2; return 1 ;;
  esac
}

run_label_for_condition() {
  case "$1" in
    params_always) echo "synth_unique_seed1_all1000_mincols_paramsalways" ;;
    params_dropout50_noprefix) echo "synth_unique_seed1_all1000_mincols_paramsdrop50noprefix" ;;
    *) echo "unknown condition: $1" >&2; return 1 ;;
  esac
}

dropout_prob_for_condition() {
  case "$1" in
    params_always) echo "0.0" ;;
    params_dropout50_noprefix) echo "0.5" ;;
    *) echo "unknown condition: $1" >&2; return 1 ;;
  esac
}

dropout_seed_for_condition() {
  case "$1" in
    params_always) echo "0" ;;
    params_dropout50_noprefix) echo "0" ;;
    *) echo "unknown condition: $1" >&2; return 1 ;;
  esac
}

IFS=',' read -r -a CONDITION_LIST <<< "${CONDITIONS}"

echo "Submitting synth_unique_seed1_all1000 params-conditioning A/B grid"
echo "baseline=no_params existing March 7, 2026 runs"
echo "base_data_csv=${BASE_DATA_CSV}"
echo "run_ts=${RUN_TS}"
echo "conditions=${CONDITIONS}"
echo "modes=${MODES}"
echo "models=${MODELS}"
echo "epochs=${EPOCHS}"
echo "best_by=${BEST_BY}"
echo "sp2013_use_param_grid=${SP2013_USE_PARAM_GRID}"
echo "loss_evals_per_epoch=${LOSS_EVALS_PER_EPOCH}"
echo

if [[ ! -f "${BASE_DATA_CSV}" && "${DRY_RUN}" != "1" ]]; then
  echo "missing base dataset: ${BASE_DATA_CSV}" >&2
  exit 1
fi

for condition in "${CONDITION_LIST[@]}"; do
  prompt_mode="$(prompt_mode_for_condition "${condition}")"
  run_label="$(run_label_for_condition "${condition}")"
  dropout_prob="$(dropout_prob_for_condition "${condition}")"
  dropout_seed="$(dropout_seed_for_condition "${condition}")"
  manifest="${OUT_BASE_DIR}/jobs_synth_unique_seed1_all1000_${condition}_2x3_${RUN_TS}.tsv"

  echo "condition=${condition}"
  echo "  data_csv=${BASE_DATA_CSV}"
  echo "  train_prompt_student_mode=${prompt_mode}"
  echo "  train_prompt_student_dropout_prob=${dropout_prob}"
  echo "  train_prompt_student_dropout_seed=${dropout_seed}"
  echo "  manifest=${manifest}"

  env \
    DATA_CSV="${BASE_DATA_CSV}" \
    RUN_TS="${RUN_TS}" \
    RUN_LABEL="${run_label}" \
    NAME_SUFFIX="${NAME_SUFFIX_BASE}" \
    MANIFEST="${manifest}" \
    MODES="${MODES}" \
    MODELS="${MODELS}" \
    PARTITION="${PARTITION}" \
    OUT_BASE_DIR="${OUT_BASE_DIR}" \
    DRY_RUN="${DRY_RUN}" \
    EPOCHS="${EPOCHS}" \
    BEST_BY="${BEST_BY}" \
    SP2013_USE_PARAM_GRID="${SP2013_USE_PARAM_GRID}" \
    EVALS_PER_EPOCH="${EVALS_PER_EPOCH}" \
    LOSS_EVALS_PER_EPOCH="${LOSS_EVALS_PER_EPOCH}" \
    EVAL_SP2013="${EVAL_SP2013}" \
    EVAL_ID_FINAL_ANSWER="${EVAL_ID_FINAL_ANSWER}" \
    SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS}" \
    SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT}" \
    SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT}" \
    VERIFY_SAVED_CHECKPOINT_LOAD="${VERIFY_SAVED_CHECKPOINT_LOAD}" \
    TRAIN_PROMPT_STUDENT_MODE="${prompt_mode}" \
    TRAIN_PROMPT_STUDENT_DROPOUT_PROB="${dropout_prob}" \
    TRAIN_PROMPT_STUDENT_DROPOUT_SEED="${dropout_seed}" \
    bash "${LAUNCHER}"

  echo
done
