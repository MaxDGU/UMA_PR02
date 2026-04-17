#!/bin/bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-${ROOT_DIR}/BBT/configs/default_pipeline.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUBPIPELINES="${SUBPIPELINES:-uma_traces,translation,distill_train,distill_eval,human_finetune,human_eval}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/results/bbt/slurm_run_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${ROOT_DIR}/results/bbt/logs"

order=(uma_traces translation distill_train distill_eval human_finetune human_eval)

contains() {
  local target="$1"
  local csv="$2"
  IFS=',' read -r -a parts <<< "${csv}"
  for p in "${parts[@]}"; do
    if [[ "${p}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

manifest_for() {
  local sp="$1"
  case "${sp}" in
    uma_traces) echo "${RUN_DIR}/01_uma_traces/manifest.json" ;;
    translation) echo "${RUN_DIR}/02_translation/manifest.json" ;;
    distill_train) echo "${RUN_DIR}/03_distill_train/manifest.json" ;;
    distill_eval) echo "${RUN_DIR}/04_distill_eval/manifest.json" ;;
    human_finetune) echo "${RUN_DIR}/05_human_finetune/manifest.json" ;;
    human_eval) echo "${RUN_DIR}/06_human_eval/manifest.json" ;;
    *) echo "" ;;
  esac
}

prev_job=""
prev_selected=""

declare -A job_ids

for sp in "${order[@]}"; do
  if ! contains "${sp}" "${SUBPIPELINES}"; then
    continue
  fi

  upstream_manifest=""
  if [[ -n "${prev_selected}" ]]; then
    upstream_manifest="$(manifest_for "${prev_selected}")"
  fi

  export_vars="ALL,SUBPIPELINE=${sp},CONFIG=${CONFIG},RUN_DIR=${RUN_DIR},PYTHON_BIN=${PYTHON_BIN},DRY_RUN=${DRY_RUN}"
  if [[ -n "${upstream_manifest}" ]]; then
    export_vars+=" ,UPSTREAM_MANIFEST=${upstream_manifest}"
  fi
  export_vars="${export_vars// ,/,}"

  sbatch_cmd=(sbatch --export="${export_vars}")
  if [[ -n "${prev_job}" ]]; then
    sbatch_cmd+=(--dependency="afterok:${prev_job}")
  fi
  sbatch_cmd+=("${ROOT_DIR}/BBT/slurm/run_subpipeline.sbatch")

  out="$(${sbatch_cmd[@]})"
  job_id="$(awk '{print $4}' <<< "${out}")"
  echo "${out}"
  echo "submitted ${sp} -> job ${job_id}"

  job_ids["${sp}"]="${job_id}"
  prev_job="${job_id}"
  prev_selected="${sp}"
done

echo "RUN_DIR=${RUN_DIR}"
for sp in "${order[@]}"; do
  if [[ -n "${job_ids[${sp}]:-}" ]]; then
    echo "${sp}: ${job_ids[${sp}]}"
  fi
done
