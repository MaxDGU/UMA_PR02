#!/bin/bash
#SBATCH --job-name=bbt_subpipe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=results/bbt/logs/bbt_%x_%j.out
#SBATCH --error=results/bbt/logs/bbt_%x_%j.err

set -euo pipefail

ROOT_DIR="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/results/bbt/logs"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${ROOT_DIR}/BBT/configs/default_pipeline.json}"
RUN_DIR="${RUN_DIR:-}"
SUBPIPELINE="${SUBPIPELINE:-}"
UPSTREAM_MANIFEST="${UPSTREAM_MANIFEST:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SUBPIPELINE}" ]]; then
  echo "SUBPIPELINE is required"
  exit 1
fi
if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR is required"
  exit 1
fi

CMD=("${PYTHON_BIN}" "${ROOT_DIR}/BBT/src/${SUBPIPELINE}.py" "--config" "${CONFIG}" "--run-dir" "${RUN_DIR}" "--python" "${PYTHON_BIN}")
if [[ -n "${UPSTREAM_MANIFEST}" ]]; then
  CMD+=("--upstream-manifest" "${UPSTREAM_MANIFEST}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

echo "Running subpipeline=${SUBPIPELINE} run_dir=${RUN_DIR}"
"${CMD[@]}"
