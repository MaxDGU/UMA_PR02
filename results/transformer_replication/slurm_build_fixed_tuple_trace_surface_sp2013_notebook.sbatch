#!/bin/bash
# Build and execute the fixed-tuple trace-surface SP2013 comparison notebook.

#SBATCH --job-name=ft472_nb
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/transformer_replication/logs/ft472_nb_%j.out
#SBATCH --error=results/transformer_replication/logs/ft472_nb_%j.err

set -euo pipefail
export PS1="${PS1:-}"

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/results/transformer_replication/logs"

module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PATH="/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40"
conda activate "${ENV_PATH}"

: "${DEFAULT_ROLLOUTS:?Set DEFAULT_ROLLOUTS to the humanlike rollout CSV.}"
: "${SURFACE_ROLLOUTS:?Set SURFACE_ROLLOUTS to the surfaced rollout CSV.}"
: "${UMA_TARGET_CSV:?Set UMA_TARGET_CSV to the merged same-tuple UMA target CSV.}"

python "${ROOT_DIR}/results/transformer_replication/build_fixed_tuple_trace_surface_sp2013_notebook.py" \
  --default_rollouts "${DEFAULT_ROLLOUTS}" \
  --surface_rollouts "${SURFACE_ROLLOUTS}" \
  --uma_target_csv "${UMA_TARGET_CSV}"
