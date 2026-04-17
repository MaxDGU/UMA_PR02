#!/bin/bash
# Merge, translate, filter, and preflight the fixed-tuple trace-surface datasets.

#SBATCH --job-name=ft472_build
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/transformer_replication/logs/ft472_build_%j.out
#SBATCH --error=results/transformer_replication/logs/ft472_build_%j.err

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

python "${ROOT_DIR}/results/transformer_replication/build_fixed_tuple_trace_surface_datasets.py"
