#!/bin/bash
# Evaluate one saved transformer checkpoint on SP2013 with a fixed student tuple.

#SBATCH --job-name=ft472_eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/transformer_replication/logs/ft472_eval_%j.out
#SBATCH --error=results/transformer_replication/logs/ft472_eval_%j.err

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

: "${RUN_DIR:?Set RUN_DIR to the training output directory.}"
: "${TARGET_CSV:?Set TARGET_CSV to the merged same-tuple UMA rollout CSV.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for evaluator outputs.}"

if [[ -z "${MAX_NEW_TOKENS:-}" ]]; then
  : "${CONTEXT_JSON:?Set CONTEXT_JSON when MAX_NEW_TOKENS is not provided.}"
  MAX_NEW_TOKENS="$(
    python - <<'PY'
import json
import os

with open(os.environ["CONTEXT_JSON"], "r", encoding="utf-8") as f:
    payload = json.load(f)
print(int(payload["context"]["shared_settings"]["max_new_tokens"]))
PY
  )"
fi

python "${ROOT_DIR}/results/transformer_replication/eval_checkpoint_sp2013_fixed_tuple.py" \
  --run_dir "${RUN_DIR}" \
  --target_csv "${TARGET_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --g "${G:-0.05}" \
  --d "${D:-0.7}" \
  --rt_mu "${RT_MU:-5}" \
  --ice "${ICE:-50}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --num_rollouts "${NUM_ROLLOUTS:-1000}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE:-64}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.95}" \
  --top_k "${TOP_K:-50}" \
  --sample_seed "${SAMPLE_SEED:-123}" \
  --progress_every "${PROGRESS_EVERY:-1000}"
