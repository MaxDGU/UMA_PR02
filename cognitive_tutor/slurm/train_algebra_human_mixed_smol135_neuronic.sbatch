#!/bin/bash
#SBATCH --job-name=alg_hmix_135m
#SBATCH --partition=neuronic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=results/transformer_replication/logs/alg_hmix_135m_%j.out
#SBATCH --error=results/transformer_replication/logs/alg_hmix_135m_%j.err

set -eo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/results/transformer_replication/logs"

if command -v module >/dev/null 2>&1; then
  module load anaconda3/2024.10 || true
fi
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-torch-env}"
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PARQUET_ROOT="${PARQUET_ROOT:-${ROOT_DIR}/cognitive_tutor/data/processed/algebra_2006_2007/translated_synthetic_algebra_v3_human_mixed/parquet}"
WORK_DIR="${WORK_DIR:-${SLURM_TMPDIR:-${ROOT_DIR}/results/transformer_replication/tmp_alg_hmix_${SLURM_JOB_ID:-local}}}"
DATA_CSV="${DATA_CSV:-${WORK_DIR}/singlevar_algebra_synth_v3_human_mixed_all.csv.gz}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/transformer_replication/smollm2_135m_algebra_human_mixed_neuronic}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}"

BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
VAL_FRAC="${VAL_FRAC:-0.02}"
TEST_FRAC="${TEST_FRAC:-0.02}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
BEST_BY="${BEST_BY:-val_loss}"

mkdir -p "${WORK_DIR}" "${OUT_DIR}"

echo "Materializing Parquet shards from ${PARQUET_ROOT}"
echo "Temporary CSV: ${DATA_CSV}"
python - <<PY
from pathlib import Path
import pandas as pd

parquet_root = Path("${PARQUET_ROOT}")
out_csv = Path("${DATA_CSV}")
parts = sorted(parquet_root.glob("*/*.parquet"))
if not parts:
    raise FileNotFoundError(f"No parquet shards found under {parquet_root}")
frames = [pd.read_parquet(path) for path in parts]
df = pd.concat(frames, ignore_index=True)
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"materialized_rows={len(df)}")
print(f"materialized_csv={out_csv}")
PY

echo "Training ${MODEL_NAME} on algebra human-mixed synthetic data"
echo "output_dir=${OUT_DIR}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
  "${ROOT_DIR}/results/transformer_replication/train_transformer_hf.py" \
    --model_name "${MODEL_NAME}" \
    --data_csv "${DATA_CSV}" \
    --output_dir "${OUT_DIR}" \
    --prompt_col instruction_nl \
    --response_col response_nl \
    --problem_col prob \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_length "${MAX_LENGTH}" \
    --val_frac "${VAL_FRAC}" \
    --test_frac "${TEST_FRAC}" \
    --split_by_problem \
    --num_workers "${NUM_WORKERS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --amp_dtype "${AMP_DTYPE}" \
    --evals_per_epoch "${EVALS_PER_EPOCH}" \
    --best_by "${BEST_BY}" \
    --save_best_only \
    --no-eval_sp2013
