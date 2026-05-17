#!/bin/bash
#SBATCH --job-name=q3evalsc
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/slurm_logs/q3evalsc_%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/slurm_logs/q3evalsc_%x_%j.err

# Required env vars (passed via --export):
#   CONDITION, BASE_MODEL, ADAPTER_DIR (optional), OUTPUT_DIR, HUMAN_CSV
#   STUDENT_G, STUDENT_D, STUDENT_RT, STUDENT_ICE  (all 4 required for conditioning)
# Optional: NUM_ROLLOUTS, TEMPERATURE, ROLLOUT_BATCH_SIZE, SAVE_ROLLOUTS

set -euo pipefail
mkdir -p /scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/slurm_logs

PYBIN="${PYTHON_BIN:?PYTHON_BIN required}"
export HF_HOME="${HF_HOME:?HF_HOME required}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_PY="/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/scripts/eval_qwen3_studentcond_sp2013.py"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"

ARGS=(
  --base_model "$BASE_MODEL"
  --condition_label "$CONDITION"
  --output_dir "$OUTPUT_DIR"
  --human_csv "$HUMAN_CSV"
  --num_rollouts "$NUM_ROLLOUTS"
  --temperature "$TEMPERATURE"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --student_g "$STUDENT_G"
  --student_d "$STUDENT_D"
  --student_rt "$STUDENT_RT"
  --student_ice "$STUDENT_ICE"
)
if [[ -n "${ADAPTER_DIR:-}" ]]; then
  ARGS+=(--adapter_dir "$ADAPTER_DIR")
fi
if [[ "${SAVE_ROLLOUTS:-0}" == "1" ]]; then
  ARGS+=(--save_rollouts)
fi

echo "[$(date)] start eval cond=$CONDITION base=$BASE_MODEL adapter=${ADAPTER_DIR:-<none>}"
echo "  profile: g=$STUDENT_G d=$STUDENT_D rt=$STUDENT_RT ice=$STUDENT_ICE"
"$PYBIN" "$EVAL_PY" "${ARGS[@]}"
echo "[$(date)] done"
