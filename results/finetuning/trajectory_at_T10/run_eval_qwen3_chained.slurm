#!/bin/bash
#SBATCH --job-name=q3eval
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/q3eval_%x_%j.out
#SBATCH --error=slurm_logs/q3eval_%x_%j.err

# Required env vars (passed via --export):
#   CONDITION       short label (e.g. q3_8b_full)
#   BASE_MODEL      HF id or local path of base
#   ADAPTER_DIR     LoRA adapter dir, or empty string for base-only
#   OUTPUT_DIR      where per-problem CSVs/JSON go
#   NUM_ROLLOUTS    optional override (default 256)
# Optional:
#   PYTHON_BIN, HF_HOME

set -euo pipefail
# Eval driver is inlined next to this sbatch (results/finetuning/trajectory_at_T10/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p slurm_logs

PYBIN="${PYTHON_BIN:?PYTHON_BIN required}"
export HF_HOME="${HF_HOME:?HF_HOME required}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

NUM_ROLLOUTS="${NUM_ROLLOUTS:-256}"
HUMAN_CSV="${HUMAN_CSV:?HUMAN_CSV required (path to siegler_fraction_human.csv)}"
TEMPERATURE="${TEMPERATURE:-}"
TEMPERATURES="${TEMPERATURES:-}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"

ARGS=(
  --base_model "$BASE_MODEL"
  --condition_label "$CONDITION"
  --output_dir "$OUTPUT_DIR"
  --human_csv "$HUMAN_CSV"
  --num_rollouts "$NUM_ROLLOUTS"
)
if [[ -n "$TEMPERATURES" ]]; then
  # sbatch --export splits on commas, so callers should pass TEMPERATURES with ';' as separator.
  TEMPERATURES="${TEMPERATURES//;/,}"
  ARGS+=(--temperatures "$TEMPERATURES")
elif [[ -n "$TEMPERATURE" ]]; then
  ARGS+=(--temperature "$TEMPERATURE")
else
  ARGS+=(--temperature 0.7)
fi
if [[ -n "$ROLLOUT_BATCH_SIZE" ]]; then
  ARGS+=(--rollout_batch_size "$ROLLOUT_BATCH_SIZE")
fi
if [[ "${SAVE_ROLLOUTS:-0}" == "1" ]]; then
  ARGS+=(--save_rollouts)
fi
if [[ -n "${ADAPTER_DIR:-}" ]]; then
  ARGS+=(--adapter_dir "$ADAPTER_DIR")
fi

echo "[$(date)] starting eval condition=$CONDITION base=$BASE_MODEL adapter=${ADAPTER_DIR:-<none>} rollouts=$NUM_ROLLOUTS"
"$PYBIN" "$SCRIPT_DIR/eval_qwen3_chained_sp2013.py" "${ARGS[@]}"
echo "[$(date)] done"
