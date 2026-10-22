#!/usr/bin/env bash
#SBATCH --job-name=c1000_spaced
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/spaced_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/spaced_%A_%a.err
# Re-roll every arm-C and arm-A-control checkpoint with SPACED problem prompts (training surface), 256 samples x 16 problems.
set -euo pipefail
C=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
read TAG CK < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $C/spaced_list.txt)
mkdir -p $C/results_spaced; OUT=$C/results_spaced/$TAG.csv; [[ -f $OUT ]] && exit 0
/home/mg7411/.conda/envs/gpt-analysis/bin/python $C/eval_spaced.py --ckpt "$CK" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$OUT" --n-samples 256
