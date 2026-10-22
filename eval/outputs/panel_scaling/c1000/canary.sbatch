#!/usr/bin/env bash
#SBATCH --job-name=c1000_canary
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/canary_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/canary_%j.err
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
/home/mg7411/.conda/envs/gpt-analysis/bin/python /scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/canary.py "$CKPT"
