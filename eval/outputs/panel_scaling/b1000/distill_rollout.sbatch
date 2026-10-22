#!/usr/bin/env bash
#SBATCH --job-name=b1000_droll
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/logs/%x_%j.err
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
TR=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication
/home/mg7411/.conda/envs/gpt-analysis/bin/python /scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/eval_seed_ckpt.py \
  --ckpt $TR/qwen3_4b_B1000_sub_v10_e1_v10regen_lora/best --base-model Qwen/Qwen3-4B-Base --domain fraction \
  --out-csv /scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/results/B1000_sub_distill.csv --n-samples 256
/home/mg7411/.conda/envs/gpt-analysis/bin/python /scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/distill_nll.py
