#!/bin/bash
#SBATCH --job-name=sft_rw
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/logs/sft_rw_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/logs/sft_rw_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam

module purge
module load anaconda3/2024.2
conda activate /scratch/gpfs/GRIFFITHS/mg7411/conda_envs/nnsight-env

cd /scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions
mkdir -p logs

echo "============================================================"
echo "Importance-weighted SFT"
echo "Start: $(date)"
echo "============================================================"

python3 -u finetune/sft_reweight.py \
    --sft_lr 1e-5 \
    --sft_epochs 20 \
    --batch_size 64 \
    --smooth_alpha 0.05 \
    --kl_beta 0.5 \
    --eval_every 5 \
    --tag sft_reweight_v1

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
