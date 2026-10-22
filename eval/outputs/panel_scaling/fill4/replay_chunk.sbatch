#!/usr/bin/env bash
#SBATCH --job-name=fill4_chunk
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/chunk_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/chunk_%A_%a.err
# Same replay as replay_one.sbatch (c=4 models), split over 8 problem chunks for learners 97 and 98.
set -euo pipefail
R=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; G=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4
SIDS=(97 98); SID=${SIDS[$((SLURM_ARRAY_TASK_ID/8))]}; K=$((SLURM_ARRAY_TASK_ID%8))
cd "$R/results/UMA_replication"
/home/mg7411/.conda/envs/gpt-analysis/bin/python run_saved_models_on_problem_set.py --models-dir $G/models_c4 --start-subjid $SID --end-subjid $SID \
  --problem-csv $G/chunks/part$K.csv --out-csv $G/raw_c4_parts/subj_${SID}_part$K.csv --overwrite --progress-every 1
echo "DONE sid=$SID part=$K"
