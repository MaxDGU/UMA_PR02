#!/usr/bin/env bash
#SBATCH --job-name=fill4
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/%x_%j.err
# Train one missing UMA learner through the full grade 1-6 GoMath curriculum with the original fork code
# (same driver as fractionGPT_repo/slurm/run_uma_parallel.slurm), saving only the grade-6 model.
set -euo pipefail
cd "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/UMA_PR02_fork/1. Model"
PYTHONPATH=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/pylib /home/mg7411/.conda/envs/gpt-analysis/bin/python -W ignore -u -c "
from simulators import *
ps={'g':[.01,.02,.03,.04,.05,.06,.07,.08,.09,.10],'d':[.1,.3,.5,.7,.9],'c':[$CVAL],'rt_mu':[3,4,5,6],'rt_sd':[1],'ice':[0,25,50,75,100]}
S=Simulation('FILL4_c$CVAL', params_sets=ps, curriculum='full')
print('params', S.cohort.params[$SID], flush=True)
S.run(start_subjid=$SID, end_subjid=$SID, save_final_only=True)
"
echo DONE
