#!/usr/bin/env bash
#SBATCH --job-name=xlate996
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/translate_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/translate_%j.err
# Paper translation recipe: trace_or_child narration, no row filter, no student prefix. Same flags as the 98/25 regen arms.
set -euo pipefail
R=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft
PS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
N=$(wc -l < $PS/subjids_all996.txt); D=$(ls $PS/raw/*.csv.done | wc -l)
echo "learners done: $D / $N"; [[ "$D" -eq "$N" ]] || { echo "NOT ALL LEARNERS DONE"; exit 1; }
cd "$R"
"$PY" results/transformer_replication/translate_uma_traces_to_nlp.py \
  --input-csv "$PS/raw/subj_*.csv" --no-include-default-uma-traces --no-include-sp2013-seeds \
  --output-csv $PS/uma_frac_all996_v10regen_nlp.csv.gz --minimal-columns --student-prompt-mode none \
  --reasoning-mode trace_or_child --chunksize 200000
"$PY" - <<'PYEOF'
import pandas as pd
PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'
sp=set(pd.read_csv('/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/sp2013.csv').prob.astype(str))
n=0; subj=set(); probs=set(); mism=0; corr=0
for ch in pd.read_csv(f'{PS}/uma_frac_all996_v10regen_nlp.csv.gz',chunksize=500000,usecols=['subjid','prob','is_correct','reasoning_quality_flags']):
    n+=len(ch); subj|=set(ch.subjid); probs|=set(ch.prob.astype(str)); mism+=ch.reasoning_quality_flags.fillna('').astype(str).str.contains('arith_claim_mismatch').sum()
    corr+=ch.is_correct.astype(str).isin(['1','True','true','1.0']).sum()
print(f'nlp rows {n:,} learners {len(subj)} probs {len(probs)} acc {100*corr/n:.1f}% arith_claim_mismatch {100*mism/n:.2f}% SP2013 leakage {len(sp&probs)}/16')
PYEOF
echo TRANSLATE_DONE
