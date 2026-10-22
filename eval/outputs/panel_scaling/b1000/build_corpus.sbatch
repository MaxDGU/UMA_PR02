#!/usr/bin/env bash
#SBATCH --job-name=b1000_build
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/logs/build_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/logs/build_%j.err
# Translate the 4 new learners (c=$CVAL) with the paper recipe, then build B1000 = arm B's exact 517,920 rows + 520 rows per new learner.
set -euo pipefail
R=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; PS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling; G=$PS/fill4; K=$PS/b1000
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
[[ $(ls $G/raw_c$CVAL/*.csv.done | wc -l) -eq 4 ]] || { echo "not all 4 replays done"; exit 1; }
cd "$R"
"$PY" results/transformer_replication/translate_uma_traces_to_nlp.py --input-csv "$G/raw_c$CVAL/subj_*.csv" \
  --no-include-default-uma-traces --no-include-sp2013-seeds --output-csv $K/fill4_c${CVAL}_nlp.csv.gz \
  --minimal-columns --student-prompt-mode none --reasoning-mode trace_or_child --chunksize 200000
"$PY" - <<PYEOF
import pandas as pd, numpy as np
K='$K'; PS='$PS'
b=pd.read_csv(f'{PS}/arms/B_all996_sub520k.csv.gz'); n=pd.read_csv(f'{K}/fill4_c${CVAL}_nlp.csv.gz')
assert list(n.columns)==list(b.columns), (set(n.columns)^set(b.columns))
sp=set(pd.read_csv('/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/UMA_replication/sp2013.csv').prob.astype(str))
f=n.reasoning_quality_flags.fillna('').astype(str)
print(f'new learners: rows {len(n):,} learners {sorted(n.subjid.unique())} acc {100*n.is_correct.astype(str).isin(["1","True","true","1.0"]).mean():.1f}% arith_mismatch {100*f.str.contains("arith_claim_mismatch").mean():.2f}% SP2013 leakage {len(sp&set(n.prob.astype(str)))}')
pick=[g.loc[np.random.RandomState(int(s)).choice(g.index,520,replace=False)] for s,g in n.groupby('subjid')]
out=pd.concat([b]+pick,ignore_index=True).sample(frac=1,random_state=0).reset_index(drop=True)
assert out.subjid.nunique()==1000 and len(out)==len(b)+2080
out.to_csv(f'{K}/B1000_sub520k.csv.gz',index=False,compression='gzip')
print(f'B1000 corpus: rows {len(out):,} learners {out.subjid.nunique()} | arm B rows {len(b):,} kept unchanged + {4*520} new')
PYEOF
echo BUILD_DONE
