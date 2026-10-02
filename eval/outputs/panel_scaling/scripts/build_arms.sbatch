#!/usr/bin/env bash
#SBATCH --job-name=arms996
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/arms_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/arms_%j.err
set -euo pipefail
/home/mg7411/.conda/envs/gpt-analysis/bin/python - <<'PYEOF'
# Arms built from ONE translated corpus so every arm shares learners, problems, text recipe and translation run.
#  A  paper25       : the paper's 25 learners, all rows          (already distilled as p25regen; linked here for the record)
#  B  all996_sub    : all 996 learners, 520 rows/learner, seed 0 (size-matched to the paper corpus; same per-learner quota method as p98sub)
#  C  all996_full   : all 996 learners, all rows                  (~20.6M)
#  D  rand25_s42/43 : 25 learners drawn uniformly from the 996, all rows (isolates panel COMPOSITION at equal learner count)
import pandas as pd, numpy as np, os, gzip
PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; SRC=f'{PS}/uma_frac_all996_v10regen_nlp.csv.gz'
allsub=[int(x) for x in open(f'{PS}/subjids_all996.txt').read().split()]
paper25=set(int(x) for x in open(f'{PS}/subjids_paper25.txt').read().split())
probs=pd.read_csv(f'{PS}/problems_20720.csv').prob.astype(str).tolist()
rng=np.random.RandomState(0); PER=520
keepB={s:set(rng.choice(probs,PER,replace=False)) for s in allsub}          # per-learner quota, like subsample.sbatch
D={}
for seed in (42,43):
    D[seed]=set(np.random.RandomState(seed).choice(allsub,25,replace=False).tolist())
    open(f'{PS}/subjids_rand25_s{seed}.txt','w').write('\n'.join(map(str,sorted(D[seed])))+'\n')
outs={'B_all996_sub':f'{PS}/arms/B_all996_sub520k.csv.gz','D_rand25_s42':f'{PS}/arms/D_rand25_s42.csv.gz','D_rand25_s43':f'{PS}/arms/D_rand25_s43.csv.gz','A_paper25_check':f'{PS}/arms/A_paper25_from996.csv.gz'}
fh={k:gzip.open(v,'wt') for k,v in outs.items()}; first={k:True for k in outs}; cnt={k:0 for k in outs}
def emit(k,df):
    if len(df)==0: return
    df.to_csv(fh[k],index=False,header=first[k]); first[k]=False; cnt[k]+=len(df)
for ch in pd.read_csv(SRC,chunksize=400000):
    ch['prob']=ch['prob'].astype(str)
    mB=[p in keepB[s] for s,p in zip(ch.subjid,ch.prob)]; emit('B_all996_sub',ch[mB])
    emit('D_rand25_s42',ch[ch.subjid.isin(D[42])]); emit('D_rand25_s43',ch[ch.subjid.isin(D[43])]); emit('A_paper25_check',ch[ch.subjid.isin(paper25)])
for k in fh: fh[k].close()
for k,v in cnt.items(): print(k,'rows',v)
# shuffle B (row order matters less for the trainer, but match subsample.sbatch which shuffled)
b=pd.read_csv(outs['B_all996_sub']); b=b.sample(frac=1,random_state=0).reset_index(drop=True); b.to_csv(outs['B_all996_sub'],index=False,compression='gzip'); print('B shuffled; learners',b.subjid.nunique())
os.symlink(SRC,f'{PS}/arms/C_all996_full.csv.gz') if not os.path.exists(f'{PS}/arms/C_all996_full.csv.gz') else None
print('ARMS_DONE')
PYEOF
