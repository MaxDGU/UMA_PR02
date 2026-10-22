#!/usr/bin/env bash
#SBATCH --job-name=c1000_build
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/build_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/build_%j.err
# Full arm-C corpus: all 996 regenerated learners (paper-recipe translation) + the 4 re-simulated learners, every problem.
# Streamed into parquet shards with only the columns the trainer reads, so loading is fast and memory-light.
set -euo pipefail
/home/mg7411/.conda/envs/gpt-analysis/bin/python - <<'PYEOF'
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, os, glob
PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; OUT=f'{PS}/c1000/corpus_parquet'
for f in glob.glob(OUT+'/*.parquet'): os.remove(f)
COLS=['subjid','prob','operation','is_correct','g','d','rt_mu','ice','source_uid','instruction_nl','response_nl','translation_version','reasoning_quality_flags']
srcs=[f'{PS}/uma_frac_all996_v10regen_nlp.csv.gz', f'{PS}/b1000/fill4_c4_nlp.csv.gz']
k=0; n=0; per={}; mism=0
for s in srcs:
    for ch in pd.read_csv(s,usecols=COLS,chunksize=1_000_000,dtype={'source_uid':str,'reasoning_quality_flags':str,'is_correct':str}):
        ch['prob']=ch['prob'].astype(str)
        pq.write_table(pa.Table.from_pandas(ch,preserve_index=False),f'{OUT}/part-{k:05d}.parquet',compression='zstd'); k+=1; n+=len(ch)
        for sid,c in ch.subjid.value_counts().items(): per[sid]=per.get(sid,0)+c
        mism+=ch.reasoning_quality_flags.fillna('').str.contains('arith_claim_mismatch').sum()
        print(f'shard {k} rows so far {n:,}',flush=True)
vals=pd.Series(per)
print(f'TOTAL rows {n:,} | learners {len(vals)} | rows per learner min {vals.min()} max {vals.max()} | arith_mismatch rows {mism}')
assert n==1000*20720 and len(vals)==1000 and vals.min()==vals.max()==20720, 'coverage check failed'
print('BUILD_OK')
PYEOF
