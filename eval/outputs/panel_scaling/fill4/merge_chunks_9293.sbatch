#!/usr/bin/env bash
#SBATCH --job-name=fill4_merge2
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/merge_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/merge_%j.err
# For each of 97/98: if the full-length replay has not already finished, cancel it and write the merged chunks in its place.
set -euo pipefail
G=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4
for SID in 92 93; do
  OUT=$G/raw_c4/subj_${SID}.csv
  if [[ -s $OUT.done ]]; then echo "sid $SID: full replay already finished, keeping it"; continue; fi
  J=$(awk -v s=$SID '$1=="c4" && $2==s {print $3}' $G/replay_jobids.txt); scancel $J || true; sleep 10
  /home/mg7411/.conda/envs/gpt-analysis/bin/python - "$SID" <<'PYEOF'
import pandas as pd, glob, sys
G='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4'; sid=sys.argv[1]
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{G}/raw_c4_parts/subj_{sid}_part*.csv'))],ignore_index=True)
assert len(d)==20720 and d.prob.nunique()==20720, (len(d), d.prob.nunique())
ok=(d.trace_steps_json.astype(str)!='[]').mean(); out=f'{G}/raw_c4/subj_{sid}.csv'
d.to_csv(out,index=False); open(out+'.done','w').write(f"{len(d)} {ok:.4f}\n")
print(f"sid {sid}: merged rows={len(d)} acc={d.is_correct.mean()*100:.1f}% trace_steps_nonempty={ok*100:.1f}%")
PYEOF
done
