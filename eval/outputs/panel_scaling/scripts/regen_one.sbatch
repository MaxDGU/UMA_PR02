#!/usr/bin/env bash
#SBATCH --job-name=regen996
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/regen_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/regen_%A_%a.err
# Identical to the validated 98-learner chain (frac100/regen_v10/regen_one.sbatch); only paths differ.
set -euo pipefail
R=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft
PS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
SID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$PS/subjids_remaining.txt")
OUT=$PS/raw/subj_${SID}.csv
if [[ -s "$OUT.done" ]]; then echo "SKIP $SID"; exit 0; fi
echo "task=$SLURM_ARRAY_TASK_ID subjid=$SID host=$(hostname) start=$(date -Iseconds)"
"$PY" $PS/xlsx_to_pkl.py --out-dir $PS/models --subjids $SID
cd "$R/results/UMA_replication"
"$PY" run_saved_models_on_problem_set.py --models-dir $PS/models --start-subjid $SID --end-subjid $SID \
  --problem-csv $PS/problems_20720.csv --out-csv "$OUT" --overwrite --progress-every 1
"$PY" - "$OUT" <<'PYEOF'
import pandas as pd, sys
d=pd.read_csv(sys.argv[1]); assert len(d)==20720, len(d)
ok=(d.trace_steps_json.astype(str)!='[]').mean()
print(f"DONE rows={len(d)} acc={d.is_correct.mean()*100:.1f}% trace_steps_nonempty={ok*100:.1f}%")
open(sys.argv[1]+'.done','w').write(f"{len(d)} {ok:.4f}\n")
PYEOF
echo "end=$(date -Iseconds)"
