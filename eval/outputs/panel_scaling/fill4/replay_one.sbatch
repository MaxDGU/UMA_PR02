#!/usr/bin/env bash
#SBATCH --job-name=fill4_replay
#SBATCH --partition=cpu
#SBATCH --account=griffith
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4/logs/%x_%j.err
# Same replay as panel_scaling/regen_one.sbatch, for one new learner and one c variant; outputs kept separate.
set -euo pipefail
R=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; G=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fill4
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python; OUT=$G/raw_c$CVAL/subj_${SID}.csv
MODELS_DIR=$G/xlsx_c$CVAL "$PY" $G/xlsx_to_pkl.py --out-dir $G/models_c$CVAL --subjids $SID
cd "$R/results/UMA_replication"
"$PY" run_saved_models_on_problem_set.py --models-dir $G/models_c$CVAL --start-subjid $SID --end-subjid $SID \
  --problem-csv /scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/problems_20720.csv --out-csv "$OUT" --overwrite --progress-every 1
"$PY" - "$OUT" <<'PYEOF'
import pandas as pd, sys
d=pd.read_csv(sys.argv[1]); assert len(d)==20720, len(d)
ok=(d.trace_steps_json.astype(str)!='[]').mean()
print(f"DONE rows={len(d)} acc={d.is_correct.mean()*100:.1f}% trace_steps_nonempty={ok*100:.1f}%")
open(sys.argv[1]+'.done','w').write(f"{len(d)} {ok:.4f}\n")
PYEOF
