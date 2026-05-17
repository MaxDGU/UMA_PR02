#!/usr/bin/env bash
# Submit per-profile SP2013 evals on a student-conditioned Qwen3-4B checkpoint.
# Reads /tmp/student_profile_run.env from the distill submission for OUT_DIR,
# HF_HOME, PYTHON_BIN. Submits 3 jobs (low/mid/high) with --dependency=afterok.
set -euo pipefail

source /tmp/student_profile_run.env

FRACTIONS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions
EVAL_SLURM=$FRACTIONS/slurm/run_eval_studentcond.slurm
EVAL_OUT_BASE=$FRACTIONS/finetune/output/qwen3_4b_studentcond_profile_evals_${RUN_TS}_v2
HUMAN_CSV=$FRACTIONS/data/siegler_fraction_human.csv
ADAPTER_DIR=$OUT_DIR/best  # train_transformer_hf saves best/ when save_best_checkpoint=1

# Fallback: if best/ doesn't exist (rare), try final/
if [[ ! -d "$ADAPTER_DIR" ]]; then
  ADAPTER_DIR=$OUT_DIR/final
fi

# Profile reps from student_profiles_4b_summary.csv (built from canonical
# SP2013 per-student accuracies in verification_full_results.csv).
declare -A PROFILE_G PROFILE_D PROFILE_RT PROFILE_ICE
PROFILE_G[low]=0.01;  PROFILE_D[low]=0.1;  PROFILE_RT[low]=3; PROFILE_ICE[low]=100
PROFILE_G[mid]=0.07;  PROFILE_D[mid]=0.1;  PROFILE_RT[mid]=5; PROFILE_ICE[mid]=25
PROFILE_G[high]=0.10; PROFILE_D[high]=0.7; PROFILE_RT[high]=4; PROFILE_ICE[high]=25

mkdir -p "$EVAL_OUT_BASE"
DEP_FLAG=""
# (No dependency — distill already finished; this is a re-submission with
# corrected profile reps.)

JIDS=""
for PROF in low mid high; do
  CONDITION="q3_4b_studentcond_${PROF}"
  EXPORT_VARS="ALL,PYTHON_BIN=$PYTHON_BIN,HF_HOME=$HF_HOME,\
CONDITION=$CONDITION,BASE_MODEL=Qwen/Qwen3-4B-Base,\
ADAPTER_DIR=$ADAPTER_DIR,OUTPUT_DIR=$EVAL_OUT_BASE,HUMAN_CSV=$HUMAN_CSV,\
NUM_ROLLOUTS=256,TEMPERATURE=1.0,ROLLOUT_BATCH_SIZE=32,SAVE_ROLLOUTS=1,\
STUDENT_G=${PROFILE_G[$PROF]},STUDENT_D=${PROFILE_D[$PROF]},\
STUDENT_RT=${PROFILE_RT[$PROF]},STUDENT_ICE=${PROFILE_ICE[$PROF]}"
  JID=$(sbatch --parsable $DEP_FLAG \
    --job-name=eval4bsc_${PROF} \
    --export="$EXPORT_VARS" \
    "$EVAL_SLURM")
  echo "profile=$PROF job=$JID"
  JIDS="$JIDS $JID"
done
echo "eval out base: $EVAL_OUT_BASE"
echo "adapter: $ADAPTER_DIR"
echo "EVAL_OUT_BASE=$EVAL_OUT_BASE" >> /tmp/student_profile_run.env
echo "ADAPTER_DIR=$ADAPTER_DIR" >> /tmp/student_profile_run.env
echo "EVAL_JIDS=$JIDS" >> /tmp/student_profile_run.env
