#!/usr/bin/env bash
# usage: submit_distill.sh ARM DATA_CSV NPROC GRAD_ACCUM WALL DEP   -> prints jobid.  Effective batch = 16*GRAD_ACCUM*NPROC = 128 for every arm.
set -euo pipefail
ARM=$1; DATA=$2; NPROC=$3; GA=$4; WALL=$5; DEP=${6:-}
ROOT_DIR=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; TR_DIR="$ROOT_DIR/results/transformer_replication"
PS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling
out_dir="$TR_DIR/qwen3_4b_${ARM}_v10_e1_v10regen_lora"
ev="ALL,REPO_ROOT=${ROOT_DIR},PYTHON_BIN=/home/mg7411/.conda/envs/gpt-analysis/bin/python,HF_HOME=/scratch/gpfs/GRIFFITHS/mg7411/.cache/huggingface"
ev+=",TASK_LABEL=${ARM}_v10regen,DATASET_KIND=fraction_panel25_default,MODEL_NAME=Qwen/Qwen3-4B-Base,DATA_CSV=${DATA},OUT_DIR=${out_dir}"
ev+=",PROMPT_PROBLEM_KIND=fraction,EPOCHS=1,BATCH_SIZE=16,EVAL_BATCH_SIZE=16,GRAD_ACCUM_STEPS=${GA},LR=5e-4,MIN_LR=1e-6,LR_SCHEDULER_TYPE=cosine"
ev+=",MAX_LENGTH=288,VAL_FRAC=0.05,TEST_FRAC=0.05,SPLIT_BY_PROBLEM=0,BEST_BY=val_loss,ID_EVAL_SIZE=64,ID_EVAL_BATCH_SIZE=4,ID_EVAL_MAX_NEW_TOKENS=240"
ev+=",USE_LORA=1,MODEL_LOAD_DTYPE=bf16,GC=1,EVAL_SP2013=0,SP2013_USE_PARAM_GRID=0,EVAL_ID_FINAL_ANSWER=1,ID_EVAL_REPORT_TRUE_TARGET=1,MOVE_SP2013_ROWS_TO_VAL=0"
ev+=",STRICT_SP2013_ONLY_VALIDATION_LAYOUT=0,TRAIN_PROMPT_STUDENT_MODE=always,TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,SAVE_EVAL_CHECKPOINTS=0"
ev+=",SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1,VERIFY_SAVED_CHECKPOINT_LOAD=0,LOCAL_FILES_ONLY=1,PREVIEW_SAMPLES=0,RESUME_FROM=none,SAVE_LAST_EVERY_UPDATES=2000"
ev+=",NPROC=${NPROC},MASTER_PORT=$((29600 + RANDOM % 200))"
DEPARG=(); [[ -n "$DEP" ]] && DEPARG=(--dependency=afterok:$DEP)
sbatch --parsable "${DEPARG[@]}" --partition=${PARTITION:-pli} ${ACCOUNT_ARG---account=nam} --gres=gpu:${NPROC} --cpus-per-task=$((4*NPROC)) --mem=$((64*NPROC))G \
  --time=$WALL --job-name=d_${ARM} --output=$PS/logs/distill_${ARM}_%j.out --error=$PS/logs/distill_${ARM}_%j.err --export="$ev" "${DISTILL_SCRIPT:-$PS/distill_ddp.sbatch}"
