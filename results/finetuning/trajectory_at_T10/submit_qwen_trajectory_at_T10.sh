#!/usr/bin/env bash
# Reproduce the Qwen3 trajectory-at-T1.0 experiment:
#   pretrained base -> distill (1 ep LoRA) -> humanFT cheap-fix (5 ep, save 1/2/3/5)
# for all four sizes (0.6B / 1.7B / 4B / 8B), and eval each stage at T=1.0
# with 256 rollouts per SP2013 problem.
#
# Wall-clock on della ailab partition (H200): ~75 min distill 1.7B (slowest),
# ~3 min humanFT 0.6B, ~5 min eval each, depending on queue concurrency.
#
# Usage:
#   REPO_ROOT=/path/to/UMA_PR02 \
#   PYTHON_BIN=$(command -v python) \
#   HF_HOME=/path/to/.cache/huggingface \
#   bash results/finetuning/trajectory_at_T10/submit_qwen_trajectory_at_T10.sh
#
# Optional knobs:
#   PARTITION (default ailab)  ACCOUNT  QOS
#   SIZES "0p6b 1p7b 4b 8b"  (subset)
#   SKIP_DISTILL=1 / SKIP_HUMANFT=1 / SKIP_EVAL=1
#   FRACTIONS_DIR (where eval script lives; defaults to ../../fractions)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT must point to the UMA_PR02 checkout}"
cd "$REPO_ROOT"

TR_DIR="$REPO_ROOT/results/transformer_replication"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN required}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
TRAJ_DIR="$REPO_ROOT/results/finetuning/trajectory_at_T10"
# Eval driver + sbatch live inside this trajectory_at_T10/ directory so the
# whole experiment is self-contained.
EVAL_PY="$TRAJ_DIR/eval_qwen3_chained_sp2013.py"
EVAL_SBATCH="$TRAJ_DIR/run_eval_qwen3_chained.slurm"

SIZES="${SIZES:-0p6b 1p7b 4b 8b}"
PARTITION="${PARTITION:-ailab}"
ACCOUNT_ARG=""
[[ -n "${ACCOUNT:-}" ]] && ACCOUNT_ARG="--account=$ACCOUNT"
QOS_ARG=""
[[ -n "${QOS:-}" ]] && QOS_ARG="--qos=$QOS"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="$TR_DIR/slurm_logs"
mkdir -p "$LOGDIR"

# ---- Per-size lookup tables ----
declare -A MODEL_NAME
MODEL_NAME[0p6b]="Qwen/Qwen3-0.6B-Base"
MODEL_NAME[1p7b]="Qwen/Qwen3-1.7B-Base"
MODEL_NAME[4b]="Qwen/Qwen3-4B-Base"
MODEL_NAME[8b]="Qwen/Qwen3-8B-Base"

# Distill: eff_batch=128, fits inside one H200 with these splits.
declare -A DISTILL_BSGA
DISTILL_BSGA[0p6b]="32 4"
DISTILL_BSGA[1p7b]="32 4"
DISTILL_BSGA[4b]="16 8"
DISTILL_BSGA[8b]="8 16"

# HumanFT (cheap-fix): eff_batch=16, very small corpus.
declare -A HFT_BSGA
HFT_BSGA[0p6b]="4 4"
HFT_BSGA[1p7b]="4 4"
HFT_BSGA[4b]="2 8"
HFT_BSGA[8b]="1 16"

declare -A DISTILL_JOB DISTILL_OUT
declare -A HFT_JOB HFT_OUT

# ============================================================================
# Stage 1: distill (1 ep LoRA r=16/alpha=32, eff_bs=128, lr=5e-4)
# ============================================================================
if [[ "${SKIP_DISTILL:-0}" != "1" ]]; then
  echo "=== Submitting distill jobs ==="
  for SIZE in $SIZES; do
    BATCH=$(echo "${DISTILL_BSGA[$SIZE]}" | awk '{print $1}')
    GA=$(echo "${DISTILL_BSGA[$SIZE]}" | awk '{print $2}')
    OUT="$TR_DIR/qwen3_${SIZE}_panel25_default_base_e1_${RUN_TS}_lora"
    EXPORT_VARS="ALL,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,HF_HOME=$HF_HOME,\
TASK_LABEL=distill_${SIZE},MODEL_NAME=${MODEL_NAME[$SIZE]},\
DATA_CSV=$REPO_ROOT/data/distillation/uma_fraction_distillation_25,\
OUT_DIR=$OUT,PROMPT_PROBLEM_KIND=fraction,EPOCHS=1,BATCH_SIZE=$BATCH,EVAL_BATCH_SIZE=$BATCH,\
GRAD_ACCUM_STEPS=$GA,LR=5e-4,MIN_LR=1e-6,LR_SCHEDULER_TYPE=cosine,MAX_LENGTH=288,\
VAL_FRAC=0.05,TEST_FRAC=0.05,SPLIT_BY_PROBLEM=0,BEST_BY=val_loss,USE_LORA=1,\
LORA_R=16,LORA_ALPHA=32,MODEL_LOAD_DTYPE=bf16,GC=1,LOCAL_FILES_ONLY=1,\
SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1"
    JID=$(sbatch --parsable --partition=$PARTITION $ACCOUNT_ARG $QOS_ARG \
      --gres=gpu:1 --cpus-per-task=6 --mem=128G --time=04:00:00 \
      --job-name="d${SIZE}" \
      --output="$LOGDIR/d${SIZE}_${RUN_TS}_%j.out" \
      --error="$LOGDIR/d${SIZE}_${RUN_TS}_%j.err" \
      --export="$EXPORT_VARS" \
      "$TR_DIR/slurm_train_qwen3_distill_1epoch.sbatch")
    DISTILL_JOB[$SIZE]=$JID
    DISTILL_OUT[$SIZE]=$OUT
    echo "  $SIZE distill job=$JID out=$OUT"
  done
fi

# ============================================================================
# Stage 2: humanFT cheap-fix (5 ep SFT, LR=1e-4, eff_bs=16, save ep 1/2/3/5)
# Continues the distill LoRA adapter; no fresh LoRA wrap.
# ============================================================================
if [[ "${SKIP_HUMANFT:-0}" != "1" ]]; then
  echo "=== Submitting humanFT jobs ==="
  for SIZE in $SIZES; do
    BATCH=$(echo "${HFT_BSGA[$SIZE]}" | awk '{print $1}')
    GA=$(echo "${HFT_BSGA[$SIZE]}" | awk '{print $2}')
    OUT="$TR_DIR/qwen_humanft_cheapfix_5ep_${SIZE}_${RUN_TS}"
    DEP=""
    [[ -n "${DISTILL_JOB[$SIZE]:-}" ]] && DEP="--dependency=afterok:${DISTILL_JOB[$SIZE]} --kill-on-invalid-dep=yes"
    # SAVE_EPOCH_LIST must be ';'-separated; the sbatch template translates back to ',' for the Python CLI
    # because sbatch --export splits the export string on commas.
    EXPORT_VARS="ALL,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,HF_HOME=$HF_HOME,\
TASK_LABEL=humanft_cheapfix_${SIZE},MODEL_NAME=${MODEL_NAME[$SIZE]},\
OUT_DIR=$OUT,INSTRUCTION_STYLE=plain,EPOCHS=5,MAX_STEPS=0,\
BATCH_SIZE=$BATCH,EVAL_BATCH_SIZE=$BATCH,GRAD_ACCUM_STEPS=$GA,\
LR=1e-4,WARMUP_RATIO=0.03,MAX_LENGTH=288,KL_BETA=0.0,\
USE_LORA=0,GC=1,EVAL_SP2013=0,LOCAL_FILES_ONLY=1,\
INIT_STRATEGY=checkpoint,MODEL_PATH=${DISTILL_OUT[$SIZE]:-$TR_DIR/qwen3_${SIZE}_panel25_default_base_e1_<TS>_lora},\
CHECKPOINT_SUBDIR=best,BEST_BY=val_nll,SAVE_EPOCH_LIST=1;2;3;5"
    JID=$(sbatch --parsable $DEP --partition=$PARTITION $ACCOUNT_ARG $QOS_ARG \
      --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00 \
      --job-name="hft${SIZE}" \
      --output="$LOGDIR/hft${SIZE}_${RUN_TS}_%j.out" \
      --error="$LOGDIR/hft${SIZE}_${RUN_TS}_%j.err" \
      --export="$EXPORT_VARS" \
      "$TR_DIR/slurm_train_qwen_humanft_lora.sbatch")
    HFT_JOB[$SIZE]=$JID
    HFT_OUT[$SIZE]=$OUT
    echo "  $SIZE humanFT job=$JID  out=$OUT"
  done
fi

# ============================================================================
# Stage 3: per-checkpoint SP2013 eval at T=1.0
# Requires the fractions/scripts/eval_qwen3_chained_sp2013.py driver.
# ============================================================================
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  EVAL_DIR="${EVAL_OUT_DIR:-$TRAJ_DIR/evals_${RUN_TS}}"
  mkdir -p "$EVAL_DIR"
  echo "=== Submitting evals ==="
  for SIZE in $SIZES; do
    # Base @ T=1.0 (no adapter).
    JID=$(sbatch --parsable --partition=$PARTITION $ACCOUNT_ARG $QOS_ARG \
      --gres=gpu:1 --cpus-per-task=4 --mem=80G --time=00:45:00 \
      --job-name="eb${SIZE}" \
      --output="$LOGDIR/eb${SIZE}_${RUN_TS}_%j.out" \
      --error="$LOGDIR/eb${SIZE}_${RUN_TS}_%j.err" \
      --export="ALL,CONDITION=traj_${SIZE}_base,BASE_MODEL=${MODEL_NAME[$SIZE]},OUTPUT_DIR=$EVAL_DIR,HUMAN_CSV=$REPO_ROOT/data/siegler_fraction_human.csv,NUM_ROLLOUTS=256,TEMPERATURE=1.0,ROLLOUT_BATCH_SIZE=64,PYTHON_BIN=$PYTHON_BIN,HF_HOME=$HF_HOME" \
      "$EVAL_SBATCH")
    echo "  $SIZE base eval=$JID"

    # Distill checkpoint @ T=1.0.
    DDEP=""
    [[ -n "${DISTILL_JOB[$SIZE]:-}" ]] && DDEP="--dependency=afterok:${DISTILL_JOB[$SIZE]} --kill-on-invalid-dep=yes"
    JID=$(sbatch --parsable $DDEP --partition=$PARTITION $ACCOUNT_ARG $QOS_ARG \
      --gres=gpu:1 --cpus-per-task=4 --mem=80G --time=00:45:00 \
      --job-name="ed${SIZE}" \
      --output="$LOGDIR/ed${SIZE}_${RUN_TS}_%j.out" \
      --error="$LOGDIR/ed${SIZE}_${RUN_TS}_%j.err" \
      --export="ALL,CONDITION=traj_${SIZE}_distill,BASE_MODEL=${MODEL_NAME[$SIZE]},ADAPTER_DIR=${DISTILL_OUT[$SIZE]:-}/best,OUTPUT_DIR=$EVAL_DIR,HUMAN_CSV=$REPO_ROOT/data/siegler_fraction_human.csv,NUM_ROLLOUTS=256,TEMPERATURE=1.0,ROLLOUT_BATCH_SIZE=64,PYTHON_BIN=$PYTHON_BIN,HF_HOME=$HF_HOME" \
      "$EVAL_SBATCH")
    echo "  $SIZE distill eval=$JID"

    # HumanFT epoch_{1,2,3,5} @ T=1.0.
    HDEP=""
    [[ -n "${HFT_JOB[$SIZE]:-}" ]] && HDEP="--dependency=afterok:${HFT_JOB[$SIZE]} --kill-on-invalid-dep=yes"
    for EP in 1 2 3 5; do
      JID=$(sbatch --parsable $HDEP --partition=$PARTITION $ACCOUNT_ARG $QOS_ARG \
        --gres=gpu:1 --cpus-per-task=4 --mem=80G --time=00:45:00 \
        --job-name="eh${SIZE}e${EP}" \
        --output="$LOGDIR/eh${SIZE}e${EP}_${RUN_TS}_%j.out" \
        --error="$LOGDIR/eh${SIZE}e${EP}_${RUN_TS}_%j.err" \
        --export="ALL,CONDITION=traj_${SIZE}_hftep${EP},BASE_MODEL=${MODEL_NAME[$SIZE]},ADAPTER_DIR=${HFT_OUT[$SIZE]:-}/epoch_${EP},OUTPUT_DIR=$EVAL_DIR,HUMAN_CSV=$REPO_ROOT/data/siegler_fraction_human.csv,NUM_ROLLOUTS=256,TEMPERATURE=1.0,ROLLOUT_BATCH_SIZE=64,PYTHON_BIN=$PYTHON_BIN,HF_HOME=$HF_HOME" \
        "$EVAL_SBATCH")
      echo "  $SIZE humanFT ep$EP eval=$JID"
    done
  done
  echo "Eval outputs land in $EVAL_DIR"
fi

echo "=== Summary ==="
printf "%s\t%s\t%s\n" "size" "distill_job" "humanft_job"
for SIZE in $SIZES; do
  printf "%s\t%s\t%s\n" "$SIZE" "${DISTILL_JOB[$SIZE]:--}" "${HFT_JOB[$SIZE]:--}"
done
