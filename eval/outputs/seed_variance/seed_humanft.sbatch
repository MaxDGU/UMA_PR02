#!/usr/bin/env bash
#SBATCH --job-name=seed_humanft
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/logs/seed_humanft_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance/logs/seed_humanft_%A_%a.err

# 4 configs x 4 seeds (43-46). Seed 42 = the existing canonical runs.
# Each task: humanFT from the frozen distill checkpoint with the exact
# canonical train_args.json settings + new seed, then rollout eval of the
# canonical checkpoint (best; epoch_3 for inst_frac).

set -euo pipefail
echo "host=$(hostname) array_id=$SLURM_ARRAY_TASK_ID job_id=$SLURM_JOB_ID"

ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft
TR=$ROOT/results/transformer_replication
SV=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance
PYTHON_BIN=/home/mg7411/.conda/envs/gpt-analysis/bin/python

export HF_HOME=/home/mg7411/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(base_frac inst_frac base_dec inst_dec)
SEEDS=(43 44 45 46)
CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / 4))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
echo "config=$CONFIG seed=$SEED"

case $CONFIG in
  base_frac)
    BASE_MODEL=Qwen/Qwen3-4B-Base
    DISTILL=$TR/qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora
    TRAIN_CSV=$ROOT/data/human_ft/data_train_nlp.csv
    VAL_CSV=$ROOT/data/human_ft/data_val_nlp.csv
    STYLE=plain; EPOCHS=1; LR=1e-4; DOMAIN=fraction; CANON=best; SAVE_LIST="" ;;
  inst_frac)
    BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
    DISTILL=$TR/qwen3_4bi_panel25_default_base_e1_20260624_233913_lora
    TRAIN_CSV=$ROOT/data/human_ft/data_train_nlp.csv
    VAL_CSV=$ROOT/data/human_ft/data_val_nlp.csv
    STYLE=plain; EPOCHS=3; LR=5e-5; DOMAIN=fraction; CANON=epoch_3; SAVE_LIST="1,2,3" ;;
  base_dec)
    BASE_MODEL=Qwen/Qwen3-4B-Base
    DISTILL=$TR/qwen3_4b_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260518_134624_lora
    TRAIN_CSV=$ROOT/data/human_ft/data_bss2021_train_nlp.csv
    VAL_CSV=$ROOT/data/human_ft/data_bss2021_val_nlp.csv
    STYLE=existing; EPOCHS=3; LR=1e-4; DOMAIN=decimal; CANON=best; SAVE_LIST="" ;;
  inst_dec)
    BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
    DISTILL=$TR/qwen3_4bi_decimal_bssmix_no3dpmul_repaired_v1_base_e1_20260624_233751_lora
    TRAIN_CSV=$ROOT/data/human_ft/data_bss2021_train_nlp.csv
    VAL_CSV=$ROOT/data/human_ft/data_bss2021_val_nlp.csv
    STYLE=existing; EPOCHS=3; LR=1e-4; DOMAIN=decimal; CANON=best; SAVE_LIST="" ;;
esac

RUN_DIR=$SV/runs/${CONFIG}_seed${SEED}
OUT_CSV=$SV/rollouts/${CONFIG}_seed${SEED}.csv
mkdir -p "$SV/runs" "$SV/rollouts" "$SV/logs"

if [[ -f "$OUT_CSV" ]]; then
  echo "SKIP: $OUT_CSV exists"
  exit 0
fi

SAVE_ARGS=()
if [[ -n "$SAVE_LIST" ]]; then SAVE_ARGS=(--save_epoch_list "$SAVE_LIST"); fi

cd "$ROOT"
echo "[$(date -Iseconds)] === TRAIN $CONFIG seed=$SEED ==="
"$PYTHON_BIN" "$ROOT/human_ft/finetune_humandata.py" \
  --init_strategy checkpoint \
  --model_path "$DISTILL" \
  --checkpoint_subdir best \
  --base_model_name "$BASE_MODEL" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --output_dir "$RUN_DIR" \
  --instruction_style "$STYLE" \
  --epochs "$EPOCHS" \
  --max_steps 0 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --grad_accum_steps 8 \
  --lr "$LR" \
  --warmup_ratio 0.03 \
  --max_length 256 \
  --num_workers 2 \
  --amp_dtype bf16 \
  --kl_beta 0.0 \
  --best_by val_nll \
  --log_every 10 \
  --seed "$SEED" \
  --no-use_lora \
  --gradient_checkpointing \
  "${SAVE_ARGS[@]}" \
  --no-eval_sp2013 \
  --no-eval_loaded_model \
  --local_files_only

echo "[$(date -Iseconds)] === EVAL $CONFIG seed=$SEED ($CANON) ==="
"$PYTHON_BIN" "$SV/eval_seed_ckpt.py" \
  --ckpt "$RUN_DIR/$CANON" \
  --base-model "$BASE_MODEL" \
  --domain "$DOMAIN" \
  --out-csv "$OUT_CSV" \
  --n-samples 100

echo "[$(date -Iseconds)] === DONE $CONFIG seed=$SEED ==="
