#!/usr/bin/env bash
#SBATCH --job-name=hft_arm
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/hft_%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/logs/hft_%x_%j.err
# Canonical human-FT (paper protocol: 1 epoch, lr 1e-4, full-FT, best by val NLL) x seeds 42-45, then SP2013 rollouts (256/problem).
set -euo pipefail
ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; TR=$ROOT/results/transformer_replication
PS=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling
SV=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DISTILL=$TR/qwen3_4b_${ARM}_v10_e1_v10regen_lora
cd "$ROOT"
[[ -f $PS/results/${ARM}_distill.csv ]] || "$PY" "$SV/eval_seed_ckpt.py" --ckpt "$DISTILL/best" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$PS/results/${ARM}_distill.csv" --n-samples 256
for SEED in 42 43 44 45; do
  RUN_DIR=$PS/hft_runs/${ARM}_seed${SEED}; OUT=$PS/results/${ARM}_dh_seed${SEED}.csv
  [[ -f "$OUT" ]] && continue
  "$PY" "$ROOT/human_ft/finetune_humandata.py" --init_strategy checkpoint --model_path "$DISTILL" --checkpoint_subdir best --base_model_name Qwen/Qwen3-4B-Base \
    --train_csv "$ROOT/data/human_ft/data_train_nlp.csv" --val_csv "$ROOT/data/human_ft/data_val_nlp.csv" --output_dir "$RUN_DIR" --instruction_style plain \
    --epochs 1 --max_steps 0 --batch_size 2 --eval_batch_size 2 --grad_accum_steps 8 --lr 1e-4 --warmup_ratio 0.03 --max_length 256 --num_workers 2 \
    --amp_dtype bf16 --kl_beta 0.0 --best_by val_nll --log_every 5 --seed $SEED --no-use_lora --gradient_checkpointing --no-eval_sp2013 --no-eval_loaded_model --local_files_only
  "$PY" "$SV/eval_seed_ckpt.py" --ckpt "$RUN_DIR/best" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$OUT" --n-samples 256
  [[ "$SEED" == "42" ]] || rm -rf "$RUN_DIR/best" "$RUN_DIR/final" "$RUN_DIR/last"
done
echo "DONE $ARM"
