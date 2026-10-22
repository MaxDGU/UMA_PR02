#!/usr/bin/env bash
#SBATCH --job-name=b1000_hft
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=42-45
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/logs/hft_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000/logs/hft_%A_%a.err
# Paper human-FT protocol (identical to arm B), one seed per array task; checkpoints KEPT for upload.
set -euo pipefail
ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; TR=$ROOT/results/transformer_replication; K=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/b1000
SV=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance; PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=$SLURM_ARRAY_TASK_ID; DISTILL=$TR/qwen3_4b_B1000_sub_v10_e1_v10regen_lora; RUN=$K/hft_runs/B1000_sub_seed$SEED; OUT=$K/results/B1000_sub_dh_seed$SEED.csv
cd "$ROOT"
[[ -d $RUN/best ]] || "$PY" human_ft/finetune_humandata.py --init_strategy checkpoint --model_path "$DISTILL" --checkpoint_subdir best --base_model_name Qwen/Qwen3-4B-Base \
  --train_csv data/human_ft/data_train_nlp.csv --val_csv data/human_ft/data_val_nlp.csv --output_dir "$RUN" --instruction_style plain \
  --epochs 1 --max_steps 0 --batch_size 2 --eval_batch_size 2 --grad_accum_steps 8 --lr 1e-4 --warmup_ratio 0.03 --max_length 256 --num_workers 2 \
  --amp_dtype bf16 --kl_beta 0.0 --best_by val_nll --log_every 5 --seed $SEED --no-use_lora --gradient_checkpointing --no-eval_sp2013 --no-eval_loaded_model --local_files_only
[[ -f $OUT ]] || "$PY" "$SV/eval_seed_ckpt.py" --ckpt "$RUN/best" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$OUT" --n-samples 256
rm -rf "$RUN/final" "$RUN/last"; echo DONE
