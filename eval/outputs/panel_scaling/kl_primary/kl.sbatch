#!/usr/bin/env bash
#SBATCH --job-name=kl_primary
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/kl_primary/logs/kl_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/kl_primary/logs/kl_%A_%a.err
# KL-anchored human-FT (kl_beta 1.0, reference = the arm's own distill checkpoint), protocol identical to every earlier
# human-FT run; evaluated with SPACED problem prompts (training format), 256 samples x 16 SP2013 problems.
set -euo pipefail
ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; K=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/kl_primary
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read ARM DISTILL SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $K/jobs.txt)
TAG=${ARM}_kl1.0_seed$SEED; RUN=$K/runs/$TAG; OUT=$K/results/${TAG}_dh.csv; cd "$ROOT"
[[ -d $RUN/best ]] || "$PY" human_ft/finetune_humandata.py --init_strategy checkpoint --model_path "$DISTILL" --checkpoint_subdir best --base_model_name Qwen/Qwen3-4B-Base \
  --train_csv data/human_ft/data_train_nlp.csv --val_csv data/human_ft/data_val_nlp.csv --output_dir "$RUN" --instruction_style plain \
  --epochs 1 --max_steps 0 --batch_size 2 --eval_batch_size 2 --grad_accum_steps 8 --lr 1e-4 --warmup_ratio 0.03 --max_length 256 --num_workers 2 \
  --amp_dtype bf16 --kl_beta 1.0 --best_by val_nll --log_every 5 --seed $SEED --no-use_lora --gradient_checkpointing --no-eval_sp2013 --no-eval_loaded_model --local_files_only
[[ -f $OUT ]] || "$PY" /scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/eval_spaced.py --ckpt "$RUN/best" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$OUT" --n-samples 256
rm -rf "$RUN/final" "$RUN/last"; echo "DONE $TAG"
