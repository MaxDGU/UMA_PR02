#!/usr/bin/env bash
#SBATCH --job-name=longhft30
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/longhft30/logs/long_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/longhft30/logs/long_%A_%a.err
# Even longer KL-anchored human FT: 30 epochs (540 updates); otherwise identical (lr 1e-4, kl_beta 1.0).
# Saves epochs 10,15,20,30 and best-by-val-NLL; each is rolled out with SPACED prompts (256 x 16).
set -euo pipefail
ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; L=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/longhft30
PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python; EV=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/eval_spaced.py
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read ARM DISTILL SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $L/jobs.txt)
TAG=${ARM}_kl1.0_ep30_seed$SEED; RUN=$L/runs/$TAG; cd "$ROOT"
[[ -d $RUN/epoch_30 ]] || "$PY" human_ft/finetune_humandata.py --init_strategy checkpoint --model_path "$DISTILL" --checkpoint_subdir best --base_model_name Qwen/Qwen3-4B-Base \
  --train_csv data/human_ft/data_train_nlp.csv --val_csv data/human_ft/data_val_nlp.csv --output_dir "$RUN" --instruction_style plain \
  --epochs 30 --save_epoch_list "10,15,20,30" --max_steps 0 --batch_size 2 --eval_batch_size 2 --grad_accum_steps 8 --lr 1e-4 --warmup_ratio 0.03 --max_length 256 --num_workers 2 \
  --amp_dtype bf16 --kl_beta 1.0 --best_by val_nll --log_every 5 --seed $SEED --no-use_lora --gradient_checkpointing --no-eval_sp2013 --no-eval_loaded_model --local_files_only
for ck in epoch_10 epoch_15 epoch_20 epoch_30 best; do OUT=$L/results/${TAG}_${ck}.csv; [[ -f $OUT ]] || "$PY" $EV --ckpt "$RUN/$ck" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$OUT" --n-samples 256; done
rm -rf "$RUN/final" "$RUN/last"; echo "DONE $TAG"
