#!/usr/bin/env bash
# Exact SLURM submissions behind the Table 1 fraction model (Qwen3-4B, 25-learner panel), copied verbatim from
# `sacct -j <id> --format=SubmitLine` on della. Paths are della paths. The wrappers run the trainers from $REPO_ROOT;
# use the main_max versions of results/transformer_replication/train_transformer_hf.py and human_ft/finetune_humandata.py
# (see README.md: the May runs had enable_input_require_grads, which the older commit a9823d7 lacks).
set -euo pipefail

# Stage 1: distillation on UMA traces. Job 7607804, submitted 2026-05-02, 1x H200, 3,596 updates, 3.5 h.
sbatch --parsable --partition=ailab --gres=gpu:1 --cpus-per-task=6 --mem=128G --time=2-00:00:00 --job-name=q34b_p25l \
  --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/slurm_logs/q34b_p25l_ailab_20260502_171918_4Bretry_%j.out \
  --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/slurm_logs/q34b_p25l_ailab_20260502_171918_4Bretry_%j.err \
  --export=ALL,REPO_ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft,PYTHON_BIN=/home/mg7411/.conda/envs/gpt-analysis/bin/python,HF_HOME=/scratch/gpfs/GRIFFITHS/mg7411/.cache/huggingface,TASK_LABEL=fraction_panel25_default_4b,DATASET_KIND=fraction_panel25_default,MODEL_NAME=Qwen/Qwen3-4B-Base,DATA_CSV=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/data/distillation/uma_fraction_distillation_25,OUT_DIR=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora,PROMPT_PROBLEM_KIND=fraction,EPOCHS=1,BATCH_SIZE=16,EVAL_BATCH_SIZE=16,GRAD_ACCUM_STEPS=8,LR=5e-4,MIN_LR=1e-6,LR_SCHEDULER_TYPE=cosine,MAX_LENGTH=288,VAL_FRAC=0.05,TEST_FRAC=0.05,SPLIT_BY_PROBLEM=0,BEST_BY=val_loss,ID_EVAL_SIZE=64,ID_EVAL_BATCH_SIZE=4,ID_EVAL_MAX_NEW_TOKENS=240,USE_LORA=1,MODEL_LOAD_DTYPE=bf16,GC=1,EVAL_SP2013=0,SP2013_USE_PARAM_GRID=0,EVAL_ID_FINAL_ANSWER=1,ID_EVAL_REPORT_TRUE_TARGET=1,MOVE_SP2013_ROWS_TO_VAL=0,STRICT_SP2013_ONLY_VALIDATION_LAYOUT=0,TRAIN_PROMPT_STUDENT_MODE=always,TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.0,SAVE_EVAL_CHECKPOINTS=0,SAVE_BEST_CHECKPOINT=1,SAVE_FINAL_CHECKPOINT=1,VERIFY_SAVED_CHECKPOINT_LOAD=0,LOCAL_FILES_ONLY=1,PREVIEW_SAMPLES=0,RESUME_FROM=none,SAVE_LAST_EVERY_UPDATES=2000 \
  results/transformer_replication/slurm_train_qwen3_distill_1epoch.sbatch

# Stage 2: human fine-tuning on Siegler 2011 (288 train / 96 val rows). Job 7637206, submitted 2026-05-03, 18 updates.
# KL_BETA=0.0 (no KL anchor). USE_LORA=0 with INIT_STRATEGY=checkpoint means "keep training the distilled LoRA
# adapter" (33.0M trainable params), not full fine-tuning; see README.md.
sbatch --parsable --partition=ailab --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00 --job-name=hftd4b_ep1 \
  --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/slurm_logs/hftd4b_ep1_ep1_20260503_124020_%j.out \
  --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/slurm_logs/hftd4b_ep1_ep1_20260503_124020_%j.err \
  --export=ALL,REPO_ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft,PYTHON_BIN=/home/mg7411/.conda/envs/gpt-analysis/bin/python,HF_HOME=/scratch/gpfs/GRIFFITHS/mg7411/.cache/huggingface,TASK_LABEL=humanft_distill_4b_ep1,MODEL_NAME=Qwen/Qwen3-4B-Base,OUT_DIR=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen_humanft_from_distill_4b_ep1_20260503_124020,TRAIN_CSV=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/data/human_ft/data_train_nlp.csv,VAL_CSV=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/data/human_ft/data_val_nlp.csv,INSTRUCTION_STYLE=plain,EPOCHS=1,MAX_STEPS=0,BATCH_SIZE=2,EVAL_BATCH_SIZE=2,GRAD_ACCUM_STEPS=8,LR=1e-4,KL_BETA=0.0,USE_LORA=0,GC=1,EVAL_SP2013=0,LOCAL_FILES_ONLY=1,INIT_STRATEGY=checkpoint,MODEL_PATH=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora,CHECKPOINT_SUBDIR=best \
  results/transformer_replication/slurm_train_qwen_humanft_lora.sbatch
