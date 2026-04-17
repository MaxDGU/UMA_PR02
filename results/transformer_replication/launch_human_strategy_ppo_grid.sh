#!/usr/bin/env bash
set -euo pipefail

CLASSIFIER_CKPT="results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best"
HUMAN_TARGET_JSON="results/human/human_strategy_targets_classifier160k.json"
SP2013_HUMAN_CSV="sp2013_human.csv"

MODELS=(
  "distilled/135m_final"
  "distilled/360m_final"
  "distilled/1p7b_final"
)

LOSSES=(
  "tv"
  "kl"
  "wasserstein"
)

for model_dir in "${MODELS[@]}"; do
  for loss_name in "${LOSSES[@]}"; do
    python results/transformer_replication/train_human_strategy_ppo.py \
      --init_model_dir "${model_dir}" \
      --classifier_ckpt "${CLASSIFIER_CKPT}" \
      --human_target_json "${HUMAN_TARGET_JSON}" \
      --sp2013_human_csv "${SP2013_HUMAN_CSV}" \
      --loss "${loss_name}" \
      --rollouts_per_prompt 32 \
      --prompt_batch_size 4 \
      --max_new_tokens 128 \
      --temperature 0.7 \
      --top_p 0.95 \
      --top_k 50 \
      --total_updates 100 \
      --local_files_only
  done
done
