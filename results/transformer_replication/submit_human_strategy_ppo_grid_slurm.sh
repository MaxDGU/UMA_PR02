#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SBATCH_SCRIPT="${ROOT_DIR}/results/transformer_replication/slurm_train_human_strategy_ppo_all.sbatch"
OUT_BASE_DIR="${OUT_BASE_DIR:-${ROOT_DIR}/results/transformer_replication}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/transformer_replication/logs}"
mkdir -p "${LOG_DIR}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-${OUT_BASE_DIR}/jobs_human_strategy_ppo_grid_${RUN_TS}.tsv}"
RECORD_PATH="${RECORD_PATH:-${OUT_BASE_DIR}/human_strategy_ppo_grid_${RUN_TS}.json}"
DRY_RUN="${DRY_RUN:-0}"

PARTITION="${PARTITION:-all}"
CONDA_ENV="${CONDA_ENV:-base}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${ROOT_DIR}/results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best}"
HUMAN_TARGET_JSON="${HUMAN_TARGET_JSON:-${ROOT_DIR}/results/human/human_strategy_targets_classifier160k.json}"
SP2013_HUMAN_CSV="${SP2013_HUMAN_CSV:-${ROOT_DIR}/sp2013_human.csv}"

ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-32}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-12}"
MIN_REASONING_CHARS="${MIN_REASONING_CHARS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
TOTAL_UPDATES="${TOTAL_UPDATES:-100}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
EMPTY_RESPONSE_PENALTY="${EMPTY_RESPONSE_PENALTY:-1.0}"
SHORT_REASONING_PENALTY="${SHORT_REASONING_PENALTY:-0.5}"
UNPARSEABLE_ANSWER_PENALTY="${UNPARSEABLE_ANSWER_PENALTY:-0.5}"
BEST_ID_INVALID_COEF="${BEST_ID_INVALID_COEF:-0.5}"
ID_EVAL_EVERY="${ID_EVAL_EVERY:-10}"
ID_EVAL_ROLLOUTS_PER_PROBLEM="${ID_EVAL_ROLLOUTS_PER_PROBLEM:-128}"
OOD_EVAL_ROLLOUTS_PER_PROBLEM="${OOD_EVAL_ROLLOUTS_PER_PROBLEM:-128}"
SAVE_EVERY="${SAVE_EVERY:-0}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-64}"
CLASSIFIER_MAX_LENGTH="${CLASSIFIER_MAX_LENGTH:-256}"
CLASSIFIER_DEVICE="${CLASSIFIER_DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
RUN_UNIT_CHECKS="${RUN_UNIT_CHECKS:-0}"

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

model_tag() {
  local path="$1"
  local base
  base="$(basename "${path}")"
  case "${base}" in
    135m_final) printf '135m' ;;
    360m_final) printf '360m' ;;
    1p7b_final) printf '1p7b' ;;
    *) printf '%s' "${base}" ;;
  esac
}

model_mem() {
  case "$(model_tag "$1")" in
    135m) printf '64G' ;;
    360m) printf '80G' ;;
    1p7b) printf '96G' ;;
    *) printf '80G' ;;
  esac
}

model_time() {
  case "$(model_tag "$1")" in
    135m) printf '1-00:00:00' ;;
    360m) printf '1-12:00:00' ;;
    1p7b) printf '2-00:00:00' ;;
    *) printf '1-12:00:00' ;;
  esac
}

model_rollout_batch() {
  case "$(model_tag "$1")" in
    135m) printf '8' ;;
    360m) printf '4' ;;
    1p7b) printf '2' ;;
    *) printf '4' ;;
  esac
}

model_policy_forward_batch() {
  case "$(model_tag "$1")" in
    135m) printf '8' ;;
    360m) printf '4' ;;
    1p7b) printf '2' ;;
    *) printf '4' ;;
  esac
}

model_train_micro_batch() {
  case "$(model_tag "$1")" in
    135m) printf '8' ;;
    360m) printf '4' ;;
    1p7b) printf '2' ;;
    *) printf '4' ;;
  esac
}

model_mini_batch() {
  case "$(model_tag "$1")" in
    135m) printf '32' ;;
    360m) printf '16' ;;
    1p7b) printf '8' ;;
    *) printf '16' ;;
  esac
}

model_gc() {
  case "$(model_tag "$1")" in
    135m) printf '0' ;;
    360m) printf '0' ;;
    1p7b) printf '1' ;;
    *) printf '1' ;;
  esac
}

printf "model\tloss\tjob_id\tjob_name\tpartition\ttime_limit\tmem\tout_dir\tlog_out\tlog_err\n" > "${MANIFEST}"

echo "Submitting human strategy PPO grid"
echo "run_ts=${RUN_TS}"
echo "partition=${PARTITION}"
echo "classifier_ckpt=${CLASSIFIER_CKPT}"
echo "human_target_json=${HUMAN_TARGET_JSON}"
echo "sp2013_human_csv=${SP2013_HUMAN_CSV}"
echo "min_new_tokens=${MIN_NEW_TOKENS} min_reasoning_chars=${MIN_REASONING_CHARS}"
echo "manifest=${MANIFEST}"

submission_tsv="$(mktemp)"
cleanup() {
  rm -f "${submission_tsv}"
}
trap cleanup EXIT

submitted=0

for model_dir in "${MODELS[@]}"; do
  model_name="$(model_tag "${model_dir}")"
  mem="$(model_mem "${model_dir}")"
  time_limit="$(model_time "${model_dir}")"
  rollout_batch_size="$(model_rollout_batch "${model_dir}")"
  policy_forward_batch_size="$(model_policy_forward_batch "${model_dir}")"
  train_micro_batch_size="$(model_train_micro_batch "${model_dir}")"
  mini_batch_size="$(model_mini_batch "${model_dir}")"
  gradient_checkpointing="$(model_gc "${model_dir}")"

  for loss_name in "${LOSSES[@]}"; do
    job_name="hppo_${model_name}_${loss_name}"
    out_dir="${OUT_BASE_DIR}/human_strategy_ppo_${model_name}_${loss_name}_${RUN_TS}"
    log_out="${LOG_DIR}/${job_name}_${RUN_TS}_%j.out"
    log_err="${LOG_DIR}/${job_name}_${RUN_TS}_%j.err"

    export_vars="ALL,REPO_ROOT=${ROOT_DIR},CONDA_ENV=${CONDA_ENV},INIT_MODEL_DIR=${ROOT_DIR}/${model_dir},CLASSIFIER_CKPT=${CLASSIFIER_CKPT},HUMAN_TARGET_JSON=${HUMAN_TARGET_JSON},SP2013_HUMAN_CSV=${SP2013_HUMAN_CSV},OUTPUT_DIR=${out_dir},LOSS_NAME=${loss_name},ROLLOUTS_PER_PROMPT=${ROLLOUTS_PER_PROMPT},PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE},ROLLOUT_BATCH_SIZE=${rollout_batch_size},POLICY_FORWARD_BATCH_SIZE=${policy_forward_batch_size},TRAIN_MICRO_BATCH_SIZE=${train_micro_batch_size},MINI_BATCH_SIZE=${mini_batch_size},PPO_EPOCHS=4,MAX_NEW_TOKENS=${MAX_NEW_TOKENS},MIN_NEW_TOKENS=${MIN_NEW_TOKENS},MIN_REASONING_CHARS=${MIN_REASONING_CHARS},TEMPERATURE=${TEMPERATURE},TOP_P=${TOP_P},TOP_K=${TOP_K},TOTAL_UPDATES=${TOTAL_UPDATES},SEED=${SEED},LR=${LR},EMPTY_RESPONSE_PENALTY=${EMPTY_RESPONSE_PENALTY},SHORT_REASONING_PENALTY=${SHORT_REASONING_PENALTY},UNPARSEABLE_ANSWER_PENALTY=${UNPARSEABLE_ANSWER_PENALTY},BEST_ID_INVALID_COEF=${BEST_ID_INVALID_COEF},ID_EVAL_EVERY=${ID_EVAL_EVERY},ID_EVAL_ROLLOUTS_PER_PROBLEM=${ID_EVAL_ROLLOUTS_PER_PROBLEM},OOD_EVAL_ROLLOUTS_PER_PROBLEM=${OOD_EVAL_ROLLOUTS_PER_PROBLEM},SAVE_EVERY=${SAVE_EVERY},CLASSIFIER_BATCH_SIZE=${CLASSIFIER_BATCH_SIZE},CLASSIFIER_MAX_LENGTH=${CLASSIFIER_MAX_LENGTH},CLASSIFIER_DEVICE=${CLASSIFIER_DEVICE},LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY},RUN_UNIT_CHECKS=${RUN_UNIT_CHECKS},GRADIENT_CHECKPOINTING=${gradient_checkpointing}"

    submit_args=(
      --parsable
      --partition="${PARTITION}"
      --time="${time_limit}"
      --gpus-per-node=1
      --cpus-per-task=8
      --mem="${mem}"
      --job-name="${job_name}"
      --output="${log_out}"
      --error="${log_err}"
      --export="${export_vars}"
    )
    if [[ -n "${ACCOUNT}" ]]; then
      submit_args+=(--account="${ACCOUNT}")
    fi
    if [[ -n "${QOS}" ]]; then
      submit_args+=(--qos="${QOS}")
    fi
    if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
      read -r -a extra_submit_args <<< "${SBATCH_EXTRA_ARGS}"
      submit_args+=("${extra_submit_args[@]}")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf "[dry-run] sbatch"
      printf " %q" "${submit_args[@]}" "${SBATCH_SCRIPT}"
      printf "\n"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${model_name}" "${loss_name}" "DRY_RUN" "${job_name}" "${PARTITION}" "${time_limit}" "${mem}" "${out_dir}" "${log_out}" "${log_err}" \
        >> "${MANIFEST}"
      continue
    fi

    submit_out="$(sbatch "${submit_args[@]}" "${SBATCH_SCRIPT}")"
    job_id="${submit_out%%;*}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${model_name}" "${loss_name}" "${job_id}" "${job_name}" "${PARTITION}" "${time_limit}" "${mem}" "${out_dir}" "${log_out}" "${log_err}" \
      >> "${MANIFEST}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_id}" "${model_name}" "${loss_name}" "${job_name}" "${out_dir}" "${log_out}" \
      >> "${submission_tsv}"
    echo "submitted ${job_id}: model=${model_name} loss=${loss_name}"
    echo "  out_dir=${out_dir}"
    submitted=$((submitted + 1))
  done
done

python - "${submission_tsv}" "${RECORD_PATH}" "${RUN_TS}" <<'PY'
import json
import sys

submission_tsv, record_path, run_ts = sys.argv[1:4]
jobs = []
with open(submission_tsv, "r", encoding="utf-8") as f:
    for line in f:
        job_id, model_name, loss_name, job_name, out_dir, log_out = line.rstrip("\n").split("\t")
        jobs.append(
            {
                "job_id": job_id,
                "model": model_name,
                "loss": loss_name,
                "job_name": job_name,
                "out_dir": out_dir,
                "log_out": log_out,
            }
        )

payload = {
    "run_ts": run_ts,
    "note": (
        "Human strategy-matching PPO grid over 3 distilled model sizes and 3 strategy-distribution losses. "
        "Targets are classifier-posteriors averaged over all train+val human responses. "
        "ID eval uses the 8 human-train problems; OOD eval uses the 16 unique sp2013_human problems."
    ),
    "jobs": jobs,
}
with open(record_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"wrote {record_path}")
PY

echo
echo "Submitted ${submitted} human strategy PPO jobs"
echo "Manifest: ${MANIFEST}"
echo "Record:   ${RECORD_PATH}"
