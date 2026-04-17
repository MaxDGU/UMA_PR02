#!/bin/bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  ROOT_DIR="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

SUBJID="${SUBJID:-472}"
ENV_PATH="/n/fs/cogai/cs1095/conda-envs/uma-transformer-l40"
G="${G:-0.05}"
D="${D:-0.7}"
RT_MU="${RT_MU:-5}"
ICE="${ICE:-50}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

UMA_DIR="${ROOT_DIR}/results/UMA_replication"
TR_DIR="${ROOT_DIR}/results/transformer_replication"
LOG_DIR="${TR_DIR}/logs"
mkdir -p "${LOG_DIR}"

TRACE_OUT_BASE="${UMA_DIR}/synth_unique_seed_traces/subjid_$(printf '%04d' "${SUBJID}")"
UMA_SP2013_BASE="${UMA_DIR}/sp2013_fixed_subjid_$(printf '%04d' "${SUBJID}")"
CONTEXT_JSON="${TR_DIR}/fixed_tuple_trace_surface_context_subjid$(printf '%04d' "${SUBJID}").json"
DEFAULT_TRAIN_CSV="${TR_DIR}/synth_unique_subjid_0472_seeds1_20_tracev10_default_nosp2013_mincols.csv.gz"
SURFACE_TRAIN_CSV="${TR_DIR}/synth_unique_subjid_0472_seeds1_20_tracev10_surface_hidden_nosp2013_mincols.csv.gz"
UMA_TARGET_CSV="${UMA_SP2013_BASE}/sp2013_subjid_$(printf '%04d' "${SUBJID}")_seed1_1000.csv.gz"

DEFAULT_RUN_DIR="${TR_DIR}/smol2_135m_fixedtuple_subjid$(printf '%04d' "${SUBJID}")_tracev10_default_lr5em4_lin_e5_${RUN_TS}"
SURFACE_RUN_DIR="${TR_DIR}/smol2_135m_fixedtuple_subjid$(printf '%04d' "${SUBJID}")_tracev10_surface_hidden_lr5em4_lin_e5_${RUN_TS}"
DEFAULT_EVAL_DIR="${DEFAULT_RUN_DIR}/sp2013_fixed_tuple_uma1000"
SURFACE_EVAL_DIR="${SURFACE_RUN_DIR}/sp2013_fixed_tuple_uma1000"

MANIFEST="${TR_DIR}/jobs_fixed_tuple_trace_surface_${RUN_TS}.tsv"

bash "${UMA_DIR}/ensure_trained_models_restored.sh"

seed_job="$(
  sbatch --parsable \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",SUBJID="${SUBJID}",SEED_START=1,PROBLEM_CSV="${UMA_DIR}/synthetic_all_unique_num1_9_den2_9.csv",PROBLEM_COL=prob,MODELS_DIR="${UMA_DIR}/trained_models",OUT_BASE_DIR="${TRACE_OUT_BASE}",INCLUDE_TRACE_STEPS=1,TRACE_MAX_SUBPROBLEMS_PER_STEP=8 \
    "${UMA_DIR}/slurm_run_synth_unique_subjid_seed_array.sbatch"
)"

build_job="$(
  sbatch --parsable \
    --dependency=afterok:${seed_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}" \
    "${TR_DIR}/slurm_build_fixed_tuple_trace_surface_datasets.sbatch"
)"

uma_sp2013_job="$(
  sbatch --parsable \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",SUBJID="${SUBJID}",SEED_START=1,SEED_COUNT=1000,OUT_BASE_DIR="${UMA_SP2013_BASE}" \
    "${UMA_DIR}/slurm_run_sp2013_fixed_subjid_seed_array.sbatch"
)"

uma_merge_job="$(
  sbatch --parsable \
    --dependency=afterok:${uma_sp2013_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",SUBJID="${SUBJID}",SEED_START=1,SEED_END=1000,OUT_BASE_DIR="${UMA_SP2013_BASE}" \
    "${UMA_DIR}/slurm_merge_fixed_subjid_sp2013_rollouts.sbatch"
)"

train_default_job="$(
  sbatch --parsable \
    --dependency=afterok:${build_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",DATA_CSV="${DEFAULT_TRAIN_CSV}",OUT_DIR="${DEFAULT_RUN_DIR}",CONTEXT_JSON="${CONTEXT_JSON}",MODEL_NAME=HuggingFaceTB/SmolLM2-135M,LR=5e-4,LR_SCHEDULER_TYPE=linear,EPOCHS=5,BATCH_SIZE=96,EVAL_BATCH_SIZE=96,ID_EVAL_BATCH_SIZE=24,GRAD_ACCUM_STEPS=1,TRAIN_PROMPT_STUDENT_MODE=always,EVAL_SP2013=0,BEST_BY=val_loss,SAVE_BEST_CHECKPOINT=1 \
    "${TR_DIR}/slurm_train_fixed_tuple_trace_surface_135m.sbatch"
)"

train_surface_job="$(
  sbatch --parsable \
    --dependency=afterok:${build_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",DATA_CSV="${SURFACE_TRAIN_CSV}",OUT_DIR="${SURFACE_RUN_DIR}",CONTEXT_JSON="${CONTEXT_JSON}",MODEL_NAME=HuggingFaceTB/SmolLM2-135M,LR=5e-4,LR_SCHEDULER_TYPE=linear,EPOCHS=5,BATCH_SIZE=96,EVAL_BATCH_SIZE=96,ID_EVAL_BATCH_SIZE=24,GRAD_ACCUM_STEPS=1,TRAIN_PROMPT_STUDENT_MODE=always,EVAL_SP2013=0,BEST_BY=val_loss,SAVE_BEST_CHECKPOINT=1 \
    "${TR_DIR}/slurm_train_fixed_tuple_trace_surface_135m.sbatch"
)"

eval_default_job="$(
  sbatch --parsable \
    --dependency=afterok:${train_default_job}:${uma_merge_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",RUN_DIR="${DEFAULT_RUN_DIR}",TARGET_CSV="${UMA_TARGET_CSV}",OUTPUT_DIR="${DEFAULT_EVAL_DIR}",CONTEXT_JSON="${CONTEXT_JSON}",G="${G}",D="${D}",RT_MU="${RT_MU}",ICE="${ICE}",NUM_ROLLOUTS=1000,ROLLOUT_BATCH_SIZE=64,TEMPERATURE=0.7,TOP_P=0.95,TOP_K=50,SAMPLE_SEED=123,PROGRESS_EVERY=1000 \
    "${TR_DIR}/slurm_eval_checkpoint_sp2013_fixed_tuple.sbatch"
)"

eval_surface_job="$(
  sbatch --parsable \
    --dependency=afterok:${train_surface_job}:${uma_merge_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",RUN_DIR="${SURFACE_RUN_DIR}",TARGET_CSV="${UMA_TARGET_CSV}",OUTPUT_DIR="${SURFACE_EVAL_DIR}",CONTEXT_JSON="${CONTEXT_JSON}",G="${G}",D="${D}",RT_MU="${RT_MU}",ICE="${ICE}",NUM_ROLLOUTS=1000,ROLLOUT_BATCH_SIZE=64,TEMPERATURE=0.7,TOP_P=0.95,TOP_K=50,SAMPLE_SEED=123,PROGRESS_EVERY=1000 \
    "${TR_DIR}/slurm_eval_checkpoint_sp2013_fixed_tuple.sbatch"
)"

notebook_job="$(
  sbatch --parsable \
    --dependency=afterok:${eval_default_job}:${eval_surface_job} \
    --export=REPO_ROOT="${ROOT_DIR}",CONDA_ENV="${ENV_PATH}",DEFAULT_ROLLOUTS="${DEFAULT_EVAL_DIR}/sp2013_fixed_tuple_rollouts.csv.gz",SURFACE_ROLLOUTS="${SURFACE_EVAL_DIR}/sp2013_fixed_tuple_rollouts.csv.gz",UMA_TARGET_CSV="${UMA_TARGET_CSV}" \
    "${TR_DIR}/slurm_build_fixed_tuple_trace_surface_sp2013_notebook.sbatch"
)"

{
  printf "stage\tjob_id\tpath_or_note\n"
  printf "synthetic_seed_rollouts\t%s\t%s\n" "${seed_job}" "${TRACE_OUT_BASE}"
  printf "build_datasets\t%s\t%s\n" "${build_job}" "${CONTEXT_JSON}"
  printf "uma_sp2013_rollouts\t%s\t%s\n" "${uma_sp2013_job}" "${UMA_SP2013_BASE}"
  printf "uma_sp2013_merge\t%s\t%s\n" "${uma_merge_job}" "${UMA_TARGET_CSV}"
  printf "train_default\t%s\t%s\n" "${train_default_job}" "${DEFAULT_RUN_DIR}"
  printf "train_surface\t%s\t%s\n" "${train_surface_job}" "${SURFACE_RUN_DIR}"
  printf "eval_default\t%s\t%s\n" "${eval_default_job}" "${DEFAULT_EVAL_DIR}"
  printf "eval_surface\t%s\t%s\n" "${eval_surface_job}" "${SURFACE_EVAL_DIR}"
  printf "notebook\t%s\t%s\n" "${notebook_job}" "${TR_DIR}/analyze_fixed_tuple_trace_surface_sp2013.ipynb"
} > "${MANIFEST}"

echo "Submitted fixed-tuple trace-surface experiment."
echo "Manifest: ${MANIFEST}"
cat "${MANIFEST}"
