#!/bin/bash
set -euo pipefail

ROOT_PREFIX="${ROOT_PREFIX:-/n/fs/cogai/cs1095}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT_PREFIX}/conda-envs/uma-transformer-l40}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${ROOT_PREFIX}/.conda/pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT_PREFIX}/.cache/pip}"
HF_HOME="${HF_HOME:-${ROOT_PREFIX}/UMA_PR02/.cache/huggingface}"
TMPDIR="${TMPDIR:-${ROOT_PREFIX}/tmp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"
export CONDA_PKGS_DIRS PIP_CACHE_DIR HF_HOME TMPDIR

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" torch
python -m pip install \
  "transformers>=4.49,<5" \
  pandas \
  numpy \
  matplotlib \
  seaborn \
  jupyter \
  huggingface_hub \
  safetensors \
  sentencepiece

python - <<'PY'
import torch
import transformers
import pandas
import numpy

print(f"python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"pandas={pandas.__version__}")
print(f"numpy={numpy.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
