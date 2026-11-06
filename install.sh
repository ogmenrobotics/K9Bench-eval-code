#!/bin/bash
# Creates the `k9bench` conda environment and installs all dependencies.
#
#   bash install.sh
#   conda activate k9bench
#
# Override the environment name with CONDA_ENV=<name> bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="${CONDA_ENV:-k9bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
FLASH_ATTN_VERSION="2.7.4.post1"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH. Install miniconda/miniforge first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env '${ENV_NAME}' already exists, reusing it."
else
    echo "Creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})..."
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "Installing ffmpeg..."
conda install -y -c conda-forge ffmpeg

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Installing flash-attn ${FLASH_ATTN_VERSION} (this compiles, it can take a while)..."
python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

echo
echo "Done. Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
