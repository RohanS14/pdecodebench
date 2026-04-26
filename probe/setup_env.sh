#!/bin/bash
# Run once from repo root on a GPU node: bash probe/setup_env.sh
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m venv probe/venv
source probe/venv/bin/activate
pip install --upgrade pip

# Detect CUDA version from nvidia-smi (handles table-border | in output)
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | tr -d '.')

if [ -z "$CUDA_VERSION" ]; then
    echo "No GPU / nvidia-smi not found — installing CPU-only torch"
    pip install torch
else
    echo "Detected CUDA $CUDA_VERSION — installing torch for cu${CUDA_VERSION}"
    pip install torch --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
fi

pip install transformers accelerate
pip install numpy scipy scikit-learn pandas openpyxl matplotlib seaborn
echo "venv setup complete: probe/venv"
