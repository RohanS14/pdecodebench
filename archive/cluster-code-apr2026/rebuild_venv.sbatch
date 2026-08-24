#!/bin/bash
#SBATCH --job-name=rebuild-venv
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ehb7466/pde-llm-eval/logs/rebuild_venv_%j.log

set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval

echo "[setup] Node: $(hostname), default python3: $(python3 --version 2>&1)"

# Load anaconda to get a stable, versioned Python
module load anaconda3/2025.06
echo "[setup] Conda python: $(python3 --version)"

# Rebuild venv with this Python
rm -rf ${WORKSPACE}/venv
python3 -m venv ${WORKSPACE}/venv
PYTHON=${WORKSPACE}/venv/bin/python

echo "[setup] Venv python: $(${PYTHON} --version)"
${PYTHON} -m pip install --upgrade pip --quiet

echo "[setup] Installing base packages..."
${PYTHON} -m pip install --quiet pandas openpyxl datasets huggingface_hub sentence-transformers

echo "[setup] Installing vllm (this takes a few minutes)..."
${PYTHON} -m pip install --quiet vllm

echo "[setup] Verifying all deps..."
${PYTHON} -c "import vllm, pandas, openpyxl, datasets, huggingface_hub, sentence_transformers; print('[setup] ALL DEPS OK')"
echo "[setup] Done at $(date -u)"
