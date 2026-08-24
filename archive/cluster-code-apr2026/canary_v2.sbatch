#!/bin/bash
#SBATCH --job-name=pde-llm-canary-v2
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/canary_v2_%j.log
#SBATCH --error=logs/canary_v2_%j.log

set -euo pipefail

# ── Critical: must be set before ANY Python or CUDA init ──────────────────
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
export PYTHONPATH=/scratch/ehb7466/pde-llm-eval:/scratch/ehb7466/pde-llm-eval/scripts
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
SCRIPT=${WORKSPACE}/scripts/run_eval.py
DATASET=${WORKSPACE}/data/pdedata_clean_v2.xlsx
RESULTS=${WORKSPACE}/results
HF_CANARY=bermaneh/pde-llm-eval-canary-v2
JOB_ID="torch:${SLURM_JOB_ID}"

mkdir -p ${RESULTS} ${WORKSPACE}/logs

echo "[canary-v2] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[canary-v2] Node: $(hostname)"
echo "[canary-v2] Prompt: Updated validity question"
echo "[canary-v2] Running: 2 models (canary), all 6 conditions, 96 samples"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Install deps ──
${PYTHON} -c "import vllm, sentence_transformers, openpyxl, datasets, huggingface_hub, pandas" 2>/dev/null \
    || ${PYTHON} -m pip install --quiet vllm sentence-transformers openpyxl datasets huggingface_hub pandas \
    || echo "[canary-v2] pip unavailable but packages may already be installed — continuing"

# ── Model 1: Qwen2.5-Coder-7B-Instruct (standard baseline) ───────────────
echo "[canary-v2] Running Qwen2.5-Coder-7B-Instruct at $(date -u)"
${PYTHON} ${SCRIPT} \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --batch_size 8 \
    --tp 1

echo "[canary-v2] Qwen2.5-Coder-7B-Instruct completed at $(date -u)"

# ── Model 2: QwQ-32B (reasoning — validates thinking suppression) ─────────
echo "[canary-v2] Running QwQ-32B at $(date -u)"
${PYTHON} ${SCRIPT} \
    --model Qwen/QwQ-32B \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --batch_size 4 \
    --tp 1

echo "[canary-v2] QwQ-32B completed at $(date -u)"
echo "[canary-v2] Canary complete at $(date -u). Check results in ${RESULTS}/"
