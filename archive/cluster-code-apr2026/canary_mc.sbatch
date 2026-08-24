#!/bin/bash
#SBATCH --job-name=pde-mc-canary
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/canary_mc_%j.log
#SBATCH --error=logs/canary_mc_%j.log

set -euo pipefail

# Load anaconda for consistent Python across login/compute nodes
module load anaconda3/2025.06

# ── Critical: must be set before ANY Python or CUDA init ──────────────────
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export PYTHONHASHSEED=0
export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
PIP=${WORKSPACE}/venv/bin/pip
HF_CLI=${WORKSPACE}/venv/bin/huggingface-cli
SCRIPT=${WORKSPACE}/run_mc_eval.py
DATASET=${WORKSPACE}/data/pdedata_clean_v2.xlsx
RESULTS=${WORKSPACE}/results_mc
HF_CANARY=bermaneh/pde-mc-logprob-canary-v1
JOB_ID="torch:${SLURM_JOB_ID}"

mkdir -p ${RESULTS} ${WORKSPACE}/logs

echo "[mc-canary] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-canary] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Install deps — skip if already installed (pip module broken on this cluster) ──
${PYTHON} -c "import vllm, sentence_transformers, openpyxl, datasets, huggingface_hub, pandas" 2>/dev/null \
    || ${PYTHON} -m pip install --quiet vllm sentence-transformers openpyxl datasets huggingface_hub pandas \
    || echo "[mc-canary] pip unavailable but packages may already be installed — continuing"


# ── Model 1: Qwen2.5-Coder-7B-Instruct ───────────────────────────────────
echo "[mc-canary] Running Qwen2.5-Coder-7B-Instruct at $(date -u)"
${PYTHON} ${SCRIPT} \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --hf_dataset ${HF_CANARY} \
    --job_id ${JOB_ID} \
    --upload_every 20 \
    --batch_size 8 \
    --tp 1 \
    --canary

echo "[mc-canary] Qwen2.5-Coder-7B-Instruct done at $(date -u)"

# ── Model 2: QwQ-32B (validates thinking suppression with max_tokens=1) ──
echo "[mc-canary] Running QwQ-32B at $(date -u)"
${PYTHON} ${SCRIPT} \
    --model Qwen/QwQ-32B \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --hf_dataset ${HF_CANARY} \
    --job_id ${JOB_ID} \
    --upload_every 20 \
    --batch_size 4 \
    --tp 1 \
    --canary

echo "[mc-canary] QwQ-32B done at $(date -u)"
echo "[mc-canary] Canary complete at $(date -u)"
