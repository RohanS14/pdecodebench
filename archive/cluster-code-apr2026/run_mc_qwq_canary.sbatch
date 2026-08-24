#!/bin/bash
#SBATCH --job-name=pde-mc-qwq-canary
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mc_qwq_canary_%j.log
#SBATCH --error=logs/mc_qwq_canary_%j.log

set -euo pipefail
module load anaconda3/2025.06

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python

echo "[qwq-canary] Job ${SLURM_JOB_ID} started at $(date -u) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} ${WORKSPACE}/qwq_no_think_test.py

echo "[qwq-canary] Done at $(date -u)"
