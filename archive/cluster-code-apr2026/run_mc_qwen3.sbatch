#!/bin/bash
#SBATCH --job-name=pde-mc-qwen3
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run_mc_qwen3_%j.log
#SBATCH --error=logs/run_mc_qwen3_%j.log

set -euo pipefail
module load anaconda3/2025.06

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
SCRIPT=${WORKSPACE}/run_mc_eval.py
DATASET=${WORKSPACE}/data/pdedata_clean_v2.xlsx
RESULTS=${WORKSPACE}/results_mc
HF_DATASET=bermaneh/pde-mc-logprob-results-v1
JOB_ID="torch:${SLURM_JOB_ID}"

mkdir -p ${RESULTS} ${WORKSPACE}/logs

echo "[mc-qwen3] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-qwen3] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} ${SCRIPT} \
    --model Qwen/Qwen3-32B \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --hf_dataset ${HF_DATASET} \
    --job_id ${JOB_ID} \
    --upload_every 20 \
    --batch_size 4 \
    --tp 1

echo "[mc-qwen3] Done at $(date -u)"
