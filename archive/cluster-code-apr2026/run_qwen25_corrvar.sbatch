#!/bin/bash
#SBATCH --job-name=pde-qwen25-cv
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/run_qwen25_corrvar_%j.log
#SBATCH --error=logs/run_qwen25_corrvar_%j.log

set -euo pipefail
module load anaconda3/2025.06

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
DATASET=${WORKSPACE}/data/pdedata_clean_v2.xlsx
RESULTS=${WORKSPACE}/results
HF_DATASET=bermaneh/pde-llm-eval-results-v1
JOB_ID="torch:${SLURM_JOB_ID}"

mkdir -p ${RESULTS} ${WORKSPACE}/logs

echo "[qwen25-cv] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[qwen25-cv] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} ${WORKSPACE}/run_eval.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --hf_dataset ${HF_DATASET} \
    --job_id ${JOB_ID} \
    --upload_every 16 \
    --batch_size 8 \
    --tp 1

echo "[qwen25-cv] Done at $(date -u)"
