#!/bin/bash
#SBATCH --job-name=pde-mc-70b
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_mc_h200_70b_%j.log
#SBATCH --error=logs/run_mc_h200_70b_%j.log

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

echo "[mc-h200-70b] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-h200-70b] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[mc-h200-70b] Running Llama-3.3-70B at $(date -u)"
${PYTHON} ${SCRIPT} \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --hf_dataset ${HF_DATASET} \
    --job_id ${JOB_ID} \
    --upload_every 20 \
    --batch_size 2 \
    --tp 2

echo "[mc-h200-70b] Done at $(date -u)"
