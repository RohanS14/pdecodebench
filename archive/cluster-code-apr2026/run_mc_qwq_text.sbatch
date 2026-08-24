#!/bin/bash
#SBATCH --job-name=pde-mc-qwq-text
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run_mc_qwq_text_%j.log
#SBATCH --error=logs/run_mc_qwq_text_%j.log

set -euo pipefail
module load anaconda3/2025.06

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
export TEMP=/scratch/ehb7466/tmp
export TMP=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
DATASET=${WORKSPACE}/data/pdedata_clean_v2.xlsx
RESULTS=${WORKSPACE}/results_mc

mkdir -p ${RESULTS} ${WORKSPACE}/logs

echo "[mc-qwq-text] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-qwq-text] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} ${WORKSPACE}/run_mc_qwq_text.py \
    --dataset ${DATASET} \
    --output_dir ${RESULTS} \
    --batch_size 4

echo "[mc-qwq-text] Done at $(date -u)"
