#!/bin/bash
#SBATCH --job-name=pde-mc-32b
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_mc_h200_32b_%j.log
#SBATCH --error=logs/run_mc_h200_32b_%j.log

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

echo "[mc-h200-32b] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-h200-32b] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_model() {
    local MODEL=$1
    local TP=${2:-1}
    echo "[mc-h200-32b] Running ${MODEL} at $(date -u)"
    ${PYTHON} ${SCRIPT} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --output_dir ${RESULTS} \
        --hf_dataset ${HF_DATASET} \
        --job_id ${JOB_ID} \
        --upload_every 20 \
        --batch_size 4 \
        --tp ${TP}
    echo "[mc-h200-32b] ${MODEL} done at $(date -u)"
}

run_model Qwen/Qwen3-32B 1
run_model Qwen/Qwen2.5-Coder-32B-Instruct 1
run_model google/gemma-3-27b-it 1

echo "[mc-h200-32b] All models complete at $(date -u)"
