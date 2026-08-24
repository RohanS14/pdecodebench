#!/bin/bash
#SBATCH --job-name=pde-mc-l40s
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_mc_l40s_%j.log
#SBATCH --error=logs/run_mc_l40s_%j.log

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

echo "[mc-l40s] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-l40s] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} -c "import vllm, openpyxl, datasets, huggingface_hub, pandas" 2>/dev/null     || ${PYTHON} -m pip install --quiet vllm openpyxl datasets huggingface_hub pandas     || echo "[mc-l40s] pip unavailable but packages may already be installed — continuing"

run_model() {
    local MODEL=$1
    echo "[mc-l40s] Running ${MODEL} at $(date -u)"
    ${PYTHON} ${SCRIPT}         --model ${MODEL}         --dataset ${DATASET}         --output_dir ${RESULTS}         --hf_dataset ${HF_DATASET}         --job_id ${JOB_ID}         --upload_every 20         --batch_size 8         --tp 1
    echo "[mc-l40s] ${MODEL} done at $(date -u)"
}

run_model meta-llama/Llama-3.1-8B-Instruct
run_model mistralai/Mistral-Nemo-Instruct-2407
run_model microsoft/phi-4

echo "[mc-l40s] All L40S models complete at $(date -u)"
