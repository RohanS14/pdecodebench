#!/bin/bash
#SBATCH --job-name=pde-pca
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_pca_%j.log
#SBATCH --error=logs/run_pca_%j.log

set -euo pipefail
module load anaconda3/2025.06

export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
export TEMP=/scratch/ehb7466/tmp
export TMP=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
mkdir -p ${WORKSPACE}/results_pca ${WORKSPACE}/logs

echo "[pca] Job ${SLURM_JOB_ID} started at $(date -u) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Extract hidden-state representations (resumable — skips existing JSONL files)
for MODEL in \
    "microsoft/phi-4" \
    "Qwen/Qwen2.5-Coder-7B-Instruct" \
    "Qwen/Qwen2.5-Coder-32B-Instruct" \
    "Qwen/Qwen3-32B" \
    "Qwen/QwQ-32B" \
    "mistralai/Mistral-Nemo-Instruct-2407" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "google/gemma-3-27b-it"
do
    echo "[pca] Extracting representations for ${MODEL} at $(date -u)"
    ${PYTHON} ${WORKSPACE}/run_pca_repr.py \
        --model ${MODEL} \
        --dataset ${WORKSPACE}/data/pdedata_clean_v2.xlsx \
        --output_dir ${WORKSPACE}/results_pca \
        --batch_size 1 \
        --max_length 2048
done

echo "[pca] Running PCA analysis and generating visualization at $(date -u)"
${PYTHON} ${WORKSPACE}/analyze_pca_repr.py \
    --input_dir ${WORKSPACE}/results_pca \
    --output ${WORKSPACE}/results_pca/pca_repr.html

echo "[pca] Done at $(date -u)"
