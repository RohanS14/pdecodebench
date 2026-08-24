#!/bin/bash
#SBATCH --job-name=pde-layer-probes
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/run_layer_probes_%j.log
#SBATCH --error=logs/run_layer_probes_%j.log

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
mkdir -p ${WORKSPACE}/results_layer_probes ${WORKSPACE}/logs

echo "[layer-probes] Job ${SLURM_JOB_ID} started at $(date -u) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} ${WORKSPACE}/run_layer_probes.py \
    --dataset ${WORKSPACE}/data/pdedata_clean_v2.xlsx \
    --output_dir ${WORKSPACE}/results_layer_probes

echo "[layer-probes] Done at $(date -u)"
