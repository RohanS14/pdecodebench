#!/bin/bash
#SBATCH --job-name=pde-phi4-l40s
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ehb7466/pde-llm-eval/logs/run_phi4_l40s_%j.log
#SBATCH --error=/scratch/ehb7466/pde-llm-eval/logs/run_phi4_l40s_%j.log

set -euo pipefail
module load anaconda3/2025.06
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
${WORKSPACE}/venv/bin/python ${WORKSPACE}/run_eval.py     --model microsoft/phi-4     --dataset ${WORKSPACE}/data/pdedata_clean_v2.xlsx     --output_dir ${WORKSPACE}/results     --hf_dataset bermaneh/pde-llm-eval-results-v1     --job_id torch:${SLURM_JOB_ID}     --upload_every 20     --batch_size 8     --tp 1
