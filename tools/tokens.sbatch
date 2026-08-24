#!/bin/bash
#SBATCH --job-name=xmgen_tokens
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_427_general
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/ehb7466/projects/pde-llm-eval/logs/tokens_%j.out
#SBATCH --error=/scratch/ehb7466/projects/pde-llm-eval/logs/tokens_%j.err
export HF_HOME=/scratch/ehb7466/projects/pde-llm-eval/cache/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/scratch/ehb7466/tmp
PY=/scratch/ehb7466/envs/pdecodebench-vllm/bin/python
[ -x "$PY" ] || { echo "no interpreter: $PY"; exit 1; }
echo "start: $(date -u)"
"$PY" -u ~/measure_tokens.py
echo "end: $(date -u)"
