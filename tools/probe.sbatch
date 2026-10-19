#!/bin/bash
#SBATCH --job-name=xmgen_probe
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_427_general
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/ehb7466/projects/pde-llm-eval/logs/probe_%j.out
#SBATCH --error=/scratch/ehb7466/projects/pde-llm-eval/logs/probe_%j.err
# One cache, HF_HOME only. See tools/README.md.
export HF_HOME=/scratch/ehb7466/hf_cache
export TMPDIR=/scratch/ehb7466/tmp
PY=/scratch/ehb7466/envs/pdecodebench-vllm/bin/python
[ -x "$PY" ] || { echo "no interpreter: $PY"; exit 1; }
echo "start: $(date -u)"
"$PY" -u $HOME/pde-llm-eval/tools/probe_configs.py
echo "end: $(date -u)"
