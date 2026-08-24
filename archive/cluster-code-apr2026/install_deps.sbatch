#!/bin/bash
#SBATCH --job-name=pde-install
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ehb7466/pde-llm-eval/logs/install_%j.log

export TMPDIR=/scratch/ehb7466/tmp
mkdir -p ${TMPDIR}

PYTHON=/scratch/ehb7466/pde-llm-eval/venv/bin/python
echo "Installing deps at $(date -u)"
${PYTHON} -m pip install --quiet vllm sentence-transformers openpyxl datasets huggingface_hub pandas
echo "Done at $(date -u)"
${PYTHON} -c "import vllm; print('vllm ok:', vllm.__version__)"
