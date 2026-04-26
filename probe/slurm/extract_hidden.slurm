#!/bin/bash
#SBATCH --job-name=pde_extract
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rs9768@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/extract_%j.out
#SBATCH --error=probe/logs/extract_%j.err

set -e
# Navigate to repo root regardless of where slurm starts the job
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

source probe/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"

python probe/extract_hidden.py \
    --dataset data/pdedata_clean_v2.xlsx \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output_dir probe/hidden_states/

echo "Done: $(date)"
