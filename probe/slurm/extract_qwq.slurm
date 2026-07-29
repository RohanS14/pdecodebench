#!/bin/bash
#SBATCH --job-name=pde_extract_qwq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/extract_qwq_%j.out
#SBATCH --error=probe/logs/extract_qwq_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rs9768@nyu.edu

set -e
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

source probe/venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:   $(date)"

python probe/extract_hidden.py \
    --dataset data/pdedata_clean_v3.xlsx \
    --model "Qwen/QwQ-32B" \
    --output_dir probe/hidden_states/

echo "Done: $(date)"
