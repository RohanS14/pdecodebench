#!/bin/bash
#SBATCH --job-name=pde_setup_env
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/setup_env_%j.out
#SBATCH --error=probe/logs/setup_env_%j.err

set -e
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

bash probe/setup_env.sh

echo "Done:  $(date)"
