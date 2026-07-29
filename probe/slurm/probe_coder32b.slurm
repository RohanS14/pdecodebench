#!/bin/bash
#SBATCH --job-name=pde_probe_coder32b
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/probe_coder32b_%j.out
#SBATCH --error=probe/logs/probe_coder32b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rs9768@nyu.edu

set -e
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

source probe/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

NPZ="probe/hidden_states/Qwen_Qwen2.5-Coder-32B-Instruct.npz"

if [ ! -f "$NPZ" ]; then
    echo "ERROR: $NPZ not found."
    exit 1
fi

mkdir -p probe/results/coder32b/

python probe/probe_hyperparam.py \
    --hidden "$NPZ" \
    --output_dir probe/results/coder32b/ \
    --pool mean_pool \
    --representation pca20 \
    --C_grid 1.0

python probe/probe_hyperparam.py \
    --hidden "$NPZ" \
    --output_dir probe/results/coder32b/ \
    --pool last_tok \
    --representation pca20 \
    --C_grid 1.0

echo "Done: $(date)"
