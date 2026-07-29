#!/bin/bash
#SBATCH --job-name=pde_probe_raw_grid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/probe_raw_grid_%j.out
#SBATCH --error=probe/logs/probe_raw_grid_%j.err
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

NPZ="probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz"

if [ ! -f "$NPZ" ]; then
    echo "ERROR: $NPZ not found."
    exit 1
fi

POOLS=(mean_pool last_tok)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}
echo "Pool: $POOL"

python probe/probe_hyperparam.py \
    --hidden "$NPZ" \
    --output_dir probe/results/ \
    --pool "$POOL" \
    --representation raw \
    --C_grid 0.01 0.1 1.0 10.0

echo "Done: $(date)"
