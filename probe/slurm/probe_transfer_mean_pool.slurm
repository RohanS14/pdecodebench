#!/bin/bash
#SBATCH --job-name=pde_transfer_mp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/probe_transfer_%j.out
#SBATCH --error=probe/logs/probe_transfer_%j.err
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
echo "Start:  $(date)"

NPZ="probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz"
if [ ! -f "$NPZ" ]; then
    echo "ERROR: $NPZ not found."
    exit 1
fi

python probe/linear_probe_clean_transfer.py \
    --hidden "$NPZ" \
    --output_dir probe/results/ \
    --pool mean_pool

echo "Done: $(date)"
