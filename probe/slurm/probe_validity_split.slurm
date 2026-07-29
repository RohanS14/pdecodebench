#!/bin/bash
#SBATCH --job-name=probe_validity_split
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/probe_validity_split_%j.out
#SBATCH --error=probe/logs/probe_validity_split_%j.err
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

for POOL in mean_pool last_tok; do
    python probe/probe_validity_split.py \
        --hidden "$NPZ" \
        --output_dir probe/results/validity_split/ \
        --pool "$POOL" \
        --C 1.0
    echo "Done: $POOL"
done

echo "Complete: $(date)"
