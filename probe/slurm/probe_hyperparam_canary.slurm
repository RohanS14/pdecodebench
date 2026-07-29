#!/bin/bash
#SBATCH --job-name=pde_probe_hp_canary
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/probe_hp_canary_%j.out
#SBATCH --error=probe/logs/probe_hp_canary_%j.err
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

# Canary: C=1.0 only, layers 0 and 14, both representations, both poolings
for POOL in mean_pool last_tok; do
    for REPR in raw pca20; do
        python probe/probe_hyperparam.py \
            --hidden "$NPZ" \
            --output_dir probe/results/canary_hp/ \
            --pool "$POOL" \
            --representation "$REPR" \
            --C_grid 1.0 \
            --layers 0 14
        echo "Done: $POOL / $REPR"
    done
done

echo "Canary complete: $(date)"
