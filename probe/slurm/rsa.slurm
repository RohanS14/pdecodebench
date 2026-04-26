#!/bin/bash
#SBATCH --job-name=pde_rsa
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/rsa_%j.out
#SBATCH --error=probe/logs/rsa_%j.err

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
    echo "ERROR: $NPZ not found. Run extract_hidden.slurm first."
    exit 1
fi

python probe/rsa_analysis.py \
    --hidden "$NPZ" \
    --layers 0,7,14,21,28 \
    --output_dir probe/figures/ \
    --pool mean_pool

python probe/rsa_analysis.py \
    --hidden "$NPZ" \
    --layers 0,7,14,21,28 \
    --output_dir probe/figures/ \
    --pool last_tok

echo "Done: $(date)"
