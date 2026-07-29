#!/bin/bash
#SBATCH --job-name=pde_rsa_32b
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/rsa_32b_%A_%a.out
#SBATCH --error=probe/logs/rsa_32b_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs9768@nyu.edu

set -e
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

source probe/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID (task $SLURM_ARRAY_TASK_ID)"
echo "Start:  $(date)"

SLUGS=(coder32b qwq32b)
NPZS=(
    "probe/hidden_states/Qwen_Qwen2.5-Coder-32B-Instruct.npz"
    "probe/hidden_states/Qwen_QwQ-32B.npz"
)

SLUG=${SLUGS[$SLURM_ARRAY_TASK_ID]}
NPZ=${NPZS[$SLURM_ARRAY_TASK_ID]}

echo "Slug: $SLUG  NPZ: $NPZ"

mkdir -p probe/results/rsa/

python probe/compute_rsa.py \
    --hidden "$NPZ" \
    --output_dir probe/results/rsa/ \
    --slug "$SLUG"

echo "Done: $(date)"
