#!/bin/bash
#SBATCH --job-name=pde_viz
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/viz_%j.out
#SBATCH --error=probe/logs/viz_%j.err
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

python probe/viz_interactive.py \
    --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
    --results_dir probe/results/ \
    --output probe/results/report.html

echo "Done: $(date)"
