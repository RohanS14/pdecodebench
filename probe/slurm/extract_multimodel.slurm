#!/bin/bash
#SBATCH --job-name=pde_extract_multi
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/extract_multi_%A_%a.out
#SBATCH --error=probe/logs/extract_multi_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rs9768@nyu.edu

set -e
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

source probe/venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:   $(date)"

MODELS=(
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    "Qwen/QwQ-32B"
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Model: $MODEL"

python probe/extract_hidden.py \
    --dataset data/pdedata_clean_v3.xlsx \
    --model "$MODEL" \
    --output_dir probe/hidden_states/

echo "Done: $(date)"
