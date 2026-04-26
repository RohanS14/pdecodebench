#!/bin/bash
#SBATCH --job-name=pde_canary
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rs9768@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --account=torch_pr_154_courant
#SBATCH --output=probe/logs/canary_%j.out
#SBATCH --error=probe/logs/canary_%j.err

set -e
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(realpath "$0")")/../.."
fi

source probe/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"

GT_SAMPLES="Wave_1,Heat_1"
NPZ="probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct_canary.npz"

mkdir -p probe/logs probe/hidden_states probe/results probe/figures

# Step 1: Extract hidden states for canary gt_samples
echo "=== Step 1: Extract ==="
python probe/extract_hidden.py \
    --dataset data/pdedata_clean_v3.xlsx \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output_dir probe/hidden_states/ \
    --gt_samples "$GT_SAMPLES"

echo "=== Step 2: RSA ==="
python probe/rsa_analysis.py \
    --hidden "$NPZ" \
    --layers 0,7,14 \
    --output_dir probe/figures/ \
    --pool mean_pool

python probe/rsa_analysis.py \
    --hidden "$NPZ" \
    --layers 0,7,14 \
    --output_dir probe/figures/ \
    --pool last_tok

echo "=== Step 3: Pooled Probe ==="
python probe/linear_probe_pooled.py \
    --hidden "$NPZ" \
    --output_dir probe/results/ \
    --pool mean_pool

python probe/linear_probe_pooled.py \
    --hidden "$NPZ" \
    --output_dir probe/results/ \
    --pool last_tok

echo "Done: $(date)"
