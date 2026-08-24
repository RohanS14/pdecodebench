#!/bin/bash
#SBATCH --job-name=pde-mc-full-v2
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mc_full_v2_%j.log
#SBATCH --error=logs/mc_full_v2_%j.log

set -euo pipefail

# ── Critical: must be set before ANY Python or CUDA init ──────────────────
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/hf_cache
export TRANSFORMERS_CACHE=/scratch/ehb7466/hf_cache
export PYTHONHASHSEED=0
export TMPDIR=/scratch/ehb7466/tmp
export PYTHONPATH=/scratch/ehb7466/pde-llm-eval:/scratch/ehb7466/pde-llm-eval/scripts
mkdir -p ${TMPDIR}

WORKSPACE=/scratch/ehb7466/pde-llm-eval
PYTHON=${WORKSPACE}/venv/bin/python
SCRIPT=${WORKSPACE}/scripts/run_mc_eval.py
DATASET=${WORKSPACE}/data/pdedata_clean_v2.xlsx
RESULTS=${WORKSPACE}/results_mc
HF_DATASET=bermaneh/pde-mc-logprob-results-v2
JOB_ID="torch:${SLURM_JOB_ID}"

mkdir -p ${RESULTS} ${WORKSPACE}/logs

echo "[mc-full-v2] Job ${SLURM_JOB_ID} started at $(date -u)"
echo "[mc-full-v2] Node: $(hostname)"
echo "[mc-full-v2] Prompt: Updated validity question (T/F)"
echo "[mc-full-v2] Running: 9 models × 96 samples × 9 questions = 7,776 total logprob evaluations"
echo "[mc-full-v2] Expected time: ~6-7 hours"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Install deps ──
${PYTHON} -c "import vllm, sentence_transformers, openpyxl, datasets, huggingface_hub, pandas" 2>/dev/null \
    || ${PYTHON} -m pip install --quiet vllm sentence-transformers openpyxl datasets huggingface_hub pandas \
    || echo "[mc-full-v2] pip unavailable but packages may already be installed — continuing"

declare -a MODELS=(
    "microsoft/phi-4"
    "Qwen/Qwen3-32B"
    "Qwen/QwQ-32B"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "google/gemma-3-27b-it"
)

TP=(
    1  # phi-4
    1  # Qwen3-32B
    1  # QwQ-32B
    1  # Qwen2.5-Coder-7B
    1  # Qwen2.5-Coder-32B
    1  # Llama-3.1-8B
    2  # Llama-3.3-70B (TP=2)
    1  # Mistral-12B
    1  # Gemma-3-27B
)

BS=(
    16  # phi-4 (MC questions are small, can batch more)
    8   # Qwen3-32B
    8   # QwQ-32B
    16  # Qwen2.5-Coder-7B
    8   # Qwen2.5-Coder-32B
    16  # Llama-3.1-8B
    4   # Llama-3.3-70B (TP=2, smaller batch)
    12  # Mistral-12B
    8   # Gemma-3-27B
)

for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    tp=${TP[$i]}
    bs=${BS[$i]}
    model_slug=$(echo ${model} | sed 's|/|__|g')

    echo ""
    echo "[mc-full-v2] Model $((i+1))/9: ${model} (TP=${tp}, BS=${bs}) at $(date -u)"

    ${PYTHON} ${SCRIPT} \
        --model "${model}" \
        --dataset ${DATASET} \
        --output_dir ${RESULTS} \
        --batch_size ${bs} \
        --tp ${tp}

    echo "[mc-full-v2] ${model} completed at $(date -u)"
done

echo ""
echo "[mc-full-v2] All 9 models complete at $(date -u)"
echo "[mc-full-v2] Results in ${RESULTS}/"
echo "[mc-full-v2] Download with: raca download torch ${RESULTS}/results.csv"
