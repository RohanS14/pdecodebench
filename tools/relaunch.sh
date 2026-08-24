#!/bin/bash
# The 2-GPU ceiling came from QOS gpu48 (auto-assigned at <=48h walltime, MaxTRESPU
# gres/gpu=2), not from any partition. gpuplus allows 96/user at equal priority.
cd ~/pde-llm-eval || exit 1
BASE=/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen
scancel 16135020 16135021 16135023 16137257 2>/dev/null
sleep 3
sub () {
  MODELS="$1" THINKING_ARMS=on OUTPUT_DIR="$BASE/$2" \
  HF_DATASET="bermaneh/pde-llm-eval-xmodal-gen-$2" UPLOAD_EVERY=128 \
  sbatch --export=ALL --qos=gpuplus --partition="$3" --account="$4" --job-name="xg_$2" \
         sbatch/run_cross_modal_consistency.sbatch
}
sub "Qwen/Qwen3.6-27B"                  qwen3-6-27b        h200_courant torch_pr_427_courant
sub "Qwen/Qwen3.8-27B"                  qwen3-8-27b        h200_courant torch_pr_427_courant
sub "allenai/Olmo-3.1-32B-Think"        olmo-3-1-32b-think h200_courant torch_pr_427_courant
sub "zai-org/GLM-4.7-Flash"             glm-4-7-flash      h200_public  torch_pr_427_general
