#!/bin/bash
cd ~/pde-llm-eval || exit 1
BASE=/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen
sub () {  # model slug partition account
  local common=(--export=ALL --partition="$3" --account="$4" --job-name="xg_$2" sbatch/run_cross_modal_consistency.sbatch)
  local env="MODELS=$1 THINKING_ARMS=on OUTPUT_DIR=$BASE/$2 HF_DATASET=bermaneh/pde-llm-eval-xmodal-gen-$2 UPLOAD_EVERY=128"
  # try gpu168 (4 GPUs/user); fall back to the default QOS if it is rejected
  if out=$(env $env sbatch --qos=gpu168 "${common[@]}" 2>&1); then
    echo "gpu168  $2 -> $out"
  else
    out=$(env $env sbatch "${common[@]}" 2>&1); echo "default $2 -> $out"
  fi
}
sub "Qwen/Qwen3.6-27B"           qwen3-6-27b        h200_courant torch_pr_427_courant
sub "Qwen/Qwen3.8-27B"           qwen3-8-27b        h200_courant torch_pr_427_courant
sub "allenai/Olmo-3.1-32B-Think" olmo-3-1-32b-think h200_courant torch_pr_427_courant
sub "zai-org/GLM-4.7-Flash"      glm-4-7-flash      h200_public  torch_pr_427_general
