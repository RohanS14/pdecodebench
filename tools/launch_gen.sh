#!/bin/bash
# One job per model. Each gets its OWN OUTPUT_DIR and its OWN HF repo, so all six
# can run concurrently: push_dataset_to_hub REPLACES a split and the uploader globs
# a whole dir, so a shared dir or repo would let jobs clobber each other's uploads.
# Split across courant (88 GPU cap) and cds (24) to spread queue pressure.
cd ~/pde-llm-eval || exit 1
BASE=/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen

submit () {  # $1=model  $2=slug  $3=partition  $4=account
  MODELS="$1" THINKING_ARMS=on \
  OUTPUT_DIR="$BASE/$2" \
  HF_DATASET="bermaneh/pde-llm-eval-xmodal-gen-$2" \
  UPLOAD_EVERY=128 \
  sbatch --export=ALL --partition="$3" --account="$4" \
         --job-name="xg_$2" sbatch/run_cross_modal_consistency.sbatch
}

submit "Qwen/Qwen3.5-27B"                            qwen3-5-27b          h200_courant torch_pr_427_courant
submit "Qwen/Qwen3.6-27B"                            qwen3-6-27b          h200_courant torch_pr_427_courant
submit "Qwen/Qwen3.8-27B"                            qwen3-8-27b          h200_courant torch_pr_427_courant
submit "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"  nemotron-3-nano-30b  h200_cds     torch_pr_1168_cds
submit "allenai/Olmo-3.1-32B-Think"                  olmo-3-1-32b-think   h200_cds     torch_pr_1168_cds
submit "zai-org/GLM-4.7-Flash"                       glm-4-7-flash        h200_cds     torch_pr_1168_cds
