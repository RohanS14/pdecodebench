#!/bin/bash
#SBATCH --job-name=test-l40s
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/ehb7466/pde-llm-eval/logs/test_public_%j.log
nvidia-smi
