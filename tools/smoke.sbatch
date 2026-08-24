#!/bin/bash
#SBATCH --job-name=xmgen_smoke
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_427_courant
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:35:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/ehb7466/projects/pde-llm-eval/logs/smoke_%j.out
#SBATCH --error=/scratch/ehb7466/projects/pde-llm-eval/logs/smoke_%j.err
VENV_PATH=/scratch/ehb7466/envs/pdecodebench-vllm
export PATH="$VENV_PATH/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/ehb7466/projects/pde-llm-eval/cache/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/scratch/ehb7466/tmp
PY="$VENV_PATH/bin/python"
echo "=== toolchain on the COMPUTE node ==="
for t in ninja gcc g++ cc nvcc cmake; do printf '%-6s -> %s\n' "$t" "$(command -v $t || echo MISSING)"; done
echo "=== minimal vLLM generate (Qwen3.5-27B, 8 tokens) ==="
"$PY" -u - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3.5-27B", tensor_parallel_size=1,
          max_model_len=69632, trust_remote_code=True, enforce_eager=True)
out = llm.chat([[{"role":"user","content":"Reply with the single word: ok"}]],
               sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
               chat_template_kwargs={"enable_thinking": True})
print("GENERATED:", repr(out[0].outputs[0].text))
print("SMOKE TEST PASSED")
EOF
echo "exit: $?"
