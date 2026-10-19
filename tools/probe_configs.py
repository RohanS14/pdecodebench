"""CPU-side loadability probe: build vLLM's ModelConfig for each model.

Registry membership is NOT loadability. gemma-4 is registered as
Gemma4ForConditionalGeneration and still dies in create_engine_config on a
per-layer head_dim. This runs the same code path, on CPU, before any GPU is held.
"""
import traceback

MODELS = [
    # All eight cross-representation checkpoints. Registry membership is not
    # loadability; two models have already been dropped for being registry-valid
    # and unloadable, and a GPU allocation is the wrong place to learn that.
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "Qwen/QwQ-32B", "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "allenai/Olmo-3.1-32B-Think",
    "zai-org/GLM-4.7-Flash",
    "google/gemma-4-31B-it",
]

from vllm.engine.arg_utils import EngineArgs

for m in MODELS:
    try:
        cfg = EngineArgs(model=m, trust_remote_code=True, max_model_len=49152,
                         enforce_eager=True).create_model_config()
        mc = cfg
        print(f"OK    {m}")
        print(f"        max_len={mc.max_model_len} dtype={mc.dtype} "
              f"arch={getattr(mc, 'architectures', '?')}")
    except Exception as e:
        print(f"FAIL  {m}")
        print(f"        {type(e).__name__}: {str(e).splitlines()[0][:200]}")
