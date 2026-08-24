"""How long does the context actually need to be for this benchmark?

Prompt length is measured PER MODEL, with that model's own tokenizer and its own
chat template, because a chars->tokens ratio is not portable across tokenizers and
the template itself adds tokens. Weights are never downloaded -- tokenizer only.
"""
import sys, numpy as np
sys.path.insert(0, "/home/ehb7466/pde-llm-eval")

from eval.consistency_prompts import (ViewSources, build_messages, load_items,
                                      load_exec_trajectories)
from eval.run_cross_modal_consistency import TOGGLEABLE

MODELS = ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B",
          "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
          "allenai/Olmo-3.1-32B-Think", "zai-org/GLM-4.7-Flash",
          "google/gemma-4-31B-it",
          "Qwen/Qwen3-32B"]          # published reference

items = load_items("/home/ehb7466/pde-llm-eval/data/multimodal_items_v1.csv")
try:
    exec_traj = load_exec_trajectories("/home/ehb7466/pde-llm-eval/data/exec_trajectories.npz")
except Exception as e:
    print("exec_traj load failed:", e); exec_traj = {}
src = ViewSources("/home/ehb7466/pde-llm-eval/data/multimodality_physics_with_trajectories.csv",
                  "/home/ehb7466/pde-llm-eval/data/merged_mod_jul28.csv", exec_traj)

msgs, meta = [], []
for it in items:
    try:
        msgs.append(build_messages(it, src))
        meta.append((it["item_id"], it["gt_sample"], it["condition"]))
    except Exception:
        pass
print(f"built {len(msgs)} prompts\n")

from transformers import AutoTokenizer
print(f"{'model':<45} {'median':>7} {'p99':>7} {'MAX':>7}  worst item")
print("-" * 100)
results = {}
for m in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        kw = {"enable_thinking": True} if m in TOGGLEABLE else {}
        n = []
        for mm in msgs:
            try:
                s = tok.apply_chat_template(mm, tokenize=False,
                                            add_generation_prompt=True, **kw)
            except Exception:
                s = tok.apply_chat_template(mm, tokenize=False,
                                            add_generation_prompt=True)
            n.append(len(tok(s, add_special_tokens=False)["input_ids"]))
        n = np.array(n)
        i = int(n.argmax())
        results[m] = int(n.max())
        print(f"{m:<45} {int(np.median(n)):>7} {int(np.percentile(n,99)):>7} "
              f"{int(n.max()):>7}  {meta[i][0]}")
    except Exception as e:
        print(f"{m:<45} TOKENIZER FAILED: {type(e).__name__}: {str(e)[:70]}")

if results:
    worst = max(results.values())
    print(f"\nworst-case prompt across all models: {worst} tokens")
    for out in (16384, 32768):
        print(f"  to guarantee {out} output tokens on EVERY item: "
              f"max_model_len >= {worst + out}")
