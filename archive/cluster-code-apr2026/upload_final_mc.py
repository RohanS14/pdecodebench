import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages/key_handler"))
try:
    from key_handler import KeyHandler; KeyHandler.set_env_key()
except: pass
sys.path.insert(0, str(Path(__file__).parent / "packages/hf_utility"))
from hf_utility import push_dataset_to_hub
from datasets import Dataset
import pandas as pd

df = pd.read_csv(Path(__file__).parent / "results_mc_combined.csv")
print(f"[upload] {len(df)} rows, {df.model.nunique()} models")
dataset = Dataset.from_pandas(df)
push_dataset_to_hub(
    dataset=dataset,
    dataset_name="pde-mc-logprob-results-v1",
    metadata={
        "script_name": "run_mc_eval.py",
        "model": "multi-model",
        "description": "MC logprob eval — 9 models, 7776 rows. QwQ-32B logprobs null (vLLM 0.19.1 thinking bug). Llama-3.3-70B has 89 null logprob_correct (correct answer outside top-10). Final artifact.",
        "experiment_name": "pde-mc-logprob",
        "artifact_status": "final",
        "canary": False,
    },
    tags=["pde-mc-logprob"],
    column_descriptions={
        "model": "Model ID",
        "title": "PDE scenario title",
        "question_type": "pde_class / phys_process / num_method / phys_valid",
        "mod_type": "Code perturbation condition",
        "correct": "Whether predicted_letter matches correct_letter",
        "logprob_correct": "Log-probability assigned to the correct answer token (null if outside top-10)",
        "entropy": "Entropy of the A/B/C/D logprob distribution (nats)",
        "margin": "logprob_correct - logprob of highest non-correct letter",
    },
)
print(f"Done — {len(df)} rows uploaded")
