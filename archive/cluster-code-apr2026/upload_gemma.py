"""Upload updated 8-model free-gen results to HF."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "packages" / "key_handler"))
try:
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except Exception as e:
    print(f"[upload] key_handler: {e}")

sys.path.insert(0, str(Path(__file__).parent / "packages" / "hf_utility"))
from hf_utility import push_dataset_to_hub
from datasets import Dataset
import pandas as pd

WORKSPACE = Path(__file__).parent

df = pd.read_csv(WORKSPACE / "results_combined.csv")
print(f"[upload] {len(df)} rows, {df['model'].nunique()} models")
print(df.groupby("model")["title"].count().to_string())

dataset = Dataset.from_pandas(df)
push_dataset_to_hub(
    dataset=dataset,
    dataset_name="pde-llm-eval-results-v1",
    metadata={
        "script_name": "run_eval.py",
        "model": "multi-model",
        "description": "Free-gen PDE eval — 8 models (752 rows). Gemma-3-27B added. Missing: Llama-3.3-70B (gated repo 403 on cluster).",
        "experiment_name": "pde-llm-eval",
        "artifact_status": "partial",
        "canary": False,
    },
    tags=["pde-llm-eval"],
    column_descriptions={
        "model": "Model ID",
        "title": "PDE scenario title (pde_class + mod_type + index)",
        "model_response": "Full model output text",
        "pde_match": "1 if model correctly identified PDE type",
        "valid_match": "1 if model correctly assessed physical validity",
    },
)
print(f"[upload] Done — {len(df)} rows uploaded to bermaneh/pde-llm-eval-results-v1")
