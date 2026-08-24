"""
Merge phys_valid rerun results into the full results dataset and upload to HF.

Strategy: start from the existing results CSV (all 7776 rows, old prompt), strip
phys_valid rows, then append new ones from the rerun output directory.

Usage (run on cluster after run_mc_valid_rerun.sbatch completes):
    python merge_valid_rerun.py \
        --base_csv    results_mc/results.csv \
        --rerun_dir   results_mc_valid_rerun \
        --output_csv  results_mc_valid_rerun/results_merged.csv \
        --hf_dataset  bermaneh/pde-mc-logprob-results-v2 \
        --job_id      torch:SLURM_JOB_ID
"""
import argparse, json, os, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "key_handler"))
try:
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except Exception as e:
    print(f"[merge] key_handler unavailable: {e}", flush=True)

QWQ_MODEL = "Qwen/QwQ-32B"

MODELS = [
    "microsoft/phi-4",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-3-27b-it",
]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv",   required=True,
                        help="Path to existing full results CSV (7776 rows, old prompt)")
    parser.add_argument("--rerun_dir",  default="results_mc_valid_rerun",
                        help="Dir with new phys_valid JSONL files (one per model)")
    parser.add_argument("--output_csv", default="results_mc_valid_rerun/results_merged.csv")
    parser.add_argument("--hf_dataset", default="bermaneh/pde-mc-logprob-results-v2")
    parser.add_argument("--job_id",     default="")
    args = parser.parse_args()

    # Load base CSV (all 7776 rows, old phys_valid prompt)
    print(f"[merge] Loading base CSV: {args.base_csv}", flush=True)
    base_df = pd.read_csv(args.base_csv)
    print(f"[merge] Base rows: {len(base_df)}", flush=True)
    assert len(base_df) == 7776, f"Expected 7776 rows in base CSV, got {len(base_df)}"

    # Strip all old phys_valid rows
    kept = base_df[base_df["question_type"] != "phys_valid"].copy()
    print(f"[merge] Kept {len(kept)} non-phys_valid rows", flush=True)
    assert len(kept) == 6912, f"Expected 6912 non-phys_valid rows, got {len(kept)}"

    # Load new phys_valid rows from rerun
    new_rows: list[dict] = []
    for model in MODELS:
        slug = model.replace("/", "__")
        rerun_path = os.path.join(args.rerun_dir, f"{slug}.jsonl")
        rows = load_jsonl(rerun_path)
        valid_rows = [r for r in rows if r.get("question_type") == "phys_valid"]

        if len(valid_rows) != 96:
            print(f"[merge] ERROR: {model} has {len(valid_rows)} phys_valid rows (expected 96)", flush=True)
            sys.exit(1)

        # Tag QwQ-32B rows with scoring_method
        for r in valid_rows:
            if model == QWQ_MODEL and "scoring_method" not in r:
                r["scoring_method"] = "text_extraction"
        new_rows.extend(valid_rows)

    new_df = pd.DataFrame(new_rows)
    print(f"[merge] New phys_valid rows: {len(new_df)}", flush=True)

    # Merge
    merged = pd.concat([kept, new_df], ignore_index=True)

    # Ensure column order; add scoring_method if missing
    base_cols = list(kept.columns)
    if "scoring_method" not in base_cols:
        base_cols = base_cols + ["scoring_method"]
    for col in base_cols:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[base_cols]

    print(f"[merge] Merged total: {len(merged)} rows", flush=True)

    # Validate
    qt_counts = merged["question_type"].value_counts().to_dict()
    print(f"[merge] Question type counts: {qt_counts}", flush=True)
    expected = {"pde_class": 864, "phys_process": 3456, "num_method": 2592, "phys_valid": 864}
    for qt, expected_count in expected.items():
        actual = qt_counts.get(qt, 0)
        if actual != expected_count:
            print(f"[merge] ERROR: {qt} count is {actual}, expected {expected_count}", flush=True)
            sys.exit(1)

    model_counts = merged["model"].value_counts().to_dict()
    for model in MODELS:
        c = model_counts.get(model, 0)
        if c != 864:
            print(f"[merge] WARNING: {model} has {c} rows (expected 864)", flush=True)

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"[merge] Written to {args.output_csv}", flush=True)

    # Upload to HF
    print(f"[merge] Uploading to {args.hf_dataset} ...", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "hf_utility"))
    try:
        from hf_utility import push_dataset_to_hub
        from datasets import Dataset
        hf_ds = Dataset.from_pandas(merged)
        push_dataset_to_hub(
            dataset=hf_ds,
            dataset_name="pde-mc-logprob-results-v2",
            experiment_slug="pde-mc-logprob",
            metadata={
                "script_name": "merge_valid_rerun.py",
                "description": (
                    "Full MC logprob results — 9 models × 96 rows × 9 question types = 7776 rows. "
                    "phys_valid rows rerun with updated prompt: "
                    "'Does this code run and produce a correct physical solution for the PDE?' "
                    "(replaces original 'Is this simulation physically valid?')"
                ),
                "experiment_name": "pde-mc-logprob",
                "job_id": args.job_id,
                "cluster": "torch",
                "artifact_status": "final",
                "canary": False,
            },
            tags=["pde-mc-logprob", "logprob-eval", "phys-valid-rerun"],
            column_descriptions={
                "model":            "HuggingFace model ID",
                "title":            "Code snippet identifier",
                "pde_class":        "Ground-truth PDE class",
                "mod_type":         "Code condition (e.g. NoComm_Valid, CorrComm)",
                "question_type":    "pde_class | phys_process | num_method | phys_valid",
                "candidate":        "Specific candidate for T/F questions; null for 4-way MC",
                "letters":          "JSON map of letter → option for this instance (A/B shuffled)",
                "correct_letter":   "Ground-truth correct letter (A/B/C/D)",
                "logprob_A":        "Log-probability of token A at position 0",
                "logprob_B":        "Log-probability of token B at position 0",
                "logprob_C":        "Log-probability of token C (null for T/F questions)",
                "logprob_D":        "Log-probability of token D (null for T/F questions)",
                "predicted_letter": "Argmax letter over applicable logprobs",
                "correct":          "predicted_letter == correct_letter",
                "logprob_correct":  "Logprob of the correct answer letter",
                "margin":           "logprob_correct - max(logprob_incorrect)",
                "entropy":          "Entropy over softmax of applicable letter logprobs",
                "finish_reason":    "vLLM finish reason (should be 'stop' for max_tokens=1)",
                "scoring_method":   "null = logprob argmax; 'text_extraction' = QwQ-32B text parse",
            },
        )
        print(f"[merge] Upload complete: {args.hf_dataset}", flush=True)
    except Exception as e:
        print(f"[merge] Upload failed: {e}", flush=True)
        raise


if __name__ == "__main__":
    main()
