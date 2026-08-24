"""
MC logprob upload helper — run as subprocess, never in the same process as vLLM.
Reads a JSONL results file from run_mc_eval.py and pushes to a private HF dataset.

Usage:
    python upload_mc_helper.py --jsonl results_mc/Qwen2.5-Coder-7B.jsonl \
                               --dataset bermaneh/pde-mc-logprob-canary-v1 \
                               --experiment pde-mc-logprob \
                               --artifact_status partial
"""
import argparse, json, os, sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "packages" / "key_handler"))
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except Exception as e:
    print(f"[upload_mc_helper] key_handler unavailable: {e}", flush=True)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "packages" / "hf_utility"))
    from hf_utility import push_dataset_to_hub
    HF_UTILITY = True
except Exception:
    HF_UTILITY = False

from datasets import Dataset
from huggingface_hub import HfApi

MC_COLUMN_DESCRIPTIONS = {
    "title":            "Dataset row identifier (code snippet)",
    "pde_class":        "Ground truth PDE class (wave/heat/burgers/navier-stokes)",
    "mod_type":         "Modification condition (NoComm_Valid/Comm_Valid/NoComm_InValid/Comm_InValid/NoComm_CorrVar/CorrComm)",
    "model":            "Model ID used for inference",
    "question_type":    "Question category: pde_class | phys_process | num_method | phys_valid",
    "candidate":        "Specific option for T/F questions (e.g. 'diffusion'); null for 4-way MC",
    "letters":          "JSON dict mapping letter → option for this question instance (records shuffle)",
    "correct_letter":   "Ground-truth correct letter (A/B/C/D) derived from GT label and shuffle",
    "logprob_A":        "Log-probability of letter token 'A' (from vLLM top-10 logprobs at position 0)",
    "logprob_B":        "Log-probability of letter token 'B'",
    "logprob_C":        "Log-probability of letter token 'C'; null for T/F if not in top-10",
    "logprob_D":        "Log-probability of letter token 'D'; null for T/F if not in top-10",
    "predicted_letter": "Argmax letter over applicable logprobs (A/B/C/D for MC; A/B for T/F)",
    "correct":          "True if predicted_letter == correct_letter",
    "logprob_correct":  "Log-probability of the correct letter",
    "margin":           "logprob_correct - max(logprob_incorrect) over applicable letters",
    "entropy":          "Shannon entropy over softmax of applicable letter logprobs (nats)",
    "finish_reason":    "vLLM stop reason — should always be 'stop' with max_tokens=1",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    """Keep last occurrence of each (title, model, question_type, candidate) tuple."""
    seen: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        key = (r.get("title"), r.get("model"),
               r.get("question_type"), r.get("candidate"))
        seen[key] = i
    return [rows[i] for i in sorted(seen.values())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",           required=True)
    parser.add_argument("--dataset",         required=True)
    parser.add_argument("--experiment",      default="pde-mc-logprob")
    parser.add_argument("--artifact_status", default="partial", choices=["partial", "final"])
    parser.add_argument("--job_id",          default="")
    parser.add_argument("--cluster",         default="torch")
    parser.add_argument("--canary",          action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.jsonl)
    if not rows:
        print("[upload_mc_helper] No rows to upload.", flush=True)
        sys.exit(0)

    # Deduplicate on full question key to handle partial checkpoint restarts
    rows = deduplicate(rows)
    print(f"[upload_mc_helper] Uploading {len(rows)} rows to {args.dataset} ({args.artifact_status})", flush=True)

    dataset = Dataset.from_list(rows)

    api = HfApi()
    try:
        api.create_repo(args.dataset, repo_type="dataset", exist_ok=True, private=True)
    except Exception as e:
        print(f"[upload_mc_helper] create_repo warning: {e}", flush=True)

    model_ids = list({r.get("model", "") for r in rows})

    if HF_UTILITY:
        push_dataset_to_hub(
            dataset=dataset,
            dataset_name=args.dataset.split("/")[-1],
            metadata={
                "script_name":     "run_mc_eval.py",
                "model":           ", ".join(model_ids),
                "description":     f"PDE MC Logprob Eval — {args.artifact_status}",
                "experiment_name": args.experiment,
                "job_id":          args.job_id,
                "cluster":         args.cluster,
                "artifact_status": args.artifact_status,
                "canary":          args.canary,
                "hyperparameters": {
                    "max_tokens": 1,
                    "logprobs":   10,
                    "temperature": 0.0,
                    "thinking":   "suppressed for Qwen3/QwQ",
                },
            },
            column_descriptions=MC_COLUMN_DESCRIPTIONS,
            tags=[args.experiment, "pde", "logprob", "mc", args.artifact_status],

        )
    else:
        dataset.push_to_hub(args.dataset, private=True)

    print(f"[upload_mc_helper] Done. {len(rows)} rows at {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
