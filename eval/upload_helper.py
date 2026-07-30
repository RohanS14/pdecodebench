"""
Upload helper for Experiment 1 (free-generation, run_eval.py).

Runs in a separate subprocess so HF imports never touch the vLLM process — vLLM's
EngineCore dies if huggingface_hub creates threads/connections inside its process.

Recovered from the cluster copy at projects/pde-llm-eval/code/upload_helper.py and
adapted to this repo's conventions: `--workspace` locates the vendored packages, matching
upload_helper_var.py, rather than assuming they sit next to this file.

Why it exists: run_mc_v3_all_models.sbatch defines an inline upload_partial() and calls it
after every model, so Experiment 2 streams partial artifacts. The free-gen sbatch uploads
only after all 10 models finish, so an 8-hour job produces nothing inspectable until the
end. Call this after each model to close that gap.

Usage (per model, from the sbatch loop):
    python upload_helper.py \
        --jsonl           results/Qwen__Qwen2.5-Coder-7B-Instruct.jsonl \
        --hf_dataset      bermaneh/pde-llm-eval-results-jul28 \
        --workspace       /home/ehb7466/pde-llm-eval \
        --job_id          torch:12345 \
        --artifact_status partial

    # or aggregate every model written so far into one repo:
    python upload_helper.py --results_dir results/ --hf_dataset ... --workspace ...
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


COLUMN_DESCRIPTIONS = {
    "title":              "Dataset row identifier, e.g. Wave_Comm_Valid_1",
    "gt_sample":          "Base problem ID, e.g. Wave_1",
    "pde_class":          "Ground truth PDE class (wave/heat/burgers/navier-stokes)",
    "mod_type":           "Modification condition (one of the 8 mod_types)",
    "source":             "Base problem provenance: human or synthetic",
    "gt_pde":             "Ground truth PDE label",
    "gt_method":          "Ground truth numerical method (/-separated if multiple)",
    "gt_behavior":        "Ground truth physical behavior (/-separated if multiple)",
    "gt_valid":           "Ground truth physical validity (True/False)",
    "model_response":     "Full raw model output — never truncated",
    "parsed_pde":         "Parsed pde field from model response",
    "parsed_method":      "Parsed method field from model response",
    "parsed_behavior":    "Parsed behavior field from model response",
    "parsed_valid":       "Parsed valid field from model response",
    "pde_match":          "Binary keyword match for PDE field (0/1)",
    "pde_embed_sim":      "Cosine embedding similarity for PDE field, range [0,1]",
    "method_any_match":   "1 if any ground-truth method token appears in the response",
    "method_recall":      "Fraction of ground-truth method tokens found, range [0,1]",
    "behavior_any_match": "1 if any ground-truth behavior token appears in the response",
    "behavior_recall":    "Fraction of ground-truth behavior tokens found, range [0,1]",
    "valid_match":        "1 if the validity prediction matches ground truth",
    "finish_reason":      "vLLM stop reason — 'length' means the output was truncated",
    "model":              "Model ID used for inference",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",            default=None,
                        help="Single model's JSONL. Mutually exclusive with --results_dir.")
    parser.add_argument("--results_dir",      default=None,
                        help="Aggregate every *.jsonl in this dir (all models so far).")
    parser.add_argument("--hf_dataset",       required=True, help="HF dataset repo id")
    parser.add_argument("--workspace",        required=True,
                        help="Dir containing packages/key_handler and packages/hf_utility")
    parser.add_argument("--experiment",       default="pde-llm-eval")
    parser.add_argument("--artifact_status",  default="partial", choices=["partial", "final"])
    parser.add_argument("--job_id",           default="local:0")
    parser.add_argument("--cluster",          default="torch")
    parser.add_argument("--canary",           default="false")
    parser.add_argument("--dataset_file",     default="data/merged_mod_jul28.csv",
                        help="Recorded in metadata so the artifact names its input dataset")
    args = parser.parse_args()

    if not args.jsonl and not args.results_dir:
        parser.error("one of --jsonl or --results_dir is required")

    is_canary = args.canary.lower() == "true"

    # Inject API keys
    try:
        sys.path.insert(0, str(Path(args.workspace) / "packages" / "key_handler"))
        from key_handler import KeyHandler
        KeyHandler.set_env_key()
    except Exception as e:
        print(f"[upload_helper] key_handler unavailable: {e}", flush=True)

    sys.path.insert(0, str(Path(args.workspace) / "packages" / "hf_utility"))
    from hf_utility import push_dataset_to_hub
    from datasets import Dataset

    if args.results_dir:
        paths = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))
        rows = [r for p in paths for r in load_jsonl(p)]
        print(f"[upload_helper] {len(rows)} rows from {len(paths)} files", flush=True)
    else:
        rows = load_jsonl(args.jsonl)
        print(f"[upload_helper] {len(rows)} rows from {args.jsonl}", flush=True)

    if not rows:
        print("[upload_helper] Nothing to upload — exiting without error.", flush=True)
        sys.exit(0)

    # Surface truncation immediately: a 'length' finish_reason is a failed row, not a datum.
    truncated = [r.get("title") for r in rows if r.get("finish_reason") == "length"]
    if truncated:
        print(f"[upload_helper] WARNING: {len(truncated)} truncated response(s): "
              f"{truncated[:10]}{' …' if len(truncated) > 10 else ''}", flush=True)

    models = sorted({r.get("model", "") for r in rows if r.get("model")})

    push_dataset_to_hub(
        dataset=Dataset.from_list(rows),
        dataset_name=args.hf_dataset.split("/")[-1],
        experiment_slug=args.experiment,
        metadata={
            "script_name":     "run_eval.py",
            "model":           ", ".join(models) if models else "unknown",
            "description":     (f"Free-generation PDE eval — {args.artifact_status}. "
                                f"{len(rows)} rows from {len(models)} model(s) on "
                                f"{args.dataset_file}."),
            "experiment_name": args.experiment,
            "job_id":          args.job_id,
            "cluster":         args.cluster,
            "artifact_status": args.artifact_status,
            "canary":          is_canary,
            "input_datasets":  [args.dataset_file],
            "hyperparameters": {"max_tokens": "per-model", "thinking": "suppressed"},
        },
        tags=[args.experiment, "free-gen", "jul28", args.artifact_status],
        column_descriptions=COLUMN_DESCRIPTIONS,
    )

    print(f"[upload_helper] Done — {len(rows)} rows at {args.hf_dataset} "
          f"({args.artifact_status})", flush=True)


if __name__ == "__main__":
    main()
