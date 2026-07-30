"""
Upload helper — runs in a separate subprocess so HF imports never touch the vLLM process.

Usage (called by run_var_logprob.py via subprocess):
    python upload_helper.py \
        --jsonl      results/Qwen__Qwen2.5-Coder-7B-Instruct.jsonl \
        --hf_dataset bermaneh/pde-var-logprob-evolution-canary-v1 \
        --model      Qwen/Qwen2.5-Coder-7B-Instruct \
        --job_id     torch:12345 \
        --canary     true \
        --workspace  /scratch/ehb7466/pde-llm-eval
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",       required=True)
    parser.add_argument("--hf_dataset", required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--job_id",     default="local:0")
    parser.add_argument("--canary",     default="false")
    parser.add_argument("--workspace",  required=True)
    parser.add_argument("--results_dir", default=None,
                        help="If set, aggregate ALL *.jsonl files in this dir (all models)")
    args = parser.parse_args()

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

    # Collect rows: either from all JSONLs in results_dir (multi-model) or just --jsonl.
    rows = []
    if args.results_dir:
        jsonl_files = sorted(Path(args.results_dir).glob("*.jsonl"))
        if not jsonl_files:
            print(f"[upload_helper] No *.jsonl files in {args.results_dir} — skipping.", flush=True)
            return
        for jf in jsonl_files:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        print(f"[upload_helper] Aggregated {len(rows)} rows from {len(jsonl_files)} files in {args.results_dir}", flush=True)
    else:
        if not Path(args.jsonl).exists():
            print(f"[upload_helper] No file at {args.jsonl} — skipping.", flush=True)
            return
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not rows:
        print("[upload_helper] 0 rows — skipping.", flush=True)
        return

    print(f"[upload_helper] Uploading {len(rows)} rows to {args.hf_dataset}", flush=True)
    ds = Dataset.from_list(rows)

    experiment_slug = "pde-var-logprob-evolution"
    push_dataset_to_hub(
        dataset=ds,
        dataset_name=args.hf_dataset.split("/")[-1],
        metadata={
            "script_name":      "run_var_logprob.py",
            "model":            args.model,
            "description":      (
                "Variable log-probability evolution: logP(gt_var | prefix) vs "
                "logP(corrupt_var | prefix) at each variable occurrence in "
                "NoComm_CorrVar and NoComm_Valid PDE code snippets."
            ),
            "experiment_name":  experiment_slug,
            "job_id":           args.job_id,
            "cluster":          args.job_id.split(":")[0],
            "artifact_status":  "partial" if is_canary else "final",
            "canary":           is_canary,
            "hyperparameters":  {
                "prompt_logprobs": 1,
                "max_tokens":      1,
                "temperature":     0.0,
                "chat_template":   False,
            },
        },
        column_descriptions={
            "title":          "Code snippet identifier",
            "mod_type":       "Condition (NoComm_CorrVar or NoComm_Valid)",
            "pde_class":      "PDE class of the snippet (burgers, heat, etc.)",
            "model":          "Model ID",
            "corrupt_var":    "Variable name as it appears in this code (foobar_N or gt_name)",
            "gt_var":         "Ground-truth meaningful variable name",
            "var_class":      "Variable classification: pde_state or pde_param",
            "occurrence_idx": "0-indexed occurrence of this variable within the snippet",
            "probe_idx":      "Global probe index within this snippet (code order)",
            "total_probes":   "Total probe points in this snippet",
            "char_offset":    "Character offset of variable name in code",
            "line_number":    "1-indexed line number of variable occurrence",
            "code_fraction":  "char_offset / len(code), normalized position [0, 1]",
            "logP_corrupt":   "Sum log-prob of corrupt_var tokens given code prefix",
            "logP_gt":        "Sum log-prob of gt_var tokens given code prefix",
            "logP_diff":      "logP_gt - logP_corrupt (positive = model prefers gt name)",
            "n_tok_corrupt":  "Number of tokens in corrupt_var",
            "n_tok_gt":       "Number of tokens in gt_var",
            "corrupt_null":   "True if logP_corrupt could not be computed",
            "gt_null":        "True if logP_gt could not be computed",
        },
        tags=[experiment_slug, "logprob", "code-understanding", "variable-names"],
    )
    print(f"[upload_helper] Done. {len(rows)} rows uploaded.", flush=True)


if __name__ == "__main__":
    main()
