"""
Variable log-probability evolution eval — vLLM prompt-scoring script.

For each probe point in probe_points.jsonl, scores:
  logP_corrupt = P(corrupt_var | prefix)   [the foobar_N name as it appears in code]
  logP_gt      = P(gt_var      | prefix)   [the ground-truth meaningful name]
  logP_diff    = logP_gt - logP_corrupt

Uses vLLM prompt_logprobs for raw code completion (no chat template).
Resumable: skips (title, corrupt_var, occurrence_idx, model) tuples in output JSONL.

Usage:
    python scripts/run_var_logprob.py \
        --model  Qwen/Qwen2.5-Coder-7B-Instruct \
        --probes data/probe_points.jsonl \
        --output_dir results/ \
        --hf_dataset bermaneh/pde-var-logprob-evolution-canary-v1 \
        [--tp 1] [--batch_size 32] [--upload_every 100]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from var_probe_utils import extract_continuation_logprob

# Inject API keys via RACA key_handler (repo lives at raca/packages/key_handler)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "key_handler"))
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except Exception as e:
    print(f"[run_var_logprob] key_handler unavailable: {e}", flush=True)


CHECKPOINT_KEY = ("title", "mod_type", "corrupt_var", "occurrence_idx")


def load_checkpoint(jsonl_path: str, model: str) -> set[tuple]:
    """Return set of checkpoint keys already processed for this model."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model") == model:
                key = tuple(row.get(k) for k in CHECKPOINT_KEY)
                done.add(key)
    print(f"[run_var_logprob] Checkpoint: {len(done)} probe points done.", flush=True)
    return done


def append_results(jsonl_path: str, results: list[dict]):
    with open(jsonl_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def init_vllm(model_id: str, tp: int):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        max_model_len=16384,
    )
    # max_tokens=1 is a formality — we only look at prompt_logprobs.
    # prompt_logprobs=1 guarantees the actual token at each position is included.
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )
    return llm, sampling_params


def build_scoring_texts(
    tokenizer,
    probe: dict,
) -> tuple[str, str, int, list[int], int, list[int]]:
    """
    Build (corrupt_text, gt_text) and their prefix/full token counts.

    Uses offset_mapping to find the token boundary rather than len(prefix_ids),
    which is wrong when BPE merges the last prefix character(s) with the first
    variable character(s) into a token that didn't appear in the prefix alone.

    Returns:
      corrupt_text, gt_text,
      prefix_count_corrupt, full_ids_corrupt,
      prefix_count_gt, full_ids_gt
    """
    prefix = probe["prefix"]
    corrupt_var = probe["corrupt_var"]
    gt_var = probe["gt_var"]

    corrupt_text = prefix + corrupt_var
    gt_text      = prefix + gt_var
    prefix_len   = len(prefix)

    def encode_with_boundary(text: str) -> tuple[int, list[int]]:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids     = enc["input_ids"]
        offsets = enc["offset_mapping"]
        # First token whose char-end extends past the prefix boundary.
        # Using end > prefix_len (not start >= prefix_len) handles the case where
        # BPE merges the last prefix character(s) with the first variable character(s)
        # into a single spanning token — that token must be counted as continuation.
        prefix_count = next(
            (i for i, (_, end) in enumerate(offsets) if end > prefix_len),
            len(ids),
        )
        return prefix_count, ids

    prefix_count_corrupt, corrupt_full_ids = encode_with_boundary(corrupt_text)
    prefix_count_gt,      gt_full_ids      = encode_with_boundary(gt_text)

    return (
        corrupt_text, gt_text,
        prefix_count_corrupt, corrupt_full_ids,
        prefix_count_gt, gt_full_ids,
    )


def run_batch(
    llm,
    sampling_params,
    texts: list[str],
) -> list:
    """Run a batch of texts through vLLM. Returns list of outputs."""
    return llm.generate(texts, sampling_params=sampling_params)


def score_probe(
    output,               # vLLM RequestOutput
    prefix_token_count: int,
    full_token_ids: list[int],
) -> float | None:
    """Extract summed logprob for the continuation tokens from prompt_logprobs."""
    return extract_continuation_logprob(
        output.prompt_logprobs,
        prefix_token_count,
        full_token_ids,
    )


def upload_results(jsonl_path: str, hf_dataset: str, model: str,
                   job_id: str, is_canary: bool, workspace: str,
                   results_dir: str | None = None):
    """Upload current JSONL to HF in a separate subprocess (never in vLLM process)."""
    upload_script = Path(__file__).parent / "upload_helper_var.py"
    cmd = [
        sys.executable, str(upload_script),
        "--jsonl",       jsonl_path,
        "--hf_dataset",  hf_dataset,
        "--model",       model,
        "--job_id",      job_id,
        "--canary",      str(is_canary).lower(),
        "--workspace",   workspace,
    ]
    if results_dir:
        cmd += ["--results_dir", results_dir]
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True)
    parser.add_argument("--probes",       default="data/probe_points.jsonl")
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--hf_dataset",   required=True)
    parser.add_argument("--tp",           type=int, default=1)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--upload_every", type=int, default=100,
                        help="Upload after every N probe points written")
    parser.add_argument("--job_id",       default="local:0")
    parser.add_argument("--canary",       action="store_true")
    args = parser.parse_args()

    workspace = str(Path(__file__).resolve().parents[1])
    os.makedirs(args.output_dir, exist_ok=True)

    model_slug = args.model.replace("/", "__")
    jsonl_path = os.path.join(args.output_dir, f"{model_slug}.jsonl")

    # Load probes
    probes = []
    with open(args.probes) as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))
    print(f"[run_var_logprob] Loaded {len(probes)} probe points.", flush=True)

    # Checkpoint
    done = load_checkpoint(jsonl_path, args.model)
    todo = [
        p for p in probes
        if tuple(p.get(k) for k in CHECKPOINT_KEY) not in done
    ]
    print(f"[run_var_logprob] Probes remaining: {len(todo)}", flush=True)

    if not todo:
        print("[run_var_logprob] All done.", flush=True)
        return

    # Init vLLM and tokenizer
    print(f"[run_var_logprob] Loading model: {args.model}", flush=True)
    llm, sampling_params = init_vllm(args.model, args.tp)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    written_since_upload = 0

    # Process in batches of 2× batch_size (corrupt + gt for each probe)
    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start: batch_start + args.batch_size]

        # Build scoring texts for each probe in the batch
        corrupt_texts: list[str] = []
        gt_texts: list[str]      = []
        meta: list[dict]         = []  # per-probe bookkeeping

        for probe in batch:
            (c_text, g_text,
             pc_corrupt, full_corrupt,
             pc_gt, full_gt) = build_scoring_texts(tokenizer, probe)

            corrupt_texts.append(c_text)
            gt_texts.append(g_text)
            meta.append({
                "probe":          probe,
                "pc_corrupt":     pc_corrupt,
                "full_corrupt":   full_corrupt,
                "pc_gt":          pc_gt,
                "full_gt":        full_gt,
            })

        # Single batched forward pass for corrupt continuations, then gt
        # (interleaving would require 2× requests anyway; sequential is simpler)
        corrupt_outputs = run_batch(llm, sampling_params, corrupt_texts)
        gt_outputs      = run_batch(llm, sampling_params, gt_texts)

        results = []
        for m, c_out, g_out in zip(meta, corrupt_outputs, gt_outputs):
            probe    = m["probe"]
            logP_c   = score_probe(c_out, m["pc_corrupt"], m["full_corrupt"])
            logP_g   = score_probe(g_out, m["pc_gt"],      m["full_gt"])

            n_tok_corrupt = len(m["full_corrupt"]) - m["pc_corrupt"]
            n_tok_gt      = len(m["full_gt"])      - m["pc_gt"]

            logP_diff = (logP_g - logP_c) if (logP_g is not None and logP_c is not None) else None

            results.append({
                # Identity
                "title":          probe["title"],
                "mod_type":       probe["mod_type"],
                "pde_class":      probe["pde_class"],
                "model":          args.model,
                # Variable info
                "corrupt_var":    probe["corrupt_var"],
                "gt_var":         probe["gt_var"],
                "var_class":      probe["var_class"],
                "occurrence_idx": probe["occurrence_idx"],
                "probe_idx":      probe["probe_idx"],
                "total_probes":   probe["total_probes"],
                # Position
                "char_offset":    probe["char_offset"],
                "line_number":    probe["line_number"],
                "code_fraction":  probe["code_fraction"],
                # Logprobs
                "logP_corrupt":   logP_c,
                "logP_gt":        logP_g,
                "logP_diff":      logP_diff,
                "n_tok_corrupt":  n_tok_corrupt,
                "n_tok_gt":       n_tok_gt,
                # Null flags
                "corrupt_null":   logP_c is None,
                "gt_null":        logP_g is None,
            })

            if logP_c is None or logP_g is None:
                print(
                    f"[run_var_logprob] WARNING: null logprob on "
                    f"{probe['title']} {probe['corrupt_var']} occ={probe['occurrence_idx']} "
                    f"(corrupt_null={logP_c is None}, gt_null={logP_g is None})",
                    flush=True,
                )

        append_results(jsonl_path, results)
        written_since_upload += len(results)

        processed = batch_start + len(batch)
        print(
            f"[run_var_logprob] {processed}/{len(todo)} probes done "
            f"({args.model})",
            flush=True,
        )

        if written_since_upload >= args.upload_every:
            upload_results(jsonl_path, args.hf_dataset, args.model,
                           args.job_id, args.canary, workspace,
                           results_dir=None)  # mid-run: only this model's partial
            written_since_upload = 0

    # Final upload — aggregate all models completed so far
    upload_results(jsonl_path, args.hf_dataset, args.model,
                   args.job_id, args.canary, workspace,
                   results_dir=args.output_dir)
    print(f"[run_var_logprob] Model {args.model} complete.", flush=True)


if __name__ == "__main__":
    main()
