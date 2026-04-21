"""
PDE LLM Eval — vLLM inference script.
Runs one model against all rows in pdedata_clean_v2.xlsx.
Resumable: skips (title, model) pairs already in the output JSONL.

Usage:
    python run_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset data/pdedata_clean_v2.xlsx \
        --output_dir results/ \
        --hf_dataset bermaneh/pde-llm-eval-canary-v1 \
        [--canary] [--job_id torch:12345] [--upload_every 20]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import pandas as pd

# Parse/score logic
sys.path.insert(0, str(Path(__file__).parent))
from parse_score import parse_response, score_row

    print(f"[run_eval] key_handler unavailable: {e}", flush=True)

PROMPT_TEMPLATE = """\
You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

Answer the following about this simulation. Be concise.

Output only:
pde: ____
method: ____
behavior: ____
valid: ____

- pde: the type of PDE being solved
- method: numerical method(s) used — list all that apply
- behavior: dominant physical process(es) — list all that apply
- valid: is this a physically valid simulation?\
"""

# Per-model settings
MODEL_CONFIGS = {
    "Qwen/Qwen3-32B":               {"max_tokens": 16384, "thinking": True},
    "Qwen/QwQ-32B":                 {"max_tokens": 16384, "thinking": True},
    # All other models
    "__default__":                  {"max_tokens": 2048, "thinking": False},
}

THINKING_MODELS = {"Qwen/Qwen3-32B", "Qwen/QwQ-32B"}


def get_model_config(model_id: str) -> dict:
    return MODEL_CONFIGS.get(model_id, MODEL_CONFIGS["__default__"])


def load_checkpoint(jsonl_path: str) -> set[tuple]:
    """Return set of (title, mod_type, model) tuples already processed."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done.add((row["title"], row["mod_type"], row["model"]))
    print(f"[run_eval] Checkpoint: {len(done)} rows already done.", flush=True)
    return done


def append_result(jsonl_path: str, result: dict):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def build_messages(code: str) -> list[dict]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(code=code)}]


def init_vllm(model_id: str, tensor_parallel_size: int = 1):
    from vllm import LLM, SamplingParams  # imported here so tests don't need vllm
    cfg = get_model_config(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=32768,
    )
    sampling_params = SamplingParams(
        max_tokens=cfg["max_tokens"],
        temperature=0.0,
    )
    return llm, sampling_params


def run_batch(llm, sampling_params, messages_batch: list[list[dict]],
              model_id: str) -> list[dict]:
    """Run a batch of messages through vLLM. Returns list of {text, finish_reason}."""
    from vllm import SamplingParams

    use_thinking = model_id in THINKING_MODELS
    chat_kwargs = {"enable_thinking": False} if use_thinking else {}

    outputs = llm.chat(
        messages_batch,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_kwargs if chat_kwargs else None,
    )

    results = []
    for out in outputs:
        text          = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        results.append({"text": text, "finish_reason": finish_reason})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          required=True)
    parser.add_argument("--dataset",        default="data/pdedata_clean_v2.xlsx")
    parser.add_argument("--output_dir",     default="results")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--tp",             type=int, default=1,
                        help="tensor_parallel_size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Sanitize model name for filename
    model_slug = args.model.replace("/", "__")
    jsonl_path = os.path.join(args.output_dir, f"{model_slug}.jsonl")

    # Load dataset
    df = pd.read_excel(args.dataset)
    print(f"[run_eval] Dataset: {len(df)} rows", flush=True)

    # Dataset integrity check
    assert len(df) == 96, f"Expected 96 rows, got {len(df)}"
    assert df["mod_type"].value_counts().to_dict() == {
        "Comm_Valid": 16, "NoComm_Valid": 16, "NoComm_InValid": 16,
        "CorrComm": 16, "NoComm_CorrVar": 16, "Comm_InValid": 16,
    }, f"Unexpected mod_type distribution: {df['mod_type'].value_counts().to_dict()}"
    print("[run_eval] Dataset integrity check passed.", flush=True)

    # Load checkpoint
    done = load_checkpoint(jsonl_path)
    # Filter on (title, mod_type, model) — titles repeat across conditions
    todo = df[~df.apply(lambda r: (r["title"], r["mod_type"], args.model) in done, axis=1)].reset_index(drop=True)
    print(f"[run_eval] Rows to process: {len(todo)}", flush=True)

    if todo.empty:
        print("[run_eval] All rows already processed.", flush=True)
        return

    # Init vLLM
    print(f"[run_eval] Loading model: {args.model}", flush=True)
    llm, sampling_params = init_vllm(args.model, args.tp)

    # Try loading sentence-transformers for embedding similarity
    embed_model = None
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[run_eval] Embedding model loaded.", flush=True)
    except Exception as e:
        print(f"[run_eval] Embedding model unavailable (will skip embed_sim): {e}", flush=True)

    new_rows  = 0
    batch_buf = []  # list of (row_dict, messages)

    def flush_batch():
        nonlocal new_rows
        if not batch_buf:
            return
        msgs_batch = [m for _, m in batch_buf]
        outputs    = run_batch(llm, sampling_params, msgs_batch, args.model)

        for (row, _), out in zip(batch_buf, outputs):
            text          = out["text"]
            finish_reason = out["finish_reason"]

            if finish_reason == "length":
                print(f"[run_eval] WARNING: truncation on {row['title']}", flush=True)

            parsed = parse_response(text)
            scores = score_row(parsed, row, embed_model)

            result = {
                "title":              row["title"],
                "pde_class":          row["pde_class"],
                "mod_type":           row["mod_type"],
                "gt_pde":             row["pde_class"],
                "gt_method":          str(row["num_method"]),
                "gt_behavior":        str(row["phys_process"]),
                "gt_valid":           bool(row["phys_valid"]),
                "model_response":     text,   # FULL — never truncated
                "parsed_pde":         parsed.get("pde"),
                "parsed_method":      parsed.get("method"),
                "parsed_behavior":    parsed.get("behavior"),
                "parsed_valid":       parsed.get("valid"),
                "finish_reason":      finish_reason,
                "model":              args.model,
                **scores,
            }
            append_result(jsonl_path, result)
            new_rows += 1

        batch_buf.clear()

    # Main loop
    for _, row in todo.iterrows():
        if (row["title"], row["mod_type"], args.model) in done:
            continue
        messages = build_messages(str(row["code"]))
        batch_buf.append((row.to_dict(), messages))
        if len(batch_buf) >= args.batch_size:
            flush_batch()

    flush_batch()  # remainder

    print(f"[run_eval] Done. {new_rows} new rows written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
