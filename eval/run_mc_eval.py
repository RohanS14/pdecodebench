"""
PDE MC Logprob Eval — vLLM inference script.
Runs one model against all rows × all question types in pdedata_clean_v2.xlsx.
Resumable: skips (title, model) pairs already fully written to output JSONL.

Each (title, model) pair generates 9 question rows atomically:
  1 × pde_class (4-way MC) + 4 × phys_process (T/F) + 3 × num_method (T/F) + 1 × phys_valid (T/F)

Usage:
    python run_mc_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset data/pdedata_clean_v2.xlsx \
        --output_dir results_mc/ \
        --hf_dataset bermaneh/pde-mc-logprob-canary-v1 \
        [--canary] [--job_id torch:12345] [--upload_every 20]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mc_questions import (
    make_all_questions,
    row_seed,
    extract_letter_logprobs,
    build_result_row,
)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "key_handler"))
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except Exception as e:
    print(f"[run_mc_eval] key_handler unavailable: {e}", flush=True)

# Per-model settings (same as run_eval.py)
THINKING_MODELS = {"Qwen/Qwen3-32B", "Qwen/QwQ-32B"}

QUESTIONS_PER_ROW = 9  # 1 + 4 + 3 + 1


def load_checkpoint(jsonl_path: str, model: str) -> set[tuple]:
    """Return set of (title, mod_type) pairs fully processed for this model (all 9 questions present)."""
    counts: dict[tuple, int] = {}
    if not os.path.exists(jsonl_path):
        return set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model") == model:
                key = (row["title"], row["mod_type"])
                counts[key] = counts.get(key, 0) + 1
    done = {key for key, count in counts.items() if count >= QUESTIONS_PER_ROW}
    print(f"[run_mc_eval] Checkpoint: {len(done)} (title, mod_type) pairs fully done.", flush=True)
    return done


def append_results(jsonl_path: str, results: list[dict]):
    with open(jsonl_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def init_vllm(model_id: str, tensor_parallel_size: int = 1):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=16384,
    )
    # logprobs=10: wide enough to capture A/B/C/D and their space-prefixed variants
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=10,
    )
    return llm, sampling_params


def run_batch(llm, sampling_params, messages_batch: list[list[dict]], model_id: str) -> list[dict]:
    """Run a batch through vLLM. Returns list of {text, finish_reason, logprobs}."""
    use_thinking = model_id in THINKING_MODELS
    chat_kwargs = {"enable_thinking": False} if use_thinking else {}

    if chat_kwargs:
        outputs = llm.chat(messages_batch, sampling_params=sampling_params,
                           chat_template_kwargs=chat_kwargs)
    else:
        outputs = llm.chat(messages_batch, sampling_params=sampling_params)

    results = []
    for out in outputs:
        o = out.outputs[0]
        results.append({
            "text":          o.text,
            "finish_reason": o.finish_reason,
            "logprobs":      o.logprobs,   # list[dict[int, Logprob]] — one per token
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True)
    parser.add_argument("--dataset",      default="data/pdedata_clean_v2.xlsx")
    parser.add_argument("--output_dir",   default="results_mc")
    parser.add_argument("--batch_size",   type=int, default=8,
                        help="Number of (title × question) prompts per vLLM batch")
    parser.add_argument("--tp",           type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_slug = args.model.replace("/", "__")
    jsonl_path = os.path.join(args.output_dir, f"{model_slug}.jsonl")

    # Load dataset
    df = pd.read_excel(args.dataset)
    print(f"[run_mc_eval] Dataset: {len(df)} rows", flush=True)
    assert len(df) == 96, f"Expected 96 rows, got {len(df)}"
    assert df["mod_type"].value_counts().to_dict() == {
        "Comm_Valid": 16, "NoComm_Valid": 16, "NoComm_InValid": 16,
        "CorrComm": 16, "NoComm_CorrVar": 16, "Comm_InValid": 16,
    }, f"Unexpected mod_type distribution: {df['mod_type'].value_counts().to_dict()}"
    print("[run_mc_eval] Dataset integrity check passed.", flush=True)

    # Load checkpoint
    done_titles = load_checkpoint(jsonl_path, args.model)
    todo = df[~df.apply(lambda r: (r["title"], r["mod_type"]) in done_titles, axis=1)].reset_index(drop=True)
    print(f"[run_mc_eval] Rows to process: {len(todo)}", flush=True)

    if todo.empty:
        print("[run_mc_eval] All titles already processed.", flush=True)
        return

    # Init vLLM
    print(f"[run_mc_eval] Loading model: {args.model}", flush=True)
    llm, sampling_params = init_vllm(args.model, args.tp)

    titles_done = 0
    # batch_buf: list of (row_dict, question_dict, messages)
    batch_buf: list[tuple[dict, dict, list]] = []

    def flush_batch():
        nonlocal titles_done
        if not batch_buf:
            return

        msgs_batch = [m for _, _, m in batch_buf]
        outputs = run_batch(llm, sampling_params, msgs_batch, args.model)

        # Group by title to write all 9 questions per title atomically
        by_title: dict[str, list[tuple[dict, dict, dict]]] = {}
        for (row, question, _), out in zip(batch_buf, outputs):
            title = row["title"]
            if title not in by_title:
                by_title[title] = []
            by_title[title].append((row, question, out))

        for title, items in by_title.items():
            result_rows = []
            for row, question, out in items:
                text          = out["text"]
                finish_reason = out["finish_reason"]
                vllm_logprobs = out["logprobs"]

                if finish_reason == "length":
                    print(f"[run_mc_eval] WARNING: length finish on {title} {question['question_type']}", flush=True)

                letter_logprobs = extract_letter_logprobs(vllm_logprobs)

                # Warn if all logprobs are None (tokenizer edge case or empty output)
                if all(v is None for v in letter_logprobs.values()):
                    print(
                        f"[run_mc_eval] WARNING: all letter logprobs are None on "
                        f"{title} / {question['question_type']} — check tokenizer",
                        flush=True,
                    )

                # Detect thinking leak: if first emitted token isn't a letter,
                # null out predicted_letter so the row isn't scored as correct/wrong
                raw_token = text.strip()
                thinking_leak = raw_token not in {"A", "B", "C", "D"}
                if thinking_leak:
                    print(
                        f"[run_mc_eval] WARNING: unexpected first token '{raw_token}' "
                        f"on {title} / {question['question_type']} — thinking leak, "
                        f"predicted_letter set to None",
                        flush=True,
                    )

                result = build_result_row(row, args.model, question, letter_logprobs, finish_reason)
                if thinking_leak:
                    result["predicted_letter"] = None
                    result["correct"] = None
                result_rows.append(result)

            # Write all 9 rows for this title atomically
            append_results(jsonl_path, result_rows)
            titles_done += 1

        batch_buf.clear()

    # Main loop — queue one title's worth of prompts at a time
    for _, row in todo.iterrows():
        row_dict = row.to_dict()
        title    = row_dict["title"]

        if (title, row_dict["mod_type"]) in done_titles:
            continue

        seed      = row_seed(title, args.model)
        questions = make_all_questions(row_dict, seed)

        for q in questions:
            messages = ([{"role": "system", "content": "/no_think"}, {"role": "user", "content": q["prompt"]}]
                        if args.model in THINKING_MODELS else
                        [{"role": "user", "content": q["prompt"]}])
            batch_buf.append((row_dict, q, messages))

        # Flush when batch is large enough (measured in prompts)
        if len(batch_buf) >= args.batch_size * QUESTIONS_PER_ROW:
            flush_batch()

    flush_batch()  # remainder

    total_rows = titles_done * QUESTIONS_PER_ROW
    print(f"[run_mc_eval] Done. {titles_done} titles → {total_rows} rows written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
