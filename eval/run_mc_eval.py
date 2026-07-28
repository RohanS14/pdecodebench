"""
PDE MC Eval — vLLM inference script.
Runs one model against all rows × all question types in pdedata_clean_v3.xlsx.
Resumable: skips (title, mod_type) pairs already fully written to output JSONL.

Each (title, mod_type) pair generates 9 question rows atomically:
  1 × pde_class (4-way MC) + 4 × phys_process (T/F) + 3 × num_method (T/F) + 1 × phys_valid (T/F)

Two scoring modes are used depending on the model:
  - Logprob mode (default): max_tokens=1, extract answer from first-token log-probabilities.
  - Text extraction mode (QwQ-32B, DeepSeek-R1-Distill-Qwen-32B): max_tokens=8192, strip
    <think> blocks, then parse the answer letter from the generated text. Logprob fields
    will be null; scoring_method="text_extraction" is set on each result row.

Usage:
    python run_mc_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset data/pdedata_clean_v3.xlsx \
        --output_dir results_mc/ \
        --batch_size 8 \
        --tp 1
"""
import argparse, json, os, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mc_questions import (
    make_all_questions,
    row_seed,
    extract_letter_logprobs,
    build_result_row,
)

# Logprob extraction fails for these models because thinking tokens bleed into the
# first generated position when max_tokens=1. Use text generation + letter parsing instead.
TEXT_EXTRACTION_MODELS = {
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}
# Qwen3-32B supports enable_thinking=False and works correctly in logprob mode.
THINKING_MODELS = {"Qwen/Qwen3-32B"}
QUESTIONS_PER_ROW = 9  # 1 + 4 + 3 + 1


def extract_answer_letter(text: str, applicable_letters: list[str]) -> str | None:
    """Parse an answer letter from free-form model output.

    Strips <think>…</think> blocks first, then tries a cascade of patterns from
    most- to least-specific.  Returns the first valid letter found, or None.
    """
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        clean = text.strip()
    valid = set(applicable_letters)
    for pat in [
        r"(?i)(?:answer\s*(?:is|:)?\s*)[*]*([A-D])[*]*",
        r"(?i)(?:correct\s+(?:answer|option|choice)\s*(?:is|:)?\s*)[*]*([A-D])[*]*",
        r"(?i)^[*]*([A-D])[*]*[.):]\s*(?:Yes|No|burgers|heat|wave|navier)",
        r"(?m)^\s*[*]*([A-D])[*]*\s*$",
        r"\b([A-D])\b",
    ]:
        for m in re.finditer(pat, clean):
            letter = m.group(1).upper()
            if letter in valid:
                return letter
    return None


def load_checkpoint(jsonl_path: str, model: str, question_types: set | None = None) -> set[tuple]:
    """Return set of (title, mod_type) pairs fully processed.

    If question_types is given, a pair is 'done' only when all requested
    question types are present for that pair.  Otherwise uses count-based check.
    """
    if not os.path.exists(jsonl_path):
        print(f"[run_mc_eval] Checkpoint: 0 pairs done (no file).", flush=True)
        return set()

    if question_types:
        present: dict[tuple, set] = {}
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("model") == model and row.get("question_type") in question_types:
                    key = (row["title"], row["mod_type"])
                    present.setdefault(key, set()).add(row["question_type"])
        done = {key for key, types in present.items() if types >= question_types}
    else:
        counts: dict[tuple, int] = {}
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
    text_mode = model_id in TEXT_EXTRACTION_MODELS
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        **({"max_model_len": 32768, "enforce_eager": True, "disable_log_stats": True}
           if text_mode else {}),
    )
    if text_mode:
        sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
    else:
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=10)
    return llm, sampling_params


def run_batch(llm, sampling_params, messages_batch: list[list[dict]], model_id: str) -> list[dict]:
    """Run a batch through vLLM. Returns list of {text, finish_reason, logprobs}."""
    if model_id in THINKING_MODELS:
        outputs = llm.chat(messages_batch, sampling_params=sampling_params,
                           chat_template_kwargs={"enable_thinking": False})
    else:
        outputs = llm.chat(messages_batch, sampling_params=sampling_params)
    return [
        {"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
         "logprobs": getattr(o.outputs[0], "logprobs", None)}
        for o in outputs
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          required=True)
    parser.add_argument("--dataset",        default="data/pdedata_clean_v3.xlsx")
    parser.add_argument("--output_dir",     default="results_mc")
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Number of prompts per vLLM batch")
    parser.add_argument("--tp",             type=int, default=1)
    parser.add_argument("--question_types", default="",
                        help="Comma-separated question types to run (default: all 9). "
                             "E.g. 'phys_valid' or 'phys_valid,pde_class'")
    args = parser.parse_args()

    filter_types: set | None = (
        {t.strip() for t in args.question_types.split(",") if t.strip()}
        if args.question_types else None
    )
    n_questions = len(filter_types) if filter_types else QUESTIONS_PER_ROW

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.replace("/", "__")
    jsonl_path = os.path.join(args.output_dir, f"{model_slug}.jsonl")

    df = pd.read_excel(args.dataset)
    print(f"[run_mc_eval] Dataset: {len(df)} rows", flush=True)
    V2_DIST = {"Comm_Valid": 16, "NoComm_Valid": 16, "NoComm_InValid": 16,
               "CorrComm": 16, "NoComm_CorrVar": 16, "Comm_InValid": 16}
    V3_DIST = {**V2_DIST, "CorrComm_Invalid": 16, "NoComm_CorrVar_InValid": 16}
    actual_dist = df["mod_type"].value_counts().to_dict()
    assert actual_dist in (V2_DIST, V3_DIST), \
        f"Unexpected mod_type distribution: {actual_dist}"
    print(f"[run_mc_eval] Dataset integrity check passed ({len(df)} rows, {len(actual_dist)} conditions).", flush=True)

    if filter_types:
        print(f"[run_mc_eval] Running only: {sorted(filter_types)}", flush=True)

    done_titles = load_checkpoint(jsonl_path, args.model, filter_types)
    todo = df[~df.apply(lambda r: (r["title"], r["mod_type"]) in done_titles, axis=1)].reset_index(drop=True)
    print(f"[run_mc_eval] Rows to process: {len(todo)}", flush=True)

    if todo.empty:
        print("[run_mc_eval] All titles already processed.", flush=True)
        return

    print(f"[run_mc_eval] Loading model: {args.model}", flush=True)
    llm, sampling_params = init_vllm(args.model, args.tp)

    text_mode = args.model in TEXT_EXTRACTION_MODELS
    titles_done = 0
    batch_buf: list[tuple[dict, dict, list]] = []

    def flush_batch():
        nonlocal titles_done
        if not batch_buf:
            return
        outputs = run_batch(llm, sampling_params, [m for _, _, m in batch_buf], args.model)
        by_title: dict[str, list] = {}
        for (row, question, _), out in zip(batch_buf, outputs):
            by_title.setdefault(row["title"], []).append((row, question, out))

        for title, items in by_title.items():
            result_rows = []
            for row, question, out in items:
                text, finish_reason = out["text"], out["finish_reason"]
                if finish_reason == "length":
                    print(f"[run_mc_eval] WARNING: length finish on {title} {question['question_type']}", flush=True)

                if text_mode:
                    predicted = extract_answer_letter(text, question["applicable_letters"])
                    if predicted is None:
                        print(f"[run_mc_eval] WARNING: no letter found for {title}/{question['question_type']}, "
                              f"text={repr(text[:120])}", flush=True)
                    null_lps = {"A": None, "B": None, "C": None, "D": None}
                    result = build_result_row(row, args.model, question, null_lps, finish_reason)
                    result["predicted_letter"] = predicted
                    result["correct"] = (predicted == question["correct_letter"]) if predicted else None
                    result["scoring_method"] = "text_extraction"
                else:
                    letter_logprobs = extract_letter_logprobs(out["logprobs"])
                    if all(v is None for v in letter_logprobs.values()):
                        print(f"[run_mc_eval] WARNING: all letter logprobs None on {title} / {question['question_type']}", flush=True)
                    thinking_leak = text.strip() not in {"A", "B", "C", "D"}
                    if thinking_leak:
                        print(f"[run_mc_eval] WARNING: thinking leak on {title} / {question['question_type']}", flush=True)
                    result = build_result_row(row, args.model, question, letter_logprobs, finish_reason)
                    if thinking_leak:
                        result["predicted_letter"] = None
                        result["correct"] = None

                result_rows.append(result)
            append_results(jsonl_path, result_rows)
            titles_done += 1
        batch_buf.clear()

    for _, row in todo.iterrows():
        row_dict = row.to_dict()
        if (row_dict["title"], row_dict["mod_type"]) in done_titles:
            continue
        seed = row_seed(row_dict["title"], args.model)
        questions = make_all_questions(row_dict, seed)
        if filter_types:
            questions = [q for q in questions if q["question_type"] in filter_types]
        for q in questions:
            if args.model in THINKING_MODELS or args.model == "Qwen/QwQ-32B":
                # Qwen3 (logprob) and QwQ (text) both benefit from /no_think
                messages = [{"role": "system", "content": "/no_think"},
                            {"role": "user", "content": q["prompt"]}]
            else:
                messages = [{"role": "user", "content": q["prompt"]}]
            batch_buf.append((row_dict, q, messages))
        if len(batch_buf) >= args.batch_size * n_questions:
            flush_batch()

    flush_batch()
    print(f"[run_mc_eval] Done. {titles_done} titles → {titles_done * n_questions} rows written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
