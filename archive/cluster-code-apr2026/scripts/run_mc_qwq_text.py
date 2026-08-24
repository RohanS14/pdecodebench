"""
QwQ-32B MC eval using text generation + letter extraction.
Used instead of run_mc_eval.py for QwQ because vLLM ignores
enable_thinking=False with max_tokens=1, causing thinking tokens
to bleed into the first position (no A/B/C/D in top-10 logprobs).

Strategy: generate up to 1024 tokens, strip <think>...</think> blocks,
then extract the answer letter from the remaining text.
Logprob fields (logprob_A/B/C/D, margin, entropy) will be null.
predicted_letter and correct are filled from the parsed text.

Usage:
    python run_mc_qwq_text.py \
        --dataset data/pdedata_clean_v2.xlsx \
        --output_dir results_mc
"""
import argparse, json, os, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from mc_questions import (
    make_all_questions,
    row_seed,
    build_result_row,
)

MODEL_ID = "Qwen/QwQ-32B"
QUESTIONS_PER_ROW = 9


def load_checkpoint(jsonl_path: str, question_types: set | None = None) -> set[tuple]:
    """Return set of (title, mod_type) pairs fully processed."""
    if not os.path.exists(jsonl_path):
        print(f"[run_mc_qwq_text] Checkpoint: 0 pairs done (no file).", flush=True)
        return set()

    if question_types:
        present: dict[tuple, set] = {}
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("model") == MODEL_ID and row.get("question_type") in question_types:
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
                if row.get("model") == MODEL_ID:
                    key = (row["title"], row["mod_type"])
                    counts[key] = counts.get(key, 0) + 1
        done = {key for key, count in counts.items() if count >= QUESTIONS_PER_ROW}

    print(f"[run_mc_qwq_text] Checkpoint: {len(done)} (title, mod_type) pairs done.", flush=True)
    return done


def append_results(jsonl_path: str, results: list[dict]):
    with open(jsonl_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def extract_answer_letter(text: str, applicable_letters: list[str]) -> str | None:
    """
    Extract the answer letter (A/B/C/D) from model output.
    Strips <think>...</think> blocks first, then searches for a bare letter.
    Returns the matched letter or None.
    """
    # 1. Strip thinking blocks
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        clean = text.strip()

    valid = set(applicable_letters)

    # 2. Explicit answer patterns (highest priority)
    for pat in [
        r"(?i)(?:answer\s*(?:is|:)?\s*)[*]*([A-D])[*]*",
        r"(?i)(?:correct\s+(?:answer|option|choice)\s*(?:is|:)?\s*)[*]*([A-D])[*]*",
        r"(?i)^[*]*([A-D])[*]*[.):]\s*(?:Yes|No|burgers|heat|wave|navier)",
        r"(?m)^\s*[*]*([A-D])[*]*\s*$",   # bare letter on its own line
        r"\b([A-D])\b",                     # any bare letter
    ]:
        for m in re.finditer(pat, clean):
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    return None


def init_vllm():
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=32768,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        max_tokens=8192,  # QwQ thinking traces can exceed 1024; generous ceiling to avoid truncation
        temperature=0.0,
    )
    return llm, sampling_params


def run_batch(llm, sampling_params, messages_batch: list[list[dict]]) -> list[dict]:
    outputs = llm.chat(
        messages_batch,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    results = []
    for out in outputs:
        o = out.outputs[0]
        results.append({"text": o.text, "finish_reason": o.finish_reason})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="data/pdedata_clean_v3.xlsx")
    parser.add_argument("--output_dir",     default="results_mc")
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--question_types", default="",
                        help="Comma-separated question types to run (default: all 9). "
                             "E.g. 'phys_valid'")
    args = parser.parse_args()

    filter_types: set | None = (
        {t.strip() for t in args.question_types.split(",") if t.strip()}
        if args.question_types else None
    )
    n_questions = len(filter_types) if filter_types else QUESTIONS_PER_ROW

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = MODEL_ID.replace("/", "__")
    jsonl_path = os.path.join(args.output_dir, f"{model_slug}.jsonl")

    df = pd.read_excel(args.dataset)
    print(f"[run_mc_qwq_text] Dataset: {len(df)} rows", flush=True)
    V2_DIST = {"Comm_Valid": 16, "NoComm_Valid": 16, "NoComm_InValid": 16,
               "CorrComm": 16, "NoComm_CorrVar": 16, "Comm_InValid": 16}
    V3_DIST = {**V2_DIST, "CorrComm_Invalid": 16, "NoComm_CorrVar_InValid": 16}
    actual_dist = df["mod_type"].value_counts().to_dict()
    assert actual_dist in (V2_DIST, V3_DIST), f"Unexpected mod_type distribution: {actual_dist}"

    if filter_types:
        print(f"[run_mc_qwq_text] Running only: {sorted(filter_types)}", flush=True)

    done_titles = load_checkpoint(jsonl_path, filter_types)
    todo = df[~df.apply(lambda r: (r["title"], r["mod_type"]) in done_titles, axis=1)].reset_index(drop=True)
    print(f"[run_mc_qwq_text] Rows to process: {len(todo)}", flush=True)

    if todo.empty:
        print("[run_mc_qwq_text] All done.", flush=True)
        return

    print(f"[run_mc_qwq_text] Loading model: {MODEL_ID}", flush=True)
    llm, sampling_params = init_vllm()

    titles_done = 0
    batch_buf: list[tuple[dict, dict, list]] = []

    def flush_batch():
        nonlocal titles_done
        if not batch_buf:
            return
        msgs_batch = [m for _, _, m in batch_buf]
        outputs = run_batch(llm, sampling_params, msgs_batch)

        by_title: dict[str, list] = {}
        for (row, question, _), out in zip(batch_buf, outputs):
            title = row["title"]
            if title not in by_title:
                by_title[title] = []
            by_title[title].append((row, question, out))

        for title, items in by_title.items():
            result_rows = []
            for row, question, out in items:
                text = out["text"]
                finish_reason = out["finish_reason"]

                if finish_reason == "length":
                    print(f"[run_mc_qwq_text] WARNING: truncated on {title} {question['question_type']}", flush=True)

                predicted = extract_answer_letter(text, question["applicable_letters"])
                if predicted is None:
                    print(
                        f"[run_mc_qwq_text] WARNING: no letter found in output for "
                        f"{title}/{question['question_type']}, text={repr(text[:120])}",
                        flush=True,
                    )

                # Build result with null logprobs — only predicted_letter from text
                null_lps = {"A": None, "B": None, "C": None, "D": None}
                result = build_result_row(row, MODEL_ID, question, null_lps, finish_reason)
                result["predicted_letter"] = predicted
                result["correct"] = (predicted == question["correct_letter"]) if predicted else None
                result["scoring_method"] = "text_extraction"
                result_rows.append(result)

            append_results(jsonl_path, result_rows)
            titles_done += 1
            if titles_done % 10 == 0:
                print(f"[run_mc_qwq_text] {titles_done} titles done", flush=True)

        batch_buf.clear()

    for _, row in todo.iterrows():
        row_dict = row.to_dict()
        if (row_dict["title"], row_dict["mod_type"]) in done_titles:
            continue

        seed = row_seed(row_dict["title"], MODEL_ID)
        questions = make_all_questions(row_dict, seed)
        if filter_types:
            questions = [q for q in questions if q["question_type"] in filter_types]

        for q in questions:
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": q["prompt"]},
            ]
            batch_buf.append((row_dict, q, messages))

        if len(batch_buf) >= args.batch_size * n_questions:
            flush_batch()

    flush_batch()

    total_rows = titles_done * n_questions
    print(f"[run_mc_qwq_text] Done. {titles_done} titles → {total_rows} rows written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
