"""
Extract hidden-state representations for all dataset rows.

Usage (from repo root):
    python probe/extract_hidden.py \
        --dataset data/pdedata_clean_v2.xlsx \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --output_dir probe/hidden_states/

Saves a .npz file per model:
    mean_pool        : (N, L, D) float16  — mean over code-token positions per layer
    last_tok         : (N, L, D) float16  — last input token per layer
    code_token_spans : (N, 2)    int32    — (start, end) token indices used for mean_pool
    titles           : (N,)      str
    mod_types        : (N,)      str
    pde_classes      : (N,)      str
    gt_samples       : (N,)      str
    phys_valid       : (N,)      bool
    phys_process     : (N,)      str      — raw "/" -separated
    num_method       : (N,)      str      — raw "/" -separated

L = num_hidden_layers + 1  (index 0 = embedding, 1..L-1 = transformer layers)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same prompt as eval/run_eval.py — kept verbatim for consistency
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

EXPECTED_MOD_TYPE_DIST = {
    "Comm_Valid": 16, "NoComm_Valid": 16, "CorrComm": 16, "NoComm_CorrVar": 16,
    "Comm_InValid": 16, "NoComm_InValid": 16, "CorrComm_Invalid": 16, "NoComm_CorrVar_InValid": 16,
}


def find_code_token_span(
    offsets: list,
    code_char_start: int,
    code_char_end: int,
) -> tuple:
    """Return (first_tok_idx, end_tok_idx) s.t. hidden_states[start:end] covers code."""
    start_tok, end_tok = None, None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue  # special / padding token
        if e > code_char_start and s < code_char_end:
            if start_tok is None:
                start_tok = i
            end_tok = i + 1
    return (start_tok, end_tok)


def validate_code_span(tokenizer, input_ids, span, code, title):
    """
    Decode the selected token span and compare to original code after whitespace
    normalization. Returns (ok, message). Cheap: decode is CPU-only, ~microseconds.
    """
    if span[0] is None or span[0] >= span[1]:
        return False, f"empty or None span: {span}"

    decoded = tokenizer.decode(
        input_ids[0, span[0]:span[1]],
        skip_special_tokens=True,
    )
    norm_decoded = " ".join(decoded.split())
    norm_code    = " ".join(code.split())

    if norm_decoded == norm_code:
        return True, "ok"

    # Tolerate slight boundary differences (BPE may include 1-2 chars from adjacent token)
    ratio = len(norm_decoded) / max(len(norm_code), 1)
    if 0.90 <= ratio <= 1.10:
        return True, f"approx match (ratio={ratio:.3f})"

    return False, (
        f"span mismatch for '{title}': "
        f"decoded={repr(norm_decoded[:80])} | code={repr(norm_code[:80])} | ratio={ratio:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/pdedata_clean_v3.xlsx")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output_dir", default="probe/hidden_states/")
    parser.add_argument("--gt_samples", default=None,
                        help="Comma-separated gt_sample IDs to filter (e.g. Wave_1,Heat_1). "
                             "Omit for full dataset run.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.", flush=True)
        sys.exit(1)

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: {args.model}", flush=True)

    # Load dataset
    df = pd.read_excel(args.dataset)
    print(f"Dataset: {len(df)} rows loaded", flush=True)

    if args.gt_samples:
        keep = [s.strip() for s in args.gt_samples.split(",")]
        df = df[df["gt_sample"].isin(keep)].reset_index(drop=True)
        print(f"Filtered to gt_samples {keep}: {len(df)} rows", flush=True)
    else:
        assert len(df) == 128, f"Expected 128 rows, got {len(df)}"
        dist = df["mod_type"].value_counts().to_dict()
        assert dist == EXPECTED_MOD_TYPE_DIST, f"Unexpected mod_type distribution: {dist}"

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Spot-check chat template — print first 20 token IDs and decoded prefix
    # so cross-model comparisons can verify inputs are equivalent
    _sample_code = "x = 1.0"
    _sample_prompt = PROMPT_TEMPLATE.format(code=_sample_code)
    _sample_fmt = tokenizer.apply_chat_template(
        [{"role": "user", "content": _sample_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    _sample_ids = tokenizer(_sample_fmt, add_special_tokens=False)["input_ids"][:20]
    print(f"Chat template spot-check (first 20 token IDs): {_sample_ids}", flush=True)
    print(f"  decoded: {repr(tokenizer.decode(_sample_ids))}", flush=True)

    # Load model in bfloat16 (native for Qwen2.5/QwQ; fp16 risks overflow on 32B models)
    print("Loading model (bfloat16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    L = n_layers + 1  # +1 for embedding layer at index 0
    D = model.config.hidden_size
    N = len(df)

    print(f"Model: {n_layers} transformer layers, hidden_dim={D}", flush=True)
    print(f"Allocating arrays: 2 × ({N}, {L}, {D}) float16 = "
          f"{2 * N * L * D * 2 / 1e6:.1f} MB", flush=True)

    all_mean_pool = np.zeros((N, L, D), dtype=np.float16)
    all_last_tok  = np.zeros((N, L, D), dtype=np.float16)
    all_spans     = np.zeros((N, 2), dtype=np.int32)
    span_failures = []

    for idx, (_, row) in enumerate(df.iterrows()):
        code = str(row["code"])
        prompt_text = PROMPT_TEMPLATE.format(code=code)
        messages = [{"role": "user", "content": prompt_text}]

        # Apply chat template — same framing as vLLM in existing evals
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Locate code char span in the formatted string
        code_char_start = formatted.find(code)
        if code_char_start == -1:
            # fallback: use <code> markers
            m = formatted.find("<code>\n")
            if m != -1:
                code_char_start = m + len("<code>\n")
                code_char_end = formatted.find("\n</code>", code_char_start)
                if code_char_end == -1:
                    code_char_end = code_char_start + len(code)
            else:
                print(f"  WARNING row {idx} ({row['title']}): code not found in formatted text",
                      flush=True)
                code_char_start = 0
                code_char_end = len(formatted)
        else:
            code_char_end = code_char_start + len(code)

        # Tokenize with offset mapping (add_special_tokens=False: chat template
        # already embedded them as literal text via tokenize=False above)
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        offsets = enc["offset_mapping"][0].tolist()

        span = find_code_token_span(offsets, code_char_start, code_char_end)
        if span[0] is None:
            print(f"  WARNING row {idx} ({row['title']}): no code tokens found in span, "
                  f"using full sequence", flush=True)
            span = (0, input_ids.shape[1])
        all_spans[idx] = span

        ok, msg = validate_code_span(tokenizer, input_ids, span, code, row["title"])
        if not ok:
            span_failures.append(f"  row {idx} ({row['title']}): {msg}")
        elif "approx" in msg:
            print(f"  WARNING row {idx} ({row['title']}): {msg}", flush=True)

        # Spot-check: print the last code token text for the first 3 rows
        if idx < 3:
            last_tok_text = tokenizer.decode(
                [input_ids[0, span[1] - 1].item()], skip_special_tokens=True
            )
            print(f"  [spot-check] row {idx} last_code_tok={repr(last_tok_text)}", flush=True)

        # Forward pass — no_grad, output_hidden_states=True
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # outputs.hidden_states: tuple of L tensors, each (1, T, D)
        # Use last code token (span[1]-1), not last prompt token.
        # The prompt ends with <|im_start|>assistant\n — identical across all
        # examples — so last prompt token gives zero-variance representations.
        last_code_pos = span[1] - 1
        for l, hs in enumerate(outputs.hidden_states):
            hs_l = hs[0]  # (T, D)
            code_hs = hs_l[span[0]:span[1]]  # (n_code_tokens, D)
            # cast bfloat16 → float32 before numpy (float16 can't represent bf16 range)
            mp = code_hs.mean(dim=0).float().cpu().numpy()
            lt = hs_l[last_code_pos].float().cpu().numpy()
            all_mean_pool[idx, l] = mp.astype(np.float16)
            all_last_tok[idx, l]  = lt.astype(np.float16)

        n_code_toks = span[1] - span[0]
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/{N}] {row['title']} ({row['mod_type']}) "
                  f"seq_len={input_ids.shape[1]} code_toks={n_code_toks}", flush=True)

    # Sanity check: assert no NaN/inf in hidden states
    for name, arr in [("mean_pool", all_mean_pool), ("last_tok", all_last_tok)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan > 0 or n_inf > 0:
            print(f"ERROR: {name} has {n_nan} NaN and {n_inf} Inf values — "
                  f"likely bf16→fp16 overflow. Check model dtype.", flush=True)
            sys.exit(1)
    print("NaN/Inf check passed for mean_pool and last_tok.", flush=True)

    # Save — always write the NPZ so GPU work is not lost
    os.makedirs(args.output_dir, exist_ok=True)
    slug = args.model.replace("/", "_")
    suffix = "_canary" if args.gt_samples else ""
    out_path = os.path.join(args.output_dir, f"{slug}{suffix}.npz")

    # Preserve existing NPZ before overwriting
    if os.path.exists(out_path):
        backup_path = out_path.replace(".npz", "_backup.npz")
        os.rename(out_path, backup_path)
        print(f"Existing NPZ renamed to: {backup_path}", flush=True)

    np.savez(
        out_path,
        mean_pool=all_mean_pool,
        last_tok=all_last_tok,
        code_token_spans=all_spans,
        titles=df["title"].values.astype(str),
        mod_types=df["mod_type"].values.astype(str),
        pde_classes=df["pde_class"].values.astype(str),
        gt_samples=df["gt_sample"].values.astype(str),
        phys_valid=df["phys_valid"].values.astype(bool),
        phys_process=df["phys_process"].values.astype(str),
        num_method=df["num_method"].values.astype(str),
        codes=df["code"].values.astype(str),
    )
    print(f"\nSaved: {out_path}", flush=True)
    print(f"  mean_pool : {all_mean_pool.shape}  dtype=float16", flush=True)
    print(f"  last_tok  : {all_last_tok.shape}  dtype=float16", flush=True)
    print(f"  code spans: min_len={int((all_spans[:,1]-all_spans[:,0]).min())} "
          f"max_len={int((all_spans[:,1]-all_spans[:,0]).max())}", flush=True)

    if span_failures:
        print(f"\nERROR: {len(span_failures)} span validation failure(s) — NPZ saved but check spans:", flush=True)
        for f in span_failures:
            print(f, flush=True)
        sys.exit(1)
    else:
        print(f"Span validation passed for all {N} rows.", flush=True)


if __name__ == "__main__":
    main()
