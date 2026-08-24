"""
PDE LLM Eval — vLLM inference script.
Runs one model against all rows in merged_mod_jul28.csv (256 rows, 8 conditions).
Resumable: skips (title, mod_type, model) tuples already in the output JSONL.

Usage:
    python run_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset data/merged_mod_jul28.csv \
        --output_dir results/ \
        --batch_size 8 \
        --tp 1
"""
import argparse, json, os, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
# dataset_io still lives in eval/ while this module lives in freegen/ (left over from
# the eval->freegen split). The cluster sbatch papers over that with PYTHONPATH, which
# means importing this module anywhere else -- a helper script, a test, a REPL --
# dies at import. Resolve the sibling directory here so the module is self-sufficient.
# eval/ goes AFTER this file's own directory, not before it. Both directories hold a
# `parse_score.py` on the cluster -- eval/'s is the older pre-split copy -- and the
# original ordering inserted eval/ at position 0, so it SHADOWED freegen/parse_score.py
# for anything importing the bare name. Locally that never showed, because the laptop
# repo has no eval/parse_score.py and the import fell through to the right file; on
# torch it resolved to the stale copy and the job died at import with
# "cannot import name 'is_no_verdict'". Appended, so `dataset_io` (which lives only in
# eval/) still resolves while this package's own modules win their own names.
sys.path.append(str(Path(__file__).resolve().parent.parent / "eval"))
from parse_score import (parse_response, score_row,
                         classify_valid_confidence, is_no_verdict)
from dataset_io import DEFAULT_MOD_DATASET, load_dataset

# Inject API keys via RACA key_handler (repo lives at raca/packages/key_handler)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "key_handler"))
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except Exception as e:
    print(f"[run_eval] key_handler unavailable: {e}", flush=True)

# Bumped 2026-08-19: the `valid` field was a compound question. NOTE this now
# DIVERGES from eval/frontier/run_belief_revision.py PROMPT_S1 and from
# probe/extract_hidden.py, both of which still carry v1 verbatim. The probe
# design requires a fixed prompt, so its existing hidden states pair with v1
# only. Reconcile deliberately before comparing across those experiments.
PROMPT_VERSION = "v2-valid-disambiguated"

# Decoding, mirrored from crossmodal/eval/run_cross_modal_consistency.py. These three
# values and K_DRAWS are the reason a per-model comparison between the two
# experiments is meaningful at all: the same eight checkpoints answer the same 32
# solver systems under the same regime on both sides. Changing any of them here
# without changing them there silently turns a capability comparison into a decoding
# comparison. Approved by the researcher on 2026-08-24, replacing temperature=0.0.
UNIFORM_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
K_DRAWS = 3
SAMPLING_SEED = 20260821

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
- valid: does this simulation produce a physically correct solution for the PDE?
  Running without error is not sufficient.\
"""

MODEL_CONFIGS = {
    # ── jul28 roster ─────────────────────────────────────────────────────────
    # NB: Qwen3-32B is recorded here as thinking=False because that is what the
    # jul28 run actually did -- see resolve_thinking() for the bug that made the
    # old value say otherwise.
    "Qwen/Qwen3-32B":                              {"max_tokens": 16384, "thinking": False,
                                                    "max_tokens_on": 32768},
    # max_tokens_on added 2026-08-24 for consistency parity. Both are ALWAYS-thinking,
    # so their arm is "on" unconditionally and this raises their budget from the jul28
    # value of 16384 to the 32768 the consistency runner gave them. The published
    # jul28 rows are already generated and unaffected; every future run of these two
    # matches the other side. Without it the paired comparison would hand the same
    # checkpoint half the budget on one side, and 60 of the 70 truncations measured in
    # the consistency arms were in exactly these two models.
    "Qwen/QwQ-32B":                                {"max_tokens": 16384, "thinking": True,
                                                    "max_tokens_on": 32768},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":    {"max_tokens": 16384, "thinking": True,
                                                    "max_tokens_on": 32768},
    # New at jul28. No parity constraint with the writeup roster, so it gets a
    # generous ceiling rather than the 2048 the other non-reasoning models use.
    "Qwen/Qwen3-Coder-30B-A3B-Instruct":           {"max_tokens": 8192,  "thinking": False},

    # ── added 2026-08-20 ─────────────────────────────────────────────────────
    # The Qwen3 ladder: one family, one prompt format, one toggle, three scales.
    # This is what turns "reasoning helps" from a cross-family comparison -- where
    # it is confounded with everything else that differs between QwQ and Llama --
    # into a within-model contrast.
    "Qwen/Qwen3-8B":                               {"max_tokens": 8192,  "thinking": False,
                                                    "max_tokens_on": 32768},
    "Qwen/Qwen3-14B":                              {"max_tokens": 8192,  "thinking": False,
                                                    "max_tokens_on": 32768},

    # Pairs against microsoft/phi-4, already in the roster: same base model family,
    # reasoning-trained, so it is a second within-family reasoning contrast.
    "microsoft/Phi-4-reasoning-plus":              {"max_tokens": 30720, "thinking": True,
                                                    "max_model_len": 32768},
    # Configurable reasoning effort -- a dose-response inside a single model, which
    # nothing else in the roster offers.
    "openai/gpt-oss-120b":                         {"max_tokens": 32768, "thinking": True,
                                                    "reasoning_effort": "medium", "tp": 2},
    # Same distillation as the Qwen-32B distill, different base: separates "R1's
    # reasoning traces" from "Qwen".
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":   {"max_tokens": 32768, "thinking": True,
                                                    "tp": 2},
    # Cheap MoE reasoning arm, ~3B active.
    "Qwen/Qwen3-30B-A3B-Thinking-2507":            {"max_tokens": 32768, "thinking": True},
    # Ceiling anchor. 22B active, but bf16 weights want four H200s.
    "Qwen/Qwen3-235B-A22B-Thinking-2507":          {"max_tokens": 32768, "thinking": True,
                                                    "tp": 4, "max_model_len": 40960},

    # ── added 2026-08-24: the cross-representation consistency roster ────────
    # These five complete the eight checkpoints the consistency experiment runs on
    # (the other three -- Qwen3-32B, QwQ-32B, R1-Distill-32B -- are already above).
    # Both experiments use the SAME 32 solver systems, so aligning the models is
    # what makes a per-model comparison possible at all: "does producing a valid
    # solver predict noticing that four views of one disagree" needs the same model
    # on both sides, and before this the two rosters shared exactly one checkpoint.
    #
    # max_tokens_on MIRRORS crossmodal/eval/run_cross_modal_consistency.py exactly,
    # rather than being re-derived for the shorter free-generation prompt. The point
    # of this run is a paired comparison, and a model given 32768 here and 131072
    # there is not the same model twice -- a budget difference would show up as a
    # capability difference. Under-budgeting is also the specific failure that cost
    # this project days: 29.6% of Nemotron's consistency draws hit a 32768 cap and
    # 24.9% still hit 65536, which is why four of these sit at 131072.
    "Qwen/Qwen3.5-27B":  {"max_tokens": 8192, "thinking": True, "max_tokens_on": 32768},
    "Qwen/Qwen3.6-27B":  {"max_tokens": 8192, "thinking": True, "max_tokens_on": 131072},
    # 65536, NOT 131072. The consistency runner lowered this on 2026-08-22 after
    # measuring 1,152 draws (p50 17,961, p90 51,500, p99 77,267): a 65,536 cap
    # truncates 2.9% of draws against 131,072's 0.7%, and those 0.7 points cost a
    # doubled straggler tail. llm.generate() writes nothing until every sequence in
    # the batch returns, so one trace running to 131k holds ~190 finished ones
    # unwritten for an extra hour and collapses GPU utilization at the batch tail --
    # which invites the cluster's sub-60% killer. Two consistency shards sat 3h at
    # 189/192 with zero rows on disk from exactly this.
    "Qwen/Qwen3.8-27B":  {"max_tokens": 8192, "thinking": True, "max_tokens_on": 65536},
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
        {"max_tokens": 8192, "thinking": True, "max_tokens_on": 131072},
    "zai-org/GLM-4.7-Flash":
        {"max_tokens": 8192, "thinking": True, "max_tokens_on": 131072},

    "__default__":                                 {"max_tokens": 2048,  "thinking": False},
}

# Models whose chat template accepts enable_thinking, so they can be run as BOTH a
# thinking and a non-thinking arm of the same experiment.
TOGGLE_THINKING_MODELS = {
    "Qwen/Qwen3-32B", "Qwen/Qwen3-14B", "Qwen/Qwen3-8B",
    # data/model_registry.csv records all five of these as reasoning_mode=toggle with
    # think tags, and the consistency arms were generated with thinking ON. Listing
    # them as toggle rather than always-on keeps `--thinking off` available as a
    # future contrast without pretending it was ever run.
    "Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "zai-org/GLM-4.7-Flash",
}
# Models that always emit reasoning and cannot be asked not to.
ALWAYS_THINKING_MODELS = {
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "microsoft/Phi-4-reasoning-plus",
    "openai/gpt-oss-120b",
}


def resolve_thinking(model_id: str, requested: str) -> str:
    """Return "on" or "off" for this arm, or exit if the model cannot honour it.

    This replaces a single set membership that read:

        use_thinking = model_id in THINKING_MODELS
        chat_kwargs = {"enable_thinking": False} if use_thinking else {}

    which disabled thinking for every model in a set named THINKING_MODELS, while
    MODEL_CONFIGS carried a `thinking` key that nothing ever read. Qwen3-32B was
    therefore recorded in the config as a thinking model and run as a non-thinking
    one, and no column in the output said which -- so its jul28 score is a
    thinking-OFF number that reads as a thinking-ON number.

    Thinking is an experimental factor, so it is now an explicit CLI arm that ends
    up in the output row and in the results filename.
    """
    if model_id in ALWAYS_THINKING_MODELS:
        if requested == "off":
            raise SystemExit(
                f"[run_eval] {model_id} always emits reasoning; --thinking off is "
                f"not achievable for it. Drop the flag or pick a toggle model.")
        return "on"
    if model_id in TOGGLE_THINKING_MODELS:
        if requested == "auto":
            return "on" if MODEL_CONFIGS.get(model_id, {}).get("thinking") else "off"
        return requested
    if requested == "on":
        raise SystemExit(
            f"[run_eval] {model_id} has no thinking mode; --thinking on would be "
            f"silently ignored, which would mislabel the arm.")
    return "off"


def get_model_config(model_id: str) -> dict:
    return MODEL_CONFIGS.get(model_id, MODEL_CONFIGS["__default__"])


def legacy_arm(model_id: str) -> str:
    """Which arm a PRE-v6 results file was actually produced under.

    Before the reasoning arm was made explicit, every toggle-capable model was run
    with enable_thinking=False and every always-reasoning model emitted reasoning it
    could not suppress. So a legacy file's arm is recoverable from the model alone,
    and Qwen3-32B's legacy rows are "off" -- which is the whole point of F9.
    """
    return "on" if model_id in ALWAYS_THINKING_MODELS else "off"


def load_checkpoint(jsonl_paths, model_id=None, arm=None) -> set[tuple]:
    """Return set of (title, mod_type, model, sample_idx) tuples already processed.

    Accepts several paths so a v6 run can resume from a PRE-v6 results file. Without
    this, renaming results to one file per arm would silently orphan every existing
    checkpoint: the documented recovery for a wall-timeout ("resubmit the same
    command") would find nothing, regenerate a finished model from zero, and leave
    the orphaned legacy file to be swept into the same HF dataset as duplicates.

    A legacy file is only honoured when the arm it was produced under matches the
    arm being requested now, so resuming a thinking-on run never inherits rows that
    were generated with reasoning disabled.
    """
    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]
    done = set()
    for path in jsonl_paths:
        if not os.path.exists(path):
            continue
        n_before = len(done)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if arm is not None:
                    # A row that records its own arm is authoritative. A legacy row
                    # records none, so infer the arm it must have run under from its
                    # model -- treating an untagged row as belonging to whichever arm
                    # is asking would let a thinking-on run resume from rows that were
                    # generated with reasoning disabled.
                    row_arm = row.get("thinking") or legacy_arm(
                        row.get("model", model_id or ""))
                    if row_arm != arm:
                        continue
                # sample_idx is part of the identity now that k=3. Without it a
                # resume that found draw 0 would treat the whole item as finished
                # and the arm would come back k=1 while the log said k=3. Legacy
                # rows carry no sample_idx and are draw 0 by construction.
                done.add((row["title"], row["mod_type"], row["model"],
                          int(row.get("sample_idx", 0))))
        print(f"[run_eval] Checkpoint: {len(done) - n_before} rows from "
              f"{os.path.basename(path)}", flush=True)
    print(f"[run_eval] Checkpoint: {len(done)} rows already done.", flush=True)
    return done


def append_result(jsonl_path: str, result: dict):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def build_messages(code: str) -> list[dict]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(code=code)}]


def arm_max_tokens(model_id: str, thinking: str) -> int:
    """Generation budget for THIS arm.

    A toggle model's thinking arm needs the 32k floor the workspace rule sets for
    reasoning models; its non-thinking twin does not, and giving both the same
    ceiling would waste hours of KV cache on an arm that answers in 200 tokens.
    """
    cfg = get_model_config(model_id)
    if thinking == "on" and "max_tokens_on" in cfg:
        return cfg["max_tokens_on"]
    return cfg["max_tokens"]


def init_vllm(model_id: str, tensor_parallel_size: int = 1, thinking: str = "off",
              worst_prompt_tokens: int = 0):
    from vllm import LLM, SamplingParams
    cfg = get_model_config(model_id)
    # Without an explicit cap, vLLM sizes the KV cache against the model's full
    # max_position_embeddings. Qwen3-235B declares 262144, which does not fit
    # alongside 470GB of weights on four H200s, and the engine aborts at init
    # rather than at generation time. The cap only has to exceed prompt +
    # max_tokens; nothing here needs a 262k window.
    budget = arm_max_tokens(model_id, thinking)
    # Headroom sized against the LONGEST PROMPT in the dataset, not a flat +4096.
    # merged_mod_jul28's two longest rows are ~16k characters, roughly 5k tokens, so
    # a 4096 constant put the real generation budget about 1k BELOW the configured
    # one on those rows while the log still printed the full number -- a silent,
    # row-dependent truncation. `worst_prompt_tokens` is passed in by the caller,
    # which has the dataset; the constant remains the fallback for callers that do
    # not.
    head = int(worst_prompt_tokens * 1.15) + 512 if worst_prompt_tokens else 4096
    max_model_len = cfg.get("max_model_len", budget + head)
    # Kernel-launch gaps in eager mode read as idle to the cluster's utilization
    # sampler, which cancelled two of this experiment's jobs mid-generation. Kept as
    # the default because it is what the published freegen arms used, but switchable
    # so a long run can turn it off, exactly as the consistency runner does.
    eager = os.environ.get("VLLM_ENFORCE_EAGER", "1") not in ("0", "false", "False")
    kw = dict(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=eager,
        max_model_len=max_model_len,
    )
    # Gated Delta Net JIT-compiles a FlashInfer kernel on first prefill. FlashInfer
    # 0.6.6 on this cluster is missing prefill_kernel_delta_rule_sm90.cuh, so "auto"
    # picks the broken path on sm90 and the engine dies AFTER the weights load -- a
    # held allocation spent on nothing. Qwen3.5/3.6/3.8 are all GDN. Same value and
    # same env override as run_cross_modal_consistency.py.
    gdn = os.environ.get("GDN_PREFILL_BACKEND", "triton")
    if gdn:
        kw["gdn_prefill_backend"] = gdn
    try:
        llm = LLM(**kw)
    except TypeError:
        # Older vLLM builds do not accept the kwarg. Falling back is correct only
        # because the models that need it will then fail loudly at prefill rather
        # than silently producing different numbers.
        kw.pop("gdn_prefill_backend", None)
        print("[run_eval] WARNING: this vLLM does not accept gdn_prefill_backend; "
              "GDN models may die at first prefill", flush=True)
        llm = LLM(**kw)
    print(f"[run_eval] max_model_len={max_model_len} max_tokens={budget} "
          f"(arm thinking={thinking}) eager={eager} gdn={gdn or 'auto'}", flush=True)
    # Decoding MIRRORS crossmodal/eval/run_cross_modal_consistency.py UNIFORM_SAMPLING
    # exactly. The two experiments run the same eight checkpoints over the same 32
    # solver systems and exist to be compared per model; a model run greedy here and
    # at 0.6/0.95/20 there is not the same model twice, and the difference would
    # surface as a capability difference. Greedy is also a measured failure mode for
    # this roster -- Nemotron truncated 30 of its first 64 consistency items in
    # repetition loops at temperature 0.
    sampling_params = SamplingParams(
        max_tokens=budget, n=K_DRAWS, seed=SAMPLING_SEED, **UNIFORM_SAMPLING)
    return llm, sampling_params


def run_batch(llm, sampling_params, messages_batch: list[list[dict]], model_id: str,
              thinking: str = "off") -> list[dict]:
    """Run a batch of messages through vLLM. Returns list of {text, finish_reason}.

    `thinking` is the already-resolved arm from resolve_thinking(), passed in rather
    than recomputed here so the value that reaches the chat template is provably the
    same one that gets written into the output rows.
    """
    from vllm import SamplingParams
    chat_kwargs = {}
    if model_id in TOGGLE_THINKING_MODELS:
        chat_kwargs["enable_thinking"] = (thinking == "on")
    effort = get_model_config(model_id).get("reasoning_effort")
    if effort:
        chat_kwargs["reasoning_effort"] = effort
    outputs = llm.chat(
        messages_batch,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_kwargs if chat_kwargs else None,
    )
    # EVERY completion, not outputs[0]. With n=K_DRAWS the engine returns k samples
    # per prompt; taking the first would throw away two thirds of what was paid for
    # and silently make k=1 again while the log still said k=3.
    return [
        [{"text": o.text, "finish_reason": o.finish_reason} for o in out.outputs]
        for out in outputs
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--dataset",    default=DEFAULT_MOD_DATASET)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tp",         type=int, default=None,
                        help="tensor_parallel_size; defaults to the model's config entry")
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                        help="Reasoning arm. 'auto' takes the model's configured "
                             "default; 'on'/'off' force it and fail loudly if the "
                             "model cannot honour the request.")
    parser.add_argument("--gt_samples", default="",
                        help="Comma-separated gt_sample IDs to restrict to (canary). "
                             "Empty = all. The integrity assertion always runs on the "
                             "full dataset first, before any subsetting.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    thinking = resolve_thinking(args.model, args.thinking)
    tp = args.tp if args.tp is not None else get_model_config(args.model).get("tp", 1)
    # One file per ARM, not per model: the same checkpoint id in two reasoning modes
    # would otherwise resume into each other and silently blend the two arms.
    model_slug = args.model.replace("/", "__")
    jsonl_path = os.path.join(args.output_dir, f"{model_slug}__think-{thinking}.jsonl")
    print(f"[run_eval] Arm: {args.model} thinking={thinking} tp={tp}", flush=True)

    df = load_dataset(args.dataset)
    print(f"[run_eval] Dataset: {len(df)} rows", flush=True)

    V2_DIST = {"Comm_Valid": 16, "NoComm_Valid": 16, "NoComm_InValid": 16,
               "CorrComm": 16, "NoComm_CorrVar": 16, "Comm_InValid": 16}
    V3_DIST = {**V2_DIST, "CorrComm_Invalid": 16, "NoComm_CorrVar_InValid": 16}
    # jul28: same 8 conditions as v3, 32 gt_samples instead of 16
    JUL28_DIST = {k: 32 for k in V3_DIST}
    actual_dist = df["mod_type"].value_counts().to_dict()
    assert actual_dist in (V2_DIST, V3_DIST, JUL28_DIST), \
        f"Unexpected mod_type distribution: {actual_dist}"
    print(f"[run_eval] Dataset integrity check passed ({len(df)} rows, {len(actual_dist)} conditions).", flush=True)

    if args.gt_samples:
        wanted = [g.strip() for g in args.gt_samples.split(",") if g.strip()]
        missing = sorted(set(wanted) - set(df["gt_sample"]))
        assert not missing, f"--gt_samples not in dataset: {missing}"
        df = df[df["gt_sample"].isin(wanted)].reset_index(drop=True)
        print(f"[run_eval] Restricted to {len(wanted)} gt_samples -> {len(df)} rows "
              f"(expected {8 * len(wanted)}).", flush=True)
        assert len(df) == 8 * len(wanted), "gt_sample subset is not a full 8-condition cross"

    # Resume from the pre-v6 filename too, but only when that file's arm is the arm
    # being asked for now.
    resume_paths = [jsonl_path]
    legacy_path = os.path.join(args.output_dir, f"{model_slug}.jsonl")
    if os.path.exists(legacy_path):
        if legacy_arm(args.model) == thinking:
            resume_paths.append(legacy_path)
            print(f"[run_eval] Legacy checkpoint {os.path.basename(legacy_path)} "
                  f"matches arm '{thinking}'; resuming from it.", flush=True)
        else:
            print(f"[run_eval] Legacy {os.path.basename(legacy_path)} was arm "
                  f"'{legacy_arm(args.model)}', not '{thinking}'; NOT resuming from it.",
                  flush=True)
    done = load_checkpoint(resume_paths, args.model, thinking)
    # An item is outstanding unless ALL k of its draws are on disk. The pre-k=3 key
    # was a 3-tuple; `done` now holds 4-tuples, so the old membership test would
    # never match, `todo` would always be the full dataset, and a finished arm would
    # reload the engine and regenerate everything to write zero rows.
    _all_draws_done = lambda r: all(
        (r["title"], r["mod_type"], args.model, k) in done for k in range(K_DRAWS))
    todo = df[~df.apply(_all_draws_done, axis=1)].reset_index(drop=True)
    print(f"[run_eval] Rows to process: {len(todo)}", flush=True)

    if todo.empty:
        print("[run_eval] All rows already processed.", flush=True)
        return

    print(f"[run_eval] Loading model: {args.model}", flush=True)
    # Longest prompt in the SUBSET actually being run, so max_model_len is sized to
    # what this job will see rather than to a flat constant. Character count is a
    # deliberate over-estimate of tokens for code, which is what we want here.
    worst_chars = int(todo["code"].astype(str).str.len().max()) if len(todo) else 0
    worst_prompt_tokens = worst_chars // 3 + 256
    print(f"[run_eval] worst prompt {worst_chars} chars "
          f"(~{worst_prompt_tokens} tokens)", flush=True)
    llm, sampling_params = init_vllm(args.model, tp, thinking,
                                     worst_prompt_tokens=worst_prompt_tokens)
    print(f"[run_eval] sampling: k={K_DRAWS} {UNIFORM_SAMPLING} "
          f"seed={SAMPLING_SEED}", flush=True)

    embed_model = None
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[run_eval] Embedding model loaded.", flush=True)
    except Exception as e:
        print(f"[run_eval] Embedding model unavailable (will skip embed_sim): {e}", flush=True)

    new_rows = 0
    batch_buf = []

    def flush_batch():
        nonlocal new_rows
        if not batch_buf:
            return
        outputs = run_batch(llm, sampling_params, [m for _, m in batch_buf],
                            args.model, thinking)
        # Flattened to (row, draw_index, completion) so the body below stays a single
        # loop. vLLM returns k completions per prompt under n=K_DRAWS; each becomes
        # its own output row, keyed by sample_idx, exactly as the consistency arms
        # are keyed.
        for (row, _), draws in zip(batch_buf, outputs):
            for sample_idx, out in enumerate(draws):
                if (row["title"], row["mod_type"], args.model, sample_idx) in done:
                    continue      # this draw survived an earlier attempt
                _write_row(row, sample_idx, out)
        batch_buf.clear()

    def _write_row(row, sample_idx, out):
        nonlocal new_rows
        text, finish_reason = out["text"], out["finish_reason"]
        if finish_reason == "length":
            print(f"[run_eval] WARNING: truncation on {row['title']} "
                  f"draw {sample_idx}", flush=True)
        # A run that never reached an answer is marked, not scored. Without this the
        # row goes to score_valid(None) -> valid_match 0, which counts a model that
        # said nothing as a model that said something false, and to
        # classify_valid_confidence(None) -> "Hedged", which inflates a published
        # figure with parse failures. The column is written for every row so the
        # count is visible rather than inferred.
        no_verdict = is_no_verdict(text, finish_reason)
        parsed = parse_response(text)
        scores = score_row(parsed, row, embed_model)
        result = {
            "title":          row["title"],
            # gt_sample / source carry the base-problem identity and the
            # human-vs-synthetic split so downstream cuts need no re-join.
            "gt_sample":      row.get("gt_sample"),
            "source":         row.get("source"),
            "pde_class":      row["pde_class"],
            "mod_type":       row["mod_type"],
            # num_char, not num_lines: num_lines is +1 on 80/256 rows (all synthetic,
            # a trailing-newline counting difference between the two parsers).
            "num_char":       int(row["num_char"]) if pd.notna(row.get("num_char")) else None,
            "invalidity_note": (None if pd.isna(row.get("invalidity_note"))
                                else row.get("invalidity_note")),
            "gt_pde":         row["pde_class"],
            "gt_method":      str(row["num_method"]),
            "gt_behavior":    str(row["phys_process"]),
            "gt_valid":       bool(row["phys_valid"]),
            "model_response": text,
            "parsed_pde":     parsed.get("pde"),
            "parsed_method":  parsed.get("method"),
            "parsed_behavior":parsed.get("behavior"),
            "parsed_valid":   parsed.get("valid"),
            # Hedge class stored at eval time so every consumer reads one label
            # rather than re-deriving it. Canonical rule: parse_score.py.
            "valid_conf":     classify_valid_confidence(parsed.get("valid")),
            "finish_reason":  finish_reason,
            "no_verdict":     bool(no_verdict),
            "sample_idx":     int(sample_idx),
            "k_draws":        K_DRAWS,
            "temperature":    UNIFORM_SAMPLING["temperature"],
            "top_p":          UNIFORM_SAMPLING["top_p"],
            "top_k":          UNIFORM_SAMPLING["top_k"],
            "sampling_seed":  SAMPLING_SEED,
            "model":          args.model,
            "thinking":       thinking,
            "prompt_version": PROMPT_VERSION,
            "dataset":        os.path.basename(args.dataset),
            **scores,
        }
        append_result(jsonl_path, result)
        new_rows += 1


    for _, row in todo.iterrows():
        # An item is queued unless ALL k of its draws are already on disk. vLLM
        # returns k completions for one prompt, so a partially-done item is
        # regenerated in full and the draws already present are skipped at write
        # time -- cheaper than issuing k separate single-draw requests, and it keeps
        # every draw of an item under one seed.
        if all((row["title"], row["mod_type"], args.model, k) in done
               for k in range(K_DRAWS)):
            continue
        batch_buf.append((row.to_dict(), build_messages(str(row["code"]))))
        if len(batch_buf) >= args.batch_size:
            flush_batch()

    flush_batch()
    print(f"[run_eval] Done. {new_rows} new rows written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
