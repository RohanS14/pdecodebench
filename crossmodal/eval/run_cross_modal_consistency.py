"""
run_cross_modal_consistency.py — vLLM generation for the cross-modal consistency
experiment (plan Part III).

Shows a model four representations of one physical system -- code, equation,
trajectory, description -- in randomized slot order, and asks whether they agree
and which one is the odd view out. Runs on open weights, so there is no API cost;
the budget is GPU-hours.

Reuses freegen/run_eval.py's shape (vLLM init, checkpoint/resume JSONL, per-model
config) with three deliberate departures, each of which that script cannot support
as written:

  1. `enable_thinking` is a PARAMETER here. run_eval.run_batch hardcodes
     {"enable_thinking": False} for thinking models, so reasoning is always off --
     which is fine for its own experiment but would erase the reasoning factor this
     one is built around.
  2. `max_model_len` is set explicitly. Prompts reach 33k tokens on the worst 2-D
     system and the thinking arm needs a long output budget on top; the default
     would silently truncate. See the MAX_MODEL_LEN comment for the measurement --
     an earlier version of this docstring said "~10.5k", which is the MEDIAN, not
     the max, and sizing the context off it truncated a third of the benchmark.
  3. Structured output comes from guided decoding where the runtime has it. Whether
     it does is a CANARY QUESTION, not an assumption: compute nodes run Python
     3.9.21, which pins vLLM older than the login node suggests, and the README is
     explicit that versions must never be checked from a login shell. The probe
     runs at startup and the chosen route is recorded on every row.

Nothing is truncated on the way out. Reasoning traces are stored in full, because
one of the questions is whether those traces reason about physics or about
identifier names.

Usage:
    python crossmodal/eval/run_cross_modal_consistency.py \\
        --model Qwen/Qwen3-32B --thinking off --limit 16
"""
import argparse
import json
import os
import subprocess
import sys
import time

# repo root: this file sits at crossmodal/<area>/, so three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from crossmodal.datagen.build_multimodal_items import MOD_DATASET, MULTIMODAL_CSV   # noqa: E402
from crossmodal.eval.consistency_prompts import (                                   # noqa: E402
    CONSISTENCY_SCHEMA, ViewSources, build_messages, load_items, load_exec_trajectories,
)
from crossmodal.eval.parse_consistency import parse_consistency, score_consistency  # noqa: E402

DEFAULT_ITEMS = "data/multimodal_items_v1.csv"

# max_tokens per arm. The house rule is the model's full useful generation length;
# below 8192 is wrong for a generative task and thinking models want far more.
# run_eval.py's MODEL_CONFIGS caps thinking at 16384, which is raised here because
# these prompts are 3-4x longer than that experiment's and the traces run longer
# with them.
# The "off" arm was 4096. That is below the house floor of 8192 for a generative
# task, and it was actively dangerous on any model that reasons regardless of the
# flag: the trace hits the cap, <think> never closes, strip_think returns "" and the
# row parses as a failure. Truncated output is failed output, so the floor is 8192.
MAX_TOKENS = {"on": 32768, "off": 8192}

# MEASURED, not estimated (2026-08-21). Every one of the 1024 prompts was rebuilt and
# tokenized with each roster model's OWN tokenizer and chat template -- a chars/token
# ratio is not portable across tokenizers, and the template adds tokens of its own:
#
#   model                          median    p99      MAX
#   Qwen3.5-27B / Qwen3.6-27B        9031   32631    32632
#   Qwen3.8-27B                      9073   32673    32674
#   Nemotron-3-Nano-30B-A3B          9008   33096    33097   <- worst case
#   GLM-4.7-Flash                    7472   26994    27140
#   Olmo-3.1-32B-Think               7050   25643    25644
#   Qwen3-32B (published reference)  8982   32522    32523   <- matches the published
#                                                              run's observed max exactly
#
# The worst item is Wave_1|X_C|obfuscated at 33,097 tokens (gemma-4 measured
# 33,217 but is no longer in the roster; see below). A full 32,768-token answer
# on THAT item needs 65,865, so the previous 49,152 was ~17k short and the docstring's
# "~10.5k prompt" was wrong by more than 3x. The consequence was not hypothetical:
# 360 of 1024 items (35.2%), spanning 12 of 32 solvers, carry prompts over 16,384
# tokens, and on those the real generation budget silently fell as low as ~16.6k while
# the log still announced 32,768. aggregate_cross_modal drops truncated rows, and the
# drop lands on the twelve LONGEST systems -- a non-random hole in the benchmark.
#
# 69632 = 68 * 1024, giving ~3.8k of slack over the 65,865 requirement.
# model_context_limit() clamps this to each model's own declared context, which is
# exactly right: Olmo-3.1 declares 65,536 and is held there, but it needs only
# 25,644 + 32,768 = 58,412, so it loses nothing.
MAX_MODEL_LEN = 69632

# Worst measured prompt across the roster (Nemotron tokenizer on
# Wave_1|X_C|obfuscated). Used to report the true worst-case budget in the log.
WORST_PROMPT_TOKENS = 33097

# Checked against the actual chat templates on 2026-08-19, not assumed:
#   Qwen/Qwen3-32B   "enable_thinking" in template -> True
#   Qwen/QwQ-32B     "enable_thinking" in template -> False
# QwQ was in TOGGLEABLE. Jinja silently ignores unknown kwargs, so passing
# enable_thinking=False to QwQ does nothing and it reasons anyway -- a factor level
# that does not exist, whose rows would then truncate against the "off" cap and
# parse as failures. QwQ is reasoning-only, like the DeepSeek distills.
#
# The thinking factor is therefore carried WITHIN model by Qwen3-32B, and the
# reasoning-only models enter as a between-model comparison. Both arms survive; only
# the model that carries the toggle changed.
#
# 2026-08-20 roster expansion, for the generational time axis. Every entry below was
# placed by READING that model's chat_template.jinja on 2026-08-20, per the QwQ
# precedent above -- membership was never inferred from the model's name:
#   Qwen3.5-27B / 3.6-27B / 3.8-27B   "enable_thinking" present, defaults TRUE
#   Nemotron-3-Nano-30B-A3B           "enable_thinking" present
#   GLM-4.7-Flash                     "enable_thinking" present
#   gemma-4-31B-it                    DROPPED 2026-08-21. Its template does carry
#                                     enable_thinking, but vLLM 0.19.1 cannot build
#                                     its ModelConfig at all: gemma-4 has a
#                                     heterogeneous config and the convertor reads a
#                                     global head_dim, raising
#                                     AmbiguousGlobalPerLayerAttributeError. Being in
#                                     vLLM's arch registry is NOT the same as being
#                                     loadable -- Gemma4ForConditionalGeneration is
#                                     registered and still fails. Cost one GPU
#                                     allocation (job 16119739) before a CPU-side
#                                     create_model_config() probe was added.
#   Olmo-3.1-32B-Think                no toggle at all; <think> only -> ALWAYS
#
# TOGGLEABLE membership matters even though only the "on" arm is run: it is the sole
# reason chat_kwargs carries enable_thinking=True at all (run_batch below passes None
# for non-members). gemma-4 defaults to FALSE, so leaving it out would silently
# produce a non-reasoning run labelled thinking=on -- the F9 failure, again.
TOGGLEABLE = {
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.8-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "zai-org/GLM-4.7-Flash",
}
ALWAYS_THINKING = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/QwQ-32B",
    "allenai/Olmo-3.1-32B-Think",
}


def supports(model, thinking):
    if model in ALWAYS_THINKING:
        return thinking == "on"
    if model in TOGGLEABLE:
        return True
    return thinking == "off"


def probe_guided_decoding():
    """Can this vLLM constrain output to a JSON schema? Answered by import, here,
    on the machine that will actually run -- never inferred from a login node."""
    try:
        from vllm.sampling_params import GuidedDecodingParams   # noqa: F401
        return "guided_json"
    except Exception:
        try:
            from vllm import SamplingParams
            return "guided_json_legacy" if "guided_json" in SamplingParams.__init__.__code__.co_varnames else "prompt_only"
        except Exception:
            return "prompt_only"


def build_sampling_params(route, max_tokens):
    from vllm import SamplingParams
    kwargs = {"temperature": 0.0, "max_tokens": max_tokens}
    if route == "guided_json":
        from vllm.sampling_params import GuidedDecodingParams
        kwargs["guided_decoding"] = GuidedDecodingParams(json=CONSISTENCY_SCHEMA)
    elif route == "guided_json_legacy":
        kwargs["guided_json"] = CONSISTENCY_SCHEMA
    return SamplingParams(**kwargs)


def model_context_limit(model):
    """The model's own maximum context, or None if it cannot be read.

    MAX_MODEL_LEN was a single hardcoded 49152 for a roster whose context lengths
    differ: Qwen3-32B tops out at 40960, and vLLM refuses to start rather than risk
    RoPE positions past the trained range producing NaNs. Job 16059710 died on
    exactly that, at engine init, after the queue wait. Ask the model instead of
    assuming, and never set VLLM_ALLOW_LONG_MAX_MODEL_LEN -- silently wrong numbers
    are worse than a job that refuses to start.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        n = getattr(cfg, "max_position_embeddings", None)
        return int(n) if n else None
    except Exception as e:
        print(f"[xmodal] could not read context limit for {model}: "
              f"{type(e).__name__}; falling back to {MAX_MODEL_LEN}", flush=True)
        return None


def init_vllm(model, tensor_parallel_size):
    from vllm import LLM
    limit = model_context_limit(model)
    max_len = min(MAX_MODEL_LEN, limit) if limit else MAX_MODEL_LEN
    # WORST_PROMPT_TOKENS is measured, not guessed -- see the MAX_MODEL_LEN comment.
    # The old log line here divided by ~11k (the MEDIAN) and so reported a worst-case
    # budget roughly 22k tokens larger than the real one, in the job log, every run.
    worst_budget = max_len - WORST_PROMPT_TOKENS
    if limit and limit < MAX_MODEL_LEN:
        print(f"[xmodal] {model} caps context at {limit}; using max_model_len={max_len} "
              f"(requested {MAX_MODEL_LEN}).", flush=True)
    print(f"[xmodal] max_model_len={max_len}; worst measured prompt is "
          f"{WORST_PROMPT_TOKENS} tokens, so the budget on the WORST item is "
          f"{worst_budget} tokens against max_tokens={MAX_TOKENS['on']} "
          f"({'OK' if worst_budget >= MAX_TOKENS['on'] else 'SHORT -- long items will truncate'})",
          flush=True)
    # Gated Delta Net linear attention (Qwen3.5/3.6/3.8) JIT-compiles a FlashInfer
    # kernel on first prefill. FlashInfer 0.6.6 on the torch cluster is missing
    # flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh, so that compile
    # fails and the engine dies (job 16133218). vLLM's own log names the fix, and
    # Triton 3.6.0 is installed, so force the Triton/FLA path instead.
    #
    # Passed as a top-level kwarg on purpose: EngineArgs.gdn_prefill_backend is
    # copied into additional_config at arg_utils.py:1964-65, which is where
    # ChunkGatedDeltaRule actually reads it. Setting neither means "auto", which
    # picks the broken FlashInfer path on sm90. Harmless for non-GDN models -- the
    # op is never instantiated.
    gdn = os.environ.get("GDN_PREFILL_BACKEND", "triton")
    return LLM(
        model=model,
        gdn_prefill_backend=gdn,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_len,
        enforce_eager=True,
    )


def load_checkpoint(path):
    """Resume set keyed by (item_id, model, thinking). Jobs on this cluster get
    rescheduled two to four times, so resumability is load-bearing, not a nicety."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["item_id"], r["model"], r["thinking"]))
    return done


def append_result(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def upload_partial(args, out_path, done_n, total_n, final=False):
    """Push what exists so far to HF, in a separate process. Never fatal."""
    if not args.hf_dataset:
        return
    # The helper moved to shared/ in the eval->freegen refactor. It used to sit
    # beside this file, and the old single-path lookup then failed os.path.exists
    # and returned silently -- so EVERY upload, including the final one, became a
    # no-op that announced itself only as one "skipping" line in an 8h log. Search
    # both, current location first.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    candidates = [os.path.join(root, "shared", "upload_helper.py"),
                  os.path.join(here, "upload_helper.py")]
    helper = next((c for c in candidates if os.path.exists(c)), None)
    if helper is None:
        print(f"[xmodal] upload_helper.py not found in any of {candidates}; "
              f"NOT uploading -- results stay on disk at {out_path}", flush=True)
        return
    cmd = [sys.executable, helper,
           "--results_dir", args.output_dir,
           "--hf_dataset", args.hf_dataset,
           "--workspace", args.workspace,
           "--experiment", "pde-llm-eval",
           "--artifact_status", "final" if final else "partial",
           "--job_id", os.environ.get("SLURM_JOB_ID", "local:0"),
           "--cluster", "torch",
           "--dataset_file", os.path.basename(args.items)]
    if args.packages_dir:
        cmd += ["--packages_dir", args.packages_dir]
    print(f"[xmodal] uploading partial ({done_n}/{total_n}) -> {args.hf_dataset}", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"[xmodal][upload] {line}", flush=True)
        if r.returncode != 0:
            print(f"[xmodal] upload returned {r.returncode}; data is safe in "
                  f"{args.output_dir}", flush=True)
            for line in (r.stderr or "").strip().splitlines()[-5:]:
                print(f"[xmodal][upload-err] {line}", flush=True)
    except Exception as e:
        # An upload problem must never take the run down with it.
        print(f"[xmodal] upload raised {type(e).__name__}: {e}; data is safe in "
              f"{args.output_dir}", flush=True)


def run(args):
    # B1 (red team 2026-08-19): load_exec_trajectories() was defined and never
    # called, so ViewSources.exec_traj was always {} and the first T_exec item
    # raised KeyError out of build_messages, killing the whole arm. SKIP_EXEC
    # defaulting to true hid it right up until someone turned the rung on.
    exec_traj = load_exec_trajectories(args.exec_npz)
    print(f"[xmodal] T_exec trajectories loaded: {len(exec_traj)}", flush=True)
    items = load_items(args.items)
    if args.conditions:
        keep = set(args.conditions.split(","))
        items = [i for i in items if i["condition"] in keep]
    if args.skip_exec:
        items = [i for i in items if i["traj_level"] != "T_exec"]
    if args.systems:
        keep = set(args.systems.split(","))
        items = [i for i in items if i["gt_sample"] in keep]

    # B2 (red team 2026-08-19): the item set carries 128 X_T_exec items covering all
    # 32 systems, but the npz has 30 keys -- NavierStokes_3 (mpi4py) and
    # NavierStokes_4 (jax) failed to execute. consistency_prompts.py documented that
    # such items are "dropped rather than shown a substitute", but no code dropped
    # them, so they raised KeyError and killed the arm. Drop them here, loudly: a
    # silently narrowed condition is worse than a missing one.
    if not args.skip_exec:
        missing = sorted({i["gt_sample"] for i in items
                          if i["traj_level"] == "T_exec"
                          and i["gt_sample"] not in exec_traj})
        if missing:
            before = len(items)
            items = [i for i in items
                     if not (i["traj_level"] == "T_exec"
                             and i["gt_sample"] in missing)]
            print(f"[xmodal] DROPPED {before - len(items)} T_exec item(s) for "
                  f"{len(missing)} system(s) with no executed trajectory: "
                  f"{','.join(missing)}. The T_exec condition covers "
                  f"{32 - len(missing)}/32 systems.", flush=True)

    out_path = os.path.join(
        args.output_dir,
        f"{args.model.replace('/', '__')}__think_{args.thinking}__consistency.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    done = load_checkpoint(out_path)
    todo = [i for i in items
            if (i["item_id"], args.model, args.thinking) not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[xmodal] model={args.model} thinking={args.thinking}", flush=True)
    print(f"[xmodal] {len(items)} items, {len(done)} already done, {len(todo)} to run",
          flush=True)
    if not todo:
        print("[xmodal] nothing to do", flush=True)
        return 0

    route = probe_guided_decoding()
    print(f"[xmodal] structured-output route: {route}", flush=True)
    if route == "prompt_only":
        print("[xmodal] NOTE: no guided decoding on this runtime; relying on the "
              "prompt contract plus the regex cascade in parse_consistency.py. "
              "Parse-failure rate is recorded per row and must be reported.",
              flush=True)

    sources = ViewSources(args.multimodal, args.dataset, exec_traj)
    llm = init_vllm(args.model, args.tp)
    sampling = build_sampling_params(route, MAX_TOKENS[args.thinking])
    chat_kwargs = ({"enable_thinking": args.thinking == "on"}
                   if args.model in TOGGLEABLE else None)

    n_fail = 0
    last_upload_at = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        messages = [build_messages(i, sources) for i in batch]
        t0 = time.time()
        outputs = llm.chat(messages, sampling_params=sampling,
                           chat_template_kwargs=chat_kwargs)
        elapsed = time.time() - t0

        for item, out in zip(batch, outputs):
            text = out.outputs[0].text
            parsed = parse_consistency(text)
            scored = score_consistency(parsed, item)
            if parsed["parse_route"] == "failed":
                n_fail += 1
            append_result(out_path, {
                "item_id": item["item_id"],
                "model": args.model,
                "thinking": args.thinking,
                "gt_sample": item["gt_sample"],
                "condition": item["condition"],
                "corrupted_view": item["corrupted_view"],
                "traj_level": item["traj_level"],
                "names": item["names"],
                "order_seed": item["order_seed"],
                "outlier_slot": item["outlier_slot"],
                "slots": [item[f"slot_{k}"] for k in range(1, 5)],
                # Full text, never truncated -- the reasoning traces are evidence.
                "response": text,
                "finish_reason": out.outputs[0].finish_reason,
                "n_prompt_tokens": len(out.prompt_token_ids or []),
                "n_output_tokens": len(out.outputs[0].token_ids or []),
                "structured_route": route,
                **parsed,
                **scored,
            })
        done_n = start + len(batch)
        print(f"[xmodal] {done_n}/{len(todo)}  ({elapsed:.0f}s this batch, "
              f"{n_fail} parse failures so far)", flush=True)

        # Partial upload. A job with an 8h wall that shows nothing until it exits is
        # the failure mode the workspace rules exist to prevent: a systematic problem
        # in the prompt or the parse would not surface until the compute was spent.
        # The JSONL is appended per item and resume is keyed on
        # (item_id, model, thinking), so the data is already on disk -- this only
        # makes it visible. Runs as its own PROCESS on purpose: huggingface_hub
        # spawns threads and connections that kill vLLM's EngineCore subprocess if
        # they are created inside this one.
        if args.upload_every and done_n // args.upload_every > last_upload_at:
            last_upload_at = done_n // args.upload_every
            upload_partial(args, out_path, done_n, len(todo))

    print(f"[xmodal] complete -> {out_path}", flush=True)
    print(f"[xmodal] parse failures: {n_fail}/{len(todo)}", flush=True)
    if args.upload_every:
        upload_partial(args, out_path, len(todo), len(todo), final=True)
    return 0


def main():
    p = argparse.ArgumentParser(description="Cross-modal consistency generation")
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-32B"))
    p.add_argument("--thinking", choices=("on", "off"),
                   default=os.environ.get("THINKING", "off"))
    p.add_argument("--items", default=os.environ.get("ITEMS", DEFAULT_ITEMS))
    p.add_argument("--multimodal", default=os.environ.get("MULTIMODAL", MULTIMODAL_CSV))
    p.add_argument("--dataset", default=os.environ.get("DATASET", MOD_DATASET))
    p.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", "results/xmodal"))
    p.add_argument("--tp", type=int, default=int(os.environ.get("TP", "1")))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "64")))
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    p.add_argument("--conditions", default=os.environ.get("CONDITIONS", ""))
    p.add_argument("--systems", default=os.environ.get("SYSTEMS", ""))
    p.add_argument("--upload_every", type=int,
                   default=int(os.environ.get("UPLOAD_EVERY", "128")),
                   help="Upload the results so far every N items. 0 disables.")
    p.add_argument("--hf_dataset", default=os.environ.get("HF_DATASET", ""),
                   help="HF repo to append partial results to.")
    p.add_argument("--workspace", default=os.environ.get("WORK_DIR", os.getcwd()))
    p.add_argument("--packages_dir", default=os.environ.get("PACKAGES_DIR", ""))
    p.add_argument("--exec_npz", default=os.environ.get(
        "EXEC_NPZ", "data/exec_trajectories.npz"),
        help="T_exec arrays from the cpu_short re-execution job.")
    p.add_argument("--skip_exec", action="store_true",
                   default=os.environ.get("SKIP_EXEC", "") == "true",
                   help="drop T_exec items, which need the cpu_short job first")
    args = p.parse_args()

    if not supports(args.model, args.thinking):
        print(f"[xmodal] {args.model} does not support thinking={args.thinking}; "
              f"nothing to do", flush=True)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
