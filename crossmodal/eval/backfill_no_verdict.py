"""backfill_no_verdict.py — finish the draws that ran out of budget mid-reasoning.

A draw is unusable when the generation hit its token cap while still inside the
reasoning block: `finish_reason == "length"` and no closing `</think>`. The model
never reached an answer, so the row carries no verdict. Across the 2026-08 roster
that is 1,434 of 20,454 draws, concentrated in two models.

The obvious fix -- re-run the whole arm at a bigger budget -- regenerates thousands
of draws that were already fine. This script regenerates only the unusable ones.

WHY IT CONTINUES RATHER THAN RESTARTS
=====================================
Re-drawing a failed slot from scratch would bias the arm, and the bias runs the same
direction as the defect being fixed. The draws being kept are conditioned on having
been short enough to finish under the old cap; a fresh unconditional draw for an
empty slot is usually short too. The merged arm would then over-represent short
reasoning relative to the distribution it claims to sample -- and short reasoning is
exactly what the easy items produce.

Continuing from the truncated prefix has no such problem. The prefix was sampled at
the same temperature under the same prompt, so appending more tokens to it is simply
finishing an interrupted sample: the result is a draw from the model's distribution
under the LARGER cap, which is precisely what the arm is supposed to contain.

It is also far cheaper. The tokens already generated are re-read as prompt, not
re-decoded, and prefill is roughly an order of magnitude faster per token than
decode.

THE OTHER FAILURE: DECODE LOOPS
==============================
A draw that never terminates is not short of budget -- it is looping. Nemotron
repeats a 12-gram hundreds of times in its tails, and its loop rate is invariant to
the cap (21.2% at 32,768 and 21.2% at 131,072). Continuing such a draw buys more of
the same loop at the highest token cost in the arm, so continuation is the wrong
repair and is off by default.

`--redraw_loops` applies the repair that does fit: a FRESH sample of the same prompt
at the same sampling parameters, with only the seed moved. See redraw_looping() for
why that is not selecting on the outcome -- briefly, every figure already conditions
on "the draw did not loop" because looped draws carry no verdict and are dropped, so
resampling estimates the same conditional quantity and changes only the sample size.
It recovers COVERAGE, not correctness: if looping tracks item difficulty, the
conditioning is biased whether the slot is dropped or redrawn, and that caveat
travels with the result either way.

Output is a MERGED arm: every good row from the source copied verbatim, plus the
backfilled rows carrying provenance (`backfilled`, `backfill_from_tokens`,
`backfill_budget`). It is written to its own directory so the source arm is never
mutated, and it is resumable on `(item_id, sample_idx)` like the main runner.

    python -m crossmodal.eval.backfill_no_verdict \
        --model zai-org/GLM-4.7-Flash --thinking on \
        --source_dir outputs/xmodal_gen/glm-4-7-flash \
        --output_dir outputs/xmodal_gen/glm-4-7-flash-backfill \
        --max_tokens 131072
"""
import argparse
import collections
import json
import os
import sys
import time
import zlib

_here = os.path.dirname(os.path.abspath(__file__))
# Two layouts in play: the packaged repo (crossmodal/eval/) and the flat copy on the
# cluster (pde-llm-eval/eval/). Insert both roots and import whichever resolves, so
# this single file runs unmodified in both places.
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
sys.path.insert(0, os.path.dirname(_here))

try:                                                                  # packaged
    from crossmodal.eval import run_cross_modal_consistency as R      # noqa: E402
    from crossmodal.eval.parse_consistency import (                   # noqa: E402
        is_looping, parse_consistency, score_consistency)
except ImportError:                                                   # flat / cluster
    from crossmodal.eval import run_cross_modal_consistency as R                 # noqa: E402
    from crossmodal.eval.parse_consistency import (                              # noqa: E402
        is_looping, parse_consistency, score_consistency)


def needs_backfill(row):
    """True when this draw never reached an answer.

    Deliberately the SAME predicate the report uses to drop rows, so the set this
    script repairs and the set the report excludes cannot drift apart.
    """
    return (str(row.get("finish_reason")) == "length"
            and "</think>" not in str(row.get("response") or ""))


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# A redraw uses a seed far from the continuation seeds so the two repair paths can
# never collide on the same (item, sample_idx) and silently produce the same sample.
REDRAW_SEED_STRIDE = 1_000_003


def redraw_looping(llm, tok, items, sources, route, gen, chat_kwargs, args,
                   loops, done, out_path, max_len):
    """Replace a looped draw with a FRESH sample at a new seed.

    A decode loop is not a budget failure. The tail is a repeating n-gram and the
    rate is invariant to the cap -- Nemotron loops on 21.2% of draws at 32,768 and
    21.2% at 131,072 -- so continuing the prefix buys more of the same loop. The
    repair that fits the failure is another draw.

    WHY THIS IS NOT CHERRY-PICKING, AND WHAT IT DOES NOT FIX
    ========================================================
    Redrawing looks like selecting on the outcome, and it is worth being exact about
    why it is not. The estimand every figure in this report already uses is the
    verdict distribution GIVEN the draw did not loop: looped draws carry no verdict
    and are dropped everywhere. Resampling until a draw terminates estimates that
    same conditional distribution -- it changes the sample size, not the estimand.
    Dropping and redrawing are equally biased; redrawing merely stops throwing away
    the item's coverage.

    So this recovers COVERAGE, not correctness. If looping correlates with item
    difficulty, conditioning on non-looped draws is biased whichever route is taken,
    and that caveat has to travel with the result either way.

    What would be cherry-picking: drawing n>1 and keeping whichever sample gave the
    nicer answer, or retrying until a draw agrees with the other two. Neither happens
    here -- n=1, the first terminating sample is taken, and every attempt is recorded
    on the row.
    """
    stats = collections.Counter()
    pending = [r for r in loops if (r["item_id"], r["sample_idx"]) not in done]
    if not pending:
        return stats
    print(f"[backfill] redrawing {len(pending)} looped draw(s), "
          f"up to {args.redraw_attempts} attempt(s) each", flush=True)

    for attempt in range(1, args.redraw_attempts + 1):
        if not pending:
            break
        nxt = []
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            prompts, params, kept = [], [], []
            for r in chunk:
                item = items.get(r["item_id"])
                if item is None:
                    continue
                messages = R.build_messages(item, sources)
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **(chat_kwargs or {}))
                room = max_len - len(tok(prompt).input_ids)
                budget = min(args.max_tokens, room)
                if budget < 256:
                    stats["no_room"] += 1
                    continue
                prompts.append(prompt)
                kept.append(r)
                # Same prompt and same sampling parameters as the original run --
                # only the seed moves. A different temperature or a repetition
                # penalty would make this a different treatment, not a redraw.
                params.append(R.build_sampling_params(
                    route, budget, gen=gen, n=1,
                    seed=args.seed + REDRAW_SEED_STRIDE * attempt
                         + int(r["sample_idx"])))
            if not prompts:
                continue
            t0 = time.time()
            outputs = llm.generate(prompts, params)
            elapsed = time.time() - t0

            for r, out in zip(kept, outputs):
                cand = out.outputs[0]
                text = cand.text
                looping = is_looping(text)
                # Retry on NO VERDICT, not on "looping". Gating the retry on the loop
                # detector meant a redraw that ran to the cap while still producing
                # novel text -- rumination, no repeated 12-gram -- was accepted on
                # attempt 1 and never resampled. That is precisely the residual this
                # pass exists to clear: GLM stalled at 62 draws, every one of them at
                # exactly 65,536 output tokens with is_looping() False, so raising
                # --redraw_attempts alone would have changed nothing. A draw that
                # never reached an answer is a draw worth another sample, whatever
                # shape its failure took.
                # The LAST attempt is written whatever it looks like, so a slot that
                # never lands ends up in the arm as a no-verdict row rather than
                # vanishing.
                no_verdict = (str(cand.finish_reason) == "length"
                              and "</think>" not in text)
                if no_verdict and attempt < args.redraw_attempts:
                    nxt.append(r)
                    continue
                item = items.get(r["item_id"])
                parsed = parse_consistency(text)
                scored = score_consistency(parsed, item) if item else {}
                new = dict(r)
                new.update(parsed)
                new.update(scored)
                new.update({
                    "response": text,
                    "finish_reason": cand.finish_reason,
                    "n_output_tokens": len(cand.token_ids or []),
                    "redrawn": True,
                    "redraw_attempt": attempt,
                    "redraw_seed": (args.seed + REDRAW_SEED_STRIDE * attempt
                                    + int(r["sample_idx"])),
                    # The discarded sample's length, kept so the artifact records
                    # that a looped draw was replaced and how big it was.
                    "redraw_discarded_tokens": r.get("n_output_tokens") or 0,
                    "redraw_still_looping": bool(looping),
                })
                R.append_result(out_path, new)
                stats["written"] += 1
                if looping:
                    stats["still_looping"] += 1
                elif needs_backfill(new):
                    stats["still_no_verdict"] += 1
                else:
                    stats["recovered"] += 1
            print(f"[backfill] redraw attempt {attempt}: "
                  f"{start + len(chunk)}/{len(pending)} "
                  f"({elapsed:.0f}s this batch, {stats['recovered']} recovered)",
                  flush=True)
        pending = nxt
    return stats


def main():
    p = argparse.ArgumentParser(description="Backfill no-verdict draws by continuation")
    p.add_argument("--model", default=os.environ.get("MODEL", ""))
    p.add_argument("--thinking", choices=("on", "off"),
                   default=os.environ.get("THINKING", "on"))
    p.add_argument("--source_dir", default=os.environ.get("SOURCE_DIR", ""),
                   help="arm to repair; read-only, never mutated")
    p.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", ""))
    p.add_argument("--max_tokens", type=int,
                   default=int(os.environ.get("BACKFILL_MAX_TOKENS", "131072")),
                   help="TOTAL budget per draw, counting the tokens already generated")
    p.add_argument("--items", default=os.environ.get("ITEMS", R.DEFAULT_ITEMS))
    p.add_argument("--multimodal", default=os.environ.get("MULTIMODAL", R.MULTIMODAL_CSV))
    p.add_argument("--dataset", default=os.environ.get("DATASET", R.MOD_DATASET))
    p.add_argument("--exec_npz", default=os.environ.get(
        "EXEC_NPZ", "data/exec_trajectories.npz"))
    p.add_argument("--tp", type=int, default=int(os.environ.get("TP", "1")))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")),
                   help="repair at most N draws. For canaries.")
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SAMPLING_SEED", "20260822")))
    p.add_argument("--upload_every", type=int,
                   default=int(os.environ.get("UPLOAD_EVERY", "128")))
    p.add_argument("--hf_dataset", default=os.environ.get("HF_DATASET", ""))
    p.add_argument("--workspace", default=os.environ.get("WORK_DIR", os.getcwd()))
    p.add_argument("--packages_dir", default=os.environ.get("PACKAGES_DIR", ""))
    p.add_argument("--repair_loops", action="store_true",
                   default=os.environ.get("REPAIR_LOOPS", "") == "1",
                   help="also continue draws whose tail is a decode loop. OFF by "
                        "default: those draws did not run out of budget, they "
                        "stopped terminating, and continuing one to a larger cap "
                        "buys more of the same loop at the highest token cost in "
                        "the arm -- they are the LONGEST draws precisely because "
                        "they never stop.")
    p.add_argument("--redraw_loops", action="store_true",
                   default=os.environ.get("REDRAW_LOOPS", "") == "1",
                   help="replace looped draws with a FRESH sample at a new seed "
                        "instead of copying them through. A loop is not a budget "
                        "failure, so continuing its prefix buys more loop; another "
                        "draw is the repair that fits. Recovers coverage, not "
                        "correctness -- see redraw_looping().")
    p.add_argument("--num_shards", type=int,
                   default=int(os.environ.get("NUM_SHARDS", "1")),
                   help="split the repair TODO list across this many jobs")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", "0")),
                   help="which slice this job repairs, 0-based")
    p.add_argument("--redraw_attempts", type=int,
                   default=int(os.environ.get("REDRAW_ATTEMPTS", "2")),
                   help="max fresh samples per looped draw. The last attempt is "
                        "written whatever it looks like, so a slot that keeps "
                        "looping stays in the arm as a no-verdict row.")
    p.add_argument("--redraw_only", action="store_true",
                   default=os.environ.get("REDRAW_ONLY", "") == "1",
                   help="redraw EVERY no-verdict draw; continue none. For a model "
                        "whose truncations are rumination rather than budget: GLM "
                        "recovered 0 of 16 when continued from 32,768 to 65,536, "
                        "and its tails show the model had already NAMED the outlier "
                        "and kept second-guessing it (tail vocabulary 16-38% unique, "
                        "no exact 12-gram repeat so is_looping misses it). More "
                        "budget cannot finish a trace that has nothing left to say.")
    p.add_argument("--dry_run", action="store_true",
                   default=os.environ.get("DRY_RUN", "") == "1",
                   help="report what would be repaired and exit without loading a model")
    args = p.parse_args()

    src_name = (f"{args.model.replace('/', '__')}"
                f"__think_{args.thinking}__consistency.jsonl")
    src_path = os.path.join(args.source_dir, src_name)
    out_path = os.path.join(args.output_dir, src_name)
    if not os.path.exists(src_path):
        print(f"[backfill] source not found: {src_path}", flush=True)
        return 2

    rows = load_rows(src_path)
    good = [r for r in rows if not needs_backfill(r)]
    no_verdict = [r for r in rows if needs_backfill(r)]
    loops = [r for r in no_verdict if is_looping(r.get("response"))]
    budget = [r for r in no_verdict if not is_looping(r.get("response"))]
    bad = no_verdict if args.repair_loops else budget
    # Three dispositions for a looped draw, and only one of them can apply:
    #   --repair_loops  continue its prefix (rarely useful; the loop resumes)
    #   --redraw_loops  fresh sample at a new seed  <- the repair that fits
    #   neither         copied through unchanged, reported as a loop
    redraw_set = loops if (args.redraw_loops and not args.repair_loops) else []
    if args.redraw_only:
        bad, redraw_set = [], list(no_verdict)

    # Shard the TODO list, not the source arm. Each shard reads the whole arm -- it
    # needs the good rows to copy through -- but repairs only its own slice and
    # writes to its own directory, so several GPUs can work one model without two
    # writers on one JSONL. consolidate_arms.py merges the slices back on
    # (item_id, sample_idx) afterwards, so nothing downstream has to know.
    # Sliced by a STABLE key rather than list position: the todo list is rebuilt on
    # every resume and its order can shift, which would silently reassign draws
    # between shards mid-run and leave some never attempted.
    if args.num_shards > 1:
        def _mine(r):
            # zlib.crc32, NOT hash(): Python randomizes string hashing per
            # process (PYTHONHASHSEED), so hash() would put a draw in a different
            # shard in every job AND on every resume -- shards would double up on
            # some draws and leave others permanently unclaimed, the exact failure
            # this split exists to avoid. crc32 over the same bytes is stable
            # across processes, machines and Python versions.
            key = f"{r.get('item_id')}|{int(r.get('sample_idx') or 0)}".encode()
            return zlib.crc32(key) % args.num_shards == args.shard
        n_before = len(bad) + len(redraw_set)
        bad = [r for r in bad if _mine(r)]
        redraw_set = [r for r in redraw_set if _mine(r)]
        print(f"[backfill] shard {args.shard}/{args.num_shards}: "
              f"{len(bad) + len(redraw_set)} of {n_before} draw(s) are mine",
              flush=True)
    skipped_loops = ([] if (args.repair_loops or args.redraw_loops) else loops)
    print(f"[backfill] {src_path}", flush=True)
    print(f"[backfill] {len(rows)} rows: {len(good)} keep, "
          f"{len(no_verdict)} without a verdict = {len(budget)} token-budget "
          f"+ {len(loops)} decode loop", flush=True)
    if redraw_set:
        disposition = (f"; {len(redraw_set)} draw(s) get a fresh sample at a new "
                       f"seed, up to {args.redraw_attempts} attempt(s)"
                       + (" [redraw_only: continuing none]" if args.redraw_only
                          else " [decode loops]"))
    elif args.repair_loops:
        disposition = ""
    else:
        disposition = (f"; the {len(loops)} loop(s) are copied through unchanged "
                       f"(--redraw_loops to resample them)")
    print(f"[backfill] repairing {len(bad)} draw(s)" + disposition, flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # Resume on the same key the merged file is written under, so a rescheduled job
    # neither duplicates a repaired draw nor re-copies a kept one.
    done = set()
    if os.path.exists(out_path):
        for r in load_rows(out_path):
            done.add((r["item_id"], r["sample_idx"]))
        print(f"[backfill] resuming: {len(done)} rows already in {out_path}",
              flush=True)

    if not bad and not redraw_set:
        # Still write the merged arm, and do it WITHOUT loading a model: with loops
        # skipped, an arm whose only broken draws were loops has nothing to generate,
        # and paying a 60 GB model load to copy rows would be pure waste.
        if not args.dry_run:
            for r in good:
                if (r["item_id"], r["sample_idx"]) not in done:
                    R.append_result(out_path, r)
            for r in skipped_loops:
                if (r["item_id"], r["sample_idx"]) not in done:
                    R.append_result(out_path, dict(r, backfilled=False,
                                                   backfill_skipped="decode loop"))
            print(f"[backfill] nothing to generate; merged arm -> {out_path}",
                  flush=True)
            if args.upload_every:
                R.upload_partial(args, out_path, 0, 0, final=True)
        else:
            print("[backfill] nothing to generate (dry run)", flush=True)
        return 0

    spent = [r.get("n_output_tokens") or 0 for r in bad] or [0]
    print(f"[backfill] continuation budget: {args.max_tokens} total per draw, "
          f"already spent median {sorted(spent)[len(spent) // 2]}, "
          f"so ~{args.max_tokens - sorted(spent)[len(spent) // 2]} new tokens each",
          flush=True)
    if args.dry_run:
        print("[backfill] dry run; no model loaded", flush=True)
        return 0

    n_copied = 0
    for r in good:
        if (r["item_id"], r["sample_idx"]) not in done:
            R.append_result(out_path, r)
            n_copied += 1
    print(f"[backfill] copied {n_copied} good rows through unchanged", flush=True)

    todo = [r for r in bad if (r["item_id"], r["sample_idx"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    redraw_todo = [r for r in redraw_set
                   if (r["item_id"], r["sample_idx"]) not in done]
    if not todo and not redraw_todo:
        print("[backfill] every repairable draw is already done", flush=True)
        return 0

    # Rebuild the prompts through the SAME path the original run used, so the
    # continuation sees a byte-identical prefix. Anything else risks continuing a
    # different prompt than the one the prefix was sampled under.
    exec_traj = R.load_exec_trajectories(args.exec_npz)
    items = {i["item_id"]: i for i in R.load_items(args.items)}
    sources = R.ViewSources(args.multimodal, args.dataset, exec_traj)

    # init_vllm sizes the context window from gen_budget(model), i.e. the budget the
    # ORIGINAL run used. Continuing to a larger total needs a larger window, so declare
    # the backfill budget as this model's budget before the engine is built. Without
    # this the window stays at 69632 and every continuation is clipped at ~36k -- the
    # exact failure the init_vllm comment warns about, reintroduced from the other side.
    R.MAX_TOKENS_BY_MODEL[args.model] = args.max_tokens

    llm = R.init_vllm(args.model, args.tp, args.thinking)
    tok = llm.get_tokenizer()
    # The window the engine ACTUALLY got, recomputed the same way init_vllm did. Some
    # models clamp it: QwQ-32B and Qwen3-32B declare 40960 and cannot be handed 131072
    # no matter what is asked for, so their truncations were context-bound rather than
    # budget-bound and there is little room to continue into. Rows with no room are
    # reported and left alone rather than being "repaired" into a second truncation.
    _ctx = R.model_context_limit(args.model)
    _want = max(R.MAX_MODEL_LEN, R.WORST_PROMPT_TOKENS + args.max_tokens)
    max_len = min(_want, _ctx) if _ctx else _want
    print(f"[backfill] engine window {max_len} tokens"
          + (f" (clamped by the model's declared {_ctx})" if _ctx and _ctx < _want else ""),
          flush=True)
    route = R.probe_guided_decoding()
    gen = dict(R.UNIFORM_SAMPLING)
    chat_kwargs = ({"enable_thinking": args.thinking == "on"}
                   if args.model in R.TOGGLEABLE else None)
    print(f"[backfill] {len(todo)} draws to continue | route={route} | "
          f"sampling={gen}", flush=True)

    n_done = n_still_short = n_still_loop = 0
    last_upload_at = 0
    # Draws with no window left to continue into. Copied through UNCHANGED so the
    # merged arm still contains every draw, and counted separately so the report of
    # what was repaired cannot quietly include them.
    no_room = []
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        prompts, params, kept = [], [], []
        for r in batch:
            item = items.get(r["item_id"])
            if item is None:
                continue
            messages = R.build_messages(item, sources)
            prefix = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **(chat_kwargs or {}))
            full = prefix + str(r.get("response") or "")
            # Measure the real position in the window rather than assuming
            # prompt + n_output_tokens: the stored count is the ORIGINAL run's, and
            # re-tokenizing the concatenation is what the engine will actually see.
            used = len(tok(full).input_ids)
            room = max_len - used
            budget_left = args.max_tokens - (r.get("n_output_tokens") or 0)
            remaining = min(room, budget_left)
            if remaining < 256:
                no_room.append(r)
                continue
            prompts.append(full)
            kept.append(r)
            # n=1: this repairs ONE specific draw, identified by sample_idx. Drawing
            # k and picking one would be selecting on the outcome.
            params.append(R.build_sampling_params(
                route, remaining, gen=gen, n=1,
                seed=args.seed + int(r["sample_idx"])))

        t0 = time.time()
        if not prompts:
            continue
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0

        for r, out in zip(kept, outputs):
            cand = out.outputs[0]
            merged = str(r.get("response") or "") + cand.text
            parsed = parse_consistency(merged)
            item = items.get(r["item_id"])
            scored = score_consistency(parsed, item) if item else {}
            new = dict(r)
            new.update(parsed)
            new.update(scored)
            new.update({
                "response": merged,
                "finish_reason": cand.finish_reason,
                "n_output_tokens": (r.get("n_output_tokens") or 0) + len(cand.token_ids or []),
                # Provenance: this row is not a single uninterrupted sample, and the
                # artifact must say so rather than presenting it as one.
                "backfilled": True,
                "backfill_from_tokens": r.get("n_output_tokens") or 0,
                "backfill_budget": args.max_tokens,
                "backfill_seed": args.seed + int(r["sample_idx"]),
            })
            R.append_result(out_path, new)
            n_done += 1
            if needs_backfill(new):
                n_still_short += 1
                if is_looping(merged):
                    n_still_loop += 1

        print(f"[backfill] {start + len(batch)}/{len(todo)} "
              f"({elapsed:.0f}s this batch, {n_still_short} still without a verdict, "
              f"{n_still_loop} of those still looping)", flush=True)

        if args.upload_every and n_done // args.upload_every > last_upload_at:
            last_upload_at = n_done // args.upload_every
            R.upload_partial(args, out_path, n_done, len(todo))

    # A draw with no window left to continue into is unfixable BY CONTINUATION, but
    # not unfixable: a fresh sample is stochastic and may simply come in shorter.
    # QwQ-32B and Qwen3-32B declare a 40,960-token context, so every one of their
    # budget-hit draws lands here -- 7 and 0 respectively -- and reporting them as
    # permanently lost was wrong. Redraw is the route that reaches them.
    if args.redraw_loops and no_room:
        print(f"[backfill] {len(no_room)} no-room draw(s) routed to redraw: "
              f"continuation cannot reach them, a fresh sample can", flush=True)
        redraw_todo = redraw_todo + [r for r in no_room
                                     if (r['item_id'], r['sample_idx']) not in done]
        no_room = []

    redraw_stats = collections.Counter()
    if redraw_todo:
        redraw_stats = redraw_looping(llm, tok, items, sources, route, gen,
                                      chat_kwargs, args, redraw_todo, done,
                                      out_path, max_len)
        print(f"[backfill] redraw: {redraw_stats['recovered']} recovered of "
              f"{redraw_stats['written']} rewritten "
              f"({redraw_stats['still_looping']} still looping, "
              f"{redraw_stats['still_no_verdict']} ran out of budget, "
              f"{redraw_stats['no_room']} had no room)", flush=True)

    for r, why in ([(x, "no context room") for x in no_room]
                   + [(x, "decode loop") for x in skipped_loops]):
        if (r["item_id"], r["sample_idx"]) not in done:
            R.append_result(out_path, dict(r, backfilled=False,
                                           backfill_skipped=why))
    if no_room:
        print(f"[backfill] {len(no_room)} draw(s) had no window left to continue "
              f"into and were copied through unchanged", flush=True)

    recovered = n_done - n_still_short
    print(f"[backfill] repaired {recovered}/{n_done} draws "
          f"({100 * recovered / max(1, n_done):.1f}%); {n_still_short} still have no "
          f"verdict, {n_still_loop} of them looping", flush=True)
    print(f"[backfill] merged arm -> {out_path}", flush=True)
    if args.upload_every:
        R.upload_partial(args, out_path, n_done, len(todo), final=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
