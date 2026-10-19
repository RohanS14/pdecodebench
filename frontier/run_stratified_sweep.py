"""
run_stratified_sweep.py — orchestrates the stratified agentic run described
in the stratified-subset plan: a shared Stage-1 cache pass (once per row,
at the SAME thinking setting as the Stage 2 it feeds) followed by that
condition's Stage-2 sweep, repeated per requested condition,
each with its own output JSONL, its own per-row logs/ directory, its own
checkpoint/resume, and its own cost cap.

Output layout (default --output_root results/frontier/stratified_64):
  <output_root>/<model_slug>/<condition>/stage1_cache/<slug>__stage1_cache.jsonl
  <output_root>/<model_slug>/<condition>/<slug>__belief_revision_agentic.jsonl (+ logs/)

where <condition> names the thinking setting explicitly (e.g. "tl_high" for
gemini-3.7-flash at thinking_level=high, "tb0" for gemini-2.5-flash at
thinking_budget=0). Paths are keyed by model so two models' sweeps can never
collide or be accidentally pooled -- they are separate experiments.

Usage:
  .venv/bin/python frontier/run_stratified_sweep.py \\
      --dataset results/frontier/stratified_64/dataset.csv \\
      --output_root results/frontier/stratified_64 \\
      --run_id_prefix stratified64
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frontier.run_belief_revision import (  # noqa: E402
    RateLimiter,
    append_result,
    estimate_cost_usd,
    load_checkpoint,
)
from frontier.run_belief_revision_agentic import (  # noqa: E402
    INVESTIGATIVE_BUDGET_DEFAULT,
    PROMPT_S1_AGENTIC,
    SUBPROCESS_TIMEOUT_DEFAULT,
    TRUNCATE_CHARS_DEFAULT,
    EPISODE_COST_CAP_DEFAULT,
    run_stage1,
    run_stage2_and_score,
)
from frontier.model_pricing import (  # noqa: E402
    condition_slug,
    estimate_prompt_cost,
    parse_thinking_setting,
)
from frontier.episode_log import write_episode_log  # noqa: E402
from shared.dataset_io import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified agentic sweep: per-condition Stage-1 cache + Stage-2 sweep, one pass per thinking condition")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--dataset", default=str(REPO_ROOT / "results" / "frontier" / "stratified_64" / "dataset.csv"))
    p.add_argument("--output_root", default=str(REPO_ROOT / "results" / "frontier" / "stratified_64"))
    p.add_argument("--run_id_prefix", required=True)
    # str, not int: a Gemini 3 condition is a level name, a 2.5 condition is a
    # numeric budget. No default -- the right conditions differ per model
    # family (2.5 has no "minimal"/level concept; 3.7 cannot disable thinking
    # and rejects "minimal" outright), so guessing here would silently run the
    # wrong experiment.
    p.add_argument("--thinking_settings", nargs="+", required=True,
                   help="Stage-2 conditions to sweep, in order. Gemini 2.5: numeric "
                        "thinking_budget(s), e.g. 0 1536. Gemini 3.x: level name(s), "
                        "e.g. high, or 'low high'.")
    p.add_argument("--budget", type=int, default=INVESTIGATIVE_BUDGET_DEFAULT)
    p.add_argument("--truncate_chars", type=int, default=TRUNCATE_CHARS_DEFAULT)
    p.add_argument("--subprocess_timeout", type=int, default=SUBPROCESS_TIMEOUT_DEFAULT)
    p.add_argument("--episode_cost_cap_usd", type=float, default=EPISODE_COST_CAP_DEFAULT,
                   help="Omit to derive the cap from this model's real pricing "
                        "(a flat dollar cap gives a pricier model less runway for the "
                        "same work).")
    p.add_argument("--max_cost_usd_stage1", type=float, default=0.5,
                   help="Session-level cap for the shared Stage-1 pass.")
    p.add_argument("--max_cost_usd_per_condition", type=float, default=3.0,
                   help="Session-level cap for EACH Stage-2 sweep independently (not shared across conditions).")
    p.add_argument("--rpm", type=float, default=8.0)
    p.add_argument("--limit", type=int, default=0, help="Cap total rows processed per phase (for testing).")
    p.add_argument("--max_retries", type=int, default=4)
    return p.parse_args()


# The old naming derived "nothink"/"think" from `thinking_budget == 0`, which
# cannot express a Gemini 3 condition at all (thinking cannot be disabled there,
# and the setting is a level name, not a number). condition_slug() names the
# actual setting, so a directory is self-describing.


def main() -> None:
    args = parse_args()

    try:
        workspace_root = REPO_ROOT.parent.parent  # mlproj -> private_projects -> raca-torch
        sys.path.insert(0, str(workspace_root / "packages" / "key_handler" / "key_handler"))
        from key_handler import KeyHandler
        KeyHandler.set_env_key()
    except ImportError:
        pass

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        sys.exit("[stratified-sweep] ERROR: GOOGLE_API_KEY not set (checked KeyHandler and os.environ).")

    from google import genai
    client = genai.Client(api_key=api_key)

    slug = args.model.replace("/", "__").replace("-", "").replace(".", "")
    # Output paths are keyed by model so two models' sweeps can never collide or
    # be accidentally pooled -- they are separate experiments, not one dataset.
    output_root = Path(args.output_root) / slug
    thinking_settings = [parse_thinking_setting(args.model, raw)
                         for raw in args.thinking_settings]
    print(f"[stratified-sweep] Model: {args.model}  conditions: "
          f"{[condition_slug(args.model, t) for t in thinking_settings]}", flush=True)
    print(f"[stratified-sweep] Output root: {output_root}", flush=True)

    df = load_dataset(args.dataset)
    print(f"[stratified-sweep] Dataset: {len(df)} rows from {args.dataset}", flush=True)

    # Each thinking condition is a fully independent pass: its own Stage-1 cache
    # (same setting as its Stage 2, so the two stages differ only in tool access)
    # and its own Stage-2 checkpoint and cost cap.
    for thinking_setting in thinking_settings:
        cond = condition_slug(args.model, thinking_setting)
        cond_dir = output_root / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: Stage-1 pass for THIS condition ─────────────────────────
        # Keyed by condition, not shared across all of them: Stage 1 must run at
        # the same thinking setting as the Stage 2 it feeds, or "does
        # investigation help" compares low-thinking-without-tools against
        # high-thinking-with-tools -- two variables at once.
        #
        # Caching is still right, for a different reason than before: on Gemini 3
        # nothing is deterministic (temperature is stripped, and confirmed live
        # that temperature=0 is not honored anyway), so the cache is what
        # guarantees every Stage-2 row sees the SAME Stage-1 draw.
        stage1_dir = cond_dir / "stage1_cache"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage1_path = stage1_dir / f"{slug}__stage1_cache.jsonl"

        done1 = load_checkpoint(stage1_path)
        todo1 = df[~df["title"].isin(done1)].reset_index(drop=True)
        print(f"[stratified-sweep] [{cond}] Stage 1: {len(todo1)} rows to process "
              f"(of {len(df)} total)", flush=True)

        limiter1 = RateLimiter(args.rpm)
        stage1_session_cost = 0.0
        new1 = 0
        for _, row in todo1.iterrows():
            if args.limit and new1 >= args.limit:
                print(f"[stratified-sweep] [{cond}] Stage 1 --limit {args.limit} reached.", flush=True)
                break
            est = estimate_prompt_cost(args.model, PROMPT_S1_AGENTIC.format(code=str(row["code"])))
            if stage1_session_cost + est > args.max_cost_usd_stage1:
                print(f"[stratified-sweep] [{cond}] Stage 1 cost limit "
                      f"${args.max_cost_usd_stage1:.2f} reached. Stopping Stage 1.", flush=True)
                break
            print(f"  [{cond} stage1 {new1+1}] {row['title']}", end=" ", flush=True)
            limiter1.wait()
            result1 = run_stage1(client, args.model, row.to_dict(),
                                 max_retries=args.max_retries,
                                 thinking_setting=thinking_setting)
            stage1_session_cost += result1["s1_cost_usd"]
            append_result(stage1_path, result1)
            new1 += 1
            print(f"valid={result1['s1_parsed_valid']} match={result1['s1_valid_match']} "
                  f"score={result1['s1_validity_score']} ${result1['s1_cost_usd']:.5f}", flush=True)

        print(f"[stratified-sweep] [{cond}] Stage 1 done: {new1} new rows -> {stage1_path}", flush=True)
        print(f"[stratified-sweep] [{cond}] Stage 1 session cost: ${stage1_session_cost:.4f}", flush=True)

        # Load the full Stage-1 cache for this condition (not just this
        # session's new rows), keyed by title.
        stage1_cache: dict[str, dict] = {}
        if stage1_path.exists():
            with open(stage1_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        stage1_cache[r["title"]] = r

        missing = [t for t in df["title"] if t not in stage1_cache]
        if missing:
            print(f"[stratified-sweep] [{cond}] WARNING: {len(missing)} row(s) have no "
                  f"Stage-1 cache entry (likely --limit or cost cap stopped Stage 1 early) "
                  f"-- these will be skipped in Stage 2 until Stage 1 is completed for "
                  f"them: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)

        # ── Phase 2: Stage-2 sweep for THIS condition ────────────────────────
        out_path = cond_dir / f"{slug}__belief_revision_agentic.jsonl"

        done2 = load_checkpoint(out_path)
        rows_available = [t for t in df["title"] if t in stage1_cache and t not in done2]
        print(f"[stratified-sweep] [{cond}] Stage 2: {len(rows_available)} rows to process", flush=True)

        limiter2 = RateLimiter(args.rpm)
        session_cost = 0.0
        new2 = 0
        truncated_turns = 0

        for title in rows_available:
            if args.limit and new2 >= args.limit:
                print(f"[stratified-sweep] [{cond}] Stage 2 --limit {args.limit} reached.", flush=True)
                break
            row = df[df["title"] == title].iloc[0].to_dict()
            # Per-episode estimate: one Stage-1-sized prompt per turn, across the
            # investigative budget plus the submit turns. Thinking is billed at
            # the output rate, so there is no separate thinking margin to add.
            est = estimate_prompt_cost(
                args.model, PROMPT_S1_AGENTIC.format(code=str(row["code"]))
            ) * (args.budget + 2)
            if session_cost + est > args.max_cost_usd_per_condition:
                print(f"[stratified-sweep] [{cond}] Stage 2 cost limit "
                      f"${args.max_cost_usd_per_condition:.2f} reached. "
                      f"Stopping this condition.", flush=True)
                break

            print(f"  [{cond} {new2+1}] {title}", end=" ", flush=True)
            limiter2.wait()
            run_id = f"{args.run_id_prefix}_{cond}"
            result = run_stage2_and_score(
                client, args.model, row, stage1_cache[title], run_id=run_id,
                budget=args.budget, truncate_chars=args.truncate_chars,
                subprocess_timeout=args.subprocess_timeout,
                episode_cost_cap_usd=args.episode_cost_cap_usd,
                thinking_setting=thinking_setting, max_retries=args.max_retries,
            )
            session_cost += result["total_cost_usd"]
            # Aborted rows (disk-safety guard fired -- see
            # agentic_sandbox.run_python_file / run_belief_revision_agentic's
            # abort propagation) are still appended to the checkpoint so
            # they are NOT silently auto-retried on the next sweep resume --
            # a human must inspect the episode dir and clear this row from
            # the checkpoint manually before it will be re-attempted.
            append_result(out_path, result)
            log_path = write_episode_log(result, str(cond_dir), run_id)
            new2 += 1
            print(f"log: {log_path}", flush=True)

            # Truncated turns are labelled, not silently reclassified as the
            # model declining to act. Counted per condition because the rate is
            # the thing that matters: it was ~1.6% on 2.5 at zero thinking, and
            # thinking competes for the same max_output_tokens budget.
            n_trunc = sum(1 for a in (result.get("action_trace") or []) if a.get("truncated"))
            truncated_turns += n_trunc

            if result.get("aborted"):
                print(
                    f"    [stratified-sweep] ABORTED (disk-safety): {title} -- "
                    f"{result.get('abort_reason')} -- inspect "
                    f"{result.get('episode_dir')} (and its _snapshots/quarantine/ "
                    f"if present) before re-running this row.",
                    flush=True,
                )
            else:
                print(
                    f"    actions={result['s2_action_count']} "
                    f"remaining_at_submit={result['actions_remaining_at_submission']} "
                    f"valid_match(s1->s2)={result['s1_valid_match']}->{result['s2_valid_match']} "
                    f"score(s1->s2)={result['s1_validity_score']}->{result['s2_validity_score']} "
                    f"${result['total_cost_usd']:.5f}"
                    + (f" [{n_trunc} TRUNCATED turn(s)]" if n_trunc else ""),
                    flush=True,
                )

        print(f"[stratified-sweep] [{cond}] Stage 2 done: {new2} new rows -> {out_path}", flush=True)
        print(f"[stratified-sweep] [{cond}] Stage 2 session cost: ${session_cost:.4f}", flush=True)
        if truncated_turns:
            print(f"[stratified-sweep] [{cond}] {truncated_turns} turn(s) hit "
                  f"max_output_tokens across this condition -- these are labelled "
                  f"truncated in the action trace, NOT model non-compliance.", flush=True)


if __name__ == "__main__":
    main()
