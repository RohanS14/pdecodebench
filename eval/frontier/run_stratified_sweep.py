"""
run_stratified_sweep.py — orchestrates the stratified agentic run described
in the stratified-subset plan: a shared Stage-1 cache pass (once per row,
since Stage 1 is 100% independent of thinking_budget) followed by two
Stage-2 sweeps (thinking_budget=0 and thinking_budget=1536 by default),
each with its own output JSONL, its own per-row logs/ directory, its own
checkpoint/resume, and its own cost cap.

Output layout (default --output_root results/frontier/stratified_64):
  <output_root>/stage1_cache/<slug>__stage1_cache.jsonl
  <output_root>/nothink/<slug>__belief_revision_agentic.jsonl (+ logs/)
  <output_root>/think/<slug>__belief_revision_agentic.jsonl (+ logs/)

Usage:
  eval/.venv/bin/python eval/frontier/run_stratified_sweep.py \\
      --dataset results/frontier/stratified_64/dataset.csv \\
      --output_root results/frontier/stratified_64 \\
      --run_id_prefix stratified64
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

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
    _PRICE_THINK,
    EPISODE_COST_CAP_DEFAULT,
    run_stage1,
    run_stage2_and_score,
)
from frontier.episode_log import write_episode_log  # noqa: E402
from dataset_io import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified agentic sweep: shared Stage-1 cache + 2 thinking-condition Stage-2 sweeps")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--dataset", default=str(REPO_ROOT / "results" / "frontier" / "stratified_64" / "dataset.csv"))
    p.add_argument("--output_root", default=str(REPO_ROOT / "results" / "frontier" / "stratified_64"))
    p.add_argument("--run_id_prefix", required=True)
    p.add_argument("--thinking_budgets", type=int, nargs="+", default=[0, 1536],
                   help="Stage-2 conditions to sweep, in order. Default matches the stratified-run plan.")
    p.add_argument("--budget", type=int, default=INVESTIGATIVE_BUDGET_DEFAULT)
    p.add_argument("--truncate_chars", type=int, default=TRUNCATE_CHARS_DEFAULT)
    p.add_argument("--subprocess_timeout", type=int, default=SUBPROCESS_TIMEOUT_DEFAULT)
    p.add_argument("--episode_cost_cap_usd", type=float, default=EPISODE_COST_CAP_DEFAULT)
    p.add_argument("--max_cost_usd_stage1", type=float, default=0.5,
                   help="Session-level cap for the shared Stage-1 pass.")
    p.add_argument("--max_cost_usd_per_condition", type=float, default=3.0,
                   help="Session-level cap for EACH Stage-2 sweep independently (not shared across conditions).")
    p.add_argument("--rpm", type=float, default=8.0)
    p.add_argument("--limit", type=int, default=0, help="Cap total rows processed per phase (for testing).")
    p.add_argument("--max_retries", type=int, default=4)
    return p.parse_args()


def _condition_dirname(thinking_budget: int) -> str:
    return "nothink" if thinking_budget == 0 else "think"


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
    output_root = Path(args.output_root)

    df = load_dataset(args.dataset)
    print(f"[stratified-sweep] Dataset: {len(df)} rows from {args.dataset}", flush=True)

    # ── Phase 1: shared Stage-1 cache pass ───────────────────────────────────
    stage1_dir = output_root / "stage1_cache"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = stage1_dir / f"{slug}__stage1_cache.jsonl"

    done1 = load_checkpoint(stage1_path)
    todo1 = df[~df["title"].isin(done1)].reset_index(drop=True)
    print(f"[stratified-sweep] Stage 1: {len(todo1)} rows to process (of {len(df)} total)", flush=True)

    limiter1 = RateLimiter(args.rpm)
    stage1_session_cost = 0.0
    new1 = 0
    for _, row in todo1.iterrows():
        if args.limit and new1 >= args.limit:
            print(f"[stratified-sweep] Stage 1 --limit {args.limit} reached.", flush=True)
            break
        est = estimate_cost_usd(PROMPT_S1_AGENTIC.format(code=str(row["code"])))
        if stage1_session_cost + est > args.max_cost_usd_stage1:
            print(f"[stratified-sweep] Stage 1 cost limit ${args.max_cost_usd_stage1:.2f} reached. Stopping Stage 1.", flush=True)
            break
        print(f"  [stage1 {new1+1}] {row['title']}", end=" ", flush=True)
        limiter1.wait()
        result1 = run_stage1(client, args.model, row.to_dict(), max_retries=args.max_retries)
        stage1_session_cost += result1["s1_cost_usd"]
        append_result(stage1_path, result1)
        new1 += 1
        print(f"valid={result1['s1_parsed_valid']} match={result1['s1_valid_match']} ${result1['s1_cost_usd']:.5f}", flush=True)

    print(f"[stratified-sweep] Stage 1 done: {new1} new rows -> {stage1_path}", flush=True)
    print(f"[stratified-sweep] Stage 1 session cost: ${stage1_session_cost:.4f}", flush=True)

    # Load the full Stage-1 cache (not just this session's new rows) keyed by title.
    stage1_cache: dict[str, dict] = {}
    if stage1_path.exists():
        import json
        with open(stage1_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    stage1_cache[r["title"]] = r

    missing = [t for t in df["title"] if t not in stage1_cache]
    if missing:
        print(f"[stratified-sweep] WARNING: {len(missing)} row(s) have no Stage-1 cache entry "
              f"(likely --limit or cost cap stopped Stage 1 early) -- these will be skipped in "
              f"every Stage-2 condition until Stage 1 is completed for them: {missing[:5]}{'...' if len(missing) > 5 else ''}",
              flush=True)

    # ── Phase 2: Stage-2 sweeps, one per thinking condition ──────────────────
    for thinking_budget in args.thinking_budgets:
        cond_dir = output_root / _condition_dirname(thinking_budget)
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_path = cond_dir / f"{slug}__belief_revision_agentic.jsonl"

        done2 = load_checkpoint(out_path)
        rows_available = [t for t in df["title"] if t in stage1_cache and t not in done2]
        print(f"[stratified-sweep] Stage 2 (thinking_budget={thinking_budget}): "
              f"{len(rows_available)} rows to process", flush=True)

        limiter2 = RateLimiter(args.rpm)
        session_cost = 0.0
        new2 = 0
        think_margin = thinking_budget * _PRICE_THINK if thinking_budget else 0.0

        for title in rows_available:
            if args.limit and new2 >= args.limit:
                print(f"[stratified-sweep] Stage 2 (thinking_budget={thinking_budget}) --limit {args.limit} reached.", flush=True)
                break
            row = df[df["title"] == title].iloc[0].to_dict()
            est = (estimate_cost_usd(PROMPT_S1_AGENTIC.format(code=str(row["code"]))) + think_margin) * (args.budget + 2)
            if session_cost + est > args.max_cost_usd_per_condition:
                print(f"[stratified-sweep] Stage 2 (thinking_budget={thinking_budget}) cost limit "
                      f"${args.max_cost_usd_per_condition:.2f} reached. Stopping this condition.", flush=True)
                break

            print(f"  [{_condition_dirname(thinking_budget)} {new2+1}] {title}", end=" ", flush=True)
            limiter2.wait()
            run_id = f"{args.run_id_prefix}_{_condition_dirname(thinking_budget)}"
            result = run_stage2_and_score(
                client, args.model, row, stage1_cache[title], run_id=run_id,
                budget=args.budget, truncate_chars=args.truncate_chars,
                subprocess_timeout=args.subprocess_timeout,
                episode_cost_cap_usd=args.episode_cost_cap_usd,
                thinking_budget=thinking_budget, max_retries=args.max_retries,
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
                    f"${result['total_cost_usd']:.5f}",
                    flush=True,
                )

        print(f"[stratified-sweep] Stage 2 (thinking_budget={thinking_budget}) done: "
              f"{new2} new rows -> {out_path}", flush=True)
        print(f"[stratified-sweep] Stage 2 (thinking_budget={thinking_budget}) session cost: ${session_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
