"""
run_belief_revision_agentic.py — Two-stage frontier model eval (Gemini API),
agentic Stage 2.

Stage 1: identical to run_belief_revision.py (imported, not duplicated).
Stage 2: manual (non-automatic) function calling. Model chooses edit_source /
         run_diagnostic / submit_final_answer turns under an investigative
         action budget; submit_final_answer never counts against it.

Output: results/frontier/<model_slug>__belief_revision_agentic.jsonl
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

from frontier.agentic_prompts import ALL_TOOL_DECLS, build_stage2_prompt  # noqa: E402
from frontier.agentic_tools import (  # noqa: E402
    EpisodeCostGuard,
    apply_unified_diff,
    next_version_filename,
    tools_available,
    truncate,
)
from frontier.agentic_sandbox import (  # noqa: E402
    run_python_file,
    setup_episode,
    snapshot_turn,
)

INVESTIGATIVE_BUDGET_DEFAULT = 6
TRUNCATE_CHARS_DEFAULT = 4000
SUBPROCESS_TIMEOUT_DEFAULT = 120
EPISODE_COST_CAP_DEFAULT = 0.50  # USD -- see notes_agentic_token_inefficiency.txt

# Max consecutive times the model may call a tool that wasn't actually declared
# to it that turn (e.g. calling edit_source after the budget/cost-guard already
# narrowed the tool list to submit_final_answer only) before the harness gives
# up and forces an empty synthetic submission. Confirmed live during the Task 8
# pilot: a real model can and does keep emitting calls for a tool name that
# isn't in that turn's declared schema -- the API does not enforce this itself,
# so the harness must reject and bound it explicitly.
MAX_PROTOCOL_VIOLATIONS = 2

# Gemini 2.5 Flash pricing (USD/token) -- mirrors run_belief_revision.py's
# constants; duplicated rather than imported since these are module-level
# literals, not behavior, and keeping this file's cost math self-contained
# avoids a cross-import just for three numbers.
_PRICE_IN = 0.15 / 1_000_000
_PRICE_OUT = 0.60 / 1_000_000
_PRICE_THINK = 3.50 / 1_000_000


def _token_cost(usage) -> float:
    if usage is None:
        return 0.0
    inp = getattr(usage, "prompt_token_count", 0) or 0
    out = getattr(usage, "candidates_token_count", 0) or 0
    think = getattr(usage, "thoughts_token_count", 0) or 0
    return inp * _PRICE_IN + out * _PRICE_OUT + think * _PRICE_THINK


def call_gemini_agentic(client, model: str, contents: list, config, max_retries: int = 4):
    """Same retry policy as run_belief_revision.call_gemini, but accepts a
    caller-built GenerateContentConfig (needed for per-turn `tools=`). Kept
    separate rather than modifying call_gemini itself -- that file is not to
    be touched by this plan."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            err = str(exc)
            retryable = any(code in err for code in ("429", "503", "500")) \
                        or "quota" in err.lower() or "rate" in err.lower()
            if retryable and attempt < max_retries:
                wait = (2 ** attempt) * 5
                print(f"    [retry {attempt+1}] {err[:80]} — wait {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise


def _build_tools(available_names: list[str]):
    from google.genai import types
    decls = [types.FunctionDeclaration(**ALL_TOOL_DECLS[name]) for name in available_names]
    return [types.Tool(function_declarations=decls)]


def _do_edit_source(work: Path, diff_text: str, timeout: int) -> tuple[str, str | None]:
    existing = sorted(p.name for p in work.glob("solver_v*.py"))
    latest_name = existing[-1]
    latest_code = (work / latest_name).read_text()

    new_code, err = apply_unified_diff(latest_code, diff_text)
    if err is not None:
        return (
            f"Model chose edit_source. diff failed to apply to {latest_name}: {err}",
            None,
        )

    new_name = next_version_filename(existing)
    (work / new_name).write_text(new_code)
    stdout, stderr, timed_out = run_python_file(new_name, work, timeout)
    suffix = " [TIMEOUT]" if timed_out else ""
    text = (
        f"Model chose edit_source. diff applied to {latest_name} and result "
        f"saved to {new_name}. Execution returned stdout={stdout!r} stderr={stderr!r}{suffix}"
    )
    return text, new_name


def _do_run_diagnostic(work: Path, script_text: str, timeout: int) -> tuple[str, str | None]:
    existing = sorted(
        int(p.stem.replace("diagnostic_", "")) for p in work.glob("diagnostic_*.py")
    )
    idx = (max(existing) + 1) if existing else 0
    new_name = f"diagnostic_{idx}.py"
    (work / new_name).write_text(script_text)
    stdout, stderr, timed_out = run_python_file(new_name, work, timeout)
    suffix = " [TIMEOUT]" if timed_out else ""
    text = (
        f"Model chose run_diagnostic. Diagnostic script saved to {new_name}. "
        f"Execution returned stdout={stdout!r} stderr={stderr!r}{suffix}"
    )
    return text, new_name


def run_agentic_stage2(
    client,
    model: str,
    title: str,
    run_id: str,
    code: str,
    prompt_s1: str,
    s1_text: str,
    *,
    budget: int = INVESTIGATIVE_BUDGET_DEFAULT,
    truncate_chars: int = TRUNCATE_CHARS_DEFAULT,
    subprocess_timeout: int = SUBPROCESS_TIMEOUT_DEFAULT,
    episode_cost_cap_usd: float = EPISODE_COST_CAP_DEFAULT,
    thinking_budget: int = 0,
    max_retries: int = 4,
) -> dict:
    """
    Run the agentic Stage 2 loop for one row. Returns a dict of per-episode
    result fields (action trace, submit answer, budget bookkeeping, cost).

    `client` is duck-typed to have `.models.generate_content(model, contents,
    config)` -- tests pass a scripted fake client (tests/test_agentic_loop.py),
    real runs pass a `google.genai.Client()`.

    `thinking_budget` defaults to 0 to match the existing static experiment's
    convention (Rohan's decision for the pilot); exposed as a parameter, not a
    hardcoded literal, since it's expected to change once Stage 2's tool-choice
    quality is evaluated with thinking enabled.
    """
    from google.genai import types

    work = setup_episode(code, title, run_id)
    snapshot_turn(title, run_id, 0)

    contents = [
        types.Content(role="user", parts=[types.Part(text=prompt_s1)]),
        types.Content(role="model", parts=[types.Part(text=s1_text)]),
        types.Content(role="user", parts=[types.Part(text=build_stage2_prompt(budget))]),
    ]

    cost_guard = EpisodeCostGuard(episode_cost_cap_usd)
    actions_used = 0
    action_trace: list[dict] = []
    tools_used: set[str] = set()
    episode_cost = 0.0
    submit_args: dict | None = None
    cost_guard_tripped = False
    protocol_violations = 0
    turn = 0

    def _empty_submission(reason: str, raw_text: str | None = None) -> dict:
        return {
            "pde": "", "method": "", "behavior": "", "valid": "",
            "pde_exp": "", "method_exp": "", "behavior_exp": "", "valid_exp": "",
            "_forced_reason": reason,
            "_raw_text": raw_text or "",
        }

    while submit_args is None:
        turn += 1
        available = tools_available(actions_used, budget, cost_guard_tripped)
        config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=_build_tools(available),
            # Force the model to call one of exactly `available` -- verified live
            # (see explore_tool_config.py exploration) that mode="ANY" with
            # allowed_function_names constrains generation itself, unlike the
            # default AUTO mode, which only suggests the tool list and does not
            # stop the model from naming an undeclared tool or replying with
            # plain text. The undeclared-tool rejection check further below is
            # left as-is for now: if ANY truly enforces this, that check should
            # never fire again in practice; if it ever does, that means the ANY
            # constraint itself failed, not that the model misbehaved -- worth
            # revisiting the rejection logic's meaning at that point, not before.
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=available
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = call_gemini_agentic(client, model, contents, config, max_retries)
        turn_cost = _token_cost(getattr(resp, "usage_metadata", None))
        episode_cost += turn_cost
        cost_guard.add(turn_cost)

        calls = getattr(resp, "function_calls", None) or []
        if not calls:
            # No tool call at all: treat as a forced, empty submission rather
            # than looping forever. Distinct from a normal forced completion
            # (budget exhausted) -- both end the episode, but this one has no
            # real answer content. The model's plain-text reply (if any) is
            # still captured in full -- never discard a model response, even
            # when it doesn't take the shape the harness expected.
            raw_text = getattr(resp, "text", None) or ""
            action_trace.append({
                "turn": turn, "tool": None, "args": {}, "result": raw_text,
                "new_filename": None, "no_function_call": True,
            })
            submit_args = _empty_submission("no_function_call", raw_text)
            break

        call = calls[0]
        name = call.name
        args = dict(call.args or {})
        contents.append(types.Content(role="model", parts=[types.Part(function_call=call)]))

        if name == "submit_final_answer":
            submit_args = args
            action_trace.append({"turn": turn, "tool": name, "args": args, "result": None, "new_filename": None})
            break

        if name not in available:
            # The model called a tool that wasn't declared to it this turn --
            # confirmed live (Task 8 pilot) that the real API does not itself
            # enforce this, so the harness must reject rather than execute it.
            # Do NOT dispatch, do NOT count it against the investigative budget
            # (it was never actually offered). Give the model a bounded number
            # of chances to correct course before forcing an empty submission,
            # so a non-compliant model can't loop indefinitely.
            protocol_violations += 1
            rejection = (
                f"Rejected: '{name}' is not an available tool this turn "
                f"(only {available} {'is' if len(available) == 1 else 'are'} available). "
                f"Call one of {available} instead."
            )
            action_trace.append({
                "turn": turn, "tool": name, "args": args,
                "result": rejection, "new_filename": None, "rejected": True,
            })
            contents.append(types.Content(
                role="tool",
                parts=[types.Part.from_function_response(name=name, response={"result": rejection})],
            ))
            if protocol_violations > MAX_PROTOCOL_VIOLATIONS:
                submit_args = _empty_submission("protocol_violations_exceeded")
                break
            continue

        tools_used.add(name)
        if name == "edit_source":
            result_text, new_filename = _do_edit_source(work, args.get("diff", ""), subprocess_timeout)
        elif name == "run_diagnostic":
            result_text, new_filename = _do_run_diagnostic(work, args.get("script", ""), subprocess_timeout)
        else:
            result_text, new_filename = f"Unknown tool: {name}", None

        actions_used += 1
        truncated = truncate(result_text, truncate_chars)
        action_trace.append({
            "turn": turn, "tool": name, "args": args,
            "result": truncated, "new_filename": new_filename,
        })

        contents.append(types.Content(
            role="tool",
            parts=[types.Part.from_function_response(name=name, response={"result": truncated})],
        ))

        snapshot_turn(title, run_id, turn)

        if cost_guard.tripped():
            cost_guard_tripped = True

    actions_remaining_at_submission = max(budget - actions_used, 0)
    return {
        "action_trace": action_trace,
        "action_count": actions_used,
        "tools_used": sorted(tools_used),
        "used_edit_source": "edit_source" in tools_used,
        "actions_remaining_at_submission": actions_remaining_at_submission,
        "cost_guard_tripped": cost_guard_tripped,
        "protocol_violations": protocol_violations,
        "episode_cost_usd": round(episode_cost, 8),
        "submit_args": submit_args,
        "episode_dir": str(work),
    }


# ── Row processing (Stage 1 reused, Stage 2 agentic) ─────────────────────────

import argparse
import hashlib
import os

from frontier.run_belief_revision import (  # noqa: E402
    PROMPT_S1,
    RateLimiter,
    append_result,
    call_gemini,
    estimate_cost_usd,
    load_checkpoint,
    token_cost,
)
from frontier.parse_frontier import parse_response, score_row  # noqa: E402
from dataset_io import DEFAULT_MOD_DATASET, load_dataset  # noqa: E402


def process_row(
    client,
    model: str,
    row: dict,
    *,
    run_id: str,
    budget: int = INVESTIGATIVE_BUDGET_DEFAULT,
    truncate_chars: int = TRUNCATE_CHARS_DEFAULT,
    subprocess_timeout: int = SUBPROCESS_TIMEOUT_DEFAULT,
    episode_cost_cap_usd: float = EPISODE_COST_CAP_DEFAULT,
    thinking_budget: int = 0,
    max_retries: int = 4,
) -> dict:
    """
    Run Stage 1 (identical to run_belief_revision.py) then the agentic Stage 2
    loop for one dataset row. Returns the full JSONL row dict.
    """
    from google.genai import types

    title = row["title"]
    code = str(row["code"])
    gt_valid = bool(row["phys_valid"])

    prompt_s1 = PROMPT_S1.format(code=code)
    contents_s1 = [types.Content(role="user", parts=[types.Part(text=prompt_s1)])]
    resp1 = call_gemini(client, model, contents_s1, max_retries)
    s1_text = resp1.text or ""
    s1_cost = token_cost(getattr(resp1, "usage_metadata", None))[3]

    p1 = parse_response(s1_text)
    scores1 = score_row(p1, row, embed_model=None)

    s2 = run_agentic_stage2(
        client, model, title, run_id, code, prompt_s1, s1_text,
        budget=budget, truncate_chars=truncate_chars,
        subprocess_timeout=subprocess_timeout,
        episode_cost_cap_usd=episode_cost_cap_usd,
        thinking_budget=thinking_budget, max_retries=max_retries,
    )

    submit = s2["submit_args"]
    # submit_final_answer's arguments arrive already structured -- parse_response
    # is bypassed entirely for Stage 2, per the design doc's Scoring section.
    scores2 = score_row(submit, row, embed_model=None)

    return {
        "model": model,
        "title": title,
        "mod_type": row["mod_type"],
        "gt_valid": gt_valid,
        "gt_pde": str(row["pde_class"]),
        "gt_method": str(row["num_method"]),
        "gt_behavior": str(row["phys_process"]),
        "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],

        # Stage 1
        "s1_response": s1_text,
        "s1_parsed_pde": p1.get("pde"),
        "s1_parsed_method": p1.get("method"),
        "s1_parsed_behavior": p1.get("behavior"),
        "s1_parsed_valid": p1.get("valid"),
        "s1_valid_match": scores1.get("valid_match"),
        "s1_pde_match": scores1.get("pde_match"),
        "s1_method_any_match": scores1.get("method_any_match"),
        "s1_behavior_any_match": scores1.get("behavior_any_match"),

        # Stage 2 (agentic)
        "s2_submit_args": submit,
        "s2_valid_match": scores2.get("valid_match"),
        "s2_pde_match": scores2.get("pde_match"),
        "s2_method_any_match": scores2.get("method_any_match"),
        "s2_behavior_any_match": scores2.get("behavior_any_match"),
        "s2_action_count": s2["action_count"],
        "s2_tools_used": s2["tools_used"],
        "s2_used_edit_source": s2["used_edit_source"],
        "actions_remaining_at_submission": s2["actions_remaining_at_submission"],
        "cost_guard_tripped": s2["cost_guard_tripped"],
        "protocol_violations": s2["protocol_violations"],
        "action_trace": s2["action_trace"],
        "episode_dir": s2["episode_dir"],

        # Bookkeeping
        "total_cost_usd": round(s1_cost + s2["episode_cost_usd"], 8),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic frontier belief-revision eval")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--dataset", default=str(REPO_ROOT / DEFAULT_MOD_DATASET))
    p.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "frontier"))
    p.add_argument("--run_id", required=True, help="Required -- see spec's Sandbox section")
    p.add_argument("--budget", type=int, default=INVESTIGATIVE_BUDGET_DEFAULT)
    p.add_argument("--truncate_chars", type=int, default=TRUNCATE_CHARS_DEFAULT)
    p.add_argument("--subprocess_timeout", type=int, default=SUBPROCESS_TIMEOUT_DEFAULT)
    p.add_argument("--episode_cost_cap_usd", type=float, default=EPISODE_COST_CAP_DEFAULT)
    p.add_argument("--thinking_budget", type=int, default=0)
    p.add_argument("--max_cost_usd", type=float, default=1.0,
                   help="Session-level cap. Conservative default (vs. run_belief_revision.py's "
                        "5.0) since per-row agentic cost is not yet characterized and Rohan has "
                        "a $5/month cap on the Google side -- this is an ESTIMATE-based guard, "
                        "not an enforced billing cap; set a real budget alert on the Google "
                        "Cloud/AI Studio side too.")
    p.add_argument("--rpm", type=float, default=8.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Use the workspace's centralized key manager, not a bare env var -- per
    # CLAUDE.md's "never hardcode keys, always use KeyHandler" rule. Falls through
    # to checking os.environ afterward in case GOOGLE_API_KEY was exported directly
    # instead (e.g. in a cluster job that can't reach the local key_handler package).
    try:
        workspace_root = REPO_ROOT.parent.parent  # mlproj -> private_projects -> raca-torch
        sys.path.insert(0, str(workspace_root / "packages" / "key_handler" / "key_handler"))
        from key_handler import KeyHandler
        KeyHandler.set_env_key()
    except ImportError:
        pass

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        sys.exit("[frontier-agentic] ERROR: GOOGLE_API_KEY not set (checked KeyHandler and os.environ).")

    from google import genai
    client = genai.Client(api_key=api_key)

    os.makedirs(args.output_dir, exist_ok=True)
    slug = args.model.replace("/", "__").replace("-", "").replace(".", "")
    out_path = Path(args.output_dir) / f"{slug}__belief_revision_agentic.jsonl"

    df = load_dataset(args.dataset)

    done = load_checkpoint(out_path)
    todo = df[~df["title"].isin(done)].reset_index(drop=True)
    print(f"[frontier-agentic] To process: {len(todo)}", flush=True)

    limiter = RateLimiter(args.rpm)
    session_cost = 0.0
    new_rows = 0

    for _, row in todo.iterrows():
        if args.limit and new_rows >= args.limit:
            print(f"[frontier-agentic] --limit {args.limit} reached.", flush=True)
            break

        est = estimate_cost_usd(PROMPT_S1.format(code=str(row["code"]))) * (args.budget + 2)
        if session_cost + est > args.max_cost_usd:
            print(f"[frontier-agentic] Cost limit ${args.max_cost_usd:.2f} reached. Stopping.", flush=True)
            break

        print(f"  [{new_rows+1}] {row['title']}", end=" ", flush=True)
        limiter.wait()
        result = process_row(
            client, args.model, row.to_dict(), run_id=args.run_id,
            budget=args.budget, truncate_chars=args.truncate_chars,
            subprocess_timeout=args.subprocess_timeout,
            episode_cost_cap_usd=args.episode_cost_cap_usd,
            thinking_budget=args.thinking_budget, max_retries=args.max_retries,
        )
        session_cost += result["total_cost_usd"]
        append_result(out_path, result)
        new_rows += 1
        print(
            f"actions={result['s2_action_count']} "
            f"remaining_at_submit={result['actions_remaining_at_submission']} "
            f"valid_match(s1->s2)={result['s1_valid_match']}->{result['s2_valid_match']} "
            f"${result['total_cost_usd']:.5f}",
            flush=True,
        )

    print(f"\n[frontier-agentic] Done. {new_rows} new rows -> {out_path}", flush=True)
    print(f"[frontier-agentic] Session cost: ${session_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
