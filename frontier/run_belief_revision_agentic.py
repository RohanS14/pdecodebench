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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frontier.agentic_prompts import (  # noqa: E402
    ALL_TOOL_DECLS,
    EMPTY_RESPONSE_FEEDBACK,
    PROMPT_S1_AGENTIC,
    TEXT_ONLY_FEEDBACK,
    build_stage2_prompt,
    build_submit_confirmation_reminder,
    build_validated_reminder_final,
    build_validated_reminder_investigative,
    parse_s1_explanations,
)
from frontier.agentic_tools import (  # noqa: E402
    EpisodeCostGuard,
    next_version_filename,
    tools_available,
    truncate,
)
from frontier.agentic_sandbox import (  # noqa: E402
    MAX_EPISODE_DIR_BYTES,
    MAX_FILE_SIZE_BYTES,
    run_python_file,
    setup_episode,
    snapshot_turn,
)
from frontier.episode_log import write_episode_log  # noqa: E402

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
_PRICE_OUT = 1.25 / 1_000_000
_PRICE_THINK = 3.50 / 1_000_000


def _token_cost(usage) -> tuple[int, int, int, float]:
    """Returns (input_tokens, output_tokens, think_tokens, cost_usd) -- mirrors
    run_belief_revision.py's token_cost() so raw counts can be persisted per
    episode, not just the derived cost (which depends on pricing constants
    that can go stale/wrong -- keeping the raw counts means cost can always
    be recomputed correctly later without re-running anything)."""
    if usage is None:
        return 0, 0, 0, 0.0
    inp = getattr(usage, "prompt_token_count", 0) or 0
    out = getattr(usage, "candidates_token_count", 0) or 0
    think = getattr(usage, "thoughts_token_count", 0) or 0
    cost = inp * _PRICE_IN + out * _PRICE_OUT + think * _PRICE_THINK
    return inp, out, think, cost


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


def _do_edit_source(
    work: Path, full_source: str, timeout: int, title: str, run_id: str,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_episode_dir_bytes: int = MAX_EPISODE_DIR_BYTES,
) -> tuple[str, str | None, str | None]:
    """
    Write full_source as a new versioned file and rerun it. Empty/whitespace-only
    full_source is a valid no-op: reruns the current latest version completely
    unchanged (a new version file is still created, matching the "never
    overwrite in place" convention, even though its content is identical to the
    previous version).

    Returns (text, new_filename, abort_reason). abort_reason is None unless a
    disk-safety guard fired in run_python_file() (see there) -- callers must
    force-end the episode rather than continue when it's set.
    """
    existing = sorted(p.name for p in work.glob("solver_v*.py"))
    latest_name = existing[-1]

    if full_source.strip():
        new_code = full_source
        rerun_note = "full rewrite"
    else:
        new_code = (work / latest_name).read_text()
        rerun_note = f"rerun of {latest_name} unchanged (empty full_source)"

    new_name = next_version_filename(existing)
    (work / new_name).write_text(new_code)
    stdout, stderr, timed_out, abort_reason = run_python_file(
        new_name, work, timeout, title, run_id,
        max_file_size_bytes=max_file_size_bytes, max_episode_dir_bytes=max_episode_dir_bytes,
    )
    suffix = " [TIMEOUT]" if timed_out else ""
    text = (
        f"Model chose edit_source ({rerun_note}). New version saved to "
        f"{new_name}. Execution returned stdout={stdout!r} stderr={stderr!r}{suffix}"
    )
    return text, new_name, abort_reason


def _do_run_diagnostic(
    work: Path, script_text: str, timeout: int, title: str, run_id: str,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_episode_dir_bytes: int = MAX_EPISODE_DIR_BYTES,
) -> tuple[str, str | None, str | None]:
    """Returns (text, new_filename, abort_reason) -- see _do_edit_source's
    abort_reason docs, identical contract."""
    existing = sorted(
        int(p.stem.replace("diagnostic_", "")) for p in work.glob("diagnostic_*.py")
    )
    idx = (max(existing) + 1) if existing else 0
    new_name = f"diagnostic_{idx}.py"
    (work / new_name).write_text(script_text)
    stdout, stderr, timed_out, abort_reason = run_python_file(
        new_name, work, timeout, title, run_id,
        max_file_size_bytes=max_file_size_bytes, max_episode_dir_bytes=max_episode_dir_bytes,
    )
    suffix = " [TIMEOUT]" if timed_out else ""
    text = (
        f"Model chose run_diagnostic. Diagnostic script saved to {new_name}. "
        f"Execution returned stdout={stdout!r} stderr={stderr!r}{suffix}"
    )
    return text, new_name, abort_reason


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
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_episode_dir_bytes: int = MAX_EPISODE_DIR_BYTES,
) -> dict:
    """
    Run the agentic Stage 2 loop for one row. Returns a dict of per-episode
    result fields (action trace, submit answer, budget bookkeeping, cost).

    max_file_size_bytes/max_episode_dir_bytes are exposed (rather than only
    living as agentic_sandbox module constants) purely so tests can override
    them with tiny values and exercise the real disk-safety guards without
    allocating real GB-scale files.

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
    episode_input_tokens = 0
    episode_output_tokens = 0
    episode_think_tokens = 0
    submit_args: dict | None = None
    cost_guard_tripped = False
    protocol_violations = 0
    consecutive_empty = 0
    submit_confirmed = False
    mode_override: str | None = None
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

        # Mode selection: VALIDATED by default everywhere -- including the
        # terminal (budget-exhausted) phase, so the model gets a chance to
        # synthesize/think before committing to submit_final_answer, not just
        # be forced straight into a structured call. ANY is used only as the
        # existing one-shot escalation mechanism (text-only -> immediate
        # escalation; two consecutive empty turns -> escalation) -- the
        # override is consumed here, not held indefinitely, so the turn after
        # an escalation (in either phase) goes back to VALIDATED. The
        # undeclared-tool rejection check further below is left as-is: it
        # still applies whenever a call is present, under either mode.
        mode = "ANY" if mode_override == "ANY" else "VALIDATED"
        mode_override = None

        # The final-turn reminder (which re-shows the original snippet and the
        # model's own Stage-1 answer) fires whenever the terminal phase is
        # reached, independent of mode -- gating this on mode == "VALIDATED"
        # would make it unreachable on an escalated ANY terminal turn. The
        # investigative reminder, by contrast, is only shown on VALIDATED
        # turns -- ANY-mode escalation turns already got their own feedback
        # message (TEXT_ONLY_FEEDBACK/EMPTY_RESPONSE_FEEDBACK) appended in the
        # turn that triggered the escalation. Per Rohan's explicit scope
        # limit, this reminder is NOT shown on a voluntary early submission
        # (budget not yet exhausted) -- only the forced-completion path.
        if available == ["submit_final_answer"]:
            reminder = build_validated_reminder_final(code, s1_text)
            contents.append(types.Content(role="user", parts=[types.Part(text=reminder)]))
        elif mode == "VALIDATED":
            reminder = build_validated_reminder_investigative(actions_used, budget)
            contents.append(types.Content(role="user", parts=[types.Part(text=reminder)]))

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=50_000,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=(thinking_budget != 0),
            ),
            tools=_build_tools(available),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=mode, allowed_function_names=available
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = call_gemini_agentic(client, model, contents, config, max_retries)
        turn_inp, turn_out, turn_think, turn_cost = _token_cost(getattr(resp, "usage_metadata", None))
        episode_cost += turn_cost
        episode_input_tokens += turn_inp
        episode_output_tokens += turn_out
        episode_think_tokens += turn_think
        cost_guard.add(turn_cost)

        calls = getattr(resp, "function_calls", None) or []
        raw_text = getattr(resp, "text", None) or ""
        has_call = bool(calls)
        has_text = bool(raw_text.strip())

        # Thought-summary extraction (only present when include_thoughts=True).
        # Purely additive: resp.text/resp.function_calls (raw_text/calls above)
        # already correctly exclude thought parts -- confirmed live via a
        # standalone smoke test before this code was written -- so none of the
        # has_call/has_text branch logic below needs to change.
        candidates = getattr(resp, "candidates", None) or []
        resp_content_parts = []
        thought_summary_text = ""
        possible_thought_leak = False
        if candidates:
            resp_content_parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
            thought_parts = [p.text for p in resp_content_parts if getattr(p, "thought", False) and getattr(p, "text", None)]
            thought_summary_text = "\n".join(thought_parts)
        if thinking_budget != 0 and raw_text.lstrip().startswith("THOUGHT:"):
            possible_thought_leak = True
            print(f"WARNING: possible thought leak into regular text at turn {turn}")

        if mode == "ANY":
            if not has_call:
                # ANY is supposed to guarantee a call; it didn't -- confirmed
                # live this can still happen (a real invalid-row pilot hit this
                # exact case). Last resort: force an empty submission rather
                # than loop forever. The model's plain-text reply (if any) is
                # still captured in full -- never discard a model response,
                # even when it doesn't take the shape the harness expected.
                action_trace.append({
                    "turn": turn, "tool": None, "args": {}, "result": raw_text,
                    "new_filename": None, "no_function_call": True,
                    "thought_summary": thought_summary_text,
                    **({"possible_thought_leak": True} if possible_thought_leak else {}),
                })
                submit_args = _empty_submission("no_function_call", raw_text)
                break
            # else: has_call -- fall through to the shared dispatch logic below.
        else:
            # mode == "VALIDATED"
            if not has_call and not has_text:
                # Empty: could be a rare glitch rather than a deliberate choice,
                # so give one retry in VALIDATED before escalating. Still costs
                # a turn -- the budget only shrinks either way, which is what
                # actually discourages stalling.
                actions_used += 1
                action_trace.append({
                    "turn": turn, "tool": None, "args": {}, "result": "",
                    "new_filename": None, "empty_response": True,
                    "thought_summary": thought_summary_text,
                    **({"possible_thought_leak": True} if possible_thought_leak else {}),
                })
                contents.append(types.Content(role="user", parts=[types.Part(text=EMPTY_RESPONSE_FEEDBACK)]))
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    mode_override = "ANY"
                if cost_guard.tripped():
                    cost_guard_tripped = True
                continue
            if not has_call and has_text:
                # Text-only: the model consciously chose not to act -- escalate
                # immediately, no retry.
                actions_used += 1
                action_trace.append({
                    "turn": turn, "tool": None, "args": {}, "result": raw_text,
                    "new_filename": None, "text_only": True,
                    "thought_summary": thought_summary_text,
                    **({"possible_thought_leak": True} if possible_thought_leak else {}),
                })
                contents.append(types.Content(role="user", parts=[types.Part(text=TEXT_ONLY_FEEDBACK)]))
                consecutive_empty = 0
                mode_override = "ANY"
                if cost_guard.tripped():
                    cost_guard_tripped = True
                continue
            # else: has_call (call-only or text+call) -- fall through.

        consecutive_empty = 0
        reasoning_text = raw_text if has_text else ""

        call = calls[0]
        name = call.name
        args = dict(call.args or {})
        # Preserve the full response content (text part + its thought_signature,
        # alongside the function_call part) rather than hand-building a minimal
        # function-call-only Content -- confirmed live that Gemini attaches
        # thought_signature to the accompanying text part, not the function_call
        # part, so dropping that part silently drops reasoning continuity when
        # thinking is on. Excludes the thought-summary part itself (not meant to
        # be replayed as future context). Falls back to the exact original
        # single-part behavior when resp has no .candidates (all fake test
        # doubles, and any real response without them).
        non_thought_parts = [p for p in resp_content_parts if not getattr(p, "thought", False)]
        model_parts = non_thought_parts if non_thought_parts else [types.Part(function_call=call)]
        contents.append(types.Content(role="model", parts=model_parts))

        if name == "submit_final_answer":
            if len(available) > 1 and not submit_confirmed:
                # Voluntary submission (budget not yet exhausted) -- unlike
                # the forced/terminal path (which already gets
                # build_validated_reminder_final BEFORE its one
                # deterministic attempt, since the harness knows in advance
                # that's the only tool available), a voluntary submission
                # can happen at any unpredictable turn, so there's no way to
                # pre-emptively ground it. Intercept once, reactively, and
                # require a second call to finalize. Confirmed live: without
                # this, the model can judge the validity of its own edited
                # code and report that for the original snippet instead.
                submit_confirmed = True
                reminder = build_submit_confirmation_reminder(code, s1_text)
                action_trace.append({
                    "turn": turn, "tool": name, "args": args, "result": reminder,
                    "new_filename": None, "reasoning_text": reasoning_text,
                    "provisional_submit": True,
                    "thought_summary": thought_summary_text,
                    **({"possible_thought_leak": True} if possible_thought_leak else {}),
                })
                contents.append(types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=name, response={"result": reminder})],
                ))
                continue
            submit_args = args
            action_trace.append({
                "turn": turn, "tool": name, "args": args, "result": None,
                "new_filename": None, "reasoning_text": reasoning_text,
                "thought_summary": thought_summary_text,
                **({"possible_thought_leak": True} if possible_thought_leak else {}),
            })
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
                "reasoning_text": reasoning_text,
                "thought_summary": thought_summary_text,
                **({"possible_thought_leak": True} if possible_thought_leak else {}),
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
            result_text, new_filename, abort_reason = _do_edit_source(
                work, args.get("full_source", ""), subprocess_timeout, title, run_id,
                max_file_size_bytes=max_file_size_bytes, max_episode_dir_bytes=max_episode_dir_bytes,
            )
        elif name == "run_diagnostic":
            result_text, new_filename, abort_reason = _do_run_diagnostic(
                work, args.get("script", ""), subprocess_timeout, title, run_id,
                max_file_size_bytes=max_file_size_bytes, max_episode_dir_bytes=max_episode_dir_bytes,
            )
        else:
            result_text, new_filename, abort_reason = f"Unknown tool: {name}", None, None

        actions_used += 1
        truncated = truncate(result_text, truncate_chars)
        action_trace.append({
            "turn": turn, "tool": name, "args": args,
            "result": truncated, "new_filename": new_filename,
            "reasoning_text": reasoning_text,
            "thought_summary": thought_summary_text,
            **({"possible_thought_leak": True} if possible_thought_leak else {}),
            **({"abort_reason": abort_reason} if abort_reason else {}),
        })

        if abort_reason:
            # Disk-safety guard fired (see agentic_sandbox.run_python_file):
            # either the oversized-write culprit couldn't be confidently
            # identified, or the episode dir's cumulative size exceeded the
            # cap. Force-end the episode immediately -- do not append this
            # turn's response to contents (there's nothing more to continue)
            # and do not snapshot_turn() (the episode is over, not just this
            # turn). Never silently retried on sweep resume; see
            # run_stratified_sweep.py's checkpoint handling.
            break

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
        "episode_input_tokens": episode_input_tokens,
        "episode_output_tokens": episode_output_tokens,
        "episode_think_tokens": episode_think_tokens,
        "submit_args": submit_args,
        "episode_dir": str(work),
        "aborted": submit_args is None and bool(action_trace) and bool(action_trace[-1].get("abort_reason")),
        "abort_reason": action_trace[-1].get("abort_reason") if action_trace and action_trace[-1].get("abort_reason") else None,
    }


# ── Row processing (Stage 1 reused, Stage 2 agentic) ─────────────────────────

import argparse
import hashlib
import os

from frontier.run_belief_revision import (  # noqa: E402
    RateLimiter,
    append_result,
    call_gemini,
    estimate_cost_usd,
    get_transition,
    load_checkpoint,
    token_cost,
)
from frontier.parse_frontier import classify_hedge, parse_response, score_row  # noqa: E402
from shared.dataset_io import DEFAULT_MOD_DATASET, load_dataset  # noqa: E402


def run_stage1(client, model: str, row: dict, max_retries: int = 4) -> dict:
    """
    Run Stage 1 only (agentic-only prompt: PROMPT_S1_AGENTIC, a superset of
    run_belief_revision.py's PROMPT_S1 that adds explanation fields -- the
    shared static-experiment prompt/parser stay untouched).

    Stage 1 is 100% independent of thinking_budget (call_gemini() hardcodes
    thinking_budget=0 and never receives the CLI flag) and fully
    deterministic (temperature=0) -- cache this once per row and reuse it
    across every thinking-condition sweep rather than re-running it, which
    would just be paying twice for identical work.
    """
    from google.genai import types

    title = row["title"]
    code = str(row["code"])
    gt_valid = bool(row["phys_valid"])

    prompt_s1 = PROMPT_S1_AGENTIC.format(code=code)
    contents_s1 = [types.Content(role="user", parts=[types.Part(text=prompt_s1)])]
    resp1 = call_gemini(client, model, contents_s1, max_retries)
    s1_text = resp1.text or ""
    s1_inp, s1_out, s1_think, s1_cost = token_cost(getattr(resp1, "usage_metadata", None))

    p1 = parse_response(s1_text)
    scores1 = score_row(p1, row, embed_model=None)
    s1_explanations = parse_s1_explanations(s1_text)

    return {
        "model": model,
        "title": title,
        "mod_type": row["mod_type"],
        "gt_valid": gt_valid,
        "gt_pde": str(row["pde_class"]),
        "gt_method": str(row["num_method"]),
        "gt_behavior": str(row["phys_process"]),
        "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],

        # Internal-only, consumed by run_stage2_and_score(); stripped before
        # a row is ever written out (see there).
        "code": code,
        "prompt_s1": prompt_s1,

        "s1_response": s1_text,
        "s1_parsed_pde": p1.get("pde"),
        "s1_parsed_method": p1.get("method"),
        "s1_parsed_behavior": p1.get("behavior"),
        "s1_parsed_valid": p1.get("valid"),
        "s1_valid_match": scores1.get("valid_match"),
        "s1_pde_match": scores1.get("pde_match"),
        "s1_method_any_match": scores1.get("method_any_match"),
        "s1_behavior_any_match": scores1.get("behavior_any_match"),
        "s1_hedge_class": classify_hedge(p1.get("valid")),
        "s1_pde_exp": s1_explanations.get("pde_exp"),
        "s1_method_exp": s1_explanations.get("method_exp"),
        "s1_behavior_exp": s1_explanations.get("behavior_exp"),
        "s1_valid_exp": s1_explanations.get("valid_exp"),

        # Bookkeeping -- raw token counts persisted (not just derived cost)
        # so cost can always be recomputed correctly later even if pricing
        # constants change or were wrong at the time this row was written.
        "s1_input_tokens": s1_inp,
        "s1_output_tokens": s1_out,
        "s1_think_tokens": s1_think,
        "s1_cost_usd": round(s1_cost, 8),
    }


def run_stage2_and_score(
    client,
    model: str,
    row: dict,
    stage1_result: dict,
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
    Run the agentic Stage 2 loop using an already-computed stage1_result
    (fresh or cached from run_stage1()) and assemble the full output row.
    This is the piece that must run once PER thinking condition -- Stage 1
    itself is reused unchanged across conditions.
    """
    title = stage1_result["title"]
    code = stage1_result["code"]
    prompt_s1 = stage1_result["prompt_s1"]
    s1_text = stage1_result["s1_response"]

    s2 = run_agentic_stage2(
        client, model, title, run_id, code, prompt_s1, s1_text,
        budget=budget, truncate_chars=truncate_chars,
        subprocess_timeout=subprocess_timeout,
        episode_cost_cap_usd=episode_cost_cap_usd,
        thinking_budget=thinking_budget, max_retries=max_retries,
    )

    out = dict(stage1_result)
    out.pop("code", None)
    out.pop("prompt_s1", None)
    s1_cost = out.pop("s1_cost_usd", 0.0)

    if s2["aborted"]:
        # Disk-safety guard force-ended the episode (see
        # agentic_sandbox.run_python_file) -- there is no submit_args to
        # score. Normal s2_* scoring fields are intentionally omitted so
        # downstream analysis/plots can't mistake this for a real answer;
        # callers (run_stratified_sweep.py) must still checkpoint this row
        # (so it isn't silently auto-retried) and flag it for human review.
        out.update({
            "thinking_budget": thinking_budget,
            "aborted": True,
            "abort_reason": s2["abort_reason"],
            "s2_action_count": s2["action_count"],
            "action_trace": s2["action_trace"],
            "episode_dir": s2["episode_dir"],
            "s2_input_tokens": s2["episode_input_tokens"],
            "s2_output_tokens": s2["episode_output_tokens"],
            "s2_think_tokens": s2["episode_think_tokens"],
            "total_cost_usd": round(s1_cost + s2["episode_cost_usd"], 8),
        })
        return out

    submit = s2["submit_args"]
    # submit_final_answer's arguments arrive already structured -- parse_response
    # is bypassed entirely for Stage 2, per the design doc's Scoring section.
    scores2 = score_row(submit, row, embed_model=None)

    out.update({
        "thinking_budget": thinking_budget,
        "aborted": False,
        "abort_reason": None,
        "transition": get_transition(out.get("s1_valid_match"), scores2.get("valid_match")),

        "s2_submit_args": submit,
        "s2_valid_match": scores2.get("valid_match"),
        "s2_pde_match": scores2.get("pde_match"),
        "s2_method_any_match": scores2.get("method_any_match"),
        "s2_behavior_any_match": scores2.get("behavior_any_match"),
        "s2_hedge_class": classify_hedge(submit.get("valid")),
        "s2_action_count": s2["action_count"],
        "s2_tools_used": s2["tools_used"],
        "s2_used_edit_source": s2["used_edit_source"],
        "actions_remaining_at_submission": s2["actions_remaining_at_submission"],
        "cost_guard_tripped": s2["cost_guard_tripped"],
        "protocol_violations": s2["protocol_violations"],
        "action_trace": s2["action_trace"],
        "episode_dir": s2["episode_dir"],

        "s2_input_tokens": s2["episode_input_tokens"],
        "s2_output_tokens": s2["episode_output_tokens"],
        "s2_think_tokens": s2["episode_think_tokens"],
        "total_cost_usd": round(s1_cost + s2["episode_cost_usd"], 8),
    })
    return out


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
    Thin wrapper: run_stage1() + run_stage2_and_score(), for callers that
    want the original single-shot combined behavior (e.g. the single-row
    pilot CLI). Prefer calling the two halves directly when Stage 1 should
    be cached/reused across multiple thinking-condition sweeps (see
    run_stratified_sweep.py).
    """
    stage1_result = run_stage1(client, model, row, max_retries=max_retries)
    return run_stage2_and_score(
        client, model, row, stage1_result, run_id=run_id,
        budget=budget, truncate_chars=truncate_chars,
        subprocess_timeout=subprocess_timeout,
        episode_cost_cap_usd=episode_cost_cap_usd,
        thinking_budget=thinking_budget, max_retries=max_retries,
    )


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

        # estimate_cost_usd() assumes ~120 flat output tokens/call and has no
        # thinking-token term, so with thinking enabled it would silently
        # undercount -- add a worst-case per-call thinking margin.
        think_margin = args.thinking_budget * _PRICE_THINK if args.thinking_budget else 0.0
        est = (estimate_cost_usd(PROMPT_S1_AGENTIC.format(code=str(row["code"]))) + think_margin) * (args.budget + 2)
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
        log_path = write_episode_log(result, args.output_dir, args.run_id)
        new_rows += 1
        print(f"  log: {log_path}", flush=True)
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
