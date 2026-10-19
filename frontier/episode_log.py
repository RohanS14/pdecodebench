"""
episode_log.py — renders a single process_row()/run_agentic_stage2() result
dict into a human-readable, turn-by-turn text log (Stage 1 response, then
every Stage 2 action_trace entry: thought_summary, reasoning_text, args,
result, flags), plus the final submitted answer and match/cost summary.

This is the same format manually built ad hoc for pilot inspection throughout
this project's agentic-harness development -- pulled out into a real function
so every run (not just manually-inspected pilots) gets one automatically,
saved next to the run's output JSONL rather than a session-scoped scratch
directory.
"""
import json
from pathlib import Path


def render_episode_log(row: dict) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"EPISODE LOG: {row.get('title')} (model={row.get('model')})")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        f"Ground truth: valid={row.get('gt_valid')}  pde={row.get('gt_pde')}  "
        f"method={row.get('gt_method')}  behavior={row.get('gt_behavior')}"
    )
    lines.append(f"Episode dir: {row.get('episode_dir')}")
    lines.append(f"Total cost: ${row.get('total_cost_usd', 0):.5f}")
    lines.append(f"Protocol violations: {row.get('protocol_violations')}")
    lines.append(f"Cost guard tripped: {row.get('cost_guard_tripped')}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("STAGE 1 (read-only, no tools)")
    lines.append("-" * 100)
    lines.append(row.get("s1_response") or "(none)")
    lines.append("")
    lines.append(
        f"Parsed: pde={row.get('s1_parsed_pde')!r} method={row.get('s1_parsed_method')!r} "
        f"behavior={row.get('s1_parsed_behavior')!r} valid={row.get('s1_parsed_valid')!r}"
    )
    lines.append(
        f"s1_valid_match={row.get('s1_valid_match')}  s1_pde_match={row.get('s1_pde_match')}"
    )
    lines.append("")
    lines.append("Explanations:")
    for f in ("s1_pde_exp", "s1_method_exp", "s1_behavior_exp", "s1_valid_exp"):
        lines.append(f"  {f}: {row.get(f)}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("STAGE 2 (agentic loop) -- turn by turn action_trace")
    lines.append("-" * 100)
    for i, turn in enumerate(row.get("action_trace") or []):
        lines.append("")
        lines.append(f"=== Turn {i + 1} (tool={turn.get('tool')}) ===")
        flags = [
            k for k in (
                "text_only", "empty_response", "no_function_call",
                "rejected", "possible_thought_leak", "provisional_submit",
            )
            if turn.get(k)
        ]
        if flags:
            lines.append(f"*** FLAGS: {flags} ***")
        lines.append("")
        lines.append("--- thought_summary ---")
        lines.append(turn.get("thought_summary") or "(none)")
        lines.append("")
        lines.append("--- reasoning_text (accompanying visible text) ---")
        lines.append(turn.get("reasoning_text") or "(none)")
        lines.append("")
        lines.append("--- args ---")
        lines.append(json.dumps(turn.get("args"), indent=2))
        lines.append("")
        lines.append("--- result ---")
        result = turn.get("result")
        lines.append(str(result)[:3000] if result else "(none / null -- terminal turn)")

    lines.append("")
    lines.append("-" * 100)
    lines.append("FINAL SUBMITTED ANSWER (Stage 2)")
    lines.append("-" * 100)
    lines.append(json.dumps(row.get("s2_submit_args"), indent=2))
    lines.append("")
    lines.append(
        f"s2_valid_match={row.get('s2_valid_match')}  s2_pde_match={row.get('s2_pde_match')}"
    )
    lines.append(
        f"s2_action_count={row.get('s2_action_count')}  "
        f"actions_remaining_at_submission={row.get('actions_remaining_at_submission')}"
    )
    lines.append(
        f"tools_used={row.get('s2_tools_used')}  used_edit_source={row.get('s2_used_edit_source')}"
    )
    return "\n".join(lines)


def write_episode_log(row: dict, output_dir: Path, run_id: str) -> Path:
    """Writes the rendered log to <output_dir>/logs/<title>__<run_id>.txt and
    returns the path. Called automatically once per row from main()'s loop --
    not something you need to invoke manually for a real run."""
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{row['title']}__{run_id}.txt"
    path.write_text(render_episode_log(row))
    return path
