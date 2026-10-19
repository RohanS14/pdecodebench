"""
agentic_tools.py — pure logic for the agentic belief-revision episode loop:
versioned-filename bookkeeping, per-turn tool availability given the
investigative budget, output truncation, and a per-episode API-cost circuit
breaker.

No network calls, no Gemini imports — this module is exercised entirely by
tests/test_agentic_tools.py with no API key required.
"""
import re

INVESTIGATIVE_TOOLS = ("edit_source", "run_diagnostic")


def next_version_filename(existing_filenames: list[str]) -> str:
    """Given files like ['solver_v0.py', 'solver_v2.py'], return the next name
    in the linear version sequence (e.g. 'solver_v3.py'). Never reuses a number."""
    versions = []
    for name in existing_filenames:
        m = re.fullmatch(r"solver_v(\d+)\.py", name)
        if m:
            versions.append(int(m.group(1)))
    next_n = (max(versions) + 1) if versions else 0
    return f"solver_v{next_n}.py"


def tools_available(actions_used: int, max_actions: int, cost_guard_tripped: bool) -> list[str]:
    """Which tool names should be offered to the model this turn.

    submit_final_answer is always available. edit_source/run_diagnostic drop out
    once the investigative budget is exhausted OR the per-episode cost guard has
    tripped -- both are treated identically: only submit_final_answer remains.
    """
    if actions_used >= max_actions or cost_guard_tripped:
        return ["submit_final_answer"]
    return ["edit_source", "run_diagnostic", "submit_final_answer"]


def truncate(text: str | None, cap: int) -> str:
    """Hard character-cap backstop. A truncated result says so explicitly --
    itself useful signal to the model that its next query should be narrower."""
    if text is None:
        return ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated, {len(text) - cap} more characters]"


class EpisodeCostGuard:
    """Tracks cumulative Stage-2 API cost for one episode; trips once at or
    over the cap. Independent of run_belief_revision.py's row-level
    --max_cost_usd session guard -- this one protects a single episode from
    an unexpectedly long or expensive agentic loop."""

    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0

    def add(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

    def tripped(self) -> bool:
        return self.spent_usd >= self.max_cost_usd
