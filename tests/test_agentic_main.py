"""
End-to-end dry run of process_row() with a fake Gemini client covering BOTH
Stage 1 (plain text response, no tools) and the agentic Stage 2 loop. Zero
network calls, zero API cost. Confirms the full JSONL row shape before any
real API key is involved.
"""
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freegen_static_judgments'))

from google.genai import types

from frontier.run_belief_revision_agentic import process_row, run_stage1, run_stage2_and_score
from frontier.agentic_sandbox import episode_dir, snapshot_root

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


class FakeUsage:
    def __init__(self):
        self.prompt_token_count = 100
        self.candidates_token_count = 20
        self.thoughts_token_count = 0


class FakeResponse:
    def __init__(self, text=None, calls=None):
        self.text = text
        self.function_calls = calls
        self.usage_metadata = FakeUsage()


class FakeModels:
    def __init__(self, stage1_text, stage2_scripted_calls):
        self._stage1_text = stage1_text
        self._stage2 = stage2_scripted_calls
        self.n_calls = 0
        self._stage2_idx = 0

    def generate_content(self, model, contents, config):
        # Stage 1 call has no tools configured; every agentic Stage-2 call does.
        if not getattr(config, "tools", None):
            self.n_calls += 1
            return FakeResponse(text=self._stage1_text)
        calls = self._stage2[self._stage2_idx]
        self._stage2_idx += 1
        self.n_calls += 1
        return FakeResponse(calls=calls)


class FakeClient:
    def __init__(self, stage1_text, stage2_scripted_calls):
        self.models = FakeModels(stage1_text, stage2_scripted_calls)


def fc(name, **kwargs):
    return types.FunctionCall(name=name, args=kwargs)


TITLE = "_test_main_Heat_Comm_Valid_1"
RUN_ID = "unittest"
for p in (episode_dir(TITLE, RUN_ID), snapshot_root(TITLE, RUN_ID)):
    if p.exists():
        shutil.rmtree(p)

row = {
    "title": TITLE,
    "mod_type": "Comm_Valid",
    "code": "u = [1, 2, 3]\nprint(sum(u))\n",
    "phys_valid": True,
    "pde_class": "heat",
    "num_method": "explicit",
    "phys_process": "diffusion",
}

stage1_text = (
    "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: yes\n"
    "pde_exp: looks like the heat equation.\n"
    "method_exp: explicit finite differences.\n"
    "behavior_exp: heat spreads out over time.\n"
    "valid_exp: boundary conditions look fine.\n"
)
submit_call = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                  pde_exp="unchanged", method_exp="unchanged", behavior_exp="unchanged", valid_exp="unchanged")
stage2_calls = [
    [submit_call],  # voluntary (budget=6, 0 actions used) -- intercepted once
    [submit_call],  # confirmed -- accepted as final
]
client = FakeClient(stage1_text, stage2_calls)

result = process_row(
    client, "gemini-2.5-flash", row, run_id=RUN_ID,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    max_retries=4,
)

check("row has title",              result["title"] == TITLE)
check("row has s1_parsed_valid",    result["s1_parsed_valid"] == "yes")
check("row has s1_valid_match",     result["s1_valid_match"] == 1)
check("row has s2_action_count",    result["s2_action_count"] == 0)
check("row has s2_submit_valid",    result["s2_submit_args"]["valid"] == "yes")
check("row has s2_valid_match via bypassed parse_response", result["s2_valid_match"] == 1, str(result.get("s2_valid_match")))
check("row has actions_remaining_at_submission", result["actions_remaining_at_submission"] == 6)
check("row has episode_dir",        "episode_dir" in result and result["episode_dir"])
check("row has total_cost_usd",     "total_cost_usd" in result)

check("row has s1_pde_exp",      result["s1_pde_exp"] == "looks like the heat equation.", str(result.get("s1_pde_exp")))
check("row has s1_method_exp",   result["s1_method_exp"] == "explicit finite differences.", str(result.get("s1_method_exp")))
check("row has s1_behavior_exp", result["s1_behavior_exp"] == "heat spreads out over time.", str(result.get("s1_behavior_exp")))
check("row has s1_valid_exp",    result["s1_valid_exp"] == "boundary conditions look fine.", str(result.get("s1_valid_exp")))

# Raw token counts persisted (not just derived cost) -- FakeUsage returns a
# fixed 100 input / 20 output / 0 think tokens on every call. 1 Stage-1 call,
# 2 Stage-2 calls (provisional + confirmed submit, since the submit call is
# voluntary here -- budget=6, 0 actions used).
check("row has s1_input_tokens",  result["s1_input_tokens"] == 100, str(result.get("s1_input_tokens")))
check("row has s1_output_tokens", result["s1_output_tokens"] == 20, str(result.get("s1_output_tokens")))
check("row has s1_think_tokens",  result["s1_think_tokens"] == 0, str(result.get("s1_think_tokens")))
check("row has s2_input_tokens (summed across both Stage-2 calls)",  result["s2_input_tokens"] == 200, str(result.get("s2_input_tokens")))
check("row has s2_output_tokens (summed across both Stage-2 calls)", result["s2_output_tokens"] == 40, str(result.get("s2_output_tokens")))
check("row has s2_think_tokens",  result["s2_think_tokens"] == 0, str(result.get("s2_think_tokens")))

# New schema additions (added for the stratified-sweep pipeline): transition,
# hedge classes, thinking_budget. gt_valid=True, s1/s2 both correctly say
# "yes" -> right->right.
check("row has thinking_budget",  result["thinking_budget"] == 0, str(result.get("thinking_budget")))
check("row has transition",       result["transition"] == "right->right", str(result.get("transition")))
check("row has s1_hedge_class",   result["s1_hedge_class"] == "Confident Yes", str(result.get("s1_hedge_class")))
check("row has s2_hedge_class",   result["s2_hedge_class"] == "Confident Yes", str(result.get("s2_hedge_class")))
check("internal-only fields (code, prompt_s1, s1_cost_usd) are not leaked into the output row",
      "code" not in result and "prompt_s1" not in result and "s1_cost_usd" not in result)

for p in (episode_dir(TITLE, RUN_ID), snapshot_root(TITLE, RUN_ID)):
    if p.exists():
        shutil.rmtree(p)


# ── run_stage1()/run_stage2_and_score() split: Stage 1 cached once, reused
# across two different thinking_budget conditions -- the actual pattern the
# stratified sweep needs. Regression test for the process_row() refactor. ──

print("\n── run_stage1() + run_stage2_and_score() split, Stage 1 cached across 2 thinking conditions ──")

TITLE2, RUN_ID2 = "_test_main_split_Heat_Comm_Valid_1", "unittest_split"
for p in (episode_dir(TITLE2, RUN_ID2), snapshot_root(TITLE2, RUN_ID2)):
    if p.exists():
        shutil.rmtree(p)

row2 = dict(row)
row2["title"] = TITLE2


class SplitFakeModels:
    """Tracks Stage-1 calls separately from Stage-2 calls, to confirm
    run_stage1() is only ever invoked once even though run_stage2_and_score()
    is called twice (once per thinking condition) against the same cached
    result."""
    def __init__(self, stage1_text, stage2_scripted_calls):
        self._stage1_text = stage1_text
        self._stage2 = stage2_scripted_calls
        self.stage1_calls = 0
        self.stage2_calls = 0
        self._stage2_idx = 0

    def generate_content(self, model, contents, config):
        if not getattr(config, "tools", None):
            self.stage1_calls += 1
            return FakeResponse(text=self._stage1_text)
        calls = self._stage2[self._stage2_idx]
        self._stage2_idx += 1
        self.stage2_calls += 1
        return FakeResponse(calls=calls)


class SplitFakeClient:
    def __init__(self, stage1_text, stage2_scripted_calls):
        self.models = SplitFakeModels(stage1_text, stage2_scripted_calls)


# 2 conditions x (provisional + confirmed submit) = 4 Stage-2 calls total.
split_stage2_calls = [[submit_call], [submit_call], [submit_call], [submit_call]]
split_client = SplitFakeClient(stage1_text, split_stage2_calls)

stage1_result = run_stage1(split_client, "gemini-2.5-flash", row2, max_retries=4)
check("stage1_result carries internal code/prompt_s1 fields for run_stage2_and_score()",
      "code" in stage1_result and "prompt_s1" in stage1_result)

row_nothink = run_stage2_and_score(
    split_client, "gemini-2.5-flash", row2, stage1_result, run_id=RUN_ID2,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    thinking_budget=0,
)
row_think = run_stage2_and_score(
    split_client, "gemini-2.5-flash", row2, stage1_result, run_id=RUN_ID2,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    thinking_budget=1536,
)

check("Stage 1 was called exactly once despite 2 Stage-2 conditions", split_client.models.stage1_calls == 1, str(split_client.models.stage1_calls))
check("Stage 2 was called 4 times (2 conditions x provisional+confirmed)", split_client.models.stage2_calls == 4, str(split_client.models.stage2_calls))
check("both output rows share identical Stage-1 fields (cached, not re-run)",
      row_nothink["s1_response"] == row_think["s1_response"]
      and row_nothink["s1_valid_match"] == row_think["s1_valid_match"]
      and row_nothink["s1_hedge_class"] == row_think["s1_hedge_class"],
      (row_nothink["s1_response"], row_think["s1_response"]))
check("the two rows differ in thinking_budget", row_nothink["thinking_budget"] == 0 and row_think["thinking_budget"] == 1536)
check("both rows have transition computed", row_nothink["transition"] == "right->right" and row_think["transition"] == "right->right")

for p in (episode_dir(TITLE2, RUN_ID2), snapshot_root(TITLE2, RUN_ID2)):
    if p.exists():
        shutil.rmtree(p)


print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
