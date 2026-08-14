"""
End-to-end dry run of process_row() with a fake Gemini client covering BOTH
Stage 1 (plain text response, no tools) and the agentic Stage 2 loop. Zero
network calls, zero API cost. Confirms the full JSONL row shape before any
real API key is involved.
"""
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from google.genai import types

from frontier.run_belief_revision_agentic import process_row
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

stage1_text = "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: yes"
stage2_calls = [
    [fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
        pde_exp="unchanged", method_exp="unchanged", behavior_exp="unchanged", valid_exp="unchanged")],
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

for p in (episode_dir(TITLE, RUN_ID), snapshot_root(TITLE, RUN_ID)):
    if p.exists():
        shutil.rmtree(p)


print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
