"""
Integration test for run_agentic_stage2() using a scripted fake Gemini client --
zero network calls, zero API cost. Exercises the full manual function-calling
protocol: a diff that fails to apply (still counts against budget), a diff that
succeeds and saves an npz, a run_diagnostic that reads it back, and a voluntary
submit_final_answer. A second scenario exercises forced completion (budget
exhausted). A third exercises the cost-guard trip.
"""
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from google.genai import types

from frontier.run_belief_revision_agentic import run_agentic_stage2
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
    def __init__(self, inp=100, out=20, think=0):
        self.prompt_token_count = inp
        self.candidates_token_count = out
        self.thoughts_token_count = think


class FakeResponse:
    def __init__(self, calls, usage=None):
        self.function_calls = calls
        self.usage_metadata = usage or FakeUsage()


class FakeModels:
    def __init__(self, scripted_calls):
        # scripted_calls: list of lists of types.FunctionCall, one entry per turn
        self._scripted = scripted_calls
        self.n_calls = 0

    def generate_content(self, model, contents, config):
        calls = self._scripted[self.n_calls]
        self.n_calls += 1
        return FakeResponse(calls)


class FakeClient:
    def __init__(self, scripted_calls):
        self.models = FakeModels(scripted_calls)


CODE = "u = [1, 2, 3]\nprint(sum(u))\n"
PROMPT_S1 = "irrelevant for this test"
S1_TEXT = "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: yes"


def fc(name, **kwargs):
    return types.FunctionCall(name=name, args=kwargs)


# ── Scenario 1: bad diff (counts against budget) -> good diff+save -> diagnostic -> voluntary submit ──

print("\n── voluntary stop, mixed success/failure actions ──")

TITLE1, RUN1 = "_test_loop_scenario1", "unittest"
for p in (episode_dir(TITLE1, RUN1), snapshot_root(TITLE1, RUN1)):
    if p.exists():
        shutil.rmtree(p)

bad_diff = (
    "--- a/solver.py\n+++ b/solver.py\n@@ -1,2 +1,2 @@\n"
    "-u = [999, 2, 3]\n+u = [1, 2, 3]\n print(sum(u))\n"
)
good_diff = (
    "--- a/solver.py\n+++ b/solver.py\n@@ -1,2 +1,3 @@\n"
    " u = [1, 2, 3]\n+import numpy as np; np.savez('h.npz', u=np.array(u))\n print(sum(u))\n"
)
diagnostic_script = (
    "import numpy as np\n"
    "d = np.load('h.npz')\n"
    "print('u contents:', d['u'].tolist())\n"
)

scripted = [
    [fc("edit_source", diff=bad_diff)],
    [fc("edit_source", diff=good_diff)],
    [fc("run_diagnostic", script=diagnostic_script)],
    [fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
        pde_exp="unchanged", method_exp="unchanged", behavior_exp="unchanged", valid_exp="confirmed via npz readback")],
]
client = FakeClient(scripted)

result = run_agentic_stage2(
    client, "gemini-2.5-flash", TITLE1, RUN1, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("action_count is 3 (bad diff + good diff + diagnostic)", result["action_count"] == 3, str(result["action_count"]))
check("used_edit_source is True", result["used_edit_source"] is True)
check("tools_used has both investigative tools", set(result["tools_used"]) == {"edit_source", "run_diagnostic"})
check("actions_remaining_at_submission is 3 (voluntary stop, budget=6)", result["actions_remaining_at_submission"] == 3, str(result))
check("cost_guard_tripped is False", result["cost_guard_tripped"] is False)
check("submit_args carries the final answer", result["submit_args"]["valid"] == "yes")
check("action_trace has 4 entries (3 investigative + submit)", len(result["action_trace"]) == 4)
check("first action's result reports the patch failure", "patch failed" in result["action_trace"][0]["result"])
check("first action created no new solver file", result["action_trace"][0]["new_filename"] is None)
check("second action's result shows execution output", "6" in result["action_trace"][1]["result"])  # sum([1,2,3])
check("diagnostic action's result shows the readback", "u contents" in result["action_trace"][2]["result"])

for p in (episode_dir(TITLE1, RUN1), snapshot_root(TITLE1, RUN1)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 2: forced completion (model never submits voluntarily, budget=2) ──

print("\n── forced completion when budget exhausted ──")

TITLE2, RUN2 = "_test_loop_scenario2", "unittest"
for p in (episode_dir(TITLE2, RUN2), snapshot_root(TITLE2, RUN2)):
    if p.exists():
        shutil.rmtree(p)

noop_diff = ""
scripted2 = [
    [fc("edit_source", diff=noop_diff)],
    [fc("run_diagnostic", script="print('looking around')\n")],
    # budget (2) is now exhausted -- only submit_final_answer is offered next turn,
    # and the harness only ever dispatches the FIRST call in a turn's response,
    # so even if the fake client scripted something else here it would be turn 3's
    # forced-submit-only turn
    [fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
        pde_exp="ran out of budget", method_exp="ran out of budget",
        behavior_exp="ran out of budget", valid_exp="ran out of budget")],
]
client2 = FakeClient(scripted2)

result2 = run_agentic_stage2(
    client2, "gemini-2.5-flash", TITLE2, RUN2, CODE, PROMPT_S1, S1_TEXT,
    budget=2, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("action_count equals the budget", result2["action_count"] == 2)
check("actions_remaining_at_submission is 0 (forced completion)", result2["actions_remaining_at_submission"] == 0)

for p in (episode_dir(TITLE2, RUN2), snapshot_root(TITLE2, RUN2)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 3: cost guard trips before budget is exhausted ──

print("\n── cost guard trips before the action budget runs out ──")

TITLE3, RUN3 = "_test_loop_scenario3", "unittest"
for p in (episode_dir(TITLE3, RUN3), snapshot_root(TITLE3, RUN3)):
    if p.exists():
        shutil.rmtree(p)

expensive_usage = FakeUsage(inp=10_000_000, out=1_000_000, think=0)  # deliberately huge -> trips a small cap

class ExpensiveFakeModels(FakeModels):
    def generate_content(self, model, contents, config):
        calls = self._scripted[self.n_calls]
        self.n_calls += 1
        return FakeResponse(calls, usage=expensive_usage)

class ExpensiveFakeClient:
    def __init__(self, scripted_calls):
        self.models = ExpensiveFakeModels(scripted_calls)

scripted3 = [
    [fc("edit_source", diff="")],
    # cost guard should have tripped after turn 1's huge usage -- turn 2 should
    # only ever be offered submit_final_answer, so script a submit here
    [fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
        pde_exp="cost capped", method_exp="cost capped", behavior_exp="cost capped", valid_exp="cost capped")],
]
client3 = ExpensiveFakeClient(scripted3)

result3 = run_agentic_stage2(
    client3, "gemini-2.5-flash", TITLE3, RUN3, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.0001,
)

check("cost_guard_tripped is True", result3["cost_guard_tripped"] is True)
check("action_count is 1 (only the one edit_source before the cap tripped)", result3["action_count"] == 1, str(result3))

for p in (episode_dir(TITLE3, RUN3), snapshot_root(TITLE3, RUN3)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 4: model keeps calling a tool that isn't declared to it anymore ──
# Regression test for the real behavior observed in the Task 8 live pilot: the
# model kept emitting edit_source calls even after the budget was exhausted and
# only submit_final_answer was declared to it. The harness must reject these
# (not dispatch them, not count them against budget) and force a synthetic
# submission after MAX_PROTOCOL_VIOLATIONS+1 consecutive violations.

print("\n── rejects tool calls for tools not declared that turn ──")

TITLE4, RUN4 = "_test_loop_scenario4", "unittest"
for p in (episode_dir(TITLE4, RUN4), snapshot_root(TITLE4, RUN4)):
    if p.exists():
        shutil.rmtree(p)

# budget=1: turn 1 legitimately uses edit_source (a no-op diff, quick and cheap).
# Turns 2+ should only ever declare submit_final_answer -- but the fake client
# keeps trying edit_source anyway, exactly like the real pilot did.
scripted4 = [
    [fc("edit_source", diff="")],           # turn 1: legitimate, uses up the budget of 1
    [fc("edit_source", diff="")],           # turn 2: should be REJECTED (not declared)
    [fc("edit_source", diff="")],           # turn 3: should be REJECTED (violation #2)
    [fc("edit_source", diff="")],           # turn 4: should be REJECTED (violation #3 -> forces submission)
]
client4 = FakeClient(scripted4)

result4 = run_agentic_stage2(
    client4, "gemini-2.5-flash", TITLE4, RUN4, CODE, PROMPT_S1, S1_TEXT,
    budget=1, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("action_count is 1 (only the legitimate turn-1 action counts)", result4["action_count"] == 1, str(result4["action_count"]))
check("protocol_violations is 3 (turns 2, 3, 4 all rejected)", result4["protocol_violations"] == 3, str(result4["protocol_violations"]))
check("submit_args is the forced empty submission", result4["submit_args"].get("_forced_reason") == "protocol_violations_exceeded", str(result4["submit_args"]))
check("action_trace has 4 entries (1 legit + 3 rejected)", len(result4["action_trace"]) == 4, str(len(result4["action_trace"])))
check("rejected entries are marked as such", all(a.get("rejected") for a in result4["action_trace"][1:]))
check("client only saw 4 turns (no extra API calls beyond the violation cap)", client4.models.n_calls == 4, str(client4.models.n_calls))

for p in (episode_dir(TITLE4, RUN4), snapshot_root(TITLE4, RUN4)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 5: model returns plain text with no function call at all ────────
# Regression test for a gap found live: a real pilot run hit this exact path
# (after a rejected tool call, the model replied with plain text instead of
# calling submit_final_answer) and the model's actual text was being silently
# discarded -- a violation of "never discard a model response." Must now be
# captured in both the action_trace and the forced submission's _raw_text.

print("\n── no function call at all -- model's text must not be discarded ──")

TITLE5, RUN5 = "_test_loop_scenario5", "unittest"
for p in (episode_dir(TITLE5, RUN5), snapshot_root(TITLE5, RUN5)):
    if p.exists():
        shutil.rmtree(p)


class NoCallResponse:
    def __init__(self, text):
        self.function_calls = None
        self.text = text
        self.usage_metadata = FakeUsage()


class NoCallFakeModels:
    def __init__(self):
        self.n_calls = 0

    def generate_content(self, model, contents, config):
        self.n_calls += 1
        return NoCallResponse("I'm not sure what to do here, apologies for the confusion.")


class NoCallFakeClient:
    def __init__(self):
        self.models = NoCallFakeModels()


client5 = NoCallFakeClient()

result5 = run_agentic_stage2(
    client5, "gemini-2.5-flash", TITLE5, RUN5, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("submit_args forced with no_function_call reason", result5["submit_args"].get("_forced_reason") == "no_function_call")
check("model's raw text is captured, not discarded", result5["submit_args"].get("_raw_text") == "I'm not sure what to do here, apologies for the confusion.", str(result5["submit_args"]))
check("action_trace records the no-function-call turn with the raw text",
      any(a.get("no_function_call") and a["result"] == "I'm not sure what to do here, apologies for the confusion." for a in result5["action_trace"]),
      str(result5["action_trace"]))
check("action_count is 0 (no investigative action ever ran)", result5["action_count"] == 0)

for p in (episode_dir(TITLE5, RUN5), snapshot_root(TITLE5, RUN5)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 6: harness actually builds a forced tool_config every turn ──────
# Verifies the wiring itself, not just that things still work when the fake
# client ignores it: on every turn, the config passed to generate_content must
# force mode="ANY" with allowed_function_names exactly equal to whatever
# tools_available() said was available that turn.

print("\n── every turn's config forces mode=ANY with the correct allowed tool names ──")

TITLE6, RUN6 = "_test_loop_scenario6", "unittest"
for p in (episode_dir(TITLE6, RUN6), snapshot_root(TITLE6, RUN6)):
    if p.exists():
        shutil.rmtree(p)


class ConfigCapturingModels(FakeModels):
    def __init__(self, scripted_calls):
        super().__init__(scripted_calls)
        self.seen_configs = []

    def generate_content(self, model, contents, config):
        self.seen_configs.append(config)
        return super().generate_content(model, contents, config)


class ConfigCapturingClient:
    def __init__(self, scripted_calls):
        self.models = ConfigCapturingModels(scripted_calls)


scripted6 = [
    [fc("edit_source", diff="")],
    [fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
        pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")],
]
client6 = ConfigCapturingClient(scripted6)

run_agentic_stage2(
    client6, "gemini-2.5-flash", TITLE6, RUN6, CODE, PROMPT_S1, S1_TEXT,
    budget=1, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

configs = client6.models.seen_configs
check("saw exactly 2 turns", len(configs) == 2, str(len(configs)))

fcc0 = configs[0].tool_config.function_calling_config
check("turn 1 mode is ANY", fcc0.mode == "ANY", str(fcc0.mode))
check("turn 1 allows all 3 tools (budget not yet used)",
      set(fcc0.allowed_function_names) == {"edit_source", "run_diagnostic", "submit_final_answer"},
      str(fcc0.allowed_function_names))

fcc1 = configs[1].tool_config.function_calling_config
check("turn 2 mode is ANY", fcc1.mode == "ANY", str(fcc1.mode))
check("turn 2 only allows submit_final_answer (budget=1 exhausted after turn 1)",
      list(fcc1.allowed_function_names) == ["submit_final_answer"],
      str(fcc1.allowed_function_names))

for p in (episode_dir(TITLE6, RUN6), snapshot_root(TITLE6, RUN6)):
    if p.exists():
        shutil.rmtree(p)


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
