"""
Integration test for run_agentic_stage2() using a scripted fake Gemini client --
zero network calls, zero API cost. Exercises the full manual function-calling
protocol: a full-file rewrite with a syntax error (still counts against
budget), a full-file rewrite that runs successfully and saves an npz, a
run_diagnostic that reads it back, and a voluntary submit_final_answer. A
second scenario exercises forced completion (budget exhausted). A third
exercises the cost-guard trip.
"""
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freegen'))

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
    def __init__(self, calls, text=None, usage=None, candidates=None):
        self.function_calls = calls
        self.text = text
        self.usage_metadata = usage or FakeUsage()
        # Only set when a scenario explicitly needs to simulate thinking-mode
        # part structure (thought-summary parts, thought_signature) -- absent
        # by default, matching every real FakeResponse used before thinking
        # support existed, so getattr(resp, "candidates", None) or [] in the
        # harness falls back to the original one-part behavior untouched.
        if candidates is not None:
            self.candidates = candidates


class FakePart:
    """Minimal stand-in for a real google.genai types.Part, for scenarios that
    need to simulate resp.candidates[0].content.parts (thought-summary
    extraction, thought_signature preservation)."""
    def __init__(self, text=None, thought=None, thought_signature=None, function_call=None):
        self.text = text
        self.thought = thought
        self.thought_signature = thought_signature
        self.function_call = function_call


class FakeCandidate:
    def __init__(self, parts):
        class _Content:
            pass
        content = _Content()
        content.parts = parts
        self.content = content


class FakeModels:
    def __init__(self, scripted_calls):
        # scripted_calls: list of turn specs, one per turn. Each entry is either
        # a plain list of types.FunctionCall (shorthand for {"calls": [...],
        # "text": None} -- the convention every existing scenario already uses),
        # or a dict {"calls": [...], "text": "...", "candidates": [...]} for
        # scenarios that need to simulate text-only / empty / text+call
        # outcomes under VALIDATED mode, or thinking-mode part structure.
        self._scripted = scripted_calls
        self.n_calls = 0

    def generate_content(self, model, contents, config):
        entry = self._scripted[self.n_calls]
        self.n_calls += 1
        if isinstance(entry, dict):
            return FakeResponse(entry.get("calls") or [], text=entry.get("text"),
                                 candidates=entry.get("candidates"))
        return FakeResponse(entry)


class FakeClient:
    def __init__(self, scripted_calls):
        self.models = FakeModels(scripted_calls)


CODE = "u = [1, 2, 3]\nprint(sum(u))\n"
PROMPT_S1 = "irrelevant for this test"
S1_TEXT = "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: yes"


def fc(name, **kwargs):
    return types.FunctionCall(name=name, args=kwargs)


# ── Scenario 1: rewrite w/ syntax error (counts against budget) -> good rewrite+save -> diagnostic -> voluntary submit ──

print("\n── voluntary stop, mixed success/failure actions ──")

TITLE1, RUN1 = "_test_loop_scenario1", "unittest"
for p in (episode_dir(TITLE1, RUN1), snapshot_root(TITLE1, RUN1)):
    if p.exists():
        shutil.rmtree(p)

bad_source = "u = [1, 2, 3\nprint(sum(u))\n"  # missing closing bracket -> SyntaxError on run
good_source = (
    "import numpy as np\n"
    "u = [1, 2, 3]\n"
    "np.savez('h.npz', u=np.array(u))\n"
    "print(sum(u))\n"
)
diagnostic_script = (
    "import numpy as np\n"
    "d = np.load('h.npz')\n"
    "print('u contents:', d['u'].tolist())\n"
)

submit_call = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                  pde_exp="unchanged", method_exp="unchanged", behavior_exp="unchanged", valid_exp="confirmed via npz readback")
scripted = [
    [fc("edit_source", full_source=bad_source)],
    [fc("edit_source", full_source=good_source)],
    [fc("run_diagnostic", script=diagnostic_script)],
    [submit_call],  # voluntary (budget not exhausted) -- gets intercepted once
    [submit_call],  # confirmed -- accepted as final
]
client = FakeClient(scripted)

result = run_agentic_stage2(
    client, "gemini-2.5-flash", TITLE1, RUN1, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("action_count is 3 (bad rewrite + good rewrite + diagnostic; neither submit call counts)", result["action_count"] == 3, str(result["action_count"]))
check("used_edit_source is True", result["used_edit_source"] is True)
check("tools_used has both investigative tools", set(result["tools_used"]) == {"edit_source", "run_diagnostic"})
check("actions_remaining_at_submission is 3 (voluntary stop, budget=6)", result["actions_remaining_at_submission"] == 3, str(result))
check("cost_guard_tripped is False", result["cost_guard_tripped"] is False)
check("submit_args carries the final answer", result["submit_args"]["valid"] == "yes")
check("action_trace has 5 entries (3 investigative + provisional submit + confirmed submit)", len(result["action_trace"]) == 5, str(len(result["action_trace"])))
check("first action's result reports a SyntaxError", "SyntaxError" in result["action_trace"][0]["result"], result["action_trace"][0]["result"])
check("first action still creates a new solver file (write always succeeds; only execution fails)",
      result["action_trace"][0]["new_filename"] is not None)
check("second action's result shows execution output", "6" in result["action_trace"][1]["result"])  # sum([1,2,3])
check("diagnostic action's result shows the readback", "u contents" in result["action_trace"][2]["result"])
check("4th entry is the intercepted provisional submit, with the recap in its result",
      result["action_trace"][3].get("provisional_submit") is True
      and CODE in result["action_trace"][3]["result"]
      and S1_TEXT in result["action_trace"][3]["result"],
      result["action_trace"][3])
check("5th entry is the real, confirmed submit (not flagged provisional)",
      not result["action_trace"][4].get("provisional_submit"), result["action_trace"][4])

for p in (episode_dir(TITLE1, RUN1), snapshot_root(TITLE1, RUN1)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 2: forced completion (model never submits voluntarily, budget=2) ──

print("\n── forced completion when budget exhausted ──")

TITLE2, RUN2 = "_test_loop_scenario2", "unittest"
for p in (episode_dir(TITLE2, RUN2), snapshot_root(TITLE2, RUN2)):
    if p.exists():
        shutil.rmtree(p)

noop_full_source = ""  # empty -> rerun current latest version unchanged
scripted2 = [
    [fc("edit_source", full_source=noop_full_source)],
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
    [fc("edit_source", full_source="")],
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
    [fc("edit_source", full_source="")],           # turn 1: legitimate, uses up the budget of 1
    [fc("edit_source", full_source="")],           # turn 2: should be REJECTED (not declared)
    [fc("edit_source", full_source="")],           # turn 3: should be REJECTED (violation #2)
    [fc("edit_source", full_source="")],           # turn 4: should be REJECTED (violation #3 -> forces submission)
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

# The model returns the SAME text-only response every turn. Under the new design
# this is: turn 1 (VALIDATED) -> text-only -> escalate to ANY; turn 2 (ANY) ->
# still no call -> last-resort fallback. Two turns, not one, to reach the same
# terminal _forced_reason.
check("client saw exactly 2 turns (text-only, then escalated ANY)", client5.models.n_calls == 2, str(client5.models.n_calls))
check("submit_args forced with no_function_call reason", result5["submit_args"].get("_forced_reason") == "no_function_call")
check("model's raw text is captured, not discarded", result5["submit_args"].get("_raw_text") == "I'm not sure what to do here, apologies for the confusion.", str(result5["submit_args"]))
check("action_trace records the no-function-call turn with the raw text",
      any(a.get("no_function_call") and a["result"] == "I'm not sure what to do here, apologies for the confusion." for a in result5["action_trace"]),
      str(result5["action_trace"]))
check("action_trace also records the preceding text-only turn",
      any(a.get("text_only") and a["result"] == "I'm not sure what to do here, apologies for the confusion." for a in result5["action_trace"]),
      str(result5["action_trace"]))
check("action_count is 1 (the text-only turn still costs a turn; the last-resort ANY turn doesn't)", result5["action_count"] == 1, str(result5["action_count"]))

for p in (episode_dir(TITLE5, RUN5), snapshot_root(TITLE5, RUN5)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 6: harness builds the right tool_config every turn ──────────────
# Verifies the wiring itself, not just that things still work when the fake
# client ignores it: VALIDATED by default everywhere, including the terminal
# (budget-exhausted) phase -- ANY is only ever a one-shot escalation, not a
# hardcoded terminal special case -- with allowed_function_names exactly equal
# to whatever tools_available() said was available that turn, in both cases.

print("\n── every turn's config has the right mode and allowed tool names ──")

TITLE6, RUN6 = "_test_loop_scenario6", "unittest"
for p in (episode_dir(TITLE6, RUN6), snapshot_root(TITLE6, RUN6)):
    if p.exists():
        shutil.rmtree(p)


class ConfigCapturingModels(FakeModels):
    def __init__(self, scripted_calls):
        super().__init__(scripted_calls)
        self.seen_configs = []
        self.seen_contents = []  # snapshot (shallow copy) of contents per call

    def generate_content(self, model, contents, config):
        self.seen_configs.append(config)
        self.seen_contents.append(list(contents))
        return super().generate_content(model, contents, config)


class ConfigCapturingClient:
    def __init__(self, scripted_calls):
        self.models = ConfigCapturingModels(scripted_calls)


scripted6 = [
    [fc("edit_source", full_source="")],
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
check("turn 1 mode is VALIDATED (more than one tool still available)", fcc0.mode == "VALIDATED", str(fcc0.mode))
check("turn 1 allows all 3 tools (budget not yet used)",
      set(fcc0.allowed_function_names) == {"edit_source", "run_diagnostic", "submit_final_answer"},
      str(fcc0.allowed_function_names))

fcc1 = configs[1].tool_config.function_calling_config
check("turn 2 (terminal, compliant single call) mode is VALIDATED, not a hardcoded ANY",
      fcc1.mode == "VALIDATED", str(fcc1.mode))
check("turn 2 only allows submit_final_answer (budget=1 exhausted after turn 1)",
      list(fcc1.allowed_function_names) == ["submit_final_answer"],
      str(fcc1.allowed_function_names))

for p in (episode_dir(TITLE6, RUN6), snapshot_root(TITLE6, RUN6)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 7: text-only turn escalates to ANY, which then succeeds ────────
# Distinct from Scenario 5 (where the escalated ANY turn ALSO fails, ending in
# the last-resort fallback): here the model declines to act once, gets
# escalated, and complies on the very next (forced) turn -- the common,
# non-pathological case the escalation mechanism is meant to handle.

print("\n── text-only turn escalates to ANY, which then succeeds ──")

TITLE7, RUN7 = "_test_loop_scenario7", "unittest"
for p in (episode_dir(TITLE7, RUN7), snapshot_root(TITLE7, RUN7)):
    if p.exists():
        shutil.rmtree(p)

submit_call7 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                   pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")
scripted7 = [
    {"calls": [], "text": "I want to check the boundary conditions before touching anything."},
    [fc("edit_source", full_source="")],  # escalated ANY turn: complies immediately
    [submit_call7],  # voluntary (budget=6, only 2 actions used) -- gets intercepted once
    [submit_call7],  # confirmed -- accepted as final
]
client7 = ConfigCapturingClient(scripted7)

result7 = run_agentic_stage2(
    client7, "gemini-2.5-flash", TITLE7, RUN7, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

configs7 = client7.models.seen_configs
check("client saw 4 turns (text-only, escalated edit_source, provisional submit, confirmed submit)",
      client7.models.n_calls == 4, str(client7.models.n_calls))
check("turn 1 (text-only) was VALIDATED", configs7[0].tool_config.function_calling_config.mode == "VALIDATED")
check("turn 2 (escalated) was forced to ANY", configs7[1].tool_config.function_calling_config.mode == "ANY")
check("turn 3 (after successful escalation) is back to VALIDATED",
      configs7[2].tool_config.function_calling_config.mode == "VALIDATED",
      configs7[2].tool_config.function_calling_config.mode)
check("action_count is 2 (text-only turn + the real edit_source; neither submit call counts)", result7["action_count"] == 2, str(result7["action_count"]))
check("action_trace's first entry is marked text_only with the model's reasoning",
      result7["action_trace"][0].get("text_only") and
      result7["action_trace"][0]["result"] == "I want to check the boundary conditions before touching anything.",
      str(result7["action_trace"][0]))
check("submit_args carries the real final answer (not a forced empty one)", result7["submit_args"]["valid"] == "yes")
check("3rd action_trace entry is the intercepted provisional submit", result7["action_trace"][2].get("provisional_submit") is True, str(result7["action_trace"][2]))

for p in (episode_dir(TITLE7, RUN7), snapshot_root(TITLE7, RUN7)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 8: two consecutive empty turns escalate to ANY, which is ALSO
# empty -> last-resort fallback via the EMPTY path (Scenario 5 covers the same
# last-resort fallback reached via the TEXT-ONLY path instead) ─────────────

print("\n── two consecutive empty turns escalate to ANY, still empty -> last resort ──")

TITLE8, RUN8 = "_test_loop_scenario8", "unittest"
for p in (episode_dir(TITLE8, RUN8), snapshot_root(TITLE8, RUN8)):
    if p.exists():
        shutil.rmtree(p)

scripted8 = [
    {"calls": [], "text": None},   # turn 1: empty -- one retry allowed, stays VALIDATED
    {"calls": [], "text": ""},     # turn 2: empty again -- escalate to ANY next turn
    {"calls": [], "text": None},   # turn 3: forced ANY, STILL empty -> last resort
]
client8 = ConfigCapturingClient(scripted8)

result8 = run_agentic_stage2(
    client8, "gemini-2.5-flash", TITLE8, RUN8, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

configs8 = client8.models.seen_configs
check("client saw 3 turns", client8.models.n_calls == 3, str(client8.models.n_calls))
check("turn 1 (first empty, retry) was VALIDATED", configs8[0].tool_config.function_calling_config.mode == "VALIDATED")
check("turn 2 (second consecutive empty) was STILL VALIDATED (retry, not yet escalated)",
      configs8[1].tool_config.function_calling_config.mode == "VALIDATED")
check("turn 3 (escalated after 2nd consecutive empty) was forced to ANY",
      configs8[2].tool_config.function_calling_config.mode == "ANY")
check("action_count is 2 (both empty VALIDATED turns cost a turn; the last-resort ANY turn doesn't)",
      result8["action_count"] == 2, str(result8["action_count"]))
check("submit_args forced with no_function_call reason (last resort)",
      result8["submit_args"].get("_forced_reason") == "no_function_call", str(result8["submit_args"]))
check("exactly 2 empty_response entries recorded",
      sum(1 for a in result8["action_trace"] if a.get("empty_response")) == 2,
      str(result8["action_trace"]))

for p in (episode_dir(TITLE8, RUN8), snapshot_root(TITLE8, RUN8)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 9: text+call outcome captures the model's reasoning text ───────

print("\n── text+call outcome: reasoning text is captured in the action_trace ──")

TITLE9, RUN9 = "_test_loop_scenario9", "unittest"
for p in (episode_dir(TITLE9, RUN9), snapshot_root(TITLE9, RUN9)):
    if p.exists():
        shutil.rmtree(p)

submit_call9 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                   pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")
scripted9 = [
    {"calls": [fc("run_diagnostic", script="print('checking boundary values')\n")],
     "text": "I want to inspect the boundary values before deciding whether to edit anything."},
    {"calls": [submit_call9],
     "text": "Based on what I found, I'm ready to submit."},  # voluntary -- intercepted once
    {"calls": [submit_call9], "text": "Confirming my answer."},  # confirmed -- accepted as final
]
client9 = FakeClient(scripted9)

result9 = run_agentic_stage2(
    client9, "gemini-2.5-flash", TITLE9, RUN9, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("action_count is 1 (only run_diagnostic counts; neither submit call counts)", result9["action_count"] == 1, str(result9["action_count"]))
check("run_diagnostic's action_trace entry captures its accompanying reasoning_text",
      result9["action_trace"][0].get("reasoning_text") ==
      "I want to inspect the boundary values before deciding whether to edit anything.",
      str(result9["action_trace"][0]))
check("the intercepted (provisional) submit's action_trace entry captures its accompanying reasoning_text",
      result9["action_trace"][1].get("provisional_submit") is True
      and result9["action_trace"][1].get("reasoning_text") == "Based on what I found, I'm ready to submit.",
      str(result9["action_trace"][1]))
check("the confirmed (final) submit's action_trace entry ALSO captures its accompanying reasoning_text",
      not result9["action_trace"][2].get("provisional_submit")
      and result9["action_trace"][2].get("reasoning_text") == "Confirming my answer.",
      str(result9["action_trace"][2]))

for p in (episode_dir(TITLE9, RUN9), snapshot_root(TITLE9, RUN9)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 10: the terminal turn's reminder must fire on BOTH the default
# VALIDATED terminal turn AND an escalated ANY terminal turn -- regression
# test for a real bug found live (the original-snippet reminder was gated on
# mode=="VALIDATED", which used to be permanently unreachable in the terminal
# phase since that phase was hardcoded to ANY). Also exercises the terminal
# phase now defaulting to VALIDATED (not ANY) and escalating exactly like the
# investigative phase does when the model doesn't act. ─────────────────────

print("\n── terminal phase defaults to VALIDATED and shows the reminder on both VALIDATED and escalated ANY turns ──")

TITLE10, RUN10 = "_test_loop_scenario10", "unittest"
for p in (episode_dir(TITLE10, RUN10), snapshot_root(TITLE10, RUN10)):
    if p.exists():
        shutil.rmtree(p)

scripted10 = [
    [fc("edit_source", full_source="")],  # turn 1: uses up the budget of 1
    {"calls": [], "text": "I want to think this through before answering."},  # turn 2: terminal, VALIDATED, text-only -> escalate
    [fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
        pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")],  # turn 3: terminal, escalated ANY, complies
]
client10 = ConfigCapturingClient(scripted10)

run_agentic_stage2(
    client10, "gemini-2.5-flash", TITLE10, RUN10, CODE, PROMPT_S1, S1_TEXT,
    budget=1, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

configs10 = client10.models.seen_configs
contents10 = client10.models.seen_contents
check("client saw 3 turns", client10.models.n_calls == 3, str(client10.models.n_calls))
check("turn 2 (terminal, first attempt) defaults to VALIDATED, not a hardcoded ANY",
      configs10[1].tool_config.function_calling_config.mode == "VALIDATED",
      configs10[1].tool_config.function_calling_config.mode)
check("turn 3 (terminal, escalated after text-only) is forced to ANY",
      configs10[2].tool_config.function_calling_config.mode == "ANY",
      configs10[2].tool_config.function_calling_config.mode)

def _texts(contents_snapshot):
    return [part.text for content in contents_snapshot for part in content.parts if getattr(part, "text", None)]

turn2_texts = _texts(contents10[1])
turn3_texts = _texts(contents10[2])
check("turn 2's contents include a reminder mentioning the budget is exhausted",
      any("investigative budget is exhausted" in t for t in turn2_texts), turn2_texts)
check("turn 2's contents include the ORIGINAL code text",
      any(CODE in t for t in turn2_texts), turn2_texts)
check("turn 2's contents include the Stage-1 answer text",
      any(S1_TEXT in t for t in turn2_texts), turn2_texts)
check("turn 3's contents ALSO include the reminder + original code, despite being escalated to ANY",
      any("investigative budget is exhausted" in t for t in turn3_texts)
      and any(CODE in t for t in turn3_texts),
      turn3_texts)

for p in (episode_dir(TITLE10, RUN10), snapshot_root(TITLE10, RUN10)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 11: thinking mode -- thought_summary extraction, and the model's
# full non-thought content (with its thought_signature) is preserved when
# appended to history, instead of the minimal function-call-only Content. ───

print("\n── thinking mode: thought_summary captured, full content preserved on model turn ──")

TITLE11, RUN11 = "_test_loop_scenario11", "unittest"
for p in (episode_dir(TITLE11, RUN11), snapshot_root(TITLE11, RUN11)):
    if p.exists():
        shutil.rmtree(p)

call11 = fc("edit_source", full_source="")
thought_part11 = FakePart(text="internal reasoning about boundary conditions...", thought=True)
text_part11 = FakePart(text="Let me rerun the original unchanged.", thought=False, thought_signature=b"sig-bytes")
call_part11 = FakePart(function_call=call11)

submit_call11 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                    pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")
scripted11 = [
    {"calls": [call11], "text": "Let me rerun the original unchanged.",
     "candidates": [FakeCandidate([thought_part11, text_part11, call_part11])]},
    [submit_call11],  # voluntary (budget=6, only 1 action used) -- intercepted once
    [submit_call11],  # confirmed -- accepted as final
]
client11 = ConfigCapturingClient(scripted11)

result11 = run_agentic_stage2(
    client11, "gemini-2.5-flash", TITLE11, RUN11, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    thinking_budget=1536,
)

check("thought_summary captured on the action_trace entry",
      result11["action_trace"][0].get("thought_summary") == "internal reasoning about boundary conditions...",
      str(result11["action_trace"][0]))
check("no possible_thought_leak flagged (text doesn't start with THOUGHT:)",
      "possible_thought_leak" not in result11["action_trace"][0])

# seen_contents[1] is the contents snapshot taken at the START of turn 2's
# call -- i.e. AFTER turn 1's model-turn append already happened.
model_turn_contents11 = client11.models.seen_contents[1]
model_turns11 = [c for c in model_turn_contents11 if c.role == "model"]
model_turn11 = model_turns11[-1]  # the turn-1 append, not the seeded Stage-1-answer turn
check("model turn role is 'model'", model_turn11.role == "model")
check("model turn's parts are the full non-thought parts (text + function_call), not just function_call",
      len(model_turn11.parts) == 2
      and getattr(model_turn11.parts[0], "text", None) == "Let me rerun the original unchanged."
      and getattr(model_turn11.parts[1], "function_call", None) is not None,
      [getattr(p, "text", None) for p in model_turn11.parts])
check("the thought-summary part's text is excluded from the replayed model turn",
      not any(getattr(p, "text", None) == "internal reasoning about boundary conditions..." for p in model_turn11.parts))
check("thought_signature survives on the replayed text part",
      getattr(model_turn11.parts[0], "thought_signature", None) == b"sig-bytes")

for p in (episode_dir(TITLE11, RUN11), snapshot_root(TITLE11, RUN11)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 12: possible_thought_leak defensive check -- a plain-text
# response starting with "THOUGHT:" while thinking is enabled and no
# .candidates marks any part as a real thought summary. ─────────────────────

print("\n── thinking mode: THOUGHT:-prefix leak into regular text is flagged ──")

TITLE12, RUN12 = "_test_loop_scenario12", "unittest"
for p in (episode_dir(TITLE12, RUN12), snapshot_root(TITLE12, RUN12)):
    if p.exists():
        shutil.rmtree(p)

submit_call12 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                    pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")
scripted12 = [
    {"calls": [submit_call12],
     "text": "THOUGHT: this looks like leaked internal reasoning, then the real answer."},
    [submit_call12],  # confirmed -- accepted as final
]
client12 = FakeClient(scripted12)

result12 = run_agentic_stage2(
    client12, "gemini-2.5-flash", TITLE12, RUN12, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    thinking_budget=1536,
)

check("possible_thought_leak is flagged True",
      result12["action_trace"][0].get("possible_thought_leak") is True,
      str(result12["action_trace"][0]))
check("the model's full text is still preserved (never discarded, even when flagged)",
      result12["action_trace"][0].get("reasoning_text", "").startswith("THOUGHT:"),
      str(result12["action_trace"][0]))

for p in (episode_dir(TITLE12, RUN12), snapshot_root(TITLE12, RUN12)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 13: voluntary submission is intercepted once and requires a
# second call -- the forced/terminal path (Scenario 6/10) is NOT affected by
# this at all, since build_validated_reminder_final already grounds it
# pre-call; this is specifically the gap that had no re-grounding before. ──

print("\n── voluntary submission gets intercepted once, requires a second call ──")

TITLE13, RUN13 = "_test_loop_scenario13", "unittest"
for p in (episode_dir(TITLE13, RUN13), snapshot_root(TITLE13, RUN13)):
    if p.exists():
        shutil.rmtree(p)

submit_call13 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                    pde_exp="x", method_exp="x", behavior_exp="x", valid_exp="x")
scripted13 = [
    [submit_call13],  # voluntary (budget=6, 0 actions used) -- intercepted
    [submit_call13],  # confirmed -- accepted as final
]
client13 = ConfigCapturingClient(scripted13)

result13 = run_agentic_stage2(
    client13, "gemini-2.5-flash", TITLE13, RUN13, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("client saw 2 turns", client13.models.n_calls == 2, str(client13.models.n_calls))
check("action_trace has 2 entries (provisional + confirmed submit)", len(result13["action_trace"]) == 2, str(len(result13["action_trace"])))
check("1st entry is the intercepted provisional submit, with the recap in its result",
      result13["action_trace"][0].get("provisional_submit") is True
      and CODE in result13["action_trace"][0]["result"]
      and S1_TEXT in result13["action_trace"][0]["result"]
      and "Call submit_final_answer again" in result13["action_trace"][0]["result"],
      result13["action_trace"][0])
check("2nd entry is the real, confirmed submit", not result13["action_trace"][1].get("provisional_submit"), result13["action_trace"][1])
check("submit_args carries the final answer", result13["submit_args"]["valid"] == "yes")
check("action_count is 0 (neither submit call counts against budget)", result13["action_count"] == 0, str(result13["action_count"]))

# The confirm-turn's contents (sent for turn 2's call) must include the
# recap as a tool-role function_response, not a hardcoded reminder gated on
# mode=="VALIDATED" the way the terminal path's pre-call reminder is.
confirm_turn_contents13 = client13.models.seen_contents[1]
tool_role_texts13 = [
    part.text
    for content in confirm_turn_contents13 if content.role == "tool"
    for part in content.parts
    if getattr(part, "function_response", None) is not None
    for part in [types.Part(text=part.function_response.response.get("result", ""))]
]
check("confirm-turn's contents include a tool-role function_response with the recap",
      any(CODE in t and S1_TEXT in t for t in tool_role_texts13), tool_role_texts13)

for p in (episode_dir(TITLE13, RUN13), snapshot_root(TITLE13, RUN13)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 14: after an intercepted voluntary submission, the model may
# choose to investigate further before actually resubmitting -- confirm this
# is allowed (available tools on the confirm-turn aren't force-narrowed) and
# the eventual second submit_final_answer call is still accepted. ──────────

print("\n── after interception, model may investigate further before resubmitting ──")

TITLE14, RUN14 = "_test_loop_scenario14", "unittest"
for p in (episode_dir(TITLE14, RUN14), snapshot_root(TITLE14, RUN14)):
    if p.exists():
        shutil.rmtree(p)

submit_call14 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                    pde_exp="revised", method_exp="revised", behavior_exp="revised", valid_exp="revised after double-checking")
scripted14 = [
    [submit_call14],                              # voluntary -- intercepted
    [fc("run_diagnostic", script="print('double-checking')\n")],  # chooses to investigate more instead of resubmitting immediately
    [submit_call14],                              # now resubmits -- accepted as final
]
client14 = ConfigCapturingClient(scripted14)

result14 = run_agentic_stage2(
    client14, "gemini-2.5-flash", TITLE14, RUN14, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
)

check("client saw 3 turns", client14.models.n_calls == 3, str(client14.models.n_calls))
check("confirm-turn's tools are NOT force-narrowed to just submit_final_answer",
      set(client14.models.seen_configs[1].tool_config.function_calling_config.allowed_function_names)
      == {"edit_source", "run_diagnostic", "submit_final_answer"},
      client14.models.seen_configs[1].tool_config.function_calling_config.allowed_function_names)
check("the run_diagnostic call in between is dispatched normally (not rejected)",
      result14["action_trace"][1]["tool"] == "run_diagnostic"
      and "double-checking" in result14["action_trace"][1]["result"],
      str(result14["action_trace"][1]))
check("action_count is 1 (only run_diagnostic counts; neither submit call does)", result14["action_count"] == 1, str(result14["action_count"]))
check("the eventual second submit_final_answer call is accepted as final",
      result14["submit_args"]["valid_exp"] == "revised after double-checking", str(result14["submit_args"]))

for p in (episode_dir(TITLE14, RUN14), snapshot_root(TITLE14, RUN14)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 15: a single-candidate oversized write is quarantined but does
# NOT abort the episode -- confirms continuity through the full loop (not
# just at the agentic_sandbox unit level): the turn still counts, its
# snapshot still gets taken, and the model can go on to submit normally on
# a later turn. Uses the same tiny threshold overrides as scenario 16 below
# so the test doesn't need to allocate real GB-scale files. ────────────────

print("\n── single-candidate oversized write: quarantined, episode continues normally ──")

TITLE15, RUN15 = "_test_loop_scenario15", "unittest"
for p in (episode_dir(TITLE15, RUN15), snapshot_root(TITLE15, RUN15)):
    if p.exists():
        shutil.rmtree(p)

oversized_write_source = "open('toolarge.bin', 'wb').write(b'x' * 500_000)\n"
submit_call15 = fc("submit_final_answer", pde="heat", method="explicit", behavior="diffusion", valid="yes",
                    pde_exp="unchanged", method_exp="unchanged", behavior_exp="unchanged", valid_exp="unaffected by the quarantine event")
scripted15 = [
    [fc("edit_source", full_source=oversized_write_source)],
    [submit_call15],  # voluntary -- intercepted
    [submit_call15],  # confirmed -- accepted as final
]
client15 = FakeClient(scripted15)

result15 = run_agentic_stage2(
    client15, "gemini-2.5-flash", TITLE15, RUN15, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    max_file_size_bytes=100_000, max_episode_dir_bytes=10_000_000,
)

check("client saw all 3 turns (episode was not aborted)", client15.models.n_calls == 3, str(client15.models.n_calls))
check("aborted is False", result15["aborted"] is False, str(result15))
check("abort_reason is None", result15["abort_reason"] is None, str(result15["abort_reason"]))
check("submit_args carries the final answer", result15["submit_args"]["valid"] == "yes")
check("first action's result mentions the write cap/quarantine", "write cap" in result15["action_trace"][0]["result"], result15["action_trace"][0]["result"])
check("episode_dir has a turn1 snapshot (episode continued normally, not aborted)",
      (snapshot_root(TITLE15, RUN15) / "turn1").exists())
check("the oversized file was actually quarantined (moved out of the work dir)",
      not (episode_dir(TITLE15, RUN15) / "toolarge.bin").exists())

for p in (episode_dir(TITLE15, RUN15), snapshot_root(TITLE15, RUN15)):
    if p.exists():
        shutil.rmtree(p)


# ── Scenario 16: ambiguous disk-safety abort (multiple oversized candidates)
# also force-ends the episode, with abort_reason == "ambiguous_oversized_write". ─

print("\n── ambiguous disk-safety abort (ambiguous_oversized_write) ──")

TITLE16, RUN16 = "_test_loop_scenario16", "unittest"
for p in (episode_dir(TITLE16, RUN16), snapshot_root(TITLE16, RUN16)):
    if p.exists():
        shutil.rmtree(p)

# Pre-seed two large files directly in the episode dir before the loop starts
# (run_agentic_stage2 calls setup_episode() internally, which mkdir(exist_ok=True)s
# and only writes solver_v0.py -- it won't disturb these), so the single
# oversized write this turn produces becomes ambiguous (3 candidates >= the
# size signature) -- same real mechanism as test_agentic_sandbox.py's
# multi-match test, exercised through the full loop.
_work16 = episode_dir(TITLE16, RUN16)
_work16.mkdir(parents=True, exist_ok=True)
(_work16 / "existing_large_1.bin").write_bytes(b"a" * 95_000)
(_work16 / "existing_large_2.bin").write_bytes(b"b" * 95_000)

scripted16 = [
    [fc("edit_source", full_source=oversized_write_source)],
]
client16 = FakeClient(scripted16)

result16 = run_agentic_stage2(
    client16, "gemini-2.5-flash", TITLE16, RUN16, CODE, PROMPT_S1, S1_TEXT,
    budget=6, truncate_chars=4000, subprocess_timeout=10, episode_cost_cap_usd=0.50,
    max_file_size_bytes=100_000, max_episode_dir_bytes=10_000_000,
)

check("ambiguous abort: aborted is True", result16["aborted"] is True, str(result16))
check("ambiguous abort: abort_reason is ambiguous_oversized_write", result16["abort_reason"] == "ambiguous_oversized_write", str(result16["abort_reason"]))
check("ambiguous abort: pre-existing files were not moved", (_work16 / "existing_large_1.bin").exists() and (_work16 / "existing_large_2.bin").exists())

for p in (episode_dir(TITLE16, RUN16), snapshot_root(TITLE16, RUN16)):
    if p.exists():
        shutil.rmtree(p)


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
