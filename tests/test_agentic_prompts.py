"""
Unit tests for agentic_prompts.py — runs locally, no network. Validates the
three tool schemas actually construct real google.genai FunctionDeclaration
objects (catches shape drift early) and checks the Stage-2 prompt text.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freegen_static_judgments'))

from frontier.agentic_prompts import (
    coerce_validity_score,
    ALL_TOOL_DECLS,
    EDIT_SOURCE_DECL,
    EMPTY_RESPONSE_FEEDBACK,
    PROMPT_S1_AGENTIC,
    RUN_DIAGNOSTIC_DECL,
    SUBMIT_FINAL_ANSWER_DECL,
    TEXT_ONLY_FEEDBACK,
    _flow,
    build_stage2_prompt,
    build_submit_confirmation_reminder,
    build_validated_reminder_final,
    build_validated_reminder_investigative,
    parse_s1_explanations,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


# ── Schema shape ──────────────────────────────────────────────────────────

print("\n── tool schema dicts ──")

check("edit_source name",            EDIT_SOURCE_DECL["name"] == "edit_source")
check("run_diagnostic name",         RUN_DIAGNOSTIC_DECL["name"] == "run_diagnostic")
check("submit_final_answer name",    SUBMIT_FINAL_ANSWER_DECL["name"] == "submit_final_answer")
check("ALL_TOOL_DECLS has all 3",    set(ALL_TOOL_DECLS.keys()) == {"edit_source", "run_diagnostic", "submit_final_answer"})

for name, decl in ALL_TOOL_DECLS.items():
    check(f"{name} has description",        isinstance(decl.get("description"), str) and len(decl["description"]) > 0)
    check(f"{name} has parameters_json_schema", "parameters_json_schema" in decl)

req = set(SUBMIT_FINAL_ANSWER_DECL["parameters_json_schema"]["required"])
check("submit_final_answer requires 4 answer + validity_score + 4 explanation fields",
      req == {"pde", "method", "behavior", "valid", "validity_score",
              "pde_exp", "method_exp", "behavior_exp", "valid_exp"},
      str(req))

# ── validity_score: graded 0-10 physical-fidelity score, alongside the binary ──
_score = SUBMIT_FINAL_ANSWER_DECL["parameters_json_schema"]["properties"]["validity_score"]
check("validity_score is declared as an integer", _score["type"] == "integer")
check("validity_score is oriented so HIGH = good (0 = no physics, 10 = exact), not severity",
      "0 means it retains none" in _score["description"]
      and "10 means it reproduces" in _score["description"])
# The score and the binary field now share the word "validity", so this
# disclaimer is load-bearing: without it the score reads as confidence in the
# yes/no answer, which is a different quantity.
check("validity_score is explicitly NOT a confidence score in the binary valid field",
      "not a confidence score in the binary valid answer field" in _score["description"])
# Both stages must describe the score identically, or an S1-vs-S2 score
# comparison is measuring two different questions.
for _frag in ["rating physical validity for the code",
              "0 means it retains none of the intended system's physics",
              "10 means it reproduces the intended system's physics exactly",
              "not a confidence score in the binary valid answer field"]:
    check(f"S1 prompt and S2 schema share wording: {_frag[:44]!r}",
          _frag in _score["description"] and _frag in PROMPT_S1_AGENTIC)
# The anchoring constraint from the design doc: naming defect categories or
# diagnostics would hand the model the vocabulary whose absence drives the
# obfuscated-condition effect this experiment measures.
_leaks = ["boundary condition", "conservation", "sign flip", "exponent",
          "stability", "CFL", "diffusion", "check whether", "for example"]
check("validity_score description leaks no defect types or diagnostics (endpoint anchors only)",
      not [w for w in _leaks if w.lower() in _score["description"].lower()],
      str([w for w in _leaks if w.lower() in _score["description"].lower()]))
check("valid_exp is asked to justify the degree of validity, not just yes/no",
      "validity_score" in SUBMIT_FINAL_ANSWER_DECL["parameters_json_schema"]["properties"]["valid_exp"]["description"])

check("PROMPT_S1_AGENTIC asks for validity_score too",
      "validity_score: ____" in PROMPT_S1_AGENTIC)

# coerce_validity_score is the ONLY real guard: confirmed live that the API
# accepts minimum/maximum in a declaration but does not enforce them.
for raw, want, label in [
    (7, 7, "in-range int"), ("7", 7, "in-range string"), (0, 0, "lower bound"),
    (10, 10, "upper bound"), (11, None, "just above range"), (-1, None, "negative"),
    (250, None, "far out of range"), (None, None, "missing"),
    ("seven", None, "non-numeric"), ("", None, "empty"), (7.5, None, "fractional"),
    (True, None, "bool is not a score"), (7.0, 7, "integral float"),
    ("8 (mostly fine)", 8, "score with trailing prose"),
]:
    check(f"coerce_validity_score: {label} -> {want!r}", coerce_validity_score(raw) == want,
          f"got {coerce_validity_score(raw)!r}")

check("edit_source requires full_source", EDIT_SOURCE_DECL["parameters_json_schema"]["required"] == ["full_source"])
check("run_diagnostic requires script",  RUN_DIAGNOSTIC_DECL["parameters_json_schema"]["required"] == ["script"])
check("submit_final_answer's description anchors 'valid' to the ORIGINAL snippet, not any edited version",
      "ORIGINAL code snippet" in SUBMIT_FINAL_ANSWER_DECL["description"]
      and "not any version you may have edited" in SUBMIT_FINAL_ANSWER_DECL["description"])
check("submit_final_answer's description clarifies self-inflicted edit errors aren't evidence about the original",
      "not evidence about the original snippet's validity" in SUBMIT_FINAL_ANSWER_DECL["description"])
check("submit_final_answer's description gives the fixed-invalid-code example",
      "generated a fixed version that was physically valid, the final answer should report invalid"
      in SUBMIT_FINAL_ANSWER_DECL["description"])


# ── _flow (collapses source-readability line-wrapping into single spaces) ────

print("\n── _flow ──")

check("_flow collapses a single line-wrap into one space",
      _flow("line one\nline two") == "line one line two")
check("_flow collapses multiple consecutive wrapped lines",
      _flow("a\nb\nc") == "a b c")
check("_flow preserves an intentional blank-line paragraph break",
      _flow("para one\nstill one\n\npara two") == "para one still one\n\npara two")
check("_flow is idempotent on already-flowed text",
      _flow("already one line") == "already one line")

check("none of the 3 tool descriptions contain a stray mid-sentence newline",
      all("\n" not in decl["description"] for decl in ALL_TOOL_DECLS.values()),
      {n: repr(d["description"]) for n, d in ALL_TOOL_DECLS.items() if "\n" in d["description"]})
check("build_stage2_prompt's output contains no stray mid-sentence newline",
      "\n" not in build_stage2_prompt(6))


# ── Real google-genai construction (catches API-shape drift) ────────────────

print("\n── google.genai.types.FunctionDeclaration construction ──")

from google.genai import types
for name, decl in ALL_TOOL_DECLS.items():
    fd = types.FunctionDeclaration(**decl)
    check(f"{name} constructs a real FunctionDeclaration", fd.name == name)


# ── build_stage2_prompt ──────────────────────────────────────────────────────

print("\n── build_stage2_prompt ──")

p6 = build_stage2_prompt(6)
check("mentions execution-success-vs-correctness sentence",
      "runs without execution errors but could be logically or physically invalid" in p6)
check("mentions the budget number", "6" in p6)
check("mentions all 3 tool names", all(n in p6 for n in ("edit_source", "run_diagnostic", "submit_final_answer")))

p3 = build_stage2_prompt(3)
check("budget number changes with argument", "3" in p3 and "budget of 6" not in p3)

check("mentions text+action preference", "both text and an action are preferred" in p6)
check("mentions avoiding reasoning in code comments", "Avoid" in p6 and "comments in the code" in p6)
check("mentions text-only answers still cost a turn and force the next one",
      "text-only answer will trigger a subsequent turn" in p6)
check("guidance paragraph's budget number matches argument", f"{6}-turn budget" in p6 and f"{3}-turn budget" in p3)


# ── per-turn reminders and feedback constants ────────────────────────────────

print("\n── per-turn reminders and feedback constants ──")

r_inv = build_validated_reminder_investigative(3, 6)
check("investigative reminder states live budget usage", "used 3 of 6 actions" in r_inv, r_inv)
check("investigative reminder asks to explain then act",
      "Explain what you want to check and why" in r_inv
      and "take an action using one of" in r_inv)

r_inv2 = build_validated_reminder_investigative(0, 6)
check("investigative reminder's budget count changes with actions_used", "used 0 of 6 actions" in r_inv2, r_inv2)

r_final = build_validated_reminder_final(
    "u = odeint(fn, Uinit, t)\nprint(u)\n",
    "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: no",
)
check("final reminder tells the model the budget is exhausted",
      "investigative budget is exhausted" in r_final and "submit_final_answer now" in r_final)
check("final reminder anchors the answer to the ORIGINAL snippet, not an edited version",
      "ORIGINAL code snippet" in r_final and "not any version you" in r_final)
check("final reminder includes the exact original code text",
      "u = odeint(fn, Uinit, t)\nprint(u)\n" in r_final, r_final)
check("final reminder includes the Stage-1 answer text",
      "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: no" in r_final, r_final)
check("final reminder uses neutral confirm-or-revise framing (not implying it must change)",
      "confirm or revise" in r_final, r_final)
check("no accidental missing-space concatenation bugs anywhere in the final reminder",
      "exhausted.You" not in r_final
      and "</original_code>For" not in r_final
      and "</stage1_answer>Provide" not in r_final,
      r_final)

r_confirm = build_submit_confirmation_reminder(
    "u = odeint(fn, Uinit, t)\nprint(u)\n",
    "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: no",
)
check("submit confirmation reminder does NOT claim the budget is exhausted (it's voluntary, not forced)",
      "budget is exhausted" not in r_confirm, r_confirm)
check("submit confirmation reminder anchors the answer to the ORIGINAL snippet, not an edited version",
      "ORIGINAL code snippet" in r_confirm and "not any version you" in r_confirm)
check("submit confirmation reminder includes the exact original code text",
      "u = odeint(fn, Uinit, t)\nprint(u)\n" in r_confirm, r_confirm)
check("submit confirmation reminder includes the Stage-1 answer text",
      "pde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: no" in r_confirm, r_confirm)
check("submit confirmation reminder tells the model to call submit_final_answer again",
      "Call submit_final_answer again" in r_confirm, r_confirm)
check("submit confirmation reminder uses neutral confirm-or-revise framing",
      "confirm or revise" in r_confirm, r_confirm)
check("no accidental missing-space concatenation bugs anywhere in the submit confirmation reminder",
      "collected.<original_code>" not in r_confirm
      and "</original_code>For" not in r_confirm
      and "</stage1_answer>Call" not in r_confirm,
      r_confirm)

check("TEXT_ONLY_FEEDBACK is non-empty and mentions taking an action",
      len(TEXT_ONLY_FEEDBACK) > 0 and "take an" in TEXT_ONLY_FEEDBACK.lower())
check("EMPTY_RESPONSE_FEEDBACK matches the exact specified sentence",
      EMPTY_RESPONSE_FEEDBACK == "No text or action was received and one turn from the budget was used. Try again.")


# ── PROMPT_S1_AGENTIC and parse_s1_explanations ──────────────────────────────

print("\n── PROMPT_S1_AGENTIC and parse_s1_explanations ──")

s1p = PROMPT_S1_AGENTIC.format(code="u = 1")
check("PROMPT_S1_AGENTIC substitutes the code", "u = 1" in s1p)
check("PROMPT_S1_AGENTIC asks for all 4 core fields",
      all(f"{f}: ____" in s1p for f in ("pde", "method", "behavior", "valid")))
check("PROMPT_S1_AGENTIC asks for all 4 explanation fields",
      all(f"{f}: ____" in s1p for f in ("pde_exp", "method_exp", "behavior_exp", "valid_exp")))

sample_s1_response = (
    "pde: heat equation\n"
    "method: explicit\n"
    "behavior: diffusion\n"
    "valid: no\n"
    "pde_exp: looks like the heat equation from the diffusion term.\n"
    "method_exp: forward Euler finite differences.\n"
    "behavior_exp: heat spreads out over time.\n"
    "valid_exp: the boundary conditions look wrong.\n"
)
exps = parse_s1_explanations(sample_s1_response)
check("parse_s1_explanations extracts pde_exp", exps["pde_exp"] == "looks like the heat equation from the diffusion term.", str(exps))
check("parse_s1_explanations extracts method_exp", exps["method_exp"] == "forward Euler finite differences.", str(exps))
check("parse_s1_explanations extracts behavior_exp", exps["behavior_exp"] == "heat spreads out over time.", str(exps))
check("parse_s1_explanations extracts valid_exp", exps["valid_exp"] == "the boundary conditions look wrong.", str(exps))

missing = parse_s1_explanations("pde: heat\nvalid: no")
check("parse_s1_explanations returns None for missing fields",
      all(v is None for v in missing.values()), str(missing))


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
