"""
Unit tests for agentic_prompts.py — runs locally, no network. Validates the
three tool schemas actually construct real google.genai FunctionDeclaration
objects (catches shape drift early) and checks the Stage-2 prompt text.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from frontier.agentic_prompts import (
    ALL_TOOL_DECLS,
    EDIT_SOURCE_DECL,
    RUN_DIAGNOSTIC_DECL,
    SUBMIT_FINAL_ANSWER_DECL,
    build_stage2_prompt,
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
check("submit_final_answer requires 4 answer + 4 explanation fields",
      req == {"pde", "method", "behavior", "valid", "pde_exp", "method_exp", "behavior_exp", "valid_exp"},
      str(req))

check("edit_source requires diff",       EDIT_SOURCE_DECL["parameters_json_schema"]["required"] == ["diff"])
check("run_diagnostic requires script",  RUN_DIAGNOSTIC_DECL["parameters_json_schema"]["required"] == ["script"])


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


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
