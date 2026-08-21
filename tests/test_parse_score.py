"""
Unit tests for parse_score.py — runs locally, no GPU, no model.
Tests parser robustness and scorer correctness against known cases.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freegen'))

from parse_score import (parse_response, score_pde, score_multival, score_valid, score_row,
                         classify_valid_confidence, valid_intent, VALID_CONF_CLASSES)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


# ── Parser tests ────────────────────────────────────────────────────────────

print("\n── parse_response ──")

# Clean format
r = parse_response("pde: wave equation\nmethod: explicit\nbehavior: oscillation\nvalid: yes")
check("clean format - pde",      r["pde"]      == "wave equation")
check("clean format - method",   r["method"]   == "explicit")
check("clean format - behavior", r["behavior"] == "oscillation")
check("clean format - valid",    r["valid"]    == "yes")

# Case insensitive fields
r = parse_response("PDE: Heat\nMethod: Implicit Euler\nBehavior: Diffusion\nValid: No")
check("case insensitive fields", r["pde"] == "Heat")
check("mixed case method",       r["method"] == "Implicit Euler")

# Thinking block stripped
r = parse_response("<think>Long reasoning about wave equations...</think>\npde: heat\nmethod: explicit\nbehavior: diffusion\nvalid: yes")
check("think block stripped",    r["pde"] == "heat")

# Partial output (model only outputs some fields)
r = parse_response("pde: burgers equation\nmethod: explicit")
check("partial output - pde",    r["pde"] == "burgers equation")
check("partial output - missing", r["behavior"] is None)

# Trailing underscores stripped
r = parse_response("pde: navier-stokes____\nmethod: spectral___")
check("trailing underscores",    r["pde"] == "navier-stokes")

# Markdown emphasis on the field labels. Qwen3-Coder-30B-A3B-Instruct answers this way
# on every row; before 2026-08-19 all four fields parsed to None for it, which scored a
# complete and correct answer as a refusal. Canary torch:16047172 caught it on 8 rows.
r = parse_response("- **pde:** Burgers' equation  \n- **method:** upwind finite volume, minmod limiter\n"
                   "- **behavior:** shock formation\n- **valid:** No")
check("markdown bold, colon inside - pde",      r["pde"] == "Burgers' equation")
check("markdown bold, colon inside - method",   r["method"] == "upwind finite volume, minmod limiter")
check("markdown bold, colon inside - behavior", r["behavior"] == "shock formation")
check("markdown bold, colon inside - valid",    r["valid"] == "No")

r = parse_response("**pde**: heat equation\n**method**: Crank-Nicolson\n**behavior**: diffusion\n**valid**: Yes")
check("markdown bold, colon outside", r["pde"] == "heat equation" and r["valid"] == "Yes")

r = parse_response("*pde*: wave equation\n`method`: leapfrog\nbehavior: oscillation\nvalid: no")
check("single-asterisk and backtick labels", r["pde"] == "wave equation" and r["method"] == "leapfrog")

r = parse_response("1. pde: wave equation\n2. method: leapfrog\n3. behavior: oscillation\n4. valid: No")
check("numbered list format", r["pde"] == "wave equation" and r["behavior"] == "oscillation")

# The plain formats every earlier model used must be untouched by the above
r = parse_response("- pde: burgers equation\n- method: explicit\n- behavior: advection\n- valid: yes")
check("plain bullet still parses", r["pde"] == "burgers equation" and r["method"] == "explicit")

# Verbose preamble before structured output
r = parse_response("Sure, here is my analysis:\npde: heat equation\nmethod: implicit\nbehavior: diffusion\nvalid: yes")
check("preamble before output",  r["pde"] == "heat equation")


# ── score_pde tests ──────────────────────────────────────────────────────────

print("\n── score_pde ──")

s = score_pde("wave equation", "wave")
check("pde exact alias match",       s["pde_match"] == 1, str(s))

s = score_pde("Navier-Stokes equations", "navier-stokes")
check("pde navier-stokes alias",     s["pde_match"] == 1, str(s))

s = score_pde("incompressible flow", "navier-stokes")
check("pde incompressible alias",    s["pde_match"] == 1, str(s))

s = score_pde("Burgers equation", "heat")
check("pde wrong class = 0",         s["pde_match"] == 0, str(s))

s = score_pde(None, "wave")
check("pde None parsed = 0",         s["pde_match"] == 0, str(s))

s = score_pde("", "wave")
check("pde empty string = 0",        s["pde_match"] == 0, str(s))


# ── score_multival tests (method) ────────────────────────────────────────────

print("\n── score_multival (method) ──")

s = score_multival("explicit Euler finite difference", "explicit", "method")
check("method explicit alias",         s["method_any_match"] == 1, str(s))
check("method single recall=1.0",      s["method_recall"] == 1.0, str(s))

s = score_multival("spectral FFT method", "explicit/spectral", "method")
check("method multi-val partial",      s["method_any_match"] == 1, str(s))
check("method recall=0.5",            s["method_recall"] == 0.5, str(s))

s = score_multival("explicit and spectral", "explicit/spectral", "method")
check("method multi-val full recall",  s["method_recall"] == 1.0, str(s))

s = score_multival("finite volume", "explicit", "method")
check("method no match = 0",           s["method_any_match"] == 0, str(s))

s = score_multival(None, "explicit", "method")
check("method None parsed = 0",        s["method_any_match"] == 0, str(s))


# ── score_multival tests (behavior) ─────────────────────────────────────────

print("\n── score_multival (behavior) ──")

s = score_multival("heat conduction spreading", "diffusion", "behavior")
check("behavior alias heat conduction",  s["behavior_any_match"] == 1, str(s))

s = score_multival("wave propagation and advection", "oscillation/advection", "behavior")
check("behavior multi-val both found",   s["behavior_recall"] == 1.0, str(s))

s = score_multival("advection transport", "oscillation/advection", "behavior")
check("behavior recall=0.5",             s["behavior_recall"] == 0.5, str(s))

s = score_multival("some other process", "diffusion", "behavior")
check("behavior no match = 0",           s["behavior_any_match"] == 0, str(s))


# ── score_valid tests ────────────────────────────────────────────────────────

print("\n── score_valid ──")

check("valid yes=True",     score_valid("yes",   True)["valid_match"]  == 1)
check("valid no=False",     score_valid("no",    False)["valid_match"] == 1)
check("valid true=True",    score_valid("true",  True)["valid_match"]  == 1)
check("valid false=False",  score_valid("false", False)["valid_match"] == 1)
check("valid mismatch",     score_valid("yes",   False)["valid_match"] == 0)
check("valid None",         score_valid(None,    True)["valid_match"]  == 0)
check("valid 'invalid'",    score_valid("invalid", False)["valid_match"] == 1)


# ── score_row integration test ───────────────────────────────────────────────

print("\n── score_row (integration) ──")

fake_row = {
    "pde_class":   "wave",
    "num_method":  "explicit/spectral",
    "phys_process": "oscillation/advection",
    "phys_valid":  True,
}
parsed = {
    "pde":      "wave equation",
    "method":   "explicit Euler and spectral FFT",
    "behavior": "oscillation and transport",
    "valid":    "yes",
}
s = score_row(parsed, fake_row)
check("row pde_match=1",              s["pde_match"] == 1, str(s))
check("row method_any_match=1",       s["method_any_match"] == 1, str(s))
check("row method_recall=1.0",        s["method_recall"] == 1.0, str(s))
check("row behavior_any_match=1",     s["behavior_any_match"] == 1, str(s))
check("row behavior_recall=1.0",      s["behavior_recall"] == 1.0, str(s))
check("row valid_match=1",            s["valid_match"] == 1, str(s))

# Worst-case: all wrong
parsed_bad = {"pde": "something else", "method": "finite volume", "behavior": "unknown", "valid": "no"}
s = score_row(parsed_bad, fake_row)
check("row all-wrong pde=0",          s["pde_match"] == 0, str(s))
check("row all-wrong method=0",       s["method_any_match"] == 0, str(s))
check("row all-wrong valid=0",        s["valid_match"] == 0, str(s))


# ── Hedge / validity-confidence classifier ──────────────────────────────────
# Canonical rule lives in parse_score.py; viz/ and eval/frontier/ both import it.
# The cases marked "strict" are the ones that separate this rule from the older
# viz copies, which had no hedge lexicon and bucketed them into a catch-all.

print("\n── classify_valid_confidence ──")

for raw, want in [
    ("yes",                                        "Confident Yes"),
    ("Valid",                                      "Confident Yes"),
    ("no",                                         "Confident No"),
    ("invalid",                                    "Confident No"),
    ("yes, the discretisation is stable",          "Uncertain Yes"),
    ("no — the CFL condition is violated",         "Confident No"),
    ("not fully valid",                            "Confident No"),
    ("this is not physically valid",               "Confident No"),
    ("physically valid for small dt",              "Uncertain Yes"),
    ("possibly valid",                             "Hedged"),          # strict
    ("cannot determine without running it",        "Hedged"),          # strict
    ("it depends on the boundary conditions",      "Hedged"),          # strict
    ("unclear",                                    "Hedged"),          # strict
    ("potentially valid, but requires context",    "Hedged"),          # strict
    ("",                                           "Hedged"),
    (None,                                         "Hedged"),
]:
    got = classify_valid_confidence(raw)
    check(f"hedge {raw!r} -> {want}", got == want, f"got {got!r}")

check("every label is a declared class",
      all(classify_valid_confidence(x) in VALID_CONF_CLASSES
          for x in ["yes", "no", "possibly", "", None, "banana"]))


# ── valid_intent ────────────────────────────────────────────────────────────
# score_valid abstains on a hedge (scores 0), so hedge-conditioned accuracy
# computed from valid_match would be 0 by construction. valid_intent reads the
# directional lean instead, including inside the Hedged bucket.

print("\n── valid_intent ──")

for raw, want in [
    ("yes",                                     True),
    ("no",                                      False),
    ("not physically valid",                    False),
    ("possibly valid",                          True),   # hedged, but leans yes
    ("potentially valid, but requires context", True),   # hedged, but leans yes
    ("might be incorrect near the boundary",    False),  # hedged, but leans no
    ("",                                        None),
    (None,                                      None),
]:
    got = valid_intent(raw)
    check(f"intent {raw!r} -> {want}", got is want, f"got {got!r}")

# pandas hands these functions a float NaN whenever the field was missing in a
# CSV round-trip. NaN is truthy, so `raw or ""` silently crashes on it.
_nan = float("nan")
check("valid_intent survives a pandas NaN", valid_intent(_nan) is None)
check("valid_intent survives a non-string", valid_intent(3.0) is None)
check("classify survives a pandas NaN", classify_valid_confidence(_nan) == "Hedged")

check("hedged rows are not all abstains",
      valid_intent("possibly valid") is not None
      and score_valid("possibly valid", True)["valid_match"] == 0,
      "a hedge must score 0 yet still carry a readable lean")


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print(f"All tests passed.")
