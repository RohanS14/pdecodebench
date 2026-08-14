"""
Unit tests for agentic_tools.py — runs locally, no GPU, no model, no network.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from frontier.agentic_tools import (
    apply_unified_diff,
    next_version_filename,
    tools_available,
    truncate,
    EpisodeCostGuard,
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


# ── apply_unified_diff ───────────────────────────────────────────────────────

print("\n── apply_unified_diff ──")

base = "a = 1\nb = 2\nc = 3\n"
good_diff = (
    "--- a/solver.py\n"
    "+++ b/solver.py\n"
    "@@ -1,3 +1,3 @@\n"
    " a = 1\n"
    "-b = 2\n"
    "+b = 99\n"
    " c = 3\n"
)
new_code, err = apply_unified_diff(base, good_diff)
check("valid diff applies", err is None, str(err))
check("valid diff produces expected content", new_code == "a = 1\nb = 99\nc = 3\n", repr(new_code))

new_code, err = apply_unified_diff(base, "")
check("empty diff is a no-op success", err is None and new_code == base)

new_code, err = apply_unified_diff(base, "   \n  ")
check("whitespace-only diff is a no-op success", err is None and new_code == base)

bad_diff = (
    "--- a/solver.py\n"
    "+++ b/solver.py\n"
    "@@ -1,3 +1,3 @@\n"
    " a = 1\n"
    "-b = 999999\n"
    "+b = 99\n"
    " c = 3\n"
)
new_code, err = apply_unified_diff(base, bad_diff)
check("diff with wrong context fails, does not raise", new_code is None and isinstance(err, str) and len(err) > 0)

# Regression: a diff whose "+" line already matches the base (but whose "-" line
# doesn't) looks "reversed" to `patch`'s heuristic -- must fail forward, not
# silently apply backwards.
reversible_looking_diff = (
    "--- a/solver.py\n"
    "+++ b/solver.py\n"
    "@@ -1,3 +1,3 @@\n"
    " a = 1\n"
    "-b = 999\n"
    "+b = 2\n"
    " c = 3\n"
)
new_code, err = apply_unified_diff(base, reversible_looking_diff)
check("reversed-looking diff fails forward, is not silently applied backwards",
      new_code is None and isinstance(err, str) and len(err) > 0, repr((new_code, err)))


# ── next_version_filename ────────────────────────────────────────────────────

print("\n── next_version_filename ──")

check("first version after v0",     next_version_filename(["solver_v0.py"]) == "solver_v1.py")
check("next after v0..v3",          next_version_filename(["solver_v0.py", "solver_v1.py", "solver_v2.py", "solver_v3.py"]) == "solver_v4.py")
check("ignores non-matching files", next_version_filename(["solver_v0.py", "diagnostic_0.py", "notes.txt"]) == "solver_v1.py")
check("out-of-order input handled", next_version_filename(["solver_v2.py", "solver_v0.py", "solver_v1.py"]) == "solver_v3.py")
check("empty list starts at v0",    next_version_filename([]) == "solver_v0.py")


# ── tools_available ──────────────────────────────────────────────────────────

print("\n── tools_available ──")

check("budget remaining, no cost trip -> all 3 tools",
      set(tools_available(2, 6, False)) == {"edit_source", "run_diagnostic", "submit_final_answer"})
check("budget exhausted -> submit only",
      tools_available(6, 6, False) == ["submit_final_answer"])
check("budget over (shouldn't happen but defensive) -> submit only",
      tools_available(7, 6, False) == ["submit_final_answer"])
check("cost guard tripped, budget remaining -> submit only",
      tools_available(1, 6, True) == ["submit_final_answer"])


# ── truncate ──────────────────────────────────────────────────────────────

print("\n── truncate ──")

check("under cap unchanged",  truncate("short", 100) == "short")
check("exact cap unchanged",  truncate("12345", 5) == "12345")
check("over cap truncated",   truncate("123456", 5).startswith("12345"))
check("over cap has marker",  "truncated" in truncate("123456", 5))
check("None becomes empty string", truncate(None, 10) == "")


# ── EpisodeCostGuard ──────────────────────────────────────────────────────

print("\n── EpisodeCostGuard ──")

g = EpisodeCostGuard(max_cost_usd=0.10)
check("not tripped initially", g.tripped() is False)
g.add(0.05)
check("not tripped under cap", g.tripped() is False)
g.add(0.05)
check("tripped at exactly cap", g.tripped() is True)

g2 = EpisodeCostGuard(max_cost_usd=0.10)
g2.add(0.5)
check("tripped well over cap", g2.tripped() is True)


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
