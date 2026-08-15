"""
Unit tests for agentic_sandbox.py — runs locally, no GPU, no model, no network.
Uses a throwaway title/run_id under results/frontier/_agentic_work/ and cleans
up after itself.
"""
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from frontier.agentic_sandbox import (
    episode_dir,
    snapshot_root,
    setup_episode,
    run_python_file,
    snapshot_turn,
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


TITLE = "_test_Heat_Comm_Valid_1"
RUN_ID = "unittest_run"

# Clean slate
for p in (episode_dir(TITLE, RUN_ID), snapshot_root(TITLE, RUN_ID)):
    if p.exists():
        shutil.rmtree(p)

try:
    # ── setup_episode ────────────────────────────────────────────────────────
    print("\n── setup_episode ──")

    work = setup_episode("x = 1\nprint(x)\n", TITLE, RUN_ID)
    check("returns the episode dir", work == episode_dir(TITLE, RUN_ID))
    check("episode dir exists", work.is_dir())
    check("solver_v0.py written", (work / "solver_v0.py").read_text() == "x = 1\nprint(x)\n")

    # ── run_python_file ──────────────────────────────────────────────────────
    print("\n── run_python_file ──")

    stdout, stderr, timed_out = run_python_file("solver_v0.py", work, timeout=10)
    check("captures stdout",      stdout.strip() == "1", repr(stdout))
    check("no stderr on success", stderr == "", repr(stderr))
    check("not timed out",        timed_out is False)

    (work / "broken.py").write_text("raise ValueError('boom')\n")
    stdout, stderr, timed_out = run_python_file("broken.py", work, timeout=10)
    check("script error surfaced in stderr, not raised", "ValueError" in stderr)
    check("not timed out on a script error", timed_out is False)

    (work / "slow.py").write_text("import time\ntime.sleep(5)\n")
    stdout, stderr, timed_out = run_python_file("slow.py", work, timeout=1)
    check("timeout surfaced, not raised", timed_out is True)

    # Regression test for a real crash found live: exc.stdout/exc.stderr on
    # subprocess.TimeoutExpired can be raw bytes even with text=True passed
    # to subprocess.run (CPython only text-decodes on communicate()'s normal
    # return path, not the exception path) -- the old code's
    # `(exc.stderr or "") + f"..."` raised TypeError: can't concat str to
    # bytes whenever the subprocess had already written non-empty stderr
    # before the timeout hit. A timeout with NO stderr output (the "slow.py"
    # case above) doesn't exercise this: empty bytes is falsy, so `or ""`
    # silently substituted "" without ever reaching the concatenation.
    (work / "slow_stderr.py").write_text(
        "import sys, time\n"
        "print('warning before hang', file=sys.stderr)\n"
        "sys.stderr.flush()\n"
        "time.sleep(5)\n"
    )
    stdout, stderr, timed_out = run_python_file("slow_stderr.py", work, timeout=1)
    check("timeout with real stderr output doesn't raise", timed_out is True)
    check("captured stderr text is a decoded str, not bytes", isinstance(stderr, str), repr(stderr))
    check("captured stderr includes the pre-timeout output", "warning before hang" in stderr, repr(stderr))
    check("timeout note appended to stderr", f"[TIMEOUT after 1s]" in stderr, repr(stderr))

    # ── snapshot_turn ─────────────────────────────────────────────────────────
    print("\n── snapshot_turn ──")

    snapshot_turn(TITLE, RUN_ID, 0)
    snap0 = snapshot_root(TITLE, RUN_ID) / "turn0"
    check("turn0 snapshot created",         snap0.is_dir())
    check("turn0 snapshot has solver_v0",   (snap0 / "solver_v0.py").exists())
    check("snapshot dir isolated from episode dir", snapshot_root(TITLE, RUN_ID) != episode_dir(TITLE, RUN_ID))

    (work / "solver_v1.py").write_text("y = 2\n")
    snapshot_turn(TITLE, RUN_ID, 1)
    snap1 = snapshot_root(TITLE, RUN_ID) / "turn1"
    check("turn1 snapshot has both versions", (snap1 / "solver_v0.py").exists() and (snap1 / "solver_v1.py").exists())
    check("turn0 snapshot untouched by turn1", not (snap0 / "solver_v1.py").exists())

finally:
    # Cleanup regardless of pass/fail
    for p in (episode_dir(TITLE, RUN_ID), snapshot_root(TITLE, RUN_ID)):
        if p.exists():
            shutil.rmtree(p)


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
