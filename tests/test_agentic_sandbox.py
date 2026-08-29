"""
Unit tests for agentic_sandbox.py — runs locally, no GPU, no model, no network.
Uses throwaway title/run_id dirs under results/frontier/_agentic_work/ and
cleans up after itself.
"""
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freegen_static_judgments'))

from frontier.agentic_sandbox import (
    episode_dir,
    snapshot_root,
    quarantine_root,
    setup_episode,
    run_python_file,
    snapshot_turn,
    episode_dir_total_bytes,
    _find_oversized_candidates,
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

    stdout, stderr, timed_out, abort_reason = run_python_file("solver_v0.py", work, 10, TITLE, RUN_ID)
    check("captures stdout",      stdout.strip() == "1", repr(stdout))
    check("no stderr on success", stderr == "", repr(stderr))
    check("not timed out",        timed_out is False)
    check("no abort",             abort_reason is None)

    (work / "broken.py").write_text("raise ValueError('boom')\n")
    stdout, stderr, timed_out, abort_reason = run_python_file("broken.py", work, 10, TITLE, RUN_ID)
    check("script error surfaced in stderr, not raised", "ValueError" in stderr)
    check("not timed out on a script error", timed_out is False)
    check("no abort on a script error", abort_reason is None)

    (work / "slow.py").write_text("import time\ntime.sleep(5)\n")
    stdout, stderr, timed_out, abort_reason = run_python_file("slow.py", work, 1, TITLE, RUN_ID)
    check("timeout surfaced, not raised", timed_out is True)
    check("no abort on a timeout", abort_reason is None)

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
    stdout, stderr, timed_out, abort_reason = run_python_file("slow_stderr.py", work, 1, TITLE, RUN_ID)
    check("timeout with real stderr output doesn't raise", timed_out is True)
    check("captured stderr text is a decoded str, not bytes", isinstance(stderr, str), repr(stderr))
    check("captured stderr includes the pre-timeout output", "warning before hang" in stderr, repr(stderr))
    check("timeout note appended to stderr", f"[TIMEOUT after 1s]" in stderr, repr(stderr))

    # Regression test for a real crash found live during the 256-row sweep:
    # a normal (non-timeout) completion whose stdout/stderr contains a byte
    # that isn't valid UTF-8 (e.g. a stray locale-encoded char in a library
    # warning) raised UnicodeDecodeError out of subprocess.run's own
    # text-decoding, uncaught, killing the whole sweep process. Not a
    # timeout, so the TimeoutExpired branch's manual decode never ran.
    (work / "bad_bytes.py").write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'before \\xa0 after\\n')\n"
        "sys.stdout.buffer.flush()\n"
    )
    stdout, stderr, timed_out, abort_reason = run_python_file("bad_bytes.py", work, 10, TITLE, RUN_ID)
    check("non-UTF8 byte in stdout doesn't raise", True)  # reaching this line at all is the assertion
    check("stdout is still a decoded str", isinstance(stdout, str), repr(stdout))
    check("surrounding valid text preserved", "before" in stdout and "after" in stdout, repr(stdout))
    check("not timed out", timed_out is False)
    check("no abort", abort_reason is None)

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
    for p in (episode_dir(TITLE, RUN_ID), snapshot_root(TITLE, RUN_ID)):
        if p.exists():
            shutil.rmtree(p)


# ── Disk-safety guard: per-file write cap (RLIMIT_FSIZE), single match ──────

print("\n── write-cap trip, single match -> quarantine, episode continues ──")

TITLE_A, RUN_A = "_test_diskguard_singlematch", "unittest"
for p in (episode_dir(TITLE_A, RUN_A), snapshot_root(TITLE_A, RUN_A)):
    if p.exists():
        shutil.rmtree(p)

try:
    work_a = setup_episode("pass\n", TITLE_A, RUN_A)
    (work_a / "bigwrite.py").write_text(
        "open('big.bin', 'wb').write(b'x' * 500_000)\n"
    )
    stdout, stderr, timed_out, abort_reason = run_python_file(
        "bigwrite.py", work_a, 10, TITLE_A, RUN_A,
        max_file_size_bytes=100_000, max_episode_dir_bytes=10_000_000,
    )
    check("write-cap trip: not a timeout", timed_out is False)
    check("write-cap trip: no abort (exactly one candidate)", abort_reason is None, str(abort_reason))
    check("write-cap trip: stderr mentions the write cap", "1GB write cap" in stderr or "write cap" in stderr, repr(stderr))
    check("write-cap trip: oversized file removed from work dir", not (work_a / "big.bin").exists())
    qroot = quarantine_root(TITLE_A, RUN_A)
    check("write-cap trip: oversized file moved to quarantine", (qroot / "big.bin").exists())
    check("write-cap trip: quarantined file size matches the cap", (qroot / "big.bin").stat().st_size <= 100_000)
finally:
    for p in (episode_dir(TITLE_A, RUN_A), snapshot_root(TITLE_A, RUN_A)):
        if p.exists():
            shutil.rmtree(p)


# ── Disk-safety guard: per-file write cap, no false trip ────────────────────

print("\n── write-cap, no false trip ──")

TITLE_B, RUN_B = "_test_diskguard_nofalsetrip", "unittest"
for p in (episode_dir(TITLE_B, RUN_B), snapshot_root(TITLE_B, RUN_B)):
    if p.exists():
        shutil.rmtree(p)

try:
    work_b = setup_episode("pass\n", TITLE_B, RUN_B)
    (work_b / "smallwrite.py").write_text(
        "open('small.bin', 'wb').write(b'x' * 50_000)\n"
    )
    stdout, stderr, timed_out, abort_reason = run_python_file(
        "smallwrite.py", work_b, 10, TITLE_B, RUN_B,
        max_file_size_bytes=100_000, max_episode_dir_bytes=10_000_000,
    )
    check("no false trip: not timed out", timed_out is False)
    check("no false trip: no abort", abort_reason is None)
    check("no false trip: stderr empty", stderr == "", repr(stderr))
    check("no false trip: file written normally", (work_b / "small.bin").stat().st_size == 50_000)
    check("no false trip: no quarantine dir created", not quarantine_root(TITLE_B, RUN_B).exists())
finally:
    for p in (episode_dir(TITLE_B, RUN_B), snapshot_root(TITLE_B, RUN_B)):
        if p.exists():
            shutil.rmtree(p)


# ── Disk-safety guard: ambiguous kill, zero matches (helper tested directly) ─

print("\n── write-cap, zero-match ambiguous case (_find_oversized_candidates in isolation) ──")

TITLE_C, RUN_C = "_test_diskguard_zeromatch", "unittest"
for p in (episode_dir(TITLE_C, RUN_C),):
    if p.exists():
        shutil.rmtree(p)

try:
    work_c = setup_episode("pass\n", TITLE_C, RUN_C)
    (work_c / "small1.bin").write_bytes(b"x" * 100)
    (work_c / "small2.bin").write_bytes(b"x" * 200)
    candidates = _find_oversized_candidates(work_c, max_file_size_bytes=100_000)
    check("zero-match: no files qualify as candidates", candidates == [], str(candidates))
finally:
    if episode_dir(TITLE_C, RUN_C).exists():
        shutil.rmtree(episode_dir(TITLE_C, RUN_C))


# ── Disk-safety guard: ambiguous kill, multiple matches (real kill, pre-existing large files) ─

print("\n── write-cap, multiple-match ambiguous case -> force-abort, nothing moved ──")

TITLE_D, RUN_D = "_test_diskguard_multimatch", "unittest"
for p in (episode_dir(TITLE_D, RUN_D), snapshot_root(TITLE_D, RUN_D)):
    if p.exists():
        shutil.rmtree(p)

try:
    work_d = setup_episode("pass\n", TITLE_D, RUN_D)
    # Two pre-existing large files (not written by the subprocess about to
    # run) both cross the candidate-size signature (>= 0.9 * 100_000 = 90_000).
    (work_d / "existing_large_1.bin").write_bytes(b"a" * 95_000)
    (work_d / "existing_large_2.bin").write_bytes(b"b" * 95_000)
    (work_d / "bigwrite.py").write_text(
        "open('big.bin', 'wb').write(b'x' * 500_000)\n"
    )
    stdout, stderr, timed_out, abort_reason = run_python_file(
        "bigwrite.py", work_d, 10, TITLE_D, RUN_D,
        max_file_size_bytes=100_000, max_episode_dir_bytes=10_000_000,
    )
    check("multi-match: abort_reason is ambiguous_oversized_write", abort_reason == "ambiguous_oversized_write", str(abort_reason))
    check("multi-match: stderr explains the ambiguity", "ambiguous" in stderr or "matched the expected size" in stderr, repr(stderr))
    check("multi-match: pre-existing file 1 untouched (not moved)", (work_d / "existing_large_1.bin").exists())
    check("multi-match: pre-existing file 2 untouched (not moved)", (work_d / "existing_large_2.bin").exists())
    check("multi-match: no quarantine dir created (nothing moved)", not quarantine_root(TITLE_D, RUN_D).exists())
finally:
    for p in (episode_dir(TITLE_D, RUN_D), snapshot_root(TITLE_D, RUN_D)):
        if p.exists():
            shutil.rmtree(p)


# ── Disk-safety guard: cumulative episode-dir cap ───────────────────────────

print("\n── cumulative dir-size cap trip (no single file over its own cap) ──")

TITLE_E, RUN_E = "_test_diskguard_cumulative", "unittest"
for p in (episode_dir(TITLE_E, RUN_E), snapshot_root(TITLE_E, RUN_E)):
    if p.exists():
        shutil.rmtree(p)

try:
    work_e = setup_episode("pass\n", TITLE_E, RUN_E)
    (work_e / "twowrites.py").write_text(
        "open('a.bin', 'wb').write(b'x' * 6000)\n"
        "open('b.bin', 'wb').write(b'y' * 6000)\n"
    )
    stdout, stderr, timed_out, abort_reason = run_python_file(
        "twowrites.py", work_e, 10, TITLE_E, RUN_E,
        max_file_size_bytes=1_000_000,  # neither individual file trips this
        max_episode_dir_bytes=10_000,   # but the cumulative total does
    )
    check("cumulative cap: execution itself succeeded (no SIGXFSZ)", timed_out is False)
    check("cumulative cap: abort_reason is cumulative_dir_size_exceeded", abort_reason == "cumulative_dir_size_exceeded", str(abort_reason))
    check("cumulative cap: stderr mentions the cumulative cap", "cumulative" in stderr, repr(stderr))
    check("cumulative cap: both files still present (this guard never moves/deletes)",
          (work_e / "a.bin").exists() and (work_e / "b.bin").exists())
finally:
    for p in (episode_dir(TITLE_E, RUN_E), snapshot_root(TITLE_E, RUN_E)):
        if p.exists():
            shutil.rmtree(p)


# ── episode_dir_total_bytes ──────────────────────────────────────────────────

print("\n── episode_dir_total_bytes ──")

TITLE_F, RUN_F = "_test_diskguard_totalbytes", "unittest"
for p in (episode_dir(TITLE_F, RUN_F),):
    if p.exists():
        shutil.rmtree(p)

try:
    work_f = setup_episode("pass\n", TITLE_F, RUN_F)  # writes solver_v0.py ("pass\n" = 5 bytes)
    (work_f / "extra.bin").write_bytes(b"z" * 1000)
    total = episode_dir_total_bytes(work_f)
    check("total_bytes sums top-level files", total == 5 + 1000, str(total))
finally:
    if episode_dir(TITLE_F, RUN_F).exists():
        shutil.rmtree(episode_dir(TITLE_F, RUN_F))


# ── snapshot dedup ───────────────────────────────────────────────────────────

print("\n── snapshot dedup: unchanged large files are not re-copied ──")

TITLE_G, RUN_G = "_test_diskguard_dedup", "unittest"
for p in (episode_dir(TITLE_G, RUN_G), snapshot_root(TITLE_G, RUN_G)):
    if p.exists():
        shutil.rmtree(p)

try:
    work_g = setup_episode("pass\n", TITLE_G, RUN_G)
    big = work_g / "data.bin"
    big.write_bytes(b"d" * 2500)  # over the small test threshold below, not the real 30MB default

    snapshot_turn(TITLE_G, RUN_G, 0, dedup_threshold_bytes=2000)
    snap0 = snapshot_root(TITLE_G, RUN_G) / "turn0"
    check("dedup: turn0 has a real copy (no previous turn to dedup against)", (snap0 / "data.bin").exists())
    check("dedup: turn0 has no placeholder", not (snap0 / "data.bin.unchanged.txt").exists())

    # Turn 1: file untouched -- should be deduped.
    snapshot_turn(TITLE_G, RUN_G, 1, dedup_threshold_bytes=2000)
    snap1 = snapshot_root(TITLE_G, RUN_G) / "turn1"
    check("dedup: turn1 has NO real copy of the unchanged file", not (snap1 / "data.bin").exists())
    placeholder1 = snap1 / "data.bin.unchanged.txt"
    check("dedup: turn1 has a placeholder instead", placeholder1.exists())
    check("dedup: placeholder references turn0's real copy", "turn0" in placeholder1.read_text())

    # Turn 2: still untouched -- placeholder should point back to turn0 (the
    # real copy), not turn1 (itself a placeholder) -- no indirection chains.
    snapshot_turn(TITLE_G, RUN_G, 2, dedup_threshold_bytes=2000)
    snap2 = snapshot_root(TITLE_G, RUN_G) / "turn2"
    placeholder2 = snap2 / "data.bin.unchanged.txt"
    check("dedup chain: turn2 also deduped", not (snap2 / "data.bin").exists() and placeholder2.exists())
    check("dedup chain: turn2's placeholder resolves to turn0 (nearest real copy), not turn1",
          "turn0" in placeholder2.read_text() and "turn1" not in placeholder2.read_text())

    # Modify the file -- next snapshot should make a real copy again.
    import time
    time.sleep(0.05)
    big.write_bytes(b"e" * 2600)
    snapshot_turn(TITLE_G, RUN_G, 3, dedup_threshold_bytes=2000)
    snap3 = snapshot_root(TITLE_G, RUN_G) / "turn3"
    check("dedup: modified file gets a real copy again (mtime changed -> not deduped)", (snap3 / "data.bin").exists())
    check("dedup: real copy has the new content", snap3.joinpath("data.bin").read_bytes() == b"e" * 2600)
finally:
    for p in (episode_dir(TITLE_G, RUN_G), snapshot_root(TITLE_G, RUN_G)):
        if p.exists():
            shutil.rmtree(p)


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
