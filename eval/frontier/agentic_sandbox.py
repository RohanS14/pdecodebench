"""
agentic_sandbox.py — filesystem + subprocess conventions for the agentic
belief-revision episode loop.

Mirrors the sandboxed-subprocess pattern already used by
extract_trajectories.py's `_work` dir and full_audit_exec.py's `_audit_exec`
dir: one scratch directory, cwd set to it, hard timeout, soft (not
hard-security) isolation. Every subprocess here runs under whatever
interpreter is currently running this script (eval/.venv, which has both
google-genai and the full numpy/scipy/jax stack the solver snippets need).

Disk-safety guards (added after a real ENOSPC incident -- see
notes_trajectory_saving_audit.txt / the disk-safety-hardening plan):
  - MAX_FILE_SIZE_BYTES: a hard per-file write cap enforced via RLIMIT_FSIZE
    (kernel-level, kills the subprocess with SIGXFSZ the instant a single
    file's write would cross the cap -- this is the only mechanism that can
    actually prevent an oversized write rather than just cleaning up after).
  - MAX_EPISODE_DIR_BYTES: a cumulative cap checked after every execution,
    regardless of outcome. RLIMIT_FSIZE is strictly per-file, so a script
    could "comply" by splitting output across several under-cap files --
    this catches that case, but only after the fact (it cannot prevent that
    one turn's writes, only stop further compounding).
  - SNAPSHOT_DEDUP_THRESHOLD_BYTES: unrelated to the two caps above -- a
    waste-reduction measure, not a safety guard. See snapshot_turn().
"""
import shutil
import subprocess
import sys
from pathlib import Path
from resource import RLIMIT_FSIZE, setrlimit

REPO_ROOT = Path(__file__).parent.parent.parent
AGENTIC_WORK_ROOT = REPO_ROOT / "results" / "frontier" / "_agentic_work"

MAX_FILE_SIZE_BYTES = 1_000_000_000          # 1GB per-file hard write cap
MAX_EPISODE_DIR_BYTES = 1_500_000_000        # 1.5GB cumulative episode-dir cap
SNAPSHOT_DEDUP_THRESHOLD_BYTES = 30_000_000  # 30MB "skip if unchanged" floor

# Fraction of MAX_FILE_SIZE_BYTES a file's on-disk size must reach to be
# considered a candidate culprit after a SIGXFSZ kill. Wide margin (0.9x)
# because the kernel kills the process precisely when a write would cross
# the cap, so the truncated file's size should land very close to it.
_CULPRIT_SIZE_FRACTION = 0.9


def episode_dir(title: str, run_id: str) -> Path:
    return AGENTIC_WORK_ROOT / title / run_id


def snapshot_root(title: str, run_id: str) -> Path:
    return AGENTIC_WORK_ROOT / title / f"{run_id}_snapshots"


def quarantine_root(title: str, run_id: str) -> Path:
    """Model-invisible location for files moved (never deleted) off an
    ambiguous/oversized-write kill. Sibling of episode_dir (via
    snapshot_root), NOT a subfolder of it -- episode_dir is the exact cwd
    every subprocess runs with, so anything left in any subfolder underneath
    it is still reachable by a later turn's script."""
    return snapshot_root(title, run_id) / "quarantine"


def setup_episode(code: str, title: str, run_id: str) -> Path:
    """Create the episode's scratch dir and write solver_v0.py (the original
    given code, copied in at episode setup). Returns the dir."""
    work = episode_dir(title, run_id)
    work.mkdir(parents=True, exist_ok=True)
    (work / "solver_v0.py").write_text(code)
    return work


def episode_dir_total_bytes(work: Path) -> int:
    """Sum of top-level file sizes in `work` (non-recursive -- every real
    episode observed writes directly into the episode dir, never
    subdirectories)."""
    total = 0
    for entry in work.iterdir():
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _find_oversized_candidates(work: Path, max_file_size_bytes: int) -> list[Path]:
    """Top-level files in `work` at or above _CULPRIT_SIZE_FRACTION of the
    configured cap -- a hard signature (the kernel kills the process exactly
    when a file crosses the cap), not a heuristic diff."""
    threshold = _CULPRIT_SIZE_FRACTION * max_file_size_bytes
    candidates = []
    for entry in work.iterdir():
        if entry.is_file() and entry.stat().st_size >= threshold:
            candidates.append(entry)
    return candidates


def run_python_file(
    filename: str,
    cwd: Path,
    timeout: int,
    title: str,
    run_id: str,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_episode_dir_bytes: int = MAX_EPISODE_DIR_BYTES,
) -> tuple[str, str, bool, str | None]:
    """Run `python3 <filename>` with cwd set to the episode dir.

    Returns (stdout, stderr, timed_out, abort_reason). Never raises on a
    script error or timeout -- both are surfaced as normal results, matching
    the "add a print/assert, rerun, see the output" debugging pattern the
    tool contract is built around.

    abort_reason is None in the normal case. It is set to a short string
    when the episode must be force-ended for disk safety:
      - "ambiguous_oversized_write": the write cap was hit but the culprit
        file could not be confidently identified (zero or multiple files
        matched the size signature) -- moving the wrong thing (or nothing)
        is worse than aborting, so this never guesses.
      - "cumulative_dir_size_exceeded": no single file crossed the per-file
        cap, but the episode dir's total size did -- RLIMIT_FSIZE is
        strictly per-file, so this catches the "many under-cap files"
        loophole. Checked after every execution regardless of outcome.
    When abort_reason is set, the model-facing explanation is folded into
    stderr (same channel used for [TIMEOUT ...] today); callers must stop
    the episode rather than continue it.

    Detecting the write-cap hit: confirmed live that CPython ignores
    SIGXFSZ by default (`signal.getsignal(signal.SIGXFSZ) == signal.SIG_IGN`)
    -- RLIMIT_FSIZE still truncates the write at the cap (confirmed exact:
    a 500_000-byte write with a 100_000-byte cap left a 100_000-byte file
    on disk), but the process is never killed by the signal; the write()
    call instead raises a normal, catchable `OSError: [Errno 27] File too
    large`, surfacing as an ordinary nonzero exit (or a caught exception and
    exit 0, if the script's own code handles it). So detection can't key off
    the exit code/signal at all -- the oversized-candidate scan below runs
    unconditionally after every execution (success, script error, or
    timeout alike), since it only ever finds something when a file actually
    landed at/near the cap.
    """
    def _limit_file_size():
        setrlimit(RLIMIT_FSIZE, (max_file_size_bytes, max_file_size_bytes))

    abort_reason = None
    try:
        proc = subprocess.run(
            [sys.executable, filename],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",  # a non-UTF8 byte in child stdout/stderr (e.g. a stray
            # locale-encoded char in a warning) must not crash the whole episode
            timeout=timeout,
            preexec_fn=_limit_file_size,
        )
        stdout, stderr, timed_out = proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        # exc.stdout/exc.stderr can be raw bytes even though text=True was
        # passed to subprocess.run above -- CPython's text-decoding happens
        # in Popen.communicate()'s normal return path, but TimeoutExpired is
        # raised from inside communicate() before that decode step runs, so
        # whatever partial output had already been captured is undecoded
        # bytes regardless of text=True. Decode explicitly before using it
        # as a str, or concatenating it with one raises TypeError.
        def _decode(chunk):
            if chunk is None:
                return ""
            if isinstance(chunk, bytes):
                return chunk.decode("utf-8", errors="replace")
            return chunk

        stdout = _decode(exc.stdout)
        stderr = _decode(exc.stderr) + f"\n[TIMEOUT after {timeout}s]"
        timed_out = True

    # Oversized-write check runs unconditionally (success, script error, or
    # timeout alike) -- see the docstring for why this can't be keyed off
    # the exit code/signal. A candidate only ever turns up here if a file
    # genuinely landed at/near the cap, so this is a no-op in the ordinary
    # case (the vast majority of turns).
    candidates = _find_oversized_candidates(cwd, max_file_size_bytes)
    if len(candidates) == 1:
        culprit = candidates[0]
        qroot = quarantine_root(title, run_id)
        qroot.mkdir(parents=True, exist_ok=True)
        dest = qroot / culprit.name
        shutil.move(str(culprit), str(dest))
        stderr += (
            f"\n[terminated: wrote a file >= {max_file_size_bytes} bytes "
            f"(1GB write cap); file moved to "
            f"{dest.relative_to(REPO_ROOT)}]"
        )
    elif len(candidates) > 1:
        stderr += (
            f"\n[terminated: wrote a file >= {max_file_size_bytes} bytes "
            f"(1GB write cap); {len(candidates)} files matched the expected "
            f"size, ambiguous -- episode aborted for manual inspection, "
            f"nothing was moved]"
        )
        abort_reason = "ambiguous_oversized_write"

    # Cumulative check runs regardless of outcome above -- RLIMIT_FSIZE only
    # bounds a single file, so this is the only guard against "many separate
    # under-cap files" ever reaching this point. Detective, not preventive:
    # whatever this execution wrote is already on disk by the time we check;
    # this only stops it compounding further across the rest of the episode.
    if abort_reason is None and episode_dir_total_bytes(cwd) >= max_episode_dir_bytes:
        stderr += (
            f"\n[terminated: episode working directory exceeded "
            f"{max_episode_dir_bytes} bytes cumulative; episode aborted for "
            f"manual inspection]"
        )
        abort_reason = "cumulative_dir_size_exceeded"

    return stdout, stderr, timed_out, abort_reason


def snapshot_turn(
    title: str,
    run_id: str,
    turn_idx: int,
    dedup_threshold_bytes: int = SNAPSHOT_DEDUP_THRESHOLD_BYTES,
) -> None:
    """Copy the full current episode dir into an isolated, model-invisible
    per-turn snapshot: results/frontier/_agentic_work/<title>/<run_id>_snapshots/turn<turn_idx>/.
    Re-running the same turn index overwrites that turn's snapshot only.

    Files >= dedup_threshold_bytes that are byte-identical (same size AND
    mtime) to their nearest previous *real* (non-placeholder) copy are not
    re-copied -- a placeholder <name>.unchanged.txt is written instead,
    pointing at the real copy. This targets duplicated data files (.npz
    etc.); small .py scripts never cross the threshold so are always copied
    normally, no extension-based special-casing needed. Turn 0 never dedups
    (no previous turn exists).
    """
    src = episode_dir(title, run_id)
    dst = snapshot_root(title, run_id) / f"turn{turn_idx}"
    if dst.exists():
        shutil.rmtree(dst)

    skip: dict[str, Path] = {}  # filename -> resolved real-copy dir
    if turn_idx > 0:
        for entry in src.iterdir():
            if not entry.is_file() or entry.stat().st_size < dedup_threshold_bytes:
                continue
            real_copy_dir = _find_nearest_real_copy(title, run_id, entry.name, turn_idx)
            if real_copy_dir is None:
                continue
            prev_file = real_copy_dir / entry.name
            st = entry.stat()
            prev_st = prev_file.stat()
            if st.st_size == prev_st.st_size and st.st_mtime == prev_st.st_mtime:
                skip[entry.name] = real_copy_dir

    def _ignore(dirpath, names):
        if Path(dirpath) == src:
            return set(skip.keys())
        return set()

    shutil.copytree(src, dst, ignore=_ignore)

    for name, real_copy_dir in skip.items():
        placeholder = dst / f"{name}.unchanged.txt"
        entry_stat = (src / name).stat()
        real_rel = (real_copy_dir / name).relative_to(REPO_ROOT)
        placeholder.write_text(
            f"unchanged since prior turn -- not re-copied to save space.\n"
            f"size: {entry_stat.st_size} bytes\n"
            f"mtime: {entry_stat.st_mtime}\n"
            f"real copy: {real_rel}\n"
        )


def _find_nearest_real_copy(title: str, run_id: str, filename: str, before_turn_idx: int) -> Path | None:
    """Walk backward from before_turn_idx - 1 to 0, returning the snapshot
    turn directory holding the nearest *real* (non-placeholder) copy of
    filename, or None if none exists. Placeholder chains always resolve to
    a real copy this way, never to another placeholder -- a human tracing
    the episode never has to follow more than one hop."""
    root = snapshot_root(title, run_id)
    for k in range(before_turn_idx - 1, -1, -1):
        candidate = root / f"turn{k}" / filename
        if candidate.exists() and candidate.is_file():
            return root / f"turn{k}"
    return None
