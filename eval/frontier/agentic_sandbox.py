"""
agentic_sandbox.py — filesystem + subprocess conventions for the agentic
belief-revision episode loop.

Mirrors the sandboxed-subprocess pattern already used by
extract_trajectories.py's `_work` dir and full_audit_exec.py's `_audit_exec`
dir: one scratch directory, cwd set to it, hard timeout, soft (not
hard-security) isolation. Every subprocess here runs under whatever
interpreter is currently running this script (eval/.venv, which has both
google-genai and the full numpy/scipy/jax stack the solver snippets need).
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
AGENTIC_WORK_ROOT = REPO_ROOT / "results" / "frontier" / "_agentic_work"


def episode_dir(title: str, run_id: str) -> Path:
    return AGENTIC_WORK_ROOT / title / run_id


def snapshot_root(title: str, run_id: str) -> Path:
    return AGENTIC_WORK_ROOT / title / f"{run_id}_snapshots"


def setup_episode(code: str, title: str, run_id: str) -> Path:
    """Create the episode's scratch dir and write solver_v0.py (the original
    given code, copied in at episode setup). Returns the dir."""
    work = episode_dir(title, run_id)
    work.mkdir(parents=True, exist_ok=True)
    (work / "solver_v0.py").write_text(code)
    return work


def run_python_file(filename: str, cwd: Path, timeout: int) -> tuple[str, str, bool]:
    """Run `python3 <filename>` with cwd set to the episode dir.

    Returns (stdout, stderr, timed_out). Never raises on a script error or
    timeout -- both are surfaced as normal results, matching the "add a
    print/assert, rerun, see the output" debugging pattern the tool contract
    is built around.
    """
    try:
        proc = subprocess.run(
            [sys.executable, filename],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n[TIMEOUT after {timeout}s]"
        return stdout, stderr, True


def snapshot_turn(title: str, run_id: str, turn_idx: int) -> None:
    """Copy the full current episode dir into an isolated, model-invisible
    per-turn snapshot: results/frontier/_agentic_work/<title>/<run_id>_snapshots/turn<turn_idx>/.
    Re-running the same turn index overwrites that turn's snapshot only."""
    src = episode_dir(title, run_id)
    dst = snapshot_root(title, run_id) / f"turn{turn_idx}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
