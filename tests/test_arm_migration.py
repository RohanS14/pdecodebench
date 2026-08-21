"""Tests for the pre-v6 -> per-arm results migration and the legacy resume path.

The hazard being guarded: run_eval appends by PATH, so renaming a file that a job is
writing splits its output. And if the legacy checkpoint is not honoured, a documented
wall-timeout resubmit regenerates a finished model from zero.
"""
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "eval"), os.path.join(ROOT, "freegen")]

from freegen.run_eval import legacy_arm, load_checkpoint  # noqa: E402

MIGRATE = os.path.join(ROOT, "freegen", "migrate_arm_filenames.py")


def _legacy(dirpath, slug, model, n=3, thinking=None):
    p = os.path.join(dirpath, f"{slug}.jsonl")
    with open(p, "w") as f:
        for i in range(n):
            row = {"title": f"T{i}", "mod_type": "Comm_Valid", "model": model}
            if thinking:
                row["thinking"] = thinking
            f.write(json.dumps(row) + "\n")
    # Age it past the live-file window so the migration will consider it.
    old = time.time() - 3 * 3600
    os.utime(p, (old, old))
    return p


# ── legacy arm inference ─────────────────────────────────────────────────────
def test_legacy_arm_matches_the_f9_diagnosis():
    """Pre-v6, toggle models were forced enable_thinking=False."""
    assert legacy_arm("Qwen/Qwen3-32B") == "off"
    assert legacy_arm("Qwen/Qwen3-Coder-30B-A3B-Instruct") == "off"
    assert legacy_arm("Qwen/QwQ-32B") == "on"
    assert legacy_arm("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") == "on"


# ── resume ───────────────────────────────────────────────────────────────────
def test_checkpoint_reads_several_files(tmp_path):
    a = _legacy(str(tmp_path), "m__a", "Qwen/Qwen3-32B", n=2)
    b = _legacy(str(tmp_path), "m__b", "Qwen/Qwen3-32B", n=3)
    # distinct titles across the two files
    with open(b, "w") as f:
        for i in range(3):
            f.write(json.dumps({"title": f"Z{i}", "mod_type": "Comm_Valid",
                                "model": "Qwen/Qwen3-32B"}) + "\n")
    done = load_checkpoint([a, b], "Qwen/Qwen3-32B", "off")
    assert len(done) == 5


def test_checkpoint_ignores_rows_from_a_different_arm(tmp_path):
    p = os.path.join(str(tmp_path), "x.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"title": "A", "mod_type": "c", "model": "m",
                            "thinking": "on"}) + "\n")
        f.write(json.dumps({"title": "B", "mod_type": "c", "model": "m",
                            "thinking": "off"}) + "\n")
    assert len(load_checkpoint([p], "m", "off")) == 1
    assert len(load_checkpoint([p], "m", "on")) == 1


def test_untagged_legacy_rows_are_accepted_for_the_matching_arm(tmp_path):
    """Legacy rows carry no `thinking` key; they must still resume."""
    p = _legacy(str(tmp_path), "legacy", "Qwen/Qwen3-32B", n=4)
    assert len(load_checkpoint([p], "Qwen/Qwen3-32B", "off")) == 4


def test_untagged_legacy_rows_are_refused_for_the_other_arm(tmp_path):
    """A thinking-ON run must not inherit rows generated with reasoning disabled."""
    p = _legacy(str(tmp_path), "legacy", "Qwen/Qwen3-32B", n=4)
    assert len(load_checkpoint([p], "Qwen/Qwen3-32B", "on")) == 0


def test_untagged_rows_of_an_always_thinking_model_resume_as_on(tmp_path):
    p = _legacy(str(tmp_path), "legacy", "Qwen/QwQ-32B", n=2)
    assert len(load_checkpoint([p], "Qwen/QwQ-32B", "on")) == 2
    assert len(load_checkpoint([p], "Qwen/QwQ-32B", "off")) == 0


def test_missing_file_is_not_an_error(tmp_path):
    assert load_checkpoint([os.path.join(str(tmp_path), "nope.jsonl")]) == set()


# ── migration ────────────────────────────────────────────────────────────────
def _run(args):
    return subprocess.run([sys.executable, MIGRATE, *args],
                          capture_output=True, text=True)


def test_dry_run_writes_nothing(tmp_path):
    p = _legacy(str(tmp_path), "Qwen__Qwen3-32B", "Qwen/Qwen3-32B")
    before = sorted(os.listdir(str(tmp_path)))
    r = _run(["--results_dir", str(tmp_path)])
    assert r.returncode == 0, r.stderr
    assert "dry run" in r.stdout
    assert sorted(os.listdir(str(tmp_path))) == before
    assert os.path.exists(p)


def test_apply_backfills_the_arm_and_renames(tmp_path):
    _legacy(str(tmp_path), "Qwen__Qwen3-32B", "Qwen/Qwen3-32B", n=3)
    r = _run(["--results_dir", str(tmp_path), "--apply"])
    assert r.returncode == 0, r.stderr
    dest = tmp_path / "Qwen__Qwen3-32B__think-off.jsonl"
    assert dest.exists()
    rows = [json.loads(l) for l in dest.read_text().splitlines() if l.strip()]
    assert len(rows) == 3
    assert all(row["thinking"] == "off" for row in rows)
    # original preserved, and NOT under a name the *.jsonl upload glob would catch
    assert (tmp_path / "Qwen__Qwen3-32B.jsonl.pre-v6").exists()
    assert not (tmp_path / "Qwen__Qwen3-32B.jsonl").exists()


def test_always_thinking_model_migrates_to_the_on_arm(tmp_path):
    _legacy(str(tmp_path), "Qwen__QwQ-32B", "Qwen/QwQ-32B", n=2)
    assert _run(["--results_dir", str(tmp_path), "--apply"]).returncode == 0
    dest = tmp_path / "Qwen__QwQ-32B__think-on.jsonl"
    assert dest.exists()
    rows = [json.loads(l) for l in dest.read_text().splitlines() if l.strip()]
    assert all(row["thinking"] == "on" for row in rows)


def test_a_freshly_written_file_is_skipped_as_live(tmp_path):
    """The core safety interlock: never migrate a file a job may be appending to."""
    p = _legacy(str(tmp_path), "Qwen__QwQ-32B", "Qwen/QwQ-32B")
    os.utime(p, None)                      # touch -> looks live
    r = _run(["--results_dir", str(tmp_path), "--apply"])
    assert r.returncode == 0
    assert "looks live" in r.stdout
    assert os.path.exists(p)
    assert not (tmp_path / "Qwen__QwQ-32B__think-on.jsonl").exists()


def test_already_migrated_files_are_left_alone(tmp_path):
    _legacy(str(tmp_path), "m__think-off", "Qwen/Qwen3-32B")
    r = _run(["--results_dir", str(tmp_path), "--apply"])
    assert "nothing to do" in r.stdout


def test_migration_is_idempotent(tmp_path):
    _legacy(str(tmp_path), "Qwen__Qwen3-32B", "Qwen/Qwen3-32B", n=3)
    assert _run(["--results_dir", str(tmp_path), "--apply"]).returncode == 0
    second = _run(["--results_dir", str(tmp_path), "--apply"])
    assert second.returncode == 0
    assert "nothing to do" in second.stdout
