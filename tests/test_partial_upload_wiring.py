"""Partial upload must actually be reachable.

The generational roster is submitted all at once and harvested as it lands, so an
upload that silently no-ops turns every 8h job into a black box. This is a
regression test for a real defect: upload_partial() looked for upload_helper.py
beside itself, the file moved to shared/ in the eval->freegen refactor, and the
lookup then failed os.path.exists and returned. Nothing raised. The only trace was
one "skipping" line buried in the log, and no artifact ever appeared.
"""
import os

import crossmodal.eval.run_cross_modal_consistency as X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(X.__file__)))
ROOT = os.path.dirname(ROOT)


def test_upload_helper_exists_where_the_uploader_looks():
    here = os.path.dirname(os.path.abspath(X.__file__))
    candidates = [os.path.join(ROOT, "shared", "upload_helper.py"),
                  os.path.join(here, "upload_helper.py")]
    assert any(os.path.exists(c) for c in candidates), (
        f"upload_helper.py missing from all of {candidates} -- every partial and "
        f"final upload would silently no-op")


def test_the_uploader_resolves_a_helper_path(monkeypatch, tmp_path, capsys):
    """It must find the helper and try to run it, not bail early."""
    ran = {}

    def fake_run(cmd, **kw):
        ran["cmd"] = cmd
        class R:
            returncode, stdout, stderr = 0, "ok", ""
        return R()

    monkeypatch.setattr(X.subprocess, "run", fake_run)

    class Args:
        hf_dataset = "org/repo"
        output_dir = str(tmp_path)
        workspace = str(tmp_path)
        items = "data/multimodal_items_v1.csv"
        packages_dir = ""

    X.upload_partial(Args(), str(tmp_path / "out.jsonl"), 128, 1024)
    assert "cmd" in ran, "uploader returned without invoking the helper"
    assert ran["cmd"][1].endswith("upload_helper.py")
    assert os.path.exists(ran["cmd"][1])


def test_no_hf_dataset_is_a_deliberate_skip():
    """An empty repo name disables upload on purpose (local dry runs); that path
    must stay quiet rather than erroring."""
    class Args:
        hf_dataset = ""
    assert X.upload_partial(Args(), "/tmp/x.jsonl", 1, 1) is None


# ── The freegen sbatch's heartbeat ────────────────────────────────────────────
# Three jobs on 2026-08-24 died mid-model (two SIGKILLed by root's utilization
# sweep, one at its wall) and 1,176 finished draws never reached HF, because
# upload_partial ran only at the END of a model in both branches of the roster
# loop. bash keeps going after a crash, so those branches cover a crash; a kill
# takes the whole script down and they never run. These pin the timer-driven
# upload that closes that hole -- the shell is not importable, so they read it.

SBATCH = os.path.join(ROOT, "sbatch", "run_freegen_jul28.sbatch")


def _sbatch_text():
    with open(SBATCH) as f:
        return f.read()


def test_heartbeat_uploader_is_armed_in_the_background():
    text = _sbatch_text()
    assert "upload_heartbeat" in text, "no heartbeat uploader in the freegen sbatch"
    assert "upload_heartbeat &" in text, (
        "the heartbeat must run in the BACKGROUND -- in the foreground it blocks the "
        "roster loop, and in vLLM's own process HF's threads kill the EngineCore")
    assert "UPLOAD_EVERY" in text, "heartbeat interval must be overridable"


def test_heartbeat_and_end_of_model_upload_take_the_same_lock():
    """push_dataset_to_hub REPLACES the split, so overlapping uploads race."""
    text = _sbatch_text()
    assert text.count('mkdir "${LOCK}"') >= 3, (
        "every upload_partial call site must take ${LOCK}: the heartbeat and both "
        "branches of the roster loop")


def test_exit_trap_flushes_and_covers_sigterm():
    text = _sbatch_text()
    assert "trap _on_exit EXIT TERM INT" in text, (
        "a wall-clock TIMEOUT and a scancel both arrive as SIGTERM; without the trap "
        "the last partial window is lost")
    assert "upload_partial partial || true" in text, (
        "the exit flush must not be able to fail the job it is trying to rescue")


def test_heartbeat_survives_set_e():
    """`[ ... ] && continue` as the last command of a loop body kills a subshell."""
    text = _sbatch_text()
    i = text.index("upload_heartbeat()")
    body = text[i:text.index("HEARTBEAT_PID=\"\"", i)]
    # Comments stripped: the comment above the fix quotes the broken form.
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert '&& continue' not in body, (
        "under `set -e` a failed test as the last command of the loop body exits the "
        "background subshell and silently disarms the heartbeat -- use if/then")


# ── Artifacts must actually reach the dashboard ───────────────────────────────
# hf_utility writes the manifest's experiment_id from metadata["experiment_id"],
# and import_experiments.py keeps ONLY manifest rows that have one. Nothing set it,
# so all eight freegen arms uploaded cleanly, verified cleanly, and were invisible
# on the Artifacts tab. "Uploaded" and "visible to the user" are not the same claim.

def test_upload_helper_sets_experiment_id_not_just_experiment_name():
    with open(os.path.join(ROOT, "shared", "upload_helper.py")) as f:
        text = f.read()
    assert '"experiment_id":' in text, (
        "metadata must carry experiment_id -- the dashboard drops manifest rows "
        "without one, so the artifact uploads and then cannot be seen")
    assert "--experiment_id" in text, (
        "the HF naming slug and the notes-folder name are different strings "
        "(pde-llm-eval vs pde-freegen-xmodal) and need separate flags")


def test_freegen_sbatch_passes_the_notes_folder_name():
    with open(SBATCH) as f:
        text = f.read()
    assert "--experiment_id" in text, (
        "the sbatch must name the experiment FOLDER, or every artifact it uploads "
        "lands in the manifest unlinked")
