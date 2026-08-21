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
