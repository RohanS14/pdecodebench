"""freegen/ modules must win their own names against the cluster's eval/ copies.

The laptop repo has no eval/parse_score.py, so a shadowing bug is INVISIBLE here and
only appears on torch, where both directories hold one. That is exactly how the first
canary submission died: four jobs, twenty seconds each, "cannot import name
'is_no_verdict' from parse_score (.../eval/parse_score.py)". These tests read the
source rather than the resolved import, because the local filesystem cannot reproduce
the condition.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEGEN = ("freegen/run_eval.py", "freegen/report.py")


def _lines(rel):
    return open(os.path.join(ROOT, rel)).read().splitlines()


def test_eval_dir_is_appended_never_inserted_at_zero():
    for rel in FREEGEN:
        for ln in _lines(rel):
            if "eval" in ln and "sys.path" in ln:
                assert "insert(0" not in ln, (
                    f"{rel}: `{ln.strip()}` puts eval/ ahead of this package's own "
                    f"directory, so eval/parse_score.py shadows freegen/parse_score.py "
                    f"on the cluster")


def test_own_directory_precedes_eval_in_every_freegen_entrypoint():
    for rel in FREEGEN:
        src = "\n".join(_lines(rel))
        own = [m.start() for m in re.finditer(r"sys\.path\.\w+\([^\n]*__file__[^\n]*\)", src)
               if "eval" not in src[m.start():m.end()]]
        ev = [m.start() for m in re.finditer(r"sys\.path\.\w+\([^\n]*eval[^\n]*\)", src)]
        assert own, f"{rel}: does not put its own directory on sys.path at all"
        assert ev, f"{rel}: does not resolve eval/ for dataset_io"
        assert min(own) < min(ev), f"{rel}: eval/ is configured before the package's own dir"


def test_is_no_verdict_is_importable_from_the_freegen_copy():
    src = open(os.path.join(ROOT, "freegen/parse_score.py")).read()
    assert "def is_no_verdict(" in src
    assert "def strip_think(" in src
