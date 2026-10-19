"""freegen_static_judgments/ modules must win their own names -- and must not put a sibling on sys.path.

History: the cluster's flat `eval/` held a stale pre-split `parse_score.py` alongside
freegen_static_judgments's real one. freegen_static_judgments/run_eval.py had to add `eval/` to sys.path anyway, because
`dataset_io` lived only there. With both directories on the path and both holding a
`parse_score.py`, ordering was the only thing standing between the job and the wrong
scorer -- and it lost: four canary jobs, twenty seconds each, "cannot import name
'is_no_verdict' from parse_score (.../eval/parse_score.py)".

The old version of this file pinned the ORDERING (eval/ appended, never inserted at 0).
That was the right guard while eval/ existed. It no longer does: dataset_io moved to
shared/ when the tree was reorganised by experiment, so freegen_static_judgments imports it by package
path and puts NO sibling directory on sys.path. This file now pins that stronger
property -- there is no second parse_score.py reachable by bare name, so the ordering
question cannot be asked at all.

These read the source rather than the resolved import: the condition was never
reproducible on the laptop, which is precisely why it reached the cluster.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEGEN = ("freegen_static_judgments/run_eval.py", "freegen_static_judgments/report.py")


def _src(rel):
    return open(os.path.join(ROOT, rel)).read()


def test_no_freegen_entrypoint_puts_eval_on_sys_path():
    for rel in FREEGEN:
        for ln in _src(rel).splitlines():
            if "sys.path" in ln and not ln.lstrip().startswith("#"):
                assert "eval" not in ln, (
                    f"{rel}: `{ln.strip()}` puts an eval/ directory back on sys.path. "
                    f"dataset_io lives in shared/ and is imported by package path; a "
                    f"bare sibling directory reintroduces the parse_score shadowing bug")


def test_dataset_io_is_imported_by_package_path_not_bare():
    for rel in FREEGEN:
        src = _src(rel)
        assert "\nfrom dataset_io import" not in src, (
            f"{rel}: imports dataset_io by bare name, which only resolves if a sibling "
            f"directory is on sys.path -- the thing this module must not do")


def test_freegen_own_directory_is_first_on_sys_path():
    for rel in FREEGEN:
        src = _src(rel)
        own = [m.start() for m in
               re.finditer(r"sys\.path\.insert\(\s*0\s*,[^\n]*__file__[^\n]*\)", src)]
        assert own, f"{rel}: does not insert its own directory at position 0"


def test_is_no_verdict_is_importable_from_the_freegen_copy():
    src = _src("freegen_static_judgments/parse_score.py")
    assert "def is_no_verdict(" in src
    assert "def strip_think(" in src


def test_there_is_exactly_one_parse_score_in_the_repo():
    """The duplicate is what made ordering load-bearing. There must not be a second."""
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "results", "data", "figures")]
        if "parse_score.py" in filenames:
            found.append(os.path.relpath(os.path.join(dirpath, "parse_score.py"), ROOT))
    assert found == ["freegen_static_judgments/parse_score.py"], f"expected one parse_score.py, found {found}"
