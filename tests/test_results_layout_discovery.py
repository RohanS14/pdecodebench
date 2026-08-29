"""Every reader of the results tree must find the per-model layout.

The results have been one directory per model since 2026-08-25:

    results/cross_modal_consistency/<model-slug>/<model>__think_on__consistency.jsonl

Three separate readers globbed "<dir>/*.jsonl" -- flat, not recursive -- and each
one, on meeting that layout, returned an EMPTY list rather than raising:

  * shared/upload_helper.py         uploaded nothing and reported success
  * cross_modal_consistency/eval/aggregate_cross_modal.py  wrote a summary over zero rows
  * viz/pde_dual_report.py          built a 5.8MB report whose entire Experiment 2
                                    half rendered as "no rows loaded" placeholders

The last one is the reason this file exists. Every cross-modal panel is written to
degrade to a placeholder instead of crashing, which is right when a job has not run
yet and wrong when the rows are on disk one directory down -- the report looked like
a faithful account of an experiment that had not happened, over 24,576 rows that had.

These tests build a tiny nested tree and assert each reader walks it. They do not
check the numbers; they check that the readers can see the files at all.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ROW = {"item_id": "Heat_1_A0_0", "gt_sample": "Heat_1", "model": "test/M",
       "thinking": "on", "condition": "A0", "corrupted_view": "none",
       "traj_level": "valid", "names": "real", "sample_idx": 0,
       "slots": ["code", "math", "trajectory", "description"],
       "outlier_slot": None, "response": "Agree: yes</think>", "finish_reason": "stop",
       "agree": True, "outlier": "none", "detection_correct": True,
       "localization_correct": None, "protocol_violation": False}


@pytest.fixture
def nested(tmp_path):
    """Two models, each in its own subdirectory -- the real layout."""
    for slug, model in [("test__m", "test/M"), ("test__n", "test/N")]:
        d = tmp_path / slug
        d.mkdir()
        with open(d / f"{slug}__think_on__consistency.jsonl", "w") as fh:
            for i in range(3):
                fh.write(json.dumps({**ROW, "model": model,
                                     "item_id": f"Heat_1_A0_{i}"}) + "\n")
    return tmp_path


def test_aggregator_walks_the_per_model_tree(nested):
    from cross_modal_consistency.eval.aggregate_cross_modal import load_rows
    rows = load_rows(str(nested))
    assert len(rows) == 6, f"aggregator saw {len(rows)} of 6 rows in a nested tree"
    assert {r["model"] for r in rows} == {"test/M", "test/N"}


def test_aggregator_refuses_an_empty_tree_instead_of_summarising_nothing(tmp_path):
    """An empty result must stop the run, not produce a summary over zero rows."""
    from cross_modal_consistency.eval.aggregate_cross_modal import load_rows
    with pytest.raises(SystemExit):
        load_rows(str(tmp_path))


def test_aggregator_skips_rescore_backups(nested):
    """A .prerescore twin must not be counted beside the file that superseded it."""
    from cross_modal_consistency.eval.aggregate_cross_modal import load_rows
    src = next(nested.glob("test__m/*.jsonl"))
    (src.parent / (src.name + ".prerescore")).write_text(src.read_text())
    assert len(load_rows(str(nested))) == 6


def test_dual_report_finds_nested_cross_modal_rows(nested, tmp_path, capsys):
    """The report must load the rows, not fall through to its placeholders."""
    import glob as _g
    paths = sorted(q for q in _g.glob(os.path.join(str(nested), "**", "*.jsonl"),
                                      recursive=True)
                   if not q.endswith((".prerescore", ".pretruncfix", ".prenormalize")))
    assert len(paths) == 2, "the report's own glob must reach into the model dirs"

    # And the source really does use a recursive glob, rather than this test asserting
    # a property of a pattern the report does not share.
    src = open(os.path.join(ROOT, "viz", "pde_dual_report.py"), encoding="utf-8").read()
    # Anchored on the load site itself. "if not rows:" also appears in exp2_headline,
    # where it guards the placeholder rather than the glob.
    i = src.index('xmodal_dir or ""')
    assert "recursive=True" in src[i - 400:i + 400], (
        "viz/pde_dual_report.py globs the cross-modal dir non-recursively; every "
        "Experiment 2 panel will silently render as a placeholder")


def test_upload_helper_globs_recursively():
    src = open(os.path.join(ROOT, "shared", "upload_helper.py"), encoding="utf-8").read()
    assert "recursive=True" in src, (
        "shared/upload_helper.py globs results_dir non-recursively; it would upload "
        "zero rows and report success")


# ── the synthetic demo must not be able to masquerade as a result ─────────────
#
# figures/table_main.tex held model-a/b/c numbers under a caption reading as a real
# result for five days, because viz/consistency/build.py defaulted to the synthetic
# CSV *and* to figures/. Two independent guards, so neither alone is load-bearing.

def test_build_does_not_default_to_the_paper_figure_directory():
    import viz.consistency.build as B
    src = open(os.path.abspath(B.__file__), encoding="utf-8").read()
    assert '"--outdir", default="figures"' not in src, (
        "viz/consistency/build.py defaults to the synthetic CSV; writing it into "
        "figures/ puts placeholder numbers next to the paper's real figures")


def test_no_judge_column_without_judge_data():
    """The real frames carry no judge_correct -- no LLM-judge pass has been run.

    _fmt renders a missing rate as "--", so the column used to appear as eight dashes
    under a caption promising "the rate at which an LLM judge finds the justification
    names the real defect". That reads as "the judge confirmed nothing", which is a
    finding; the truth is that the measurement does not exist.
    """
    import pandas as pd
    from viz.consistency import tables
    from viz.consistency.constants import SCHEMA_COLUMNS

    df = pd.DataFrame([{c: None for c in SCHEMA_COLUMNS} for _ in range(4)])
    df["condition"] = ["A0", "A-C", "A-C", "A0"]
    df["true_outlier"] = ["none", "C", "C", "none"]
    df["pred_outlier"] = ["none", "C", "D", "C"]
    df["pred_agree"] = ["yes", "no", "no", "no"]
    df["judge_correct"] = None

    tex = tables.main_results_latex(df)
    # The header row, not the whole document -- the caption legitimately mentions
    # judge-confirmed in order to say the column is absent.
    header = next(l for l in tex.splitlines() if l.startswith("condition &"))
    assert "judge-confirmed" not in header, "judge column emitted with no judge data"
    assert header.count("&") == 3, f"expected 4 columns, got: {header}"
    assert r"\begin{tabular}{lrrr}" in tex, "column spec must match the header"
    assert "No LLM-judge pass has been run" in tex, (
        "the caption must say the measurement is absent, not omit it silently")


def test_latex_rows_end_in_exactly_one_row_break():
    """A row terminated with \\\\\\\\ instead of \\\\ does not compile."""
    import pandas as pd
    from viz.consistency import tables
    from viz.consistency.constants import SCHEMA_COLUMNS

    df = pd.DataFrame([{c: None for c in SCHEMA_COLUMNS} for _ in range(2)])
    df["condition"] = ["A0", "A-C"]
    df["true_outlier"] = ["none", "C"]
    df["pred_outlier"] = ["none", "C"]
    df["pred_agree"] = ["yes", "no"]

    for line in tables.main_results_latex(df).splitlines():
        if line.startswith(("A0 ", "A-C ")):
            assert line.endswith(r" \\") and not line.endswith(r"\\\\"), line
