"""The paired detection/localization figure, and the identity that justifies pairing it.

The two panels are only allowed to share a row axis because they are one measurement
split by denominator: panel A's dot sits where panel B's hatch ends. That identity is
the figure's whole argument, so it is asserted here rather than trusted -- and the one
thing that can break it (a draw that ended without a verdict, which B counts and A
drops) is asserted to be *reported* rather than silently absorbed.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from viz.consistency import claim_report as CR
from viz.consistency import figures as F
from viz.consistency.constants import SCHEMA_COLUMNS
from viz.consistency.sensitivity import SIGNAL_CONDITIONS, detection_sensitivity
from viz.consistency.synth import Effects, generate


@pytest.fixture(scope="module")
def df():
    return generate(Effects(n_solvers=8, models=("m1", "m2")))


def test_pair_builds_two_panels_on_a_shared_row_axis(df):
    fig = CR.fig_detection_blame_pair(df)
    axa, axb = fig.axes[0], fig.axes[1]
    assert axa.get_ylim() == axb.get_ylim()
    # B carries no y labels of its own; A names every row exactly once.
    assert [t.get_text() for t in axb.get_yticklabels() if t.get_text()] == []
    assert axa.get_yticklabels()[-1].get_text() == "nothing corrupted"
    plt.close(fig)


def test_panels_agree_row_by_row_when_every_draw_has_a_verdict(df):
    """A's flag rate is 1 minus B's 'does not identify disagreement' share."""
    fig = CR.fig_detection_blame_pair(df)
    assert fig._pair_residual < 1e-9, fig._pair_residual
    plt.close(fig)

    # ...and again against the numbers directly, not just the figure's own report.
    tally = CR._unconditional_counts(df)
    rates = {r["condition"]: r["hit_rate"]
             for r in detection_sensitivity(df).rows if not r["empty"]}
    for cond, rate in rates.items():
        counts, total = tally[cond]
        assert np.isclose(1.0 - counts[CR.MISS] / total, rate), cond


def test_no_verdict_draws_are_reported_not_absorbed(df):
    """The one thing that breaks the identity has to appear on the figure."""
    d_all = df.copy()
    nv = np.zeros(len(d_all), dtype=bool)
    target = d_all["condition"].eq(SIGNAL_CONDITIONS[0]).to_numpy()
    nv[np.flatnonzero(target)[: max(1, int(target.sum() // 3))]] = True
    d_all["no_verdict"] = nv

    fig = CR.fig_detection_blame_pair(df, d_all=d_all)
    assert fig._pair_residual > 0.005, fig._pair_residual
    text = " ".join(t.get_text() for t in fig.texts)
    assert "without a verdict" in text
    plt.close(fig)


def test_segments_sum_to_the_row_total(df):
    """UNCLEAR is a subtraction, so a parse gap shows as a sliver, never as a drop."""
    for cond, (counts, total) in CR._unconditional_counts(df).items():
        if total:
            assert sum(counts.values()) == total, cond


def test_pair_and_standalone_draw_the_same_bars():
    """Neither figure classifies draws itself; both ask the same helper.

    Scoped to each function's own source rather than counted across the module -- a
    module-wide count is a test about how many times a name appears, which fails when
    a third caller is added and passes when a caller quietly stops using it.
    """
    import inspect
    for fn in (CR.fig_blame_stack_unconditional, CR.fig_detection_blame_pair):
        src = inspect.getsource(fn)
        assert "_unconditional_counts(" in src, fn.__name__
        assert "counts[MISS]" not in src.replace("counts.get(MISS", ""), fn.__name__
    # ...and the control row goes through the same classifier as every other row.
    assert "_classify_blame(" in inspect.getsource(CR._clean_counts)
    assert "_classify_blame(" in inspect.getsource(CR._unconditional_counts)


def test_the_control_row_is_drawn_and_marked_correct_on_the_agree_segment(df):
    """The row the standalone figure omits, and the reason the marker is a caret.

    On items where nothing was corrupted the correct answer is "says all four agree",
    not a named view -- so correctness cannot be an outline around a modality
    segment, and the control row cannot be left blank without hiding where blame goes
    when there is no signal to respond to.
    """
    counts, total = CR._clean_counts(df)
    assert total > 0 and sum(counts.values()) == total
    assert counts[CR.MISS] > 0

    fig = CR.fig_detection_blame_pair(df)
    axb = fig.axes[1]
    # One caret per drawn row: seven corrupted plus the control.
    carets = [ln for ln in axb.lines if ln.get_marker() == "^"]
    assert len(carets) == len(CR._unconditional_counts(df)) + 1

    # The control row's caret sits inside its "says all four agree" segment, which
    # runs from the flag rate to 1.0 -- i.e. it is the rightmost caret on the panel.
    lowest = min(carets, key=lambda ln: ln.get_ydata()[0])
    x = lowest.get_xdata()[0]
    start = 1.0 - counts[CR.MISS] / total
    assert start < x < 1.0, (start, x)
    plt.close(fig)


def test_correctness_is_a_caret_not_a_second_box(df):
    """An outline beside the hatched segment's own border was two dark rectangles of
    similar weight where only one meant anything."""
    fig = CR.fig_detection_blame_pair(df)
    axb = fig.axes[1]
    outlines = [p for p in axb.patches
                if p.get_facecolor()[3] == 0 and p.get_linewidth() > 0]
    assert not outlines, "correct answer is still drawn as an unfilled box"
    assert any(ln.get_marker() == "^" for ln in axb.lines)
    plt.close(fig)


def test_pair_survives_an_empty_frame():
    fig = CR.fig_detection_blame_pair(pd.DataFrame(columns=list(SCHEMA_COLUMNS)))
    assert fig is not None
    plt.close(fig)


def test_pair_is_exported_with_the_other_paper_figures():
    assert "fig8_detect_localize" in F.FIGURES


def test_pair_stays_out_of_the_frozen_report(df, tmp_path):
    """consistency_claims.html must keep rebuilding byte for byte, so the paired
    figure is gated behind the same flag the unconditional one is."""
    out = str(tmp_path / "frozen.html")
    CR.build(df, out=out, blame_unconditional=False)
    assert "both questions on one row axis" not in io.open(out).read()
    CR.build(df, out=out, blame_unconditional=True, d_all=df)
    assert "both questions on one row axis" in io.open(out).read()
