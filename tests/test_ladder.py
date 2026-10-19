"""Tests for the ladder comparison — the claims the figure is allowed to make.

Each test here pins one property the spec fixes: metrics on the common item set
only, agreement well-defined, the reference band being exactly the singleton range,
no trend line anywhere, and the verdict's inside/outside word agreeing with the
numbers it quotes. The last one matters most: that word is the whole finding, and
nothing else would catch it drifting from the data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from viz.consistency import ladder as L
from viz.consistency.constants import NONE


def _draws(model, item, solver, true_o, preds):
    return [{"model": model, "item_id": item, "sample_idx": i, "solver_id": solver,
             "true_outlier": true_o, "pred_outlier": p, "pred_agree": "no"}
            for i, p in enumerate(preds)]


def _frame(rows):
    return pd.DataFrame(rows)


def test_roles_are_declared_not_inferred():
    cfg = L.load_roles()
    assert [m["short"] for m in cfg["ladder"]] == [
        "Qwen3-32B", "Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.8-27B"]
    assert len(cfg["reference"]) == 4
    lo, hi = L.assert_scale_band(cfg)
    assert 27.0 <= lo <= hi <= 33.0


def test_disjoint_item_sets_raise():
    """A model scored on its own items is not being compared with anything."""
    rows = []
    for i in range(10):
        rows += _draws("A", f"a{i}", "S1", "C", ["C", "C", "C"])
        rows += _draws("B", f"b{i}", "S1", "C", ["C", "C", "C"])
    with pytest.raises(ValueError, match="same material"):
        L.common_items(_frame(rows), ["A", "B"])


def test_common_items_reports_what_each_model_lost():
    rows = []
    for i in range(10):
        rows += _draws("A", f"i{i}", "S1", "C", ["C", "C", "C"])
    for i in range(9):                      # B is missing i9
        rows += _draws("B", f"i{i}", "S1", "C", ["C", "C", "C"])
    items, rep = L.common_items(_frame(rows), ["A", "B"])
    assert len(items) == 9
    assert rep["lost"] == {"A": 1, "B": 0}
    assert rep["frac_of_smallest"] == 1.0


def test_agreement_is_one_when_all_draws_identical():
    rows = []
    for i in range(8):
        rows += _draws("A", f"i{i}", f"S{i % 3}", "C", ["C", "C", "C"])
    t = _frame(rows)
    out = L.per_model(t, set(t["item_id"]), ["A"], n_boot=20)
    assert out["agreement_rate"].iloc[0] == 1.0
    assert out["loc_acc"].iloc[0] == 1.0


def test_agreement_is_zero_when_every_draw_differs():
    rows = []
    for i in range(8):
        rows += _draws("A", f"i{i}", f"S{i % 3}", "C", ["C", "T", "D"])
    t = _frame(rows)
    out = L.per_model(t, set(t["item_id"]), ["A"], n_boot=20)
    assert out["agreement_rate"].iloc[0] == 0.0
    # Committed on every draw, right on one of three.
    assert out["loc_acc"].iloc[0] == pytest.approx(1 / 3)


def test_uncommitted_draws_excluded_from_agreement_denominator():
    """An item where a draw named nothing is not evidence about agreement."""
    rows = _draws("A", "i0", "S1", "C", ["C", "C", NONE])
    rows += _draws("A", "i1", "S2", "C", ["C", "C", "C"])
    t = _frame(rows)
    out = L.per_model(t, set(t["item_id"]), ["A"], n_boot=20)
    assert out["agreement_rate"].iloc[0] == 1.0     # i0 dropped, i1 unanimous


def test_localization_ignores_clean_items():
    """Localization is undefined where there is no outlier to find."""
    rows = _draws("A", "clean", "S1", NONE, ["C", "C", "C"])
    rows += _draws("A", "dirty", "S2", "T", ["T", "T", "T"])
    t = _frame(rows)
    out = L.per_model(t, set(t["item_id"]), ["A"], n_boot=20)
    assert out["loc_acc"].iloc[0] == 1.0


def test_reference_band_spans_exactly_min_to_max_of_singletons(monkeypatch):
    lad = pd.DataFrame({"short": list("abcd"), "release": ["2025-01-01"] * 4,
                        "loc_acc": [.4, .5, .6, .7], "loc_lo": [.3] * 4,
                        "loc_hi": [.8] * 4, "agreement_rate": [.5] * 4,
                        "agr_lo": [.4] * 4, "agr_hi": [.6] * 4})
    ref = pd.DataFrame({"short": list("wxyz"),
                        "loc_acc": [.20, .35, .55, .62], "loc_lo": [.1] * 4,
                        "loc_hi": [.9] * 4, "agreement_rate": [.3] * 4,
                        "agr_lo": [.2] * 4, "agr_hi": [.4] * 4})
    # Record the span's arguments rather than reading the patch back. axhspan
    # stores its path in axes-fraction coordinates and resolves the data values
    # through a blended transform at draw time, so the vertices say nothing about
    # which numbers were passed -- which is the thing under test.
    calls = []
    real = plt.Axes.axhspan

    def spy(self, ymin, ymax, **kw):
        calls.append((ymin, ymax))
        return real(self, ymin, ymax, **kw)

    monkeypatch.setattr(plt.Axes, "axhspan", spy)
    L.fig8_ladder(lad, ref, baseline=0.25)
    assert calls, "no reference band was drawn"
    # Exactly min-to-max of the singletons, on both panels.
    assert all(c == (0.20, 0.62) or c == (0.3, 0.3) for c in calls), calls
    assert (0.20, 0.62) in calls


def test_no_trend_line_is_drawn():
    """The ordinal panel connects the ladder and nothing else.

    A fitted line would appear as an extra Line2D spanning the axis. The ladder
    series, the reference median and the baseline are the only lines allowed.
    """
    lad = pd.DataFrame({"short": list("abcd"), "release": ["2025-01-01"] * 4,
                        "loc_acc": [.4, .5, .6, .7], "loc_lo": [.3] * 4,
                        "loc_hi": [.8] * 4, "agreement_rate": [.5] * 4,
                        "agr_lo": [.4] * 4, "agr_hi": [.6] * 4})
    ref = pd.DataFrame({"short": list("wx"), "loc_acc": [.2, .6],
                        "loc_lo": [.1] * 2, "loc_hi": [.9] * 2,
                        "agreement_rate": [.3, .4], "agr_lo": [.2] * 2,
                        "agr_hi": [.5] * 2})
    fig = L.fig8_ladder(lad, ref, baseline=0.25)
    for src in (L.fig8_ladder, L.fig8_ladder_dated, L._panel):
        src_text = src.__doc__ or ""
        assert "polyfit" not in src_text
    import inspect
    for src in (L.fig8_ladder, L.fig8_ladder_dated, L._panel):
        code = inspect.getsource(src)
        for banned in ("polyfit", "linregress", "regplot", "np.poly", "lstsq"):
            assert banned not in code, f"{src.__name__} fits a line ({banned})"
    assert fig is not None


def test_verdict_inside_outside_matches_the_numbers():
    """The word must be computed from the spread it quotes, not asserted."""
    def lad_of(a, b, halfwidth=0.005):
        return pd.DataFrame({
            "short": ["first", "m2", "m3", "last"],
            "loc_acc": [a, a, b, b],
            "loc_lo": [v - halfwidth for v in (a, a, b, b)],
            "loc_hi": [v + halfwidth for v in (a, a, b, b)]})
    # Reference spread is 10pp.
    ref = pd.DataFrame({"short": list("wxyz"), "loc_acc": [.40, .44, .47, .50]})

    big = L.verdict_line(lad_of(0.30, 0.70), ref, "loc_acc", "acc")
    assert "outside" in big and "+40pp" in big

    small = L.verdict_line(lad_of(0.50, 0.55), ref, "loc_acc", "acc")
    assert "inside" in small, small

    # Overlapping endpoint intervals => zero is credible => no reliable change.
    flat = L.verdict_line(lad_of(0.50, 0.505, halfwidth=0.05), ref, "loc_acc", "acc")
    assert "no reliable change" in flat, flat


def test_bootstrap_resamples_solvers_not_draws():
    """Two solvers, one perfect and one hopeless: the CI must be wide.

    Resampling draws would put ~1500 independent observations behind a 50% mean and
    return a CI a couple of points wide. Resampling the two SYSTEMS can return all
    of one or all of the other, so the interval has to reach toward 0 and 1.
    """
    rows = []
    for i in range(40):
        rows += _draws("A", f"good{i}", "S_good", "C", ["C", "C", "C"])
        rows += _draws("A", f"bad{i}", "S_bad", "C", ["T", "T", "T"])
    t = _frame(rows)
    out = L.per_model(t, set(t["item_id"]), ["A"], n_boot=400)
    assert out["loc_acc"].iloc[0] == pytest.approx(0.5)
    assert out["loc_lo"].iloc[0] < 0.05 and out["loc_hi"].iloc[0] > 0.95
