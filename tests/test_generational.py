"""The generational trend: d' per model against release date.

These tests pin the two things that would produce a confident wrong answer rather
than an error -- plotting response bias as if it were sensitivity, and computing
intervals as if 1024 items were 1024 independent observations when there are only
32 solvers behind them.
"""
import numpy as np
import pandas as pd
import pytest

from viz.consistency import generational as G
from viz.consistency.constants import SCHEMA_COLUMNS


def _rows(model, release, n_solvers=8, hit=0.6, fa=0.2, family="F", params=30.0,
          seed=0):
    """Synthetic arm with a known hit and false-alarm rate."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_solvers):
        for i in range(8):                       # corrupted items
            out.append(dict(
                run_id=f"{model}-{s}-c{i}", solver_id=f"S{s}", condition="A-C",
                true_outlier="C", traj_level="", naming="real", reasoning="on",
                model=model, order="C,T,D,M",
                pred_agree=("no" if rng.random() < hit else "yes"),
                pred_outlier="C", pred_pde_class="", pred_method="",
                justification="", pde_class="Heat", numerical_method="fd",
                release_date=release, params_total_b=params, family=family))
        for i in range(8):                       # clean items
            out.append(dict(
                run_id=f"{model}-{s}-a{i}", solver_id=f"S{s}", condition="A0",
                true_outlier="none", traj_level="", naming="real", reasoning="on",
                model=model, order="C,T,D,M",
                pred_agree=("no" if rng.random() < fa else "yes"),
                pred_outlier="none", pred_pde_class="", pred_method="",
                justification="", pde_class="Heat", numerical_method="fd",
                release_date=release, params_total_b=params, family=family))
    return out


def _frame(*arms):
    return pd.DataFrame([r for a in arms for r in a]).reindex(
        columns=[c for c in SCHEMA_COLUMNS if c != "judge_correct"])


def test_empty_frame_returns_an_empty_table_not_an_error():
    out = G.by_release(_frame([]).iloc[0:0])
    assert out.empty


def test_a_model_with_no_release_date_is_dropped_not_dated_zero():
    """An unregistered model must vanish from the trend, never be plotted at a
    guessed date -- a fabricated x position is worse than a missing point."""
    rows = _rows("unknown/model", release="", seed=1)
    out = G.by_release(_frame(rows))
    assert out.empty


def test_high_hit_rate_with_high_false_alarms_does_not_beat_a_discriminating_arm():
    """The reason the y-axis is d' and not hit rate.

    'shouty' says "disagree" almost always: hit 0.95, but false alarms 0.90. 'sharp'
    hits less often, 0.70, but false-alarms at 0.10. Ranked on hit rate shouty wins;
    ranked on d' -- which is what detection means -- sharp wins by a wide margin.
    """
    df = _frame(_rows("shouty", "2025-01-01", hit=0.95, fa=0.90, seed=2),
                _rows("sharp", "2026-01-01", hit=0.70, fa=0.10, seed=3))
    out = G.by_release(df, n_boot=300).set_index("model")
    assert out.loc["shouty", "hit_rate"] > out.loc["sharp", "hit_rate"]
    assert out.loc["sharp", "dprime"] > out.loc["shouty", "dprime"]


def test_intervals_are_clustered_over_solvers_not_rows():
    """A row-level bootstrap would be ~sqrt(32) too narrow. Holding the rates fixed
    and multiplying the ITEMS per solver must not shrink the interval much, because
    n is the number of solvers; multiplying the SOLVERS must."""
    few = G.by_release(_frame(_rows("m", "2026-01-01", n_solvers=4, seed=4)),
                       n_boot=800)
    many = G.by_release(_frame(_rows("m", "2026-01-01", n_solvers=32, seed=4)),
                        n_boot=800)
    w_few = float(few["hi"].iloc[0] - few["lo"].iloc[0])
    w_many = float(many["hi"].iloc[0] - many["lo"].iloc[0])
    assert w_many < w_few, "more solvers must tighten a solver-clustered interval"


def test_trend_needs_at_least_three_dated_points():
    two = _frame(_rows("a", "2025-01-01", seed=5), _rows("b", "2026-01-01", seed=6))
    assert G.trend(G.by_release(two, n_boot=200)) is None


def test_trend_returns_none_when_every_model_shares_one_date():
    """No date spread means no slope, and fitting one would divide by zero."""
    same = _frame(_rows("a", "2026-01-01", seed=7),
                  _rows("b", "2026-01-01", seed=8),
                  _rows("c", "2026-01-01", seed=9))
    assert G.trend(G.by_release(same, n_boot=200)) is None


def test_trend_reports_how_many_models_it_fitted():
    """A slope through a handful of models is a description of those models. The
    count travels with the number so a caption cannot quietly omit it."""
    df = _frame(_rows("a", "2025-01-01", hit=0.55, fa=0.35, seed=10),
                _rows("b", "2025-07-01", hit=0.65, fa=0.25, seed=11),
                _rows("c", "2026-01-01", hit=0.80, fa=0.15, seed=12))
    t = G.trend(G.by_release(df, n_boot=300))
    assert t is not None
    assert t["n_models"] == 3
    assert t["span_days"] > 300
    assert t["slope_per_year"] > 0        # improving by construction


@pytest.mark.parametrize("hit,fa,expected", [
    (1.0, 1.0, True),    # says "disagree" to everything
    (0.0, 0.0, True),    # says "agree" to everything
    (1.0, 0.0, False),   # PERFECT detector -- both rates extreme, maximally informative
    (0.0, 1.0, False),   # perfectly anti-correlated -- also informative
])
def test_degenerate_means_all_one_way_not_merely_extreme(hit, fa, expected):
    """Regression: an earlier version flagged degenerate when BOTH rates were in
    (0, 1), which caught hit=1.0/fa=0.0 -- the best possible detector. trend()
    filters on ~degenerate, so the strongest model in the roster was silently
    dropped from the fit, and with few points that turned a real slope into None."""
    df = _frame(_rows("m", "2026-01-01", hit=hit, fa=fa, seed=13))
    out = G.by_release(df, n_boot=200)
    assert bool(out["degenerate"].iloc[0]) is expected


def test_a_perfect_detector_still_reaches_the_trend_fit():
    df = _frame(_rows("a", "2025-01-01", hit=0.55, fa=0.35, seed=20),
                _rows("b", "2025-07-01", hit=0.70, fa=0.20, seed=21),
                _rows("c", "2026-01-01", hit=1.0, fa=0.0, seed=22))
    t = G.trend(G.by_release(df, n_boot=300))
    assert t is not None and t["n_models"] == 3


def test_arm_filter_selects_a_single_reasoning_level():
    df = _frame(_rows("m", "2026-01-01", seed=14))
    df.loc[df.index[:64], "reasoning"] = "off"
    both = G.by_release(df, n_boot=200)
    on_only = G.by_release(df, n_boot=200, arm="on")
    assert set(both["reasoning"]) == {"on", "off"}
    assert set(on_only["reasoning"]) == {"on"}
