"""Tests for the paired obfuscation analysis and its figure.

Aimed at the ways a paired analysis silently goes wrong: shares that do not close,
solvers dropped without anyone noticing, and a bootstrap that reports an effect the
data does not contain (or misses one it does).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.consistency import obfuscation as OB          # noqa: E402
from viz.consistency.constants import (MODALITIES, NAMING_LEVELS,  # noqa: E402
                                       NONE, SCHEMA_COLUMNS)
from viz.consistency.synth import Effects, generate    # noqa: E402

BOOT = 800          # smaller than production; the estimator is the same


def _frame(rows):
    d = pd.DataFrame(rows)
    for c in SCHEMA_COLUMNS:
        if c not in d:
            d[c] = ""
    return d


def _synthetic(n_solvers=24, shift=None, seed=1):
    """Solvers x conditions x namings, with an optional injected C-ward shift."""
    rng = np.random.default_rng(seed)
    rows = []
    conds = ["A0", "A-T-rand", "A-D", "A-M", "A-C"]
    for s in range(n_solvers):
        for cond in conds:
            true_out = {"A0": NONE, "A-T-rand": "T", "A-D": "D",
                        "A-M": "M", "A-C": "C"}[cond]
            for naming in NAMING_LEVELS:
                for rep in range(8):
                    p = {"C": 0.2, "T": 0.2, "D": 0.2, "M": 0.2, NONE: 0.2}
                    if shift and naming == NAMING_LEVELS[1]:
                        take = shift / 4.0
                        for k in ("T", "D", "M", NONE):
                            p[k] -= take
                        p["C"] += shift
                    cats = list(p)
                    pred = rng.choice(cats, p=[max(p[k], 0.0) for k in cats]
                                      / np.sum([max(p[k], 0.0) for k in cats]))
                    rows.append(dict(
                        run_id=f"r{s}-{cond}-{naming}-{rep}", solver_id=f"S{s:02d}",
                        condition=cond, true_outlier=true_out, naming=naming,
                        reasoning="on", model="m", pred_agree="no",
                        pred_outlier=pred, justification=""))
    return _frame(rows)


# ── shares close ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("slice_name", ["innocent", "guilty"])
def test_shares_sum_to_one_within_each_naming_condition(slice_name):
    d = _synthetic()
    matched, _, _ = OB.match_pairs(d, slice_name)
    shares = OB._solver_shares(matched)
    for naming in NAMING_LEVELS:
        tot = shares[naming].sum(axis=1).dropna()
        assert len(tot)
        assert np.allclose(tot.to_numpy(), 1.0, atol=1e-6)


@pytest.mark.parametrize("slice_name", ["innocent", "guilty"])
def test_reported_mean_shares_also_close(slice_name):
    res = OB.paired_blame_shift(_synthetic(), slice_name, n_boot=BOOT)
    for key in ("real", "obf"):
        assert abs(sum(r[key] for r in res.rows) - 1.0) < 1e-6


def test_five_rows_in_the_specified_order():
    res = OB.paired_blame_shift(_synthetic(), "innocent", n_boot=BOOT)
    assert [r["category"] for r in res.rows] == list(MODALITIES) + [NONE]


# ── the slices are disjoint and never pooled ─────────────────────────────────
def test_slices_are_disjoint_and_cover_the_frame():
    d = _synthetic()
    a, _, ua = OB.match_pairs(d, "innocent")
    b, _, ub = OB.match_pairs(d, "guilty")
    assert set(a["true_outlier"]).isdisjoint({"C"})
    assert set(b["true_outlier"]) == {"C"}
    assert len(a) + len(b) + ua + ub == len(d)


# ── pairing drops are reported, not swallowed ────────────────────────────────
def test_unpaired_cells_are_dropped_and_counted():
    d = _synthetic(n_solvers=6)
    # Remove one naming condition from one (solver, condition, reasoning, model) cell.
    victim = (d["solver_id"].eq("S00") & d["condition"].eq("A-D")
              & d["naming"].eq(NAMING_LEVELS[1]))
    assert victim.any()
    res = OB.paired_blame_shift(d[~victim], "innocent", n_boot=BOOT)
    assert res.dropped_cells == 1, "an unpaired cell must be counted, not swallowed"


def test_solver_missing_a_naming_condition_entirely_is_excluded():
    d = _synthetic(n_solvers=6)
    full = OB.paired_blame_shift(d, "innocent", n_boot=BOOT).n_solvers
    gone = d["solver_id"].eq("S00") & d["naming"].eq(NAMING_LEVELS[1])
    res = OB.paired_blame_shift(d[~gone], "innocent", n_boot=BOOT)
    assert res.n_solvers == full - 1
    assert res.dropped_cells > 0


def test_unparseable_verdicts_are_excluded_and_counted():
    d = _synthetic(n_solvers=6)
    d.loc[d.index[:10], "pred_outlier"] = ""
    res = OB.paired_blame_shift(d, "innocent", n_boot=BOOT)
    assert res.dropped_unparsed > 0


# ── the estimator recovers a known effect, and invents none ──────────────────
def test_injected_shift_is_recovered_within_bootstrap_error():
    shift = 0.12                       # +12pp toward code under obfuscation
    res = OB.paired_blame_shift(_synthetic(n_solvers=40, shift=shift, seed=7),
                                "innocent", n_boot=BOOT)
    code = next(r for r in res.rows if r["category"] == "C")
    assert code["diff"] == pytest.approx(shift, abs=0.04)
    assert code["lo"] < shift < code["hi"]
    assert code["significant"]


def test_zero_true_effect_produces_all_ns():
    res = OB.paired_blame_shift(_synthetic(n_solvers=40, shift=None, seed=11),
                                "innocent", n_boot=BOOT)
    assert not any(r["significant"] for r in res.rows), \
        "a null frame must not yield a significant reallocation"


def test_zero_effect_caption_says_nothing_survives():
    import matplotlib
    matplotlib.use("Agg")
    from viz.consistency import figures as F
    fig, st, cap = F.fig5_obfuscation_contrast(
        _synthetic(n_solvers=40, shift=None, seed=11), n_boot=BOOT)
    assert "No modality's share change survives correction." in cap
    matplotlib.pyplot.close(fig)


# ── caption contents ─────────────────────────────────────────────────────────
def test_caption_states_the_required_facts():
    import matplotlib
    matplotlib.use("Agg")
    from viz.consistency import figures as F
    d = _synthetic(n_solvers=30, shift=0.12, seed=3)
    fig, st, cap = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    assert "n=30 solvers" in cap                       # n solvers
    assert f"{BOOT:,} bootstrap" in cap                # bootstrap count
    assert "Bonferroni-corrected alpha" in cap
    assert "obfuscated - real" in cap                  # the sign convention
    assert "NOT the corrupted one" in cap              # slice definition
    assert "renormalised over" in cap.lower()          # the renormalisation
    matplotlib.pyplot.close(fig)


def test_bootstrap_resamples_solvers_not_rows():
    """A row-level bootstrap would shrink the interval; guard the width."""
    d = _synthetic(n_solvers=8, shift=None, seed=5)
    res = OB.paired_blame_shift(d, "innocent", n_boot=BOOT)
    widths = [r["hi"] - r["lo"] for r in res.rows]
    # With only 8 solvers the interval must be wide; a row-level bootstrap over
    # ~2000 rows would collapse it to near zero.
    assert min(widths) > 0.01


# ── figures ──────────────────────────────────────────────────────────────────
def test_figure_builds_and_returns_stats_and_caption():
    import matplotlib
    matplotlib.use("Agg")
    from viz.consistency import figures as F
    d = _synthetic(n_solvers=20, shift=0.1, seed=9)
    fig, st, cap = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    assert fig is not None and isinstance(cap, str) and len(cap) > 60
    assert set(st) >= {"rows", "raw_rows", "none", "guilty", "delta_is"}
    matplotlib.pyplot.close(fig)


def test_figure_survives_an_empty_frame():
    import matplotlib
    matplotlib.use("Agg")
    from viz.consistency import figures as F
    fig, st, cap = F.fig5_obfuscation_contrast(_frame([]).iloc[0:0], n_boot=BOOT)
    assert fig is not None and isinstance(cap, str)
    matplotlib.pyplot.close(fig)


def test_figure_fits_the_text_column_and_respects_the_font_floor():
    import matplotlib
    matplotlib.use("Agg")
    from viz.consistency import figures as F, style
    fig, _st, _cap = F.fig5_obfuscation_contrast(_synthetic(n_solvers=12),
                                                n_boot=BOOT)
    assert fig.get_size_inches()[0] <= style.TEXT_WIDTH_IN + 1e-9
    sizes = [t.get_fontsize() for ax in fig.axes for t in ax.texts]
    sizes += [t.get_fontsize() for ax in fig.axes
              for t in ax.get_xticklabels() + ax.get_yticklabels()]
    assert sizes and min(sizes) >= style.MIN_FONT_PT - 1e-9
    matplotlib.pyplot.close(fig)
