"""Guarantees for the blame-share contrast, now an APPENDIX-ONLY analysis.

Blame share was demoted from Q4 because it measures where blame goes, not whether
it goes to the correct view: a shift between two wrong views moves it and leaves
correctness unchanged. The figure and its stats are still built, so the sign
convention, renormalisation and encoding guarantees are still tested here; the
verdict-line consistency rule moved to test_prior_weakening.py, which enforces it
against the outcome that actually answers the question.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.consistency import claims as C          # noqa: E402
from viz.consistency import figures as F         # noqa: E402
from viz.consistency import obfuscation as OB    # noqa: E402
from viz.consistency.constants import (DELTA_IS, MODALITIES,  # noqa: E402
                                       NAMING_LEVELS, NONE, SCHEMA_COLUMNS)

BOOT = 600


def _frame(rows):
    d = pd.DataFrame(rows)
    for c in SCHEMA_COLUMNS:
        if c not in d:
            d[c] = ""
    return d


def _synth(n_solvers=24, obf_toward=None, amount=0.0, seed=2):
    """Paired frame; `obf_toward` gains `amount` of blame share under obfuscation."""
    rng = np.random.default_rng(seed)
    rows = []
    conds = {"A0": NONE, "A-T-rand": "T", "A-D": "D", "A-M": "M", "A-C": "C"}
    for s in range(n_solvers):
        for cond, true_out in conds.items():
            for naming in NAMING_LEVELS:
                for rep in range(10):
                    p = {m: 0.2 for m in MODALITIES}
                    p[NONE] = 0.2
                    if obf_toward and naming == NAMING_LEVELS[1]:
                        for k in p:
                            p[k] -= amount / len(p)
                        p[obf_toward] += amount
                    cats = list(p)
                    w = np.array([max(p[k], 1e-9) for k in cats])
                    rows.append(dict(
                        run_id=f"{s}{cond}{naming}{rep}", solver_id=f"S{s:02d}",
                        condition=cond, true_outlier=true_out, naming=naming,
                        reasoning="on", model="m",
                        pred_agree="no", pred_outlier=rng.choice(cats, p=w / w.sum()),
                        justification=""))
    return _frame(rows)


def _numbers(text):
    """Every signed pp value in a rendered verdict line, to 1dp."""
    return [round(float(x), 1) for x in re.findall(r"([+-]\d+\.\d)pp", text)]


# ── the regression test: verdict numbers must exist in the figure's stats ────
def test_stats_and_caption_are_generated_from_one_object():
    """The caption may quote only numbers present in the stats it was built from."""
    d = _synth(obf_toward="C", amount=0.10)
    fig, st, cap = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    pool = set()
    for r in st["rows"] + st["raw_rows"] + ([st["none"]] if st["none"] else []):
        for k in ("diff", "lo", "hi"):
            pool.add(round(100 * r[k], 1))
    for value in _numbers(cap):
        assert value in pool, f"caption quotes {value:+.1f}pp, absent from its stats"


def test_verdict_and_figure_agree_on_sign_for_a_known_shift():
    """Obfuscated blame on code EXCEEDS real; delta must be positive in both."""
    d = _synth(obf_toward="C", amount=0.14)
    fig, st, _ = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    code = next(r for r in st["rows"] if r["category"] == "C")
    assert code["diff"] > 0, "figure has the sign backwards"
    assert code["diff"] == pytest.approx(code["obf"] - code["real"], abs=1e-9)


def test_sign_convention_is_defined_once_and_used():
    assert DELTA_IS == "obfuscated - real"
    st = F.obfuscation_stats(_synth(), n_boot=BOOT)
    assert st["delta_is"] == DELTA_IS


def test_delta_matches_obfuscated_minus_real_numerically():
    st = F.obfuscation_stats(_synth(obf_toward="T", amount=0.10), n_boot=BOOT)
    for r in st["rows"]:
        assert r["diff"] == pytest.approx(r["obf"] - r["real"], abs=1e-9)


# ── renormalisation ──────────────────────────────────────────────────────────
def test_renormalised_modality_shares_sum_to_one():
    st = F.obfuscation_stats(_synth(), n_boot=BOOT)
    assert len(st["rows"]) == len(MODALITIES)
    for key in ("real", "obf"):
        assert abs(sum(r[key] for r in st["rows"]) - 1.0) < 1e-6


def test_raw_five_way_shares_are_kept_for_the_details_block():
    st = F.obfuscation_stats(_synth(), n_boot=BOOT)
    cats = {r["category"] for r in st["raw_rows"]}
    assert NONE in cats and cats >= set(MODALITIES)
    for key in ("real", "obf"):
        assert abs(sum(r[key] for r in st["raw_rows"]) - 1.0) < 1e-6


def test_none_is_promoted_out_of_the_modality_rows():
    st = F.obfuscation_stats(_synth(), n_boot=BOOT)
    assert NONE not in {r["category"] for r in st["rows"]}
    assert st["none"] is not None
    text, ok = F.none_statement(st)
    assert ok and "all four representations agree" in text


# ── significance encoding ────────────────────────────────────────────────────
def test_significance_uses_the_corrected_interval_not_the_raw_one():
    st = F.obfuscation_stats(_synth(obf_toward="C", amount=0.05), n_boot=BOOT)
    for r in st["rows"]:
        by_corrected = not (r["clo"] <= 0.0 <= r["chi"])
        assert r["significant"] == (by_corrected and not r["thin"]), \
            "significance was judged on the uncorrected 95% interval"


def test_null_fixture_yields_no_significant_rows_and_says_so():
    """Seed 3 is verified clean. Seeds 17 and 18 are NOT, and that is expected --
    see the calibration test below: at a corrected alpha of 0.0125 across four rows
    a false positive should appear in roughly 5% of runs, and it does."""
    d = _synth(obf_toward=None, amount=0.0, n_solvers=30, seed=3)
    fig, st, cap = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    assert not any(r["significant"] for r in st["rows"])
    assert "No modality's share change survives correction." in cap


@pytest.mark.slow
def test_bootstrap_interval_is_calibrated_under_a_true_null():
    """The property the single-seed test above can only sample.

    Point estimates must be unbiased and the intervals must cover the null at close
    to their nominal rate. A miscalibrated interval would make every claim in this
    section unreliable, and would not be visible from any one fixture.
    """
    diffs, cover95, cover_corr, n = [], 0, 0, 0
    for seed in range(40):
        r = OB.paired_blame_shift(
            _synth(obf_toward=None, amount=0.0, n_solvers=30, seed=seed),
            "innocent", n_boot=300, renormalise=True)
        for row in r.rows:
            n += 1
            diffs.append(row["diff"])
            cover95 += (row["lo"] <= 0 <= row["hi"])
            cover_corr += (row["clo"] <= 0 <= row["chi"])
    mean = float(np.mean(diffs))
    assert abs(mean) < 0.005, f"point estimate is biased: {100 * mean:+.3f}pp"
    assert 0.90 <= cover95 / n <= 0.99, f"95% coverage is {cover95 / n:.3f}"
    assert cover_corr / n >= 0.955, f"corrected coverage is {cover_corr / n:.3f}"


# ── layout guarantees ────────────────────────────────────────────────────────
def test_guilty_row_shares_the_axis_with_the_modality_rows():
    d = _synth(obf_toward="C", amount=0.08)
    fig, st, _ = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    assert len(fig.axes) == 1, "the guilty row was given its own axis"
    matplotlib.pyplot.close(fig)


def test_no_bars_are_drawn():
    """A bar from zero double-encodes a signed difference."""
    d = _synth(obf_toward="C", amount=0.08)
    fig, st, _ = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    from matplotlib.patches import Rectangle
    bars = [p for ax in fig.axes for p in ax.patches
            if isinstance(p, Rectangle) and p.get_width() > 0
            and p.get_height() < 1.0]
    assert not bars, "the contrast figure is drawing bars"
    matplotlib.pyplot.close(fig)


def test_symmetric_limits_rounded_to_an_even_number():
    d = _synth(obf_toward="C", amount=0.08)
    fig, st, _ = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    lo, hi = fig.axes[0].get_xlim()
    assert lo == pytest.approx(-hi), "x-limits are not symmetric"
    assert abs(hi % 2.0) < 1e-9, f"limit {hi} is not rounded to an even number"
    matplotlib.pyplot.close(fig)


def test_greyscale_separates_significant_rows_by_fill_alone():
    """Filled vs hollow must survive with every colour removed."""
    d = _synth(obf_toward="C", amount=0.14, n_solvers=30)
    from viz.consistency import style
    style.apply("light")
    fig, st, _ = F.fig5_obfuscation_contrast(d, n_boot=BOOT)
    ax = fig.axes[0]
    coll = [c for c in ax.collections if len(c.get_offsets())]
    filled, hollow = 0, 0
    for c in coll:
        lw = np.atleast_1d(c.get_linewidths())[0]
        (hollow := hollow + 1) if lw > 0 else (filled := filled + 1)
    assert filled and hollow, (
        "every row rendered the same way; significance is not encoded by fill")
    matplotlib.pyplot.close(fig)
