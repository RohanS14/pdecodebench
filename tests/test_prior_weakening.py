"""Tests for the prior-weakening section (Q4).

The hard rule is the point: a verdict line may not cite a quantity whose interval
includes zero. The previous build quoted "trajectory -4.3pp" from a row its own
figure drew as non-significant, which is how an underpowered exploratory row became
a stated finding.
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

from viz.consistency import claims as C            # noqa: E402
from viz.consistency import figures as F           # noqa: E402
from viz.consistency import prior_weakening as PW   # noqa: E402
from viz.consistency.constants import (MODALITIES, NAMING_LEVELS,  # noqa: E402
                                       NONE, SCHEMA_COLUMNS)

BOOT = 600


def _frame(rows):
    d = pd.DataFrame(rows)
    for c in SCHEMA_COLUMNS:
        if c not in d:
            d[c] = ""
    return d


def _synth(n_solvers=32, reps=8, seed=1,
           p_commit=(0.6, 0.6), p_correct=(0.6, 0.6), guilty_correct=None):
    """p_* are (real, obfuscated). guilty_correct overrides p_correct for C items."""
    rng = np.random.default_rng(seed)
    conds = {"A0": NONE, "A-T-rand": "T", "A-D": "D", "A-M": "M", "A-C": "C"}
    rows = []
    for s in range(n_solvers):
        for cond, true_out in conds.items():
            for ni, naming in enumerate(NAMING_LEVELS):
                for rep in range(reps):
                    if true_out == NONE:
                        agree = "yes" if rng.random() < 0.6 else "no"
                        pred = NONE if agree == "yes" else rng.choice(list(MODALITIES))
                    else:
                        commit = rng.random() < p_commit[ni]
                        agree = "no" if commit else "yes"
                        if not commit:
                            pred = NONE
                        else:
                            pc = (guilty_correct[ni] if (guilty_correct
                                                         and true_out == "C")
                                  else p_correct[ni])
                            pred = (true_out if rng.random() < pc
                                    else rng.choice([m for m in MODALITIES
                                                     if m != true_out]))
                    rows.append(dict(
                        run_id=f"{s}{cond}{naming}{rep}", solver_id=f"S{s:02d}",
                        condition=cond, true_outlier=true_out, naming=naming,
                        reasoning="on", model="m", pred_agree=agree,
                        pred_outlier=pred, justification=""))
    return _frame(rows)


# ── the identity ─────────────────────────────────────────────────────────────
def test_overall_accuracy_equals_detection_times_conditional():
    from viz.consistency.adapter import from_xmodal
    r = PW.analyse(_synth(), n_boot=BOOT)
    assert np.isfinite(r.identity_max_err)
    assert r.identity_max_err < 1e-6


# ── the hard rule ────────────────────────────────────────────────────────────
def _cited(text):
    return [round(float(x), 1) for x in re.findall(r"([+-]\d+\.\d)pp", text)]


def test_verdict_cites_no_quantity_whose_interval_includes_zero():
    """Fails on any build that quotes an n.s. row as a finding."""
    d = _synth(p_commit=(0.6, 0.55), p_correct=(0.6, 0.6))
    r = PW.analyse(d, n_boot=BOOT)
    v = C.q4(d, stats=r)
    allowed = set()
    for cst in [r.primary, r.interaction]:
        if cst is not None and cst.significant:
            for val in (cst.diff, cst.lo, cst.hi, cst.real, cst.obf):
                if np.isfinite(val):
                    allowed.add(round(100 * val, 1))
    # The primary's own estimate and interval may always be reported, because the
    # sentence states explicitly when it includes zero.
    for val in (r.primary.diff, r.primary.lo, r.primary.hi):
        allowed.add(round(100 * val, 1))
    for value in _cited(v.sentence):
        assert value in allowed, (
            f"verdict cites {value:+.1f}pp, which is not the primary test or a "
            f"resolved contrast")


def test_verdict_never_cites_an_exploratory_row():
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    v = C.q4(d, stats=r)
    for cst in r.per_outlier:
        if np.isfinite(cst.diff):
            assert f"{100 * cst.diff:+.1f}pp" not in v.sentence, (
                f"verdict quotes the exploratory {cst.name} row")


def test_unresolved_interaction_produces_the_fixed_closing_sentence():
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    if not r.interaction.significant:
        v = C.q4(d, stats=r)
        assert "not resolved at this sample size" in v.sentence


# ── recovery of injected effects ─────────────────────────────────────────────
def test_primary_reported_even_when_detection_moves_the_other_way():
    """Obfuscation improves conditional correctness but lowers commitment."""
    d = _synth(n_solvers=40, reps=12, seed=5,
               p_commit=(0.70, 0.50), p_correct=(0.55, 0.75))
    r = PW.analyse(d, n_boot=BOOT)
    assert r.primary.diff > 0, "primary should be positive"
    assert r.detection.diff < 0, "detection should be negative"
    v = C.q4(d, stats=r)
    assert f"{100 * r.primary.diff:+.1f}pp" in v.sentence
    assert f"{100 * r.detection.diff:+.1f}pp" not in v.sentence


def test_interaction_recovers_an_injected_asymmetry():
    """Obfuscation helps where code is innocent and hurts where code is guilty."""
    d = _synth(n_solvers=40, reps=14, seed=11,
               p_correct=(0.50, 0.75), guilty_correct=(0.75, 0.45))
    r = PW.analyse(d, n_boot=BOOT)
    assert r.interaction.diff > 0.15, r.interaction.diff
    assert r.interaction.significant


def test_no_injected_effect_leaves_the_primary_unresolved():
    r = PW.analyse(_synth(n_solvers=32, reps=10, seed=21), n_boot=BOOT)
    assert not r.primary.significant


# ── power reporting ──────────────────────────────────────────────────────────
def test_mde_is_finite_and_reported_in_the_caption():
    d = _synth()
    fig, r, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    assert np.isfinite(r.primary.mde) and r.primary.mde > 0
    assert all(np.isfinite(c.mde) for c in r.per_outlier)
    assert "80% power" in cap
    assert "pp" in cap


def test_caption_states_the_single_primary_test_and_no_correction():
    d = _synth()
    fig, r, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    assert "exploratory" in cap.lower()
    assert "80% power" in cap
    assert "details block" in cap


def test_trajectory_subtypes_are_pooled_not_crossed():
    """Four extra rows would multiply comparisons this design cannot power."""
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    names = {c.name for c in r.per_outlier}
    assert names <= set(MODALITIES)
    assert not any("rand" in n or "shuf" in n for n in names)


# ── figure guarantees ────────────────────────────────────────────────────────
def test_exploratory_rows_always_render_hollow():
    """Even a significant exploratory CI must not be drawn as a finding."""
    d = _synth(n_solvers=40, reps=14, seed=11,
               p_correct=(0.45, 0.80), guilty_correct=(0.80, 0.40))
    r = PW.analyse(d, n_boot=BOOT)
    assert any(np.isfinite(c.lo) and not (c.lo <= 0 <= c.hi)
               for c in r.per_outlier), "fixture produced no strong exploratory row"
    for c in r.per_outlier:
        assert not c.significant, "an exploratory row reported itself as significant"


def test_blame_share_is_absent_from_the_section():
    d = _synth()
    fig, r, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    assert "blame share" not in cap.lower()
    rows = dict(C._q4_rows(r))
    note = [k for k in rows if "blame share" in k]
    assert note, "the appendix note explaining why blame share is unused is missing"


def test_figure_builds_and_fits_the_column():
    """Two axes by design: levels on the left, the paired change on the right."""
    from viz.consistency import style
    d = _synth()
    fig, r, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    assert fig.get_size_inches()[0] <= style.TEXT_WIDTH_IN + 1e-9
    assert len(fig.axes) == 1
    lo, hi = fig.axes[0].get_xlim()
    assert 0 <= lo < hi <= 100, "the axis is not an absolute percentage scale"
    matplotlib.pyplot.close(fig)


# ── the figure must identify its own experiment ──────────────────────────────
def test_figure_names_both_conditions_on_its_face():
    """A delta-only figure cannot say which experiment it is. This one must."""
    from viz.consistency import claim_report as CR, style
    style.apply("light")
    d = _synth()
    fig, r, _ = F.fig7_prior_weakening(d, n_boot=BOOT)
    svg = CR._svg(fig)
    for needle in ("real names", "obfuscated"):
        assert needle in svg, f"figure never says {needle!r}"
    assert "named the right view" in svg, "the axis does not state the outcome"
    # "corrupted", not "broken": the figure calls a tampered-with view corrupted,
    # which is the term the rest of the consistency report uses. Experiment 1 keeps
    # "broken" for physically invalid CODE -- a different thing, and conflating the
    # two in one document is what this rename was for.
    assert "corrupted" in svg, "rows do not say what population they are a percent of"


def test_single_panel_with_two_marks_per_row():
    """One panel, one denominator. Two marks per row: real and obfuscated."""
    d = _synth()
    fig, r, _ = F.fig7_prior_weakening(d, n_boot=BOOT)
    assert len(fig.axes) == 1, "the figure grew a second panel again"
    ax = fig.axes[0]
    n_marks = sum(len(c.get_offsets()) for c in ax.collections)
    n_rows = len(ax.get_yticks())
    assert n_marks == 2 * n_rows, f"{n_marks} marks across {n_rows} rows"
    matplotlib.pyplot.close(fig)


def test_every_row_shares_one_denominator_semantics():
    """The failure that made the old figure unreadable: four denominators, one axis."""
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    shown = [r.overall] + list(r.per_outlier)
    for cst in shown:
        assert "items where" in cst.denom or "something was broken" in cst.denom \
            or "broken" in cst.denom, cst.denom
    # The decomposition rows have OTHER denominators and must not be on the figure.
    fig, _, _ = F.fig7_prior_weakening(d, n_boot=BOOT)
    assert len(fig.axes[0].get_yticks()) == len(shown)
    matplotlib.pyplot.close(fig)


def test_per_representation_rows_are_shown_and_marked_exploratory():
    """The question is asked of each representation, but never reported as tested."""
    d = _synth()
    from viz.consistency import style
    style.apply("light")
    fig, r, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    # headline + one row per representation, and nothing else
    assert len(fig.axes[0].get_yticks()) == 1 + len(MODALITIES)
    matplotlib.pyplot.close(fig)
    assert len(r.per_outlier) == len(MODALITIES)
    for c in r.per_outlier:
        assert c.exploratory
        assert not c.significant, "an exploratory row reported itself as significant"
    assert "exploratory" in cap.lower() and "80% power" in cap


def test_per_representation_uses_the_same_outcome_as_the_headline():
    """Both must be 'named the right view, of all broken items' for that view."""
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    for c in r.per_outlier:
        assert c.conditional is not None, "the conditional companion is missing"
        # overall <= conditional, since overall multiplies in the commit rate
        assert c.real <= c.conditional.real + 1e-9


def test_evidence_clause_matches_the_primary_sign_and_significance():
    pos = _synth(n_solvers=40, reps=14, seed=31, p_correct=(0.45, 0.80))
    r = PW.analyse(pos, n_boot=BOOT)
    assert r.primary.significant and r.primary.diff > 0
    text, ok = F.evidence_statement(r)
    assert ok and "MORE often right" in text
    assert "LESS often right" not in text and "includes zero" not in text

    neg = _synth(n_solvers=40, reps=14, seed=32, p_correct=(0.80, 0.45))
    rn = PW.analyse(neg, n_boot=BOOT)
    assert rn.primary.significant and rn.primary.diff < 0
    text_n, _ = F.evidence_statement(rn)
    assert "LESS often right" in text_n and "MORE often right" not in text_n


def test_null_primary_produces_the_power_clause():
    r = PW.analyse(_synth(n_solvers=32, reps=10, seed=41), n_boot=BOOT)
    text, ok = F.evidence_statement(r)
    assert not ok
    assert "includes zero" in text and "could only detect" in text


# ── every row must state its own denominator ─────────────────────────────────
def test_each_row_carries_its_denominator():
    """One shared percent axis over four different populations is only readable if
    each row says which population it is a percent OF."""
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    for cst in [r.overall, r.primary, r.detection, r.specificity] + r.per_outlier:
        assert cst.n_items > 0, f"{cst.name} has no recorded denominator size"
        assert cst.denom, f"{cst.name} does not say what its denominator is"


def test_denominators_are_the_ones_the_maths_actually_uses():
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    corrupted = int(d["true_outlier"].ne(NONE).sum())
    clean = int(d["true_outlier"].eq(NONE).sum())
    committed = int((d["true_outlier"].ne(NONE) & d["pred_agree"].eq("no")).sum())
    assert r.overall.n_items == corrupted
    assert r.detection.n_items == corrupted
    assert r.specificity.n_items == clean
    assert r.primary.n_items == committed
    for cst in r.per_outlier:
        assert cst.n_items == int(d["true_outlier"].eq(cst.name).sum())


def test_flag_rate_row_is_labelled_as_not_a_correctness_measure():
    """It is the one row on the figure that is not 'correctly identified'."""
    d = _synth()
    r = PW.analyse(d, n_boot=BOOT)
    assert "NOT correctness" in r.detection.denom
    from viz.consistency import claim_report as CR, style
    style.apply("light")
    fig, _, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    svg = CR._svg(fig)
    # It is not a correctness measure, so it must not share the figure's one axis.
    assert "FLAGGED something" not in svg
    assert "details block" in cap


def test_caption_warns_that_rates_are_solver_means_not_pooled():
    d = _synth()
    fig, r, cap = F.fig7_prior_weakening(d, n_boot=BOOT)
    matplotlib.pyplot.close(fig)
    assert "means over solvers" in cap
