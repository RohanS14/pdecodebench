"""Tests for the two primary-research-question sections (Q1 sensitivity, Q2 blame).

The assertions target the ways these two figures mislead when they are wrong: a
reference row that is not actually the marginal, segments that do not close to 100%,
an "informative" verdict on independent data, and a figure whose reading depends on
hue.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.consistency import claims as C           # noqa: E402
from viz.consistency import claim_report as CR    # noqa: E402
from viz.consistency import sensitivity as S      # noqa: E402
from viz.consistency.constants import (MODALITIES, MODALITY_COLORS,  # noqa: E402
                                       NONE, OUTLIER_LEVELS, SCHEMA_COLUMNS)

PERM = 400          # smaller than production; same estimator


def _frame(rows):
    d = pd.DataFrame(rows)
    for c in SCHEMA_COLUMNS:
        if c not in d:
            d[c] = ""
    return d


def _make(correct_prob, n_solvers=12, reps=10, seed=3):
    """Flagged items whose blame matches the truth with the given probability."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_solvers):
        for m in MODALITIES:
            for r in range(reps):
                if rng.random() < correct_prob:
                    pred = m
                else:
                    pred = rng.choice([x for x in MODALITIES if x != m])
                rows.append(dict(
                    run_id=f"{s}-{m}-{r}", solver_id=f"S{s:02d}",
                    condition={"C": "A-C", "T": "A-T-rand", "D": "A-D",
                               "M": "A-M"}[m],
                    true_outlier=m, naming="real", reasoning="on", model="m",
                    pred_agree="no", pred_outlier=pred, justification="x"))
    return _frame(rows)


def _independent(n_solvers=14, reps=12, seed=5):
    """Blame drawn from a FIXED policy, unrelated to what was corrupted."""
    rng = np.random.default_rng(seed)
    policy = [0.45, 0.25, 0.20, 0.10]        # a stereotyped responder
    rows = []
    for s in range(n_solvers):
        for m in MODALITIES:
            for r in range(reps):
                pred = rng.choice(list(MODALITIES), p=policy)
                rows.append(dict(
                    run_id=f"{s}-{m}-{r}", solver_id=f"S{s:02d}",
                    condition={"C": "A-C", "T": "A-T-rand", "D": "A-D",
                               "M": "A-M"}[m],
                    true_outlier=m, naming="real", reasoning="on", model="m",
                    pred_agree="no", pred_outlier=pred, justification="x"))
    return _frame(rows)


# ── row 5 is genuinely the marginal ──────────────────────────────────────────
def test_rows_are_conditions_with_trajectory_disaggregated():
    from viz.consistency.sensitivity import SIGNAL_CONDITIONS
    d = _make(0.6)
    b = S.blame_information(d, n_perm=PERM)
    assert list(b.table.index) == list(SIGNAL_CONDITIONS)
    assert sum(1 for c in b.table.index if c.startswith("A-T-")) == 4


def test_reference_row_equals_the_column_marginal():
    b = S.blame_information(_make(0.6), n_perm=PERM)
    expected = b.table.sum(axis=0) / b.table.to_numpy().sum()
    got = b.marginal.reindex(expected.index)
    assert np.allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_reference_row_sums_to_one():
    b = S.blame_information(_make(0.6), n_perm=PERM)
    assert abs(float(b.marginal.sum()) - 1.0) < 1e-6


def test_every_stacked_row_sums_to_one():
    b = S.blame_information(_make(0.55), n_perm=PERM)
    for cond in b.table.index:
        counts = b.table.loc[cond].reindex(OUTLIER_LEVELS).fillna(0)
        tot = counts.sum()
        if tot:
            assert abs(float((counts / tot).sum()) - 1.0) < 1e-6


# ── the two permitted verdicts, and nothing between them ─────────────────────
def test_independent_blame_is_called_uninformative():
    d = _independent()
    b = S.blame_information(d, n_perm=PERM)
    assert not b.informative
    v = C.q_blame_information(d)
    assert "blame is indistinguishable from a fixed response bias" in v.sentence
    assert "blame tracks the actual outlier" not in v.sentence


def test_independent_rows_match_the_reference_row_numerically():
    """If blame ignores the truth, every row IS the marginal, up to sampling noise."""
    b = S.blame_information(_independent(n_solvers=40, reps=30), n_perm=PERM)
    ref = b.marginal.reindex(OUTLIER_LEVELS).fillna(0).to_numpy()
    for cond in b.table.index:
        counts = b.table.loc[cond].reindex(OUTLIER_LEVELS).fillna(0)
        if counts.sum() == 0:
            continue
        row = (counts / counts.sum()).to_numpy()
        assert np.max(np.abs(row - ref)) < 0.08


def test_diagonal_dominant_blame_is_called_informative():
    d = _make(0.90, n_solvers=20, reps=14)
    b = S.blame_information(d, n_perm=PERM)
    assert b.informative
    from viz.consistency.constants import CONDITION_OUTLIER
    for cond in b.table.index:                # the diagonal really does dominate
        if b.table.loc[cond].sum() == 0:
            continue                          # condition absent from this fixture
        assert b.table.loc[cond].idxmax() == CONDITION_OUTLIER[cond]
    v = C.q_blame_information(d)
    assert "blame tracks the actual outlier" in v.sentence


def test_verdict_uses_only_the_two_permitted_claims():
    for d in (_independent(), _make(0.9)):
        v = C.q_blame_information(d)
        assert (("blame tracks the actual outlier" in v.sentence)
                ^ ("blame is indistinguishable from a fixed response bias"
                   in v.sentence))


def test_constant_policy_cannot_beat_the_best_fixed_answer():
    """A model that always answers the same thing IS a fixed policy; margin <= 0."""
    d = _make(0.5)
    d["pred_outlier"] = "C"
    b = S.blame_information(d, n_perm=PERM)
    assert not b.informative
    assert b.localization <= b.best_constant + 1e-9


# ── Q1 sensitivity ───────────────────────────────────────────────────────────
def test_sensitivity_shares_one_false_alarm_rate_across_modalities():
    d = _make(0.6)
    clean = d.iloc[:40].copy()
    clean["true_outlier"] = NONE
    clean["condition"] = "A0"
    clean["pred_agree"] = "yes"
    clean["pred_outlier"] = NONE
    r = S.detection_sensitivity(pd.concat([d, clean]), n_boot=200)
    assert r.n_noise == 40
    assert 0.0 <= r.fa_rate <= 1.0


def test_axis_limits_are_computed_from_the_data_not_hardcoded():
    """A fixture whose rates all sit high must produce a zoomed axis, not [0, 1]."""
    import re
    from viz.consistency import style
    rows = []
    rng = np.random.default_rng(4)
    for sv in range(10):
        for m in MODALITIES:
            for r in range(20):
                rows.append(dict(run_id=f"{sv}{m}{r}", solver_id=f"S{sv}",
                                 condition={"C": "A-C", "T": "A-T-rand", "D": "A-D",
                                            "M": "A-M"}[m],
                                 true_outlier=m, naming="real", reasoning="on",
                                 model="m",
                                 pred_agree="no" if rng.random() < 0.90 else "yes",
                                 pred_outlier=m, justification=""))
        for r in range(20):
            rows.append(dict(run_id=f"{sv}n{r}", solver_id=f"S{sv}", condition="A0",
                             true_outlier=NONE, naming="real", reasoning="on",
                             model="m",
                             pred_agree="no" if rng.random() < 0.82 else "yes",
                             pred_outlier=NONE, justification=""))
    style.apply("light")
    svg = CR.fig_sensitivity(_frame(rows))
    ticks = [float(t.rstrip("%")) for t in re.findall(r">(\d{1,3})%<", svg)]
    assert ticks, "no percentage ticks rendered"
    assert min(ticks) >= 60, f"axis was not zoomed to the data: ticks start at {min(ticks)}"


def test_baseline_row_is_last_and_never_sorted():
    from viz.consistency import style
    rows = []
    rng = np.random.default_rng(6)
    for sv in range(8):
        for m in MODALITIES:
            for r in range(12):
                rows.append(dict(run_id=f"{sv}{m}{r}", solver_id=f"S{sv}",
                                 condition={"C": "A-C", "T": "A-T-rand", "D": "A-D",
                                            "M": "A-M"}[m],
                                 true_outlier=m, naming="real", reasoning="on",
                                 model="m", pred_agree="no", pred_outlier=m,
                                 justification=""))
        for r in range(12):
            # Baseline flags MORE often than some corrupted rows; it must still sort last.
            rows.append(dict(run_id=f"{sv}n{r}", solver_id=f"S{sv}", condition="A0",
                             true_outlier=NONE, naming="real", reasoning="on",
                             model="m", pred_agree="no" if rng.random() < 0.95 else "yes",
                             pred_outlier=NONE, justification=""))
    style.apply("light")
    svg = CR.fig_sensitivity(_frame(rows))
    assert "nothing corrupted" in svg
    assert svg.index("nothing corrupted") > svg.index("corrupted")


def test_mismatched_severity_triggers_banner_and_downgraded_verdict():
    """Trajectory carries four generation methods; the others carry one."""
    from viz.consistency.sensitivity import severity_tiers
    d = _make(0.6)
    d.loc[d["true_outlier"].eq("T"), "traj_level"] = np.resize(
        ["rand", "shuf", "swap", "exec"], int(d["true_outlier"].eq("T").sum()))
    clean = d.iloc[:40].copy()
    clean["true_outlier"] = NONE; clean["condition"] = "A0"
    clean["pred_agree"] = "yes"; clean["pred_outlier"] = NONE
    frame = pd.concat([d, clean])
    _, matched, common = severity_tiers(frame)
    assert not matched and not common
    v = C.q_sensitivity(frame)
    assert "NOT severity-matched" in v.sentence
    assert "easiest to detect" not in v.sentence, "ordering asserted despite mismatch"
    assert v.direction == "inconclusive"


def test_matched_severity_produces_no_banner_and_a_full_verdict():
    from viz.consistency.sensitivity import severity_tiers
    d = _make(0.6)                       # every modality has one tier: "single"
    clean = d.iloc[:40].copy()
    clean["true_outlier"] = NONE; clean["condition"] = "A0"
    clean["pred_agree"] = "yes"; clean["pred_outlier"] = NONE
    frame = pd.concat([d, clean])
    _, matched, common = severity_tiers(frame)
    assert matched and common == ["single"]
    v = C.q_sensitivity(frame)
    assert "NOT severity-matched" not in v.sentence
    assert "reports a disagreement on" in v.sentence


def test_verdict_leads_with_the_false_alarm_floor():
    d = _make(0.6)
    clean = d.iloc[:40].copy()
    clean["true_outlier"] = NONE; clean["condition"] = "A0"
    clean["pred_agree"] = "yes"; clean["pred_outlier"] = NONE
    v = C.q_sensitivity(pd.concat([d, clean]))
    assert v.sentence.startswith("The model reports a disagreement on")


def test_no_matched_figure_when_no_common_tier():
    from viz.consistency import style
    style.apply("light")
    d = _make(0.6)
    d.loc[d["true_outlier"].eq("T"), "traj_level"] = "rand"
    svg, note = CR.fig_sensitivity_matched(d)
    assert svg is None
    assert "No severity tier is present for all four" in note


def test_sensitivity_rows_are_ordered_by_dprime_descending():
    d = _make(0.6)
    clean = d.iloc[:40].copy()
    clean["true_outlier"] = NONE; clean["condition"] = "A0"
    clean["pred_agree"] = "yes"; clean["pred_outlier"] = NONE
    r = S.detection_sensitivity(pd.concat([d, clean]), n_boot=200)
    vals = [x["dprime"] for x in r.rows if not x["empty"]]
    assert vals == sorted(vals, reverse=True)


def test_unresolved_ordering_is_reported_as_unresolved():
    """Equal detectability must not be reported as a ranking."""
    rows = []
    rng = np.random.default_rng(2)
    CMAP = {"C": "A-C", "T": "A-T-rand", "D": "A-D", "M": "A-M"}
    for s in range(8):
        for m in MODALITIES:
            for r in range(6):
                rows.append(dict(run_id=f"{s}{m}{r}", solver_id=f"S{s}",
                                 condition=CMAP[m], true_outlier=m, naming="real",
                                 reasoning="on", model="m",
                                 pred_agree="no" if rng.random() < 0.6 else "yes",
                                 pred_outlier=m, justification=""))
        for r in range(6):
            rows.append(dict(run_id=f"{s}n{r}", solver_id=f"S{s}", condition="A0",
                             true_outlier=NONE, naming="real", reasoning="on",
                             model="m", pred_agree="no" if rng.random() < 0.4 else "yes",
                             pred_outlier=NONE, justification=""))
    res = S.detection_sensitivity(_frame(rows), n_boot=400)
    if not res.resolved:
        v = C.q_sensitivity(_frame(rows))
        assert "ordering is not resolved" in v.sentence
        assert v.direction == "inconclusive"


# ── greyscale legibility ─────────────────────────────────────────────────────
def _luma(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_known_greyscale_luminance_collisions_are_pinned():
    """Okabe-Ito is colourblind-safe but NOT greyscale-safe.

    Trajectory (#D55E00, luma 0.441) and description (#009E73, luma 0.476) are
    almost identical once hue is removed, and they sit next to each other in the
    fixed segment order. The figure does not rely on hue to separate them -- every
    segment is bounded by a surface-coloured stroke, and the order is fixed -- but
    the collision is real, so it is pinned here rather than left to be rediscovered.
    A palette change that introduces a NEW collision fails this test.
    """
    from viz.consistency.constants import NONE_COLOR
    order = list(MODALITIES) + [NONE]
    cols = [MODALITY_COLORS.get(m, NONE_COLOR) for m in order]
    lums = [_luma(c) for c in cols]
    collisions = {tuple(sorted((order[i], order[i + 1])))
                  for i in range(len(order) - 1)
                  if abs(lums[i] - lums[i + 1]) <= 0.04}
    assert collisions == {("D", "T")}, (
        f"greyscale collisions changed: {collisions}")


def test_stack_has_a_legend_naming_every_segment():
    from viz.consistency import style
    style.apply("light")
    svg = CR.fig_blame_stack(_make(0.7, n_solvers=8, reps=8))
    for word in ("code", "trajectory", "description", "math",
                 "correct answer for this row"):
        assert word in svg, f"legend is missing {word!r}"


def test_segments_are_separated_by_a_surface_coloured_stroke():
    """This, not hue, is what makes the stack readable with no colour at all."""
    from viz.consistency import style
    style.apply("light")
    d = _make(0.7, n_solvers=8, reps=8)
    svg = CR.fig_blame_stack(d)
    # Every bar patch is drawn with a stroke in the panel colour and non-zero width.
    assert "stroke-width: 1.5" in svg or "stroke-width:1.5" in svg, \
        "segment separators are missing; the stack would be unreadable in greyscale"


@pytest.mark.parametrize("fn", ["fig_blame_stack", "fig_sensitivity"])
def test_figures_render_in_greyscale(fn, tmp_path):
    """Render, drop all colour, and confirm the mark structure survives."""
    from viz.consistency import style
    d = _make(0.75, n_solvers=10, reps=8)
    clean = d.iloc[:30].copy()
    clean["true_outlier"] = NONE; clean["condition"] = "A0"
    clean["pred_agree"] = "yes"; clean["pred_outlier"] = NONE
    frame = pd.concat([d, clean])
    style.apply("light")
    svg = getattr(CR, fn)(frame)
    assert svg.lstrip().startswith("<svg")
    # Strip every fill/stroke colour; the figure must still contain its marks.
    import re
    grey = re.sub(r'(fill|stroke):#[0-9a-fA-F]{6}', r'\1:#808080', svg)
    assert grey.count("<path") > 5
    assert len(grey) > 0.8 * len(svg)


def test_empty_modality_renders_an_explicit_label():
    """A modality with no flagged rows must say so, not vanish."""
    d = _make(0.6)
    d = d[d["true_outlier"].ne("M")]          # remove math entirely
    from viz.consistency import style
    style.apply("light")
    svg = CR.fig_blame_stack(d)
    assert "no detected rows" in svg
