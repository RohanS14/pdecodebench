"""Hand-checked fixture tests for crossmodal.plotting.metrics.

The fixture is deliberately tiny and every expected number below was worked out by
hand from it, not read back off the implementation. The two definitions most likely
to drift are pinned hardest: blame-matrix row normalization, and the fact that
localization accuracy is conditional on correct detection.
"""
import math

import numpy as np
import pandas as pd
import pytest

from viz.consistency import metrics as M
from viz.consistency.constants import NONE, OUTLIER_LEVELS


def _row(cond, true_out, agree, pred_out, judge=False, naming="real",
         reasoning="on", model="m", pde="Heat"):
    return {
        "run_id": f"{cond}-{true_out}-{agree}-{pred_out}-{naming}-{reasoning}",
        "solver_id": "S00", "pde_class": pde, "numerical_method": "spectral",
        "condition": cond, "true_outlier": true_out, "naming": naming,
        "reasoning": reasoning, "model": model, "order": "C,T,D,M",
        "pred_agree": agree, "pred_outlier": pred_out, "pred_pde_class": pde,
        "pred_method": "spectral", "justification": "x", "judge_correct": judge,
    }


@pytest.fixture
def fx():
    """12 rows, hand-tabulated below.

      #  cond  true  agree pred   detected? det_ok? loc_eligible? loc_ok?
      1  A0    none  yes   none   no        YES     no            -
      2  A0    none  yes   none   no        YES     no            -
      3  A0    none  no    C      yes       no      no            -     (false alarm)
      4  A-C   C     no    C      yes       YES     yes           YES
      5  A-C   C     no    T      yes       YES     yes           no
      6  A-C   C     yes   none   no        no      no            -     (missed)
      7  A-T-swap T  no    T      yes       YES     yes           YES
      8  A-T-swap T  no    T      yes       YES     yes           YES
      9  A-D   D     no    C      yes       YES     yes           no
     10  A-D   D     no    C      yes       YES     yes           no
     11  A-D   D     yes   none   no        no      no            -     (missed)
     12  A-M   M     no    M      yes       YES     yes           YES
    """
    return pd.DataFrame([
        _row("A0", NONE, "yes", NONE),
        _row("A0", NONE, "yes", NONE),
        _row("A0", NONE, "no", "C"),
        _row("A-C", "C", "no", "C", judge=True),
        _row("A-C", "C", "no", "T"),
        _row("A-C", "C", "yes", NONE),
        _row("A-T-swap", "T", "no", "T", judge=True),
        _row("A-T-swap", "T", "no", "T"),
        _row("A-D", "D", "no", "C"),
        _row("A-D", "D", "no", "C"),
        _row("A-D", "D", "yes", NONE),
        _row("A-M", "M", "no", "M", judge=True),
    ])


# ── wilson ───────────────────────────────────────────────────────────────────
def test_wilson_zero_n_is_nan():
    lo, hi = M.wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_stays_inside_unit_interval_at_the_boundaries():
    # The reason Wilson is used at all: a Wald interval on 0/12 runs negative.
    for k, n in ((0, 12), (12, 12), (1, 200)):
        lo, hi = M.wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_known_value():
    # 5/10 -> textbook Wilson 95% interval.
    lo, hi = M.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2365, abs=1e-4)
    assert hi == pytest.approx(0.7635, abs=1e-4)


# ── detection ────────────────────────────────────────────────────────────────
def test_detection_overall_specificity_recall(fx):
    d = M.detection_metrics(fx).set_index("metric")
    # detection_correct on rows 1,2 (A0 said yes) + 4,5,7,8,9,10,12 = 9 of 12
    assert int(d.loc["overall", "k"]) == 9
    assert int(d.loc["overall", "n"]) == 12
    # specificity: A0 only, 2 of 3 correctly said "they agree"
    assert int(d.loc["specificity", "k"]) == 2
    assert int(d.loc["specificity", "n"]) == 3
    # recall: corrupted only, flagged on 7 of 9
    assert int(d.loc["recall", "k"]) == 7
    assert int(d.loc["recall", "n"]) == 9


def test_specificity_counts_saying_yes_not_saying_no(fx):
    """A0 rows are correct when they DON'T flag. Inverting this is the classic bug."""
    d = M.detection_metrics(fx).set_index("metric")
    assert d.loc["specificity", "rate"] == pytest.approx(2 / 3)


# ── localization: the conditional definition ─────────────────────────────────
def test_localization_is_conditional_on_correct_detection(fx):
    """Denominator is corrupted AND flagged (7), never all corrupted (9)."""
    loc = M.localization_accuracy(fx)
    assert int(loc.loc[0, "n"]) == 7          # rows 4,5,7,8,9,10,12
    assert int(loc.loc[0, "k"]) == 4          # rows 4,7,8,12
    assert loc.loc[0, "rate"] == pytest.approx(4 / 7)
    # The unconditional figure would be 4/9; guard against silently reverting.
    assert loc.loc[0, "rate"] != pytest.approx(4 / 9)


def test_localization_by_condition_excludes_missed_items(fx):
    loc = M.localization_accuracy(fx, by=["condition"]).set_index("condition")
    # A-C: rows 4,5 flagged (row 6 missed, excluded) -> 1/2
    assert int(loc.loc["A-C", "n"]) == 2
    assert loc.loc["A-C", "rate"] == pytest.approx(0.5)
    # A-D: rows 9,10 flagged, both blamed C -> 0/2
    assert int(loc.loc["A-D", "n"]) == 2
    assert loc.loc["A-D", "rate"] == pytest.approx(0.0)
    # A0 has no eligible rows at all, so it must be absent rather than 0.
    assert "A0" not in loc.index


# ── blame matrix normalization ───────────────────────────────────────────────
def test_blame_matrix_is_full_5x5_even_when_cells_are_empty(fx):
    bm = M.blame_matrix(fx)
    assert len(bm) == len(OUTLIER_LEVELS) ** 2 == 25
    assert set(bm["true_outlier"]) == set(OUTLIER_LEVELS)
    assert set(bm["pred_outlier"]) == set(OUTLIER_LEVELS)


def test_blame_matrix_rows_normalize_to_one(fx):
    bm = M.blame_matrix(fx)
    sums = bm.groupby("true_outlier")["row_frac"].sum()
    for level in OUTLIER_LEVELS:
        observed = bm[bm["true_outlier"].eq(level)]["n"].sum()
        if observed:
            assert sums[level] == pytest.approx(1.0)


def test_blame_matrix_hand_checked_cells(fx):
    bm = M.blame_matrix(fx).set_index(["true_outlier", "pred_outlier"])
    # true=C, 3 rows: pred C once, pred T once, pred none once
    assert int(bm.loc[("C", "C"), "n"]) == 1
    assert bm.loc[("C", "C"), "row_frac"] == pytest.approx(1 / 3)
    assert int(bm.loc[("C", NONE), "n"]) == 1
    # true=D, 3 rows: blamed C twice, none once -- never D itself
    assert int(bm.loc[("D", "C"), "n"]) == 2
    assert bm.loc[("D", "C"), "row_frac"] == pytest.approx(2 / 3)
    assert int(bm.loc[("D", "D"), "n"]) == 0
    # true=none, 3 rows: 2 none, 1 C
    assert bm.loc[("none", NONE), "row_frac"] == pytest.approx(2 / 3)


def test_blame_matrix_empty_row_is_nan_not_zero(fx):
    """A row with no observations must not read as 'never blamed'."""
    only_c = fx[fx["true_outlier"].eq("C")]
    bm = M.blame_matrix(only_c).set_index(["true_outlier", "pred_outlier"])
    assert np.isnan(bm.loc[("M", "M"), "row_frac"])
    assert int(bm.loc[("M", "M"), "n"]) == 0


# ── per-modality ─────────────────────────────────────────────────────────────
def test_per_modality_detection_and_false_blame(fx):
    pm = M.per_modality_rates(fx).set_index(["modality", "metric"])
    # C corrupted on rows 4,5,6; flagged on 4,5 -> 2/3
    assert pm.loc[("C", "detection_rate"), "rate"] == pytest.approx(2 / 3)
    # C falsely blamed: among the 9 rows where true != C, blamed C on rows 3,9,10
    assert int(pm.loc[("C", "false_blame_rate"), "n"]) == 9
    assert int(pm.loc[("C", "false_blame_rate"), "k"]) == 3
    # D is never blamed anywhere
    assert pm.loc[("D", "false_blame_rate"), "rate"] == pytest.approx(0.0)


def test_false_blame_denominator_includes_clean_rows(fx):
    """Blaming a view on an A0 item is a false blame; A0 must be in the denominator."""
    pm = M.per_modality_rates(fx).set_index(["modality", "metric"])
    n = int(pm.loc[("C", "false_blame_rate"), "n"])
    assert n == len(fx) - int(fx["true_outlier"].eq("C").sum())


# ── outcomes / judge / table ─────────────────────────────────────────────────
def test_outcome_breakdown_shares_sum_to_one(fx):
    br = M.outcome_breakdown(fx, by=["condition"])
    for _, g in br.groupby("condition"):
        assert g["share"].sum() == pytest.approx(1.0)


def test_outcome_labels_are_hand_checked(fx):
    d = M.prepare(fx)
    got = list(M.outcome_of(d))
    # "correct" is reached from BOTH halves of the design: agreeing about a clean
    # item (row 1) and flagging a corrupted one at the right view (row 4).
    assert got[0] == "correct"                    # A0, said agree
    assert got[2] == "false alarm"                # A0, flagged
    assert got[3] == "correct"                    # A-C, flagged + right view
    assert got[4] == "flagged, wrong view"        # A-C, flagged + wrong view
    assert got[5] == "missed the disagreement"    # A-C, never flagged


def test_judge_rate_by_condition(fx):
    j = M.judge_rate(fx).set_index("condition")
    assert j.loc["A-T-swap", "rate"] == pytest.approx(0.5)   # 1 of 2
    assert j.loc["A-D", "rate"] == pytest.approx(0.0)   # 0 of 3


def test_slicing_partitions_the_rows(fx):
    """Any `by` slice must account for every row, or a facet is silently dropped."""
    whole = M.detection_metrics(fx)
    whole_n = int(whole[whole["metric"].eq("overall")]["n"].iloc[0])
    sliced = M.detection_metrics(fx, by=["naming", "reasoning"])
    assert int(sliced[sliced["metric"].eq("overall")]["n"].sum()) == whole_n


def test_main_table_has_a_row_per_present_condition(fx):
    t = M.main_table(fx)
    # Conditions present in the fixture, in CONDITIONS order. The trajectory rungs
    # that the fixture does not exercise are absent rather than zero-filled.
    assert list(t["condition"]) == ["A0", "A-C", "A-T-swap", "A-D", "A-M"]
    assert int(t[t.condition.eq("A0")]["n"].iloc[0]) == 3
    # A0 localization is undefined, and must be NaN rather than 0.0
    assert np.isnan(t[t.condition.eq("A0")]["localization_rate"].iloc[0])


def test_prepare_does_not_mutate_input(fx):
    before = fx.copy()
    M.prepare(fx)
    pd.testing.assert_frame_equal(fx, before)


def test_empty_frame_returns_empty_not_raises():
    empty = pd.DataFrame(columns=list(fx_columns()))
    assert M.detection_metrics(empty).empty
    assert M.localization_accuracy(empty).empty
    assert M.blame_matrix(empty).empty
    assert M.outcome_breakdown(empty).empty


def fx_columns():
    from viz.consistency.constants import SCHEMA_COLUMNS
    return SCHEMA_COLUMNS


# ── trajectory corruption ladder ─────────────────────────────────────────────
# The trajectory is the one view whose corruption spans gross to subtle. Collapsing
# the rungs reports the mean of a wide range as if it were one difficulty.

def test_trajectory_conditions_all_map_to_the_T_outlier():
    from viz.consistency.constants import (CONDITION_OUTLIER, TRAJ_CONDITIONS,
                                           CONDITIONS, TRAJ_LEVELS)
    assert len(TRAJ_CONDITIONS) == len(TRAJ_LEVELS) == 4
    for c in TRAJ_CONDITIONS:
        assert c in CONDITIONS
        assert CONDITION_OUTLIER[c] == "T"


def test_traj_ladder_separates_the_rungs():
    from viz.consistency.synth import Effects, generate
    from viz.consistency.constants import TRAJ_LEVELS
    df = generate(Effects(n_solvers=8, models=("m",)))
    lad = M.traj_ladder(df)
    assert list(lad["traj_level"]) == list(TRAJ_LEVELS)   # ladder order, not alphabetical
    assert (lad["n"] > 0).all()


def test_traj_ladder_recovers_the_injected_ordering():
    """rand is grossest, exec subtlest; detection must fall across the rungs."""
    from viz.consistency.synth import Effects, generate
    df = generate(Effects(n_solvers=32, models=("m",)))
    rates = M.traj_ladder(df).set_index("traj_level")["rate"]
    assert rates["rand"] > rates["shuf"] > rates["swap"] > rates["exec"]


def test_blame_matrix_by_condition_gives_a_row_per_rung():
    from viz.consistency.synth import Effects, generate
    from viz.consistency.constants import CONDITIONS, OUTLIER_LEVELS
    df = generate(Effects(n_solvers=4, models=("m",)))
    bm = M.blame_matrix(df, row_by="condition")
    assert len(bm) == len(CONDITIONS) * len(OUTLIER_LEVELS) == 40
    sums = bm.groupby("condition")["row_frac"].sum()
    for c in CONDITIONS:
        assert sums[c] == pytest.approx(1.0)


def test_blame_matrix_default_still_collapses_to_5x5():
    """The lumped view stays available; splitting is a choice, not a replacement."""
    from viz.consistency.synth import Effects, generate
    df = generate(Effects(n_solvers=4, models=("m",)))
    assert len(M.blame_matrix(df)) == 25
