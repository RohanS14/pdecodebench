"""Tests for viz/consistency/stability.py — the Q4 item-level stability block."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.consistency import stability as S            # noqa: E402
from viz.consistency.constants import NONE            # noqa: E402


def _frame(rows):
    """rows: (system, condition, naming, order, draw, true, pred) -> tidy frame."""
    out = []
    for sysname, cond, naming, order, draw, true_o, pred in rows:
        out.append({
            "model": "M", "solver_id": sysname, "system": sysname,
            "condition": cond, "naming": naming, "order": order,
            "sample_idx": draw, "true_outlier": true_o, "pred_outlier": pred,
            "pred_agree": "no" if pred != NONE else "yes",
        })
    return pd.DataFrame(out)


def _both_namings(sysname, cond, true_o, pred_real, pred_obf, n_draws=3):
    rows = []
    for d in range(n_draws):
        rows.append((sysname, cond, S.REAL, "0", d, true_o, pred_real))
        rows.append((sysname, cond, S.OBF, "0", d, true_o, pred_obf))
    return rows


# ── categories are mutually exclusive and sum to the pair count ──────────────

def test_categories_partition_the_pairs():
    rows = []
    rows += _both_namings("A", "X_C", "C", "C", "C")      # same_correct
    rows += _both_namings("B", "X_C", "C", "T", "T")      # same_wrong
    rows += _both_namings("C", "X_C", "C", "T", "C")      # flip_to_correct
    rows += _both_namings("D", "X_C", "C", "C", "T")      # flip_to_wrong
    rows += _both_namings("E", "X_C", "C", "T", "D")      # wrong_to_wrong
    rows += _both_namings("F", "X_C", "C", "C", NONE)     # commit_asymmetry
    res = S.decompose(_frame(rows), "outlier", "modal")
    assert res["n"] == 6
    assert sum(res["counts"].values()) == res["n"]
    assert set(res["counts"]) <= set(S.CATEGORIES)
    assert res["counts"] == {
        S.SAME_CORRECT: 1, S.SAME_WRONG: 1, S.FLIP_TO_CORRECT: 1,
        S.FLIP_TO_WRONG: 1, S.WRONG_TO_WRONG: 1, S.COMMIT_ASYM: 1}


def test_every_category_is_reachable_and_distinct():
    seen = {S.classify(a, b, t, "outlier")
            for a, b, t in [("C", "C", "C"), ("T", "T", "C"), ("T", "C", "C"),
                            ("C", "T", "C"), ("T", "D", "C"), ("C", NONE, "C")]}
    assert seen == set(S.CATEGORIES)


# ── identical answers under both namings => nothing moved ───────────────────

def test_identical_answers_yield_full_stability_and_a_no_change_verdict():
    rows = []
    for i, (true_o, pred) in enumerate(
            [("C", "C"), ("T", "T"), ("D", "M"), ("M", NONE), (NONE, NONE)]):
        rows += _both_namings(f"S{i}", "X_C", true_o, pred, pred)
    t = _frame(rows)
    res = S.decompose(t, "outlier", "modal")
    assert res["n"] == 5
    stable = sum(res["counts"].get(c, 0) for c in S.STABLE)
    assert stable == res["n"], res["counts"]
    assert S.instability(res["counts"], res["n"]) == pytest.approx(0.0)

    # ... and the generated verdict must say names changed nothing. With the floor
    # also at zero the difference is exactly zero, so the clause is the null one.
    v = S.verdict(S.bars(t, "outlier"))
    assert "0% of items" in v
    assert "no more than sampling noise" in v
    assert "partly driven by identifiers" not in v


def test_identical_answers_leave_the_floor_at_zero_too():
    rows = []
    for i in range(4):
        rows += _both_namings(f"S{i}", "X_C", "C", "C", "C")
    floor = S.decompose(_frame(rows), "outlier", "draw", side_b=S.REAL, draw_b=1)
    assert S.instability(floor["counts"], floor["n"]) == pytest.approx(0.0)


# ── the floor is computed within one naming condition, never across ─────────

def test_noise_floor_never_reads_an_obfuscated_row():
    """Real draws agree with each other; obfuscated answers differ wildly.

    If the floor leaked across namings it would pick that up. It must not: the floor
    is movement from resampling alone.
    """
    rows = []
    for i in range(6):
        rows += [(f"S{i}", "X_C", S.REAL, "0", d, "C", "C") for d in range(3)]
        rows += [(f"S{i}", "X_C", S.OBF, "0", d, "C", "M") for d in range(3)]
    t = _frame(rows)
    floor = S.decompose(t, "outlier", "draw", side_b=S.REAL, draw_b=1)
    assert S.instability(floor["counts"], floor["n"]) == pytest.approx(0.0)
    manip = S.decompose(t, "outlier", "draw", side_b=S.OBF)
    assert S.instability(manip["counts"], manip["n"]) == pytest.approx(1.0)


def test_noise_floor_sees_within_condition_movement():
    rows = []
    for i in range(4):
        # draw 0 says C, draw 1 says T -- movement from resampling alone.
        rows += [(f"S{i}", "X_C", S.REAL, "0", 0, "C", "C"),
                 (f"S{i}", "X_C", S.REAL, "0", 1, "C", "T"),
                 (f"S{i}", "X_C", S.REAL, "0", 2, "C", "C"),
                 (f"S{i}", "X_C", S.OBF, "0", 0, "C", "C"),
                 (f"S{i}", "X_C", S.OBF, "0", 1, "C", "C"),
                 (f"S{i}", "X_C", S.OBF, "0", 2, "C", "C")]
    floor = S.decompose(_frame(rows), "outlier", "draw", side_b=S.REAL, draw_b=1)
    assert S.instability(floor["counts"], floor["n"]) == pytest.approx(1.0)


# ── pairing keeps presentation order out of the manipulation ────────────────

def test_pairs_are_matched_on_presentation_order():
    """Two orders per cell. A pairing that ignored order would compare order 0's
    real answer against order 1's obfuscated one and call the difference naming."""
    rows = []
    for order, pred in (("0", "C"), ("1", "M")):
        for d in range(3):
            rows.append(("S0", "X_C", S.REAL, order, d, "C", pred))
            rows.append(("S0", "X_C", S.OBF, order, d, "C", pred))
    res = S.decompose(_frame(rows), "outlier", "modal")
    assert res["n"] == 2
    # Both orders answered identically across namings, so nothing moved.
    assert S.instability(res["counts"], res["n"]) == pytest.approx(0.0)


# ── verdict level ───────────────────────────────────────────────────────────

def test_verdict_level_cannot_produce_wrong_to_wrong_or_asymmetry():
    rows = []
    rows += _both_namings("A", "X_C", "C", "C", NONE)   # flag -> noflag
    rows += _both_namings("B", "X_C", "C", "T", "D")    # flagged both times
    rows += _both_namings("C", "A0", NONE, NONE, "C")   # noflag -> flag
    res = S.decompose(_frame(rows), "verdict", "modal")
    assert res["counts"].get(S.WRONG_TO_WRONG, 0) == 0
    assert res["counts"].get(S.COMMIT_ASYM, 0) == 0
    assert sum(res["counts"].values()) == res["n"] == 3


def test_clean_items_score_saying_agree_as_correct():
    rows = _both_namings("A", "A0", NONE, NONE, NONE)
    res = S.decompose(_frame(rows), "outlier", "modal")
    assert res["counts"] == {S.SAME_CORRECT: 1}


# ── dropped pairs are counted, not silently discarded ───────────────────────

def test_unparseable_side_is_dropped_and_reported():
    rows = []
    rows += _both_namings("A", "X_C", "C", "C", "C")
    # Obfuscated side has no readable answer on any draw.
    rows += [("B", "X_C", S.REAL, "0", d, "C", "C") for d in range(3)]
    rows += [("B", "X_C", S.OBF, "0", d, "C", "") for d in range(3)]
    res = S.decompose(_frame(rows), "outlier", "modal")
    assert res["n"] == 1
    assert res["dropped"] == 1


def test_modal_takes_the_majority_not_the_first_draw():
    rows = [("A", "X_C", S.REAL, "0", 0, "C", "T"),
            ("A", "X_C", S.REAL, "0", 1, "C", "C"),
            ("A", "X_C", S.REAL, "0", 2, "C", "C"),
            ("A", "X_C", S.OBF, "0", 0, "C", "C"),
            ("A", "X_C", S.OBF, "0", 1, "C", "C"),
            ("A", "X_C", S.OBF, "0", 2, "C", "C")]
    res = S.decompose(_frame(rows), "outlier", "modal")
    assert res["counts"] == {S.SAME_CORRECT: 1}


# ── directional blame: shares, arrows, churn ────────────────────────────────

def _pair_rows(specs, n_draws=3):
    """specs: (system, condition, true, pred_real, pred_obf) -> tidy frame."""
    rows = []
    for sysname, cond, true_o, pr, po in specs:
        rows += _both_namings(sysname, cond, true_o, pr, po, n_draws=n_draws)
    return _frame(rows)


def test_blame_shares_sum_to_one_within_each_naming():
    t = _pair_rows([("A", "X_C", "C", "C", "T"), ("B", "X_C", "C", "T", NONE),
                    ("C", "X_D", "D", "D", "D"), ("D", "A0", NONE, NONE, "M"),
                    ("E", "X_M", "M", "M", "C")])
    pairs, _ = S.paired_answers(t, "outlier", "modal")
    sh = S.shares(pairs, S.BLAME_LEVELS)
    assert sum(sh["real"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(sh["obf"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sh["n"] == 5


def test_verdict_shares_also_sum_to_one():
    t = _pair_rows([("A", "X_C", "C", "C", NONE), ("B", "A0", NONE, NONE, "T")])
    pairs, _ = S.paired_answers(t, "verdict", "modal")
    sh = S.shares(pairs, S.VERDICT_LEVELS)
    assert sum(sh["real"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(sh["obf"].values()) == pytest.approx(1.0, abs=1e-6)


def test_unchanged_blame_gives_zero_length_arrows_and_zero_churn():
    t = _pair_rows([("A", "X_C", "C", "C", "C"), ("B", "X_D", "D", "T", "T"),
                    ("C", "A0", NONE, NONE, NONE), ("D", "X_M", "M", "M", "M")])
    pairs, _ = S.paired_answers(t, "outlier", "modal")
    rows, n = S.direction_rows(pairs, S.BLAME_LEVELS, n_boot=200)
    assert n == 4
    assert S.churn(pairs) == pytest.approx(0.0)
    for r in rows:                       # zero-length arrows
        assert r["delta"] == pytest.approx(0.0)
        assert not r["sig"]


def test_high_churn_with_identical_marginals_still_gives_zero_length_arrows():
    """The case the deleted figures existed to show, and the reason the churn
    sentence has to survive them: the distribution does not move at all while every
    single item changes its answer."""
    t = _pair_rows([("A", "X_C", "C", "C", "T"), ("B", "X_C", "C", "T", "C"),
                    ("C", "X_D", "D", "D", "M"), ("D", "X_M", "M", "M", "D")])
    pairs, _ = S.paired_answers(t, "outlier", "modal")
    rows, n = S.direction_rows(pairs, S.BLAME_LEVELS, n_boot=200)
    assert S.churn(pairs) == pytest.approx(1.0)
    for r in rows:
        assert r["delta"] == pytest.approx(0.0), (r["level"], r["delta"])
    line = S.churn_line(rows, S.churn(pairs), S.BLAME_LABELS)
    assert "at most 0.0pp" in line and "100% of individual items" in line
    v = S.direction_verdict(rows, S.churn(pairs), S.BLAME_LABELS)
    assert "not resolved" in v or "no direction that this sample size resolves" in v
    assert "100% of individual items" in v


def test_verdict_names_a_view_only_when_its_ci_excludes_zero():
    rows = [{"level": "T", "real": 0.5, "obf": 0.4, "real_n": 1, "obf_n": 1,
             "delta": -0.10, "lo": -0.15, "hi": -0.05, "sig": True},
            {"level": "C", "real": 0.2, "obf": 0.3, "real_n": 1, "obf_n": 1,
             "delta": 0.10, "lo": -0.02, "hi": 0.22, "sig": False}]
    v = S.direction_verdict(rows, 0.30, S.BLAME_LABELS)
    assert "trajectory" in v and "-10.0pp" in v
    assert "code" not in v               # n.s. row must not be presented as a finding

    allns = [dict(r, sig=False, lo=-0.2, hi=0.2) for r in rows]
    v2 = S.direction_verdict(allns, 0.30, S.BLAME_LABELS)
    assert "resolve" in v2
    for word in ("moves blame off trajectory (-10.0pp).",):
        assert word not in v2


def test_churn_line_leads_with_a_resolved_row_when_one_exists():
    rows = [{"level": "T", "real": 0.5, "obf": 0.4, "real_n": 1, "obf_n": 1,
             "delta": -0.044, "lo": -0.09, "hi": -0.01, "sig": True}]
    line = S.churn_line(rows, 0.29, S.BLAME_LABELS)
    assert line.startswith("Blame moves off trajectory by 4.4pp")
    assert "at most" not in line


def test_no_stacked_bar_is_rendered_in_the_section():
    """fig_stability (the deleted stacked chart) must not be reachable from
    build_block. The transition data survives only as a table."""
    import inspect
    src = inspect.getsource(S.build_block)
    assert "fig_stability" not in src
    assert "counts_table" in src          # ... as a table
    assert "fig_blame_direction" in src
