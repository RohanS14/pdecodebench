"""
Unit tests for the Experiment 2 Part II components that were previously
smoke-tested only: probe/cross_modal.py, probe/geometry_battery.py and
datagen/extract_trajectories.py.

Runs locally, no GPU, no model.

The emphasis is on the places where a silent wrong answer is possible:
  * retrieval scored within-group must use 1/group_size as chance, not 1/N
  * a blow-up must be a CATEGORY, not an unbounded number that would dominate
    any correlation it enters
  * the trajectory text must not name the PDE class or numerical method it is
    being used to test for
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
sys.path.insert(0, os.path.join(ROOT, "cross_modal_consistency", "datagen"))

from cross_modal import align, retrieval, unit, word_multiset  # noqa: E402
from geometry_battery import (  # noqa: E402
    logo_acc, participation_ratio, twonn_id,
)
from extract_trajectories import (  # noqa: E402
    VALID_OF, divergence, render_text, state_field,
)


# =============================== cross_modal =================================

def _perfect(n):
    return np.eye(n) * 2.0 - 1.0          # diagonal strictly highest


def test_retrieval_perfect_match_is_top1_one():
    r = retrieval(_perfect(10))
    assert r["top1"] == 1.0
    assert r["mrr"] == 1.0
    assert r["median_rank"] == 1.0


def test_retrieval_worst_case_ranks_last():
    """Diagonal strictly lowest -> the correct answer ranks last every time."""
    M = np.ones((6, 6))
    np.fill_diagonal(M, -1.0)
    r = retrieval(M)
    assert r["top1"] == 0.0
    assert r["median_rank"] == 6.0


def test_retrieval_global_chance_is_one_over_n():
    assert retrieval(np.zeros((8, 8)))["chance_top1"] == pytest.approx(1 / 8)


def test_retrieval_within_group_chance_is_one_over_group_size():
    """
    THE test for the within-class rule. With 4 groups of 8, chance must be 1/8,
    not 1/32. Reporting 1/32 here would make a category-level hit look like an
    instance-level one, which is the exact confusion §15 exists to prevent.
    """
    groups = np.repeat(np.arange(4), 8)
    r = retrieval(np.zeros((32, 32)), groups)
    assert r["chance_top1"] == pytest.approx(1 / 8)
    assert retrieval(np.zeros((32, 32)))["chance_top1"] == pytest.approx(1 / 32)


def test_retrieval_within_group_ignores_out_of_group_competitors():
    """A strong out-of-group distractor must not be able to beat the answer."""
    n = 6
    groups = np.array([0, 0, 0, 1, 1, 1])
    M = np.zeros((n, n))
    np.fill_diagonal(M, 1.0)
    M[0, 4] = 5.0                       # huge score, but group 1 -> not a candidate
    assert retrieval(M, groups)["top1"] == 1.0
    assert retrieval(M)["top1"] < 1.0   # globally it does steal the top slot


def test_retrieval_random_is_near_chance():
    rng = np.random.default_rng(0)
    accs = [retrieval(rng.standard_normal((20, 20)))["top1"] for _ in range(20)]
    assert abs(np.mean(accs) - 1 / 20) < 0.06


def test_word_multiset_splits_on_non_alphanumeric():
    m = word_multiset("du/dt = alpha * d2u_dx2")
    assert m["du"] == 1 and m["dt"] == 1 and m["alpha"] == 1
    assert "=" not in m and "*" not in m


def test_word_multiset_is_case_insensitive():
    assert word_multiset("Alpha ALPHA alpha")["alpha"] == 3


def test_align_returns_shared_keys_in_sorted_order():
    a = np.array(["c", "a", "b"])
    b = np.array(["b", "c", "d"])
    ia, ib, common = align(a, b)
    assert common == ["b", "c"]
    assert list(a[ia]) == common and list(b[ib]) == common


def test_unit_rows_have_norm_one_and_zero_is_safe():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    U = unit(X)
    assert U[0] @ U[0] == pytest.approx(1.0)
    assert not np.isnan(U).any()


# ============================= geometry_battery ==============================

def test_participation_ratio_of_rank_one_data_is_one():
    d = np.random.default_rng(0).standard_normal(32)
    X = np.outer(np.linspace(-1, 1, 40), d)
    assert participation_ratio(X) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_recovers_k_equal_variance_directions():
    rng = np.random.default_rng(1)
    k, D, n = 5, 64, 600
    basis = np.linalg.qr(rng.standard_normal((D, k)))[0]
    X = rng.standard_normal((n, k)) @ basis.T
    assert participation_ratio(X) == pytest.approx(k, rel=0.15)


def test_participation_ratio_is_scale_invariant():
    X = np.random.default_rng(2).standard_normal((50, 10))
    assert participation_ratio(X) == pytest.approx(participation_ratio(1000 * X),
                                                   rel=1e-9)


def test_twonn_id_of_a_line_in_high_dimensions_is_about_one():
    """Points must be SAMPLED, not evenly spaced — see the lattice test below."""
    rng = np.random.default_rng(3)
    d = rng.standard_normal(50)
    X = np.outer(np.sort(rng.random(300)), d)
    assert twonn_id(X) == pytest.approx(1.0, abs=0.4)


def test_twonn_id_returns_nan_on_a_regular_lattice():
    """
    On evenly spaced points the two nearest neighbours are equidistant (one each
    side), so mu = r2/r1 = 1 everywhere and the slope is numerical noise — the
    estimator returned ~8 for a perfectly 1D line before this guard. NaN is the
    correct answer; a plausible-looking number would be reported as a real
    dimensionality.
    """
    d = np.random.default_rng(9).standard_normal(50)
    assert np.isnan(twonn_id(np.outer(np.linspace(0, 1, 300), d)))
    assert np.isnan(twonn_id(np.linspace(0, 1, 200)[:, None]))


@pytest.mark.parametrize("d", [1, 2, 3, 5, 8])
def test_twonn_id_recovers_known_dimensionality(d):
    """
    Averaged over seeds: a single draw of 500 points carries ~15% spread, so a
    per-seed tolerance would either be flaky or so loose it tests nothing. The
    mean over 8 draws is within ~6% of truth for every d checked here.
    """
    vals = [twonn_id(np.random.default_rng(s).standard_normal((500, d)))
            for s in range(8)]
    assert float(np.mean(vals)) == pytest.approx(d, rel=0.10)


def test_twonn_id_grows_with_true_dimensionality():
    rng = np.random.default_rng(4)
    lo = twonn_id(rng.standard_normal((400, 2)))
    hi = twonn_id(rng.standard_normal((400, 8)))
    assert lo < hi


def test_twonn_id_returns_nan_when_too_few_points():
    assert np.isnan(twonn_id(np.zeros((2, 5))))


def test_logo_acc_perfect_on_separable_data():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(5)
    y = np.array([0, 1] * 12)
    groups = np.repeat(np.arange(12), 2)
    X = rng.standard_normal((24, 8)) * 0.05 + y[:, None] * 8.0
    acc = logo_acc(X, y, groups, lambda: LogisticRegression(max_iter=500))
    assert acc == pytest.approx(1.0)


def test_logo_acc_never_trains_on_the_held_out_group():
    """Group identity alone must not be decodable -> chance on random labels."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 40)
    groups = np.repeat(np.arange(20), 2)
    X = rng.standard_normal((40, 6))
    assert logo_acc(X, y, groups, lambda: LogisticRegression(max_iter=500)) < 0.8


# =========================== extract_trajectories ============================

def _summary(**kw):
    base = {"shape": [16], "mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0,
            "l2": 4.0, "absmean": 0.5, "tv": 2.0, "nan": False,
            "profile": [float(i) for i in range(16)], "rms": 1.0}
    base.update(kw)
    return base


def test_divergence_of_identical_runs_is_zero():
    a = _summary()
    assert divergence(a, dict(a))["rel_l2"] == pytest.approx(0.0, abs=1e-12)


def test_divergence_detects_a_real_difference():
    a = _summary()
    b = _summary(profile=[float(i) + 1.0 for i in range(16)])
    d = divergence(a, b)
    assert d["kind"] == "field"
    assert d["rel_l2"] > 0


def test_blowup_is_a_category_not_a_huge_number():
    """
    An invalid run producing NaN has unbounded relative L2. If that leaked out as
    a number it would dominate the ‖Δh‖ correlation in §15.1 on its own.
    """
    d = divergence(_summary(), _summary(nan=True))
    assert d["kind"] == "blowup"
    assert np.isnan(d["rel_l2"])


def test_shape_change_is_its_own_category():
    d = divergence(_summary(), _summary(shape=[32]))
    assert d["kind"] == "shape_change"
    assert np.isnan(d["rel_l2"])


def test_near_zero_profile_falls_back_instead_of_exploding():
    """
    Regression test for a real bug: a symmetric 2D field averaged to ~0, and that
    near-zero profile became the divergence denominator, yielding rel_l2 = 4.6e7.
    """
    z = [0.0] * 16
    a = _summary(profile=z, rms=1.0)
    b = _summary(profile=z, rms=2.0)
    d = divergence(a, b)
    assert d["kind"] == "degenerate_profile"
    assert 0.0 <= d["rel_l2"] <= 1.0


def test_divergence_is_bounded_for_ordinary_fields():
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = _summary(profile=list(rng.standard_normal(16)))
        b = _summary(profile=list(rng.standard_normal(16)))
        assert 0.0 <= divergence(a, b)["rel_l2"] <= 2.0


def test_divergence_handles_missing_summary():
    assert divergence(None, _summary())["kind"] == "unavailable"


def test_render_text_never_names_the_class_or_method():
    """The trajectory text is used to RETRIEVE the class; it must not state it."""
    import re
    rng = np.random.default_rng(8)
    banned = re.compile(r"burgers|heat|wave|navier|stokes|explicit|implicit|"
                        r"spectral|diffusion|advection", re.I)
    for _ in range(30):
        s = _summary(profile=list(rng.standard_normal(16)),
                     min=float(rng.normal()), max=float(rng.normal()) + 3,
                     tv=abs(float(rng.normal())) * 10)
        assert not banned.search(render_text(s, True, None))


def test_render_text_reports_failure_and_divergence_distinctly():
    assert "failed" in render_text(None, False, "ValueError: x").lower()
    assert "diverged" in render_text(_summary(nan=True), True, None).lower()


def test_render_text_handles_ok_run_with_no_field():
    assert "no recognisable field" in render_text(None, True, None)


def test_state_field_picks_the_largest_array():
    fields = {"small": _summary(shape=[8]), "big": _summary(shape=[64]),
              "tiny": None}
    key, summ = state_field(fields)
    assert key == "big"
    assert summ["shape"] == [64]


def test_state_field_ignores_degenerate_arrays():
    assert state_field({"scalarish": _summary(shape=[2])}) == (None, None)


def test_valid_of_covers_every_invalid_condition():
    """Every invalid mod_type must map to its valid twin, or pairs go missing."""
    assert set(VALID_OF) == {"Comm_InValid", "NoComm_InValid",
                             "CorrComm_Invalid", "NoComm_CorrVar_InValid"}
    assert set(VALID_OF.values()) == {"Comm_Valid", "NoComm_Valid",
                                      "CorrComm", "NoComm_CorrVar"}


def test_trajectory_json_round_trips():
    """Summaries must be JSON-serialisable — they are written with json.dump."""
    json.loads(json.dumps({"a": _summary()}))
