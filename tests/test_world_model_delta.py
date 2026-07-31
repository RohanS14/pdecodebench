"""
Unit tests for probe/world_model_delta.py — runs locally, no GPU, no model.

These are power-and-null tests on synthetic hidden states with a known ground
truth. The point is that `within_cos` on its own cannot distinguish the two ways
a high score can arise, so we verify the control statistics do:

    regime   | what was planted                       | within_cos | gap | match@1
    ---------|----------------------------------------|------------|-----|--------
    signal   | per-solver defect direction, shared     |    high    | high|  high
             | across all 4 surface conditions         |            |     |
    null     | pure noise                             |    ~0      | ~0  | chance
    generic  | ONE global defect direction, all solvers|    high    | ~0  | chance

The `generic` row is the reason the cross-solver control exists: a single "this
code is broken" direction produces a `within_cos` that looks like a strong result.
If a future refactor breaks the control, this test fails.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probe"))
from world_model_delta import (  # noqa: E402
    COND_NAMES, CONDITIONS, build_pair_index, compute_deltas, cos_matrix,
    generic_transfer_acc, match_acc, permutation_gap, principal_angles, unit,
)

N_SOLVERS, L, D = 24, 3, 96
LAYER = 1


def make_data(regime: str, seed: int = 0) -> dict:
    """Synthetic NPZ-shaped dict with a planted ground truth."""
    rng = np.random.default_rng(seed)
    solvers = [f"Sample_{i}" for i in range(N_SOLVERS)]
    mod_types = [m for pair in CONDITIONS.values() for m in pair]

    base = rng.standard_normal((N_SOLVERS, len(COND_NAMES), L, D)) * 10.0
    solver_dir = rng.standard_normal((N_SOLVERS, L, D))
    global_dir = rng.standard_normal((L, D))

    H, gt, mt, pde = [], [], [], []
    for si, s in enumerate(solvers):
        for ci, cname in enumerate(COND_NAMES):
            for is_inv, m in enumerate(CONDITIONS[cname]):
                h = base[si, ci].copy()
                if is_inv:
                    noise = rng.standard_normal((L, D))
                    if regime == "signal":
                        h = h + 0.3 * solver_dir[si] + 0.3 * noise
                    elif regime == "null":
                        h = h + 0.3 * noise
                    elif regime == "generic":
                        h = h + 0.3 * global_dir + 0.3 * noise
                    else:
                        raise ValueError(regime)
                H.append(h)
                gt.append(s)
                mt.append(m)
                pde.append(["wave", "heat", "burgers", "navier-stokes"][si % 4])
    return {
        "mean_pool": np.asarray(H, dtype=np.float32),
        "gt_samples": np.array(gt),
        "mod_types": np.array(mt),
        "pde_classes": np.array(pde),
    }


def deltas_for(regime: str) -> np.ndarray:
    data = make_data(regime)
    solvers, idx = build_pair_index(data)
    return compute_deltas(data["mean_pool"], solvers, idx, LAYER)


@pytest.fixture(scope="module")
def d():
    return {r: deltas_for(r) for r in ("signal", "null", "generic")}


def _pair(delta):
    """First two surface conditions, as (S, D) each."""
    return delta[:, 0, :], delta[:, 1, :]


# --- structural -------------------------------------------------------------

def test_conditions_are_the_eight_mod_types():
    flat = [m for pair in CONDITIONS.values() for m in pair]
    assert len(flat) == 8 and len(set(flat)) == 8


def test_build_pair_index_rejects_ragged_grid():
    data = make_data("null")
    for k in ("mean_pool", "gt_samples", "mod_types", "pde_classes"):
        data[k] = data[k][1:]                      # drop one cell
    with pytest.raises(ValueError, match="missing"):
        build_pair_index(data)


def test_delta_is_invalid_minus_valid():
    data = make_data("null")
    solvers, idx = build_pair_index(data)
    delta = compute_deltas(data["mean_pool"], solvers, idx, LAYER)
    H = data["mean_pool"]
    v, iv = CONDITIONS[COND_NAMES[0]]
    expect = (H[idx[(solvers[0], iv)], LAYER].astype(np.float64)
              - H[idx[(solvers[0], v)], LAYER].astype(np.float64))
    np.testing.assert_allclose(delta[0, 0], expect, rtol=1e-6)


def test_delta_is_float64_not_float32():
    # Δh is a small difference of large vectors; precision is the whole ballgame
    assert deltas_for("null").dtype == np.float64


# --- power: the planted signal is detected ----------------------------------

def test_signal_gap_is_large_and_significant(d):
    A, B = _pair(d["signal"])
    gap, within, cross, p, _ = permutation_gap(A, B, 2000, np.random.default_rng(0))
    assert within > 0.3
    assert abs(cross) < 0.1
    assert gap > 0.3
    assert p < 0.01


def test_signal_match_acc_far_above_chance(d):
    A, B = _pair(d["signal"])
    top1, mrr = match_acc(cos_matrix(A, B))
    assert top1 > 0.8
    assert mrr > 0.8


# --- null: nothing planted, nothing found -----------------------------------

def test_null_gap_is_zero(d):
    A, B = _pair(d["null"])
    gap, within, _, p, _ = permutation_gap(A, B, 2000, np.random.default_rng(0))
    assert abs(within) < 0.1
    assert abs(gap) < 0.1
    assert p > 0.05


def test_null_match_acc_at_chance(d):
    A, B = _pair(d["null"])
    top1, _ = match_acc(cos_matrix(A, B))
    assert top1 < 4.0 / N_SOLVERS


# --- the confound: high within_cos that is NOT a solver-specific representation

def test_generic_direction_inflates_within_cos(d):
    """A single global defect direction makes within_cos look like a result."""
    A, B = _pair(d["generic"])
    assert float(np.mean(np.diag(cos_matrix(A, B)))) > 0.3


def test_generic_direction_is_caught_by_the_cross_solver_gap(d):
    """...but the gap, which is the headline number, correctly reads ~0."""
    A, B = _pair(d["generic"])
    gap, _, cross, p, _ = permutation_gap(A, B, 2000, np.random.default_rng(0))
    assert cross > 0.3               # the inflation is in BOTH within and cross
    assert abs(gap) < 0.1
    assert p > 0.05


def test_generic_direction_is_caught_by_match_acc(d):
    A, B = _pair(d["generic"])
    top1, _ = match_acc(cos_matrix(A, B))
    assert top1 < 4.0 / N_SOLVERS


def test_generic_transfer_separates_the_two_regimes(d):
    """generic_transfer_acc is the mirror image of match_acc, by construction."""
    sig_a, sig_b = _pair(d["signal"])
    gen_a, gen_b = _pair(d["generic"])
    assert generic_transfer_acc(gen_a, gen_b) > 0.9      # shared direction: found
    assert generic_transfer_acc(sig_a, sig_b) < 0.7      # per-solver: averaged away


# --- geometry ---------------------------------------------------------------

def test_random_direction_scores_near_zero(d):
    rng = np.random.default_rng(1)
    g = rng.standard_normal((N_SOLVERS, D))
    for regime in ("signal", "null", "generic"):
        A = d[regime][:, 0, :]
        r = float(np.mean(np.sum(unit(A) * unit(g), axis=1)))
        assert abs(r) < 0.1, f"{regime}: random cosine {r} — cosine is reading dimensionality"


def test_unit_handles_zero_vector_without_nan():
    v = np.zeros((2, 5))
    assert not np.isnan(unit(v)).any()


def test_principal_angles_bounds_and_ordering(d):
    A, B = _pair(d["signal"])
    ang = principal_angles(A, B, 5)
    assert len(ang) == 5
    assert np.all(ang >= -1e-9) and np.all(ang <= 90 + 1e-9)
    assert np.all(np.diff(ang) >= -1e-9)          # ascending


def test_principal_angle_of_subspace_with_itself_is_zero(d):
    # Tolerance is 1e-3 deg, not machine epsilon: arccos has sqrt(eps) sensitivity
    # near a singular value of 1, so float64 lands around 1e-6 deg here.
    A = d["signal"][:, 0, :]
    assert principal_angles(A, A, 3).max() < 1e-3


def test_permutation_gap_is_deterministic_given_seed(d):
    A, B = _pair(d["signal"])
    a = permutation_gap(A, B, 500, np.random.default_rng(7))
    b = permutation_gap(A, B, 500, np.random.default_rng(7))
    assert a == b
