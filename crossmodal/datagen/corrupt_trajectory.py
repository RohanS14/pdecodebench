"""
corrupt_trajectory.py — the four wrong-variants of the trajectory view
(cross-modal consistency experiment, plan Part III).

The dataset ships ONE wrong trajectory per system, and it turns out to be another
system's correct trajectory: the 64 stored trajectories contain only 32 unique
values, in a perfect derangement (Heat_1_wrong is byte-identical to
Navier_Stokes_6, Wave_3_wrong to Heat_1, and so on). That is a far grosser
corruption than the other three views receive -- code, math and description each
get a single sign flip -- so using it alone would measure corruption severity
rather than which modality a model trusts.

So severity becomes an axis. Four rungs, from structurally empty to physically
subtle:

    T_rand   same shape, i.i.d. values matched to the valid trajectory's mean and
             variance. A floor: detectable from smoothness alone, no PDE knowledge
             needed. A model that misses this is not reading the numbers at all.

    T_shuf   a permutation of the valid trajectory's OWN values. Preserves the
             shape and every marginal statistic exactly -- mean, variance, min,
             max, the entire histogram -- so nothing but the arrangement
             distinguishes it. This is the sharpest control in the design:
             detecting it requires reading spatiotemporal structure, and no
             summary statistic can help.

    T_swap   the delivered cross-system trajectory. Coherent physics, wrong
             referent.

    T_exec   the actual output of the invalid solver, decimated to the same frames.
             Coherent physics, right system, wrong dynamics. The real test.

Reading the accuracy curve across the four says WHAT KIND of wrongness a model can
see, which no single rung can.

Determinism: every draw is seeded from the system name, so rebuilding produces
byte-identical output the way datagen/build_jul28.py already does. Nothing here
depends on global RNG state.
"""
import ast
import hashlib

import numpy as np

# T_exec arrives from re-execution rather than from the CSV, so it must be
# decimated the same way the dataset was. Verified by executing the solvers and
# matching frames: Heat_1's stored frames are indices 0, 111, 222, ..., 1000 of a
# 1001-step history and Burgers_1's are 0, 22, ..., 199 of 200 -- exactly
# linspace(0, N-1, 10), inclusive of both t=0 and the final state. Not the first
# 10 steps, not log-spaced.
DATASET_FRAMES = 10

LADDER = ("T_rand", "T_shuf", "T_swap", "T_exec")


def _rng(system, level):
    """Per-(system, level) RNG seeded from the names, so the build is reproducible
    and a system's draw does not shift when an unrelated system is added."""
    key = f"{system}::{level}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def decimate_frames(a, n_frames=DATASET_FRAMES):
    """Sample n_frames evenly over the whole run, endpoints included.

    Applied to a re-executed solver's full history so T_exec lands on the same
    frame convention as everything from the CSV.
    """
    T = a.shape[0]
    if T <= n_frames:
        return a
    return a[np.linspace(0, T - 1, n_frames).round().astype(int)]


def make_random(valid, system):
    """T_rand: same shape, moments matched, no structure."""
    rng = _rng(system, "T_rand")
    out = rng.normal(float(valid.mean()), float(valid.std()), size=valid.shape)
    assert out.shape == valid.shape
    return out


def make_shuffled(valid, system):
    """T_shuf: the valid trajectory's own values, rearranged.

    A full permutation across space and time together. Shuffling only within each
    frame, or only permuting whole frames, would be weaker -- both leave part of
    the arrangement intact.
    """
    rng = _rng(system, "T_shuf")
    out = rng.permutation(valid.ravel()).reshape(valid.shape)
    assert out.shape == valid.shape
    # The defining property: identical multiset of values, so no position-blind
    # statistic can tell this from the valid trajectory.
    assert np.array_equal(np.sort(out.ravel()), np.sort(valid.ravel()))
    return out


def make_time_shuffled(valid, system):
    """T_timeshuf: the frames themselves permuted, each one left intact.

    Every frame remains an individually valid physical field; only the ordering is
    non-causal -- diffusion running backwards, a pulse un-spreading. No statistical
    tell at all, and it isolates temporal causality specifically. Proposed as a
    fifth rung rather than assumed, since it costs another 32 items per factor cell.
    """
    rng = _rng(system, "T_timeshuf")
    order = rng.permutation(valid.shape[0])
    if valid.shape[0] > 1 and np.array_equal(order, np.arange(valid.shape[0])):
        order = order[::-1]          # never hand back the identity
    return valid[order]


def make_swapped(wrong_text):
    """T_swap: the delivered `_wrong` column, parsed as-is.

    Its shape differs from the valid twin in 30 of 32 cases and flips 1-D <-> 2-D
    in 14, which would give the outlier away for free. That is handled downstream:
    render_trajectory_table.resample puts every candidate on the item's own grid,
    so all four rungs print at identical size and precision.
    """
    return np.asarray(ast.literal_eval(wrong_text), dtype=float)


def build_ladder(valid, wrong_text, system, include_time_shuffle=False):
    """All wrong-variants for one system, keyed by level name.

    T_exec is absent here -- it requires executing the invalid solver, which
    happens in the cpu_short job, and is merged in afterwards.
    """
    out = {
        "T_rand": make_random(valid, system),
        "T_shuf": make_shuffled(valid, system),
        "T_swap": make_swapped(wrong_text),
    }
    if include_time_shuffle:
        out["T_timeshuf"] = make_time_shuffled(valid, system)
    return out
