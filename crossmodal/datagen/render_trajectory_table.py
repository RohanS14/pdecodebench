"""
render_trajectory_table.py — the single point through which every trajectory the
cross-modal consistency experiment shows a model must pass (plan Part III).

Why one renderer for all four rungs of the corruption ladder: the trajectory view
is corrupted four different ways, and three of those arrive from the CSV while
`T_exec` arrives from re-execution. If the rungs were formatted independently the
outlier would be identifiable from formatting alone -- array shape, float
precision, or pipeline fingerprint -- with no physics involved. Measured on the
delivered data:

  * 30/32 swapped trajectories have a DIFFERENT array shape from their valid twin,
    and 14/32 flip 1-D <-> 2-D. `Heat_4_wrong` is (10,41,41,2) for code declaring
    a 1-D n=50 grid.
  * 20/64 stored trajectories are printed without scientific notation, losing
    significant digits relative to the %e-formatted ones.

Both are erased here rather than in the data: every trajectory in an item is
resampled onto ONE grid, fixed by that item's valid trajectory, and printed with
one format string. The rendered tables are then byte-identical in size and
precision, so shape and formatting cannot leak which view is the odd one out.

The dataset carries no axis metadata at all -- no dt, no time span, no grid
spacing, no coordinates, no variable name -- so the header states normalized
coordinates rather than inventing physical units it cannot support.

Usage:
    from crossmodal.datagen.render_trajectory_table import parse_trajectory, choose_grid, render
    a = parse_trajectory(row["Trajectory"])
    grid = choose_grid(a.shape)          # fixed by the VALID trajectory of the item
    text = render(a, grid)               # every rung of the ladder uses this same grid
"""
import ast
import json

import numpy as np

# One number renders as e.g. "-1.2345e-03" -> 11 chars plus a separator, roughly
# 4 tokens. These budgets were calibrated against all 32 valid trajectories by
# downsampling, linearly re-interpolating back to full resolution, and measuring
# relative L2 error. Two results drove the choice:
#
#   * For 1-D fields, FRAMES matter more than probes. At a fixed ~400-number
#     budget, 10 frames x 40 probes gives median error 0.062 while 5 frames x 80
#     probes gives 0.179 -- these fields are spatially smooth but the dynamics are
#     the whole point, so halving the time axis costs more than doubling space
#     buys. Beyond ~50 probes the return flattens (0.049 at 50, 0.040 at 64,
#     0.032 at 80), so probes are capped rather than spent to the budget.
#   * 2-D fields are far more expensive and never get cheap. See RENDER_HARD_CASES.
BUDGET_NUMBERS = 2000
MAX_FRAMES = 10
MAX_PROBES_1D = 64

# Systems whose field cannot be faithfully rendered on ANY prompt-feasible uniform
# grid, measured as relative L2 reconstruction error at a 10x24x24 sampling:
#
#   Heat_7          err 0.998 and FLAT in resolution -- 0.03% of cells exceed 1% of
#                   peak and peak/rms is 96, i.e. a near-point source that a uniform
#                   probe grid simply misses. More resolution does not help.
#   NavierStokes_4  err 0.618   high-frequency vorticity
#   NavierStokes_8  err 0.620   high-frequency vorticity
#   Wave_7          err 0.423   high-frequency
#
# The other 28 systems land at or below ~0.25. These four are recorded per item so
# the analysis can check whether trajectory-condition accuracy tracks rendering
# fidelity, and they are reported as a limitation rather than silently included.
RENDER_HARD_CASES = ("Heat_7", "NavierStokes_4", "NavierStokes_8", "Wave_7")

FMT = "%+.4e"   # always signed, so every finite number is exactly 11 chars wide
# 12, not 11: a blown-up solver reaches three-digit exponents (+1.2345e+308) where
# a well-behaved one stays at two (+1.2345e-03). Padding to the widest possible
# float64 rendering is what makes the column width independent of magnitude.
FMT_WIDTH = 12

# Non-finite values must occupy the SAME width as a finite one. Many invalid
# solvers blow up -- "Blows up to NaN" is a common invalidity_note -- and Python
# renders nan in 3 characters against 11 for a float. Left alone, that made the
# T_exec table visibly shorter than the other three rungs and identified the
# corrupted view from character count, with no physics involved.
#
# Whether a blow-up makes T_exec EASY is a separate, legitimate question: a NaN
# field genuinely does not match a smooth description, and the repo already treats
# blow-up as a category rather than a large number. That is recorded per item as a
# covariate (see has_nonfinite) so the condition can be reported split by it,
# rather than hidden by the formatting.
_NONFINITE = {"nan": "nan".rjust(FMT_WIDTH),
              "inf": "+inf".rjust(FMT_WIDTH),
              "-inf": "-inf".rjust(FMT_WIDTH)}


def _fmt(v):
    if v != v:
        return _NONFINITE["nan"]
    if v == float("inf"):
        return _NONFINITE["inf"]
    if v == float("-inf"):
        return _NONFINITE["-inf"]
    return (FMT % v).rjust(FMT_WIDTH)


def has_nonfinite(a):
    """Whether a trajectory contains NaN or infinity -- recorded per item."""
    return bool(not np.isfinite(np.asarray(a, dtype=float)).all())

HEADER = (
    "Numerical solution field, sampled on a uniform normalized grid.\n"
    "Rows are successive time samples from the start of the run (t=0.00) to the "
    "end (t=1.00) in equal steps.\n"
    "Within a row, values run over the spatial domain from 0.00 to 1.00 in equal "
    "steps along each axis.\n"
    "No physical units, grid spacing, or time step are available for this field."
)


def parse_trajectory(text):
    """Parse a stored trajectory into a (T, X, Y, C) float array.

    The column holds nested brackets, comma separated. json.loads is tried first
    because it reads BOTH the delivered format and NaN/Infinity, which
    ast.literal_eval cannot -- and non-finite values are not hypothetical here:
    many invalid solvers blow up, which is the point of them, so a T_exec
    trajectory routinely contains NaN. ast.literal_eval is kept as a fallback for
    any Python-repr-only spelling.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = ast.literal_eval(text)
    a = np.asarray(parsed, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"expected a 4-D (T, X, Y, C) trajectory, got shape {a.shape}")
    return a


def choose_grid(shape, budget=BUDGET_NUMBERS):
    """Pick the render grid for an item from its VALID trajectory's shape.

    Frames are taken first (up to MAX_FRAMES), then the remaining budget goes to
    space. That ordering is measured, not assumed -- see BUDGET_NUMBERS. 1-D
    probes are capped at MAX_PROBES_1D because the error curve flattens there and
    the extra numbers would only inflate the prompt.

    n_c is never reduced: dropping a velocity component changes what the view is,
    not merely how finely it is sampled.

    Returns (n_t, n_x, n_y, n_c).
    """
    T, X, Y, C = shape
    n_t = min(T, MAX_FRAMES)
    per_frame = max(1, budget // (n_t * C))

    if Y > 1:
        side = max(2, int(np.sqrt(per_frame)))
        return (n_t, min(X, side), min(Y, side), C)
    return (n_t, min(X, min(per_frame, MAX_PROBES_1D)), 1, C)


def _axis_index(n_src, n_out):
    """Nearest-index sampling at normalized coordinates, endpoints included.

    This is the same rule the dataset itself used to decimate to 10 frames --
    verified by execution: Heat_1's stored frames are indices 0,111,...,1000 of a
    1001-step history, exactly linspace(0, N-1, 10).
    """
    if n_src == 1:
        return np.zeros(n_out, dtype=int)
    return np.linspace(0, n_src - 1, n_out).round().astype(int)


def resample(a, grid):
    """Resample any trajectory onto the item's grid.

    Handles donors whose shape differs from the receiver's, which is the whole
    point for T_swap: 30/32 swapped trajectories have a different shape, and 14/32
    are a different dimensionality entirely. Channels wrap modulo the donor's own
    count so a 1-channel donor can fill a 2-channel receiver without inventing a
    zero field that would itself be a tell.
    """
    n_t, n_x, n_y, n_c = grid
    T, X, Y, C = a.shape
    out = a[_axis_index(T, n_t)][:, _axis_index(X, n_x)][:, :, _axis_index(Y, n_y)]
    return out[:, :, :, np.arange(n_c) % C]


def render(a, grid):
    """Render a trajectory as the uniform numeric table shown to the model."""
    s = resample(a, grid)
    n_t, n_x, n_y, n_c = s.shape
    lines = [HEADER, ""]
    for t in range(n_t):
        frac = 0.0 if n_t == 1 else t / (n_t - 1)
        for c in range(n_c):
            tag = f"t={frac:.2f}" + (f" component={c}" if n_c > 1 else "")
            values = " ".join(_fmt(v) for v in s[t, :, :, c].ravel())
            lines.append(f"{tag}: {values}")
    return "\n".join(lines)


def _interp_axis(a, axis, n_out):
    n = a.shape[axis]
    if n == n_out:
        return a
    src, dst = np.linspace(0, 1, n), np.linspace(0, 1, n_out)
    return np.apply_along_axis(lambda y: np.interp(dst, src, y), axis, a)


def reconstruction_error(a, grid):
    """Relative L2 error from rendering `a` on `grid` and interpolating back.

    The fidelity number recorded per item. Preferred over counting local extrema,
    which was the first thing I tried and is unusable here: it is a ratio whose
    denominator collapses on near-monotone fields (Heat_5, Heat_6) and which
    aliasing can inflate as easily as destroy, so it ranked grids
    non-monotonically. L2 reconstruction error is monotone in resolution and
    directly answers "how much of the field survives the render".
    """
    r = resample(a, grid)
    for ax, n in enumerate(a.shape):
        if r.shape[ax] != n:
            r = _interp_axis(r, ax, n)
    den = float(np.sqrt((a ** 2).mean()))
    if den == 0.0:
        return 0.0
    return float(np.sqrt(((r - a) ** 2).mean()) / den)
