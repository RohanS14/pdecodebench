"""The fixed vocabulary of the experiment. Every list the figures iterate lives here.

Plotting code that hardcodes ("C", "T", "D", "M") keeps running the day a fifth
representation is added -- it just quietly describes less than the caption claims.
Reading the levels from one module makes that failure a KeyError instead.
"""

# ── representations ──────────────────────────────────────────────────────────
MODALITIES = ("C", "T", "D", "M")
MODALITY_LABELS = {
    "C": "code",
    "T": "trajectory",
    "D": "description",
    "M": "math",
}

NONE = "none"
# Rows and columns of the blame matrix, in the order the spec fixes them.
OUTLIER_LEVELS = (NONE,) + MODALITIES

# ── design ───────────────────────────────────────────────────────────────────
# The trajectory is the one view whose corruption can be dialled from gross to
# subtle, so it gets a ladder rather than a single condition. Collapsing these four
# into one "A-T" averages a shape-matched noise field together with the invalid
# solver's real output -- two corruptions that differ by orders of magnitude in how
# detectable they are -- and reports the mean as if it were one difficulty.
TRAJ_LEVELS = ("rand", "shuf", "swap", "exec")
TRAJ_LEVEL_LABELS = {
    "rand": "random values, correct shape",
    "shuf": "real values, permuted",
    "swap": "another system's trajectory",
    "exec": "the invalid solver's real output",
}
TRAJ_CONDITIONS = tuple(f"A-T-{lvl}" for lvl in TRAJ_LEVELS)

CONDITIONS = ("A0", "A-C") + TRAJ_CONDITIONS + ("A-D", "A-M")
CONDITION_OUTLIER = {
    "A0": NONE, "A-C": "C", "A-D": "D", "A-M": "M",
    **{c: "T" for c in TRAJ_CONDITIONS},
}
# The trajectory rungs all share true_outlier "T", so this direction is one-to-many
# and only meaningful for the non-trajectory views.
OUTLIER_CONDITION = {v: k for k, v in CONDITION_OUTLIER.items() if v != "T"}
CONDITION_TRAJ_LEVEL = {c: c.rsplit("-", 1)[1] for c in TRAJ_CONDITIONS}

NAMING_LEVELS = ("real", "obfuscated")

# THE sign convention for every naming contrast, defined once. A figure computing
# obfuscated-minus-real beside a verdict computing real-minus-obfuscated produced
# two statements about the same data with opposite signs; both call sites now import
# this and the label that goes with it.
DELTA_IS = "obfuscated - real"
DELTA_LABEL = "change in blame share when identifiers are obfuscated (pp)"
REASONING_LEVELS = ("on", "off")
PDE_CLASSES = ("Heat", "Wave", "Burgers", "NavierStokes")

NUMERICAL_METHODS = (
    "finite difference", "finite volume", "spectral",
    "Crank-Nicolson", "explicit Euler", "Runge-Kutta 4",
)

SCHEMA_COLUMNS = (
    "run_id", "solver_id", "pde_class", "numerical_method", "condition",
    "true_outlier", "traj_level", "naming", "reasoning", "model", "order",
    "pred_agree", "pred_outlier", "pred_pde_class", "pred_method",
    "justification", "judge_correct",
    # Model metadata, joined from data/model_registry.csv rather than carried in
    # the results rows -- a results row records what the model answered, not when
    # the model was released. These three are what the generational figure needs.
    "release_date", "params_total_b", "family",
)

# Columns any slice may legally group by. release_date and params_total_b are
# deliberately absent: they are continuous covariates for the trend figure, not
# categories to facet on.
SLICE_COLUMNS = ("naming", "reasoning", "model", "pde_class", "traj_level", "family")

# ── colour ───────────────────────────────────────────────────────────────────
# Okabe-Ito. Colourblind-safe across all four, and separable in greyscale print,
# which some workshop proceedings still produce. One modality keeps one colour in
# every figure in the set -- that consistency is what lets a reader carry a colour
# from the blame matrix to the trust scatter without a second look at the legend.
MODALITY_COLORS = {
    "C": "#0072B2",   # blue
    "T": "#D55E00",   # vermillion
    "D": "#009E73",   # bluish green
    "M": "#CC79A7",   # reddish purple
}
NONE_COLOR = "#7A7A7A"
OUTLIER_COLORS = {NONE: NONE_COLOR, **MODALITY_COLORS}

# Conditions are the control axis, not a finding, so they stay neutral: the only
# categorical hue in the figure set means "representation".
# The four trajectory rungs share a ramp of their own so the ladder reads as one
# family rather than four unrelated conditions.
CONDITION_GREYS = {
    "A0":       "#DCDCDC",
    "A-C":      "#B4B4B4",
    "A-T-rand": "#A9C2DD",
    "A-T-shuf": "#8AA9CC",
    "A-T-swap": "#6B90BB",
    "A-T-exec": "#4C77AA",
    "A-D":      "#6A6A6A",
    "A-M":      "#454545",
}

# Outcome bars are a ramp from "right" to "wrong", with hatching carrying the two
# error kinds so the stack survives greyscale and colourblind rendering both.
# "correct agree" and "detected, correct outlier" are the SAME event -- the model
# gave the right answer -- reached from the two halves of the design. Splitting them
# forced one legend to span two disjoint sets (only A0 can be a false alarm, only a
# corrupted item can be missed), so no reader could compare bars. Merged, the bottom
# segment is simply accuracy, and its height is comparable across every condition.
OUTCOMES = (
    "correct",
    "flagged, wrong view",
    "missed the disagreement",
    "false alarm",
)
OUTCOME_COLORS = {
    "correct":                 "#3C3C3C",
    "flagged, wrong view":     "#8E8E8E",
    "missed the disagreement": "#C4C4C4",
    "false alarm":             "#E4E4E4",
}
# Dark ground: the ramp inverts so "correct" is the brightest segment rather than
# a near-black block that disappears into the panel.
OUTCOME_COLORS_DARK = {
    "correct":                 "#cfd8e8",
    "flagged, wrong view":     "#8592ae",
    "missed the disagreement": "#4a5570",
    "false alarm":             "#2a3145",
}
OUTCOME_HATCH = {
    "correct":                 "",
    "flagged, wrong view":     "//",
    "missed the disagreement": "",
    "false alarm":             "..",
}
# Which conditions each outcome can occur in at all. Used to caption the legend so a
# structurally-zero segment does not read as a measured zero.
OUTCOME_SCOPE = {
    "correct":                 "all",
    "flagged, wrong view":     "corrupted only",
    "missed the disagreement": "corrupted only",
    "false alarm":             "A0 only",
}

SEQUENTIAL_CMAP = "Greys"
