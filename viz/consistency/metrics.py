"""Every number the figures and tables draw. Tidy DataFrames out, nothing printed.

Two definitional choices are worth stating, because they are the ones a reader will
challenge and the ones the tests pin:

* **Localization accuracy is conditional on correct detection.** The denominator is
  corrupted items the model actually flagged (`is_corrupted & detected`), not all
  corrupted items. Unconditional localization mixes two failures -- never noticing,
  and noticing but pointing at the wrong view -- into one number, and the whole
  point of the two-step framing is to keep them apart.

* **False-blame rate for modality m is P(pred_outlier == m | true_outlier != m).**
  The denominator includes A0 rows, where blaming anything is already wrong. A
  modality can therefore have a high false-blame rate purely by being the model's
  default guess on clean items, which is exactly the over-trust effect the
  experiment is looking for.

Every proportion carries a Wilson interval. Wilson rather than normal-approximation
because these rates sit near 0 and 1 often enough -- a model that never abstains, a
modality never blamed -- that a Wald interval would run outside [0, 1] and quietly
render as a bar poking past the axis.
"""
import math

import numpy as np
import pandas as pd

from .constants import (CONDITIONS, MODALITIES, NONE, OUTLIER_LEVELS, TRAJ_LEVELS)

Z95 = 1.959963984540054

PROP_COLUMNS = ["k", "n", "rate", "lo", "hi"]


def wilson_ci(k, n, z=Z95):
    """Wilson score interval for k successes in n trials. (nan, nan) when n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    k, n = float(k), float(n)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def prepare(df):
    """Add the derived booleans every metric is phrased in. Never mutates `df`.

    `detected` is the model's claim that something disagrees; `detection_correct`
    is whether that claim matched reality, which on A0 means having said *yes,
    they agree*. Conflating the two is the easiest way to report a false-alarm-
    prone model as a sensitive one.
    """
    d = df.copy()
    d["is_corrupted"] = d["true_outlier"].ne(NONE)
    d["detected"] = d["pred_agree"].eq("no")
    d["detection_correct"] = d["detected"].eq(d["is_corrupted"])
    d["localization_correct"] = (
        d["is_corrupted"] & d["detected"] & d["pred_outlier"].eq(d["true_outlier"]))
    # Only meaningful where the model both should have and did flag the item; the
    # figures read this column and must not count clean rows as localization misses.
    d["localization_eligible"] = d["is_corrupted"] & d["detected"]
    if "judge_correct" in d:
        d["judge_correct"] = d["judge_correct"].astype(bool)
    return d


def _empty(by, extra=()):
    return pd.DataFrame(columns=[*by, *extra, *PROP_COLUMNS])


def _proportion(d, by, flag, den_mask=None):
    """k/n of `flag` within `den_mask`, grouped by `by`, with Wilson intervals."""
    by = list(by)
    if den_mask is not None:
        d = d[den_mask]
    if d.empty:
        return _empty(by)
    f = d[flag].astype(bool)
    if by:
        g = f.groupby([d[c] for c in by], dropna=False).agg(["sum", "count"])
        g.index.names = by
        out = g.reset_index().rename(columns={"sum": "k", "count": "n"})
    else:
        out = pd.DataFrame([{"k": int(f.sum()), "n": int(len(f))}])
    out["k"] = out["k"].astype(int)
    out["n"] = out["n"].astype(int)
    out["rate"] = np.where(out["n"] > 0, out["k"] / out["n"].replace(0, np.nan), np.nan)
    ci = [wilson_ci(k, n) for k, n in zip(out["k"], out["n"])]
    out["lo"] = [c[0] for c in ci]
    out["hi"] = [c[1] for c in ci]
    return out


# ── detection ────────────────────────────────────────────────────────────────
def detection_metrics(df, by=()):
    """Overall detection accuracy, plus specificity (A0 only) and recall (corrupted).

    Returned long, one row per (slice, metric), so a caller can facet on `metric`
    without knowing which slices exist.
    """
    d = prepare(df)
    by = list(by)
    parts = []
    for metric, flag, mask in (
            ("overall", "detection_correct", None),
            # On clean items the correct answer is "they agree", so specificity is
            # the rate of NOT flagging.
            ("specificity", "detection_correct", ~d["is_corrupted"]),
            ("recall", "detected", d["is_corrupted"]),
    ):
        p = _proportion(d, by, flag, mask)
        if p.empty:
            continue
        p.insert(len(by), "metric", metric)
        parts.append(p)
    if not parts:
        return _empty(by, ("metric",))
    return pd.concat(parts, ignore_index=True)


def localization_accuracy(df, by=()):
    """P(pred_outlier == true_outlier | corrupted AND detected). See module docstring."""
    d = prepare(df)
    return _proportion(d, list(by), "localization_correct", d["localization_eligible"])


# ── blame ────────────────────────────────────────────────────────────────────
def blame_matrix(df, by=(), row_by="true_outlier"):
    """Blame matrix, long, with raw counts and row fractions.

    `row_by="true_outlier"` gives the 5x5 view. `row_by="condition"` splits the
    trajectory into its four corruption rungs, giving 8x5 -- the same columns, but
    rows that distinguish a shape-matched noise field from the invalid solver's own
    output. Those two are not one condition and should not share a row.

    Reindexed over the full OUTLIER_LEVELS x OUTLIER_LEVELS grid so absent cells
    come back as 0 rather than missing. A heatmap built from a sparse frame draws a
    different shape per slice, which makes two panels silently non-comparable.
    """
    d = prepare(df)
    by = list(by)
    if d.empty:
        return pd.DataFrame(columns=[*by, row_by, "pred_outlier",
                                     "n", "row_total", "row_frac"])
    row_levels = OUTLIER_LEVELS if row_by == "true_outlier" else CONDITIONS
    grid = pd.MultiIndex.from_product([row_levels, OUTLIER_LEVELS],
                                      names=[row_by, "pred_outlier"])
    frames = []
    groups = d.groupby(by, dropna=False) if by else [((), d)]
    for key, g in groups:
        counts = (g.groupby([row_by, "pred_outlier"], dropna=False).size()
                  .reindex(grid, fill_value=0).rename("n").reset_index())
        totals = counts.groupby(row_by)["n"].transform("sum")
        counts["row_total"] = totals
        # A row with no observations has an undefined fraction. NaN, not 0: zero
        # would draw as "never blamed", which is a claim the data does not make.
        counts["row_frac"] = np.where(totals > 0, counts["n"] / totals.replace(0, np.nan),
                                      np.nan)
        if by:
            key = key if isinstance(key, tuple) else (key,)
            for col, val in zip(by, key):
                counts.insert(0, col, val)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def per_modality_rates(df, by=()):
    """Detection rate and false-blame rate for each modality. See module docstring."""
    d = prepare(df)
    by = list(by)
    rows = []
    for m in MODALITIES:
        det = _proportion(d, by, "detected", d["true_outlier"].eq(m))
        if not det.empty:
            det.insert(len(by), "metric", "detection_rate")
            det.insert(len(by), "modality", m)
            rows.append(det)
        blamed = d["pred_outlier"].eq(m)
        tmp = d.assign(_blamed=blamed)
        fb = _proportion(tmp, by, "_blamed", d["true_outlier"].ne(m))
        if not fb.empty:
            fb.insert(len(by), "metric", "false_blame_rate")
            fb.insert(len(by), "modality", m)
            rows.append(fb)
    if not rows:
        return _empty(by, ("modality", "metric"))
    return pd.concat(rows, ignore_index=True)


def traj_ladder(df, by=()):
    """Detection rate for each trajectory corruption rung, with Wilson intervals.

    The trajectory's detectability spans the widest range of any view in the design,
    so a single "trajectory detection rate" is an average over conditions that differ
    by a factor of two or more. This is that average taken apart.
    """
    d = prepare(df)
    if "traj_level" not in d:
        return _empty(list(by), ("traj_level",))
    d = d[d["true_outlier"].eq("T") & d["traj_level"].astype(str).ne("")]
    if d.empty:
        return _empty(list(by), ("traj_level",))
    out = _proportion(d, ["traj_level", *list(by)], "detected")
    order = {lvl: i for i, lvl in enumerate(TRAJ_LEVELS)}
    return out.sort_values("traj_level", key=lambda c: c.map(order)).reset_index(drop=True)


# ── justification ────────────────────────────────────────────────────────────
def judge_rate(df, by=()):
    """LLM-judge agreement rate, grouped by condition plus any extra slice."""
    d = prepare(df)
    if "judge_correct" not in d:
        return _empty(["condition", *by])
    return _proportion(d, ["condition", *list(by)], "judge_correct")


# ── outcome decomposition ────────────────────────────────────────────────────
def outcome_of(d):
    """One mutually exclusive outcome label per row, for the stacked bars."""
    clean, det = ~d["is_corrupted"], d["detected"]
    correct_view = d["pred_outlier"].eq(d["true_outlier"])
    return np.select(
        [clean & ~det,                        # agreed about an item that agrees
         clean & det,                         # flagged a clean item
         ~clean & ~det,                       # missed a real disagreement
         ~clean & det & correct_view,         # flagged it and named the right view
         ~clean & det & ~correct_view],
        ["correct", "false alarm", "missed the disagreement",
         "correct", "flagged, wrong view"],
        default="missed the disagreement")


def outcome_breakdown(df, by=("condition",)):
    """Share of each outcome within every slice. Shares sum to 1 per group."""
    d = prepare(df)
    if d.empty:
        return pd.DataFrame(columns=[*by, "outcome", "n", "total", "share"])
    d = d.assign(outcome=outcome_of(d))
    by = list(by)
    n = d.groupby([*by, "outcome"], dropna=False).size().rename("n").reset_index()
    n["total"] = n.groupby(by, dropna=False)["n"].transform("sum")
    n["share"] = n["n"] / n["total"]
    return n


def main_table(df):
    """Condition x {detection acc, localization acc, judge_correct, n}, with CIs."""
    d = prepare(df)
    det = detection_metrics(d, by=["condition"])
    det = det[det["metric"].eq("overall")].set_index("condition")
    loc = localization_accuracy(d, by=["condition"]).set_index("condition")
    jud = judge_rate(d).set_index("condition")
    rows = []
    for c in CONDITIONS:
        if c not in det.index:
            continue
        row = {"condition": c, "n": int(det.loc[c, "n"])}
        for name, src in (("detection", det), ("localization", loc), ("judge", jud)):
            if c in src.index:
                row[f"{name}_rate"] = float(src.loc[c, "rate"])
                row[f"{name}_lo"] = float(src.loc[c, "lo"])
                row[f"{name}_hi"] = float(src.loc[c, "hi"])
                row[f"{name}_n"] = int(src.loc[c, "n"])
            else:
                row.update({f"{name}_rate": np.nan, f"{name}_lo": np.nan,
                            f"{name}_hi": np.nan, f"{name}_n": 0})
        rows.append(row)
    return pd.DataFrame(rows)
