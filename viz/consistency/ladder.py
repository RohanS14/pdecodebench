"""ladder.py — the model comparison, built around the Qwen ladder.

The eight models in this roster are NOT eight points on a time axis. Four of them
are consecutive releases from one lab (Qwen3-32B -> 3.5 -> 3.6 -> 3.8) and admit an
ordering; the other four are singletons from four different labs and admit none.
Plotting all eight against calendar date and fitting a line manufactures a
field-wide trajectory out of four unrelated models that happen to have dates.

So: the ladder carries the only trend claim, drawn on an ORDINAL axis of release
position because four points do not support a time axis. The singletons appear as a
band showing how far apart two contemporary 30B models can be for reasons that have
nothing to do with time. That band is the yardstick — a ladder movement smaller than
the between-model spread is not a generational finding.

Roles are declared in data/models.yaml and never inferred from the data.

Metrics here are deliberately not hit rate. QwQ-32B posts a 0.93 hit rate against a
0.75 false-alarm rate; Qwen3.5 posts 0.99 against 0.91. Both are close to answering
"these disagree" to everything, and any accuracy-like measure that ignores the false
alarms reads that as skill. The two reported instead are:

  conditional localization accuracy -- given the model committed to naming a view,
      how often was it the corrupted one. Immune to flag-everything behaviour,
      because a model that flags everything still has to point somewhere.
  agreement_rate -- across the k=3 draws of one item, how often all three name the
      SAME view. A model can be accurate by luck on single draws at T=0.6;
      agreement is what separates a localization from a sample of a prior.

Every metric is computed on the intersection of item ids across all eight models,
so no model is scored on an easier subset than another.

Bootstrap resamples SOLVER SYSTEMS, not draws. There are 32 solvers behind 1024
items behind 3072 draws; resampling draws would report an interval roughly sqrt(96)
too narrow for the number of independent physical systems actually observed.
"""
import os

import numpy as np
from matplotlib.patches import Patch
import pandas as pd

from . import style
from .constants import MODALITY_LABELS, MODALITIES, NONE

import matplotlib.pyplot as plt

ROLES_YAML = "data/models.yaml"
N_BOOT = 2000
BOOT_SEED = 20260822
# The intersection must retain most of the smallest model's items, or the models are
# not being compared on the same material and no per-model difference is readable.
MIN_INTERSECTION_FRAC = 0.80


def load_roles(path=ROLES_YAML):
    """Declared ladder/reference split. Raises rather than guessing."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("ladder", "reference", "band"):
        if key not in cfg:
            raise ValueError(f"{path} is missing '{key}'")
    return cfg


def assert_scale_band(cfg):
    """All eight models within the declared parameter band.

    Roughly-held scale is what makes a cross-model comparison mean anything, so this
    is asserted rather than assumed, and the measured range is returned for the
    caption instead of the caption repeating a number nobody checked.
    """
    ps = [m["params_b"] for m in cfg["ladder"] + cfg["reference"]]
    lo, hi = cfg["band"]["params_b_min"], cfg["band"]["params_b_max"]
    bad = [m for m in cfg["ladder"] + cfg["reference"]
           if not (lo <= m["params_b"] <= hi)]
    if bad:
        raise ValueError("outside the declared parameter band "
                         f"{lo}-{hi}B: " + ", ".join(
                             f"{m['short']} ({m['params_b']}B)" for m in bad))
    return min(ps), max(ps)


def tidy(d, raw):
    """One row per draw, carrying only what the ladder metrics need.

    `d` is the adapter's schema frame (which resolved each model's slot answer into
    the view it actually accused) and `raw` is the results frame it was built from,
    which is where item_id and sample_idx live. They are positionally aligned by
    construction; this asserts that rather than trusting it.
    """
    if len(d) != len(raw):
        raise ValueError(f"frames not aligned: {len(d)} vs {len(raw)}")
    return pd.DataFrame({
        "model": d["model"].astype(str).values,
        "item_id": raw["item_id"].astype(str).values,
        "sample_idx": pd.to_numeric(raw["sample_idx"], errors="coerce").values,
        "solver_id": d["solver_id"].astype(str).values,
        "true_outlier": d["true_outlier"].astype(str).values,
        "pred_outlier": d["pred_outlier"].astype(str).values,
        "pred_agree": d["pred_agree"].astype(str).values,
    })


def common_items(t, models):
    """Item ids present for EVERY model, with a report of what each model lost.

    Without this the ladder's newest model -- still generating -- would be scored on
    whatever prefix of the item set it had reached, against the older models' full
    1024. Any difference would then be partly a difference in which items were seen.
    """
    per = {m: set(t.loc[t["model"] == m, "item_id"]) for m in models}
    missing = [m for m, s in per.items() if not s]
    if missing:
        raise ValueError("no rows for: " + ", ".join(missing))
    inter = set.intersection(*per.values())
    smallest = min(len(s) for s in per.values())
    frac = len(inter) / smallest if smallest else 0.0
    report = {
        "n_common": len(inter),
        "smallest_model_items": smallest,
        "frac_of_smallest": frac,
        "contributed": {m: len(s) for m, s in per.items()},
        "lost": {m: len(s) - len(inter) for m, s in per.items()},
    }
    if frac < MIN_INTERSECTION_FRAC:
        raise ValueError(
            f"common item set is {len(inter)} items, only {frac:.0%} of the "
            f"smallest model's {smallest}. Models are not being compared on the "
            f"same material; refusing to report metrics.")
    return inter, report


def _committed(s):
    """A draw that named an actual view. NONE and unparsed answers are not."""
    return s.isin(MODALITIES)


def _loc_acc(g):
    """Correct outlier | committed, on corrupted items only.

    Restricted to corrupted items because localization is undefined when there is
    no outlier to find: on a clean item the correct behaviour is to name nothing.
    """
    g = g[g["true_outlier"].ne(NONE) & _committed(g["pred_outlier"])]
    if not len(g):
        return np.nan
    return float((g["pred_outlier"] == g["true_outlier"]).mean())


def _by_item(g):
    """Per item: did all committed draws agree, and what did the majority say."""
    rows = []
    for item, sub in g.groupby("item_id", sort=False):
        c = sub[_committed(sub["pred_outlier"])]
        allc = len(c) == len(sub) and len(sub) > 0
        votes = c["pred_outlier"].value_counts()
        rows.append({
            "item_id": item,
            "solver_id": sub["solver_id"].iloc[0],
            "true_outlier": sub["true_outlier"].iloc[0],
            "all_committed": allc,
            "unanimous": bool(allc and len(votes) == 1),
            "modal": votes.index[0] if len(votes) else "",
        })
    return pd.DataFrame(rows)


def _agreement(items):
    """P(all k draws name the same view | all k committed)."""
    e = items[items["all_committed"]]
    return float(e["unanimous"].mean()) if len(e) else np.nan


def _modal_acc(items):
    """Accuracy of the majority-vote outlier, corrupted items only."""
    e = items[items["true_outlier"].ne(NONE) & items["modal"].ne("")]
    return float((e["modal"] == e["true_outlier"]).mean()) if len(e) else np.nan


def _ratio_ci(counts, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI for a ratio, resampling SOLVER SYSTEMS with replacement.

    `counts` is one (numerator, denominator) pair PER SOLVER, reduced once before
    any resampling. That reduction is what makes this affordable: the first version
    re-derived per-item statistics inside every bootstrap iteration, which is
    2,000 iterations x 8 models x 3 metrics x ~1,000 items of Python-level groupby
    -- it did not finish, and the report build hung silently behind it. Resampling
    32 precomputed pairs is arithmetic.

    The unit is still the solver: 32 systems sit behind 1024 items behind 3072
    draws, and resampling the larger unit would report an interval far too narrow
    for the number of independent physical systems actually observed.
    """
    num = np.asarray([c[0] for c in counts], dtype=float)
    den = np.asarray([c[1] for c in counts], dtype=float)
    if len(num) < 2 or den.sum() <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    n = num[idx].sum(axis=1)
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(d > 0, n / d, np.nan)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _loc_counts(g):
    """Per solver: (correct, committed) over corrupted items."""
    e = g[g["true_outlier"].ne(NONE) & _committed(g["pred_outlier"])]
    if not len(e):
        return []
    ok = (e["pred_outlier"] == e["true_outlier"])
    by = e.groupby("solver_id", sort=False)
    return list(zip(ok.groupby(e["solver_id"], sort=False).sum(), by.size()))


def _item_counts(items, num_col, mask):
    """Per solver: (numerator, eligible) over items."""
    e = items[mask]
    if not len(e):
        return []
    by = e.groupby("solver_id", sort=False)
    return list(zip(by[num_col].sum(), by.size()))


def best_constant_baseline(t, items):
    """Best score available from always naming one fixed view.

    A model that learned nothing but "blame the code" scores this. Reporting
    localization accuracy without it invites reading a prior as a skill.
    """
    e = t[t["item_id"].isin(items) & t["true_outlier"].ne(NONE)]
    if not len(e):
        return np.nan, ""
    rates = {m: float((e["true_outlier"] == m).mean()) for m in MODALITIES}
    best = max(rates, key=rates.get)
    return rates[best], best


def per_model(t, items, models, n_boot=N_BOOT):
    """Every metric for every model, on the common item set only."""
    out = []
    for m in models:
        g = t[(t["model"] == m) & t["item_id"].isin(items)].copy()
        it = _by_item(g)
        eligible_agr = it["all_committed"]
        eligible_mod = it["true_outlier"].ne(NONE) & it["modal"].ne("")
        it = it.assign(
            _unan=it["unanimous"].astype(float),
            _modok=(it["modal"] == it["true_outlier"]).astype(float))
        loc_lo, loc_hi = _ratio_ci(_loc_counts(g), n_boot)
        agr_lo, agr_hi = _ratio_ci(_item_counts(it, "_unan", eligible_agr), n_boot)
        mod_lo, mod_hi = _ratio_ci(_item_counts(it, "_modok", eligible_mod), n_boot)
        out.append({
            "model": m, "n_items": int(g["item_id"].nunique()),
            "n_draws": int(len(g)), "n_solvers": int(g["solver_id"].nunique()),
            "loc_acc": _loc_acc(g), "loc_lo": loc_lo, "loc_hi": loc_hi,
            "agreement_rate": _agreement(it), "agr_lo": agr_lo, "agr_hi": agr_hi,
            "modal_accuracy": _modal_acc(it), "mod_lo": mod_lo, "mod_hi": mod_hi,
        })
    return pd.DataFrame(out)


def verdict_line(lad, ref, metric="loc_acc", label="conditional localization accuracy"):
    """The generated verdict. inside/outside is COMPUTED, never written by hand.

    The comparison that matters is not whether the ladder moved but whether it moved
    by more than four contemporary models of the same size differ from each other.
    """
    lad = lad.dropna(subset=[metric])
    if len(lad) < 2:
        return "Too few ladder models with data to state a change."
    a, b = lad[metric].iloc[0], lad[metric].iloc[-1]
    delta = (b - a) * 100
    lo_c, hi_c = f"{metric[:3]}_lo", f"{metric[:3]}_hi"
    lo_c = "loc_lo" if metric == "loc_acc" else "agr_lo"
    hi_c = "loc_hi" if metric == "loc_acc" else "agr_hi"
    # Delta CI from the endpoint intervals: if they overlap, zero is credible.
    d_lo = (lad[metric].iloc[-1] - lad[hi_c].iloc[0]) * 100
    d_hi = (lad[hi_c].iloc[-1] - lad[lo_c].iloc[0]) * 100
    d_lo = min(d_lo, (lad[lo_c].iloc[-1] - lad[hi_c].iloc[0]) * 100)
    spread = (ref[metric].max() - ref[metric].min()) * 100 if len(ref) else np.nan

    first, last = lad["short"].iloc[0], lad["short"].iloc[-1]
    if d_lo <= 0 <= d_hi:
        return (f"Across four Qwen releases ({first} to {last}), {label} moves from "
                f"{a * 100:.0f}% to {b * 100:.0f}% ({delta:+.0f}pp, 95% CI "
                f"{d_lo:+.0f} to {d_hi:+.0f}pp). That interval includes zero, so "
                f"the ladder shows no reliable change.")
    word = "outside" if abs(delta) > spread else "inside"
    return (f"Across four Qwen releases ({first} to {last}), {label} moves from "
            f"{a * 100:.0f}% to {b * 100:.0f}% ({delta:+.0f}pp, 95% CI "
            f"{d_lo:+.0f} to {d_hi:+.0f}pp). The spread across four contemporary "
            f"27–33B models from other families is {spread:.0f}pp, so this change "
            f"is {word} the between-model noise.")


def _spread(values, min_gap):
    """Nudge labels apart while keeping their order and staying near their values.

    Two singletons a point apart printed at their true heights overlap into an
    unreadable smear -- R1-Distill and QwQ differ by 4pp on agreement and their
    labels sat on top of each other. This keeps each label next to its own line
    without letting any two collide.
    """
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def _panel(ax, lad, ref, metric, lo_c, hi_c, title, baseline=None, ylim=None,
           baseline_view=""):
    c = style.colors()
    x = np.arange(len(lad))
    lo_y, hi_y = ylim if ylim else (0.0, 1.0)

    # Reference band FIRST, behind everything: it is context, not a series.
    ref = ref[np.isfinite(ref[metric])] if len(ref) else ref
    if len(ref):
        ax.axhspan(ref[metric].min(), ref[metric].max(), color=c["muted"],
                   alpha=0.11, zorder=0)
        ax.axhline(ref[metric].median(), color=c["muted"], linewidth=0.9,
                   alpha=0.5, zorder=1)
        # Each singleton gets a tick into the right margin at its OWN height, with
        # a leader line so a nudged label still reads against the right value. They
        # have no position on a release ordering, so they get no x.
        vals = ref[metric].to_numpy(dtype=float)
        ys = _spread(vals, min_gap=(hi_y - lo_y) * 0.052)
        xr = len(lad) - 0.5
        for (_, r), v, y in zip(ref.iterrows(), vals, ys):
            ax.plot([xr - 0.12, xr + 0.10], [v, v], color=c["muted"],
                    linewidth=0.9, alpha=0.75, clip_on=False, zorder=2)
            if abs(y - v) > 1e-9:
                ax.plot([xr + 0.10, xr + 0.22], [v, y], color=c["muted"],
                        linewidth=0.7, alpha=0.55, clip_on=False, zorder=2)
            ax.annotate(r["short"], xy=(xr + 0.24, y), va="center", ha="left",
                        fontsize=8, color=c["muted"], annotation_clip=False)

    if baseline is not None and np.isfinite(baseline):
        ax.axhline(baseline, color=c["fg"], linewidth=1.0, linestyle="--",
                   alpha=0.55, zorder=1)
        # No in-axes label. The line is named in the figure legend instead, which
        # carries both the strategy and its value -- so the mark is still decoded,
        # without a second piece of text inside a panel that already holds four
        # ladder points, their CIs, four singleton labels and a shaded band.

    yerr = np.vstack([lad[metric] - lad[lo_c], lad[hi_c] - lad[metric]])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    ax.errorbar(x, lad[metric], yerr=yerr, fmt="-o", color="#0072B2",
                ecolor="#0072B2", elinewidth=1.4, capsize=4, markersize=7,
                markeredgecolor=c["sep"], markeredgewidth=0.8, linewidth=2.0,
                zorder=3)
    for xi, v, top in zip(x, lad[metric], lad[hi_c]):
        if not np.isfinite(v):
            continue
        anchor = top if np.isfinite(top) else v
        ax.annotate(f"{v * 100:.0f}%", xy=(xi, anchor), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color="#4d9fe8", fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(lad["short"], fontsize=9)
    ax.set_xlim(-0.5, len(lad) - 0.5)
    ax.set_ylim(lo_y, hi_y)
    step = 0.05 if (hi_y - lo_y) <= 0.45 else 0.1
    ax.set_yticks(np.arange(np.ceil(lo_y / step) * step, hi_y + 1e-9, step))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=10.5, pad=16, loc="left")


def _limits(lad, ref, metrics, baseline=None):
    """Crop the y-axis to the data. A 0-100% axis spent 40% of its height empty."""
    vals = []
    for m, lo_c, hi_c in metrics:
        for f in (lad, ref):
            if not len(f):
                continue
            for col in (m, lo_c, hi_c):
                if col in f:
                    vals += [v for v in f[col].to_numpy(dtype=float)
                             if np.isfinite(v)]
    if baseline is not None and np.isfinite(baseline):
        vals.append(baseline)
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08
    return (max(0.0, np.floor((lo - pad) * 20) / 20), min(1.03, hi + pad * 1.6))


def fig8_ladder(lad, ref, baseline=None, baseline_view=""):
    """Two panels SIDE BY SIDE on a shared ordinal release axis. No regression line.

    Was stacked, on the reasoning that two columns at the 5.5in text width left each
    panel about two inches of plot area -- too narrow for unrotated tick labels and
    for the singleton names, which sit OUTSIDE the axes on the right. That reasoning
    was sound at 5.5in and is what changed: this figure is now laid out at ~1.95x the
    text column, so each panel gets more plot area side by side than it had stacked,
    and the pair fits on one screen instead of two.

    The right margin per panel is reserved explicitly through `wspace`. Without it
    the left panel's singleton labels, which are drawn with clip_on=False, print
    straight over the right panel's y-axis.
    """
    style.apply(style.theme())
    metrics = [("loc_acc", "loc_lo", "loc_hi"),
               ("agreement_rate", "agr_lo", "agr_hi")]
    ylim = _limits(lad, ref, metrics, baseline)
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.95, 4.15), sharey=True)
    _panel(axes[0], lad, ref, *metrics[0],
           "Localization accuracy, given the model committed to a view",
           baseline=baseline, ylim=ylim, baseline_view=baseline_view)
    _panel(axes[1], lad, ref, *metrics[1],
           "Agreement: all 3 draws named the same view", ylim=ylim)
    axes[0].set_ylabel("proportion", fontsize=9)
    # sharey blanks the right panel's tick LABELS but not its ticks; the labels are
    # what would be duplicated, and the axis is shared so one copy is the right count.
    axes[1].tick_params(labelleft=False)
    # right: outer margin for the SECOND panel's singleton labels.
    # wspace: the gap that holds the FIRST panel's. Both are label room, not padding.
    fig.subplots_adjust(right=0.855, wspace=0.42, top=0.845, bottom=0.225,
                        left=0.075)
    # Title built from the DATA, not written out. Two things were wrong with the
    # literal: "singletons" is this codebase's private word for "one release from a
    # lab, with no siblings in the roster" and means nothing to a reader; and the
    # "27-33B" was the DECLARED band from models.yaml, not the range these eight
    # models actually occupy (27.8-32.8B), so it was a hardcoded number standing in
    # for a measured one.
    sizes = [v for v in list(lad.get("params_b", [])) + list(ref.get("params_b", []))
             if np.isfinite(v)]
    band = (f"{min(sizes):.1f}\u2013{max(sizes):.1f}B" if sizes else "similar-size")
    fig.suptitle(f"Four Qwen releases, against four contemporary {band} models "
                 f"from other labs", fontsize=11.5, y=0.975)
    # A legend, because THREE marks on this figure carry meaning and none of them
    # said so: the shaded band, the line through it, and the dashed floor. A reader
    # had to leave the figure and find the caption to decode any one of them.
    c = style.colors()
    handles = [
        plt.Line2D([], [], color="#0072B2", marker="o", linewidth=2.0,
                   markersize=6, label="the 4 Qwen releases, oldest to newest (95% CI)"),
        # These two entries name the statistic AND point at where the reader can
        # see it. "range" and "median" alone said what the marks WERE without
        # saying what they were OF -- and the answer is already on the panel: the
        # band spans the four ticks in the right margin, which carry the model
        # names. Tying the legend to those ticks is what makes the band legible.
        Patch(facecolor=c["muted"], alpha=0.11,
              label="shaded band = spread of the 4 models named at right"),
        plt.Line2D([], [], color=c["muted"], linewidth=0.9, alpha=0.5,
                   label="line = their median"),
    ]
    if baseline is not None and np.isfinite(baseline):
        # Value lives here now. Generic wording plus the number, so the legend says
        # what the floor IS without asserting which view produces it inside the
        # panel -- and it updates from the data if the dominant class ever changes.
        who = MODALITY_LABELS.get(baseline_view, baseline_view)
        handles.append(plt.Line2D(
            [], [], color=c["fg"], linewidth=1.0, linestyle="--", alpha=0.55,
            label=(f"floor: always name one fixed view ({baseline * 100:.0f}%"
                   + (f", &ldquo;{who}&rdquo;" if False else f" \u2014 {who}")
                   + ")")))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.055))
    return fig


def fig8_ladder_dated(lad, ref, baseline=None):
    """The same two panels with calendar date on x. CONFOUNDED — details block only.

    Kept out of the headline because four points cannot support a time axis, and
    because the gaps between Qwen releases encode nothing the ordinal axis does not:
    a reader sees a slope and reads a rate, when what exists is four measurements.
    """
    style.apply(style.theme())
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.0, 3.4))
    c = style.colors()
    x = pd.to_datetime(lad["release"]).map(lambda v: v.toordinal()).to_numpy(float)
    for ax, metric, lo_c, hi_c, title in (
            (axes[0], "loc_acc", "loc_lo", "loc_hi",
             "Localization accuracy | committed"),
            (axes[1], "agreement_rate", "agr_lo", "agr_hi",
             "Agreement across 3 draws")):
        if len(ref):
            ax.axhspan(ref[metric].min(), ref[metric].max(), color=c["muted"],
                       alpha=0.13, zorder=0)
        if metric == "loc_acc" and baseline is not None and np.isfinite(baseline):
            ax.axhline(baseline, color=c["fg"], linewidth=1.0, linestyle="--",
                       alpha=0.6)
        yerr = np.vstack([lad[metric] - lad[lo_c], lad[hi_c] - lad[metric]])
        ax.errorbar(x, lad[metric], yerr=np.where(np.isfinite(yerr), yerr, 0.0),
                    fmt="-o", color="#0072B2", ecolor="#0072B2", elinewidth=1.1,
                    capsize=3, markersize=5, linewidth=1.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s)[:7] for s in lad["release"]], rotation=25,
                           ha="right", fontsize=6.6)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8.5, pad=5)
    axes[0].set_ylabel("proportion")
    fig.suptitle("Same data on a calendar axis — CONFOUNDED, shown for completeness",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    return fig


def by_true_outlier(t, items, models):
    """Localization accuracy split by which view was actually corrupted.

    Exploratory: the split divides each model's corrupted items four ways, so the
    per-cell n is small and no single cell should be quoted on its own.
    """
    rows = []
    for m in models:
        g = t[(t["model"] == m) & t["item_id"].isin(items)]
        for view in MODALITIES:
            sub = g[g["true_outlier"].eq(view) & _committed(g["pred_outlier"])]
            rows.append({"model": m, "true_outlier": view, "n": int(len(sub)),
                         "loc_acc": (float((sub["pred_outlier"] == view).mean())
                                     if len(sub) else np.nan)})
    return pd.DataFrame(rows)
