"""Claim-driven results report: one question per section, verdict before figure.

Every sentence a reader sees is generated in claims.py from the frame. There is no
hand-written interpretive path in this file -- the only prose here is the fixed
question wording and structural labels. A verdict whose interval crosses the null
renders as "inconclusive" because that is what the dataclass says, not because
someone remembered to soften it.

Figures are inline SVG so they stay sharp and text-searchable, and so individual
cells can carry ids: every bar segment and blame-matrix cell is clickable and opens
the runs behind it. Sampling is seeded, so the same report regenerates identically.
"""
import base64
import html
import io
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import claims as C
from . import metrics as M
from . import style
from .constants import (CONDITIONS, MODALITIES, MODALITY_LABELS, NAMING_LEVELS,
                        MODALITY_COLORS, NONE, NONE_COLOR, OUTLIER_LEVELS)

SEED = 20260820
SAMPLES_PER_CELL = 5
# The FROZEN artifact this report is built from, not the 8-model consolidated
# roster. consistency_claims.html reports the original three models only.
HF_URL = "https://huggingface.co/datasets/bermaneh/pde-llm-eval-xmodal-consistency-frozen-v1"


def _svg(fig):
    """Inline SVG, with the XML prologue stripped so it embeds in the document."""
    buf = io.StringIO()
    # `Date: None` drops the <dc:date> stamp matplotlib otherwise writes into every
    # SVG. Together with the `svg.hashsalt` pinned in style.RC (which fixes the
    # otherwise-random clip-path and glyph ids) this makes the report byte-identical
    # across rebuilds, so an md5 comparison against the published file actually means
    # "nothing changed" instead of "you rebuilt it".
    fig.savefig(buf, format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _esc(x):
    return html.escape(str(x if x is not None else ""))


# ── drill-down ───────────────────────────────────────────────────────────────
def build_drilldown(d, defects=None):
    """cell id -> up to N sampled runs. Deterministic under a fixed seed.

    Seeded and index-sorted so the same frame always yields the same five traces:
    a report whose examples reshuffle on every rebuild cannot be cited in a paper.
    """
    rng = np.random.default_rng(SEED)
    defects = defects or {}
    out = {}

    def rec(i):
        return {
            "run_id": str(d.at[i, "run_id"]),
            "solver_id": str(d.at[i, "solver_id"]),
            "condition": str(d.at[i, "condition"]),
            "naming": str(d.at[i, "naming"]),
            "reasoning": str(d.at[i, "reasoning"]),
            "model": str(d.at[i, "model"]),
            "true_outlier": str(d.at[i, "true_outlier"]),
            "pred_outlier": str(d.at[i, "pred_outlier"]),
            "pred_agree": str(d.at[i, "pred_agree"]),
            "justification": str(d.at[i, "justification"])[:2000],
            "defect": str(defects.get(str(d.at[i, "solver_id"]), "")),
        }

    def sample(sub, key):
        if sub.empty:
            return
        idx = sorted(sub.index.tolist())
        if len(idx) > SAMPLES_PER_CELL:
            pick = rng.choice(len(idx), SAMPLES_PER_CELL, replace=False)
            idx = [idx[i] for i in sorted(pick)]
        out[key] = {"n": int(len(sub)), "runs": [rec(i) for i in idx]}

    for cond in CONDITIONS:
        for pred in OUTLIER_LEVELS:
            sample(d[d["condition"].eq(cond) & d["pred_outlier"].eq(pred)],
                   f"cell|{cond}|{pred}")
        sample(d[d["condition"].eq(cond)], f"bar|{cond}")
    # Q2's stacked segments are keyed by (true_outlier, pred_outlier) and restricted
    # to the flagged subset, which is the subset that figure is actually about.
    det = d[d["pred_agree"].eq("no")]
    for cond in CONDITIONS:
        for pred in OUTLIER_LEVELS:
            sample(det[det["condition"].eq(cond) & det["pred_outlier"].eq(pred)],
                   f"tp|{cond}|{pred}")
    return out


# ── figures ──────────────────────────────────────────────────────────────────
def _baseline_lines(ax, base, key, xmax):
    """Draw the reference strategies, labelled on the plot rather than in a legend."""
    c = style.colors()
    for i, (name, spec) in enumerate(base.items()):
        v = spec.get(key)
        if v is None or not np.isfinite(v):
            continue
        ax.axhline(v, color=c["muted"], linewidth=0.8,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(spec["label"], (xmax, v), xytext=(3, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=style.ANNOT_PT - 0.5, color=c["muted"])


def fig_rate_by_condition(d, kind, base):
    """One varying factor: condition. Everything else pooled and stated in caption."""
    style.apply(style.theme())
    c = style.colors()
    dd = M.prepare(d)
    rows = []
    for cond in CONDITIONS:
        sub = dd[dd["condition"].eq(cond)]
        if kind == "detection":
            k, n = int(sub["detected"].sum()), len(sub)
        else:
            sub = sub[sub["localization_eligible"]]
            k, n = int(sub["localization_correct"].sum()), len(sub)
        if not n:
            continue
        lo, hi = M.wilson_ci(k, n)
        rows.append((cond, k / n, lo, hi, n))
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.9))
    if not rows:
        style.empty_axes(ax, "no rows")
        return _svg(fig)
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    err = np.array([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]])
    for i, (cond, v, lo, hi, n) in enumerate(rows):
        thin = n < C.MIN_N
        b = ax.bar(i, v, width=0.66,
                   color=c["bar"] if not thin else c["faint"],
                   hatch="///" if thin else "",
                   edgecolor=c["muted"] if thin else "none",
                   linewidth=0.6, **style.hatch_kw())
        b[0].set_gid(f"bar|{cond}")
        ax.annotate(f"n={n:,}", (i, 0), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=style.ANNOT_PT - 1,
                    color=c["panel"] if not thin else c["muted"])
    ax.errorbar(x, vals, yerr=np.nan_to_num(err), fmt="none",
                ecolor=c["fg"], elinewidth=0.8, capsize=2)
    _baseline_lines(ax, base, kind, len(rows) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0].replace("A-T-", "T:") for r in rows],
                       rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("% flagged as inconsistent" if kind == "detection"
                  else "% naming the right view")
    ax.grid(True, axis="y", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    fig.subplots_adjust(right=0.72)
    return _svg(fig)



def _obfuscation_design_note(d):
    """State the manipulation before showing its effect.

    Every row of this section is a difference between two versions of the SAME item,
    and nothing on the figure says what the two versions are. Without that a reader
    cannot tell which experiment they are looking at, let alone what a positive
    number means.
    """
    n_real = int((d["naming"] == NAMING_LEVELS[0]).sum())
    n_obf = int((d["naming"] == NAMING_LEVELS[1]).sum())
    n_solv = int(d["solver_id"].nunique())
    return (
        '<div class="designnote">'
        '<b>What is manipulated.</b> Every item is shown twice: once with the '
        'solver\u2019s real variable names, and once with those names replaced by '
        'meaningless ones. Nothing else changes \u2014 same solver, same corruption, '
        'same four views, identical physics. '
        f'{n_real:,} items with real names, {n_obf:,} obfuscated, across '
        f'{n_solv} solvers.<br>'
        '<b>The question.</b> Do meaningless identifiers weaken the model\u2019s '
        'lexical prior on the code, and does that make it better at identifying '
        'which representation is actually the outlier?<br>'
        '<b>How it is measured.</b> By CORRECTNESS, not by where blame lands. The '
        'primary outcome is: given the model committed to a verdict, how often that '
        'verdict named the view that was really corrupted. Every number below is '
        '<i>obfuscated minus real</i> on the same solver, so zero means '
        '\u201cthe names made no difference\u201d.'
        '</div>')


def _obfuscation_block(d):
    """(svg, stats, caption) for the prior-weakening section."""
    from . import figures as F
    try:
        fig, r, cap = F.fig7_prior_weakening(d)
        stmt, _ = F.evidence_statement(r)
        return f'<p class="promoted">{_esc(stmt)}</p>' + _svg(fig), r, cap
    except Exception as exc:                                    # noqa: BLE001
        return None, None, f"Could not render the figure: {exc}"


def _obfuscation_split_block(d):
    """The same contrast with trajectory opened into its four corruption methods.

    A second figure rather than a replacement for the first. The pooled figure is the
    one carrying the tested contrast; this one is the follow-up question -- do the
    four trajectory corruptions respond to obfuscation alike -- and it is drawn below
    with its own caption so the exploratory rows cannot be mistaken for the result.
    Returns "" on failure so a missing companion never takes the section down with it.
    """
    from . import figures as F
    try:
        fig, _r, cap = F.fig7b_prior_weakening_split(d)
    except Exception as exc:                                    # noqa: BLE001
        return f'<p class="figcap">Could not render the split figure: {_esc(exc)}</p>'
    return ('<h4 class="subfig">Trajectory, split into its four corruptions</h4>'
            + _svg(fig) + f'<p class="figcap">{cap}</p>')


def _inject_titles(svg, tips):
    """Give gid'd SVG elements a <title> child, which browsers show as a tooltip.

    matplotlib writes gid as the element id but has no notion of hover text, so the
    tooltips are attached here rather than faked with JS overlays that would not
    survive the figure being saved out of the page.
    """
    for gid, text in tips.items():
        needle = f'id="{gid}"'
        i = svg.find(needle)
        if i == -1:
            continue
        j = svg.find(">", i)
        if j == -1:
            continue
        if svg[j - 1] == "/":                       # self-closing: expand it
            tag_start = svg.rfind("<", 0, i)
            tag_name = svg[tag_start + 1:svg.find(" ", tag_start)]
            svg = (svg[:j - 1] + f"><title>{_esc(text)}</title></{tag_name}>"
                   + svg[j + 1:])
        else:
            svg = svg[:j + 1] + f"<title>{_esc(text)}</title>" + svg[j + 1:]
    return svg


def fig_sensitivity(d, tier=None, verbose=False):
    """Single-panel dot plot. Flag rate per corrupted condition, against the A0 floor.

    Replaces a two-panel version that encoded the same quantity twice (rate on the
    left, d' as bars on the right), hid the baseline in an unlabelled dashed line,
    and fixed the x-range to [0, 1] so most of the panel was empty. Here the baseline
    is a DATA ROW -- the thing every other row has to be read against -- and d' is a
    text column, because it is a derived number, not a second measurement.

    Trajectory is disaggregated: its four generation methods differ more from each
    other than whole modalities do.
    """
    from .sensitivity import detection_sensitivity, severity_tiers
    style.apply(style.theme())
    c = style.colors()
    src = d if tier is None else _restrict_tier(d, tier)
    r = detection_sensitivity(src)
    rows = [x for x in r.rows if not x["empty"]]
    # Wide and short. The old allocation gave every row 0.42in and then added four
    # rows' worth of padding on top of a 1.0in constant, which on a seven-row figure
    # spent more than half the panel on margin, group gaps and the trailing note. The
    # row pitch is what has to be legible; everything else is packing.
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.26 * (len(rows) + 2.0) + 0.85))
    if not rows or not np.isfinite(r.fa_rate):
        style.empty_axes(ax, "no rows")
        return _svg(fig)

    # Grouped, not globally sorted. A single sorted list asserts a ranking ACROSS
    # representations, which this design cannot support: trajectory carries four
    # corruption methods and the other views carry one each. Grouping makes the
    # comparison the data does support -- within trajectory, and among the
    # single-method views -- the local one, and stops the eye making the other.
    traj = sorted([x for x in rows if x["condition"].startswith("A-T-")],
                  key=lambda x: -x["hit_rate"])
    other = sorted([x for x in rows if not x["condition"].startswith("A-T-")],
                   key=lambda x: -x["hit_rate"])
    # The blocks carry no in-figure headers. The gap between them is the whole signal:
    # every row already names its own corruption, so a banner over each block restated
    # what the labels say and cost a line of height per group. Why the blocks exist --
    # trajectory's four methods are comparable with each other, the three
    # single-method views with each other, and the two sets are not severity-matched
    # -- is in the caption, which is where an argument about the design belongs.
    groups = [g for g in (traj, other) if g]
    ordered, y_corrupt = [], []
    y = 0.0
    for members in reversed(groups):
        for x in reversed(members):          # highest rate at the TOP of its block
            y += 1.0
            ordered.append(x)
            y_corrupt.append(y)
        y += 0.6
    rows = ordered
    y_base = -0.75

    # Axis limits from the data, rounded outward to the nearest 0.05.
    lo_v = min([x["lo_rate"] for x in rows] + [r.fa_rate])
    hi_v = max([x["hi_rate"] for x in rows] + [r.fa_rate])
    xlo = max(0.0, np.floor((lo_v - 0.05) / 0.05) * 0.05)
    xhi = min(1.0, np.ceil((hi_v + 0.05) / 0.05) * 0.05)

    # The floor, shaded across the whole plot: everything left of the dashed line is
    # what the model does when nothing is wrong.
    ax.axvspan(xlo, r.fa_rate, color=c["muted"], alpha=0.10, zorder=0)
    ax.axvline(r.fa_rate, color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)

    tips = {}
    for yi, x in zip(y_corrupt, rows):
        col = MODALITY_COLORS[x["modality"]] if not x["thin"] else c["muted"]
        ax.plot([x["lo_rate"], x["hi_rate"]], [yi, yi], color=col, linewidth=1.3,
                solid_capstyle="round", zorder=3)
        dot = ax.scatter([x["hit_rate"]], [yi], s=46, color=col, zorder=4)
        dot.set_gid(f"hit|{x['condition']}")
        tips[f"hit|{x['condition']}"] = (
            f"{x['label']}: flagged {x['n_hit']} of {x['n_signal']} "
            f"({100 * x['hit_rate']:.1f}%), d'={x['dprime']:.2f}")
        # No d' column: it is a monotone transform of this dot's distance from the
        # same baseline, so printing it beside the dot encodes one quantity twice.
        # The values are in the details table.
    # Baseline as a data row, below the rule, always last, always grey.
    ax.plot([r.fa_lo, r.fa_hi], [y_base, y_base], color=c["muted"], linewidth=1.3,
            solid_capstyle="round", zorder=3)
    ax.scatter([r.fa_rate], [y_base], s=46, color=c["muted"], zorder=4)
    ax.axhline(0.15, color=c["muted"], linewidth=0.6)
    ax.set_yticks(y_corrupt + [y_base])
    # `x["label"]` is the terse form and still carries the generator's own codes --
    # "trajectory - exec corrupted". This figure accepted `verbose` and then never
    # used it, so the four trajectory rungs kept their internal names here long after
    # the blame figures had been given readable ones, and the report spoke two
    # vocabularies for the same four rows. In the clear form the trailing "corrupted"
    # goes too: both group headers already say it, and the baseline row still names
    # itself. The terse branch is what the frozen report rebuilds from.
    if verbose:
        from .sensitivity import row_caption_corrupted
        labels = ([row_caption_corrupted(x["condition"]) for x in rows]
                  + ["nothing corrupted"])
    else:
        labels = [x["label"] + " corrupted" for x in rows] + ["nothing corrupted"]
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_color(c["muted"])
    ax.set_ylim(y_base - 0.55, max(y_corrupt) + 0.55)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(np.arange(xlo, xhi + 1e-9, 0.05))
    ax.set_xticklabels([f"{100 * t:.0f}%" for t in np.arange(xlo, xhi + 1e-9, 0.05)])
    ax.set_xlabel("items the model flagged as disagreeing")
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    # Offset in POINTS, not axes fraction. As a fraction of a shorter axes the same
    # -0.30 lands on top of the x-label; a fixed offset keeps the note the same
    # distance below the tick labels whatever height the figure ends up at.
    ax.annotate(f"Shaded region: flagged even when all four representations agree "
                f"({100 * r.fa_rate:.1f}%).",
                (0, 0), xycoords="axes fraction", xytext=(0, -30),
                textcoords="offset points", fontsize=style.ANNOT_PT,
                color=c["muted"], va="top")
    return _inject_titles(_svg(fig), tips)


def _restrict_tier(d, tier):
    t = d.get("traj_level", "").fillna("").astype(str)
    return d[np.where(t.eq(""), "single", t) == tier]


def fig_sensitivity_matched(d):
    """The same figure restricted to a severity tier common to all four modalities.

    Returns (svg or None, note). There is no such tier in this dataset -- trajectory
    carries four generation methods and the other views carry one each -- so this
    returns None and says why rather than inventing a comparison.
    """
    from .sensitivity import severity_tiers
    _, matched, common = severity_tiers(d)
    if matched:
        return None, "Corruptions are severity-matched; no restricted figure needed."
    if not common:
        return None, ("No severity tier is present for all four representations "
                      "(trajectory has four generation methods: rand, shuf, swap, "
                      "exec; code, description and math have one each), so no "
                      "severity-matched version of this figure can be drawn.")
    return fig_sensitivity(d, tier=common[0]), (
        f"Restricted to the severity tier common to all four representations: "
        f"{common[0]}.")


# ── shared blame-category bookkeeping ────────────────────────────────────────
# Named at module scope rather than inside the figure that first used them,
# because two figures now draw these bars and a category whose definition lives
# in one of them is a category the other can silently redefine.
MISS = "__miss__"            # flagged nothing: said the representations agree
UNCLEAR = "__unclear__"      # flagged it, but named no view we can read
NOVERDICT = "__noverdict__"  # never finished: budget truncation or a decode loop


def _blame_order_labels():
    order = list(MODALITIES) + [NONE, UNCLEAR, MISS, NOVERDICT]
    label = {**MODALITY_LABELS, NONE: "named none",
             UNCLEAR: "flagged, view unreadable",
             MISS: "does not identify disagreement",
             NOVERDICT: "no verdict \u2014 truncated or looping"}
    return order, label


def _blame_frame(src):
    """`prepare` plus the no-verdict flag, which the schema does not carry."""
    p = M.prepare(src)
    p["no_verdict"] = (src["no_verdict"].to_numpy()
                       if "no_verdict" in src.columns else False)
    return p


def _classify_blame(sub):
    """(counts_by_category, total) for one already-selected set of draws."""
    total = int(len(sub))
    if total == 0:
        return {}, 0
    # Order matters. A no-verdict draw is classified BEFORE the flag/miss split:
    # most of them carry an `agree=yes` the regex scavenged out of reasoning the
    # model never finished, so treating them as answers would file them under
    # "does not identify disagreement" and manufacture that finding.
    nv = sub["no_verdict"].astype(bool)
    counts = {NOVERDICT: int(nv.sum())}
    ans = sub[~nv]
    det = ans["detected"]
    named = ans["pred_outlier"].where(det)
    for m in MODALITIES:
        counts[m] = int(named.eq(m).sum())
    counts[NONE] = int(named.eq(NONE).sum())
    counts[MISS] = int((~det).sum())
    # Whatever is left is a flagged row whose named view we could not read. Derived
    # by subtraction rather than by a parse predicate so the segments are guaranteed
    # to sum to the row total -- a stacked bar that silently drops rows is worse than
    # one that shows an unexplained sliver.
    counts[UNCLEAR] = total - sum(counts.values())
    return counts, total


def _unconditional_counts(src):
    """{condition: (counts_by_category, row_total)} over EVERY corrupted draw.

    The arithmetic behind the unconditional blame bars, lifted out of the figure
    that used to own it so the paired figure draws the same numbers rather than a
    second implementation of them.
    """
    from .sensitivity import SIGNAL_CONDITIONS
    p = _blame_frame(src)
    corrupted = p[p["is_corrupted"]]
    return {cond: _classify_blame(corrupted[corrupted["condition"].eq(cond)])
            for cond in SIGNAL_CONDITIONS}


def _clean_counts(src):
    """The same categories over the CONTROL items, where nothing was corrupted.

    The unconditional figure leaves this row out because there is no outlier to
    localize on it. But the model still answers: on roughly half these items it says
    the four representations disagree and then names one. Those names are the only
    place in the data where its blame prior is visible with NO signal to respond to
    -- whatever it points at here, it points at without evidence. On this row the
    correct answer is the "says all four agree" segment, which is why the paired
    figure marks correctness with a caret under the segment rather than by outlining
    it: the meaning of a category flips between this row and the seven above it.
    """
    p = _blame_frame(src)
    return _classify_blame(p[~p["is_corrupted"]])


def fig_blame_stack_unconditional(d, d_all=None, verbose=False):
    """The blame figure with EVERY corrupted item in the denominator.

    `fig_blame_stack` conditions on the model having flagged the item, which answers
    "when it says something disagrees, does it know which thing". That is the right
    denominator for that question, but it hides how often the model never gets to
    the question at all: a model that flags 10% of corruptions and localizes those
    perfectly scores 100% there, identically to one that flags everything and is
    always right.

    So this version divides by all corrupted items for the condition and adds the
    category the conditional figure cannot show -- the model said the four
    representations AGREE, on an item where one of them was corrupted. Read together,
    the first figure is skill-given-attempt and this one is skill-per-opportunity.

    Deliberately NOT clickable. The drill-down keys (`tp|cond|cat`) address the
    conditional table, so wiring the same gids here would open a set of runs whose
    count disagrees with the bar the reader just clicked.
    """
    from .constants import (MODALITY_LABELS as ML, MODALITY_COLORS, NONE_COLOR,
                            NONE, MODALITIES, CONDITION_OUTLIER)
    from .sensitivity import SIGNAL_CONDITIONS, row_label, row_caption
    from . import metrics as M
    style.apply(style.theme())
    c = style.colors()

    # `d_all` carries every draw the models wrote, no-verdict ones included. Without
    # it the row totals are unequal -- the exclusion is not uniform across conditions
    # -- and the shares rest on a denominator that quietly lost the model's longest
    # deliberations.
    src = d if d_all is None else d_all
    tally = _unconditional_counts(src)
    # Same treatment as the conditional figure above: height from the row count, and
    # the legend beside the axes rather than under them. This one had the worse of
    # the two layouts -- a seven-entry legend in three columns at y=-0.42, which is
    # nearly three data rows of height spent naming the segments.
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.235 * len(SIGNAL_CONDITIONS)
                                                 + 0.95))
    if not any(t for _, t in tally.values()):
        style.empty_axes(ax, "no corrupted items")
        return _svg(fig)

    order, label = _blame_order_labels()

    # No pooled row. Pooling across conditions averages corruptions that differ by
    # orders of magnitude in detectability (trajectory-rand vs code), so the summary
    # bar reads as a fact about the model when it is mostly a fact about the mix.
    rows = list(SIGNAL_CONDITIONS)
    ypos = [len(rows) - i for i in range(len(rows))]
    # Which categories actually occur, so the legend cannot advertise an empty one.
    # `none` is the pred_outlier the parser assigns exactly when the model said the
    # views AGREE, which is already the MISS bucket -- so "flagged, named none" is
    # empty by construction in this dataset (0 of 14,030 flagged corrupted rows).
    # Kept in the draw order rather than deleted: it costs nothing, and a future
    # parser that does emit it would otherwise drop those rows silently.
    seen = set()
    share_max = {}          # biggest share any single row gives each category
    for row_i, cond in enumerate(rows):
        yi = ypos[row_i]
        counts, total = tally.get(cond, ({}, 0))
        if total == 0:
            ax.annotate("no corrupted rows", (0.01, yi), va="center",
                        fontsize=style.ANNOT_PT, color=c["muted"])
            continue

        left = 0.0
        true_m = CONDITION_OUTLIER[cond]
        for cat in order:
            n_here = counts.get(cat, 0)
            if n_here <= 0:
                continue
            seen.add(cat)
            w = n_here / total
            share_max[cat] = max(share_max.get(cat, 0.0), w)
            if cat == MISS:
                bar = ax.barh(yi, w, left=left, height=0.62,
                              facecolor=c["panel"], hatch="///",
                              edgecolor=c["muted"], linewidth=1.2,
                              **style.hatch_kw())
            elif cat == NOVERDICT:
                bar = ax.barh(yi, w, left=left, height=0.62,
                              facecolor=c["faint"], hatch="xxx",
                              edgecolor=c["muted"], linewidth=1.2,
                              **style.hatch_kw())
            elif cat == UNCLEAR:
                bar = ax.barh(yi, w, left=left, height=0.62,
                              facecolor=NONE_COLOR, alpha=0.35,
                              edgecolor=c["panel"], linewidth=1.5)
            else:
                col = MODALITY_COLORS.get(cat, NONE_COLOR)
                bar = ax.barh(yi, w, left=left, height=0.62, color=col,
                              edgecolor=c["panel"], linewidth=1.5)
            if w >= 0.07:
                ax.text(left + w / 2, yi, f"{100 * w:.0f}%", ha="center",
                        va="center", fontsize=style.ANNOT_PT,
                        color=c["muted"] if cat == MISS else c["bg"], zorder=5)
            if cat == true_m:
                ax.barh(yi, w, left=left, height=0.62, facecolor="none",
                        edgecolor=c["fg"], linewidth=1.5, zorder=4)
            left += w
    # No per-row n. Every row of THIS figure has the same denominator by
    # construction -- the benchmark assigns each item exactly one condition, 128
    # apiece -- so printing it seven times down the margin repeats one number and
    # invites the reading that the rows have denominators worth comparing. The
    # caption states it once. (The conditional figure keeps its n's: there the
    # denominators are flagged items and genuinely do differ per row.)

    ax.set_yticks(ypos)
    ax.set_yticklabels([row_caption(c_, verbose) for c_ in rows])
    if verbose:
        # The captions no longer repeat "was corrupted" on every row, so the axis
        # carries it once. Without this the rows read as blame targets, not causes.
        ax.set_ylabel("which view was corrupted")
    # 1.13 left a margin for the per-row "n = ..." labels that used to sit outside
    # the bars. With those gone the extra 13% was dead space that pushed the 100%
    # tick away from the plot edge.
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    # Parallel to the conditional figure's "share of flagged items, by the view the
    # model blamed". The second clause is what ties the colours to the legend; without
    # it the axis describes only the denominator and the segments look unexplained.
    ax.set_xlabel("share of all items with this corruption, "
                  "by the view the model blamed")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=MODALITY_COLORS[m], label=label[m])
               for m in MODALITIES if m in seen]
    if NONE in seen:
        handles.append(Patch(facecolor=NONE_COLOR, label=label[NONE]))
    # UNCLEAR is the residual bucket -- total minus everything we could classify --
    # not a measured category, and in this dataset it is ONE draw in 21,379 (0.005%,
    # all of it in A-C). At that size the segment is well under a pixel, so a legend
    # entry names something the reader cannot find and puts a parser edge case on the
    # same footing as the four representations. It is still drawn and still counted,
    # so the segments continue to sum to the row total; only the legend line goes.
    # The threshold, not a special case for this number: if a future parse regression
    # pushes it up to a visible share, it names itself again without an edit here.
    if UNCLEAR in seen and share_max.get(UNCLEAR, 0.0) >= 0.005:
        handles.append(Patch(facecolor=NONE_COLOR, alpha=0.35,
                             label=label[UNCLEAR]))
    if MISS in seen:
        handles.append(Patch(facecolor=c["panel"], hatch="///",
                             edgecolor=c["muted"], label=label[MISS],
                             **style.hatch_kw()))
    # The outlined segment marks the row's true outlier. It is drawn on every row,
    # so it needs naming here exactly as it is in the conditional figure above --
    # an unexplained outline reads as emphasis rather than as the correct answer.
    if NOVERDICT in seen:
        handles.append(Patch(facecolor=c["faint"], hatch="xxx",
                             edgecolor=c["muted"], label=label[NOVERDICT],
                             **style.hatch_kw()))
    handles.append(Patch(facecolor="none", edgecolor=c["fg"], linewidth=1.5,
                         label="correct answer for this row"))
    # One column, to the right of the axes. The x-limit is 1.02 here (no "n = "
    # column to clear), so the legend can sit close in.
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncol=1, frameon=False, fontsize=style.ANNOT_PT,
              handlelength=1.1, handletextpad=0.5, labelspacing=0.5,
              borderaxespad=0.0)
    return _svg(fig)


# The paired figure is the one figure in this package that does NOT fit the 5.5in
# text column, and is meant not to. Two panels of seven rows each, with the row
# captions carried on a shared axis, is a landscape object: squeezed into the column
# its bars are shorter than their own labels and panel B's small segments close up.
# It is drawn to be placed full width (\textwidth in a wide template, a `figure*`,
# or a rotated float), so the type is scaled by LESS than the width -- the extra
# space goes to the data, not to bigger letters.
PAIR_WIDTH_IN = 10.6
PAIR_TYPE_SCALE = 1.35

# Two geometries for the same figure, same data and same arithmetic. "short" spends
# about a third less height for a float that has to share a page with body text, and
# it buys that back out of WHITE SPACE only -- the type scale, the panel widths and
# the bar labels are identical in both, so the compact one is not a shrunken figure,
# it is the same figure with the air taken out.
#
# The floor on `pitch` is not the bar: it is the caret, which lives in the gap
# BETWEEN rows and cannot be allowed to touch the bar above or below it. So `bar_h`
# rises as `pitch` falls -- the gap gives up proportionally more than the bar does --
# and `caret_ms` comes down with it. `span_pad` is the figsize formula's old `+ 2.0`,
# the allowance for the block gap, the control row and the axis padding, which all
# shrink together with the row pitch or the rows would spread back out to fill the
# box they were just given less of.
PAIR_GEOMETRY = {
    "default": dict(pitch=0.245, furniture=1.20, span_pad=2.00, block_gap=0.60,
                    y_base=-0.62, pad_lo=0.50, pad_hi=0.55, bar_h=0.62,
                    caret_dy=0.44, caret_ms=4.6, legend_dy=-0.02, note_dy=-0.135),
    "short": dict(pitch=0.170, furniture=1.02, span_pad=1.55, block_gap=0.42,
                  y_base=-0.56, pad_lo=0.62, pad_hi=0.46, bar_h=0.70,
                  caret_dy=0.47, caret_ms=3.9, legend_dy=-0.015, note_dy=-0.115),
}

def fig_detection_blame_pair(d, d_all=None, verbose=True,
                             width_in=PAIR_WIDTH_IN,
                             type_scale=PAIR_TYPE_SCALE,
                             geometry="default"):
    """Detection and localization on ONE shared row axis. Returns a Figure.

    The two questions this report answers in separate sections are the same
    measurement split by denominator, and separating them lets a reader carry the
    wrong impression from the first to the second. Panel A is "does the model
    notice at all" -- the flag rate for each corruption against the rate at which the model
    flags items where nothing is wrong. Panel B is "and which view does it blame" over
    the SAME denominator, every corrupted item, so the hatched segment is exactly
    the mass that never reached panel B's question.

    That identity is the reason for the shared y-axis: read across a row and the
    dot in A sits at the right-hand edge of the hatch in B. It only holds while no
    draw is missing a verdict -- those are counted in B (a consumed opportunity)
    and dropped from A (scoring a run that produced no verdict as if it had produced
    a wrong one invents an answer) -- so the residual is measured below and stated
    on the figure whenever it is not zero.

    Rows are ordered by detectability within two blocks, matching `fig_sensitivity`:
    trajectory's four generation methods are comparable with each other and the
    three single-method views with each other, but the two blocks are not
    severity-matched, so there is no global ranking to draw.

    The control row is drawn in both panels, which the standalone blame figure does
    not do. On the items where nothing was corrupted the model still says the four
    views disagree about half the time and then names one, and that distribution is
    its blame prior with no signal present -- the thing every row above it should be
    read against. Correctness is marked with a caret under the correct segment rather
    than by outlining it, because on that row the correct answer is "says all four
    agree" and an outline convention would have to invert.

    Not clickable, for the same reason the standalone unconditional figure is not:
    the drill-down is keyed on the conditional denominators.

    `geometry` picks a row pitch out of PAIR_GEOMETRY: "default" is the published
    figure, "short" is the same figure at about two thirds the height for a float
    that has to share a page. Nothing but white space differs between them.
    """
    from .sensitivity import detection_sensitivity, row_caption
    from .constants import CONDITION_OUTLIER
    style.apply(style.theme())
    c = style.colors()
    g = PAIR_GEOMETRY[geometry]
    # Every point size in this figure is derived, so the whole thing rescales from
    # `type_scale` alone rather than from fourteen literals that would drift apart.
    pt_a = style.ANNOT_PT * type_scale
    pt_b = style.BASE_PT * type_scale
    pt_t = style.TICK_PT * type_scale
    lw = type_scale                      # line weights track the type, not the width

    r = detection_sensitivity(d)
    tally = _unconditional_counts(d if d_all is None else d_all)
    rows = [x for x in r.rows if not x["empty"]]

    # Same two-block grouping as fig_sensitivity, and deliberately a copy of it
    # rather than a call into it: that function draws the frozen report's Q1 figure
    # and has to keep reproducing it byte for byte, so it is not refactored to serve
    # a second caller. The ARITHMETIC is shared (detection_sensitivity,
    # _unconditional_counts); only the ~20 lines of layout are duplicated.
    traj = sorted([x for x in rows if x["condition"].startswith("A-T-")],
                  key=lambda x: -x["hit_rate"])
    other = sorted([x for x in rows if not x["condition"].startswith("A-T-")],
                   key=lambda x: -x["hit_rate"])
    groups = [g for g in (traj, other) if g]
    ordered, ypos = [], []
    y = 0.0
    for members in reversed(groups):
        for x in reversed(members):          # highest rate at the TOP of its block
            y += 1.0
            ordered.append(x)
            ypos.append(y)
        y += g["block_gap"]
    rows = ordered
    # Closer to the block than a full row gap. The baseline is a reference row, not
    # an eighth corruption, but panel B leaves it empty and a wide gap there reads as
    # a missing row rather than as a rule.
    y_base = g["y_base"]

    fig, (axa, axb) = plt.subplots(
        1, 2, sharey=True,
        # Row pitch grows with the TYPE, not with the width -- that is what makes
        # this a landscape figure rather than a scaled-up square one. The trailing
        # constant is the axis furniture (tick labels, x-label), a fixed number of
        # text lines however many rows there are.
        figsize=(width_in,
                 g["pitch"] * type_scale * (len(rows) + g["span_pad"])
                 + g["furniture"] * type_scale),
        gridspec_kw=dict(width_ratios=[1.0, 1.12], wspace=0.09))
    axa.tick_params(labelsize=pt_t)
    axb.tick_params(labelsize=pt_t)
    if not rows or not np.isfinite(r.fa_rate):
        style.empty_axes(axa, "no rows")
        style.empty_axes(axb, "no rows")
        return fig

    # ── panel A: did it notice ────────────────────────────────────────────────
    lo_v = min([x["lo_rate"] for x in rows] + [r.fa_rate])
    hi_v = max([x["hi_rate"] for x in rows] + [r.fa_rate])
    xlo = max(0.0, np.floor((lo_v - 0.05) / 0.05) * 0.05)
    xhi = min(1.0, np.ceil((hi_v + 0.05) / 0.05) * 0.05)
    axa.axvspan(xlo, r.fa_rate, color=c["muted"], alpha=0.10, zorder=0)
    axa.axvline(r.fa_rate, color=c["muted"], linewidth=1.0 * lw,
                linestyle=(0, (4, 3)), zorder=1)
    for yi, x in zip(ypos, rows):
        col = MODALITY_COLORS[x["modality"]] if not x["thin"] else c["muted"]
        axa.plot([x["lo_rate"], x["hi_rate"]], [yi, yi], color=col,
                 linewidth=1.3 * lw, solid_capstyle="round", zorder=3)
        axa.scatter([x["hit_rate"]], [yi], s=42 * lw ** 2, color=col, zorder=4)
    axa.plot([r.fa_lo, r.fa_hi], [y_base, y_base], color=c["muted"],
             linewidth=1.3 * lw, solid_capstyle="round", zorder=3)
    axa.scatter([r.fa_rate], [y_base], s=42 * lw ** 2, color=c["muted"], zorder=4)
    axa.axhline(0.15, color=c["muted"], linewidth=0.6)
    axa.set_xlim(xlo, xhi)
    # Tick density from the panel's own range AND from how much room it got. At
    # text-column width a 5-point grid over a 60-point range put "100%" hard against
    # panel B's "0"; at full width there is room for the finer grid again.
    span = xhi - xlo
    fine = width_in >= 8.0
    step = ((0.10 if span > 0.45 else 0.05) if fine
            else (0.20 if span > 0.45 else (0.10 if span > 0.20 else 0.05)))
    ticks = np.arange(xlo, xhi + 1e-9, step)
    axa.set_xticks(ticks)
    axa.set_xticklabels([f"{100 * t:.0f}%" for t in ticks])
    axa.set_xlabel("% of items flagged as disagreeing", fontsize=pt_b)
    axa.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    axa.set_axisbelow(True)
    # "at all" is dropped, not lost: panel A's x-label ("items flagged as
    # disagreeing") already says the question is detection rather than
    # localization, and with it the title overran panel A and collided with B's.
    axa.set_title("A   Does the model notice a disagreement?", loc="left",
                  fontsize=pt_b, color=c["fg"], pad=7 * lw)

    axa.set_yticks(list(ypos) + [y_base])
    axa.set_yticklabels([row_caption(x["condition"], verbose) for x in rows]
                        + ["nothing corrupted"])
    axa.get_yticklabels()[-1].set_color(c["muted"])
    axa.set_ylim(y_base - g["pad_lo"], max(ypos) + g["pad_hi"])
    if verbose:
        # The rows are causes, not blame targets. Panel B's x-label says "by the view
        # the model blamed" and panel A's says "flagged"; without this the shared
        # captions down the left read as the answer rather than as the manipulation.
        axa.set_ylabel("which view was corrupted", fontsize=pt_b, labelpad=6 * lw)

    # ── panel B: and which view does it blame ─────────────────────────────────
    order, label = _blame_order_labels()
    # "does not identify disagreement" is a MISS on the seven corrupted rows and the
    # CORRECT answer on the control row, so in this figure the category is named for
    # what the model did rather than for whether it was right. Correctness is carried
    # by the caret instead, which can point at a different category row by row.
    label = {**label, MISS: "says all four agree"}
    seen, share_max = set(), {}
    residual = 0.0

    # The control row is drawn, not left blank. On ~47% of items where nothing was
    # corrupted the model still names a view, and that distribution is its blame
    # prior with no signal present -- the baseline every row above should be read
    # against, and the reason trajectory dominance there is a finding rather than an
    # artefact of the corruptions.
    clean_counts, clean_total = _clean_counts(d if d_all is None else d_all)
    brows = [(yi, tally.get(x["condition"], ({}, 0)), CONDITION_OUTLIER[x["condition"]],
              x["hit_rate"]) for yi, x in zip(ypos, rows)]
    if clean_total:
        brows.append((y_base, (clean_counts, clean_total), MISS, None))

    for yi, (counts, total), correct_cat, hit_rate in brows:
        if total == 0:
            axb.annotate("no corrupted rows", (0.01, yi), va="center",
                         fontsize=pt_a, color=c["muted"])
            continue
        left = 0.0
        for cat in order:
            n_here = counts.get(cat, 0)
            if n_here <= 0:
                continue
            seen.add(cat)
            w = n_here / total
            share_max[cat] = max(share_max.get(cat, 0.0), w)
            if cat == MISS:
                axb.barh(yi, w, left=left, height=g["bar_h"], facecolor=c["panel"],
                         hatch="///", edgecolor=c["muted"], linewidth=1.0 * lw,
                         **style.hatch_kw())
            elif cat == NOVERDICT:
                axb.barh(yi, w, left=left, height=g["bar_h"], facecolor=c["faint"],
                         hatch="xxx", edgecolor=c["muted"], linewidth=1.0 * lw,
                         **style.hatch_kw())
            elif cat == UNCLEAR:
                axb.barh(yi, w, left=left, height=g["bar_h"], facecolor=NONE_COLOR,
                         alpha=0.35, edgecolor=c["panel"], linewidth=1.5 * lw)
            else:
                axb.barh(yi, w, left=left, height=g["bar_h"],
                         color=MODALITY_COLORS.get(cat, NONE_COLOR),
                         edgecolor=c["panel"], linewidth=1.5 * lw)
            # The width threshold falls with the panel: at full width a 5%
            # segment is wide enough to hold its own label, and those small
            # segments are exactly what the extra width was spent on.
            if w >= (0.05 if fine else 0.08):
                axb.text(left + w / 2, yi, f"{100 * w:.0f}%", ha="center",
                         va="center", fontsize=pt_a,
                         color=c["muted"] if cat == MISS else c["bg"], zorder=5)
            # A caret under the correct segment, not a second box around it. The
            # outline this replaces was `fg` at 1.5pt sitting immediately beside the
            # hatched segment's `muted` border at 1.2pt: two dark rectangles of
            # similar weight, one meaning "right answer" and the other meaning
            # nothing, close enough that a reader could not tell at a glance which
            # segment was being endorsed. A marker outside the bar cannot be confused
            # with a bar edge at all, and it can point at the hatch on the control
            # row -- which the outline convention could not do without inverting.
            if cat == correct_cat:
                axb.plot([left + w / 2], [yi - g["caret_dy"]], marker="^",
                         markersize=g["caret_ms"] * lw, color=c["fg"], clip_on=False,
                         zorder=6, linestyle="none")
            left += w
        # The cross-panel identity, measured rather than asserted: the "says all four
        # agree" segment should begin exactly where panel A's dot sits.
        if hit_rate is not None:
            residual = max(residual,
                           abs((1.0 - counts.get(MISS, 0) / total) - hit_rate))
    axb.axhline(0.15, color=c["muted"], linewidth=0.6)
    axb.set_xlim(0, 1.02)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    axb.set_xlabel("% blamed, of items with this corruption", fontsize=pt_b)
    axb.set_title("B   And which view does the model blame?", loc="left",
                  fontsize=pt_b, color=c["fg"], pad=7 * lw)
    for side in ("top", "right", "left"):
        axb.spines[side].set_visible(False)
    axa.spines["top"].set_visible(False)
    axa.spines["right"].set_visible(False)
    # sharey leaves B with A's tick MARKS but none of its labels, which reads as a
    # row of unexplained dashes floating to the left of the bars.
    axb.tick_params(axis="y", left=False)

    handles = [Patch(facecolor=MODALITY_COLORS[m], label=label[m])
               for m in MODALITIES if m in seen]
    if NONE in seen:
        handles.append(Patch(facecolor=NONE_COLOR, label=label[NONE]))
    if UNCLEAR in seen and share_max.get(UNCLEAR, 0.0) >= 0.005:
        handles.append(Patch(facecolor=NONE_COLOR, alpha=0.35, label=label[UNCLEAR]))
    if MISS in seen:
        handles.append(Patch(facecolor=c["panel"], hatch="///",
                             edgecolor=c["muted"], label=label[MISS],
                             **style.hatch_kw()))
    if NOVERDICT in seen:
        handles.append(Patch(facecolor=c["faint"], hatch="xxx",
                             edgecolor=c["muted"], label=label[NOVERDICT],
                             **style.hatch_kw()))
    handles.append(Line2D([], [], marker="^", markersize=g["caret_ms"] * lw, color=c["fg"],
                          linestyle="none", label="correct answer for this row"))
    # Under both panels, not beside B. A right-hand legend would have to come out of
    # panel B's width, and B is the panel carrying seven segments across a full
    # 0-100% range; the row pitch and the segment widths are what have to stay
    # legible at column width.
    #
    # Anchored BELOW the figure box (negative y) rather than inside a reserved
    # margin: these figures are saved with bbox_inches="tight", which grows the
    # canvas to include any artist hanging off it, so an explicit bottom margin
    # would be added to -- not consumed by -- the legend, and the fixed reservation
    # would have to be retuned every time the row count changes.
    # One row of entries when the width allows it: wrapping six short labels onto
    # two rows at full width buys nothing and costs a line of height.
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, g["legend_dy"] * type_scale),
               bbox_transform=fig.transFigure, ncol=len(handles) if fine else 3,
               frameon=False, fontsize=pt_a, handlelength=1.1,
               handletextpad=0.5, columnspacing=1.6)

    # Line breaks written out, not left to `wrap=True`: wrap measures against the
    # FIGURE width, so on the wide layout it produced a last line of two words that
    # then collided with the legend.
    # The descriptive note is gone -- it restated the caption, and at full width two
    # centred lines under the axes read as a second figure. What stays is the GUARD:
    # A's flag rate and B's "says all four agree" share are the same denominator seen
    # from two sides, so they must sum to 1 exactly. If they ever do not, the figure
    # has to say so on its face rather than in a docstring.
    if residual > 0.005:
        fig.text(0.5, g["note_dy"] * type_scale,
                 f"The panels part company by up to {100 * residual:.1f} points: B "
                 f"also counts draws that ended without a verdict, which A drops.",
                 ha="center", va="top", fontsize=pt_a, color=c["muted"],
                 linespacing=1.5)
    fig._pair_residual = residual        # for tests and for the report caption
    return fig

def fig_blame_stack(d, annotate=False, hide_empty=False, verbose=False):
    """Four true-outlier rows plus the pooled marginal reference. Segments clickable.

    `annotate` prints each share on its segment and the row denominator at the right.
    It defaults to OFF so that this function keeps producing exactly the figure that
    is in the published consistency_claims.html: that report is a frozen artifact,
    and a shared figure helper that silently changed its appearance would rewrite it
    the next time anyone reran its build script. The expanded report opts in.

    `hide_empty` drops any blame category with zero rows across the whole table --
    in this dataset that is "none", which the parser emits exactly when the model
    said the views AGREE, so it can never co-occur with the flagged subset this
    figure is drawn from. It is 0 of 14,030 flagged corrupted rows. Also OFF by
    default, for the same frozen-artifact reason.
    """
    from .sensitivity import blame_information
    from .constants import MODALITY_LABELS as ML, NONE_COLOR
    from .sensitivity import row_caption
    style.apply(style.theme())
    c = style.colors()
    from .sensitivity import SIGNAL_CONDITIONS, row_label
    from .constants import CONDITION_OUTLIER
    b = blame_information(d, n_perm=1)      # the table is all this figure needs
    # Height from the row count, not a constant. Eight stacked bars in a fixed 3.2in
    # box left the rows further apart than they need to be, and the legend used to
    # sit UNDER the x-label in three columns, which added a band as tall as two data
    # rows to a figure that is already the tallest in the report. The legend now
    # stands beside the axes instead, where it costs width -- which this figure has
    # to spare, its bars being a 0-100% axis -- rather than height.
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.235 * len(SIGNAL_CONDITIONS)
                                                 + 0.95))
    if b.table is None:
        style.empty_axes(ax, "no flagged items")
        return _svg(fig)
    conds = list(SIGNAL_CONDITIONS)
    levels = list(OUTLIER_LEVELS)
    if hide_empty:
        levels = [cat for cat in levels
                  if cat in b.table.columns and int(b.table[cat].sum()) > 0]
    labels = [row_caption(c, verbose) for c in conds]
    ypos = [len(conds) - i for i in range(len(conds))]
    tips = {}
    for row_i, cond in enumerate(conds):
        yi = ypos[row_i]
        m = CONDITION_OUTLIER[cond] if cond else None
        if cond is None:
            shares = b.marginal.reindex(levels).fillna(0.0)
            total = int(b.table.to_numpy().sum())
        else:
            counts = b.table.loc[cond].reindex(levels).fillna(0)
            total = int(counts.sum())
            if total == 0:
                ax.annotate("no detected rows", (0.01, yi), va="center",
                            fontsize=style.ANNOT_PT, color=c["muted"])
                continue
            shares = counts / total
        left = 0.0
        for cat in levels:
            w = float(shares.get(cat, 0.0))
            if w <= 0:
                continue
            col = MODALITY_COLORS.get(cat, NONE_COLOR)
            bar = ax.barh(yi, w, left=left, height=0.62, color=col,
                          edgecolor=c["panel"], linewidth=1.5,
                          alpha=0.55 if cond is None else 1.0)
            # The share, printed on the segment. Without it the only way to read a
            # value is to hover, which is unavailable in print and on a phone.
            # Narrow segments are left unlabelled rather than overplotted -- below
            # about 7% the text is wider than the segment it would sit in.
            if annotate and w >= 0.07:
                ax.text(left + w / 2, yi, f"{100 * w:.0f}%", ha="center",
                        va="center", fontsize=style.ANNOT_PT, color=c["bg"],
                        zorder=5)
            if cond is not None:
                gid = f"tp|{cond}|{cat}"
                bar[0].set_gid(gid)
                n_here = int(b.table.loc[cond, cat]) if cat in b.table.columns else 0
                tips[gid] = (f"{row_label(cond)} corrupted \u2192 blamed "
                             f"{ML.get(cat, cat)}: {n_here} of {total} "
                             f"({100*w:.1f}%)")
                if cat == m:
                    # The diagonal, outlined so it is findable without a legend.
                    ax.barh(yi, w, left=left, height=0.62, facecolor="none",
                            edgecolor=c["fg"], linewidth=1.5, zorder=4)
            left += w
        # Row denominator, so a share can be read back to a count. A row of
        # percentages with no n invites reading a 3-item row as if it were a 600-item
        # one, and the rows here differ by more than that.
        if annotate:
            ax.annotate(f"n = {total:,}", (1.015, yi), va="center", ha="left",
                        fontsize=style.ANNOT_PT, color=c["muted"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=style.TICK_PT)
    ax.set_xlim(0, 1.13 if annotate else 1)
    ax.set_xlabel("share of flagged items, by the view the model blamed")
    ax.set_ylim(min(ypos) - 0.55, max(ypos) + 0.55)
    # A legend, because five fixed segments in every bar cannot be named any other
    # way; the outlined segment marks the correct answer for its row.
    handles = [Patch(facecolor=MODALITY_COLORS.get(cat, NONE_COLOR),
                     edgecolor=c["panel"], linewidth=1.2,
                     label=ML.get(cat, cat)) for cat in levels]
    handles.append(Patch(facecolor="none", edgecolor=c["fg"], linewidth=1.5,
                         label="correct answer for this row"))
    # Anchored past the right edge of the axes -- clear of the "n = " column, which
    # sits at x=1.015 in DATA coords and so is still inside the 1.13 x-limit.
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=1, fontsize=style.TICK_PT, frameon=False,
              handlelength=1.1, handletextpad=0.5, labelspacing=0.5,
              borderaxespad=0.0)
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"]); ax.set_axisbelow(True)
    return _inject_titles(_svg(fig), tips)


def fig_permutation_null(d):
    """Small evidence panel: MI under the null, with the observed value marked."""
    from .sensitivity import blame_information
    style.apply(style.theme())
    c = style.colors()
    b = blame_information(d)
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 1.15))
    if b.null is None:
        style.empty_axes(ax, "no flagged items")
        return _svg(fig)
    ax.hist(b.null, bins=60, color=c["bar2"], edgecolor="none")
    ax.axvline(b.mi, color=MODALITY_COLORS["T"], linewidth=1.6)
    ax.annotate(f"observed {b.mi:.3f} bits\np = {b.p_value:.4f}",
                (b.mi, ax.get_ylim()[1]), xytext=(-6, -4),
                textcoords="offset points", ha="right", va="top",
                fontsize=style.ANNOT_PT, color=c["fg"])
    ax.set_xlabel("mutual information under random reassignment of the true view (bits)")
    ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    return _svg(fig)


def fig_blame_levels(d):
    """How often each view is blamed when it is NOT the corrupted one, ranked.

    This replaced a six-row pairwise-contrast plot. The contrasts were correct but
    answered "is code blamed more than math", which is a question about the
    statistics rather than about the model; the ranked levels answer the question
    the section actually asks. The contrasts still exist, in the details block.
    """
    style.apply(style.theme())
    c = style.colors()
    dd = M.prepare(d)
    rows = []
    for m in MODALITIES:
        den = dd[dd["true_outlier"].ne(m)]
        k, n = int(den["pred_outlier"].eq(m).sum()), len(den)
        if not n:
            continue
        lo, hi = M.wilson_ci(k, n)
        rows.append((m, k / n, lo, hi, n))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.3))
    if not rows:
        style.empty_axes(ax, "no rows")
        return _svg(fig)
    y = np.arange(len(rows))
    for i, (m, v, lo, hi, n) in enumerate(rows):
        ax.barh(i, 100 * v, height=0.6, color=MODALITY_COLORS[m])
        ax.plot([100 * lo, 100 * hi], [i, i], color=c["fg"], linewidth=1.1)
        ax.annotate(f"{100 * v:.1f}%   n={n:,}", (100 * hi, i), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=style.ANNOT_PT, color=c["fg"])
    ax.set_yticks(y)
    ax.set_yticklabels([MODALITY_LABELS[r[0]] for r in rows])
    ax.set_xlabel("how often this view is blamed when it is NOT the corrupted one (%)")
    ax.set_xlim(0, max(100 * r[3] for r in rows) * 1.35)
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    return _svg(fig)


def fig_obfuscation_overall(d):
    """What stripping identifiers does to the experiment as a whole.

    The per-view contrast answers "which view moved"; this answers "did anything
    move at all", which is the question the obfuscation factor exists for and which
    no figure was reporting.
    """
    style.apply(style.theme())
    c = style.colors()
    dd = M.prepare(d)
    measures = [
        ("notices something is wrong",
         lambda s: (int(s[s["is_corrupted"]]["detected"].sum()),
                    len(s[s["is_corrupted"]]))),
        ("false alarm on a clean set",
         lambda s: (int(s[~s["is_corrupted"]]["detected"].sum()),
                    len(s[~s["is_corrupted"]]))),
        ("names the right view, once it notices",
         lambda s: (int(s[s["localization_eligible"]]["localization_correct"].sum()),
                    len(s[s["localization_eligible"]]))),
    ]
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.6))
    yy = np.arange(len(measures))[::-1]
    h = 0.32
    for offset, naming, col in ((h / 2, NAMING_LEVELS[0], c["bar"]),
                                (-h / 2, NAMING_LEVELS[1], c["bar2"])):
        sub = dd[dd["naming"].eq(naming)]
        for yi, (_, fn) in zip(yy, measures):
            k, n = fn(sub)
            if not n:
                continue
            v = 100 * k / n
            lo, hi = M.wilson_ci(k, n)
            ax.barh(yi + offset, v, height=h, color=col,
                    label=naming if yi == yy[0] else None)
            ax.plot([100 * lo, 100 * hi], [yi + offset] * 2, color=c["fg"],
                    linewidth=1.0)
    for yi, (label, fn) in zip(yy, measures):
        kr, nr = fn(dd[dd["naming"].eq(NAMING_LEVELS[0])])
        ko, no = fn(dd[dd["naming"].eq(NAMING_LEVELS[1])])
        diff, lo, hi = C.newcombe_diff(kr, nr, ko, no)
        sig = not C.crosses_null(lo, hi)
        ax.annotate(f"{C.pp(diff)}" + ("" if sig else "  (n.s.)"),
                    (max(100 * kr / nr if nr else 0, 100 * ko / no if no else 0), yi),
                    xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=style.ANNOT_PT,
                    color=c["fg"] if sig else c["muted"])
    ax.set_yticks(yy)
    ax.set_yticklabels([m[0] for m in measures])
    ax.set_xlabel("% of items  —  real names vs obfuscated names")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=style.TICK_PT)
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    return _svg(fig)


# Reader-facing names for the two axes of the blame matrix. Kept next to the figure
# rather than in constants.py because they are presentation, not design: the codes in
# CONDITIONS/OUTLIER_LEVELS stay canonical and every gid, drill-down key and stored
# column continues to use them.
_BLAMED_LABELS = {
    # "none" is the model answering that the four representations agree, i.e. it named
    # no outlier at all. Left as the bare word it reads like an empty column.
    "none": "none — said they agree",
}
def _condition_labels():
    """Built from the same table the y-axes use, so the two figures cannot drift.

    Computed on call rather than at import: row_caption lives in .sensitivity, which
    this module imports lazily inside functions to keep the import graph acyclic.
    """
    from .sensitivity import row_caption
    return {"A0": "nothing corrupted",
            **{c: row_caption(c, verbose=True) for c in CONDITIONS if c != "A0"}}


def _unconditional_caption_stats(d_all):
    """Row n, no-verdict count, and how many of those carry a scavenged verdict.

    Returned as display strings so the caption never has to know whether the rows
    came out equal: while an arm is still filling they can differ, and asserting
    "every row has the same n" would then be false.
    """
    if d_all is None or not len(d_all):
        return "n varies by row", 0, "some", "unevenly", "unevenly"
    corrupted = d_all[d_all["condition"].astype(str).ne("A0")]
    per_row = corrupted.groupby("condition").size()
    if len(set(per_row)) == 1:
        row_n = f"{int(per_row.iloc[0]):,} draws"
    else:
        row_n = (f"{int(per_row.min()):,}\u2013{int(per_row.max()):,} draws, "
                 "not yet equal while an arm is still filling")
    nv = d_all[d_all["no_verdict"].astype(bool)] if "no_verdict" in d_all else d_all.iloc[:0]
    scav = "some"
    if len(nv) and "pred_agree" in nv:
        n = int(nv["pred_agree"].astype(str).isin(("yes", "no")).sum())
        scav = f"{n:,}"
    # Which conditions lose the most and the least to no-verdict draws. These were
    # written into the caption by hand ("8.1% for corrupted code against 4.2% for
    # trajectory-rand") and went stale on the first repair pass; the rung was also
    # named there by its generator code, which no figure says any more.
    hi_s = lo_s = "unevenly across rows"
    if "no_verdict" in corrupted and len(corrupted):
        from .sensitivity import row_caption_corrupted
        share = corrupted.groupby("condition")["no_verdict"].mean().sort_values()
        if len(share) > 1:
            def _fmt(cond):
                return (f"{100 * float(share[cond]):.1f}% for "
                        f"{row_caption_corrupted(cond)}")
            hi_s, lo_s = _fmt(share.index[-1]), _fmt(share.index[0])
    return row_n, len(nv), scav, hi_s, lo_s


def fig_blame_matrix(d):
    """Condition x blamed view, row-normalized, every cell individually clickable."""
    style.apply(style.theme())
    c = style.colors()
    bm = M.blame_matrix(d, row_by="condition")
    grid = bm.pivot(index="condition", columns="pred_outlier", values="row_frac") \
             .reindex(index=CONDITIONS, columns=OUTLIER_LEVELS)
    cnt = bm.pivot(index="condition", columns="pred_outlier", values="n") \
            .reindex(index=CONDITIONS, columns=OUTLIER_LEVELS).fillna(0)
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.42 * len(CONDITIONS) + 1.3))
    cmap = plt.get_cmap(style.colors()["cmap"])
    for i, cond in enumerate(CONDITIONS):
        for j, pred in enumerate(OUTLIER_LEVELS):
            v = grid.to_numpy(dtype=float)[i, j]
            n = int(cnt.to_numpy()[i, j])
            face = c["panel"] if not np.isfinite(v) else cmap(v)
            # Rectangles rather than imshow: imshow is one artist, so no cell could
            # carry its own id and the whole point of the drill-down would be lost.
            r = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                              edgecolor=c["panel"], linewidth=1.2)
            r.set_gid(f"cell|{cond}|{pred}")
            ax.add_patch(r)
            if n < C.MIN_N:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor=c["muted"],
                                           linewidth=0.0, alpha=0.55,
                                           **style.hatch_kw()))
            ax.text(j, i, f"{n}", ha="center", va="center",
                    fontsize=style.ANNOT_PT,
                    color="#0b0d14" if (np.isfinite(v) and v > 0.62) else c["fg"])
    ax.set_xlim(-0.5, len(OUTLIER_LEVELS) - 0.5)
    ax.set_ylim(len(CONDITIONS) - 0.5, -0.5)
    # Both axes carried internal codes: the columns read "none" (which reads as "no
    # data", not "the model said they agree"), and the rows read A0 / A-C / T:rand /
    # A-M, which are the spec's condition ids and mean nothing to a reader. The cell
    # gids keep the codes -- the drill-down is keyed on them -- but nothing on screen
    # needs to.
    ax.set_xticks(range(len(OUTLIER_LEVELS)))
    ax.set_xticklabels([_BLAMED_LABELS.get(v, MODALITY_LABELS.get(v, v))
                        for v in OUTLIER_LEVELS], rotation=30, ha="right")
    ax.set_yticks(range(len(CONDITIONS)))
    _cond_lab = _condition_labels()
    ax.set_yticklabels([_cond_lab.get(x, x) for x in CONDITIONS])
    ax.set_xlabel("which view the model named as the odd one out")
    ax.set_ylabel("which view was actually corrupted")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return _svg(fig)


# ── tables ───────────────────────────────────────────────────────────────────
def latex_table(rows, caption, label):
    """booktabs source for the numbers under a figure, so the paper cannot drift."""
    body = "\n".join(f"{_tex(a)} & {_tex(b)} \\\\" for a, b in rows)
    return ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lr}\n\\toprule\n"
            "quantity & value \\\\\n\\midrule\n" + body +
            f"\n\\bottomrule\n\\end{{tabular}}\n\\caption{{{_tex(caption)}}}\n"
            f"\\label{{{label}}}\n\\end{{table}}")


def _tex(x):
    return (str(x).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&").replace("−", "$-$"))


def html_table(rows):
    body = "".join(f"<tr><td>{_esc(a)}</td><td class='num'>{_esc(b)}</td></tr>"
                   for a, b in rows)
    return f"<table class='raw'><tr><th>quantity</th><th>value</th></tr>{body}</table>"


def crossing_table(d, kind="detection"):
    """The full crossing behind a headline figure: condition x naming x reasoning."""
    dd = M.prepare(d)
    out = []
    for cond in CONDITIONS:
        for naming in NAMING_LEVELS:
            for reasoning in sorted(dd["reasoning"].dropna().unique()):
                sub = dd[dd["condition"].eq(cond) & dd["naming"].eq(naming)
                         & dd["reasoning"].eq(reasoning)]
                if kind == "localization":
                    sub = sub[sub["localization_eligible"]]
                    k = int(sub["localization_correct"].sum())
                else:
                    k = int(sub["detected"].sum())
                n = len(sub)
                if not n:
                    continue
                lo, hi = M.wilson_ci(k, n)
                out.append((f"{cond} · {naming} · reasoning={reasoning}",
                            f"{C.pct(k / n)} [{C.pct(lo)}, {C.pct(hi)}]  n={n}"
                            + ("  ⚠ thin" if n < C.MIN_N else "")))
    return out


# ── assembly ─────────────────────────────────────────────────────────────────
def build(d, out="viz/consistency_claims.html", defects=None, theme="dark",
          annotate=False, blame_unconditional=False, d_all=None,
          verbose_labels=False):
    style.apply(theme)
    base = C.baselines(d)
    obf_svg, obf_stats, obf_cap = _obfuscation_block(d)
    obf_design = _obfuscation_design_note(d)
    obf_split = _obfuscation_split_block(d) if obf_svg else ""
    verdicts = C.all_verdicts(d, shared={"obfuscation": obf_stats}
                              if obf_stats else None)
    drill = build_drilldown(d, defects)

    from .sensitivity import severity_tiers
    _sev_table, _sev_matched, _sev_common = severity_tiers(d)
    matched_svg, matched_note = fig_sensitivity_matched(d)
    sev_banner = ("" if _sev_matched else
                  '<div class="warnbanner">Corruptions are not severity-matched '
                  'across representations. Row ordering may reflect how each '
                  'corruption was generated rather than what the model trusts.</div>')
    figs = {
        "q1": (sev_banner + fig_sensitivity(d, verbose=verbose_labels)
               + (matched_svg or ""),
               "Left: how often each corruption is flagged, against the SHARED rate "
               "at which consistent items are flagged (the dashed line). The visual "
               "reading is how far right of that line each row sits. Right: the same "
               "thing as a sensitivity index. Rows are ordered by sensitivity; a "
               "hatched row has fewer than 20 items. Rows are grouped, not globally "
               "sorted: trajectory's four corruption methods can be compared with "
               "each other, and the three single-method views with each other, but "
               "the two groups are not severity-matched. Only the corruption "
               "varies; models, naming and reasoning are pooled. " + matched_note),
        "q2": (fig_blame_stack(d, annotate=annotate,
                               hide_empty=blame_unconditional,
                               verbose=verbose_labels),
               "One row per corruption. Segments are always in the order code, "
               "trajectory, description, math, none, and the outlined segment is the "
               "correct answer for that row &mdash; a row that is mostly its own "
               "outline is a row the model got right. Every segment is clickable. "
               "Trajectory is split into its four generation methods, because they "
               "differ more from each other than whole modalities do. The pooled "
               "blame distribution, and how far each row departs from it, is in the "
               "details block."),
        "q3": (fig_blame_levels(d),
               "How often the model blames each view on the items where that view is "
               "NOT the corrupted one. Only the view varies; naming and reasoning are "
               "pooled. Bars are 95% intervals. The six pairwise differences are in "
               "the details block."),
        "q4": ((obf_design + obf_svg + obf_split) if obf_svg else obf_svg, obf_cap),
        "q5": (None, ""),
    }


    sections = []
    for qid, title, v in verdicts:
        svg, caption = figs.get(qid, (None, ""))
        if qid == "q3":
            svg = svg + fig_blame_matrix(d)
            caption += ("<br><br><b>Below: the full blame matrix.</b> Read a row as "
                        "&ldquo;when THIS view was corrupted, where did the blame "
                        "go?&rdquo; The number in each cell is a count of draws; the "
                        "shading is that count as a share of its row, so the diagonal "
                        "lighting up is the model being right. The first column is the "
                        "model answering that all four representations agree &mdash; a "
                        "bright cell there is a miss, not a correct answer. The top row "
                        "is the control where nothing was corrupted, so for that row "
                        "the first column is the correct answer and everything else is "
                        "a false alarm. A cell crossed by <b>diagonal lines</b> holds "
                        f"fewer than {C.MIN_N} draws &mdash; too few to read a rate "
                        "from, so the shading there is marking how rare the cell is "
                        "rather than measuring anything. The pale gaps between cells "
                        "are only separators. Every cell is clickable and opens the "
                        "runs behind it.")
        # Opt-in only, for the same reason `annotate` is: consistency_claims.html is
        # a frozen artifact and its build script must keep producing it byte for byte.
        if qid == "q2" and blame_unconditional and svg is not None:
            svg = svg + fig_blame_stack_unconditional(d, d_all=d_all,
                                                       verbose=verbose_labels)
            # These were hardcoded ("2,784 draws", "709 of Nemotron's 907") and went
            # stale the moment a repair pass landed -- by 2026-08-23 Nemotron's count
            # had moved 907 -> 734 while the caption still said 907. A caption that
            # states numbers the figure above it no longer holds is worse than one
            # that states none, so both now come from the frame being plotted.
            _rn, _nv, _sc, _hi_s, _lo_s = _unconditional_caption_stats(d_all)
            caption += (
                "<br><br><b>Below: the same rows over every corrupted item.</b> The "
                "figure above divides by the items the model flagged, so it measures "
                "localization <i>given</i> that it noticed something. This one "
                "divides by all items where that representation was corrupted and "
                "adds the category the first cannot show &mdash; <i>does not "
                "identify disagreement</i> (hatched), where the model answered that "
                "the four representations agree. A row that is mostly hatched never "
                "reached the question the figure above is asking, however well it "
                "scores there. Same colours and same outlined-correct-answer "
                "convention; not clickable, because these denominators are not the "
                "ones the drill-down indexes."
                f"<br><br><b>Every row has the same n</b> ({_rn}), because "
                "this is the one figure that keeps the draws which ended without a "
                "verdict &mdash; budget truncation or a decode loop &mdash; instead "
                "of dropping them. Every other figure in this report excludes them, "
                "which is right there: scoring a run that produced no verdict as if "
                "it had produced a wrong one invents an answer. But those draws are "
                "consumed opportunities, and excluding them here would both unbalance "
                f"the rows (the loss runs {_hi_s} against {_lo_s}) "
                "and quietly remove the model&rsquo;s longest "
                "deliberations from the denominator. They are counted separately "
                f"rather than as misses on purpose: {_sc} of the {_nv:,} no-verdict "
                "draws carry an &ldquo;agree&rdquo; the parser scavenged out of "
                "reasoning the model never finished, so folding them into &ldquo;does "
                "not identify disagreement&rdquo; would manufacture that finding.")
            # The paper figure: the two questions this report answers in separate
            # sections, on one row axis. It comes AFTER both of its halves rather
            # than replacing them, because each half still carries its own verdict,
            # details table and -- for the conditional figure -- the drill-down.
            _pair = fig_detection_blame_pair(d, d_all=d_all, verbose=verbose_labels)
            _resid = getattr(_pair, "_pair_residual", 0.0)
            # A ruled heading, not just another SVG appended to the same <figure>.
            # Three stacked figures inside one bordered box ran together -- the
            # paired figure's panel titles sat directly under the previous figure's
            # legend with nothing between them, so it read as a fourth row of that
            # figure rather than as a separate one.
            svg = (svg + '<h4 class="subfig">Both questions on one row axis</h4>'
                   + _svg(_pair))
            caption += (
                "<br><br><b>Below: both questions on one row axis.</b> This is the "
                "figure for the paper. Panel A is the detection question from the "
                "section above &mdash; how often each corruption is flagged, against "
                "the rate at which the model flags items where nothing is wrong "
                "(dashed line, shaded region). Panel B is the blame figure "
                "immediately above it, over the same denominator. They are drawn on "
                "a shared row axis because they are one measurement split two ways: "
                "read across a row and panel A&rsquo;s dot sits at the right-hand "
                "edge of panel B&rsquo;s hatch"
                + (" &mdash; here to within "
                   f"{100 * _resid:.1f} percentage points, the draws that ended "
                   "without a verdict, which B counts and A drops."
                   if _resid > 0.005 else
                   " &mdash; here exactly, because no draw in this roster ended "
                   "without a verdict.")
                + " Rows are ordered by detectability within two blocks: "
                "trajectory&rsquo;s four generation methods are comparable with each "
                "other and the three single-method views with each other, but the "
                "two blocks are not severity-matched, so there is no global ranking "
                "to draw."
                "<br><br><b>The control row is drawn here, which the figure above "
                "omits.</b> On the items where nothing was corrupted the model still "
                "says the four views disagree about half the time and then names "
                "one, and that distribution is its blame prior with no signal to "
                "respond to \u2014 whatever it points at there, it points at without "
                "evidence, and every row above should be read against it. That is "
                "also why correctness is marked with a caret <b>under</b> the correct "
                "segment rather than by outlining it: on the control row the correct "
                "answer is <i>says all four agree</i>, so the marker has to be able "
                "to move to a different category, and an outline sitting beside the "
                "hatched segment&rsquo;s own border was two dark rectangles of "
                "similar weight where only one meant anything. Not clickable, for "
                "the same reason the figure above it is not.")
        if svg is None:
            figblock = ("<div class='pending'><b>No figure.</b> "
                        f"{_esc(v.detail or 'This question has no measured data.')}"
                        "</div>")
        else:
            figblock = f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"

        det_rows = v.rows or []
        cross = crossing_table(d, "detection") if qid == "q1" else []
        tex = latex_table(det_rows or cross, title, f"tab:{qid}")
        sections.append(f"""
<section id="{qid}">
  <h2>{_esc(title)}</h2>
  <p class="verdict v-{v.direction}">{_esc(v.sentence)}</p>
  {figblock}
  <details>
    <summary>Details — full crossing, raw numbers, n per cell</summary>
    {html_table(det_rows) if det_rows else ""}
    {("<h4>Full crossing</h4>" + html_table(cross)) if cross else ""}
    <button class="copytex" data-tex="{_esc(tex)}">copy as LaTeX table</button>
  </details>
</section>""")

    doc = TEMPLATE.format(
        cssvars=CSS_THEMES.get(theme, CSS_THEMES["dark"]),
        sections="\n".join(sections),
        drill=json.dumps(drill), n=len(d), hf=HF_URL,
        models=_esc(", ".join(sorted(d["model"].astype(str).unique()))))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(doc)
    print(f"[claims] wrote {out} ({os.path.getsize(out)/1e6:.2f} MB, "
          f"{len(verdicts)} questions, {len(drill)} drill-down cells)")
    return out


# One palette per ground. The page and the figures must agree: a white raster
# dropped into a dark report reads as a foreign object, and the reverse is worse.
# Dark is the default so build_cross_modal_claims_frozen.sh keeps reproducing the frozen
# consistency_claims.html byte for byte; the expanded report opts into light.
CSS_THEMES = {
    "dark":  "--accent:#7eb8ff; --bg:#0d0f18; --blue2:#cfe0ff; --deep:#0a0c14; --dim:#5a6274; --dim2:#6b7a99; --drawer:#0f1119; --fg:#e0e0e0; --green:#8fd694; --hi:#2a3450; --line:#1e2130; --link2:#8fa6c9; --muted:#8592ae; --ok:#4fa96a; --orange:#f2a97e; --panel:#12141e; --panel2:#141826; --panel3:#1b2032; --raised:#171d30; --sunk:#12182a; --tagbg:#1d2540; --tagline:#26304a; --text2:#cfd8e8; --text3:#b8c2d6; --text4:#c8cddb; --warn:#c9a227; --warn2:#e0c88f",
    "light": "--accent:#1a5fb4; --bg:#ffffff; --blue2:#1a4f9c; --deep:#f7f8fb; --dim:#7b8493; --dim2:#6b7280; --drawer:#ffffff; --fg:#16181d; --green:#1f7a45; --hi:#d9e2f3; --line:#dfe3ea; --link2:#1f5aa6; --muted:#5b6472; --ok:#1f7a45; --orange:#a85520; --panel:#ffffff; --panel2:#f4f6fa; --panel3:#eef1f7; --raised:#eef1f7; --sunk:#f4f6fa; --tagbg:#e6edfb; --tagline:#c9d6ef; --text2:#2a313c; --text3:#39404d; --text4:#333a45; --warn:#8a6d0f; --warn2:#6b5410",
}


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-representation consistency — results</title>
<style>
  :root {{ {cssvars}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); line-height:1.6;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          font-size:15px; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:42px 26px 120px; }}
  h1 {{ font-size:1.5rem; font-weight:400; margin:0 0 6px; }}
  .sub {{ color:var(--muted); font-size:0.82rem; margin-bottom:30px; }}
  h2 {{ font-size:1.12rem; font-weight:500; margin:0 0 12px; }}
  section {{ border-top:1px solid var(--line); padding-top:26px; margin-top:34px; }}
  .verdict {{ font-size:1.0rem; padding:13px 16px; border-radius:7px;
              border-left:3px solid var(--dim); background:var(--panel2); margin:0 0 18px; }}
  .v-supported {{ border-left-color:var(--ok); }}
  .v-inconclusive {{ border-left-color:var(--warn); }}
  .v-unmeasured {{ border-left-color:var(--dim2); color:var(--muted); }}
  figure {{ margin:0 0 14px; background:var(--panel); border:1px solid var(--line);
            border-radius:8px; padding:14px; overflow-x:auto; }}
  figure svg {{ max-width:100%; height:auto; display:block; margin:0 auto; }}
  .designnote {{ font-size:0.82rem; color:var(--text3); background:var(--panel);
                 border:1px solid var(--line); border-radius:7px; padding:13px 16px;
                 margin:0 0 14px; line-height:1.65; }}
  .designnote b {{ color:var(--fg); }}
  .promoted {{ font-size:0.95rem; color:var(--text2); background:var(--panel2);
               border-left:3px solid var(--accent); border-radius:6px;
               padding:12px 15px; margin:0 0 14px; }}
  .figsays {{ color:var(--muted); font-size:13px; margin:8px 2px 0; }}
  /* The companion figure inside a section: its own heading and its own caption, so
     the exploratory rows below the rule are never read as part of the section's
     result. Styled lighter than a section heading -- it is a sub-figure, not a
     sixth question. */
  h4.subfig {{ color:var(--text2); font-size:0.86rem; font-weight:600;
               margin:26px 0 8px; padding-top:16px;
               border-top:1px solid var(--line); }}
  .figcap {{ color:var(--muted); font-size:0.78rem; margin:9px 2px 0; }}
  figcaption {{ color:var(--muted); font-size:0.78rem; margin-top:10px;
                border-top:1px solid var(--line); padding-top:9px; }}
  details {{ background:var(--panel); border:1px solid var(--line); border-radius:7px;
             padding:0 14px; }}
  summary {{ cursor:pointer; color:var(--muted); font-size:0.8rem; padding:11px 0; }}
  table.raw {{ width:100%; border-collapse:collapse; font-size:0.79rem;
               margin-bottom:12px; }}
  table.raw td, table.raw th {{ border-bottom:1px solid var(--line); padding:5px 7px;
                                text-align:left; }}
  table.raw th {{ color:var(--dim); font-weight:500; }}
  table.raw td.num {{ font-family:ui-monospace,Menlo,monospace; text-align:right;
                      color:var(--text2); }}
  h4 {{ color:var(--dim); font-size:0.74rem; text-transform:uppercase;
        letter-spacing:0.08em; margin:16px 0 6px; }}
  button {{ background:var(--panel3); color:var(--fg); border:1px solid var(--line);
            border-radius:5px; padding:6px 11px; font-size:0.75rem; cursor:pointer;
            margin:4px 0 14px; font-family:inherit; }}
  button:hover {{ border-color:var(--accent); color:var(--accent); }}
  .pending {{ border:1px dashed var(--hi); color:var(--muted); padding:20px;
              border-radius:7px; font-size:0.85rem; }}
  [id^="cell|"], [id^="bar|"] {{ cursor:pointer; }}
  [id^="cell|"]:hover, [id^="bar|"]:hover {{ opacity:0.72; }}
  #drawer {{ position:fixed; top:0; right:0; width:min(620px,94vw); height:100%;
             background:var(--drawer); border-left:1px solid var(--line); overflow-y:auto;
             transform:translateX(100%); transition:transform .18s ease; z-index:50;
             padding:20px 22px 60px; }}
  #drawer.open {{ transform:none; }}
  #drawer h3 {{ margin:0 0 4px; font-size:0.98rem; font-weight:500; }}
  #drawer .meta {{ color:var(--dim); font-size:0.76rem; margin-bottom:16px; }}
  .run {{ border:1px solid var(--line); border-radius:6px; padding:12px;
          margin-bottom:12px; background:var(--panel); }}
  .run .hdr {{ font-family:ui-monospace,monospace; font-size:0.73rem;
               color:var(--muted); margin-bottom:7px; }}
  .run .lbl {{ color:var(--dim); font-size:0.68rem; text-transform:uppercase;
               letter-spacing:0.07em; margin-top:8px; }}
  .run .just {{ white-space:pre-wrap; font-size:0.79rem; color:var(--text2); }}
  .run .defect {{ font-size:0.79rem; color:var(--warn2); }}
  .close {{ position:absolute; top:14px; right:18px; }}
  .toast {{ position:fixed; bottom:22px; left:50%; transform:translateX(-50%);
            background:var(--panel3); border:1px solid var(--accent); color:var(--accent);
            padding:9px 16px; border-radius:6px; font-size:0.8rem; opacity:0;
            transition:opacity .2s; z-index:99; }}
  .toast.show {{ opacity:1; }}
</style></head><body>
<div class="wrap">
  <h1>Cross-representation consistency</h1>
  <div class="sub">{n:,} runs · models: {models} · every verdict below is computed
    from the data, not written. Intervals are 95% Wilson; differences use Newcombe's
    hybrid score interval. Cells with n&nbsp;&lt;&nbsp;20 are hatched and excluded
    from verdict lines. <a href="{hf}" style="color:var(--accent)">source dataset</a>
  </div>

  {sections}
</div>

<div id="drawer">
  <button class="close" onclick="closeDrawer()">close</button>
  <h3 id="dtitle">runs</h3>
  <div class="meta" id="dmeta"></div>
  <div id="druns"></div>
</div>
<div class="toast" id="toast"></div>

<script>
const DRILL = {drill};
const VIEW = {{none:'none', C:'code', T:'trajectory', D:'description', M:'math'}};

function toast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1400);
}}
function copy(text, msg) {{
  navigator.clipboard.writeText(text).then(() => toast(msg),
    () => toast('copy failed — select manually'));
}}
function closeDrawer() {{ document.getElementById('drawer').classList.remove('open'); }}

function openCell(key) {{
  const entry = DRILL[key];
  const drawer = document.getElementById('drawer');
  const parts = key.split('|');
  const title = parts[0] === 'cell'
      ? `${{parts[1]}} → blamed ${{VIEW[parts[2]] || parts[2]}}`
      : `${{parts[1]}} — all runs`;
  document.getElementById('dtitle').textContent = title;
  if (!entry) {{
    document.getElementById('dmeta').textContent = 'No runs in this cell.';
    document.getElementById('druns').innerHTML = '';
    drawer.classList.add('open');
    return;
  }}
  document.getElementById('dmeta').textContent =
      `${{entry.n}} run(s) in this cell — showing ${{entry.runs.length}}, ` +
      `sampled with a fixed seed so this list is reproducible.`;
  document.getElementById('druns').innerHTML = entry.runs.map(r => `
    <div class="run">
      <div class="hdr">${{r.run_id}}</div>
      <div>${{r.solver_id}} · ${{r.condition}} · ${{r.naming}} · reasoning=${{r.reasoning}}</div>
      <div class="lbl">said</div>
      <div>agree=${{r.pred_agree || '(none)'}} · blamed ${{VIEW[r.pred_outlier] || '(none)'}}
           · actually corrupted: ${{VIEW[r.true_outlier] || r.true_outlier}}</div>
      <div class="lbl">justification (verbatim)</div>
      <div class="just">${{(r.justification || '(empty)').replace(/</g,'&lt;')}}</div>
      ${{r.defect ? `<div class="lbl">ground-truth defect</div>
                     <div class="defect">${{r.defect.replace(/</g,'&lt;')}}</div>` : ''}}
      <button onclick="copy('${{r.run_id}}', 'run_id copied')">copy citation</button>
    </div>`).join('');
  drawer.classList.add('open');
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('[id^="cell|"], [id^="bar|"]').forEach(el => {{
    el.addEventListener('click', () => openCell(el.id));
  }});
  document.querySelectorAll('.copytex').forEach(b => {{
    b.addEventListener('click', () => copy(b.dataset.tex, 'LaTeX table copied'));
  }});
  document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDrawer(); }});
}});
</script></body></html>"""
