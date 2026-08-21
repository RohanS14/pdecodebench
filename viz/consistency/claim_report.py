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

from . import claims as C
from . import metrics as M
from . import style
from .constants import (CONDITIONS, MODALITIES, MODALITY_LABELS, NAMING_LEVELS,
                        MODALITY_COLORS, NONE, NONE_COLOR, OUTLIER_LEVELS)

SEED = 20260820
SAMPLES_PER_CELL = 5
HF_URL = "https://huggingface.co/datasets/bermaneh/pde-llm-eval-xmodal-consistency"


def _svg(fig):
    """Inline SVG, with the XML prologue stripped so it embeds in the document."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor())
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
                   edgecolor=c["muted"] if thin else "none", linewidth=0.6)
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


def fig_sensitivity(d, tier=None):
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
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.42 * (len(rows) + 4) + 1.0))
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
    groups = [g for g in (("trajectory, by how it was corrupted", traj),
                          ("one corruption method each", other)) if g[1]]
    ordered, y_corrupt, headers = [], [], []
    y = 0.0
    for gi, (gname, members) in enumerate(reversed(groups)):
        for x in reversed(members):          # highest rate at the TOP of its block
            y += 1.0
            ordered.append(x)
            y_corrupt.append(y)
        headers.append((gname, y + 0.42))
        y += 0.75
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
    for gname, gy in headers:
        ax.annotate(gname, (0.012, gy), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=style.ANNOT_PT,
                    color=c["muted"], style="italic")

    ax.set_yticks(y_corrupt + [y_base])
    labels = [x["label"] + " corrupted" for x in rows] + ["nothing corrupted"]
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_color(c["muted"])
    ax.set_ylim(y_base - 0.7, max(y_corrupt) + 1.1)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(np.arange(xlo, xhi + 1e-9, 0.05))
    ax.set_xticklabels([f"{100 * t:.0f}%" for t in np.arange(xlo, xhi + 1e-9, 0.05)])
    ax.set_xlabel("items the model flagged as disagreeing")
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    ax.annotate(f"Shaded region: flagged even when all four representations agree "
                f"({100 * r.fa_rate:.1f}%).",
                (0, -0.30), xycoords="axes fraction", fontsize=style.ANNOT_PT,
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


def fig_blame_stack(d):
    """Four true-outlier rows plus the pooled marginal reference. Segments clickable."""
    from .sensitivity import blame_information
    from .constants import MODALITY_LABELS as ML, NONE_COLOR
    style.apply(style.theme())
    c = style.colors()
    from .sensitivity import SIGNAL_CONDITIONS, row_label
    from .constants import CONDITION_OUTLIER
    b = blame_information(d, n_perm=1)      # the table is all this figure needs
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 3.2))
    if b.table is None:
        style.empty_axes(ax, "no flagged items")
        return _svg(fig)
    conds = list(SIGNAL_CONDITIONS)
    labels = [f"{row_label(c)} was corrupted" for c in conds]
    ypos = [len(conds) - i for i in range(len(conds))]
    tips = {}
    for row_i, cond in enumerate(conds):
        yi = ypos[row_i]
        m = CONDITION_OUTLIER[cond] if cond else None
        if cond is None:
            shares = b.marginal.reindex(OUTLIER_LEVELS).fillna(0.0)
            total = int(b.table.to_numpy().sum())
        else:
            counts = b.table.loc[cond].reindex(OUTLIER_LEVELS).fillna(0)
            total = int(counts.sum())
            if total == 0:
                ax.annotate("no detected rows", (0.01, yi), va="center",
                            fontsize=style.ANNOT_PT, color=c["muted"])
                continue
            shares = counts / total
        left = 0.0
        for cat in OUTLIER_LEVELS:
            w = float(shares.get(cat, 0.0))
            if w <= 0:
                continue
            col = MODALITY_COLORS.get(cat, NONE_COLOR)
            bar = ax.barh(yi, w, left=left, height=0.62, color=col,
                          edgecolor=c["panel"], linewidth=1.5,
                          alpha=0.55 if cond is None else 1.0)
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
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=style.TICK_PT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of flagged items, by the view the model blamed")
    ax.set_ylim(min(ypos) - 0.7, max(ypos) + 0.7)
    # A legend, because five fixed segments in every bar cannot be named any other
    # way; the outlined segment marks the correct answer for its row.
    handles = [Patch(facecolor=MODALITY_COLORS.get(cat, NONE_COLOR),
                     edgecolor=c["panel"], linewidth=1.2,
                     label=ML.get(cat, cat)) for cat in OUTLIER_LEVELS]
    handles.append(Patch(facecolor="none", edgecolor=c["fg"], linewidth=1.5,
                         label="correct answer for this row"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=style.TICK_PT)
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
    ax.set_xlabel("how often this view is blamed when it is NOT the broken one (%)")
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
                                           linewidth=0.0, alpha=0.55))
            ax.text(j, i, f"{n}", ha="center", va="center",
                    fontsize=style.ANNOT_PT,
                    color="#0b0d14" if (np.isfinite(v) and v > 0.62) else c["fg"])
    ax.set_xlim(-0.5, len(OUTLIER_LEVELS) - 0.5)
    ax.set_ylim(len(CONDITIONS) - 0.5, -0.5)
    ax.set_xticks(range(len(OUTLIER_LEVELS)))
    ax.set_xticklabels([MODALITY_LABELS.get(v, v) for v in OUTLIER_LEVELS],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([x.replace("A-T-", "T:") for x in CONDITIONS])
    ax.set_xlabel("view the model blamed")
    ax.set_ylabel("condition")
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
def build(d, out="viz/consistency_claims.html", defects=None, theme="dark"):
    style.apply(theme)
    base = C.baselines(d)
    obf_svg, obf_stats, obf_cap = _obfuscation_block(d)
    obf_design = _obfuscation_design_note(d)
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
        "q1": (sev_banner + fig_sensitivity(d)
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
        "q2": (fig_blame_stack(d),
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
               "NOT the broken one. Only the view varies; naming and reasoning are "
               "pooled. Bars are 95% intervals. The six pairwise differences are in "
               "the details block."),
        "q4": ((obf_design + obf_svg) if obf_svg else obf_svg, obf_cap),
        "q5": (None, ""),
    }

    sections = []
    for qid, title, v in verdicts:
        svg, caption = figs.get(qid, (None, ""))
        if qid == "q3":
            svg = svg + fig_blame_matrix(d)
            caption += ("<br>Below: the full blame matrix, row-normalized. "
                        "Every cell is clickable and opens the runs behind it.")
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
        sections="\n".join(sections),
        drill=json.dumps(drill), n=len(d), hf=HF_URL,
        models=_esc(", ".join(sorted(d["model"].astype(str).unique()))))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(doc)
    print(f"[claims] wrote {out} ({os.path.getsize(out)/1e6:.2f} MB, "
          f"{len(verdicts)} questions, {len(drill)} drill-down cells)")
    return out


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-representation consistency — results</title>
<style>
  :root {{ --bg:#0d0f18; --panel:#12141e; --line:#1e2130; --fg:#e0e0e0;
           --muted:#8592ae; --dim:#5a6274; --accent:#7eb8ff; }}
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
              border-left:3px solid var(--dim); background:#141826; margin:0 0 18px; }}
  .v-supported {{ border-left-color:#4fa96a; }}
  .v-inconclusive {{ border-left-color:#c9a227; }}
  .v-unmeasured {{ border-left-color:#6b7a99; color:var(--muted); }}
  figure {{ margin:0 0 14px; background:var(--panel); border:1px solid var(--line);
            border-radius:8px; padding:14px; overflow-x:auto; }}
  figure svg {{ max-width:100%; height:auto; display:block; margin:0 auto; }}
  .designnote {{ font-size:0.82rem; color:#b8c2d6; background:#12141e;
                 border:1px solid #1e2130; border-radius:7px; padding:13px 16px;
                 margin:0 0 14px; line-height:1.65; }}
  .designnote b {{ color:#e0e0e0; }}
  .promoted {{ font-size:0.95rem; color:#cfd8e8; background:#141826;
               border-left:3px solid #7eb8ff; border-radius:6px;
               padding:12px 15px; margin:0 0 14px; }}
  .figsays {{ color:var(--muted); font-size:13px; margin:8px 2px 0; }}
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
                      color:#cfd8e8; }}
  h4 {{ color:var(--dim); font-size:0.74rem; text-transform:uppercase;
        letter-spacing:0.08em; margin:16px 0 6px; }}
  button {{ background:#1b2032; color:var(--fg); border:1px solid var(--line);
            border-radius:5px; padding:6px 11px; font-size:0.75rem; cursor:pointer;
            margin:4px 0 14px; font-family:inherit; }}
  button:hover {{ border-color:var(--accent); color:var(--accent); }}
  .pending {{ border:1px dashed #2a3450; color:var(--muted); padding:20px;
              border-radius:7px; font-size:0.85rem; }}
  [id^="cell|"], [id^="bar|"] {{ cursor:pointer; }}
  [id^="cell|"]:hover, [id^="bar|"]:hover {{ opacity:0.72; }}
  #drawer {{ position:fixed; top:0; right:0; width:min(620px,94vw); height:100%;
             background:#0f1119; border-left:1px solid var(--line); overflow-y:auto;
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
  .run .just {{ white-space:pre-wrap; font-size:0.79rem; color:#cfd8e8; }}
  .run .defect {{ font-size:0.79rem; color:#e0c88f; }}
  .close {{ position:absolute; top:14px; right:18px; }}
  .toast {{ position:fixed; bottom:22px; left:50%; transform:translateX(-50%);
            background:#1b2032; border:1px solid var(--accent); color:var(--accent);
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
