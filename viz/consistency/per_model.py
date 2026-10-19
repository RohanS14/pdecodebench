"""Per-model appendix: the pooled analyses, repeated one checkpoint at a time.

WHAT THIS IS FOR
================
Every headline number in the report pools eight checkpoints. Pooling is the right
default -- the design crosses 32 solver systems with 8 conditions and 2 naming
levels, and no single model has the power to carry that on its own -- but it cannot
distinguish "all eight models do this" from "two models do this hard enough to move
the mean". This module answers only that question. Nothing here is a new result and
nothing here recomputes a pooled one.

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not touch claim_report.build(). The pooled figures, verdicts and drill-downs
are produced there and are left exactly as they are; this section is injected into
the finished document afterwards, the same way the ladder and the raw-response
browser are. That is a hard constraint, not a stylistic one: claim_report.build()
also produces the frozen viz/consistency_claims.html, so anything added inside it
would rewrite the published report as a side effect.

EVERY QUANTITY IS COMPUTED BY THE POOLED CODE PATH
==================================================
Detection calls sensitivity.detection_sensitivity per model -- the same bootstrap
over solver systems, the same log-linear d' correction, the same 95% percentile
interval. Obfuscation calls prior_weakening.analyse per model, which is the function
behind the pooled obfuscation figure. Innocent-blame reuses the denominator that
fig_blame_levels uses (ALL draws where the view was not the corrupted one, clean items
and "said they agree" included), because a per-model panel computed against a
different denominator would look like a disagreement with the pooled figure when it
was only a disagreement with itself.

The three obfuscation outcomes the pooled analysis separates -- overall correctness,
correctness GIVEN the model committed, and how often it commits at all -- are kept
separate here too. They are not interchangeable: obfuscation is already known to make
models decline more often, and overall accuracy multiplies that into correctness.
"""
import html as _h

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from . import metrics as M
from . import style
from .claim_report import _svg
from .constants import (MODALITIES, MODALITY_COLORS, MODALITY_LABELS, NONE,
                        NONE_COLOR, NAMING_LEVELS)

# 8 checkpoints in 4x2. Four across keeps each panel wide enough for a 0-100% axis
# with readable ticks; eight across would make every panel a sliver, and two across
# would run the section four screens long.
NCOL = 4
# Wider than the 5.5in text column on purpose. These are small multiples in an HTML
# appendix, not a paper figure -- the constraint here is that eight panels stay
# comparable, which needs width, and the report already carries 7-8in figures.
GRID_W = 1.95


def _panels(n, per_h, ncol=NCOL):
    """A grid sized from the panel count, with shared x and a flat axes list."""
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, sharex=True,
                             figsize=style.figsize(GRID_W, per_h * nrow + 0.75))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:                       # unused cells in the last row
        ax.set_visible(False)
    return fig, axes[:n], nrow


def _pct(v):
    return "&mdash;" if not np.isfinite(v) else f"{100 * v:.1f}%"


def _label(ax, lo, hi, y, v, c, muted=False):
    """Print a rate beside its interval, flipping side when it would run off.

    A dot plot with no numbers forces the reader to measure against the axis, and
    the whole point of this section is reading exact values off eight panels at
    once. Rows whose interval reaches the right edge get their label on the LEFT
    instead, so nothing is clipped and nothing overlaps the next panel.
    """
    if not np.isfinite(v):
        return
    right = np.isfinite(hi) and hi <= 0.72
    ax.annotate(f"{100 * v:.0f}", (hi if right else lo, y),
                xytext=(3 if right else -3, 0), textcoords="offset points",
                va="center", ha="left" if right else "right",
                fontsize=style.TICK_PT - 2.5,
                color=c["muted"] if muted else c["fg"])


def _pp(v):
    return "&mdash;" if not np.isfinite(v) else f"{100 * v:+.1f}"


# ── 1. detection ─────────────────────────────────────────────────────────────
def detection_by_model(d, models, n_boot=None):
    """sensitivity.detection_sensitivity, once per checkpoint. Same method, same CIs."""
    from .sensitivity import detection_sensitivity
    out = {}
    for mid, _short in models:
        sub = d[d["model"].astype(str).eq(mid)]
        if sub.empty:
            continue
        out[mid] = detection_sensitivity(
            sub, **({"n_boot": n_boot} if n_boot else {}))
    return out


def fig_detection_facets(res, models):
    """One panel per model; rows are conditions; the dashed line is that model's floor.

    The x-axis is shared and pinned to [0, 1] rather than fitted per panel. A fitted
    axis would make every model look equally spread and is the one thing these
    panels must not do -- the comparison IS the axis.
    """
    from .sensitivity import row_caption_corrupted
    style.apply(style.theme())
    c = style.colors()
    have = [(mid, s) for mid, s in models if mid in res]
    fig, axes, nrow = _panels(len(have), per_h=1.55)
    if not have:
        return fig

    # Row order is fixed ACROSS panels and taken from the first model, not sorted
    # per panel. Sorting each panel by its own rate would put a different condition
    # on each line and destroy the only comparison this figure exists to support.
    ref = res[have[0][0]]
    conds = [r["condition"] for r in ref.rows if not r["empty"]]
    ypos = {cond: len(conds) - i for i, cond in enumerate(conds)}

    for k, (mid, short) in enumerate(have):
        ax, r = axes[k], res[mid]
        by = {x["condition"]: x for x in r.rows}
        if np.isfinite(r.fa_rate):
            ax.axvspan(0, r.fa_rate, color=c["muted"], alpha=0.10, zorder=0)
            ax.axvline(r.fa_rate, color=c["muted"], linewidth=1.0,
                       linestyle=(0, (4, 3)), zorder=1)
        for cond in conds:
            x = by.get(cond)
            if x is None or x["empty"]:
                continue
            yi = ypos[cond]
            col = MODALITY_COLORS[x["modality"]] if not x["thin"] else c["muted"]
            ax.plot([x["lo_rate"], x["hi_rate"]], [yi, yi], color=col,
                    linewidth=1.2, solid_capstyle="round", zorder=3)
            ax.scatter([x["hit_rate"]], [yi], s=26, color=col, zorder=4)
            _label(ax, x["lo_rate"], x["hi_rate"], yi, x["hit_rate"], c)
        # The clean row, below the rule, in grey -- same convention as the pooled
        # figure, where the baseline is a DATA row rather than a hidden annotation.
        if np.isfinite(r.fa_rate):
            ax.plot([r.fa_lo, r.fa_hi], [0, 0], color=c["muted"], linewidth=1.2,
                    solid_capstyle="round", zorder=3)
            ax.scatter([r.fa_rate], [0], s=26, color=c["muted"], zorder=4)
            _label(ax, r.fa_lo, r.fa_hi, 0, r.fa_rate, c, muted=True)
        ax.axhline(0.5, color=c["muted"], linewidth=0.5)
        ax.set_title(short, fontsize=style.TICK_PT, color=c["fg"], pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(conds) + 0.6)
        ax.set_yticks([ypos[cnd] for cnd in conds] + [0])
        if k % NCOL == 0:
            ax.set_yticklabels([row_caption_corrupted(cnd) for cnd in conds]
                               + ["nothing corrupted"], fontsize=style.TICK_PT - 1.5)
            ax.get_yticklabels()[-1].set_color(c["muted"])
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"],
                           fontsize=style.TICK_PT - 1)
        ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
        ax.set_axisbelow(True)
        if k // NCOL == nrow - 1:
            ax.set_xlabel("flagged as disagreeing", fontsize=style.TICK_PT)
    fig.tight_layout(w_pad=0.6, h_pad=1.1)
    return fig


def detection_table(res, models):
    from .sensitivity import row_caption_corrupted
    have = [(mid, s) for mid, s in models if mid in res]
    if not have:
        return ""
    conds = [r["condition"] for r in res[have[0][0]].rows if not r["empty"]]
    head = ("<tr><th>model</th><th>nothing corrupted</th>"
            + "".join(f"<th>{_h.escape(row_caption_corrupted(c))}</th>"
                      for c in conds) + "</tr>")
    body = []
    for mid, short in have:
        r = res[mid]
        by = {x["condition"]: x for x in r.rows}
        cells = [f"<td>{_pct(r.fa_rate)} <span style='color:var(--dim)'>"
                 f"({r.n_fa}/{r.n_noise})</span></td>"]
        for cnd in conds:
            x = by.get(cnd)
            cells.append("<td>&mdash;</td>" if x is None or x["empty"] else
                         f"<td>{_pct(x['hit_rate'])} "
                         f"<span style='color:var(--dim)'>({x['n_hit']}/"
                         f"{x['n_signal']}, d&prime;={x['dprime']:+.2f})</span></td>")
        body.append(f"<tr><td>{_h.escape(short)}</td>" + "".join(cells) + "</tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


# ── 2. blame ─────────────────────────────────────────────────────────────────
# Predicted-outlier buckets, in a fixed draw order. "said they agree" is its OWN
# bucket and is never folded into a wrong view: the pooled unconditional figure
# makes the same distinction, and collapsing it would convert a model that declines
# into a model that guesses wrong.
_SAID_AGREE = "__agree__"
_BLAME_ORDER = list(MODALITIES) + [_SAID_AGREE]
_BLAME_LABEL = {**MODALITY_LABELS, _SAID_AGREE: "said all four agree"}


def blame_by_model(d, models):
    """Per model: blame distribution per true outlier, plus the derived rates."""
    out = {}
    for mid, _short in models:
        sub = M.prepare(d[d["model"].astype(str).eq(mid)])
        if sub.empty:
            continue
        rec = {"dist": {}, "n_by_true": {}}
        for m in MODALITIES:
            den = sub[sub["true_outlier"].eq(m)]
            n = len(den)
            rec["n_by_true"][m] = n
            if not n:
                continue
            named = den["pred_outlier"].where(den["detected"])
            counts = {cat: int(named.eq(cat).sum()) for cat in MODALITIES}
            counts[_SAID_AGREE] = int((~den["detected"]).sum())
            # Anything the parser could not read is folded into no bucket; it is
            # reported as the shortfall so the segments never silently sum short.
            rec["dist"][m] = {"counts": counts, "n": n,
                              "unread": n - sum(counts.values())}
        # Correct outlier GIVEN the model committed -- the pooled conditional
        # quantity, denominator = corrupted AND flagged.
        elig = sub[sub["localization_eligible"]]
        k, n = int(elig["localization_correct"].sum()), len(elig)
        rec["cond_loc"] = (k, n, k / n if n else float("nan"),
                           *M.wilson_ci(k, n))
        # Detection on corrupted items and on clean items, same definitions as
        # detection_sensitivity's hit and false-alarm rates.
        corr = sub[sub["is_corrupted"]]
        kc, nc = int(corr["detected"].sum()), len(corr)
        rec["corrupt_flag"] = (kc, nc, kc / nc if nc else float("nan"))
        clean = sub[~sub["is_corrupted"]]
        kk, nn = int(clean["detected"].sum()), len(clean)
        rec["clean_flag"] = (kk, nn, kk / nn if nn else float("nan"),
                             *M.wilson_ci(kk, nn))
        # Innocent-blame, with fig_blame_levels' denominator EXACTLY: every draw
        # where this view was not the corrupted one, clean items and declines included.
        rec["innocent"] = {}
        for m in MODALITIES:
            den = sub[sub["true_outlier"].ne(m)]
            k2, n2 = int(den["pred_outlier"].eq(m).sum()), len(den)
            rec["innocent"][m] = (k2, n2, k2 / n2 if n2 else float("nan"),
                                  *M.wilson_ci(k2, n2))
        rec["n_scored"] = len(sub)
        out[mid] = rec
    return out


def fig_blame_facets(res, models):
    """One panel per model, four stacked bars each: where blame went, per true view."""
    style.apply(style.theme())
    c = style.colors()
    have = [(mid, s) for mid, s in models if mid in res]
    fig, axes, nrow = _panels(len(have), per_h=1.30)
    if not have:
        return fig
    rows = [m for m in MODALITIES]
    for k, (mid, short) in enumerate(have):
        ax, rec = axes[k], res[mid]
        for i, m in enumerate(rows):
            yi = len(rows) - i
            dist = rec["dist"].get(m)
            if not dist:
                continue
            left = 0.0
            for cat in _BLAME_ORDER:
                w = dist["counts"].get(cat, 0) / dist["n"]
                if w <= 0:
                    continue
                if cat == _SAID_AGREE:
                    ax.barh(yi, w, left=left, height=0.62, facecolor=c["panel"],
                            hatch="///", edgecolor=c["hatch"], linewidth=1.0,
                            **style.hatch_kw())
                else:
                    ax.barh(yi, w, left=left, height=0.62,
                            color=MODALITY_COLORS[cat],
                            edgecolor=c["panel"], linewidth=1.0)
                if cat == m:           # the diagonal: blame that landed correctly
                    ax.barh(yi, w, left=left, height=0.62, facecolor="none",
                            edgecolor=c["fg"], linewidth=1.3, zorder=4)
                # 14%, not the pooled figure's 7%: these panels are a quarter the
                # width, so the text needs a proportionally wider segment to sit in.
                # Narrower segments are left unlabelled rather than overplotted --
                # every value is in the details table underneath.
                if w >= 0.14:
                    ax.text(left + w / 2, yi, f"{100 * w:.0f}", ha="center",
                            va="center", fontsize=style.TICK_PT - 2.5,
                            color=c["muted"] if cat == _SAID_AGREE else c["bg"],
                            zorder=5)
                left += w
        ax.set_title(short, fontsize=style.TICK_PT, color=c["fg"], pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.3, len(rows) + 0.7)
        ax.set_yticks([len(rows) - i for i in range(len(rows))])
        if k % NCOL == 0:
            ax.set_yticklabels([f"{MODALITY_LABELS[m]} corrupted" for m in rows],
                               fontsize=style.TICK_PT - 1.5)
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"], fontsize=style.TICK_PT - 1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if k // NCOL == nrow - 1:
            ax.set_xlabel("share of items", fontsize=style.TICK_PT)
    handles = [Patch(facecolor=MODALITY_COLORS[m], label=MODALITY_LABELS[m])
               for m in MODALITIES]
    handles.append(Patch(facecolor=c["panel"], hatch="///", edgecolor=c["hatch"],
                         label=_BLAME_LABEL[_SAID_AGREE], **style.hatch_kw()))
    handles.append(Patch(facecolor="none", edgecolor=c["fg"], linewidth=1.3,
                         label="blamed the view that was actually corrupted"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=style.TICK_PT, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.6, h_pad=1.1, rect=(0, 0.07, 1, 1))
    return fig


def fig_innocent_facets(res, models):
    """The pooled section-3 question, per model: who gets blamed while innocent.

    Drawn as one grouped panel rather than eight, because the comparison here is
    BETWEEN models on four bars, not within a model -- and four bars per panel is
    below the size where a small multiple earns its whitespace.
    """
    style.apply(style.theme())
    c = style.colors()
    have = [(mid, s) for mid, s in models if mid in res]
    fig, ax = plt.subplots(figsize=style.figsize(1.35, 0.30 * len(have) + 1.15))
    if not have:
        style.empty_axes(ax, "no rows")
        return fig
    h = 0.19
    for j, m in enumerate(MODALITIES):
        off = (j - (len(MODALITIES) - 1) / 2) * h
        ys, vs, los, his = [], [], [], []
        for i, (mid, _s) in enumerate(have):
            k, n, v, lo, hi = res[mid]["innocent"][m]
            ys.append(len(have) - 1 - i + off)
            vs.append(100 * v); los.append(100 * lo); his.append(100 * hi)
        ax.barh(ys, vs, height=h, color=MODALITY_COLORS[m],
                label=MODALITY_LABELS[m])
        for y, lo, hi, v in zip(ys, los, his, vs):
            ax.plot([lo, hi], [y, y], color=c["fg"], linewidth=0.9)
            ax.annotate(f"{v:.1f}", (hi, y), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=style.TICK_PT - 2, color=c["fg"])
    ax.set_yticks([len(have) - 1 - i for i in range(len(have))])
    ax.set_yticklabels([s for _m, s in have], fontsize=style.TICK_PT)
    ax.set_xlabel("blamed while innocent (% of draws where this view was NOT corrupted)")
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False,
              fontsize=style.TICK_PT, handlelength=1.1, handletextpad=0.5,
              labelspacing=0.5, borderaxespad=0.0)
    return fig


def blame_table(res, models):
    have = [(mid, s) for mid, s in models if mid in res]
    head = ("<tr><th>model</th><th>correct outlier | committed</th>"
            + "".join(f"<th>{MODALITY_LABELS[m]} blamed innocent</th>"
                      for m in MODALITIES) + "</tr>")
    body = []
    for mid, short in have:
        r = res[mid]
        k, n, v, lo, hi = r["cond_loc"]
        cells = [f"<td>{_pct(v)} <span style='color:var(--dim)'>({k}/{n}, "
                 f"{_pct(lo)}&ndash;{_pct(hi)})</span></td>"]
        for m in MODALITIES:
            k2, n2, v2, lo2, hi2 = r["innocent"][m]
            cells.append(f"<td>{_pct(v2)} <span style='color:var(--dim)'>"
                         f"({k2}/{n2})</span></td>")
        body.append(f"<tr><td>{_h.escape(short)}</td>" + "".join(cells) + "</tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


def blame_dist_table(res, models):
    have = [(mid, s) for mid, s in models if mid in res]
    head = ("<tr><th>model</th><th>view actually corrupted</th><th>n</th>"
            + "".join(f"<th>blamed {_BLAME_LABEL[c]}</th>" for c in _BLAME_ORDER)
            + "<th>unreadable</th></tr>")
    body = []
    for mid, short in have:
        for i, m in enumerate(MODALITIES):
            dist = res[mid]["dist"].get(m)
            if not dist:
                continue
            cells = "".join(
                f"<td>{_pct(dist['counts'].get(cat, 0) / dist['n'])} "
                f"<span style='color:var(--dim)'>({dist['counts'].get(cat, 0)})"
                f"</span></td>" for cat in _BLAME_ORDER)
            body.append(
                f"<tr><td>{_h.escape(short) if i == 0 else ''}</td>"
                f"<td>{MODALITY_LABELS[m]}</td><td>{dist['n']:,}</td>"
                + cells + f"<td>{dist['unread']}</td></tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


# ── 3. obfuscation ───────────────────────────────────────────────────────────
def obfuscation_by_model(d, models, n_boot=None):
    """prior_weakening.analyse per model -- the function behind the pooled figure."""
    from . import prior_weakening as PW
    out = {}
    for mid, _short in models:
        sub = d[d["model"].astype(str).eq(mid)]
        if sub.empty:
            continue
        r = PW.analyse(sub, **({"n_boot": n_boot} if n_boot else {}))
        if r.overall is not None and np.isfinite(r.overall.diff):
            out[mid] = r
    return out


def fig_obfuscation_facets(res, models):
    """One panel per model. Filled = real names, hollow = obfuscated, line = the gap.

    Rows are the pooled contrast plus the four per-representation rows, in the same
    order and with the same colours as the pooled figure, so a reader can lay the
    two side by side.
    """
    style.apply(style.theme())
    c = style.colors()
    have = [(mid, s) for mid, s in models if mid in res]
    fig, axes, nrow = _panels(len(have), per_h=1.55)
    if not have:
        return fig

    lo_all, hi_all = [], []
    for mid, _s in have:
        r = res[mid]
        for cst in [r.overall] + list(r.per_outlier):
            lo_all += [100 * min(cst.real, cst.obf)]
            hi_all += [100 * max(cst.real, cst.obf)]
    xlo = max(0.0, np.floor((min(lo_all) - 6) / 10) * 10)
    xhi = min(100.0, np.ceil((max(hi_all) + 6) / 10) * 10)

    for k, (mid, short) in enumerate(have):
        ax, r = axes[k], res[mid]
        rows = [(r.overall, None)] + [(x, x.name) for x in r.per_outlier]
        ypos = [float(len(rows) - 1) + 0.85] + [float(len(rows) - 1 - i)
                                               for i in range(len(rows) - 1)]
        for yi, (cst, key) in zip(ypos, rows):
            head = key is None
            col = c["bar"] if head else MODALITY_COLORS.get(key, c["muted"])
            # The interval belongs to the PAIRED difference and is anchored at the
            # real-names dot, so its far end is where the obfuscated dot could sit.
            # Marginal intervals on the two dots would be the wrong quantity twice
            # over: both arms are the same solvers, and the reader would compare two
            # overlapping bars instead of asking whether the GAP clears zero.
            # Without it this panel printed a signed delta per row with nothing to
            # say which were separable -- and per model, at 32 solvers, most are not.
            if np.isfinite(cst.lo) and np.isfinite(cst.hi):
                a = 100 * cst.real
                ax.plot([a + 100 * cst.lo, a + 100 * cst.hi], [yi, yi], color=col,
                        linewidth=0.7, alpha=0.5, zorder=1)
            ax.plot([100 * cst.real, 100 * cst.obf], [yi, yi], color=col,
                    linewidth=1.3, alpha=0.9, zorder=2)
            ax.scatter([100 * cst.real], [yi], s=34 if head else 24, color=col,
                       zorder=4)
            ax.scatter([100 * cst.obf], [yi], s=34 if head else 24,
                       facecolor=c["panel"], edgecolor=col, linewidth=1.3, zorder=4)
            # The gap is the quantity; printing it per row means the four
            # exploratory rows are readable without measuring against the axis.
            # The head row says whether it cleared zero. The four below it do NOT
            # get a significance mark: analyse() flags them exploratory, and a mark
            # would promote an underpowered per-representation split to a result.
            txt = f"{100 * cst.diff:+.1f}"
            if head and not cst.significant:
                txt += " n.s."
            ax.annotate(txt, (1.02, yi),
                        xycoords=("axes fraction", "data"), va="center", ha="left",
                        fontsize=style.TICK_PT - 2.5,
                        color=c["fg"] if head else c["muted"],
                        weight="bold" if head else "normal")
        ax.axhline(len(rows) - 1 + 0.42, color=c["muted"], linewidth=0.5)
        ax.set_title(short, fontsize=style.TICK_PT, color=c["fg"], pad=3)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(min(ypos) - 0.55, max(ypos) + 0.55)
        ax.set_yticks(ypos)
        if k % NCOL == 0:
            ax.set_yticklabels(
                ["ALL corrupted items"]
                + [f"{MODALITY_LABELS.get(x.name, x.name)} corrupted"
                   for x in r.per_outlier], fontsize=style.TICK_PT - 1.5)
        else:
            ax.set_yticklabels([])
        ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=style.TICK_PT - 1)
        if k // NCOL == nrow - 1:
            ax.set_xlabel("% naming the right view", fontsize=style.TICK_PT)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=c["muted"],
                   markeredgecolor=c["muted"], markersize=7,
                   label="real identifiers (comments removed in both arms)"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=c["panel"],
                   markeredgecolor=c["muted"], markeredgewidth=1.3, markersize=7,
                   label="obfuscated identifiers"),
        plt.Line2D([], [], color=c["muted"], linewidth=0.7, alpha=0.6,
                   label="95% CI on the paired difference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=style.TICK_PT, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=2.4, h_pad=1.1, rect=(0, 0.06, 1, 1))
    return fig


def obfuscation_table(res, models):
    """All three outcomes the pooled analysis keeps apart, kept apart per model."""
    have = [(mid, s) for mid, s in models if mid in res]
    head = ("<tr><th>model</th><th>outcome</th><th>real</th><th>obfuscated</th>"
            "<th>obf &minus; real (pp)</th><th>95% CI</th><th>solvers</th>"
            "<th>n items</th></tr>")
    body = []
    for mid, short in have:
        r = res[mid]
        named = [("named the right view, of ALL corrupted items", r.overall),
                 ("correct GIVEN it committed", r.primary),
                 ("committed to a verdict at all", r.detection)]
        named += [(f"&mdash; {MODALITY_LABELS.get(x.name, x.name)} was corrupted", x)
                  for x in r.per_outlier]
        for i, (lab, cst) in enumerate(named):
            if cst is None or not np.isfinite(cst.diff):
                continue
            body.append(
                f"<tr><td>{_h.escape(short) if i == 0 else ''}</td><td>{lab}</td>"
                f"<td>{_pct(cst.real)}</td><td>{_pct(cst.obf)}</td>"
                f"<td>{_pp(cst.diff)}</td>"
                f"<td>{_pp(cst.lo)} to {_pp(cst.hi)}"
                + ("" if not cst.exploratory else
                   " <span style='color:var(--dim)'>(exploratory)</span>")
                + f"</td><td>{cst.n_solvers}</td><td>{cst.n_items:,}</td></tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


# ── 4. summary ───────────────────────────────────────────────────────────────
def summary_table(blame, obf, models, dropped=None):
    head = ("<tr><th>model</th><th>scored n</th><th>no-verdict</th>"
            "<th>clean disagreement rate</th>"
            "<th>corrupted-item disagreement rate</th>"
            "<th>correct outlier | disagreement</th>"
            "<th>trajectory blamed innocent</th>"
            "<th>obfuscation effect (pp)</th></tr>")
    body = []
    for mid, short in models:
        b = blame.get(mid)
        if b is None:
            continue
        _kc, nc, vc = b["corrupt_flag"]
        kk, nn, vk, _lo, _hi = b["clean_flag"]
        _k, n_el, v_loc, _l, _hh = b["cond_loc"]
        _kt, nt, vt, _lt, _ht = b["innocent"]["T"]
        o = obf.get(mid)
        ocell = "&mdash;"
        if o is not None and np.isfinite(o.overall.diff):
            ocell = (f"{_pp(o.overall.diff)} <span style='color:var(--dim)'>"
                     f"({_pp(o.overall.lo)} to {_pp(o.overall.hi)})</span>")
        drop = "&mdash;" if dropped is None else f"{dropped.get(mid, 0):,}"
        body.append(
            f"<tr><td>{_h.escape(short)}</td><td>{b['n_scored']:,}</td>"
            f"<td>{drop}</td>"
            f"<td>{_pct(vk)} <span style='color:var(--dim)'>({kk}/{nn})</span></td>"
            f"<td>{_pct(vc)} <span style='color:var(--dim)'>({nc:,})</span></td>"
            f"<td>{_pct(v_loc)} <span style='color:var(--dim)'>({n_el:,})</span></td>"
            f"<td>{_pct(vt)} <span style='color:var(--dim)'>({nt:,})</span></td>"
            f"<td>{ocell}</td></tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


# ── section ──────────────────────────────────────────────────────────────────
def _spread(vals):
    v = [x for x in vals if np.isfinite(x)]
    return (max(v) - min(v)) if len(v) > 1 else float("nan")


def build_section(d, models, dropped=None, n_boot=None):
    """The whole appendix as one HTML <section>. Pure function of `d`."""
    det = detection_by_model(d, models, n_boot=n_boot)
    blame = blame_by_model(d, models)
    obf = obfuscation_by_model(d, models, n_boot=n_boot)

    det_svg = _svg(fig_detection_facets(det, models))
    blame_svg = _svg(fig_blame_facets(blame, models))
    inn_svg = _svg(fig_innocent_facets(blame, models))
    obf_svg = _svg(fig_obfuscation_facets(obf, models))

    # Read-outs stated from the computed numbers rather than asserted in prose, so
    # the text cannot drift from the figures on a rebuild.
    n_models = len(blame)
    floors = [blame[m]["clean_flag"][2] for m, _s in models if m in blame]
    every_above = sum(
        1 for mid, _s in models if mid in det
        and all(x["hit_rate"] > det[mid].fa_rate
                for x in det[mid].rows if not x["empty"]))
    traj_top = sum(
        1 for mid, _s in models if mid in blame
        and blame[mid]["innocent"]["T"][2] == max(
            blame[mid]["innocent"][m][2] for m in MODALITIES))
    obf_neg = sum(1 for mid, _s in models if mid in obf and obf[mid].overall.diff < 0)
    obf_sig = sum(1 for mid, _s in models if mid in obf
                  and obf[mid].overall.significant)

    return (
        '<section id="per-model">'
        '<h2>Per-model breakdowns</h2>'
        '<p class="sub" style="margin-bottom:18px">Every analysis in the main '
        f'report pools all {n_models} checkpoints. These panels show the same '
        'quantities computed separately for each model, to check whether the '
        'qualitative findings hold across checkpoints or are carried by a few. '
        '<b>This is an appendix, not a new result.</b> Nothing here is recomputed '
        'differently: detection calls the same bootstrap over solver systems as the '
        'pooled figure, obfuscation calls the same paired analysis, and '
        'innocent-blame uses the same denominator. Per-model cells are a fraction '
        'of the pooled n, so intervals are wide and no single panel settles '
        'anything on its own.</p>'

        '<h3 class="pmh">1 &middot; Disagreement detection, by model</h3>'
        '<p class="sub">One panel per checkpoint. Each dot is how often that model '
        'flagged a disagreement for that corruption, with the 95% interval; the '
        'dashed line and shaded band are <i>that model&rsquo;s own</i> rate on '
        'clean items, which is the floor its other rows have to clear. The x-axis '
        'is identical in every panel and pinned to 0&ndash;100% &mdash; a fitted '
        'per-panel axis would make every model look equally spread, which is the '
        'one thing these panels must not do. Row order is fixed across panels for '
        'the same reason. '
        f'<b>{every_above} of {len(det)} models</b> put every corruption above '
        'their own clean-item floor. Clean-item flag rates themselves range '
        f'{_pct(min(floors))} to {_pct(max(floors))}, a spread of '
        f'{100 * _spread(floors):.0f}pp &mdash; the models differ far more in how '
        'readily they cry foul than in what they can detect.</p>'
        f'<figure>{det_svg}</figure>'
        '<p class="sub"><b>Caveat, carried over from the pooled analysis.</b> The '
        'corruptions are <b>not severity-matched</b> across representations: '
        'trajectory carries four generation methods and the other three views carry '
        'one each. Ordering within trajectory, and among the three single-method '
        'views, is comparable; ordering between those two groups is not.</p>'
        '<details><summary class="sub">Details &mdash; flag rate, count and '
        'd&prime; per model per condition</summary>'
        f'<div style="overflow-x:auto">{detection_table(det, models)}</div>'
        '</details>'

        '<h3 class="pmh">2 &middot; Outlier localization and blame, by model</h3>'
        '<p class="sub">Each bar is all the items where one view really was the '
        'corrupted one, divided by where that model put the blame. The hatched segment '
        'is the model saying all four agree &mdash; kept as its own category and '
        '<b>never folded into a wrong answer</b>, since declining and guessing '
        'wrong are different failures and the pooled figure separates them too. The '
        'outlined segment is blame that landed on the view that was actually '
        'corrupted.</p>'
        f'<figure>{blame_svg}</figure>'
        '<p class="sub">Below: the pooled section&rsquo;s question asked per model '
        '&mdash; how often each view is blamed on the draws where it was '
        '<i>not</i> the corrupted one. Denominator is every such draw, clean items and '
        'declines included, exactly as in the pooled figure. '
        f'<b>Trajectory takes the most innocent blame in {traj_top} of '
        f'{len(blame)} models</b>, which is the per-model form of the pooled '
        'finding.</p>'
        f'<figure>{inn_svg}</figure>'
        '<details><summary class="sub">Details &mdash; conditional localization, '
        'innocent-blame rates, and the full blame distribution</summary>'
        f'<div style="overflow-x:auto">{blame_table(blame, models)}</div>'
        '<p class="sub" style="margin-top:16px">Full distribution, one row per '
        '(model, corrupted view). <b>unreadable</b> is the shortfall between the '
        'segments and the row total &mdash; a flagged draw whose named view the '
        'parser could not resolve. It is shown rather than absorbed so the segments '
        'can be checked against n.</p>'
        f'<div style="overflow-x:auto">{blame_dist_table(blame, models)}</div>'
        '</details>'

        '<h3 class="pmh">3 &middot; Identifier obfuscation, by model</h3>'
        '<p class="sub">Filled dot = real variable names, hollow = obfuscated, and '
        'the line between them is the effect; the number beside each model is '
        '<i>obfuscated &minus; real</i> on the pooled top row. Same outcome as the '
        'pooled obfuscation figure: of the items where something really was corrupted, '
        'how often the model named the view that was corrupted. '
        f'<b>{obf_neg} of {len(obf)} models move in the negative direction</b>, and '
        f'<b>{obf_sig}</b> have an interval excluding zero on their own. Paired '
        'within solver across the naming factor, bootstrapped over solver systems.</p>'
        f'<figure>{obf_svg}</figure>'
        '<p class="sub">The four per-representation rows in each panel are '
        '<b>exploratory</b>: they divide one model&rsquo;s solvers four ways, and '
        'at that size the smallest detectable effect is larger than most of the '
        'effects being looked for.</p>'
        '<details><summary class="sub">Details &mdash; all three outcomes, kept '
        'separate</summary>'
        '<p class="sub">The pooled analysis distinguishes overall correctness, '
        'correctness <i>given the model committed</i>, and how often it commits at '
        'all &mdash; because obfuscation is already known to make models decline '
        'more often, and overall accuracy multiplies that into correctness. All '
        'three are reported here rather than one standing in for the others.</p>'
        f'<div style="overflow-x:auto">{obfuscation_table(obf, models)}</div>'
        '</details>'

        '<h3 class="pmh">4 &middot; Summary</h3>'
        '<p class="sub">One row per checkpoint. <b>scored n</b> is the draws that '
        'reached a verdict and entered these figures; <b>no-verdict</b> is what was '
        'dropped before scoring, so effective sample size differs slightly across '
        'models. Counts and denominators are in parentheses. The obfuscation column '
        'is <i>obfuscated &minus; real</i> in percentage points on the primary '
        'outcome, with its 95% interval.</p>'
        f'<div style="overflow-x:auto">{summary_table(blame, obf, models, dropped)}</div>'
        '</section>')
