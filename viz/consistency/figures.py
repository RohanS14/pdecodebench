"""The four paper figures. Each takes a DataFrame and returns a matplotlib Figure.

No function here hardcodes a condition, modality, or outcome list -- they all come
from `constants`, so adding a fifth representation changes one tuple rather than
four plotting functions that would otherwise keep drawing a 4-wide figure without
complaining.

Every figure degrades gracefully. A missing slice (reasoning=on not yet run, a
model absent, a blame row with no observations) draws as a labelled blank rather
than raising or, worse, silently rescaling so a partial run looks complete.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch

from . import generational as G
from . import metrics as M
from . import style
from .constants import (CONDITIONS, MODALITIES, MODALITY_COLORS, MODALITY_LABELS,
                        NAMING_LEVELS, NONE, OUTCOME_COLORS, OUTCOME_COLORS_DARK,
                        OUTCOME_HATCH,
                        OUTCOME_SCOPE, OUTCOMES, OUTLIER_LEVELS, REASONING_LEVELS,
                        SEQUENTIAL_CMAP, TRAJ_LEVELS, TRAJ_LEVEL_LABELS)

_LEVEL_LABEL = {NONE: "none", **MODALITY_LABELS}


def _num(v):
    """Anything that is not a real number becomes nan, so callers can test finiteness."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _tick_labels(levels):
    return [_LEVEL_LABEL.get(v, v) for v in levels]


# ── fig 1 ────────────────────────────────────────────────────────────────────
def fig1_blame_matrix(df, cmap=None, row_by="condition"):
    """Row-normalized 5x5 blame matrix, one panel per naming condition.

    Row-normalized because the question is "given this view was corrupted, where
    did the blame go" -- a column-normalized or raw-count heatmap answers a
    different question and invites reading the diagonal as accuracy when the row
    totals differ. Raw counts are annotated so the reader can still see how much
    data each cell rests on.
    """
    style.apply(style.theme())
    bm = M.blame_matrix(df, by=["naming"], row_by=row_by)
    row_levels = OUTLIER_LEVELS if row_by == "true_outlier" else CONDITIONS
    namings = [n for n in NAMING_LEVELS if n in set(bm.get("naming", []))]
    if not namings:
        fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.6))
        style.empty_axes(ax, "no rows")
        return fig

    height = 1.05 + 0.30 * len(row_levels)
    fig, axes = plt.subplots(
        1, len(namings), figsize=style.figsize(1.0, height), squeeze=False,
        gridspec_kw=dict(bottom=0.26, top=0.90, left=0.17, right=0.88, wspace=0.12))
    axes = axes[0]
    im = None
    for ax, naming in zip(axes, namings):
        sub = bm[bm["naming"].eq(naming)]
        grid = (sub.pivot(index=row_by, columns="pred_outlier", values="row_frac")
                .reindex(index=row_levels, columns=OUTLIER_LEVELS))
        counts = (sub.pivot(index=row_by, columns="pred_outlier", values="n")
                  .reindex(index=row_levels, columns=OUTLIER_LEVELS).fillna(0))
        vals = np.ma.masked_invalid(grid.to_numpy(dtype=float))
        im = ax.imshow(vals, cmap=cmap or style.colors()["cmap"], vmin=0, vmax=1, aspect="auto")
        # Hairline separators: the cells are a matrix, not a continuous field, and
        # without them adjacent pale cells read as one block.
        ax.set_xticks(np.arange(len(OUTLIER_LEVELS) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_levels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=style.colors()["sep"], linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(len(OUTLIER_LEVELS)))
        ax.set_yticks(range(len(row_levels)))
        ax.set_xticklabels(_tick_labels(OUTLIER_LEVELS), rotation=45, ha="right")
        ax.set_title(naming, pad=5, fontsize=style.BASE_PT)
        # Only the leftmost panel is labelled: repeating the row names on panel two
        # ran them straight through panel one's rightmost cells.
        if ax is axes[0]:
            ax.set_yticklabels(_tick_labels(row_levels))
            ax.set_ylabel("condition (what was corrupted)"
                          if row_by == "condition" else "actually corrupted")
        else:
            ax.set_yticklabels([])
        for i in range(len(row_levels)):
            for j in range(len(OUTLIER_LEVELS)):
                v = grid.to_numpy(dtype=float)[i, j]
                if np.isnan(v):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=style.ANNOT_PT, color=style.colors()["muted"])
                    continue
                ax.text(j, i, f"{int(counts.to_numpy()[i, j])}", ha="center",
                        va="center", fontsize=style.ANNOT_PT,
                        color=("white" if v > 0.55 else style.colors()["fg"])
                        if style.theme() == "light"
                        else ("#0b0d14" if v > 0.62 else style.colors()["fg"]))
        for s in ax.spines.values():
            s.set_visible(False)
    if im is not None:
        # Explicit axes rather than stealing space from the panels, so the bar can
        # never be repositioned out from under the layout it was sized against.
        cax = fig.add_axes([0.905, 0.26, 0.018, 0.64])
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cb.set_label("share of row", fontsize=style.TICK_PT, labelpad=3)
        cb.ax.tick_params(labelsize=style.TICK_PT, length=2)
        cb.outline.set_visible(False)
    # One axis label for the pair, not the same word twice under adjacent panels.
    # Bottom space is reserved in the gridspec above, before the colorbar exists.
    fig.supxlabel("view the model blamed", fontsize=style.BASE_PT, y=0.02)
    return fig


# ── fig 2 ────────────────────────────────────────────────────────────────────
def fig2_trust_scatter(df):
    """Detection rate against false-blame rate, one point per modality per naming.

    The arrow is the finding: if obfuscating identifiers moves a modality's point,
    the model's trust in that representation was keyed to its names rather than to
    the physics it encodes.
    """
    style.apply(style.theme())
    pm = M.per_modality_rates(df, by=["naming"])
    # The trajectory ladder gets its own panel rather than a vertical stack inside
    # the scatter: at four rungs the rung labels landed on top of the modality
    # labels, and the ladder shares an x with the T point anyway, so it carries no
    # information the scatter's x-axis can add.
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=style.figsize(1.0, 2.9),
        gridspec_kw=dict(width_ratios=[2.6, 1.0], wspace=0.42))
    _traj_ladder_axes(df, axl)
    if pm.empty:
        style.empty_axes(ax, "no rows")
        return fig

    wide = pm.pivot_table(index=["modality", "naming"], columns="metric",
                          values="rate")
    drawn = False
    for m in MODALITIES:
        pts = {}
        for naming in NAMING_LEVELS:
            if (m, naming) in wide.index:
                r = wide.loc[(m, naming)]
                # A slice can be missing a whole metric column, not just a value:
                # an A0-only run has no corrupted rows, so `detection_rate` never
                # gets built. Coerce through nan rather than indexing into None.
                x = _num(r.get("false_blame_rate", np.nan))
                y = _num(r.get("detection_rate", np.nan))
                if np.isfinite(x) and np.isfinite(y):
                    pts[naming] = (x, y)
        if not pts:
            continue
        drawn = True
        c = MODALITY_COLORS[m]
        for naming, (x, y) in pts.items():
            filled = naming == NAMING_LEVELS[0]
            ax.scatter([x], [y], s=34,
                       color=c if filled else style.colors()["panel"],
                       edgecolor=c, linewidth=1.1, zorder=3)
        if len(pts) == len(NAMING_LEVELS):
            (x0, y0), (x1, y1) = pts[NAMING_LEVELS[0]], pts[NAMING_LEVELS[1]]
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                color=c, linewidth=0.9, alpha=0.85, zorder=2,
                shrinkA=4.5, shrinkB=4.5))
        lx, ly = list(pts.values())[0]
        ax.annotate(MODALITY_LABELS[m], (lx, ly), textcoords="offset points",
                    xytext=(6, 5), fontsize=style.ANNOT_PT, color=c)
    if not drawn:
        style.empty_axes(ax, "no rows")
        return fig

    ax.set_xlabel("false-blame rate")
    ax.set_ylabel("detection rate")
    ax.grid(True, linewidth=0.4, color=style.colors()["faint"])
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#555555",
                   markeredgecolor="#555555", markersize=5, label=NAMING_LEVELS[0]),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                   markeredgecolor="#555555", markersize=5, label=NAMING_LEVELS[1]),
    ]
    ax.legend(handles=handles, loc="best", fontsize=style.TICK_PT)
    return fig


def _traj_ladder_axes(df, ax):
    """Detection rate per trajectory corruption rung, grossest at the top.

    This is the axis the single "trajectory" point in the scatter averages over. The
    spread across it is wider than the spread between whole modalities, which is the
    argument for not treating the trajectory as one condition.
    """
    lad = M.traj_ladder(df)
    if lad.empty:
        style.empty_axes(ax, "no corrupted\ntrajectories")
        return
    levels = list(lad["traj_level"])
    y = np.arange(len(levels))[::-1]
    rate = lad["rate"].to_numpy(dtype=float)
    err = np.vstack([rate - lad["lo"].to_numpy(dtype=float),
                     lad["hi"].to_numpy(dtype=float) - rate])
    ax.barh(y, rate, height=0.62, color=MODALITY_COLORS["T"], alpha=0.85,
            xerr=np.nan_to_num(err), error_kw=dict(elinewidth=0.7, ecolor=style.colors()["fg"]))
    ax.set_yticks(y)
    ax.set_yticklabels(levels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("detection rate")
    ax.set_title("trajectory rungs", pad=4, color=MODALITY_COLORS["T"])
    ax.grid(True, axis="x", linewidth=0.4, color=style.colors()["faint"])
    ax.set_axisbelow(True)


# ── fig 3 ────────────────────────────────────────────────────────────────────
def _outcome_colors():
    return OUTCOME_COLORS_DARK if style.theme() == "dark" else OUTCOME_COLORS


def fig3_outcome_stack(df):
    """Outcome composition per condition, faceted by reasoning setting."""
    style.apply(style.theme())
    d = M.prepare(df)
    present = [r for r in REASONING_LEVELS if r in set(d.get("reasoning", []))]
    # A facet the run has not produced still gets an axes, labelled blank, so the
    # figure reports the gap instead of hiding it behind a rescaled single panel.
    facets = present or [None]
    fig, axes = plt.subplots(1, max(len(facets), 1),
                             figsize=style.figsize(1.0, 3.0), squeeze=False,
                             sharey=True)
    axes = axes[0]
    for ax, facet in zip(axes, facets):
        sub = df if facet is None else df[df["reasoning"].eq(facet)]
        if sub.empty:
            style.empty_axes(ax, f"reasoning={facet}\nnot run")
            continue
        br = M.outcome_breakdown(sub, by=["condition"])
        conds = [c for c in CONDITIONS if c in set(br["condition"])]
        if not conds:
            style.empty_axes(ax, "no rows")
            continue
        piv = (br.pivot(index="condition", columns="outcome", values="share")
               .reindex(index=conds, columns=list(OUTCOMES)).fillna(0.0))
        bottom = np.zeros(len(conds))
        x = np.arange(len(conds))
        for outcome in OUTCOMES:
            vals = piv[outcome].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, width=0.70,
                   color=_outcome_colors()[outcome], hatch=OUTCOME_HATCH[outcome],
                   edgecolor=style.colors()["sep"], linewidth=0.7, label=outcome)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("A-T-", "T:") for c in conds],
                           rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title("reasoning: " + (facet if facet else "all"), pad=4)
        if ax is axes[0]:
            ax.set_ylabel("share of items")
    c = style.colors()
    handles = [Patch(facecolor=_outcome_colors()[o], hatch=OUTCOME_HATCH[o],
                     edgecolor=c["sep"],
                     label=f"{o}  ({OUTCOME_SCOPE[o]})") for o in OUTCOMES]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=style.TICK_PT, bbox_to_anchor=(0.5, -0.30))
    return fig


# ── fig 4 ────────────────────────────────────────────────────────────────────
def fig4_justification_gap(df):
    """Localization accuracy against judge-confirmed justification, per condition.

    Picking the right slot and explaining the right defect are different claims,
    and the gap between the paired bars is how much of the localization score is
    unsupported by the model's own stated reason.
    """
    style.apply(style.theme())
    loc = M.localization_accuracy(df, by=["condition"]).set_index("condition")
    jud = M.judge_rate(df).set_index("condition")
    conds = [c for c in CONDITIONS if c in loc.index or c in jud.index]
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.9))
    if not conds:
        style.empty_axes(ax, "no rows")
        return fig

    x = np.arange(len(conds))
    w = 0.36
    def _col(src, c, k):
        return float(src.loc[c, k]) if c in src.index else np.nan
    lv = np.array([_col(loc, c, "rate") for c in conds])
    jv = np.array([_col(jud, c, "rate") for c in conds])
    lerr = np.array([[_col(loc, c, "rate") - _col(loc, c, "lo") for c in conds],
                     [_col(loc, c, "hi") - _col(loc, c, "rate") for c in conds]])
    jerr = np.array([[_col(jud, c, "rate") - _col(jud, c, "lo") for c in conds],
                     [_col(jud, c, "hi") - _col(jud, c, "rate") for c in conds]])

    # NaN heights draw no bar at all, which is the point: A0 has no corrupted items,
    # so localization is undefined there. Substituting zero would draw a full-height
    # axis slot reading "0% accuracy" -- a claim about the model rather than about
    # the design. The undefined slot is labelled n/a instead.
    have_judge_pre = bool(np.isfinite(jv).any())
    off = (-w / 2) if have_judge_pre else 0.0
    ax.bar(x + off, lv, width=w, color=style.colors()["bar"],
           label="localization accuracy",
           yerr=np.nan_to_num(lerr), error_kw=dict(elinewidth=0.7, ecolor=style.colors()["fg"]))
    have_judge = bool(np.isfinite(jv).any())
    if have_judge:
        ax.bar(x + w / 2, jv, width=w, color=style.colors()["bar2"],
               edgecolor=style.colors()["sep"], linewidth=0.4, label="judge-confirmed",
               yerr=np.nan_to_num(jerr), error_kw=dict(elinewidth=0.7, ecolor=style.colors()["muted"]))
    else:
        # An empty legend swatch reads as "judge rate is zero". Say what is actually
        # true instead: the column does not exist in this run.
        ax.text(0.5, 0.94, "judge-confirmed rate unavailable — no LLM-judge pass "
                           "has been run over the justifications",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=style.ANNOT_PT, color=style.colors()["note_fg"],
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=style.colors()["note_bg"],
                          edgecolor=style.colors()["note_edge"], linewidth=0.6))

    for xi, a in zip(x, lv):
        if not np.isfinite(a):
            ax.annotate("n/a", (xi - w / 2, 0.0), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=style.ANNOT_PT,
                        color=style.colors()["muted"])

    for xi, a, b in zip(x, lv, jv):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        top = max(a, b)
        ax.annotate(f"{a - b:+.2f}", (xi, top), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=style.ANNOT_PT,
                    color=style.colors()["fg"])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("A-T-", "T:") for c in conds],
                       rotation=45, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", linewidth=0.4, color=style.colors()["faint"])
    ax.set_axisbelow(True)
    ax.set_ylabel("rate")
    ax.legend(loc="upper left", fontsize=style.TICK_PT)
    return fig


def fig_generational_trend(df, arm="on"):
    """Detection d' against model release date, one point per model.

    The y-axis is d' rather than hit rate on purpose: see generational.py. Marker
    area encodes total parameters, so the reader can see that size is held in a
    narrow band rather than having to take it on trust, and colour encodes family,
    so a Qwen-only trend stays visually distinguishable from a field-wide one.

    Degrades to a labelled empty axes when the registry is missing, when no model
    carries a release date, or when only one date is present -- the partial-run
    shapes tests/test_consistency_figures.py enforces for every other figure here.
    """
    style.apply(style.theme())
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 3.0))
    try:
        rows = G.by_release(df, arm=arm)
    except (KeyError, ValueError):
        rows = None
    if rows is None or rows.empty:
        style.empty_axes(ax, "no dated models\n(model_registry.csv missing?)")
        return fig

    c = style.colors()
    x = pd.to_datetime(rows["release_date"])
    fams = [f for f in dict.fromkeys(rows["family"]) if f]
    # Okabe-Ito, same family of hues the modality figures use, so the report reads
    # as one palette. Families are nominal, so any stable assignment will do.
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#F0E442", "#000000")
    cmap = {f: palette[i % len(palette)] for i, f in enumerate(fams)}

    sizes = rows["params_total_b"].to_numpy(dtype=float)
    sizes = np.where(np.isfinite(sizes), sizes, 30.0)
    for fam in fams:
        m = rows["family"].eq(fam).to_numpy()
        ax.errorbar(x[m], rows["dprime"][m],
                    yerr=[rows["dprime"][m] - rows["lo"][m],
                          rows["hi"][m] - rows["dprime"][m]],
                    fmt="none", ecolor=c["muted"], elinewidth=1.0, capsize=2.5,
                    zorder=1)
        ax.scatter(x[m], rows["dprime"][m], s=6.0 * sizes[m], zorder=2,
                   color=cmap[fam], edgecolor=c["sep"], linewidth=0.7, label=fam)

    # A degenerate arm is drawn hollow: d' is finite only via the Hautus
    # correction, so the point exists but carries no discrimination information.
    deg = rows["degenerate"].to_numpy(dtype=bool)
    if deg.any():
        ax.scatter(x[deg], rows["dprime"][deg], s=6.0 * sizes[deg], zorder=3,
                   facecolor="none", edgecolor=c["fg"], linewidth=1.6)

    ax.axhline(0.0, color=c["muted"], linewidth=0.9, linestyle=":", zorder=0)
    ax.set_ylabel("detection d′ (Hautus)")
    ax.set_xlabel("model release date")
    fig.autofmt_xdate(rotation=30)

    t = G.trend(rows)
    sub = (f"slope {t['slope_per_year']:+.2f} d′/year, "
           f"R²={t['r2']:.2f}, {t['n_models']} models over "
           f"{t['span_days']}d" if t else
           f"{len(rows)} model(s) — too few dated points to fit a trend")
    ax.set_title("Does cross-modal detection improve with model generation?\n"
                 + sub, pad=6)
    if fams:
        ax.legend(frameon=False, fontsize=7, loc="best", title=None)
    fig.tight_layout()
    return fig


# fig5* return (Figure, caption); build_all unwraps them. Kept out of FIGURES so
# callers that iterate it keep getting a bare Figure back.
CAPTIONED_FIGURES = {
    "fig5_obfuscation_contrast": None,      # bound below, after the defs
}

FIGURES = {
    "fig1_blame_matrix": fig1_blame_matrix,
    "fig2_trust_scatter": fig2_trust_scatter,
    "fig3_outcome_stack": fig3_outcome_stack,
    "fig4_justification_gap": fig4_justification_gap,
    "fig6_generational_trend": fig_generational_trend,
}


def build_all(df, outdir="figures"):
    """Build and save every figure. Returns {name: (pdf, png)}; captions in .captions."""
    out = {}
    for name, fn in FIGURES.items():
        fig = fn(df)
        out[name] = style.save(fig, name, outdir=outdir)
        plt.close(fig)
    captions = {}
    for name, fn in CAPTIONED_FIGURES.items():
        fig, cap = fn(df)
        out[name] = style.save(fig, name, outdir=outdir)
        captions[name] = cap
        plt.close(fig)
    build_all.captions = captions
    return out


# ── fig 5: the naming manipulation ───────────────────────────────────────────
def _blame_color(cat):
    from .constants import NONE_COLOR
    return MODALITY_COLORS.get(cat, NONE_COLOR)


def _row_label(cat):
    return MODALITY_LABELS.get(cat, cat)


def obfuscation_stats(df, n_boot=None):
    """The single source both the figure and the verdict line read from.

    Returned as one object so the two cannot diverge: the previous versions computed
    real-minus-obfuscated on a per-row slice with uncorrected alpha (verdict) and
    obfuscated-minus-real on a fixed slice with Bonferroni (figure), and produced
    opposite signs for the same effect.
    """
    from . import obfuscation as OB
    from .constants import DELTA_IS
    kw = {"n_boot": n_boot} if n_boot else {}
    ren = OB.paired_blame_shift(df, "innocent", renormalise=True, **kw)
    raw = OB.paired_blame_shift(df, "innocent", **kw)
    guilty = OB.paired_guilty_recall(df, **kw)
    by_cond = OB.paired_refusal_by_condition(df, **kw)
    none_row = next((r for r in raw.rows if r["category"] == NONE), None)
    return {
        "delta_is": DELTA_IS,
        "rows": ren.rows,                 # four modalities, renormalised over blame
        "raw_rows": raw.rows,             # the un-renormalised five-way shares
        "none": none_row,
        "guilty": guilty.rows[0] if guilty.rows else None,
        # The slice disaggregated: does obfuscation quiet the model uniformly, or
        # only when particular views are corrupted?
        "by_condition": [r for r in by_cond
                         if r["condition"] != "A0"],
        "n_solvers": ren.n_solvers,
        "n_boot": ren.n_boot,
        "alpha_corrected": ren.alpha_corrected,
        "dropped_cells": ren.dropped_cells,
        "dropped_unparsed": ren.dropped_unparsed,
    }


def fig5_obfuscation_contrast(df, n_boot=None):
    """One panel, one axis. Returns (Figure, stats, caption).

    No bars: a bar from zero double-encodes a signed difference and reads as a
    magnitude. Dot plus interval says exactly what was estimated and how well.
    """
    from .constants import DELTA_LABEL, NONE_COLOR
    style.apply(style.theme())
    c = style.colors()
    st = obfuscation_stats(df, n_boot=n_boot)
    rows = sorted([r for r in st["rows"]], key=lambda r: -abs(r["diff"]))
    guilty = st["guilty"]
    n_rows = len(rows) + (1 if guilty else 0)
    n_rows = n_rows + len(st.get("by_condition") or [])
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.44 * (n_rows + 4) + 1.4))
    if not rows:
        style.empty_axes(ax, "no paired observations")
        return fig, st, "No paired observations."

    none_r = st.get("none")
    by_cond = st.get("by_condition") or []
    # Blocks are laid out bottom-up: guilty, then the disaggregated refusal rows,
    # then the blame composition, then the promoted overall refusal row on top.
    y = 0.0
    y_guilty = y if guilty else None
    if guilty:
        y += 1.55
    y_cond = []
    for _ in by_cond:
        y_cond.append(y); y += 0.82
    if by_cond:
        y += 0.75
    y_mod = []
    for _ in rows:
        y_mod.append(y); y += 0.82
    y_mod = list(reversed(y_mod))
    y += 0.75
    y_none = y

    def _mark(yi, r, col, emphasis=False):
        filled = r["significant"]
        ax.plot([100 * r["lo"], 100 * r["hi"]], [yi, yi], color=col,
                linewidth=2.0 if emphasis else 1.3, alpha=1.0 if filled else 0.75,
                solid_capstyle="round", zorder=3)
        # Fill encodes significance; the STROKE always carries the row's own colour,
        # so a non-significant row is still identifiable as itself.
        ax.scatter([100 * r["diff"]], [yi], s=70 if emphasis else 46, zorder=4,
                   color=col if filled else c["panel"],
                   edgecolor=col, linewidth=0 if filled else 1.6)
        lab = f"{100 * r['diff']:+.1f}" + ("" if filled else "  n.s.")
        ax.annotate(lab, (1.005, yi), xycoords=("axes fraction", "data"),
                    va="center", ha="left",
                    fontsize=style.ANNOT_PT + (1 if emphasis else 0),
                    color=c["fg"], weight="bold" if emphasis else "normal")

    if none_r:
        _mark(y_none, none_r, c["bar"], emphasis=True)
        ax.axhline(y_none - 0.42, color=c["muted"], linewidth=0.6)
    for yi, r in zip(y_mod, rows):
        _mark(yi, r, MODALITY_COLORS.get(r["category"], NONE_COLOR))
    if by_cond:
        ax.axhline(max(y_cond) + 0.86, color=c["muted"], linewidth=0.6)
        for yi, r in zip(reversed(y_cond), by_cond):
            _mark(yi, r, MODALITY_COLORS.get(r["modality"], NONE_COLOR))
    if guilty:
        ax.axhline(y_guilty + 0.6, color=c["muted"], linewidth=0.6)
        _mark(y_guilty, guilty, MODALITY_COLORS["C"])

    bounds = [abs(100 * v) for r in rows + by_cond + ([guilty] if guilty else [])
                                       + ([none_r] if none_r else [])
              for v in (r["lo"], r["hi"]) if np.isfinite(v)]
    lim = max(2.0, np.ceil(max(bounds) / 2.0) * 2.0) if bounds else 2.0
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color=c["fg"], linewidth=1.4, zorder=2)
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)

    from .sensitivity import row_label
    labels, ypos = [], []
    if none_r:
        labels.append("DECLINES TO BLAME\n(all items)")
        ypos.append(y_none)
    for yi, r in zip(y_mod, rows):
        labels.append(MODALITY_LABELS.get(r["category"], r["category"]))
        ypos.append(yi)
    for yi, r in zip(reversed(y_cond), by_cond):
        labels.append(row_label(r["condition"]))
        ypos.append(yi)
    if guilty:
        labels.append("names the broken code\n(A-C items only)")
        ypos.append(y_guilty)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    # Levels as text under each label -- this replaces the whole left dumbbell panel,
    # where three of five shifts were under 1.2pp on a 0-40 axis and unreadable.
    sub_rows = ([(y_none, none_r, "of all items")] if none_r else []) \
        + [(yi, r, "of blame") for yi, r in zip(y_mod, rows)] \
        + [(yi, r, "not blamed") for yi, r in zip(reversed(y_cond), by_cond)] \
        + ([(y_guilty, guilty, "of A-C items")] if guilty else [])
    for yi, r, unit in sub_rows:
        ax.annotate(f"{100 * r['real']:.1f}% \u2192 {100 * r['obf']:.1f}%  {unit}",
                    (-0.005, yi - 0.32), xycoords=("axes fraction", "data"),
                    # Spec asked for 10px here; that is ~7.5pt and would break the
                    # 8pt paper floor the module enforces, so it sits at the floor.
                    ha="right", va="center", fontsize=style.ANNOT_PT,
                    color=c["muted"])
    ax.set_ylim(min(ypos) - 0.8, max(ypos) + 0.7)
    # Covers all three quantities on the axis: a blame share, a refusal rate, and a
    # recall. They are all percentage-point changes; each row names its own base.
    heads = []
    if rows:
        heads.append(("WHERE blame goes, among items it did blame "
                      "(shares add to 100%)", max(y_mod) + 0.48))
    if by_cond:
        heads.append(("HOW OFTEN it declines to blame anything, per corruption",
                      max(y_cond) + 0.62))
    for htext, hy in heads:
        ax.annotate(htext, (0.012, hy), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=style.ANNOT_PT,
                    color=c["muted"], style="italic")
    ax.set_xlabel("change under obfuscation (percentage points)")
    ax.annotate("Left of zero: goes down under obfuscation.  "
                "Hollow marks: interval includes zero.",
                (0, -0.26), xycoords="axes fraction", fontsize=style.ANNOT_PT,
                color=c["muted"], va="top")
    return fig, st, _obf_caption(st)


def _obf_caption(st):
    sig = [r for r in st["rows"] if r["significant"]]
    parts = [
        f"Blame composition among the four representations, renormalised over "
        f"DETECTED items only (pred_agree == 'no') so the four rows sum to 100% "
        f"within each naming condition; 'none' is reported separately above because "
        f"it is a refusal to blame, not a representation.",
        f"Slice: rows where the code view is NOT the corrupted one. Paired "
        f"within-solver over n={st['n_solvers']} solvers; {st['n_boot']:,} bootstrap "
        f"resamples of SOLVERS. Delta is {st['delta_is']}. Significance at a "
        f"Bonferroni-corrected alpha of {st['alpha_corrected']:.3f}.",
    ]
    bc = st.get("by_condition") or []
    if bc:
        same_sign = all(r["diff"] > 0 for r in bc) or all(r["diff"] < 0 for r in bc)
        n_sig_bc = sum(1 for r in bc if r["significant"])
        # Built outside the f-string: a nested implicit concatenation inside a
        # replacement field is a syntax error before Python 3.12.
        direction = ("same way" if same_sign else "in both directions")
        gloss = (" \u2014 a response-policy effect rather than something about a "
                 "particular corruption" if same_sign else "")
        parts.append(
            f"The lower block disaggregates the slice: the change in how often NO "
            f"view is blamed, per corruption, with trajectory split into its four "
            f"methods. All {len(bc)} shifts point the {direction}{gloss}; "
            f"{n_sig_bc} survives correction.")
    if not sig:
        parts.append("No modality's share change survives correction.")
    else:
        big = max(sig, key=lambda r: abs(r["diff"]))
        parts.append(f"Largest surviving shift: "
                     f"{MODALITY_LABELS.get(big['category'], big['category'])} "
                     f"{100 * big['diff']:+.1f}pp.")
    if st["dropped_unparsed"]:
        parts.append(f"{st['dropped_unparsed']} rows carried no parseable verdict and "
                     f"are excluded.")
    return " ".join(parts)


def none_statement(st):
    """The promoted 'none' line, generated. Returns (text, ok)."""
    r = st.get("none")
    if not r:
        return ("The refusal-to-blame rate could not be computed.", False)
    return (f"Obfuscation raises \u201call four representations agree\u201d from "
            f"{100 * r['real']:.1f}% to {100 * r['obf']:.1f}% "
            f"({100 * r['diff']:+.1f}pp, {100 * r['lo']:+.1f} to "
            f"{100 * r['hi']:+.1f} CI"
            + ("" if r["significant"] else ", not significant") + ").", True)


CAPTIONED_FIGURES["fig5_obfuscation_contrast"] = (
    lambda d: fig5_obfuscation_contrast(d)[::2])


# ── fig 7: prior weakening ───────────────────────────────────────────────────
def fig7_prior_weakening(df, n_boot=None):
    """One question, one denominator, five rows. Returns (Figure, stats, caption).

    Rebuilt from a version that had grown to eight rows across FOUR different
    denominators, two panels and three separate significance encodings. The mixed
    denominators were the core problem -- a single percent axis cannot mean four
    things at once. Every row here is the same quantity: of the items in this
    category where something really was broken, how often the model named the view
    that was actually broken. So the rows are comparable and the axis means one thing.

    The decomposition -- did it commit, was it right given it committed, how it does
    on clean items -- explains this number but does not answer the question, so it
    lives in the details block.
    """
    from . import prior_weakening as PW
    from .constants import MODALITY_LABELS as ML
    style.apply(style.theme())
    c = style.colors()
    r = PW.analyse(df, **({"n_boot": n_boot} if n_boot else {}))
    if r.overall is None or not np.isfinite(r.overall.diff):
        fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.0))
        style.empty_axes(ax, "no paired observations")
        return fig, r, "No paired observations."

    rows = [r.overall] + list(r.per_outlier)
    # Laid out wide and short rather than tall: the figure carries five one-line rows,
    # so the height it needs is set by the row pitch, not by the column width. The
    # earlier version reserved a full row-height of empty axes below the last row for
    # an in-axes legend and put the pooled row 1.55 pitches above the rest, which
    # together cost about a third of the panel to whitespace. The legend now sits on
    # one 8pt line above the axes and the head row sits 0.9 of a pitch clear of the
    # separator -- still visibly set apart, without the gap reading as a missing row.
    ypos = [float(len(rows) - 1) + 0.9] + [float(len(rows) - 1 - i)
                                           for i in range(len(rows) - 1)]

    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.30 * len(rows) + 1.0))
    for yi, cst in zip(ypos, rows):
        head = cst is r.overall
        col = c["bar"] if head else MODALITY_COLORS.get(cst.name, c["muted"])
        ax.plot([100 * cst.real, 100 * cst.obf], [yi, yi], color=col,
                linewidth=1.5, alpha=0.9, zorder=2)
        ax.scatter([100 * cst.real], [yi], s=76 if head else 56, color=col, zorder=4)
        ax.scatter([100 * cst.obf], [yi], s=76 if head else 56,
                   facecolor=c["panel"], edgecolor=col, linewidth=1.8, zorder=4)
        ax.annotate(f"{100 * cst.diff:+.1f} pp", (1.02, yi),
                    xycoords=("axes fraction", "data"), va="center", ha="left",
                    fontsize=style.ANNOT_PT + (1 if head else 0), color=c["fg"],
                    weight="bold" if head else "normal")

    ax.axhline(len(rows) - 1 + 0.45, color=c["muted"], linewidth=0.6)
    ax.set_yticks(ypos)
    # One line per label, not two. At this pitch a wrapped label would collide with
    # its neighbour, and the n belongs beside the category rather than under it.
    ax.set_yticklabels(
        [f"ALL broken items (n={r.overall.n_items:,})"]
        + [f"{ML.get(x.name, x.name)} was broken (n={x.n_items:,})"
           for x in rows[1:]])
    lo = min(100 * min(x.real, x.obf) for x in rows)
    hi = max(100 * max(x.real, x.obf) for x in rows)
    ax.set_xlim(max(0.0, np.floor((lo - 8) / 10) * 10),
                min(100.0, np.ceil((hi + 8) / 10) * 10))
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    ax.set_xlabel("% of these items where the model named the right view")
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=c["muted"],
                   markeredgecolor=c["muted"], markersize=8, label="real names"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=c["panel"],
                   markeredgecolor=c["muted"], markeredgewidth=1.8, markersize=8,
                   label="obfuscated names"),
    ]
    # Above the axes, not inside it: an in-axes legend needs an empty corner, and on
    # a panel this short there is no corner the dumbbells do not reach.
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, frameon=False, fontsize=style.TICK_PT,
              borderaxespad=0.0, handletextpad=0.4, columnspacing=1.2)
    return fig, r, _prior_caption(r)


def fig7b_prior_weakening_split(df, n_boot=None):
    """fig7, with the trajectory row opened up into its four corruption methods.

    A companion to fig7_prior_weakening, not a replacement. The pooled figure answers
    the question the section is under and carries the one contrast this design powers;
    this one asks whether trajectory's four corruptions respond to obfuscation alike,
    which the pooled row averages over. Every trajectory rung here is exploratory --
    32 solvers split four ways -- and the head row is the same tested contrast as in
    the pooled figure, reprinted so the rungs have something to be read against.

    Row order matches fig7 with trajectory expanded IN PLACE (code, the four rungs,
    description, math), so the two figures can be read against each other row by row.
    """
    from . import prior_weakening as PW
    from .constants import MODALITY_LABELS as ML
    from .sensitivity import TRAJ_SHORT
    style.apply(style.theme())
    c = style.colors()
    kw = {"n_boot": n_boot} if n_boot else {}
    r = PW.analyse(df, **kw)
    rungs = PW.per_trajectory_rung(df, **kw)
    if r.overall is None or not np.isfinite(r.overall.diff):
        fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.0))
        style.empty_axes(ax, "no paired observations")
        return fig, r, "No paired observations."
    if not rungs:
        fig, ax = plt.subplots(figsize=style.figsize(1.0, 2.0))
        style.empty_axes(ax, "no trajectory rungs in this data")
        return fig, r, "No trajectory rungs present."

    by_name = {x.name: x for x in r.per_outlier}
    # (contrast, label, colour-key). Trajectory's own pooled row is dropped: the four
    # rungs it averages are right below it, and printing both invites reading the
    # pooled row as a fifth, independent measurement.
    body = []
    for m in ("C", "T", "D", "M"):
        if m == "T":
            for x in rungs:
                body.append((x, f"trajectory \u2014 {TRAJ_SHORT[x.name]}", "T"))
        elif m in by_name:
            body.append((by_name[m], f"{ML.get(m, m)} was broken", m))

    rows = [(r.overall, f"ALL broken items", None)] + body
    ypos = [float(len(rows) - 1) + 0.9] + [float(len(rows) - 1 - i)
                                           for i in range(len(rows) - 1)]

    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.30 * len(rows) + 1.0))
    for yi, (cst, _, key) in zip(ypos, rows):
        head = key is None
        col = c["bar"] if head else MODALITY_COLORS.get(key, c["muted"])
        ax.plot([100 * cst.real, 100 * cst.obf], [yi, yi], color=col,
                linewidth=1.5, alpha=0.9, zorder=2)
        ax.scatter([100 * cst.real], [yi], s=76 if head else 56, color=col, zorder=4)
        ax.scatter([100 * cst.obf], [yi], s=76 if head else 56,
                   facecolor=c["panel"], edgecolor=col, linewidth=1.8, zorder=4)
        ax.annotate(f"{100 * cst.diff:+.1f} pp", (1.02, yi),
                    xycoords=("axes fraction", "data"), va="center", ha="left",
                    fontsize=style.ANNOT_PT + (1 if head else 0), color=c["fg"],
                    weight="bold" if head else "normal")

    ax.axhline(len(rows) - 1 + 0.45, color=c["muted"], linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lab} (n={cst.n_items:,})" for cst, lab, _ in rows])
    lo = min(100 * min(x[0].real, x[0].obf) for x in rows)
    hi = max(100 * max(x[0].real, x[0].obf) for x in rows)
    ax.set_xlim(max(0.0, np.floor((lo - 8) / 10) * 10),
                min(100.0, np.ceil((hi + 8) / 10) * 10))
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    ax.set_xlabel("% of these items where the model named the right view")
    ax.grid(True, axis="x", linewidth=0.4, color=c["faint"])
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=c["muted"],
                   markeredgecolor=c["muted"], markersize=8, label="real names"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=c["panel"],
                   markeredgecolor=c["muted"], markeredgewidth=1.8, markersize=8,
                   label="obfuscated names"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, frameon=False, fontsize=style.TICK_PT,
              borderaxespad=0.0, handletextpad=0.4, columnspacing=1.2)
    return fig, r, _split_caption(r, rungs)


def _split_caption(r, rungs):
    p = r.overall
    sig = [x for x in rungs if np.isfinite(x.lo) and not (x.lo <= 0 <= x.hi)]
    mdes = [x.mde for x in rungs if np.isfinite(x.mde)]
    return (
        f"The same measurement as the pooled figure, with trajectory opened up into "
        f"the four ways it was corrupted. Filled dot = real variable names, hollow = "
        f"obfuscated; every row is of the items where THAT view was the broken one, "
        f"how often the model named it. The head row is the one tested contrast "
        f"({100 * p.diff:+.1f}pp, {100 * p.lo:+.1f} to {100 * p.hi:+.1f} CI, "
        f"n={p.n_solvers} solvers) and is identical to the pooled figure's. "
        f"Trajectory's own pooled row is not drawn: it is the average of the four "
        f"rungs below it, and showing both would read as five measurements where "
        f"there are four. "
        + (f"Every row below the rule is EXPLORATORY. Splitting trajectory four ways "
           f"divides the same {p.n_solvers} solvers across four denominators, so the "
           f"smallest effect detectable at 80% power on a rung runs "
           f"{100 * min(mdes):.1f}pp to {100 * max(mdes):.1f}pp"
           if mdes else "Every row below the rule is EXPLORATORY")
        + (f"; the {len(sig)} rung interval(s) excluding zero are suggestive only."
           if sig else "; no rung interval excludes zero."))


def evidence_statement(r):
    """Leads with the question as asked, then the decomposition that explains it."""
    p = r.overall
    if p is None or not np.isfinite(p.diff):
        return ("The primary contrast could not be computed.", False)
    head = (f"Of every item where something really was wrong, the model named the "
            f"right view {100 * p.real:.1f}% of the time with real variable names "
            f"and {100 * p.obf:.1f}% with obfuscated ones \u2014 "
            f"{100 * p.diff:+.1f}pp ({100 * p.lo:+.1f} to {100 * p.hi:+.1f} CI, "
            f"n={p.n_solvers} solvers).")
    cond, det = r.primary, r.detection
    if p.significant and p.diff > 0:
        tail = ("Removing the lexical shortcut made it MORE often right about which "
                "view was broken: the prior was getting in the way.")
    elif p.significant and p.diff < 0:
        tail = ("Removing the lexical shortcut made it LESS often right, so the "
                "names were carrying information it was using.")
        # Say WHERE the loss came from, because the two components differ.
        if det is not None and det.significant and not cond.significant:
            tail += (f" The loss is not in its judgement but in its willingness to "
                     f"judge: accuracy given that it committed barely moves "
                     f"({100 * cond.diff:+.1f}pp, interval includes zero), while how "
                     f"often it commits at all falls {100 * det.diff:+.1f}pp "
                     f"({100 * det.lo:+.1f} to {100 * det.hi:+.1f}).")
    else:
        tail = (f"That interval includes zero. With n={p.n_solvers} solvers this test "
                f"could only detect {100 * p.mde:.1f}pp or larger, so it does not "
                f"distinguish a grounded identification from a lexical one.")
    return (head + " " + tail, p.significant)


def _prior_caption(r):
    p = r.overall
    if not r.per_outlier:
        return ""
    sig = [x for x in r.per_outlier if not (x.lo <= 0 <= x.hi)]
    tail = (f"the {len(sig)} whose interval excludes zero are suggestive, not "
            f"established." if sig
            else "none of them can settle anything on its own.")
    return (
        f"Every row is the same quantity: of the items in that category where "
        f"something really was broken, how often the model named the view that was "
        f"actually broken. Filled dot = real variable names, hollow = obfuscated. "
        f"Paired within solver across naming; n={p.n_solvers} solvers; "
        f"{r.n_boot:,} bootstrap resamples of solvers. "
        f"The pooled top row is the tested one: {100 * p.diff:+.1f}pp "
        f"({100 * p.lo:+.1f} to {100 * p.hi:+.1f} CI). The four representation rows "
        f"are exploratory -- at this sample size the smallest effect detectable at "
        f"80% power ranges from "
        f"{100 * min(x.mde for x in r.per_outlier):.1f}pp to "
        f"{100 * max(x.mde for x in r.per_outlier):.1f}pp, so {tail} "
        f"Rates are means over solvers, each weighted equally, as the paired design "
        f"requires. Whether the model committed at all, whether it was right given "
        f"that it committed, and how it behaves on clean items are in the details "
        f"block: they explain this number but do not answer the question.")


def interaction_statement(r):
    """The pre-specified asymmetry, stated whether or not it resolves."""
    it = r.interaction
    if it is None or not np.isfinite(it.diff):
        return ("The asymmetry contrast could not be computed.", False)
    resolved = it.significant
    body = (f"Asymmetry test: obfuscation changes correctness by "
            f"{100 * it.real:+.1f}pp where the code is innocent and "
            f"{100 * it.obf:+.1f}pp where the code is the culprit — a difference "
            f"of {100 * it.diff:+.1f}pp ({100 * it.lo:+.1f} to {100 * it.hi:+.1f} CI"
            + ("" if resolved else ", which includes zero") + ").")
    if not resolved:
        body += (f" At n={it.n_solvers} solvers this contrast could only detect an "
                 f"effect of {100 * it.mde:.1f}pp or larger, so it does not settle "
                 f"the question either way.")
    return body, resolved
