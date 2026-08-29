"""
Belief-revision figure: horizontal layout, shared condition labels, one axis row.

Replaces the 2x2 grid of vertical stacked bars. All four models and all four
transition categories are retained; the space saving comes from writing the
condition labels once and sharing a single x-axis row.

Data contract
-------------
A tidy DataFrame with one row per (model, condition) cell and these columns:

    model      str   display name, plotted left-to-right in MODEL_ORDER
    validity   str   'valid' or 'invalid'
    cue        str   one of CUE_ORDER
    RR         int   right in stage 1, right in stage 2
    WR         int   wrong  -> right   (evidence corrected it)
    RW         int   right  -> wrong   (evidence broke it)
    WW         int   wrong in both

Counts, not proportions. Rows are normalised to 100% internally so unequal n
per cell does not silently distort the bars.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# --- style -----------------------------------------------------------------
#
# Prefer importing the rcParams your other figures already use, so every figure
# in the paper stays typographically consistent:
#
#     plt.style.use("path/to/your_paper.mplstyle")
#
# or, if the settings live in a module:
#
#     from your_pkg.plotstyle import apply_style; apply_style()
#
# The block below is a stand-in that matches the serif setting used in the
# cross-consistency figures. Delete it once the shared style is wired in.

RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,        # embed as TrueType, not Type 3 — required by
    "ps.fonttype": 42,         # most camera-ready checkers
}
plt.rcParams.update(RC)

MODEL_ORDER = ["2.5-flash-tb0", "2.5-flash-tb8192", "3.7-flash-low", "3.7-flash-high"]

# The reasoning factor is currently buried in the model suffixes, and the two
# families use different notation (token budgets vs. named levels). Split it out
# into a two-tier column header: family on top, thinking budget beneath.
# Budgets are NOT comparable across families — say so in the caption.
MODEL_META = {
    "2.5-flash-tb0":    ("2.5-flash", "no thinking"),
    "2.5-flash-tb8192": ("2.5-flash", "thinking (8192)"),
    "3.7-flash-low":    ("3.7-flash", "low thinking"),
    "3.7-flash-high":   ("3.7-flash", "high thinking"),
}
CUE_ORDER = ["+ comment", "no comment", "corrupt comment", "obfuscated var"]

# Semantic order matters: correct outcomes first so the green/blue boundary
# (final accuracy) sits at a readable position, wrong outcomes trailing.
CATS = ["RR", "WR", "RW", "WW"]
COLORS = {
    "RR": "#2ca02c",   # kept correct
    "WR": "#1f77b4",   # revised into correct
    "RW": "#d62728",   # revised out of correct
    "WW": "#9e9e9e",   # stayed wrong
}
LABELS = {
    "RR": "stayed right (right → right)",
    "WR": "fixed (wrong → right)",
    "RW": "broke (right → wrong)",
    "WW": "stayed wrong (wrong → wrong)",
}

BAR_H = 0.80
BLOCK_GAP = 0.6          # blank rows between the valid and invalid blocks


def _y_positions():
    """Row centres, valid block on top, invalid below, gap between."""
    valid = np.arange(len(CUE_ORDER), dtype=float)
    invalid = valid + len(CUE_ORDER) + BLOCK_GAP
    return valid, invalid


def plot(df, outfile="figures/agent_belief_revision.png",
         annotate_accuracy=False, seg_label_min=12.0):
    """
    annotate_accuracy : print RR+WR in a column at the right of each panel.
    seg_label_min     : in-bar labels for segments at least this wide (%).
                        Set to None to suppress in-bar labels entirely.
    """
    missing = set(MODEL_ORDER) - set(df["model"].unique())
    if missing:
        raise ValueError(f"missing models in df: {sorted(missing)}")

    valid_y, invalid_y = _y_positions()
    ypos = {"valid": valid_y, "invalid": invalid_y}

    fig, axes = plt.subplots(
        1, len(MODEL_ORDER),
        figsize=(12.4, 2.9),
        sharey=True,
        gridspec_kw={"wspace": 0.30},
    )

    for ax, model in zip(axes, MODEL_ORDER):
        sub = df[df["model"] == model]
        for validity in ("valid", "invalid"):
            for i, cue in enumerate(CUE_ORDER):
                row = sub[(sub["validity"] == validity) & (sub["cue"] == cue)]
                if row.empty:
                    continue
                counts = row.iloc[0][CATS].to_numpy(dtype=float)
                total = counts.sum()
                if total == 0:
                    continue
                pct = 100.0 * counts / total          # normalise per cell
                left = 0.0
                y = ypos[validity][i]
                for cat, w in zip(CATS, pct):
                    if w <= 0:
                        continue
                    ax.barh(y, w, left=left, height=BAR_H,
                            color=COLORS[cat], edgecolor="none", zorder=2)
                    if seg_label_min is not None and w >= seg_label_min:
                        ax.text(left + w / 2, y, f"{w:.0f}",
                                va="center", ha="center", fontsize=7.5,
                                color="white", zorder=3)
                    left += w
                if annotate_accuracy:
                    acc = pct[CATS.index("RR")] + pct[CATS.index("WR")]
                    ax.text(104, y, f"{acc:.0f}%", va="center", ha="left",
                            fontsize=8.5, color="#222222",
                            fontweight="bold" if acc < 60 else "normal",
                            clip_on=False, zorder=4)

        ax.set_title(MODEL_META[model][1], fontsize=9.5, pad=3)
        if annotate_accuracy:
            ax.text(104, -0.95, "final", fontsize=8, style="italic",
                    color="#555555", ha="left", va="center",
                    clip_on=False, zorder=4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100%"], fontsize=8)
        ax.margins(x=0)
        ax.xaxis.grid(True, color="#e0e0e0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(axis="y", length=0)

    # shared row labels, written once on the leftmost axis
    ax0 = axes[0]
    ax0.set_yticks(np.concatenate([valid_y, invalid_y]))
    ax0.set_yticklabels(CUE_ORDER * 2, fontsize=9)
    ax0.invert_yaxis()

    # block headers sit outside the axes, aligned to each block
    for validity, label in (("valid", "VALID CODE"), ("invalid", "INVALID CODE")):
        y = ypos[validity].mean()
        ax0.text(-0.78, y, label, transform=ax0.get_yaxis_transform(),
                 rotation=90, va="center", ha="center",
                 fontsize=9, fontweight="bold", color="#222222")

    # family headers spanning each pair of panels, with a bracket rule
    fig.canvas.draw()
    families = {}
    for ax, model in zip(axes, MODEL_ORDER):
        families.setdefault(MODEL_META[model][0], []).append(ax)
    for family, group in families.items():
        x0 = min(a.get_position().x0 for a in group)
        x1 = max(a.get_position().x1 for a in group)
        y = max(a.get_position().y1 for a in group)
        # Offsets in FIGURE fractions have to be read against this figure's
        # height (2.9in): the original 0.185/0.205 put the rule 0.54in above axes
        # whose titles end 0.19in above them, so a third of the band was empty.
        # Derived from the title's own extent instead, so the header stays put if
        # the figure is ever made taller.
        top = max(a.title.get_window_extent(fig.canvas.get_renderer()).y1
                  for a in group) / fig.bbox.height
        fig.add_artist(plt.Line2D([x0, x1], [top + 0.030, top + 0.030],
                                  color="#999999", lw=0.8,
                                  transform=fig.transFigure))
        fig.text((x0 + x1) / 2, top + 0.045, family, ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color="#111111")

    handles = [mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in CATS]
    # Both were hung off the bottom of the canvas in figure fractions -- -0.06 and
    # -0.24 on a 2.9in figure is 0.17in and 0.70in below the axes, against tick
    # labels only 0.11in tall. Anchored to the axes in POINTS instead, which is what
    # the spacing actually is, and does not have to be retuned when the figure is
    # resized.
    # Centred on the FIGURE (it labels all four panels) but positioned off the axes
    # bottom in points, so it clears the tick labels by a fixed distance whatever
    # height the figure is given.
    y0 = min(a.get_position().y0 for a in axes)
    fig.text(0.5, y0 - 26 / (fig.get_figheight() * 72), "share of episodes",
             ha="center", va="top", fontsize=9, color="#444444")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.045),
               bbox_transform=fig.transFigure)

    # PNG only. Nothing in this repo reads the PDFs, and a directory carrying two
    # files per figure is two to keep straight and twice as much to retire.
    import os
    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"[agentic] wrote {outfile}")
    return fig


# --- synthetic data so the module runs standalone ---------------------------

def _synthetic(seed=0):
    """Accuracy totals follow the source figure; internal splits are invented."""
    acc = {
        "2.5-flash-tb0":     [50, 59, 25, 44, 97, 97, 94, 91],
        "2.5-flash-tb8192":  [56, 62, 56, 66, 94, 91, 97, 94],
        "3.7-flash-low":     [91, 94, 91, 94, 97, 94, 97, 97],
        "3.7-flash-high":    [94, 94, 94, 94, 97, 97, 97, 97],
    }
    rng = np.random.default_rng(seed)
    rows, n = [], 32
    for model, vals in acc.items():
        for j, a in enumerate(vals):
            validity = "valid" if j < 4 else "invalid"
            cue = CUE_ORDER[j % 4]
            right = int(round(n * a / 100))
            wr = int(rng.integers(0, max(1, right // 3) + 1))
            rr = right - wr
            wrong = n - right
            rw = int(rng.integers(0, max(1, wrong // 2) + 1)) if wrong else 0
            ww = wrong - rw
            rows.append(dict(model=model, validity=validity, cue=cue,
                             RR=rr, WR=wr, RW=rw, WW=ww))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = _synthetic()
    assert (df[CATS].sum(axis=1) == 32).all(), "cells must sum to n"
    plot(df)