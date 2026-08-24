"""One shared matplotlib style. Import and call `apply()` before building anything.

Sized for a NeurIPS single-column workshop paper: 5.5in of text width, nothing
below 8pt. Those two numbers are the whole constraint set, so they live here as
named constants rather than as literals sprinkled through four figure modules.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DARK_CMAP = LinearSegmentedColormap.from_list(
    "panel_to_accent", ["#12141e", "#1e2c46", "#2f4d78", "#4a7fb5", "#9ec8f0"])
# The light ground used "Greys", chosen when this theme existed only for greyscale
# print. On screen that renders the blame matrix black and white, throwing away the
# one channel a heatmap has. This ramp runs the SAME hue as the light accent
# (#1a5fb4) from page-white to a deep blue, so the matrix reads as part of the report
# rather than a foreign object, and it stays monotone in lightness -- which is what
# keeps it honest under greyscale printing and for colour-vision deficiency, since
# the ordering survives even when the hue does not.
PAPER_CMAP = LinearSegmentedColormap.from_list(
    "white_to_accent", ["#ffffff", "#dbe7f7", "#9dc0e8", "#5189c4", "#1a5fb4"])
for _cm, _name in ((DARK_CMAP, "panel_to_accent"), (PAPER_CMAP, "white_to_accent")):
    try:
        matplotlib.colormaps.register(_cm, name=_name)
    except (ValueError, AttributeError):
        pass

# The figures are built headless in CI and on the cluster; picking the backend at
# import time avoids a DISPLAY-dependent crash that only shows up off a laptop.
if not os.environ.get("MPL_INTERACTIVE"):
    matplotlib.use("Agg")

TEXT_WIDTH_IN = 5.5
MIN_FONT_PT = 8

# Anything a reader has to decode at 100% zoom sits at 8pt or above. Base text is
# 9 so that the 8pt annotations still read as subordinate to it.
BASE_PT = 9
TICK_PT = 8
ANNOT_PT = 8

RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "font.family": "serif",
    # DejaVu ships with matplotlib, so the figures render identically on a machine
    # without the paper's Times installed instead of silently falling back.
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": BASE_PT,
    "axes.titlesize": BASE_PT,
    "axes.labelsize": BASE_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "figure.titlesize": BASE_PT,

    "axes.linewidth": 0.7,
    "axes.edgecolor": "#4A4A4A",
    "axes.labelcolor": "#1A1A1A",
    "axes.labelpad": 4.0,
    "axes.titlepad": 6.0,
    "axes.titleweight": "regular",
    "text.color": "#1A1A1A",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "xtick.labelcolor": "#1A1A1A",
    "ytick.labelcolor": "#1A1A1A",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.linewidth": 0.4,
    "grid.color": "#DDDDDD",

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "figure.constrained_layout.use": False,

    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.4,
    "legend.labelspacing": 0.45,
    "errorbar.capsize": 2,
    "hatch.linewidth": 0.5,
    # Fixes the clip-path/glyph ids matplotlib would otherwise derive from a random
    # salt per process. See the note in claim_report._svg -- without both this and
    # the dropped date stamp, two identical builds differ in a few hundred bytes.
    "svg.hashsalt": "pde-llm-eval-consistency",
    "pdf.fonttype": 42,      # embed TrueType, not Type 3 -- required by most CFPs
    "ps.fonttype": 42,
    # Keep SVG text as <text>, not glyph outlines. The default ("path") renders
    # every label as a <path>, which makes the figure unsearchable, unselectable and
    # invisible to a screen reader -- and quietly falsified the claim that these
    # inline figures are text-searchable.
    "svg.fonttype": "none",
}


# The paper wants white; the dashboard is dark. Same figures, two grounds -- a white
# raster dropped into a dark report reads as a foreign object pasted in, which is
# exactly how these looked before the theme became switchable.
PALETTE = {
    "light": dict(bg="white", panel="white", fg="#1A1A1A", muted="#4A4A4A",
                  faint="#EEEEEE", rule="#4A4A4A", sep="white",
                  # Hatching is a texture, not data: it says "this band is the
                  # residual", and on white it was drawn in `muted` -- the same
                  # near-black as the percentage printed on top of it, so the
                  # number had to be read through the lines. Lighter fixes that
                  # without dropping the texture. Dark keeps `muted`, which is
                  # already low-contrast against its panel, so the frozen report
                  # renders byte-for-byte as published.
                  hatch="#C6CBD4",
                  bar="#4A4A4A", bar2="#B8B8B8", cmap="white_to_accent",
                  note_fg="#8A6D3B", note_bg="#FDF6E3", note_edge="#E0D6BC"),
    "dark":  dict(bg="#12141e", panel="#12141e", fg="#e0e0e0", muted="#8592ae",
                  faint="#1e2130", rule="#3a4258", sep="#12141e", hatch="#8592ae",
                  bar="#7eb8ff", bar2="#46527a", cmap="panel_to_accent",
                  note_fg="#d9b877", note_bg="#241d10", note_edge="#7a5c2a"),
}
_ACTIVE = {"theme": "light"}


def colors():
    """Palette for the theme currently applied."""
    return PALETTE[_ACTIVE["theme"]]


def theme():
    return _ACTIVE["theme"]


def apply(theme="light"):
    """Install the house style. Idempotent, so figure modules may each call it."""
    _ACTIVE["theme"] = theme if theme in PALETTE else "light"
    c = colors()
    rc = dict(RC)
    rc.update({
        "figure.facecolor": c["bg"], "savefig.facecolor": c["bg"],
        "axes.facecolor": c["panel"], "axes.edgecolor": c["rule"],
        "axes.labelcolor": c["fg"], "text.color": c["fg"],
        "xtick.color": c["muted"], "ytick.color": c["muted"],
        "xtick.labelcolor": c["fg"], "ytick.labelcolor": c["fg"],
        "grid.color": c["faint"],
    })
    plt.rcParams.update(rc)


def figsize(width_frac=1.0, height_in=2.6):
    """Width as a fraction of the 5.5in text column."""
    return (TEXT_WIDTH_IN * width_frac, height_in)


def save(fig, name, outdir="figures"):
    """Write vector PDF and 300dpi PNG side by side. Returns both paths.

    Both formats every time: the PDF goes in the paper, the PNG goes in the
    dashboard and the slide deck, and keeping them in lockstep here means the two
    can never drift to different versions of the same figure.
    """
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}.pdf")
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    fig.savefig(png, dpi=300, facecolor=fig.get_facecolor())
    return pdf, png


def empty_axes(ax, message="no data"):
    """Render an axes that has nothing to show as a labelled blank, not a crash.

    A figure with a missing cell -- reasoning=on absent, a model not yet run -- has
    to keep building, and the gap has to be visible. Dropping the panel silently
    would make a partial run look like a complete one.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=ANNOT_PT, color=colors()["muted"], transform=ax.transAxes)
    return ax


def hatch_kw(edge="muted"):
    """`hatchcolor=` only when it would actually change the ink.

    matplotlib >= 3.11 lets a patch's hatch take a colour of its own, which is what
    keeps a percentage readable when it is printed on top of the hatch. But the kwarg
    is written into the SVG whenever it is passed -- even when it resolves to exactly
    the edge colour -- so passing it unconditionally rewrites every byte of the frozen
    dark report without changing a single pixel of it. Dark's hatch IS its `muted`, so
    on that ground this returns nothing at all and the published file still rebuilds
    byte for byte.
    """
    c = colors()
    return {} if c["hatch"] == c[edge] else {"hatchcolor": c["hatch"]}
