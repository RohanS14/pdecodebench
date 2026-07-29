"""Export four paper-quality PNG figures from raw Coder-7B pilot data.

1a: pde_class accuracy vs layer (both pools + BoW + chance)
1b: pde_class mod-type accuracy at best layer (both pools + BoW)
1c: phys_valid AUROC vs layer (both pools + chance)
1d: phys_valid mod-type accuracy at best AUROC layer (both pools + BoW)

Also saves a single 2×2 panel figure (fig_panel.png).
Layout: A (top-left), B (top-right), C (bottom-left), D (bottom-right).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import sqrt
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
MOD_TYPES = [
    "Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]
MT_LABELS = [
    "Clean+\nComment", "Clean,\nNo Comment", "Corrupt\nComment", "Corrupt\nVariable",
    "Invalid+\nComment", "Invalid,\nNo Comment", "CorrComment\n+Invalid", "CorrVar\n+Invalid",
]
POOL_COLORS = {"mean_pool": "#1f77b4", "last_tok": "#d62728"}
POOL_LABELS = {"mean_pool": "mean_pool", "last_tok": "last_tok"}
BOW_COLOR    = "#ff7f0e"
CHANCE_ACC   = 0.25
CHANCE_AUROC = 0.5

RESULTS = Path("probe/results")
OUT_DIR = Path("probe/results/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def wilson_ci(k, n):
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def probe_rows(df):
    sub = df[df["layer"] != "bow"].copy()
    sub["layer"] = sub["layer"].astype(int)
    return sub.sort_values("layer")

def bow_row(df):
    r = df[df["layer"] == "bow"]
    return r.iloc[0] if not r.empty else None

# ── Load data ─────────────────────────────────────────────────────────────────
raw = {}
for pool in ("mean_pool", "last_tok"):
    df = pd.read_csv(RESULTS / f"probe_pooled_{pool}.csv")
    raw[pool] = df

pde_dfs   = {p: raw[p][raw[p]["label"] == "pde_class"].copy()  for p in ("mean_pool", "last_tok")}
auroc_dfs = {p: raw[p][raw[p]["label"] == "phys_valid"].copy() for p in ("mean_pool", "last_tok")}


# ── Panel builders ────────────────────────────────────────────────────────────

def plot_acc_vs_layer(ax, dfs, chance, ylabel="LOGO-CV Accuracy"):
    """pde_class accuracy vs layer. Chance shown as labelled legend entry."""
    for pool in ("mean_pool", "last_tok"):
        df = probe_rows(dfs[pool])
        layers = df["layer"].values
        accs   = df["accuracy"].values
        ci_lo  = df["ci_low"].values
        ci_hi  = df["ci_high"].values
        color  = POOL_COLORS[pool]
        ax.plot(layers, accs, color=color, lw=1.8, label=POOL_LABELS[pool])
        ax.fill_between(layers, ci_lo, ci_hi, color=color, alpha=0.15)

    br = bow_row(dfs["mean_pool"])
    if br is not None:
        bv = float(br["accuracy"])
        blo = float(br.get("ci_low", np.nan))
        bhi = float(br.get("ci_high", np.nan))
        if not (np.isnan(blo) or np.isnan(bhi)):
            ax.axhspan(blo, bhi, color=BOW_COLOR, alpha=0.12, label="_nolegend_")
        ax.axhline(bv, color=BOW_COLOR, ls="--", lw=1.5, label="BoW")

    ax.axhline(chance, color="gray", ls=":", lw=1.0, label="chance")
    ax.set_xlabel("Layer (0 = embedding)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=9)


def plot_auroc_vs_layer(ax, dfs, chance):
    """phys_valid AUROC vs layer. Chance shown as labelled legend entry."""
    for pool in ("mean_pool", "last_tok"):
        df = probe_rows(dfs[pool]).dropna(subset=["auroc"])
        layers = df["layer"].values
        aurocs = df["auroc"].values
        ci_lo  = df["auroc_ci_low"].values if "auroc_ci_low" in df.columns else aurocs
        ci_hi  = df["auroc_ci_high"].values if "auroc_ci_high" in df.columns else aurocs
        color  = POOL_COLORS[pool]
        ax.plot(layers, aurocs, color=color, lw=1.8, label=POOL_LABELS[pool])
        ax.fill_between(layers, ci_lo, ci_hi, color=color, alpha=0.15)

    ax.axhline(chance, color="gray", ls=":", lw=1.0, label="chance")
    br = bow_row(dfs["mean_pool"])
    if br is not None:
        bav = float(br.get("auroc", np.nan))
        if not np.isnan(bav):
            balo = float(br.get("auroc_ci_low", np.nan))
            bahi = float(br.get("auroc_ci_high", np.nan))
            if not (np.isnan(balo) or np.isnan(bahi)):
                ax.axhspan(balo, bahi, color=BOW_COLOR, alpha=0.12, label="_nolegend_")
            ax.axhline(bav, color=BOW_COLOR, ls="--", lw=1.5, label="BoW")
    ax.set_xlabel("Layer (0 = embedding)", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.tick_params(labelsize=9)


def plot_modtype_bars(ax, dfs, chance=0.25, best_by="accuracy"):
    """Grouped bar chart of per-mod-type accuracy at best layer. No chance label."""
    n_mt  = len(MOD_TYPES)
    grp_w = 0.75
    bar_w = grp_w / 3
    x     = np.arange(n_mt)

    for pi, pool in enumerate(("mean_pool", "last_tok")):
        df = probe_rows(dfs[pool])
        if best_by == "auroc":
            best_row = df.dropna(subset=["auroc"]).pipe(lambda d: d.loc[d["auroc"].idxmax()])
        else:
            best_row = df.loc[df["accuracy"].idxmax()]

        vals = [float(best_row.get(f"mt_{mt}", np.nan)) for mt in MOD_TYPES]
        err_lo, err_hi = [], []
        for v in vals:
            if not np.isnan(v):
                k = round(v * 16)
                lo, hi = wilson_ci(k, 16)
                err_lo.append(v - lo); err_hi.append(hi - v)
            else:
                err_lo.append(0); err_hi.append(0)

        offset = (pi - 1) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=POOL_COLORS[pool], label=POOL_LABELS[pool],
               yerr=[err_lo, err_hi], capsize=2, error_kw=dict(lw=1.0))

    br = bow_row(dfs["mean_pool"])
    if br is not None:
        bow_vals = [float(br.get(f"mt_{mt}", np.nan)) for mt in MOD_TYPES]
        bow_err_lo, bow_err_hi = [], []
        for v in bow_vals:
            if not np.isnan(v):
                k = round(v * 16)
                lo, hi = wilson_ci(k, 16)
                bow_err_lo.append(v - lo); bow_err_hi.append(hi - v)
            else:
                bow_err_lo.append(0); bow_err_hi.append(0)
        ax.bar(x + bar_w, bow_vals, width=bar_w * 0.9, color=BOW_COLOR, label="BoW",
               yerr=[bow_err_lo, bow_err_hi], capsize=2, error_kw=dict(lw=1.0))

    # Chance: grey dotted line only, no text label
    ax.axhline(chance, color="gray", ls=":", lw=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(MT_LABELS, fontsize=7, rotation=35, ha="right")
    ax.set_ylabel("Accuracy at best layer", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=9)
    ax.margins(x=0.03)


# ── Individual figures ────────────────────────────────────────────────────────
INDIVIDUAL_SIZE = (3.5, 3.0)

for name, fn, dfs, kw in [
    ("1a", plot_acc_vs_layer,   pde_dfs,   dict(chance=CHANCE_ACC)),
    ("1b", plot_modtype_bars,   pde_dfs,   dict(chance=CHANCE_ACC, best_by="accuracy")),
    ("1c", plot_auroc_vs_layer, auroc_dfs, dict(chance=CHANCE_AUROC)),
    ("1d", plot_modtype_bars,   auroc_dfs, dict(chance=CHANCE_AUROC, best_by="auroc")),
]:
    fig, ax = plt.subplots(figsize=INDIVIDUAL_SIZE)
    fn(ax, dfs, **kw)
    ax.legend(fontsize=8, framealpha=0.85, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.70)
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2×2 panel figure ─────────────────────────────────────────────────────────
# A=top-left, B=top-right, C=bottom-left, D=bottom-right
PANEL_SIZE = (14.0, 5.5)

fig, axes = plt.subplots(2, 2, figsize=PANEL_SIZE,
                         gridspec_kw=dict(hspace=0.55, wspace=0.38))
ax_a, ax_b = axes[0, 0], axes[0, 1]
ax_c, ax_d = axes[1, 0], axes[1, 1]

plot_acc_vs_layer(ax_a,   pde_dfs,   chance=CHANCE_ACC)
plot_modtype_bars(ax_b,   pde_dfs,   chance=CHANCE_ACC,   best_by="accuracy")
plot_auroc_vs_layer(ax_c, auroc_dfs, chance=CHANCE_AUROC)
plot_modtype_bars(ax_d,   auroc_dfs, chance=CHANCE_AUROC, best_by="auroc")

# Row titles
for ax, lbl in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")]:
    ax.text(-0.14, 1.06, lbl, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

ax_a.set_title("PDE Class", fontsize=11, fontweight="bold", pad=6)
ax_b.set_title("PDE Class", fontsize=11, fontweight="bold", pad=6)
ax_c.set_title("Physical Validity", fontsize=11, fontweight="bold", pad=6)
ax_d.set_title("Physical Validity", fontsize=11, fontweight="bold", pad=6)

# ── Shared legend ─────────────────────────────────────────────────────────────
# Collect entries from line-graph panels (a and c have "chance"); bar panels have no chance entry.
# Build manually so we control order: mean_pool, last_tok, BoW (from a), chance (from a).
handles_a, labels_a = ax_a.get_legend_handles_labels()
legend_entries = list(zip(handles_a, labels_a))  # mean_pool, last_tok, BoW (dashed), chance

# Place legend on the right side, vertically centered
fig.legend(
    [h for h, _ in legend_entries],
    [l for _, l in legend_entries],
    loc="center right",
    ncol=1,
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(1.0, 0.5),
    borderaxespad=0.3,
)

fig.subplots_adjust(right=0.88)

out_panel = OUT_DIR / "fig_panel.png"
fig.savefig(out_panel, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_panel}")
