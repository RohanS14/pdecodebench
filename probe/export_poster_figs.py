"""Export two poster-quality static PNG figures, matching the HTML viz exactly.

Fig 1: Pooled probe — accuracy vs layer (pde_class)
Fig 2: Pooled probe — mod-type breakdown (pde_class, best layer)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path

# ── Constants matching viz_interactive.py exactly ────────────────────────────
MOD_TYPES = [
    "Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]
POOL_COLORS = {"mean_pool": "#1f77b4", "last_tok": "#d62728"}
BOW_COLOR   = "#ff7f0e"
CHANCE      = 0.25  # pde_class is 4-class

RESULTS = Path("probe/results")
OUT_DIR = Path("probe/results/poster")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def wilson_ci(k, n):
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

# ── Load correct CSVs (probe_pooled_*, not probe_hyperparam_*) ────────────────
dfs = {}
for pool in ("mean_pool", "last_tok"):
    df = pd.read_csv(RESULTS / f"probe_pooled_{pool}.csv")
    dfs[pool] = df[df["label"] == "pde_class"].copy()

# ── Fig 1: Accuracy vs Layer ──────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7.5, 3.4))

bow_plotted = False
for pool in ("mean_pool", "last_tok"):
    df = dfs[pool]

    # Probe rows only (exclude BoW)
    probe = df[df["layer"] != "bow"].copy()
    probe["layer"] = probe["layer"].astype(int)
    probe = probe.sort_values("layer")

    layers = probe["layer"].values
    accs   = probe["accuracy"].values
    ci_lo  = probe["ci_low"].values
    ci_hi  = probe["ci_high"].values
    color  = POOL_COLORS[pool]
    label  = "mean_pool" if pool == "mean_pool" else "last_tok"

    ax1.plot(layers, accs, color=color, lw=1.8, label=label)
    ax1.fill_between(layers, ci_lo, ci_hi, color=color, alpha=0.15)

    # BoW horizontal line — mean_pool only (both pools have same value)
    if pool == "mean_pool":
        bow_row = df[df["layer"] == "bow"]
        if not bow_row.empty:
            bv = float(bow_row["accuracy"].iloc[0])
            ax1.axhline(bv, color=BOW_COLOR, ls="--", lw=1.5,
                        label=f"BoW ({bv:.2f})")

# Chance line
ax1.axhline(CHANCE, color="gray", ls=":", lw=1.0, label=f"Chance ({CHANCE:.2f})")

ax1.set_xlabel("Layer (0 = embedding)", fontsize=11)
ax1.set_ylabel("LOGO-CV Accuracy", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.tick_params(labelsize=9)
ax1.legend(fontsize=9, framealpha=0.85, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig1.tight_layout()
fig1.subplots_adjust(right=0.78)
out1 = OUT_DIR / "fig1_accuracy_vs_layer.png"
fig1.savefig(out1, dpi=300)
plt.close(fig1)
print(f"Saved: {out1}")

# ── Fig 2: Mod-type breakdown at best layer ───────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(609/96, 263/96))

n_mt   = len(MOD_TYPES)
n_grps = 3   # mean_pool, last_tok, BoW
grp_w  = 0.75
bar_w  = grp_w / n_grps
x      = np.arange(n_mt)

for pi, pool in enumerate(("mean_pool", "last_tok")):
    df = dfs[pool]
    probe = df[df["layer"] != "bow"].copy()
    probe["layer"] = probe["layer"].astype(int)
    best_row = probe.loc[probe["accuracy"].idxmax()]
    best_layer = int(best_row["layer"])

    vals   = [float(best_row.get(f"mt_{mt}", np.nan)) for mt in MOD_TYPES]
    err_lo, err_hi = [], []
    for v in vals:
        if not np.isnan(v):
            k = round(v * 16)
            lo, hi = wilson_ci(k, 16)
            err_lo.append(v - lo)
            err_hi.append(hi - v)
        else:
            err_lo.append(0); err_hi.append(0)

    offset = (pi - 1) * bar_w  # -bar_w, 0 for mean_pool/last_tok; BoW gets +bar_w
    label  = pool
    ax2.bar(
        x + offset, vals, width=bar_w * 0.9,
        color=POOL_COLORS[pool], label=label,
        yerr=[err_lo, err_hi], capsize=2, error_kw=dict(lw=1.0),
    )

# BoW bars — from the bow row (same for both pools, use mean_pool)
bow_row = dfs["mean_pool"][dfs["mean_pool"]["layer"] == "bow"]
if not bow_row.empty:
    bow_vals = [float(bow_row.iloc[0].get(f"mt_{mt}", np.nan)) for mt in MOD_TYPES]
    offset = 1 * bar_w
    ax2.bar(
        x + offset, bow_vals, width=bar_w * 0.9,
        color=BOW_COLOR, label="BoW",
    )

ax2.axhline(CHANCE, color="gray", ls=":", lw=1.0, label=f"Chance ({CHANCE:.2f})")

ax2.set_xticks(x)
MT_LABELS = [
    "Clean+Comment",
    "Clean,\nNo Comment",
    "Corrupt\nComment",
    "Corrupt\nVariable",
    "Invalid+Comment",
    "Invalid,\nNo Comment",
    "CorrComment\n+Invalid",
    "CorrVar\n+Invalid",
]
ax2.set_xticklabels(MT_LABELS, fontsize=7, rotation=30, ha="right")
ax2.set_ylabel("Accuracy at best layer", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis="y", labelsize=9)
ax2.legend(fontsize=9, framealpha=0.85, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig2.tight_layout()
fig2.subplots_adjust(right=0.72)
out2 = OUT_DIR / "fig2_modtype_breakdown.png"
fig2.savefig(out2, dpi=300)
plt.close(fig2)
print(f"Saved: {out2}")
