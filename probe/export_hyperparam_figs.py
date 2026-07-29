"""Export C-sensitivity figure for pde_class — mean_pool, raw + pca20, all C values.

Matches the HTML hyperparam sweep page exactly:
  - Color = C value
  - Solid line = raw (D=3584), dashed = PCA-20
  - CI bands per curve
  - Chance dotted gray line

Output: probe/results/paper/hyperparam_pde_class_c_sensitivity.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

RESULTS = Path("probe/results")
OUT_DIR = Path("probe/results/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_COLORS  = {0.01: "#1f77b4", 0.1: "#2ca02c", 1.0: "#d62728", 10.0: "#9467bd"}
REPR_DASH = {"raw": "-", "pca20": "--"}
REPR_LABEL = {"raw": "raw", "pca20": "PCA-20"}
CHANCE = 0.25
LABEL = "pde_class"

c_handles = [Line2D([0], [0], color=C_COLORS[C], lw=2, label=f"C={C:g}") for C in sorted(C_COLORS)]
repr_handles = [
    Line2D([0], [0], color="black", ls="-",  lw=2, label="raw"),
    Line2D([0], [0], color="black", ls="--", lw=2, label="PCA-20"),
    Line2D([0], [0], color="gray",  ls=":",  lw=1, label="chance"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

for ax, pool in zip(axes, ("mean_pool", "last_tok")):
    repr_paths = {
        "raw":   RESULTS / f"probe_hyperparam_raw_{pool}.csv",
        "pca20": RESULTS / f"probe_hyperparam_pca20_{pool}.csv",
    }
    for repr_, path in repr_paths.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        sub = df[df["label"] == LABEL].copy()
        sub["layer"] = sub["layer"].astype(int)
        for C in sorted(C_COLORS):
            c_sub = sub[sub["C"] == C].sort_values("layer")
            if c_sub.empty:
                continue
            layers = c_sub["layer"].values
            accs   = c_sub["accuracy"].values
            ci_lo  = c_sub["ci_low"].values
            ci_hi  = c_sub["ci_high"].values
            ax.plot(layers, accs, color=C_COLORS[C], ls=REPR_DASH[repr_], lw=1.8)
            ax.fill_between(layers, ci_lo, ci_hi, color=C_COLORS[C], alpha=0.10)

    ax.axhline(CHANCE, color="gray", ls=":", lw=1.0)
    ax.set_title(pool, fontsize=11, fontweight="bold")
    ax.set_xlabel("Layer (0 = embedding)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=9)

axes[0].set_ylabel("LOGO-CV Accuracy", fontsize=10)

axes[1].legend(handles=c_handles + repr_handles, fontsize=8.5, framealpha=0.85,
               loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

fig.tight_layout()
fig.subplots_adjust(right=0.86)

out = OUT_DIR / "hyperparam_pde_class_c_sensitivity.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
