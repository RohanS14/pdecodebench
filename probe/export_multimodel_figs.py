"""Export multimodel probe v2 figures (mean_pool only, C=1.0, all 3 models).

Outputs:
  probe/results/paper/multimodel_pde_accuracy_vs_layer.png
  probe/results/paper/multimodel_phys_auroc_vs_layer.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("probe/results")
OUT_DIR = Path("probe/results/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLUGS = [
    ("Coder-7B",  RESULTS / "probe_hyperparam_pca20_mean_pool.csv",              28, "#1f77b4"),
    ("Coder-32B", RESULTS / "coder32b/probe_hyperparam_pca20_mean_pool.csv",     64, "#ff7f0e"),
    ("QwQ-32B",   RESULTS / "qwq32b/probe_hyperparam_pca20_mean_pool.csv",       64, "#2ca02c"),
]


def get_sub(path, label, n_layers):
    df = pd.read_csv(path)
    sub = df[(df["label"] == label) & (df["C"] == 1.0)].copy()
    sub = sub[sub["layer"].apply(lambda x: str(x).lstrip("-").isdigit())].copy()
    sub["layer"] = sub["layer"].astype(int)
    sub = sub.sort_values("layer")
    sub["rel"] = sub["layer"] / n_layers
    return sub


def plot_line_fig(label, metric, ci_lo_col, ci_hi_col, chance, ylabel, ylim, outname):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, path, n_layers, color in SLUGS:
        sub = get_sub(path, label, n_layers)
        if metric == "auroc":
            sub = sub.dropna(subset=["auroc"])
        rel  = sub["rel"].values
        vals = sub[metric].values
        ci_lo = sub[ci_lo_col].values if ci_lo_col in sub.columns else vals
        ci_hi = sub[ci_hi_col].values if ci_hi_col in sub.columns else vals
        rgba = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
        ax.plot(rel, vals, color=color, lw=1.8, label=name)
        ax.fill_between(rel, ci_lo, ci_hi, color=color, alpha=0.15)

    ax.axhline(chance, color="gray", ls=":", lw=1.0, label="chance")
    ax.set_xlabel("Relative layer depth (layer / n_layers)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    out = OUT_DIR / outname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_line_fig(
    label="pde_class", metric="accuracy",
    ci_lo_col="ci_low", ci_hi_col="ci_high",
    chance=0.25, ylabel="LOGO-CV Accuracy", ylim=(0, 1.05),
    outname="multimodel_pde_accuracy_vs_layer.png",
)

plot_line_fig(
    label="phys_valid", metric="auroc",
    ci_lo_col="auroc_ci_low", ci_hi_col="auroc_ci_high",
    chance=0.5, ylabel="AUROC", ylim=(0.4, 1.05),
    outname="multimodel_phys_auroc_vs_layer.png",
)
