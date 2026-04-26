"""
Representational Similarity Analysis (RSA) for PDE hidden states.

Usage:
    python probe/rsa_analysis.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --layers 0,7,14,21,28 \
        --output_dir probe/figures/ \
        --pool mean_pool

Produces:
    rsa_layer{l}_{pool}.png       — 96×96 cosine-distance heatmap per layer
    rsa_block_score_{pool}.png    — within/between class distance ratio vs layer depth
    rsa_modtype_compare_l{l}.png  — side-by-side 16×16 heatmaps for 3 mod_types
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from probe_utils import load_data, PDE_CLASSES, MOD_TYPES

PDE_COLORS = {
    "wave": "#4C72B0",
    "heat": "#DD8452",
    "burgers": "#55A868",
    "navier-stokes": "#C44E52",
}


def compute_rdm(reps: np.ndarray) -> np.ndarray:
    """Cosine distance matrix for (N, D) reps."""
    reps_f = reps.astype(np.float32)
    norms = np.linalg.norm(reps_f, axis=1, keepdims=True) + 1e-8
    reps_norm = reps_f / norms
    return 1.0 - reps_norm @ reps_norm.T


def block_rdm_score(rdm: np.ndarray, pde_labels: np.ndarray) -> float:
    """
    Within/between class distance ratio.
    Lower = better clustering (within distances are small relative to between).
    Returns within_mean / between_mean.
    """
    N = len(rdm)
    within, between = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if pde_labels[i] == pde_labels[j]:
                within.append(rdm[i, j])
            else:
                between.append(rdm[i, j])
    if not between:
        return float("nan")
    return float(np.mean(within) / (np.mean(between) + 1e-8))


def plot_rdm(rdm: np.ndarray, order: np.ndarray, pde_labels: np.ndarray,
             mod_type_labels: np.ndarray, title: str, out_path: str):
    rdm_sorted = rdm[np.ix_(order, order)]
    pde_sorted = pde_labels[order]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rdm_sorted, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine distance")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Example index (sorted by PDE class)")
    ax.set_ylabel("Example index (sorted by PDE class)")

    # Draw class boundary lines
    boundaries = []
    prev = pde_sorted[0]
    for i, p in enumerate(pde_sorted):
        if p != prev:
            boundaries.append(i)
            prev = p
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=1.5)
        ax.axvline(b - 0.5, color="white", linewidth=1.5)

    # Legend
    seen = {}
    for pde, color in PDE_COLORS.items():
        seen[pde] = mpatches.Patch(color=color, label=pde)
    # Annotate color bands on axes using class labels
    # (tick marks at class centers)
    class_positions = {}
    for i, p in enumerate(pde_sorted):
        class_positions.setdefault(p, []).append(i)
    tick_pos = [int(np.mean(positions)) for p, positions in class_positions.items()]
    tick_labels = list(class_positions.keys())
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--layers", default="0,7,14,21,28",
                        help="Comma-separated layer indices to plot")
    parser.add_argument("--output_dir", default="probe/figures/")
    parser.add_argument("--pool", default="mean_pool",
                        choices=["mean_pool", "last_tok"],
                        help="Which pooling to use")
    parser.add_argument("--compare_layer", type=int, default=None,
                        help="Layer index for mod_type comparison plot (default: middle layer)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.hidden)
    reps_all = data[args.pool].astype(np.float32)  # (N, L, D)
    pde_labels = data["pde_classes"]
    mod_type_labels = data["mod_types"]
    gt_samples = data["gt_samples"]
    titles = data["titles"]

    N, L, D = reps_all.shape
    print(f"Loaded: {N} examples, {L} layers, {D} dim  [{args.pool}]", flush=True)

    # Sort order: by pde_class, then gt_sample, then mod_type
    pde_order_map = {c: i for i, c in enumerate(PDE_CLASSES)}
    mt_order_map = {mt: i for i, mt in enumerate(MOD_TYPES)}
    sort_key = [
        (pde_order_map.get(pde_labels[i].lower(), 99),
         gt_samples[i],
         mt_order_map.get(mod_type_labels[i], 99))
        for i in range(N)
    ]
    order = np.array(sorted(range(N), key=lambda i: sort_key[i]))

    layer_ids = [int(x) for x in args.layers.split(",")]
    layer_ids = [l for l in layer_ids if 0 <= l < L]

    # --- Per-layer RDM heatmaps ---
    block_scores = []
    for l_idx in range(L):
        reps_l = reps_all[:, l_idx, :]
        rdm = compute_rdm(reps_l)
        score = block_rdm_score(rdm, pde_labels)
        block_scores.append(score)

        if l_idx in layer_ids:
            print(f"  Layer {l_idx}: block_score={score:.4f}", flush=True)
            out = os.path.join(args.output_dir,
                               f"rsa_layer{l_idx:02d}_{args.pool}.png")
            plot_rdm(rdm, order, pde_labels, mod_type_labels,
                     title=f"RSA — Layer {l_idx} ({args.pool})", out_path=out)
            print(f"    Saved: {out}", flush=True)

    # --- Block score vs layer ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(L), block_scores, marker="o", markersize=4, linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ratio=1 (no clustering)")
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("Within/between distance ratio\n(lower = better PDE clustering)")
    ax.set_title(f"RSA block score vs layer depth [{args.pool}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Mark selected layers
    for l_idx in layer_ids:
        ax.axvline(l_idx, color="red", linestyle=":", alpha=0.5)
    plt.tight_layout()
    score_path = os.path.join(args.output_dir, f"rsa_block_score_{args.pool}.png")
    plt.savefig(score_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Block score plot: {score_path}", flush=True)

    # --- Mod-type comparison at one layer ---
    compare_l = args.compare_layer
    if compare_l is None:
        # pick layer with lowest block score (best clustering)
        compare_l = int(np.argmin(block_scores))
        print(f"Auto-selected layer {compare_l} for mod-type comparison "
              f"(best block score: {block_scores[compare_l]:.4f})", flush=True)

    compare_mod_types = ["Comm_Valid", "CorrComm", "NoComm_CorrVar"]
    fig, axes = plt.subplots(1, len(compare_mod_types), figsize=(5 * len(compare_mod_types), 5))
    fig.suptitle(f"RSA mod-type comparison — Layer {compare_l} [{args.pool}]", fontsize=12)

    for ax, mt in zip(axes, compare_mod_types):
        mask = np.array([m == mt for m in mod_type_labels])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            ax.set_title(f"{mt}\n(no data)")
            continue

        # Sort within this mod_type by pde_class
        pde_sub = pde_labels[idx]
        sub_order = sorted(range(len(idx)), key=lambda i: pde_order_map.get(pde_sub[i].lower(), 99))
        idx_sorted = idx[sub_order]

        reps_sub = reps_all[idx_sorted, compare_l, :]
        rdm_sub = compute_rdm(reps_sub)
        pde_sub_sorted = pde_labels[idx_sorted]

        im = ax.imshow(rdm_sub, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="Cosine dist")

        class_pos = {}
        for i, p in enumerate(pde_sub_sorted):
            class_pos.setdefault(p, []).append(i)
        for p, positions in class_pos.items():
            center = int(np.mean(positions))
            ax.text(center, -1.5, p, ha="center", fontsize=7, rotation=30)

        # class boundaries
        prev = pde_sub_sorted[0]
        for i, p in enumerate(pde_sub_sorted):
            if p != prev:
                ax.axhline(i - 0.5, color="white", linewidth=1)
                ax.axvline(i - 0.5, color="white", linewidth=1)
                prev = p

        score_sub = block_rdm_score(rdm_sub, pde_sub_sorted)
        ax.set_title(f"{mt}\nblock={score_sub:.3f} (N={len(idx)})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    mt_path = os.path.join(args.output_dir,
                           f"rsa_modtype_compare_l{compare_l:02d}_{args.pool}.png")
    plt.savefig(mt_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mod-type comparison: {mt_path}", flush=True)

    print("\nBlock scores by layer:", flush=True)
    for l_idx, s in enumerate(block_scores):
        marker = " <-- selected" if l_idx in layer_ids else ""
        print(f"  Layer {l_idx:2d}: {s:.4f}{marker}", flush=True)


if __name__ == "__main__":
    main()
