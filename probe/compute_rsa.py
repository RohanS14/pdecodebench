"""
Compute RSA metrics and PCA-2 coordinates from a hidden-state NPZ.
Outputs small CSVs suitable for cross-model comparison in the viz.

Usage:
    python probe/compute_rsa.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/rsa/ \
        --slug coder7b

Outputs (in output_dir):
    rsa_block_{slug}.csv      — layer, pool, pde_block_score, valid_block_score
    rsa_drift_{slug}.csv      — mod_type, pool, mean_dist, sem, best_layer
    pca2_{slug}.csv           — title, pde_class, mod_type, phys_valid, gt_sample,
                                 pc1, pc2, pool, layer
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from probe_utils import load_data

POOLS = ["mean_pool", "last_tok"]
MOD_TYPES = [
    "Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]


def compute_rdm(reps: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8
    reps_n = reps / norms
    cos_sim = reps_n @ reps_n.T
    return 1.0 - np.clip(cos_sim, -1, 1)


def block_rdm_score(rdm: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(rdm[i, j])
            else:
                between.append(rdm[i, j])
    if not within or not between:
        return float("nan")
    return float(np.mean(within) / np.mean(between))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--output_dir", default="probe/results/rsa/")
    parser.add_argument("--slug", required=True,
                        help="Model slug for output filenames (e.g. coder7b, coder32b, qwq32b)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.hidden)

    pde_labels = data["pde_classes"]
    valid_labels = np.array(["valid" if v else "invalid" for v in data["phys_valid"]])
    mod_type_labels = data["mod_types"]
    gt_samples = data["gt_samples"]
    titles = data["titles"]
    N, L, D = data["mean_pool"].shape

    print(f"Loaded: N={N}, L={L}, D={D}", flush=True)

    # ── Block scores ──
    block_rows = []
    best_pde_layer = {}
    for pool in POOLS:
        reps_all = data[pool].astype(np.float32)
        pde_scores, valid_scores = [], []
        for l in range(L):
            rdm = compute_rdm(reps_all[:, l, :])
            pde_scores.append(block_rdm_score(rdm, pde_labels))
            valid_scores.append(block_rdm_score(rdm, valid_labels))
            block_rows.append({
                "layer": l,
                "pool": pool,
                "pde_block_score": pde_scores[-1],
                "valid_block_score": valid_scores[-1],
            })
        best_pde_layer[pool] = int(np.nanargmin(pde_scores))
        print(f"  {pool}: pde best={min(pde_scores):.4f} @ layer {best_pde_layer[pool]}", flush=True)

    block_csv = os.path.join(args.output_dir, f"rsa_block_{args.slug}.csv")
    pd.DataFrame(block_rows).to_csv(block_csv, index=False)
    print(f"Saved: {block_csv}", flush=True)

    # ── Mod-type drift from Comm_Valid at best pde_class layer ──
    drift_rows = []
    for pool in POOLS:
        best_l = best_pde_layer[pool]
        reps_best = data[pool].astype(np.float32)[:, best_l, :]
        norms = np.linalg.norm(reps_best, axis=1, keepdims=True) + 1e-8
        reps_n = reps_best / norms

        drifts = {mt: [] for mt in MOD_TYPES if mt != "Comm_Valid"}
        for gt in np.unique(gt_samples):
            gt_mask = gt_samples == gt
            cv_idx = np.where(gt_mask & (mod_type_labels == "Comm_Valid"))[0]
            if len(cv_idx) == 0:
                continue
            cv_rep = reps_n[cv_idx[0]]
            for mt in MOD_TYPES:
                if mt == "Comm_Valid":
                    continue
                mt_idx = np.where(gt_mask & (mod_type_labels == mt))[0]
                if len(mt_idx) == 0:
                    continue
                dist = float(1.0 - float(cv_rep @ reps_n[mt_idx[0]]))
                drifts[mt].append(dist)

        for mt in drifts:
            vals = drifts[mt]
            drift_rows.append({
                "mod_type": mt,
                "pool": pool,
                "mean_dist": float(np.mean(vals)) if vals else float("nan"),
                "sem": float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "best_layer": best_l,
                "n": len(vals),
            })

    drift_csv = os.path.join(args.output_dir, f"rsa_drift_{args.slug}.csv")
    pd.DataFrame(drift_rows).to_csv(drift_csv, index=False)
    print(f"Saved: {drift_csv}", flush=True)

    # ── PCA-2 at best pde_class layer (mean_pool) ──
    pca2_rows = []
    for pool in POOLS:
        best_l = best_pde_layer[pool]
        reps = data[pool].astype(np.float32)[:, best_l, :]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(reps)
        var_explained = pca.explained_variance_ratio_
        print(f"  PCA-2 [{pool}] layer {best_l}: var explained = {var_explained[0]:.3f}, {var_explained[1]:.3f}", flush=True)
        for i in range(N):
            pca2_rows.append({
                "title": titles[i],
                "pde_class": pde_labels[i],
                "mod_type": mod_type_labels[i],
                "phys_valid": bool(data["phys_valid"][i]),
                "gt_sample": gt_samples[i],
                "pc1": float(coords[i, 0]),
                "pc2": float(coords[i, 1]),
                "pool": pool,
                "layer": best_l,
                "var_pc1": float(var_explained[0]),
                "var_pc2": float(var_explained[1]),
            })

    pca2_csv = os.path.join(args.output_dir, f"pca2_{args.slug}.csv")
    pd.DataFrame(pca2_rows).to_csv(pca2_csv, index=False)
    print(f"Saved: {pca2_csv}", flush=True)


if __name__ == "__main__":
    main()
