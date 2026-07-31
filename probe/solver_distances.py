"""
Distances between solvers under each perturbation — the plain-language view of
Experiment 2.

Three questions, each with one number you can read off a chart:

  1. HOW FAR does each perturbation move a solver's representation?
     Relative displacement ‖h(perturbed) − h(baseline)‖ / ‖h(baseline)‖, per solver.

  2. Which moves it MORE — changing the physics, or changing the description?
     Physics edits are measured within a fixed surface (valid → invalid). Surface
     edits are measured on fixed physics (Comm_Valid → each other valid arm).

  3. Does the ARRANGEMENT of solvers survive the perturbation?
     The 32×32 solver-by-solver distance matrix under each condition, and its
     correlation with the unperturbed matrix. If the model represents the physics,
     solvers should keep their relative positions when only the description moves.

Writes a JSON blob consumed by viz_solver_distances.py.

Usage:
    python probe/solver_distances.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --out probe/results/solver_distances.json
"""
import argparse
import json
import os

import numpy as np

BASELINE = "Comm_Valid"

# Perturbations that change ONLY the description; the implemented physics is fixed.
SURFACE = {
    "NoComm_Valid": "comments removed",
    "CorrComm": "comments made misleading",
    "NoComm_CorrVar": "comments removed + identifiers obfuscated",
}
# Perturbations that change the PHYSICS, each measured against its own surface twin
# so the description is held fixed.
PHYSICS = {
    "Comm_InValid": ("Comm_Valid", "physics broken (comments kept)"),
    "NoComm_InValid": ("NoComm_Valid", "physics broken (no comments)"),
    "CorrComm_Invalid": ("CorrComm", "physics broken (misleading comments)"),
    "NoComm_CorrVar_InValid": ("NoComm_CorrVar", "physics broken (obfuscated)"),
}


def reldist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise ‖a−b‖ / ‖a‖ — displacement as a fraction of the baseline's size."""
    return np.linalg.norm(a - b, axis=-1) / np.clip(np.linalg.norm(a, axis=-1), 1e-12, None)


def dist_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine distance between solvers."""
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - Xn @ Xn.T


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--out", default="probe/results/solver_distances.json")
    ap.add_argument("--rsa_layer", type=int, default=None,
                    help="layer used as the READING POINT for the matrices and the "
                         "summary table; every curve is computed across all layers. "
                         "Default = 40%% of depth.")
    args = ap.parse_args()

    d = np.load(args.hidden, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    gt = data["gt_samples"].astype(str)
    mt = data["mod_types"].astype(str)
    pde = data["pde_classes"].astype(str)
    model = str(data["model_name"]) if "model_name" in data else "unknown"
    L = data["mean_pool"].shape[1]
    rsa_layer = args.rsa_layer if args.rsa_layer is not None else int(round(0.4 * (L - 1)))

    idx = {(g, m): i for i, (g, m) in enumerate(zip(gt, mt))}
    solvers = sorted(set(gt))
    # order solvers by PDE class so block structure is visible in the matrices
    order = sorted(solvers, key=lambda s: (pde[idx[(s, BASELINE)]], s))
    cls = [str(pde[idx[(s, BASELINE)]]) for s in order]

    out = {"model": model, "n_layers": int(L), "rsa_layer": rsa_layer,
           "solvers": order, "pde_class": cls, "pools": {}}

    for pool in ("mean_pool", "last_tok"):
        H = data[pool]
        pert = {}

        for cond, label in SURFACE.items():
            per_layer = []
            for l in range(L):
                a = np.stack([H[idx[(s, BASELINE)], l] for s in order]).astype(np.float64)
                b = np.stack([H[idx[(s, cond)], l] for s in order]).astype(np.float64)
                per_layer.append(reldist(a, b))
            per_layer = np.array(per_layer)            # (L, S)
            pert[cond] = {"kind": "surface", "label": label,
                          "vs": BASELINE,
                          "mean_by_layer": per_layer.mean(axis=1).round(5).tolist(),
                          "per_solver_at_rsa": per_layer[rsa_layer].round(5).tolist()}

        for cond, (base, label) in PHYSICS.items():
            per_layer = []
            for l in range(L):
                a = np.stack([H[idx[(s, base)], l] for s in order]).astype(np.float64)
                b = np.stack([H[idx[(s, cond)], l] for s in order]).astype(np.float64)
                per_layer.append(reldist(a, b))
            per_layer = np.array(per_layer)
            pert[cond] = {"kind": "physics", "label": label, "vs": base,
                          "mean_by_layer": per_layer.mean(axis=1).round(5).tolist(),
                          "per_solver_at_rsa": per_layer[rsa_layer].round(5).tolist()}

        # --- solver-by-solver geometry, at EVERY layer.
        # Reporting this at a single hand-picked layer would be circular — the
        # layer that maximises an effect is not evidence for it. The curve is
        # computed across all layers and the chosen layer is only a reading point.
        preserved_by_layer = {c: [] for c in list(SURFACE) + list(PHYSICS)}
        for l in range(L):
            bX = np.stack([H[idx[(s, BASELINE)], l] for s in order]).astype(np.float64)
            bD = upper(dist_matrix(bX))
            for cond in preserved_by_layer:
                X = np.stack([H[idx[(s, cond)], l] for s in order]).astype(np.float64)
                preserved_by_layer[cond].append(round(spearman(bD, upper(dist_matrix(X))), 4))

        base_X = np.stack([H[idx[(s, BASELINE)], rsa_layer] for s in order]).astype(np.float64)
        base_D = dist_matrix(base_X)
        mats = {BASELINE: base_D.round(4).tolist()}
        preserved = {}
        for cond in list(SURFACE) + list(PHYSICS):
            X = np.stack([H[idx[(s, cond)], rsa_layer] for s in order]).astype(np.float64)
            mats[cond] = dist_matrix(X).round(4).tolist()
            preserved[cond] = preserved_by_layer[cond][rsa_layer]

        # how much of the geometry is PDE class? within-class vs between-class distance
        cls_arr = np.array(cls)
        same = cls_arr[:, None] == cls_arr[None, :]
        iu = np.triu_indices(len(order), k=1)
        wi = float(base_D[iu][same[iu]].mean())
        bw = float(base_D[iu][~same[iu]].mean())

        out["pools"][pool] = {
            "perturbations": pert,
            "matrices": mats,
            "geometry_preserved": preserved,
            "geometry_preserved_by_layer": preserved_by_layer,
            "within_class_dist": round(wi, 4),
            "between_class_dist": round(bw, 4),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(f"Saved: {args.out}", flush=True)
    print(f"Model {model} | {len(order)} solvers | RSA layer {rsa_layer}/{L-1}", flush=True)
    for pool in ("mean_pool", "last_tok"):
        p = out["pools"][pool]
        s = np.mean([v["mean_by_layer"][rsa_layer] for v in p["perturbations"].values()
                     if v["kind"] == "surface"])
        ph = np.mean([v["mean_by_layer"][rsa_layer] for v in p["perturbations"].values()
                      if v["kind"] == "physics"])
        print(f"\n{pool} @ layer {rsa_layer}:", flush=True)
        print(f"  mean displacement — description change {s:.4f} | physics change {ph:.4f} "
              f"| ratio physics/description {ph/s:.2f}", flush=True)
        print(f"  solver geometry within-class {p['within_class_dist']:.4f} vs "
              f"between-class {p['between_class_dist']:.4f}", flush=True)
        print(f"  geometry preserved (Spearman vs unperturbed):", flush=True)
        for k, v in sorted(p["geometry_preserved"].items(), key=lambda x: -x[1]):
            print(f"      {k:24s} {v:+.3f}", flush=True)


if __name__ == "__main__":
    main()
