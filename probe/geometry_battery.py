"""
Experiment 2, Part II — representation geometry (plan §16).

Four observations that bear on "does the model represent physics", each chosen
because it can come out negative in an informative way:

  16.1  intrinsic dimensionality   is the physics manifold lower-dimensional than
                                   the code manifold, and where does that separate?
  16.2  linear vs kernel probe     is pde_class linearly encoded, or present but
                                   curved? A large gap changes how every linear
                                   result in this experiment should be read.
  16.3  validity direction         fit phys_valid on 3 classes, test on the 4th.
                                   Cross-class transfer is a strong claim about a
                                   general representation of physical correctness.
  16.4  anisotropy baseline        the Δh analysis escapes anisotropy by
                                   differencing. Nothing here does, so every
                                   number is reported raw AND mean-centred.

Usage (from repo root):
    python probe/geometry_battery.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/ --pool mean_pool
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

LEAK_SOLVERS = {"Burgers_6", "Burgers_7", "Burgers_8",
                "NavierStokes_5", "NavierStokes_7"}


def participation_ratio(X: np.ndarray) -> float:
    """(Σλ)² / Σλ² — effective number of dimensions carrying the variance."""
    Xc = X - X.mean(axis=0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = lam.sum()
    return float(s * s / (lam ** 2).sum()) if s > 0 else 0.0


def twonn_id(X: np.ndarray) -> float:
    """
    Two-NN intrinsic dimensionality (Facco et al.). Robust to curvature in a way
    PCA-based estimates are not, which is the point: a curved manifold has low ID
    but high PCA rank, and 16.2 predicts exactly that situation.
    """
    Xc = X - X.mean(axis=0)
    n = len(Xc)
    if n < 4:
        return float("nan")
    d = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    srt = np.sort(d, axis=1)
    r1, r2 = srt[:, 0], srt[:, 1]
    ok = (r1 > 0) & np.isfinite(r2)
    if ok.sum() < 4:
        return float("nan")
    mu = np.sort(r2[ok] / r1[ok])

    # Degeneracy guard. On a regular lattice the two nearest neighbours are
    # equidistant (one either side), so mu == 1 for every interior point, log(mu)
    # collapses to numerical noise and the slope below is meaningless — it returns
    # ~8 for a perfectly 1-dimensional line. TwoNN assumes points drawn from a
    # density, not a grid. Return NaN rather than a plausible-looking number: a
    # bogus dimensionality printed in the report is worse than a missing one.
    if float(np.median(mu)) < 1 + 1e-9:
        return float("nan")

    F = np.arange(1, len(mu) + 1) / len(mu)
    keep = F < 1.0
    x, y = np.log(mu[keep]), -np.log(1 - F[keep])
    if len(x) < 2 or np.sum(x * x) <= 0:
        return float("nan")
    return float(np.sum(x * y) / np.sum(x * x))       # slope through the origin


def logo_acc(X, y, groups, clf_factory) -> float:
    """Leave-one-group-out accuracy. Groups are solvers, never rows."""
    accs = []
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = clf_factory().fit(sc.transform(X[tr]), y[tr])
        accs.append(float(np.mean(clf.predict(sc.transform(X[te])) == y[te])))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--output_dir", default="probe/results/")
    ap.add_argument("--pool", default="mean_pool", choices=["mean_pool", "last_tok"])
    ap.add_argument("--layers", default=None)
    ap.add_argument("--condition", default="Comm_Valid")
    args = ap.parse_args()

    d = np.load(args.hidden, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    H = data[args.pool]
    N, L, D = H.shape
    model_name = str(data["model_name"]) if "model_name" in data else "unknown"
    mt = data["mod_types"].astype(str)
    gt = data["gt_samples"].astype(str)
    pde = data["pde_classes"].astype(str)
    valid = data["phys_valid"].astype(int)

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(L)))
    rows = []

    def emit(layer, stat, value, **kw):
        r = {"model": model_name, "pool": args.pool, "layer": layer,
             "rel_depth": layer / (L - 1) if L > 1 else 0.0,
             "stat": stat, "value": float(value)}
        r.update(kw)
        rows.append(r)

    # --- 16.1 / 16.4 dimensionality, raw and mean-centred --------------------
    # "physics-varying" = one row per solver at a fixed condition (physics differs,
    # surface fixed). "code-varying" = one solver across all 8 conditions (code
    # surface differs, physics fixed) pooled over solvers.
    phys_sel = np.where(mt == args.condition)[0]
    print(f"physics-varying set: {len(phys_sel)} rows ({args.condition})", flush=True)

    for layer in layers:
        Xp = H[phys_sel, layer].astype(np.float64)
        emit(layer, "participation_ratio", participation_ratio(Xp), manifold="physics")
        emit(layer, "twonn_id", twonn_id(Xp), manifold="physics")

        # code-varying: within-solver spread across the 8 conditions
        prs, ids_ = [], []
        for s in np.unique(gt):
            idx = np.where(gt == s)[0]
            if len(idx) >= 4:
                Xs = H[idx, layer].astype(np.float64)
                prs.append(participation_ratio(Xs))
                ids_.append(twonn_id(Xs))
        if prs:
            emit(layer, "participation_ratio", float(np.mean(prs)), manifold="code")
            emit(layer, "twonn_id", float(np.nanmean(ids_)), manifold="code")

        # anisotropy: mean cosine between raw representations. High = the space is
        # dominated by a common direction, which inflates every raw similarity.
        Xn = Xp / np.linalg.norm(Xp, axis=1, keepdims=True).clip(1e-12)
        iu = np.triu_indices(len(Xn), k=1)
        emit(layer, "mean_pairwise_cos_raw", float((Xn @ Xn.T)[iu].mean()))
        Xc = Xp - Xp.mean(axis=0)
        Xcn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True).clip(1e-12)
        emit(layer, "mean_pairwise_cos_centered", float((Xcn @ Xcn.T)[iu].mean()))

    # --- 16.2 linear vs kernel probe for pde_class ---------------------------
    yp = pde[phys_sel]
    gp = gt[phys_sel]
    for layer in layers:
        X = H[phys_sel, layer].astype(np.float64)
        lin = logo_acc(X, yp, gp, lambda: LogisticRegression(max_iter=500,
                                                             random_state=42))
        knn = logo_acc(X, yp, gp, lambda: KNeighborsClassifier(
            n_neighbors=min(5, max(1, len(X) // 6))))
        emit(layer, "pde_class_linear_acc", lin)
        emit(layer, "pde_class_knn_acc", knn)
        emit(layer, "pde_class_curvature_gap", knn - lin)

    # --- 16.3 validity direction across classes ------------------------------
    # Leak solvers are excluded here: their invalid variant may be detectable from
    # a cadence change, which would let them carry a "validity" result that is not
    # about physical correctness at all.
    clean = np.array([s not in LEAK_SOLVERS for s in gt])
    for layer in layers:
        for held in np.unique(pde):
            tr = np.where((pde != held) & clean)[0]
            te = np.where((pde == held) & clean)[0]
            if len(np.unique(valid[tr])) < 2 or len(te) == 0:
                continue
            sc = StandardScaler().fit(H[tr, layer].astype(np.float64))
            clf = LogisticRegression(max_iter=500, random_state=42).fit(
                sc.transform(H[tr, layer].astype(np.float64)), valid[tr])
            acc = float(np.mean(clf.predict(
                sc.transform(H[te, layer].astype(np.float64))) == valid[te]))
            emit(layer, "validity_cross_class_acc", acc, held_out_class=held,
                 n_test=len(te))

        if layer % 5 == 0 or layer == layers[-1]:
            sel = [r for r in rows if r["layer"] == layer
                   and r["stat"] == "validity_cross_class_acc"]
            m = np.mean([r["value"] for r in sel]) if sel else float("nan")
            print(f"  layer {layer:3d}/{L-1}  validity cross-class acc {m:.3f}",
                  flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    slug = model_name.replace("/", "_")
    out = os.path.join(args.output_dir, f"geometry_{slug}_{args.pool}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)", flush=True)

    pr = df[(df.stat == "participation_ratio")]
    for man in pr.manifold.dropna().unique():
        s = pr[pr.manifold == man]["value"]
        print(f"  participation ratio ({man:7s}): median {s.median():.2f}", flush=True)
    gap = df[df.stat == "pde_class_curvature_gap"]["value"]
    print(f"  kNN − linear (pde_class): max {gap.max():+.3f} at layer "
          f"{df.loc[gap.idxmax(), 'layer'] if len(gap) else 'n/a'}", flush=True)
    if gap.max() > 0.15:
        print("  NOTE: pde_class is substantially more decodable non-linearly. "
              "Linear results elsewhere in this experiment understate the "
              "structure that is present.", flush=True)
    va = df[df.stat == "validity_cross_class_acc"].groupby("layer")["value"].mean()
    if len(va):
        print(f"  validity cross-class transfer: peak {va.max():.3f} at layer "
              f"{va.idxmax()} (chance 0.500)", flush=True)
    an = df[df.stat == "mean_pairwise_cos_raw"]["value"]
    print(f"  anisotropy (raw mean pairwise cos): median {an.median():.3f} "
          f"— raw similarities are inflated by this much", flush=True)


if __name__ == "__main__":
    sys.exit(main())
