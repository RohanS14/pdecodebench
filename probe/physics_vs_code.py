"""
Experiment 2, Part II — physics vs. code dissociation.

The Δh analysis (world_model_delta.py) varies how a fixed program is *described*.
It cannot separate "represents the physics" from "represents the program", because
its four surface conditions are byte-identical modulo comments and identifier
names. This script attacks that separation directly.

    14.1  variance partitioning   does pde_class explain representational
                                  distance AFTER partialling out code similarity?
    14.2  method invariance       do same-PDE/different-algorithm solvers cluster
                                  above same-algorithm/different-PDE ones?
    14.3  process transfer        exploratory only — see the confound audit, which
                                  this script prints rather than hides.

Usage (from repo root):
    python probe/physics_vs_code.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/ \
        --pool mean_pool \
        --condition Comm_Valid

`--condition` selects which of the 8 mod_types supplies the representation. Run it
for `Comm_Valid` AND `NoComm_CorrVar`: with comments gone and identifiers
obfuscated, pde_class structure that survives is much harder to attribute to
lexical form.

Significance uses a Mantel-style permutation over SOLVER identity, not over the
496 pairs. Pairwise distances are not independent observations and treating them
as such would inflate every p-value in the file.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_similarity import build_similarity_matrices  # noqa: E402

# Measured against merged_mod_jul28.csv — see plan §13. Recorded here so the
# confounds travel with the code that would otherwise overstate the results.
CONFOUND_NOTES = {
    "burgers_all_explicit": "burgers is 8/8 explicit — excluded from 14.2",
    "process_class_confound": (
        "diffusion is 8/8 in heat and NS, 0/8 in wave; advection is 8/8 in burgers, "
        "0/8 in heat and wave; oscillation is 8/8 wave and 0 elsewhere. Process "
        "directions are near-duplicates of class directions. 14.3 is exploratory."
    ),
    "burgers_diffusion_triple_confound": (
        "the 3 burgers solvers labelled diffusion are exactly Burgers_6/7/8 — the "
        "cadence-leak solvers, all synthetic"
    ),
}
METHOD_CLASSES = ["heat", "wave", "navier-stokes"]  # burgers excluded, see above


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return 1.0 - Xn @ Xn.T


def upper(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² of y on X with an intercept. Least-squares, no sklearn dependency."""
    A = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def partial_mantel(D_rep: np.ndarray, D_target: np.ndarray,
                   nuisance: list, n_perm: int, rng) -> dict:
    """
    Incremental R² of `D_target` over `nuisance`, with a Mantel permutation test.

    The permutation shuffles solver identity and rebuilds the target matrix, which
    preserves the dependency structure among pairwise distances. Permuting the
    upper-triangle entries directly would destroy it and produce a null that is far
    too narrow — the classic way RSA p-values get inflated.
    """
    y = upper(D_rep)
    Xn = np.column_stack([upper(m) for m in nuisance]) if nuisance else np.empty((len(y), 0))
    Xf = np.column_stack([Xn, upper(D_target)]) if Xn.size else upper(D_target)[:, None]

    r2_nuis = ols_r2(y, Xn)
    r2_full = ols_r2(y, Xf)
    obs = r2_full - r2_nuis

    S = D_rep.shape[0]
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(S)
        Dt = D_target[np.ix_(p, p)]
        Xp = np.column_stack([Xn, upper(Dt)]) if Xn.size else upper(Dt)[:, None]
        null[k] = ols_r2(y, Xp) - r2_nuis

    return {
        "incremental_r2": obs,
        "r2_nuisance_only": r2_nuis,
        "r2_full": r2_full,
        "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
        "null_hi95": float(np.percentile(null, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--output_dir", default="probe/results/")
    ap.add_argument("--pool", default="mean_pool", choices=["mean_pool", "last_tok"])
    ap.add_argument("--condition", default="Comm_Valid",
                    help="which mod_type supplies the representation")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.hidden, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    H = data[args.pool]
    N, L, D = H.shape
    model_name = str(data["model_name"]) if "model_name" in data else \
        os.path.basename(args.hidden).replace(".npz", "")

    mt = data["mod_types"].astype(str)
    sel = np.where(mt == args.condition)[0]
    if len(sel) == 0:
        raise SystemExit(f"condition {args.condition} not present in NPZ")

    solvers = data["gt_samples"].astype(str)[sel]
    order = np.argsort(solvers)
    sel, solvers = sel[order], solvers[order]
    S = len(sel)

    pde = data["pde_classes"].astype(str)[sel]
    method = data["num_method"].astype(str)[sel]
    process = data["phys_process"].astype(str)[sel]
    src = (data["sources"].astype(str)[sel] if "sources" in data
           else np.array(["unknown"] * S))
    codes = data["codes"].astype(str)[sel]

    print(f"Model     : {model_name}", flush=True)
    print(f"Pool      : {args.pool}   Condition: {args.condition}", flush=True)
    print(f"Solvers   : {S}", flush=True)
    print("Confounds recorded in this run:", flush=True)
    for k, v in CONFOUND_NOTES.items():
        print(f"  - {k}: {v}", flush=True)
    print("", flush=True)

    # --- design matrices ----------------------------------------------------
    print("Building code-similarity regressors...", flush=True)
    sim = build_similarity_matrices(list(codes))

    same_class = (pde[:, None] == pde[None, :]).astype(float)
    D_class = 1.0 - same_class                      # distance form

    def method_set(m):
        return set(str(m).split("/"))
    meth_j = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            a, b = method_set(method[i]), method_set(method[j])
            meth_j[i, j] = 1.0 - (len(a & b) / len(a | b) if a | b else 1.0)
    D_method = meth_j

    nuisance = [sim["token_jaccard"], sim["ast_ngram"], sim["len_diff"]]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(L)))
    rng = np.random.default_rng(args.seed)
    rows = []

    def emit(layer, stat, value, **kw):
        r = {"model": model_name, "pool": args.pool, "condition": args.condition,
             "layer": layer, "rel_depth": layer / (L - 1) if L > 1 else 0.0,
             "stat": stat, "value": float(value)}
        r.update(kw)
        rows.append(r)

    # --- 14.1 variance partitioning ----------------------------------------
    print("14.1 variance partitioning (Mantel permutation over solvers)...", flush=True)
    for layer in layers:
        D_rep = cosine_distance_matrix(H[sel, layer].astype(np.float64))

        # each regressor's own marginal correlation, for context
        y = upper(D_rep)
        for name, M in [("pde_class", D_class), ("num_method", D_method),
                        ("token_jaccard", sim["token_jaccard"]),
                        ("ast_ngram", sim["ast_ngram"]),
                        ("len_diff", sim["len_diff"])]:
            r = float(np.corrcoef(y, upper(M))[0, 1])
            emit(layer, "marginal_r", r, regressor=name)

        # the number that matters: pde_class AFTER code similarity
        res = partial_mantel(D_rep, D_class, nuisance, args.n_perm, rng)
        for k, v in res.items():
            emit(layer, f"pde_class_{k}", v, regressor="pde_class")

        # and the same for algorithm, so the two can be compared head to head
        res_m = partial_mantel(D_rep, D_method, nuisance, args.n_perm, rng)
        for k, v in res_m.items():
            emit(layer, f"num_method_{k}", v, regressor="num_method")

        # how much do the code regressors explain on their own?
        emit(layer, "code_only_r2", ols_r2(y, np.column_stack([upper(m) for m in nuisance])),
             regressor="code_similarity")

        if layer % 5 == 0 or layer == layers[-1]:
            print(f"  layer {layer:3d}/{L-1}  pde_class ΔR²={res['incremental_r2']:+.4f} "
                  f"(p={res['p']:.4g})  num_method ΔR²={res_m['incremental_r2']:+.4f}",
                  flush=True)

    # --- 14.2 method invariance --------------------------------------------
    print("14.2 method invariance (heat/wave/NS only — burgers is 8/8 explicit)...",
          flush=True)
    keep = np.array([p in METHOD_CLASSES for p in pde])
    n_keep = int(keep.sum())
    same_c_diff_m, diff_c_same_m = [], []
    for i, j in itertools.combinations(range(S), 2):
        if not (keep[i] and keep[j]):
            continue
        sc = pde[i] == pde[j]
        sm = method_set(method[i]) == method_set(method[j])
        if sc and not sm:
            same_c_diff_m.append((i, j))
        elif not sc and sm:
            diff_c_same_m.append((i, j))
    print(f"  cells: same-PDE/diff-method n={len(same_c_diff_m)}, "
          f"diff-PDE/same-method n={len(diff_c_same_m)}  (from {n_keep} solvers)",
          flush=True)

    if same_c_diff_m and diff_c_same_m:
        for layer in layers:
            D_rep = cosine_distance_matrix(H[sel, layer].astype(np.float64))
            a = float(np.mean([D_rep[i, j] for i, j in same_c_diff_m]))
            b = float(np.mean([D_rep[i, j] for i, j in diff_c_same_m]))
            emit(layer, "same_pde_diff_method_dist", a)
            emit(layer, "diff_pde_same_method_dist", b)
            # negative => same physics binds tighter than same algorithm
            emit(layer, "physics_minus_algorithm", a - b)
            emit(layer, "n_same_pde_diff_method", len(same_c_diff_m))
            emit(layer, "n_diff_pde_same_method", len(diff_c_same_m))
    else:
        print("  SKIPPED — one of the two cells is empty", flush=True)

    # --- 14.3 process transfer (exploratory) --------------------------------
    print("14.3 process-direction transfer (EXPLORATORY — see confounds)...", flush=True)
    for proc in ["diffusion", "advection"]:
        y_proc = np.array([int(proc in p.lower()) for p in process])
        cells = {c: (int(((pde == c) & (y_proc == 1)).sum()),
                     int(((pde == c) & (y_proc == 0)).sum()))
                 for c in sorted(set(pde))}
        print(f"  {proc}: " + "  ".join(f"{c}={p}+/{n}-" for c, (p, n) in cells.items()),
              flush=True)

        # a class is usable as a TEST set only if it contains both labels
        usable = [c for c, (p, n) in cells.items() if p > 0 and n > 0]
        if not usable:
            print(f"    no class has both labels — {proc} transfer is undefined, "
                  f"the label is a relabelling of pde_class", flush=True)
            for layer in layers:
                emit(layer, "process_transfer_undefined", 1.0, regressor=proc)
            continue

        for test_c in usable:
            tr = np.where(pde != test_c)[0]
            te = np.where(pde == test_c)[0]
            if len(np.unique(y_proc[tr])) < 2:
                continue
            for layer in layers:
                X = H[sel, layer].astype(np.float64)
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
                pos = Xn[tr][y_proc[tr] == 1].mean(axis=0)
                neg = Xn[tr][y_proc[tr] == 0].mean(axis=0)
                w = pos - neg
                w /= np.linalg.norm(w) or 1.0
                scores = Xn[te] @ w
                thr = float(np.median((Xn[tr] @ w)))
                acc = float(np.mean((scores > thr).astype(int) == y_proc[te]))
                emit(layer, "process_transfer_acc", acc, regressor=proc,
                     test_class=test_c, n_test=len(te),
                     n_test_pos=int(y_proc[te].sum()))

    os.makedirs(args.output_dir, exist_ok=True)
    slug = model_name.replace("/", "_")
    out = os.path.join(args.output_dir,
                       f"physics_vs_code_{slug}_{args.pool}_{args.condition}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)", flush=True)

    # --- summary ------------------------------------------------------------
    inc = df[df.stat == "pde_class_incremental_r2"].set_index("layer")["value"]
    ps = df[df.stat == "pde_class_p"].set_index("layer")["value"]
    best = inc.idxmax()
    print(f"\npde_class ΔR² after partialling out code similarity: "
          f"peak {inc.max():+.4f} at layer {best} "
          f"({best/(L-1):.2f} rel depth), p={ps[best]:.4g}", flush=True)
    incm = df[df.stat == "num_method_incremental_r2"].set_index("layer")["value"]
    print(f"num_method ΔR² for comparison:                        "
          f"peak {incm.max():+.4f} at layer {incm.idxmax()}", flush=True)
    co = df[df.stat == "code_only_r2"]["value"]
    print(f"code-similarity regressors alone explain R² up to {co.max():.4f}",
          flush=True)
    if inc.max() < 0.01:
        print("\nNOTE: pde_class adds almost nothing beyond code similarity at any "
              "layer. On this dataset that is a PHYSICS-NEGATIVE result and should "
              "be reported as one.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
