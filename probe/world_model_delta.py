"""
Experiment 2 — World Model: are physical defects represented consistently across
descriptions?

For each solver s, surface condition c, layer l and pooling p:

    Δh(s, c, l, p) = h(invalid | s, c, l, p) − h(valid | s, c, l, p)

The primary question is whether the *same* physical edit produces a similar
representational change under *different* descriptions of the same program, i.e.
whether cos(Δh(s,c), Δh(s,c')) is high — and, critically, higher than it would be
for two *different* solvers, which is what separates "the model represents this
specific physical intervention" from "the model has one generic broken-code
direction".

Usage (from repo root):
    python probe/world_model_delta.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/ \
        --pool mean_pool

Writes one tidy long-form CSV per (model, pool):
    probe/results/world_model_delta_{model_slug}_{pool}.csv

Columns:
    model, pool, layer, rel_depth, stat, condition_a, condition_b,
    gt_sample, source, pde_class, stratum, value

`stat` is one of:
    within_cos        cos(Δh(s,c), Δh(s,c')) — the primary statistic, per solver
    cross_cos         cos(Δh(s,c), Δh(s',c')), s ≠ s' — generic-direction control
    cross_cos_sameclass  as cross_cos but restricted to matching pde_class
    random_cos        cos(Δh(s,c), g), g ~ N(0, I) — geometry sanity check, ≈ 0
    gap               mean(within_cos) − mean(cross_cos), per layer
    gap_perm_p        permutation p-value for `gap`
    gap_perm_null_hi  95th percentile of the permutation null for `gap`
    delta_norm_ratio  ‖Δh‖ / ‖h_valid‖ — float precision + effect-size audit
    surface_norm      ‖h(c) − h(c')‖ between two *valid* arms (surface-only move)
    physics_norm      ‖Δh(s,c)‖ (physics-only move, same surface)
    match_acc         top-1 defect identification across conditions (chance = 1/S)
    match_mrr         mean reciprocal rank for the same retrieval
    generic_transfer_acc  does a SHARED defect direction transfer c → c' (chance = 0.5)
    subspace_angle    k-th principal angle (deg) between span{Δh(·,c)}, span{Δh(·,c')}

The two statistics that matter most are `gap` and `match_acc`. `within_cos` alone
is not evidence: a single global "this code is broken" direction drives it near 1.0
while leaving both `gap` and `match_acc` at chance. That separation is verified
against synthetic data in tests/test_world_model_delta.py.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

# The 8 mod_types factor into 4 surface conditions × {valid, invalid}.
# Verified against data/merged_mod_jul28.csv: comment-stripped source is identical
# across S_plain / S_bare / S_mislead, and S_obf is AST-isomorphic to them.
CONDITIONS = {
    "S_plain":   ("Comm_Valid",     "Comm_InValid"),
    "S_bare":    ("NoComm_Valid",   "NoComm_InValid"),
    "S_mislead": ("CorrComm",       "CorrComm_Invalid"),
    "S_obf":     ("NoComm_CorrVar", "NoComm_CorrVar_InValid"),
}
COND_NAMES = list(CONDITIONS)

# Pre-registered strata (see probe/plans/exp2_world_model.md §7). These are NOT
# dropped — they are labelled so the primary analysis can be reported with and
# without them, rather than silently excluding rows.
LEAK_SOLVERS = {"Burgers_6", "Burgers_7", "Burgers_8",
                "NavierStokes_5", "NavierStokes_7"}
GRID_SOLVERS = {"Heat_2"}


def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_pair_index(data: dict) -> tuple:
    """
    Return (solvers, idx) where idx[(gt_sample, mod_type)] -> row index.
    Fails loudly if any solver is missing any of the 8 conditions — the whole
    experiment is undefined on a ragged grid.
    """
    gt = data["gt_samples"].astype(str)
    mt = data["mod_types"].astype(str)

    idx = {}
    for i, (g, m) in enumerate(zip(gt, mt)):
        key = (g, m)
        if key in idx:
            raise ValueError(f"duplicate (gt_sample, mod_type) in NPZ: {key}")
        idx[key] = i

    solvers = sorted(set(gt))
    needed = [m for pair in CONDITIONS.values() for m in pair]
    missing = [(g, m) for g in solvers for m in needed if (g, m) not in idx]
    if missing:
        raise ValueError(
            f"{len(missing)} missing (gt_sample, mod_type) cells, e.g. {missing[:5]}"
        )
    return solvers, idx


def compute_deltas(H: np.ndarray, solvers: list, idx: dict, layer: int) -> np.ndarray:
    """
    Δ[s, c, :] = h(invalid) − h(valid) for solver s under condition c at `layer`.
    Returns (S, C, D) float64 — float64 because the whole point is that Δ is a
    small difference of large vectors.
    """
    S, C, D = len(solvers), len(COND_NAMES), H.shape[2]
    delta = np.zeros((S, C, D), dtype=np.float64)
    for si, s in enumerate(solvers):
        for ci, c in enumerate(COND_NAMES):
            valid_mt, invalid_mt = CONDITIONS[c]
            h_valid = H[idx[(s, valid_mt)], layer].astype(np.float64)
            h_inval = H[idx[(s, invalid_mt)], layer].astype(np.float64)
            delta[si, ci] = h_inval - h_valid
    return delta


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine between rows of A (n, D) and rows of B (m, D) -> (n, m)."""
    return unit(A) @ unit(B).T


def permutation_gap(dc: np.ndarray, dc2: np.ndarray, n_perm: int,
                    rng: np.random.Generator) -> tuple:
    """
    dc, dc2: (S, D) defect vectors for two conditions, aligned by solver.

    Observed gap = mean of the matched diagonal − mean of the off-diagonal.
    Null: permute solver identity in the second condition. The off-diagonal mean
    is (near-)invariant under permutation, so the null is carried by the diagonal
    term — which is exactly the "same solver" assumption we are testing.

    The independent unit is the solver (S of them), not the S² pairs.
    """
    M = cos_matrix(dc, dc2)                       # (S, S)
    S = M.shape[0]
    off = ~np.eye(S, dtype=bool)
    obs_within = float(np.mean(np.diag(M)))
    obs_cross = float(np.mean(M[off]))
    obs_gap = obs_within - obs_cross

    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(S)
        # Guard: a permutation with fixed points leaks true pairs into the null,
        # which biases p upward (conservative). Left in deliberately — resampling
        # to derangements only would understate the null's true spread.
        null[k] = float(np.mean(M[np.arange(S), perm])) - obs_cross

    p = float((np.sum(null >= obs_gap) + 1) / (n_perm + 1))
    return obs_gap, obs_within, obs_cross, p, float(np.percentile(null, 95))


def generic_transfer_acc(dc_train: np.ndarray, dc_test: np.ndarray) -> float:
    """
    Does a SHARED, solver-independent defect direction exist and survive a change
    of description? Leave-one-solver-out: fit the mean defect direction on all
    other solvers under condition c, check it points the held-out solver's defect
    the right way under condition c'. Chance = 0.5.

    This is the decodable counterpart to `cross_cos`, NOT to `within_cos` — the
    mean over solvers deliberately averages away anything solver-specific. On
    synthetic data with per-solver defect directions this correctly reads ~0.5,
    and on data with one global defect direction it reads ~1.0.
    """
    S = dc_train.shape[0]
    hits = 0
    for s in range(S):
        others = np.delete(np.arange(S), s)
        w = unit(dc_train[others].mean(axis=0))
        hits += int(float(unit(dc_test[s]) @ w) > 0)
    return hits / S


def match_acc(M: np.ndarray) -> tuple:
    """
    Defect identification: given solver s's defect under condition c, can we pick
    out that same solver's defect under condition c' from all S candidates?

    M is the (S, S) cosine matrix between conditions. Top-1 accuracy and mean
    reciprocal rank over the rows. Chance = 1/S (3.1% at S=32), so this is a far
    sharper test than any 0.5-chance statistic — and it is the one that directly
    answers the RQ, because it can only succeed if the defect representation is
    solver-specific AND stable across descriptions.

    A global "broken code" direction scores at chance here even though it drives
    `within_cos` near 1.0, which is exactly the confound we need separated.
    """
    S = M.shape[0]
    order = np.argsort(-M, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(S)])
    return float(np.mean(ranks == 1)), float(np.mean(1.0 / ranks))


def principal_angles(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """Principal angles (degrees) between the top-k subspaces of A and B (both (S, D))."""
    k = int(min(k, A.shape[0], B.shape[0], A.shape[1]))
    Qa = np.linalg.svd(unit(A).T, full_matrices=False)[0][:, :k]
    Qb = np.linalg.svd(unit(B).T, full_matrices=False)[0][:, :k]
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def stratum_of(solver: str) -> str:
    if solver in LEAK_SOLVERS:
        return "cadence_leak"
    if solver in GRID_SOLVERS:
        return "grid_change"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", required=True, help="NPZ from extract_hidden.py")
    ap.add_argument("--output_dir", default="probe/results/")
    ap.add_argument("--pool", default="mean_pool", choices=["mean_pool", "last_tok"])
    ap.add_argument("--layers", default=None,
                    help="Comma-separated layer indices. Omit for all layers.")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--subspace_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_npz(args.hidden)
    H = data[args.pool]                       # (N, L, D)
    N, L, D = H.shape
    model_name = str(data["model_name"]) if "model_name" in data else \
        os.path.basename(args.hidden).replace(".npz", "")
    model_slug = model_name.replace("/", "_")

    print(f"Model : {model_name}", flush=True)
    print(f"Pool  : {args.pool}   hidden {H.shape}  dtype={H.dtype}", flush=True)
    if H.dtype == np.float16:
        print("WARNING: hidden states are float16. Δh is a small difference of large "
              "vectors; cosines may be dominated by rounding. Re-extract in float32.",
              flush=True)

    solvers, idx = build_pair_index(data)
    S = len(solvers)
    print(f"Solvers: {S} × {len(COND_NAMES)} conditions", flush=True)

    # Per-solver metadata, taken from the Comm_Valid row of each solver
    gt_arr = data["gt_samples"].astype(str)
    pde_arr = data["pde_classes"].astype(str)
    src_arr = (data["sources"].astype(str) if "sources" in data
               else np.array(["unknown"] * N))
    code_arr = data["codes"].astype(str) if "codes" in data else None

    solver_pde = {s: pde_arr[idx[(s, "Comm_Valid")]] for s in solvers}
    solver_src = {s: src_arr[idx[(s, "Comm_Valid")]] for s in solvers}
    solver_stratum = {s: stratum_of(s) for s in solvers}

    # Zero-length-edit subset: solvers where the physical edit changes no code
    # length at all, so ‖Δh‖ cannot be driven by a token-count difference.
    if code_arr is not None:
        for s in solvers:
            dv = len(code_arr[idx[(s, "Comm_InValid")]]) - len(code_arr[idx[(s, "Comm_Valid")]])
            if dv == 0 and solver_stratum[s] == "clean":
                solver_stratum[s] = "clean_zero_len_delta"
    n_clean = sum(1 for s in solvers if solver_stratum[s].startswith("clean"))
    print(f"Strata : {n_clean} clean "
          f"({sum(1 for s in solvers if solver_stratum[s] == 'clean_zero_len_delta')} "
          f"zero-length-edit), "
          f"{sum(1 for s in solvers if solver_stratum[s] == 'cadence_leak')} cadence_leak, "
          f"{sum(1 for s in solvers if solver_stratum[s] == 'grid_change')} grid_change",
          flush=True)

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(L)))
    rng = np.random.default_rng(args.seed)
    rows = []

    def emit(layer, stat, value, ca="", cb="", solver="", k=None):
        rows.append({
            "model": model_name,
            "pool": args.pool,
            "layer": layer,
            "rel_depth": layer / (L - 1) if L > 1 else 0.0,
            "stat": stat,
            "condition_a": ca,
            "condition_b": cb,
            "gt_sample": solver,
            "source": solver_src.get(solver, ""),
            "pde_class": solver_pde.get(solver, ""),
            "stratum": solver_stratum.get(solver, ""),
            "k": k if k is not None else "",
            "value": float(value),
        })

    same_class = np.array([[solver_pde[a] == solver_pde[b] for b in solvers]
                           for a in solvers])
    off_diag = ~np.eye(S, dtype=bool)

    for layer in layers:
        delta = compute_deltas(H, solvers, idx, layer)          # (S, C, D)

        # --- norm audit: is Δh big enough to be real, and how does it compare
        # --- to a purely surface-level move of the same representation?
        for ci, c in enumerate(COND_NAMES):
            valid_mt, _ = CONDITIONS[c]
            for si, s in enumerate(solvers):
                h_valid = H[idx[(s, valid_mt)], layer].astype(np.float64)
                dn = float(np.linalg.norm(delta[si, ci]))
                hn = float(np.linalg.norm(h_valid))
                emit(layer, "physics_norm", dn, ca=c, solver=s)
                emit(layer, "delta_norm_ratio", dn / hn if hn else np.nan, ca=c, solver=s)

        # surface-only moves: between the *valid* arms of two conditions. Same
        # physics, different description — the scale reference for physics_norm.
        for ca, cb in itertools.combinations(COND_NAMES, 2):
            for s in solvers:
                ha = H[idx[(s, CONDITIONS[ca][0])], layer].astype(np.float64)
                hb = H[idx[(s, CONDITIONS[cb][0])], layer].astype(np.float64)
                emit(layer, "surface_norm", np.linalg.norm(ha - hb), ca=ca, cb=cb, solver=s)

        # --- primary statistic + controls, per condition pair
        for ai, bi in itertools.combinations(range(len(COND_NAMES)), 2):
            ca, cb = COND_NAMES[ai], COND_NAMES[bi]
            A, B = delta[:, ai, :], delta[:, bi, :]
            M = cos_matrix(A, B)

            for si, s in enumerate(solvers):
                emit(layer, "within_cos", M[si, si], ca=ca, cb=cb, solver=s)

            emit(layer, "cross_cos", float(np.mean(M[off_diag])), ca=ca, cb=cb)
            sc = off_diag & same_class
            if sc.any():
                emit(layer, "cross_cos_sameclass", float(np.mean(M[sc])), ca=ca, cb=cb)

            gap, within, cross, p, null_hi = permutation_gap(A, B, args.n_perm, rng)
            emit(layer, "gap", gap, ca=ca, cb=cb)
            emit(layer, "gap_perm_p", p, ca=ca, cb=cb)
            emit(layer, "gap_perm_null_hi", null_hi, ca=ca, cb=cb)

            # Defect identification, both directions (the cosine matrix is not
            # symmetric in its row-normalisation, so run it each way).
            for Mx, x, y in ((M, ca, cb), (M.T, cb, ca)):
                top1, mrr = match_acc(Mx)
                emit(layer, "match_acc", top1, ca=x, cb=y)
                emit(layer, "match_mrr", mrr, ca=x, cb=y)

            # transfer runs in both directions — it is not symmetric
            emit(layer, "generic_transfer_acc", generic_transfer_acc(A, B), ca=ca, cb=cb)
            emit(layer, "generic_transfer_acc", generic_transfer_acc(B, A), ca=cb, cb=ca)

            for k, ang in enumerate(principal_angles(A, B, args.subspace_k)):
                emit(layer, "subspace_angle", ang, ca=ca, cb=cb, k=k)

        # --- geometry sanity check: a random direction must score ~0. If this is
        # --- not ~0, the cosine statistic is picking up dimensionality, not signal.
        g = rng.standard_normal((S, D))
        for ci, c in enumerate(COND_NAMES):
            r = float(np.mean(np.sum(unit(delta[:, ci, :]) * unit(g), axis=1)))
            emit(layer, "random_cos", r, ca=c)

        if layer % 5 == 0 or layer == layers[-1]:
            sel = [r for r in rows if r["layer"] == layer and r["stat"] == "gap"]
            mg = np.mean([r["value"] for r in sel]) if sel else float("nan")
            print(f"  layer {layer:3d}/{L-1}  mean gap = {mg:+.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"world_model_delta_{model_slug}_{args.pool}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)", flush=True)

    # --- console summary: the one number the experiment is about
    gaps = df[df.stat == "gap"].groupby("layer")["value"].mean()
    best = gaps.idxmax()
    ps = df[(df.stat == "gap_perm_p") & (df.layer == best)]["value"]
    print(f"\nPeak mean gap: layer {best} ({best/(L-1):.2f} rel depth), "
          f"gap={gaps[best]:+.4f}, max p across pairs={ps.max():.4g}", flush=True)

    ma = df[(df.stat == "match_acc") & (df.layer == best)]["value"]
    print(f"Defect ID    : top-1 {ma.mean():.3f} at layer {best} "
          f"(chance {1/S:.3f}) — the solver-specific test", flush=True)
    gt = df[(df.stat == "generic_transfer_acc") & (df.layer == best)]["value"]
    print(f"Generic dir  : {gt.mean():.3f} (chance 0.500) — shared, "
          f"solver-independent defect direction", flush=True)

    l0 = gaps.get(0, float("nan"))
    print(f"Layer-0 gap  : {l0:+.4f}  "
          f"{'<-- RED FLAG: embeddings see only token identity' if l0 > 0.05 else '(ok)'}",
          flush=True)

    nr = df[df.stat == "delta_norm_ratio"]["value"]
    print(f"‖Δh‖/‖h‖     : median {nr.median():.4g}, min {nr.min():.4g}", flush=True)
    if nr.median() < 1e-3:
        print("WARNING: Δh is tiny relative to h. float32 is required here; "
              "verify the extraction dtype.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
