"""
RSA dimensionality control experiment.

For each model:
  1. Load hidden states at best pde_class layer (from rsa_block_{slug}.csv)
  2. Compute raw drift + symmetric and asymmetric normalized drift
  3. Repeat under random projection at k = 20, 50, 100 (10 repeats each)

Outputs:
  probe/results/rsa_dimcheck/drift_{slug}.csv          -- raw
  probe/results/rsa_dimcheck/drift_{slug}_rp{k}.csv   -- random projection
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
GT_MOD = "Comm_Valid"
MOD_TYPES = [
    "Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]
POOL = "mean_pool"
N_BOOT = 10_000
N_REPEATS = 10
RNG_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """Bootstrap 95% CI for mean of values (1-D array, n=16)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def load_best_layer(rsa_block_csv: Path, pool: str = POOL) -> int:
    """Return layer with minimum pde_block_score (tightest clustering)."""
    df = pd.read_csv(rsa_block_csv)
    sub = df[df["pool"] == pool].copy()
    best = sub.loc[sub["pde_block_score"].idxmin(), "layer"]
    return int(best)


def load_hidden(npz_path: Path, pool: str, layer: int):
    """
    Load hidden states from NPZ.
    Returns:
      H: (128, D) float32 array at the given layer
      meta: dict with mod_types, pde_classes, gt_samples (all length-128 string arrays)
    """
    data = np.load(npz_path, allow_pickle=True)
    # shape: (128, n_layers, D)
    H_all = data[pool].astype(np.float32)
    H = H_all[:, layer, :]  # (128, D)
    meta = {
        "mod_types":   data["mod_types"],
        "pde_classes": data["pde_classes"],
        "gt_samples":  data["gt_samples"],
    }
    return H, meta


def compute_drift(H: np.ndarray, meta: dict):
    """
    Compute raw drift, symmetric control, asymmetric control, and ratios.

    Returns a DataFrame with one row per mod_type (excluding GT_MOD) per gt_sample.
    Columns: gt_sample, mod_type, d_task, d_control_sym, d_control_asym
    """
    mod_types  = meta["mod_types"]
    pde_classes = meta["pde_classes"]
    gt_samples  = meta["gt_samples"]

    unique_gt = sorted(set(gt_samples))
    unique_mt = [m for m in MOD_TYPES if m != GT_MOD]

    # Index lookups
    def idx(gt, mt):
        matches = np.where((gt_samples == gt) & (mod_types == mt))[0]
        assert len(matches) == 1, f"Expected 1 match for ({gt}, {mt}), got {len(matches)}"
        return matches[0]

    # Pre-compute symmetric control per gt_sample:
    # d_control_sym(g) = mean over g' (cross-class) of dist(h(g,GT), h(g',GT))
    gt_class = {g: pde_classes[idx(g, GT_MOD)] for g in unique_gt}

    d_control_sym_per_g = {}
    for g in unique_gt:
        h_gt_g = H[idx(g, GT_MOD)]
        cross_class_dists = []
        for g2 in unique_gt:
            if gt_class[g2] == gt_class[g]:
                continue
            h_gt_g2 = H[idx(g2, GT_MOD)]
            cross_class_dists.append(cosine_dist(h_gt_g, h_gt_g2))
        assert len(cross_class_dists) == 12, f"Expected 12 cross-class samples for {g}, got {len(cross_class_dists)}"
        d_control_sym_per_g[g] = float(np.mean(cross_class_dists))

    rows = []
    for g in unique_gt:
        h_gt_g = H[idx(g, GT_MOD)]
        d_ctrl_sym = d_control_sym_per_g[g]

        for mt in unique_mt:
            h_m = H[idx(g, mt)]
            d_task = cosine_dist(h_gt_g, h_m)

            # Asymmetric control: dist from h(g,m) to cross-class GT
            asym_dists = []
            for g2 in unique_gt:
                if gt_class[g2] == gt_class[g]:
                    continue
                h_gt_g2 = H[idx(g2, GT_MOD)]
                asym_dists.append(cosine_dist(h_m, h_gt_g2))
            d_ctrl_asym = float(np.mean(asym_dists))

            assert d_ctrl_sym > 1e-6, f"d_control_sym near zero for {g}"
            assert d_ctrl_asym > 1e-6, f"d_control_asym near zero for ({g}, {mt})"

            rows.append({
                "gt_sample":       g,
                "mod_type":        mt,
                "d_task":          d_task,
                "d_control_sym":   d_ctrl_sym,
                "d_control_asym":  d_ctrl_asym,
                "r_sym":           d_task / d_ctrl_sym,
                "r_asym":          d_task / d_ctrl_asym,
            })

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(g,m) rows into per-m summary with bootstrap CIs."""
    out = []
    for mt, grp in df.groupby("mod_type"):
        vals_d_task    = grp["d_task"].values
        vals_r_sym     = grp["r_sym"].values
        vals_r_asym    = grp["r_asym"].values
        vals_ctrl_sym  = grp["d_control_sym"].values
        vals_ctrl_asym = grp["d_control_asym"].values

        mean_dt, _, _            = bootstrap_mean_ci(vals_d_task)
        mean_rs, rs_lo, rs_hi    = bootstrap_mean_ci(vals_r_sym)
        mean_ra, ra_lo, ra_hi    = bootstrap_mean_ci(vals_r_asym)

        out.append({
            "mod_type":          mt,
            "mean_d_task":       mean_dt,
            "sem_d_task":        float(vals_d_task.std() / np.sqrt(len(vals_d_task))),
            "mean_d_control_sym":  float(vals_ctrl_sym.mean()),
            "sem_d_control_sym":   float(vals_ctrl_sym.std() / np.sqrt(len(vals_ctrl_sym))),
            "mean_d_control_asym": float(vals_ctrl_asym.mean()),
            "sem_d_control_asym":  float(vals_ctrl_asym.std() / np.sqrt(len(vals_ctrl_asym))),
            "r_sym":             mean_rs,
            "r_sym_ci_lo":       rs_lo,
            "r_sym_ci_hi":       rs_hi,
            "r_asym":            mean_ra,
            "r_asym_ci_lo":      ra_lo,
            "r_asym_ci_hi":      ra_hi,
        })
    return pd.DataFrame(out)


def random_project(H: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Project (N, D) → (N, k) using R ~ N(0, 1/k), then L2-normalize rows."""
    D = H.shape[1]
    R = rng.standard_normal((k, D)) / np.sqrt(k)
    H_proj = H @ R.T  # (N, k)
    norms = np.linalg.norm(H_proj, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (H_proj / norms).astype(np.float32)


def run_random_projection(H: np.ndarray, meta: dict, k: int, n_repeats: int = N_REPEATS):
    """Average r_sym and r_asym over n_repeats random projections at dim k."""
    rng = np.random.default_rng(RNG_SEED)
    rep_rows = []  # list of aggregate DataFrames, one per repeat
    for _ in range(n_repeats):
        H_proj = random_project(H, k, rng)
        df = compute_drift(H_proj, meta)
        agg = aggregate(df)
        rep_rows.append(agg)

    # Average r_sym and r_asym over repeats; keep other columns from first repeat
    base = rep_rows[0].copy()
    for col in ["r_sym", "r_sym_ci_lo", "r_sym_ci_hi",
                "r_asym", "r_asym_ci_lo", "r_asym_ci_hi",
                "mean_d_task", "sem_d_task",
                "mean_d_control_sym", "mean_d_control_asym"]:
        base[col] = np.mean([r[col].values for r in rep_rows], axis=0)

    # Std of r_sym and r_asym means across repeats
    base["r_sym_rep_std"]  = np.std([r["r_sym"].values  for r in rep_rows], axis=0)
    base["r_asym_rep_std"] = np.std([r["r_asym"].values for r in rep_rows], axis=0)
    return base


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden",     required=True, help="Path to NPZ hidden states")
    parser.add_argument("--rsa_block",  required=True, help="Path to rsa_block_{slug}.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--slug",       required=True, help="Model slug")
    parser.add_argument("--proj_dims",  nargs="+", type=int, default=[20, 50, 100])
    args = parser.parse_args()

    npz_path   = Path(args.hidden)
    block_csv  = Path(args.rsa_block)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        sys.exit(f"ERROR: NPZ not found: {npz_path}")
    if not block_csv.exists():
        sys.exit(f"ERROR: rsa_block CSV not found: {block_csv}")

    # 1. Best layer
    best_layer = load_best_layer(block_csv, pool=POOL)
    print(f"[{args.slug}] Best pde_class layer ({POOL}): {best_layer}")

    # 2. Load hidden states
    H, meta = load_hidden(npz_path, POOL, best_layer)
    print(f"[{args.slug}] Hidden state shape at layer {best_layer}: {H.shape}")

    # 3. Raw drift
    print(f"[{args.slug}] Computing raw drift...")
    df_raw = compute_drift(H, meta)
    agg_raw = aggregate(df_raw)
    agg_raw["best_layer"] = best_layer
    agg_raw["pca_dim"]    = "raw"
    out_raw = out_dir / f"drift_{args.slug}.csv"
    agg_raw.to_csv(out_raw, index=False)
    print(f"[{args.slug}] Saved: {out_raw}")

    # Sanity: print CorrVar raw d_task
    corrvar = agg_raw[agg_raw["mod_type"] == "NoComm_CorrVar"]
    if not corrvar.empty:
        print(f"[{args.slug}] CorrVar mean_d_task = {corrvar['mean_d_task'].iloc[0]:.4f}  "
              f"r_sym = {corrvar['r_sym'].iloc[0]:.4f} [{corrvar['r_sym_ci_lo'].iloc[0]:.4f}, {corrvar['r_sym_ci_hi'].iloc[0]:.4f}]")

    # 4. Random projection
    for k in args.proj_dims:
        print(f"[{args.slug}] Random projection k={k} ({N_REPEATS} repeats)...")
        agg_rp = run_random_projection(H, meta, k)
        agg_rp["best_layer"] = best_layer
        agg_rp["pca_dim"]    = k
        out_rp = out_dir / f"drift_{args.slug}_rp{k}.csv"
        agg_rp.to_csv(out_rp, index=False)
        print(f"[{args.slug}] Saved: {out_rp}")
        corrvar_rp = agg_rp[agg_rp["mod_type"] == "NoComm_CorrVar"]
        if not corrvar_rp.empty:
            print(f"  CorrVar r_sym (k={k}) = {corrvar_rp['r_sym'].iloc[0]:.4f}  "
                  f"rep_std = {corrvar_rp['r_sym_rep_std'].iloc[0]:.4f}")

    print(f"[{args.slug}] Done.")


if __name__ == "__main__":
    main()
