"""
probe-validity-followup: phys_valid probe on NoComm vs Comm subsets.

Tests whether phys_valid probe signal survives when docstring comments are absent.

Usage:
    python probe/probe_validity_split.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/validity_split/ \
        --pool mean_pool \
        --C 1.0 \
        --layers 0 14 28   # optional subset

Outputs:
    validity_split_{pool}.csv
    Columns: split, layer, pool, C, accuracy, ci_low, ci_high,
             auroc, auroc_ci_low, auroc_ci_high, n_rows
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from probe_utils import load_data, bootstrap_ci

NOCOMM_TYPES = {"NoComm_Valid", "NoComm_InValid", "NoComm_CorrVar", "NoComm_CorrVar_InValid"}
COMM_TYPES   = {"Comm_Valid", "CorrComm", "Comm_InValid", "CorrComm_Invalid"}

SPLITS = {
    "full":   None,   # all rows
    "nocomm": NOCOMM_TYPES,
    "comm":   COMM_TYPES,
}


def run_logo_phys_valid(X_all, y, groups, C):
    """LOGO-CV for phys_valid on a given subset. Returns (acc, lo, hi, auroc, auroc_lo, auroc_hi)."""
    logo = LeaveOneGroupOut()
    fold_accs, fold_aurocs = [], []

    for train_idx, test_idx in logo.split(X_all, y, groups):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2:
            fold_accs.append(float(np.mean(y_te == y_tr[0])))
            fold_aurocs.append(float("nan"))
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        fold_accs.append(float(np.mean(pred == y_te)))

        try:
            proba = clf.predict_proba(X_te)
            auroc = roc_auc_score(y_te, proba[:, 1])
        except ValueError:
            auroc = float("nan")
        fold_aurocs.append(auroc)

    acc, lo, hi = bootstrap_ci(fold_accs)
    valid_aurocs = [a for a in fold_aurocs if not np.isnan(a)]
    if valid_aurocs:
        auroc, auroc_lo, auroc_hi = bootstrap_ci(valid_aurocs)
    else:
        auroc, auroc_lo, auroc_hi = float("nan"), float("nan"), float("nan")

    return acc, lo, hi, auroc, auroc_lo, auroc_hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--output_dir", default="probe/results/validity_split/")
    parser.add_argument("--pool", default="mean_pool", choices=["mean_pool", "last_tok"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"validity_split_{args.pool}.csv")

    data = load_data(args.hidden)
    reps = data[args.pool].astype(np.float32)   # (N, L, D)
    N, L, D = reps.shape
    mod_types = data["mod_types"]
    groups    = data["gt_samples"]
    y_full    = data["phys_valid"].astype(np.int32)

    # Verify dataset composition
    unique_mt, counts = np.unique(mod_types, return_counts=True)
    print(f"Dataset: N={N}, L={L}, D={D}, pool={args.pool}", flush=True)
    print("mod_type counts:", dict(zip(unique_mt, counts)), flush=True)
    for split_name, allowed in SPLITS.items():
        mask = np.ones(N, dtype=bool) if allowed is None else np.isin(mod_types, list(allowed))
        y_sub = y_full[mask]
        vals, cnts = np.unique(y_sub, return_counts=True)
        print(f"  {split_name}: n={mask.sum()}, phys_valid dist={dict(zip(vals.tolist(), cnts.tolist()))}", flush=True)

    layers_to_run = args.layers if args.layers is not None else list(range(L))

    # Resume
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        done = set(zip(existing["split"].astype(str), existing["layer"].astype(str)))
        rows = existing.to_dict("records")
        print(f"Resuming: {len(rows)} rows done, skipping {len(done)} (split,layer) combos.", flush=True)
    else:
        done = set()
        rows = []

    for split_name, allowed in SPLITS.items():
        mask = np.ones(N, dtype=bool) if allowed is None else np.isin(mod_types, list(allowed))
        X_split = reps[mask]
        y_split = y_full[mask]
        g_split = groups[mask]
        n_split = int(mask.sum())

        print(f"\n--- Split: {split_name} (n={n_split}) ---", flush=True)

        for l in layers_to_run:
            key = (split_name, str(l))
            if key in done:
                continue

            X = X_split[:, l, :]
            acc, lo, hi, auroc, auroc_lo, auroc_hi = run_logo_phys_valid(X, y_split, g_split, args.C)

            row = {
                "split": split_name, "layer": l, "pool": args.pool, "C": args.C,
                "n_rows": n_split,
                "accuracy": acc, "ci_low": lo, "ci_high": hi,
                "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
            }
            rows.append(row)
            done.add(key)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  layer={l}: acc={acc:.3f} auroc={auroc:.3f}", flush=True)

    print(f"\nDone. {out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
