"""
Hyperparameter sweep for the pooled linear probe.

Sweeps over C (regularization) and representation type (raw D=3584 vs PCA-20).
PCA is fit inside each LOGO fold on training data only to prevent leakage.

Usage:
    python probe/probe_hyperparam.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/ \
        --pool mean_pool \
        --representation raw \
        --C_grid 0.01 0.1 1.0 10.0

Outputs:
    probe/results/probe_hyperparam_{representation}_{pool}.csv
    Columns: label, layer, pool, representation, C,
             accuracy, ci_low, ci_high,
             auroc, auroc_ci_low, auroc_ci_high,
             mt_{mod_type} (per-mod_type accuracy at best C)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from probe_utils import (
    load_data, extract_label_arrays, bootstrap_ci,
    MOD_TYPES, BINARY_PROCESS_LABELS, BINARY_METHOD_LABELS,
)

LABEL_ORDER = (
    ["pde_class"]
    + [f"process_{p}" for p in BINARY_PROCESS_LABELS]
    + [f"method_{m}" for m in BINARY_METHOD_LABELS]
    + ["phys_valid"]
)

CHANCE = {
    "pde_class": 0.25,
    **{f"process_{p}": 0.5 for p in BINARY_PROCESS_LABELS},
    **{f"method_{m}": 0.5 for m in BINARY_METHOD_LABELS},
    "phys_valid": 0.5,
}


def run_probe_fold(X_train, X_test, y_train, y_test, C, n_pca, mt_test):
    """Single LOGO fold: optional PCA → scale → logistic regression."""
    if n_pca is not None:
        k = min(n_pca, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=k, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        pred = np.full(len(y_test), y_train[0])
        auroc = float("nan")
    else:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        try:
            if len(clf.classes_) == 2:
                auroc = roc_auc_score(y_test, proba[:, 1])
            else:
                auroc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except ValueError:
            auroc = float("nan")

    fold_acc = float(np.mean(pred == y_test))
    mt_acc = {}
    for mt in MOD_TYPES:
        mask = mt_test == mt
        if mask.any():
            mt_acc[mt] = float(np.mean(pred[mask] == y_test[mask]))

    return fold_acc, auroc, mt_acc


def run_logo_sweep(X_all, y, groups, mod_types, C, n_pca):
    """LOGO-CV over all folds for a given (C, n_pca)."""
    logo = LeaveOneGroupOut()
    fold_accs, fold_aurocs = [], []
    # per-mod_type: accumulate per-fold accuracy values
    mt_fold_accs = {mt: [] for mt in MOD_TYPES}

    for train_idx, test_idx in logo.split(X_all, y, groups):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mt_test = mod_types[test_idx]

        fold_acc, auroc, mt_acc = run_probe_fold(
            X_train, X_test, y_train, y_test, C, n_pca, mt_test
        )
        fold_accs.append(fold_acc)
        fold_aurocs.append(auroc)
        for mt in MOD_TYPES:
            if mt in mt_acc:
                mt_fold_accs[mt].append(mt_acc[mt])

    acc, lo, hi = bootstrap_ci(fold_accs)
    valid_aurocs = [a for a in fold_aurocs if not np.isnan(a)]
    auroc, auroc_lo, auroc_hi = (
        bootstrap_ci(valid_aurocs) if valid_aurocs else (float("nan"),) * 3
    )
    mt_result = {
        mt: float(np.mean(vals)) if vals else float("nan")
        for mt, vals in mt_fold_accs.items()
    }

    return acc, lo, hi, auroc, auroc_lo, auroc_hi, mt_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--output_dir", default="probe/results/")
    parser.add_argument("--pool", default="mean_pool", choices=["mean_pool", "last_tok"])
    parser.add_argument("--representation", default="raw", choices=["raw", "pca20"])
    parser.add_argument("--C_grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Subset of layer indices to run (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_pca = 20 if args.representation == "pca20" else None

    data = load_data(args.hidden)
    reps = data[args.pool].astype(np.float32)   # (N, L, D)
    N, L, D = reps.shape
    groups = data["gt_samples"]
    mod_types = data["mod_types"]
    label_arrays = extract_label_arrays(data)

    out_csv = os.path.join(
        args.output_dir, f"probe_hyperparam_{args.representation}_{args.pool}.csv"
    )

    layers_to_run = args.layers if args.layers is not None else list(range(L))
    print(f"Probing {N} examples, layers={layers_to_run}, D={D} [{args.pool}] "
          f"representation={args.representation} C_grid={args.C_grid}", flush=True)

    # Resume: load any existing rows and skip already-done (label, C, layer) combos
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        done = set(
            zip(existing["label"].astype(str),
                existing["C"].astype(float).round(6),
                existing["layer"].astype(str))
        )
        rows = existing.to_dict("records")
        print(f"Resuming — {len(rows)} rows already done, {len(done)} (label,C,layer) combos skipped.",
              flush=True)
    else:
        done = set()
        rows = []

    for label_name in LABEL_ORDER:
        y = label_arrays[label_name]
        print(f"\n--- {label_name} (chance={CHANCE[label_name]:.2f}) ---", flush=True)

        for C in args.C_grid:
            layer_accs = []
            for l in layers_to_run:
                key = (label_name, round(C, 6), str(l))
                if key in done:
                    # recover acc for best-layer reporting
                    match = existing[
                        (existing["label"] == label_name) &
                        (existing["C"].round(6) == round(C, 6)) &
                        (existing["layer"].astype(str) == str(l))
                    ]
                    layer_accs.append(float(match["accuracy"].iloc[0]) if not match.empty else 0.0)
                    continue

                X = reps[:, l, :]
                acc, lo, hi, auroc, auroc_lo, auroc_hi, mt_acc = run_logo_sweep(
                    X, y, groups, mod_types, C, n_pca
                )
                layer_accs.append(acc)
                row = {
                    "label": label_name, "layer": l, "pool": args.pool,
                    "representation": args.representation, "C": C,
                    "accuracy": acc, "ci_low": lo, "ci_high": hi,
                    "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
                }
                row.update({f"mt_{mt}": mt_acc.get(mt, float("nan")) for mt in MOD_TYPES})
                rows.append(row)
                done.add(key)

                # checkpoint after every layer
                pd.DataFrame(rows).to_csv(out_csv, index=False)

            if layer_accs:
                best_l = layers_to_run[int(np.argmax(layer_accs))]
                print(f"  C={C:.2f}: best layer={best_l} acc={max(layer_accs):.3f}", flush=True)
            print(f"  [checkpoint] {out_csv} ({len(rows)} rows)", flush=True)

    print(f"\nDone. Results saved: {out_csv} ({len(rows)} rows total)", flush=True)


if __name__ == "__main__":
    main()
