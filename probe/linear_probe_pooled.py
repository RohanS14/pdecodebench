"""
Linear probe — all 96 rows, LOGO-CV by gt_sample.

Answers: which layers encode PDE properties; does test accuracy drop for
corrupted mod_types in the test folds?

Usage:
    python probe/linear_probe_pooled.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --output_dir probe/results/ \
        --pool mean_pool

Outputs:
    probe/results/probe_pooled_{pool}.csv
    probe/figures/probe_pooled_acc_vs_layer_{label}_{pool}.png
    probe/figures/probe_pooled_modtype_{pool}.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from probe_utils import (
    load_data, extract_label_arrays, run_logo_probe, bootstrap_ci,
    PDE_CLASSES, MOD_TYPES, BINARY_PROCESS_LABELS, BINARY_METHOD_LABELS,
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


def run_bow_probe(codes, y, groups, mod_types):
    """TF-IDF bag-of-words baseline with same LOGO-CV."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        analyzer="word",
        token_pattern=r"[A-Za-z_]\w*",
    )
    logo = LeaveOneGroupOut()
    mt_true_pred = {mt: ([], []) for mt in MOD_TYPES}
    all_true, all_pred = [], []

    for train_idx, test_idx in logo.split(np.zeros(len(y)), y, groups):
        codes_train = [codes[i] for i in train_idx]
        codes_test  = [codes[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mt_test = mod_types[test_idx]

        X_train = vectorizer.fit_transform(codes_train).toarray().astype(np.float32)
        X_test  = vectorizer.transform(codes_test).toarray().astype(np.float32)

        n_classes = len(np.unique(y_train))
        if n_classes < 2:
            pred = np.full(len(y_test), y_train[0])
        else:
            clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                     random_state=42)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        for mt in MOD_TYPES:
            mask = mt_test == mt
            if mask.any():
                mt_true_pred[mt][0].extend(y_test[mask].tolist())
                mt_true_pred[mt][1].extend(pred[mask].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = float(np.mean(all_true == all_pred))
    mt_acc = {}
    for mt in MOD_TYPES:
        t, p = mt_true_pred[mt]
        mt_acc[mt] = float(np.mean(np.array(t) == np.array(p))) if t else float("nan")
    return acc, mt_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--output_dir", default="probe/results/")
    parser.add_argument("--pool", default="mean_pool",
                        choices=["mean_pool", "last_tok"])
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic regression regularization strength")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = args.output_dir.replace("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data = load_data(args.hidden)
    reps = data[args.pool].astype(np.float32)   # (N, L, D)
    N, L, D = reps.shape
    groups    = data["gt_samples"]
    mod_types = data["mod_types"]

    # Raw codes for BoW baseline — stored in npz by extract_hidden.py
    if "codes" in data:
        raw_codes = [str(c) for c in data["codes"]]
    else:
        # fallback: try loading from dataset
        raw_codes = None
        try:
            df_raw = pd.read_excel("data/pdedata_clean_v2.xlsx")
            title_to_code = {str(r["title"]): str(r["code"]) for _, r in df_raw.iterrows()}
            raw_codes = [title_to_code.get(str(t), "") for t in data["titles"]]
        except Exception as e:
            print(f"BoW baseline: could not load raw codes ({e}), skipping BoW.", flush=True)

    label_arrays = extract_label_arrays(data)

    print(f"Probing {N} examples, {L} layers, {D} dim [{args.pool}]", flush=True)
    print(f"Groups (gt_sample): {len(np.unique(groups))} unique", flush=True)
    print(f"Labels: {LABEL_ORDER}", flush=True)

    rows = []

    for label_name in LABEL_ORDER:
        y = label_arrays[label_name]
        print(f"\n--- {label_name} (chance={CHANCE[label_name]:.2f}) ---", flush=True)

        # BoW baseline
        if raw_codes is not None:
            bow_acc, bow_mt = run_bow_probe(raw_codes, y, groups, mod_types)
            print(f"  BoW: {bow_acc:.3f}", flush=True)
            bow_row = {"label": label_name, "layer": "bow", "pool": args.pool,
                       "accuracy": bow_acc, "ci_low": float("nan"), "ci_high": float("nan")}
            bow_row.update({f"mt_{mt}": bow_mt.get(mt, float("nan")) for mt in MOD_TYPES})
            rows.append(bow_row)

        # Per-layer probes
        layer_accs = []
        for l in range(L):
            X = reps[:, l, :]
            res = run_logo_probe(X, y, groups, C=args.C, mod_types=mod_types)
            acc, lo, hi = bootstrap_ci(res["per_fold_acc"])
            valid_aurocs = [a for a in res["per_fold_auroc"] if not np.isnan(a)]
            auroc, auroc_lo, auroc_hi = (
                bootstrap_ci(valid_aurocs) if valid_aurocs else (float("nan"),) * 3
            )
            layer_accs.append(acc)
            row = {"label": label_name, "layer": l, "pool": args.pool,
                   "accuracy": acc, "ci_low": lo, "ci_high": hi,
                   "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi}
            row.update({f"mt_{mt}": res["per_modtype_acc"].get(mt, float("nan"))
                        for mt in MOD_TYPES})
            rows.append(row)

            if l % 5 == 0 or l == L - 1:
                print(f"  Layer {l:2d}: acc={acc:.3f} [{lo:.3f}, {hi:.3f}]  "
                      f"auroc={auroc:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]", flush=True)

        # Plot accuracy vs layer
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(L), layer_accs, marker="o", markersize=4, linewidth=1.5,
                label=f"{label_name} [{args.pool}]")

        # Add BoW line
        if raw_codes is not None:
            ax.axhline(bow_acc, color="orange", linestyle="--", linewidth=1.5,
                       label=f"BoW ({bow_acc:.3f})")

        ax.axhline(CHANCE[label_name], color="gray", linestyle=":", linewidth=1,
                   label=f"Chance ({CHANCE[label_name]:.2f})")
        ax.set_xlabel("Layer (0 = embedding)")
        ax.set_ylabel("LOGO-CV Accuracy")
        ax.set_title(f"Linear probe: {label_name} — all 96 rows [{args.pool}]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(0, 1.05)

        fig_path = os.path.join(fig_dir,
                                f"probe_pooled_acc_vs_layer_{label_name}_{args.pool}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Incremental save after each label
        out_csv = os.path.join(args.output_dir, f"probe_pooled_{args.pool}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  [checkpoint] {out_csv} ({len(rows)} rows)", flush=True)

    # Final save
    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, f"probe_pooled_{args.pool}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}", flush=True)

    # --- Mod-type breakdown plot (best layer per label) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(LABEL_ORDER))
    width = 0.12
    layer_rows = results_df[results_df["layer"] != "bow"].copy()
    layer_rows["layer"] = layer_rows["layer"].astype(int)
    mt_colors = plt.cm.tab10(np.linspace(0, 0.6, len(MOD_TYPES)))

    for mt_idx, mt in enumerate(MOD_TYPES):
        mt_col = f"mt_{mt}"
        vals = []
        for label_name in LABEL_ORDER:
            sub = layer_rows[layer_rows["label"] == label_name]
            best_layer_row = sub.loc[sub["accuracy"].idxmax()]
            vals.append(best_layer_row[mt_col] if mt_col in best_layer_row else float("nan"))
        offset = (mt_idx - len(MOD_TYPES) / 2) * width
        ax.bar(x + offset, vals, width, label=mt, color=mt_colors[mt_idx], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy at best layer")
    ax.set_title(f"Probe accuracy by mod_type (best layer per label) [{args.pool}]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    mt_fig_path = os.path.join(fig_dir, f"probe_pooled_modtype_{args.pool}.png")
    plt.tight_layout()
    plt.savefig(mt_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mod-type breakdown plot: {mt_fig_path}", flush=True)


if __name__ == "__main__":
    main()
