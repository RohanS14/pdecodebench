"""
Shared utilities for linear probe scripts.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

PDE_CLASSES = ["wave", "heat", "burgers", "navier-stokes"]
MOD_TYPES = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
             "Comm_InValid", "NoComm_InValid"]

# Binary label names and how to extract them from raw "/" -separated strings
BINARY_PROCESS_LABELS = ["diffusion", "advection", "oscillation", "restoration"]
BINARY_METHOD_LABELS  = ["explicit", "implicit", "spectral"]


def load_data(npz_path: str) -> dict:
    """Load NPZ and return dict with numpy arrays."""
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def extract_label_arrays(data: dict) -> dict:
    """
    Return a dict of label_name → 1-D numpy array (length N).
    pde_class: int (0-3)
    process_*: binary int
    method_*:  binary int
    phys_valid: binary int
    """
    labels = {}

    # 4-class PDE
    pde_map = {c: i for i, c in enumerate(PDE_CLASSES)}
    labels["pde_class"] = np.array([pde_map.get(p.lower(), -1)
                                    for p in data["pde_classes"]], dtype=np.int32)

    # Binary physical process labels
    for proc in BINARY_PROCESS_LABELS:
        labels[f"process_{proc}"] = np.array(
            [int(proc in str(p).lower().replace("difffusion", "diffusion"))
             for p in data["phys_process"]], dtype=np.int32
        )

    # Binary numerical method labels
    for meth in BINARY_METHOD_LABELS:
        labels[f"method_{meth}"] = np.array(
            [int(meth in str(m).lower()) for m in data["num_method"]], dtype=np.int32
        )

    # Physical validity
    labels["phys_valid"] = data["phys_valid"].astype(np.int32)

    return labels


def run_logo_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float = 1.0,
    mod_types: np.ndarray = None,
) -> dict:
    """
    Leave-One-Group-Out logistic regression probe.

    Returns:
        accuracy       : overall accuracy across all test folds
        per_fold_acc   : list of per-fold accuracy values (for bootstrap CI)
        per_modtype_acc: dict mod_type → accuracy (if mod_types provided)
    """
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    fold_accs = []
    fold_aurocs = []
    mt_true_pred = {mt: ([], []) for mt in MOD_TYPES} if mod_types is not None else None

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        n_classes = len(np.unique(y_train))
        if n_classes < 2:
            pred = np.full(len(y_test), y_train[0])
            fold_aurocs.append(float("nan"))
        else:
            clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs",
                                     random_state=42)
            clf.fit(X_train_s, y_train)
            pred = clf.predict(X_test_s)
            proba = clf.predict_proba(X_test_s)
            try:
                if len(clf.classes_) == 2:
                    fold_auroc = roc_auc_score(y_test, proba[:, 1])
                else:
                    fold_auroc = roc_auc_score(
                        y_test, proba, multi_class="ovr", average="macro"
                    )
            except ValueError:
                fold_auroc = float("nan")  # only 1 class in y_test
            fold_aurocs.append(fold_auroc)

        fold_accs.append(float(np.mean(pred == y_test)))
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

        if mt_true_pred is not None:
            mt_test = mod_types[test_idx]
            for mt in MOD_TYPES:
                mask = mt_test == mt
                if mask.any():
                    mt_true_pred[mt][0].extend(y_test[mask].tolist())
                    mt_true_pred[mt][1].extend(pred[mask].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    result = {
        "accuracy": float(np.mean(all_true == all_pred)),
        "per_fold_acc": fold_accs,
        "per_fold_auroc": fold_aurocs,
    }

    if mt_true_pred is not None:
        mt_acc = {}
        for mt in MOD_TYPES:
            t, p = mt_true_pred[mt]
            mt_acc[mt] = float(np.mean(np.array(t) == np.array(p))) if t else float("nan")
        result["per_modtype_acc"] = mt_acc

    return result


def bootstrap_ci(values: list, n_boot: int = 10000, seed: int = 42,
                 ci: float = 0.95) -> tuple:
    """Bootstrap CI over a list of per-fold accuracy values."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(arr.mean()), lo, hi
