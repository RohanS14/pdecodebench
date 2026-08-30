"""Map the real cross-modal results onto the plotting module's schema.

The eval writes the vocabulary the eval cared about; the plotting module reads the
vocabulary the figures are phrased in. This is the one place that translates, so no
figure has to know that `corrupted_view` says "trajectory" while the blame matrix
axis says "T", or that the real `outlier` field names a SLOT rather than a view.

The slot detail matters and is the reason this file exists rather than a rename dict:
the model answers "view_2", which means whatever sat in position 2 for that item. It
can only be resolved through that row's own `slots` permutation, and getting it wrong
would silently produce a plausible, wrong blame matrix.

`judge_correct` has no counterpart in the real run -- no LLM-judge pass has been run
over the justifications -- so it is left absent rather than invented, and the panels
that need it degrade to "not available" instead of showing a fabricated rate.
"""
import ast

import numpy as np
import pandas as pd

from .constants import CONDITION_OUTLIER, NONE, SCHEMA_COLUMNS

# eval vocabulary -> figure vocabulary
VIEW_CODE = {"code": "C", "trajectory": "T", "description": "D", "math": "M",
             "none": NONE, "": NONE}

# X_* is the eval's condition naming; the figures use A-*, and the trajectory rungs
# carry their level in the name so a per-condition axis shows the ladder.
CONDITION_MAP = {
    "A0": "A0", "X_C": "A-C", "X_D": "A-D", "X_M": "A-M",
    "X_T_rand": "A-T-rand", "X_T_shuf": "A-T-shuf",
    "X_T_swap": "A-T-swap", "X_T_exec": "A-T-exec",
}
TRAJ_MAP = {"T_rand": "rand", "T_shuf": "shuf", "T_swap": "swap",
            "T_exec": "exec", "valid": ""}


def _slots(v):
    """The four view names in presentation order, from whatever the column holds."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v]
    if isinstance(v, str) and v.strip():
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return [p.strip() for p in v.split(",") if p.strip()]
    return []


def _resolve_outlier(pred, slots):
    """Turn the model's slot answer into the view it actually accused.

    "view_3" is meaningless without this row's permutation, and the permutation is
    per item. Resolving it against a fixed order would look fine and be wrong.
    """
    if not isinstance(pred, str) or not pred.strip():
        return ""
    p = pred.strip().lower()
    if p in ("none", "no", "nan"):
        return NONE
    if p.startswith("view_"):
        try:
            i = int(p.split("_", 1)[1]) - 1
        except ValueError:
            return ""
        if 0 <= i < len(slots):
            return VIEW_CODE.get(slots[i], "")
        return ""
    return VIEW_CODE.get(p, "")


def from_xmodal(df, items_csv="data/multimodal_items_v1.csv",
                registry_csv="data/model_registry.csv"):
    """Real results -> the plotting schema. Returns a DataFrame of SCHEMA_COLUMNS."""
    d = df.copy()
    slots = d["slots"].map(_slots)

    out = pd.DataFrame({
        # item_id repeats across arms, so the arm has to be part of the key.
        "run_id": (d["item_id"].astype(str) + "|" + d["model"].astype(str)
                   + "|" + d.get("thinking", "").astype(str)),
        "solver_id": d["gt_sample"].astype(str),
        "condition": d["condition"].map(CONDITION_MAP).fillna(d["condition"]),
        "true_outlier": d["corrupted_view"].astype(str).map(VIEW_CODE).fillna(NONE),
        "traj_level": d.get("traj_level", "").astype(str).map(TRAJ_MAP).fillna(""),
        "naming": d["names"].astype(str),
        "reasoning": d.get("thinking", "").astype(str),
        "model": d["model"].astype(str),
        "order": [",".join(VIEW_CODE.get(x, "?") for x in s) for s in slots],
        "pred_agree": d["agree"].astype(str).str.strip().str.lower(),
        "pred_outlier": [_resolve_outlier(p, s)
                         for p, s in zip(d["outlier"], slots)],
        "pred_pde_class": d.get("system_pde_class", "").astype(str),
        "pred_method": d.get("system_num_method", "").astype(str),
        "justification": d.get("justification", "").astype(str),
    })

    # A row that never produced a parseable verdict is not a "yes"; leave it as the
    # empty string so prepare() reads it as un-flagged rather than as agreement.
    out["pred_agree"] = out["pred_agree"].where(
        out["pred_agree"].isin(["yes", "no"]), "")

    # Ground-truth physics comes from the item table, not from the model's answer.
    try:
        items = pd.read_csv(items_csv)
        key = "gt_sample"
        cols = {c: c for c in ("pde_class", "num_method") if c in items.columns}
        if key in items.columns and cols:
            lut = items.drop_duplicates(key).set_index(key)
            out["pde_class"] = out["solver_id"].map(lut.get("pde_class", pd.Series(dtype=str)))
            out["numerical_method"] = out["solver_id"].map(
                lut.get("num_method", pd.Series(dtype=str)))
    except (FileNotFoundError, ValueError):
        pass
    for c in ("pde_class", "numerical_method"):
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("")

    # Model metadata -- release date, size, family -- for the generational figure.
    # A results row cannot carry these: it records what the model said, not when the
    # model shipped. Missing rows are left empty rather than guessed, so a model
    # absent from the registry drops out of the trend figure instead of being
    # plotted at a fabricated date.
    try:
        reg = pd.read_csv(registry_csv)
        if "model_id" in reg.columns:
            lut = reg.drop_duplicates("model_id").set_index("model_id")
            for col in ("release_date", "params_total_b", "family"):
                if col in lut.columns:
                    out[col] = out["model"].map(lut[col])
    except (FileNotFoundError, ValueError):
        pass
    for c in ("release_date", "family"):
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("")
    if "params_total_b" not in out:
        out["params_total_b"] = float("nan")
    out["params_total_b"] = pd.to_numeric(out["params_total_b"], errors="coerce")

    # judge_correct is deliberately NOT created. See the module docstring.
    return out.reindex(columns=[c for c in SCHEMA_COLUMNS if c != "judge_correct"])


# One repo per model for the generational expansion, NOT one shared repo.
# push_dataset_to_hub REPLACES a split rather than appending, and upload_helper globs
# a whole results dir, so N concurrent jobs sharing either would clobber each other's
# uploads and the artifact could shrink mid-run. Per-model repos let all six run at
# once with partial uploads intact and zero contention. Missing repos are skipped, so
# this list can name models that have not been run yet.
# The ORIGINAL 4,096-row, 28-column artifact backing viz/consistency_claims.html:
# 3 models (Qwen3-32B think-on and think-off, QwQ-32B, R1-Distill-32B). Renamed
# 2026-08-25 with an explicit -frozen-v1 suffix, because the unsuffixed name was
# indistinguishable from the 8-model consolidated roster and a blanket rename
# silently repointed this constant at that roster -- exactly the substitution the
# GENERATIONAL_REPOS comment below warns against.
FROZEN_REPO = "bermaneh/pde-llm-eval-xmodal-consistency-frozen-v1"

# The generational roster, CONSOLIDATED 2026-08-25 into a single repo carrying `model`
# as a column. It was ten per-model repos while the campaign ran, because concurrent
# jobs each uploaded their own arm and push_dataset_to_hub REPLACES a split; those ten
# have been deleted and every row they held is in the repo below. It still does NOT
# replace the frozen artifact -- build_cross_modal_claims_frozen.sh stays pinned to FROZEN_REPO, and
# folding these eight models into that report would rewrite published pooled panels.
GENERATIONAL_REPOS = (
    "bermaneh/pde-llm-eval-cross-modal-consistency",
)

# The frozen 4096 rows plus whatever generational arms exist. build_cross_modal_claims_frozen.sh must
# NOT use this default -- it is pinned to FROZEN_REPO alone, or the new models would
# be folded into the published report.
DEFAULT_REPOS = (FROZEN_REPO,) + GENERATIONAL_REPOS


def load_real(repo=DEFAULT_REPOS, **kw):
    """Load one or more result repos and map them onto the plotting schema.

    A repo that does not exist yet is skipped with a warning rather than raising --
    the generational repo has no rows until its first arm uploads, and every figure
    in this package is built during a run, not after it.
    """
    from datasets import load_dataset
    repos = (repo,) if isinstance(repo, str) else tuple(repo)
    frames = []
    for r in repos:
        try:
            frames.append(load_dataset(r, split="train").to_pandas())
        except Exception as exc:                                  # noqa: BLE001
            print(f"[adapter] skipping {r}: {type(exc).__name__}: {exc}")
    if not frames:
        raise RuntimeError(f"no readable result repos among {repos}")
    return from_xmodal(pd.concat(frames, ignore_index=True), **kw)
