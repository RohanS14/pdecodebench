"""
build_multimodal_items.py — builds the item set for the cross-modal consistency
experiment (plan Part III).

Each item shows a model four independent representations of one physical system --
solver code, symbolic equation, numerical trajectory, natural-language description
-- in randomized slot order behind a neutral legend, and asks whether they agree
and if not which one is the odd view out. Either all four agree, or exactly one is
corrupted and the other three form the majority that determines the answer.

The item set is a MANIFEST, not materialized text: one row per (system, condition,
name level, order seed) carrying identifiers, ground truth, and covariates. The
view text is materialized at prompt-build time by cross_modal_consistency/eval/consistency_prompts.py.
Two reasons: the rendered trajectory tables run 8-25 KB each, so inlining them
would make this file hundreds of MB for no gain; and it keeps rendering in exactly
one place, which is what stops the four corruption rungs from being distinguishable
by their formatting.

The code views come from the jul28 benchmark rather than from the new CSV. That is
sound because the two agree: all 64 rows of
data/multimodality_physics_with_trajectories.csv map 1:1 onto
data/merged_base_jul28.csv -- 40 code bodies byte-identical, the other 24 identical
once comments and blank lines are stripped. Taking them from merged_mod_jul28.csv
buys the comment-free requirement (the spec wants natural language in exactly one
view) and the AST-obfuscated identifier variants for the lexical factor, both
already audited at 49/49 by datagen/full_audit.py.

Determinism: seeded from names throughout, so consecutive runs produce a
byte-identical CSV, matching datagen/build_jul28.py's contract.

Usage:
    python cross_modal_consistency/datagen/build_multimodal_items.py --out data/multimodal_items_v1.csv
"""
import argparse
import csv
import hashlib
import os
import re
import sys

import numpy as np

# repo root: this file sits at cross_modal_consistency/<area>/, so three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from cross_modal_consistency.datagen.corrupt_trajectory import build_ladder                    # noqa: E402
from cross_modal_consistency.datagen.render_trajectory_table import (                          # noqa: E402
    RENDER_HARD_CASES, choose_grid, parse_trajectory, reconstruction_error,
)

csv.field_size_limit(10 ** 9)

MULTIMODAL_CSV = "data/multimodality_physics_with_trajectories.csv"
MOD_DATASET = "data/merged_mod_jul28.csv"
EQUATIONS = "data/equations_jul28.csv"

VIEWS = ("code", "math", "trajectory", "description")

# One all-agree control plus one condition per corrupted view. The trajectory view
# gets four, because the delivered corruption is far grosser than the other three
# receive -- see cross_modal_consistency/datagen/corrupt_trajectory.py.
CONDITIONS = (
    ("A0", None, "valid"),
    ("X_C", "code", "valid"),
    ("X_M", "math", "valid"),
    ("X_D", "description", "valid"),
    ("X_T_rand", "trajectory", "T_rand"),
    ("X_T_shuf", "trajectory", "T_shuf"),
    ("X_T_swap", "trajectory", "T_swap"),
    ("X_T_exec", "trajectory", "T_exec"),
)

NAME_LEVELS = ("real", "obfuscated")
ORDER_SEEDS = (0, 1)

MOD_FOR_NAMES = {
    ("real", True): "NoComm_Valid",
    ("real", False): "NoComm_InValid",
    ("obfuscated", True): "NoComm_CorrVar",
    ("obfuscated", False): "NoComm_CorrVar_InValid",
}

# These leak their own identity through the code view in EVERY condition, so they
# cannot support the obfuscation contrast. Derived by inspecting the imports and
# string literals that survive comment stripping and AST renaming, not assumed:
#
#   NavierStokes_4  `from jax_cfd.base import ...`   -- "cfd" names the domain
#   NavierStokes_3  `from mpi4py_fft import PFFT`    -- FFT names the method
#   Heat_4          `print(f'CFL: {...} < 0.5')`     -- a Courant condition implies
#                                                       an explicit scheme
#
# No renaming hides an import path or a string literal. Flagged and held out of the
# obfuscation-specific analysis, not dropped from the experiment.
#
# Correction on record: an earlier note listed Wave_1 and Wave_2 here on the
# strength of 'Please choose a correct boundary condition'. That literal is generic
# to any PDE and names neither class nor method, so both are clean; Heat_4, which
# the note omitted, is the real third case.
IDENTITY_LEAK = ("Heat_4", "NavierStokes_3", "NavierStokes_4")

# Systems sharing a math or description view with another system, so they carry
# less instance-level information than the count of 32 suggests. Heat_1 and Heat_3
# share both and differ only in t_steps; Burgers_1-4 share one description.
NEAR_DUPLICATE = ("Heat_1", "Heat_3", "Burgers_1", "Burgers_2", "Burgers_3", "Burgers_4")


def canonical(name):
    """'Navier_Stokes_2 ' -> 'NavierStokes_2'.

    The new CSV writes Navier_Stokes_* where the rest of the repo writes
    NavierStokes_*, and 'Wave_8 ' carries a trailing space. Both silently break a
    naive join.
    """
    return name.strip().replace("Navier_Stokes", "NavierStokes")


def blank_line_runs(code):
    """Count runs of blank lines -- a surface cue that separates wrong code from
    valid in roughly 20 of 32 pairs. Recorded rather than repaired (the data is
    used as delivered), then partialled out in the analysis."""
    return len(re.findall(r"\n[ \t]*\n", code))


def slot_order(system_index, system, condition, seed, corrupted_view):
    """Assign the four views to slots 1-4.

    Position is a factor this experiment measures -- whether a model's choice of
    outlier tracks where the view sits -- so the corrupted view's slot is
    COUNTERBALANCED rather than drawn freely. A free permutation left the outlier
    in slot 1 on 184 items and slot 4 on 256, a spread wide enough to confound the
    position contrast it was supposed to support.

    The corrupted view lands in slot (system_index + 2*seed) mod 4, so within each
    (condition, seed) the 32 systems distribute exactly 8 to each slot, and each
    system sees two different slots across the two seeds. The remaining three views
    fill the other slots under a seed derived from the names, keeping the layout
    reproducible and stable against unrelated systems changing.
    """
    key = f"{system}::{condition}::{seed}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    if corrupted_view is None:
        return [VIEWS[i] for i in rng.permutation(len(VIEWS))]

    target = (system_index + 2 * seed) % len(VIEWS)
    others = [v for v in VIEWS if v != corrupted_view]
    others = [others[i] for i in rng.permutation(len(others))]
    slots = []
    for i in range(len(VIEWS)):
        slots.append(corrupted_view if i == target else others.pop(0))
    return slots


def load_multimodal(path):
    rows = {}
    for r in csv.DictReader(open(path, newline="")):
        rows[canonical(r["Example Name"])] = r
    return rows


def load_code_variants(path):
    """{gt_sample: {mod_type: code}} for the four comment-free conditions."""
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        if r["mod_type"] in set(MOD_FOR_NAMES.values()):
            out.setdefault(r["gt_sample"], {})[r["mod_type"]] = r["code"]
    return out


def load_meta(path):
    """Ground-truth PDE class and numerical method, plus the invalidity note used
    later to grade the model's free-text justification."""
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        if r["mod_type"] == "NoComm_Valid":
            out[r["gt_sample"]] = {
                "pde_class": r["pde_class"],
                "num_method": r["num_method"],
                "phys_process": r["phys_process"],
                "source": r["source"],
            }
        if r["mod_type"] == "NoComm_InValid":
            out.setdefault(r["gt_sample"], {})["invalidity_note"] = r["invalidity_note"]
    return out


FIELDNAMES = [
    "item_id", "gt_sample", "pde_class", "num_method", "phys_process", "source",
    "condition", "corrupted_view", "traj_level",
    "names", "order_seed",
    "slot_1", "slot_2", "slot_3", "slot_4", "outlier_slot",
    "gt_pde_class", "gt_num_method", "invalidity_note",
    "desc_len_valid", "desc_len_wrong", "desc_len_delta",
    "code_len_valid", "code_len_wrong", "code_len_delta",
    "code_blank_runs_valid", "code_blank_runs_wrong",
    "traj_shape_valid", "traj_shape_swap", "render_grid", "render_recon_err",
    "flag_render_hard", "flag_identity_leak", "flag_near_duplicate",
    "needs_exec",
]


def build(mm_path, mod_path, out_path, include_time_shuffle=False):
    mm = load_multimodal(mm_path)
    codes = load_code_variants(mod_path)
    meta = load_meta(mod_path)

    systems = sorted(k for k in mm if not k.endswith("_wrong"))
    conditions = list(CONDITIONS)
    if include_time_shuffle:
        conditions.append(("X_T_timeshuf", "trajectory", "T_timeshuf"))

    rows = []
    for system_index, system in enumerate(systems):
        valid_row, wrong_row = mm[system], mm[system + "_wrong"]
        valid_traj = parse_trajectory(valid_row["Trajectory"])
        grid = choose_grid(valid_traj.shape)
        ladder = build_ladder(valid_traj, wrong_row["Trajectory"], system,
                              include_time_shuffle=include_time_shuffle)
        m = meta.get(system, {})

        cov = {
            "desc_len_valid": len(valid_row["Written Description"]),
            "desc_len_wrong": len(wrong_row["Written Description"]),
            "code_len_valid": len(valid_row["Code"]),
            "code_len_wrong": len(wrong_row["Code"]),
            "code_blank_runs_valid": blank_line_runs(valid_row["Code"]),
            "code_blank_runs_wrong": blank_line_runs(wrong_row["Code"]),
            "traj_shape_valid": "x".join(map(str, valid_traj.shape)),
            "traj_shape_swap": "x".join(map(str, ladder["T_swap"].shape)),
            "render_grid": "x".join(map(str, grid)),
            "render_recon_err": round(reconstruction_error(valid_traj, grid), 4),
        }
        cov["desc_len_delta"] = cov["desc_len_wrong"] - cov["desc_len_valid"]
        cov["code_len_delta"] = cov["code_len_wrong"] - cov["code_len_valid"]

        for condition, corrupted_view, traj_level in conditions:
            for names in NAME_LEVELS:
                for seed in ORDER_SEEDS:
                    slots = slot_order(system_index, system, condition, seed,
                                       corrupted_view)
                    outlier = "" if corrupted_view is None else slots.index(corrupted_view) + 1
                    rows.append({
                        "item_id": f"{system}|{condition}|{names}|{seed}",
                        "gt_sample": system,
                        "pde_class": m.get("pde_class", ""),
                        "num_method": m.get("num_method", ""),
                        "phys_process": m.get("phys_process", ""),
                        "source": m.get("source", ""),
                        "condition": condition,
                        "corrupted_view": corrupted_view or "none",
                        "traj_level": traj_level,
                        "names": names,
                        "order_seed": seed,
                        "slot_1": slots[0], "slot_2": slots[1],
                        "slot_3": slots[2], "slot_4": slots[3],
                        "outlier_slot": outlier,
                        "gt_pde_class": m.get("pde_class", ""),
                        "gt_num_method": m.get("num_method", ""),
                        "invalidity_note": m.get("invalidity_note", ""),
                        "flag_render_hard": int(system in RENDER_HARD_CASES),
                        "flag_identity_leak": int(system in IDENTITY_LEAK),
                        "flag_near_duplicate": int(system in NEAR_DUPLICATE),
                        # T_exec is the one rung that cannot be built from the CSV;
                        # it is merged in after the cpu_short re-execution job.
                        "needs_exec": int(traj_level == "T_exec"),
                        **cov,
                    })

    rows.sort(key=lambda r: (r["gt_sample"], r["condition"], r["names"], r["order_seed"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    missing = [s for s in systems if len(codes.get(s, {})) != 4]
    return rows, missing


def main():
    p = argparse.ArgumentParser(description="Build the cross-modal consistency item set")
    p.add_argument("--multimodal", default=MULTIMODAL_CSV)
    p.add_argument("--dataset", default=MOD_DATASET)
    p.add_argument("--out", default="data/multimodal_items_v1.csv")
    p.add_argument("--time_shuffle", action="store_true",
                   help="add the T_timeshuf rung (frame permutation)")
    args = p.parse_args()

    rows, missing = build(args.multimodal, args.dataset, args.out, args.time_shuffle)
    print(f"[build] {len(rows)} items -> {args.out}")
    if missing:
        print(f"[build] WARNING: {len(missing)} systems lack all four NoComm variants: {missing}")


if __name__ == "__main__":
    main()
