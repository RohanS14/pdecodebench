"""
build_jul28.py — Build the jul28 dataset release from two independent
sources: Shreya's synthetic-generated snippets (data/newcode_jul28.txt,
parsed by parse_newcode.py) and the fixed/expanded human-generated snippets
(data/Physics_Code_HumanGen.xlsx, parsed by parse_humangen.py).

Each source is fully self-contained: CorrComm/NoComm_CorrVar donor selection
is drawn ONLY from within that source's own 16-sample pool (no donor ever
crosses the human/synthetic boundary) so each _mod file is independently
reproducible and interpretable on its own.

Outputs (all CSV, all under data/):
  synthetic_base_jul28.csv  — 32 rows (16 synthetic gt_samples x Valid/InValid)
  human_base_jul28.csv      — 32 rows (16 human gt_samples x Valid/InValid)
  merged_base_jul28.csv     — 64 rows = concat of the above, + `source` column
  synthetic_mod_jul28.csv   — 128 rows (16 x 8 mod_types, own-pool donors)
  human_mod_jul28.csv       — 128 rows (16 x 8 mod_types, own-pool donors)
  merged_mod_jul28.csv      — 256 rows = concat of the above, + `source` column
"""

import re
import sys

import pandas as pd

sys.path.insert(0, "datagen")
import parse_newcode
import parse_humangen
from corrupt_comment import generate_corrcomm_rows
from augment_foobar_vars import generate_foobar_rows
from audit_dataset import audit

BASE_COL_ORDER = [
    "title", "gt_sample", "pde_class", "phys_process", "phys_valid",
    "num_method", "num_lines", "num_char", "num_comments",
    "invalidity_note", "code",
]

MOD_COL_ORDER = [
    "title", "code", "num_lines", "num_char",
    "pde_class", "phys_process", "phys_valid", "num_method",
    "corruption_source_id", "corruption_source_pde",
    "injected_comments", "delta_comments", "num_comments",
    "gt_sample", "mod_type", "invalidity_note",
]


def _finalize(rows: list[dict], col_order: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in col_order:
        if col not in df.columns:
            df[col] = pd.NA
    # `title` is the tiebreaker: base files have no `mod_type`, so they'd sort on
    # `gt_sample` alone -- two tied rows per value, and pandas' default quicksort
    # isn't stable, so the Valid/InValid pair order shuffled between runs.
    sort_keys = [c for c in ["gt_sample", "mod_type"] if c in col_order] + ["title"]
    return df[col_order].sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


def _recompute_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """num_lines/num_char/num_comments are never trusted from source; recompute from code."""
    df = df.copy()
    df["num_lines"] = df["code"].map(lambda c: len(str(c).split("\n")))
    df["num_char"] = df["code"].map(lambda c: len(str(c)))
    df["num_comments"] = df["code"].map(
        lambda c: sum(1 for l in str(c).split("\n") if l.strip().startswith("#"))
    )
    return df


# Cosmetic drift between Heat_3's valid and invalid variants, present in the source
# workbook. Its real physical error is the Toeplitz stencil sign
# (toeplitz([-2.0, 1.0, ...]) -> toeplitz([-2.0, -1.0, ...])), which is deliberately NOT
# touched here. Everything below changes no value: int literals respelled as floats, and
# two boundary constants renamed U0/Un -> B0/Bn. Left alone they make the invalid twin
# differ from the valid one outside the physics, which is exactly the kind of
# label-correlated artifact the audit exists to catch.
_HEAT3_COSMETIC_REPAIRS = [
    ("k = 1.0", "k = 1"),
    ("T0 = 0.0", "T0 = 0"),
    ("B0 = 1.0", "U0 = 1"),
    ("Bn = -1.0", "Un = -1"),
    ("b[0] = B0", "b[0] = U0"),
    ("b[n-1] = Bn", "b[n-1] = Un"),
]


# Five synthetic samples widen a snapshot guard in the invalid variant only:
# `if n % 10 == 0:` becomes `if n < 30 or n % 10 == 0:`. That is not the physical error --
# each of the five has a single sign flip that is. It is instrumentation the author added
# knowing the run diverges early, so it correlates perfectly with the label (5/5 invalid,
# 0/5 valid) and, unlike a comment leak, survives every surface condition including
# obfuscation, where it reads `if foobar_15 < 30 or foobar_15 % 10 == 0:`.
#
# Aligning the guard to the valid twin is behavior-preserving, not merely harmless: the
# list each guard gates is appended to and never read (parse_newcode.py strips the plotting
# code that consumed it), verified by AST over all five -- 0 genuine Load references, the
# `.append` receiver excluded. No computed value, stored array or execution outcome moves.
_SNAPSHOT_GUARD_REPAIRS = {
    "Burgers_6":       ("if n < 30 or n % 10 == 0:",  "if n % 10 == 0:"),
    "Burgers_7":       ("if n < 40 or n % 13 == 0:",  "if n % 13 == 0:"),
    "Burgers_8":       ("if n < 40 or n % 2 == 0:",   "if n % 2 == 0:"),
    "NavierStokes_5":  ("if n < 60 or n % 300 == 0:", "if n % 300 == 0:"),
    "NavierStokes_7":  ("if n < 30 or n % 62 == 0:",  "if n % 62 == 0:"),
}

# Heat_2's invalid variant refines the grid tenfold (n = 100 -> n = 1000) alongside its
# intended sign flip, `np.matmul(A, u) + b` -> `np.matmul(A, -u) - b`. The refinement is
# not part of the error and is not needed to produce it: the flip negates the whole RHS,
# so A's eigenvalues turn positive and the solution grows exponentially. That is
# ill-posedness (anti-diffusion), not a stability threshold, so it is grid-independent.
#
# Measured on the live data, aligning n back to 100 strengthens the documented failure
# mode rather than weakening it. Against the note "Has spiking negative temperatures at
# u[1]": at n=1000, u[:,1] is negative on 1 of 300 timesteps and 51 cells go negative; at
# n=100 it is negative on 45 timesteps and 4465 cells go negative. The tenfold refinement
# makes the system 100x stiffer (the stencil scales as alpha/dx^2 with dx = 1/n), which
# pushes LSODA into tiny steps and truncates the very spiking the note describes.
_HEAT2_GRID_REPAIR = ("n = 1000", "n = 100")

# The six samples carrying a label-correlated artifact outside their physics. Both states
# of each are shipped: the repaired one in the canonical files, and both side by side in
# data/leak_ablation_jul28.csv, so the leak can be measured rather than only removed.
_LEAK_AFFECTED = set(_SNAPSHOT_GUARD_REPAIRS) | {"Heat_2"}


def normalize_source_defects(core_df: pd.DataFrame, apply_leak_repairs: bool = True) -> pd.DataFrame:
    """Repairs that must land before any condition is derived.

    `apply_leak_repairs=False` skips repairs 4 and 5 only, reproducing the pre-repair code
    so the ablation file can carry both states of each affected sample. Everything else --
    whitespace, Heat_3, num_method -- still applies, so the two states differ in nothing
    but the leak itself.

    1. Trailing whitespace. Insignificant to Python but not to a diff: it made Wave_1's
       valid and invalid twins differ on two lines that are otherwise identical. Present
       in 54 rows across 9 human samples, so it is stripped everywhere rather than patched
       at the one site where it happened to matter.
    2. Heat_3's cosmetic valid/invalid drift (see _HEAT3_COSMETIC_REPAIRS).
    3. num_method token order. `spectral/explicit` and `explicit/spectral` name the same
       pair but compare as different strings, and donor eligibility is a string comparison,
       so the unnormalised form let a donor pass the "different method" test while sharing
       the receiver's actual method set. Tokens are sorted into a canonical order.

       Note this reshuffles CorrComm donor assignments: the candidate pool changes, and
       donors are drawn from it by seeded RNG. The constraints still hold and no jul28
       results exist yet, but donor identities are not comparable with earlier builds.
       Any dominant-method-first meaning in the original ordering is not preserved -- it
       was not encoded consistently (only one value of seven was out of order).
    4. The snapshot-guard leak in five synthetic samples (see _SNAPSHOT_GUARD_REPAIRS).
    5. Heat_2's grid refinement (see _HEAT2_GRID_REPAIR).

    Every repair here is asserted to have fired. These are literal substitutions against
    source text, so a source edit that changes the spelling would otherwise turn one into
    a silent no-op -- and a no-op looks exactly like a clean dataset.
    """
    df = core_df.copy()

    df["code"] = df["code"].map(
        lambda c: "\n".join(l.rstrip() for l in str(c).split("\n"))
    )

    is_heat3 = (df["gt_sample"] == "Heat_3") & (~df["phys_valid"].astype(bool))
    if is_heat3.any():
        def repair(code):
            for old, new in _HEAT3_COSMETIC_REPAIRS:
                code = code.replace(old, new)
            return code
        df.loc[is_heat3, "code"] = df.loc[is_heat3, "code"].map(repair)

    if not apply_leak_repairs:
        df["num_method"] = df["num_method"].map(
            lambda v: "/".join(sorted(str(v).split("/"))) if pd.notna(v) else v
        )
        return _recompute_metadata(df)

    invalid = ~df["phys_valid"].astype(bool)

    for gt, (widened, aligned) in _SNAPSHOT_GUARD_REPAIRS.items():
        rows = invalid & (df["gt_sample"] == gt)
        if not rows.any():
            continue
        hits = df.loc[rows, "code"].str.count(re.escape(widened))
        if not (hits == 1).all():
            raise RuntimeError(
                f"normalize_source_defects: expected exactly one {widened!r} in each "
                f"invalid {gt} row, found counts {sorted(hits.unique())}"
            )
        df.loc[rows, "code"] = df.loc[rows, "code"].str.replace(widened, aligned, regex=False)

    heat2 = invalid & (df["gt_sample"] == "Heat_2")
    if heat2.any():
        coarse, refined = _HEAT2_GRID_REPAIR
        # anchored: `n = 1000` is a whole assignment line, so a substring match cannot
        # catch a longer name ending in n, or a wider literal like 10000
        pattern = rf"(?m)^{re.escape(coarse)}$"
        hits = df.loc[heat2, "code"].str.count(pattern)
        if not (hits == 1).all():
            raise RuntimeError(
                f"normalize_source_defects: expected exactly one `{coarse}` assignment in "
                f"each invalid Heat_2 row, found counts {sorted(hits.unique())}"
            )
        df.loc[heat2, "code"] = df.loc[heat2, "code"].str.replace(pattern, refined, regex=True)

    df["num_method"] = df["num_method"].map(
        lambda v: "/".join(sorted(str(v).split("/"))) if pd.notna(v) else v
    )

    return _recompute_metadata(df)


def normalize_comm_invalid(core_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every Comm_InValid row as (Comm_Valid's comments) + (NoComm_InValid's code).

    Comm_InValid's whole purpose is "correct comments, broken code": the comments state
    the intent and the code fails to implement it. That only holds if the comments are
    *inherited from the valid variant* rather than authored alongside the invalid code.
    Where they were authored alongside it, the author's knowledge of the failure leaks in
    -- five synthetic samples had comments like "capture densely early since the blow-up
    is very fast", which announce the label instead of describing the intent.

    This applies build_comm_invalid.py's position-based injection uniformly to all 32
    gt_samples. It was already the protocol for the human Wave/Heat half (8/8 comments
    identical to Comm_Valid); the other 24 took Comm_InValid as given from source and so
    drifted. Runs BEFORE generate_corrcomm_rows, because CorrComm_Invalid uses
    Comm_InValid's code as its base.

    Note this can leave a comment describing code the invalid variant removed (Burgers_4's
    "# Lax-Friedrichs dissipation"). That is on-design, not a defect: the comment states
    what the code should do, and the code no longer does it.
    """
    from build_comm_invalid import build_comm_invalid

    derived = build_comm_invalid(core_df)
    if derived.empty:
        raise RuntimeError("normalize_comm_invalid: derived no Comm_InValid rows")

    expected = set(core_df.loc[core_df["mod_type"] == "Comm_InValid", "gt_sample"])
    got = set(derived["gt_sample"])
    if got != expected:
        raise RuntimeError(
            f"normalize_comm_invalid: expected Comm_InValid for {sorted(expected)}, "
            f"derived {sorted(got)}"
        )

    kept = core_df[core_df["mod_type"] != "Comm_InValid"]
    out = pd.concat([kept, derived[core_df.columns]], ignore_index=True)
    return out.sort_values(["gt_sample", "mod_type"]).reset_index(drop=True)


def build_source(core_rows: list[dict], build_base_fn,
                 apply_leak_repairs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """core_rows: 64 rows (16 gt_samples x 4 core mod_types). Returns
    (base_df [32 rows], mod_df [128 rows, own-16-pool donors])."""
    core_df = _finalize(core_rows, MOD_COL_ORDER)
    core_df = normalize_source_defects(core_df, apply_leak_repairs=apply_leak_repairs)
    core_df = normalize_comm_invalid(core_df)

    # base files carry the Comm_Valid/Comm_InValid pair, so they must be built from the
    # normalized rows too, not from the raw source ones
    base_rows = build_base_fn(core_df.to_dict("records"))
    base_df = _finalize(base_rows, BASE_COL_ORDER)

    corrcomm_rows = generate_corrcomm_rows(core_df)
    foobar_rows = generate_foobar_rows(core_df)
    new_rows_df = pd.DataFrame(corrcomm_rows + foobar_rows)
    for col in MOD_COL_ORDER:
        if col not in new_rows_df.columns:
            new_rows_df[col] = pd.NA

    mod_df = pd.concat([core_df[MOD_COL_ORDER], new_rows_df[MOD_COL_ORDER]], ignore_index=True)
    mod_df = mod_df.sort_values(["gt_sample", "mod_type"]).reset_index(drop=True)
    # single source of truth for the three derived counts, after every transform has run
    return _recompute_metadata(base_df), _recompute_metadata(mod_df)


def build_leak_ablation(merged_mod: pd.DataFrame, unrepaired_mod: pd.DataFrame) -> pd.DataFrame:
    """Pair each affected invalid row with its pre-repair self, for measuring the leak.

    `aligned` rows are lifted from the canonical file rather than rebuilt, so the ablation
    cannot silently disagree with the dataset it is an ablation of. The two states are
    asserted to differ in `code` and nothing else -- same donors, same labels, same
    metadata -- so a model's accuracy gap between them is attributable to the leak alone.
    """
    key = ["gt_sample", "mod_type"]
    sel = lambda df: (df["gt_sample"].isin(_LEAK_AFFECTED)) & (~df["phys_valid"].astype(bool))

    aligned = merged_mod[sel(merged_mod)].copy()
    widened = unrepaired_mod[sel(unrepaired_mod)].copy()
    if not (len(aligned) == len(widened) == 4 * len(_LEAK_AFFECTED)):
        raise RuntimeError(
            f"build_leak_ablation: expected {4 * len(_LEAK_AFFECTED)} rows per variant, "
            f"got aligned={len(aligned)} widened={len(widened)}"
        )

    a = aligned.set_index(key).sort_index()
    w = widened.set_index(key).sort_index()
    if list(a.index) != list(w.index):
        raise RuntimeError("build_leak_ablation: variant row sets differ")
    if (a["code"] == w["code"]).any():
        same = [i for i in a.index if a.loc[i, "code"] == w.loc[i, "code"]]
        raise RuntimeError(f"build_leak_ablation: no leak difference on {same}")
    for col in [c for c in a.columns if c not in ("code", "num_lines", "num_char", "num_comments")]:
        if not a[col].fillna("~").eq(w[col].fillna("~")).all():
            raise RuntimeError(
                f"build_leak_ablation: variants differ in {col!r}; the pair must differ "
                f"in code only or the comparison is confounded"
            )

    out = pd.concat([w.reset_index().assign(leak_variant="widened"),
                     a.reset_index().assign(leak_variant="aligned")], ignore_index=True)
    return out.sort_values(["gt_sample", "mod_type", "leak_variant"],
                           kind="mergesort").reset_index(drop=True)


def report_comment_stats(human_mod: pd.DataFrame, synthetic_mod: pd.DataFrame) -> None:
    h = human_mod[human_mod["mod_type"] == "Comm_Valid"]["num_comments"]
    s = synthetic_mod[synthetic_mod["mod_type"] == "Comm_Valid"]["num_comments"]
    print("\n--- Comment count comparison (Comm_Valid rows) ---")
    print(f"  human:     n={len(h)}  mean={h.mean():.2f}  min={h.min()}  max={h.max()}")
    print(f"  synthetic: n={len(s)}  mean={s.mean():.2f}  min={s.min()}  max={s.max()}")
    ratio = s.mean() / h.mean() if h.mean() else float("inf")
    if ratio > 1.5 or ratio < 1 / 1.5:
        print(f"  FLAG: means differ by {ratio:.2f}x -- CorrComm donor comment counts are "
              f"systematically different in scale between the two groups (each group's donor "
              f"pool is still internally consistent, since donors never cross groups, but this "
              f"is worth knowing if you compare CorrComm behavior across sources).")
    else:
        print(f"  OK: comparable comment density between groups (ratio {ratio:.2f}x).")


def main():
    print("=== Building synthetic (Shreya's newcode_jul28.txt) ===")
    with open(parse_newcode.TAG_REVIEW_PATH, newline="", encoding="utf-8") as f:
        import csv
        tag_rows = list(csv.DictReader(f))
    synthetic_core = parse_newcode.get_new_base_rows(tag_rows)
    synthetic_base, synthetic_mod = build_source(synthetic_core, parse_newcode.build_base_rows)
    print(f"  synthetic_base: {len(synthetic_base)} rows")
    print(f"  synthetic_mod:  {len(synthetic_mod)} rows")

    print("\n=== Building human (Physics_Code_HumanGen.xlsx) ===")
    human_core = parse_humangen.build_core_mod_rows()
    human_base, human_mod = build_source(human_core, parse_humangen.build_base_rows)
    print(f"  human_base: {len(human_base)} rows")
    print(f"  human_mod:  {len(human_mod)} rows")

    print("\n=== Merging ===")
    synthetic_base = synthetic_base.assign(source="synthetic")
    human_base = human_base.assign(source="human")
    synthetic_mod = synthetic_mod.assign(source="synthetic")
    human_mod = human_mod.assign(source="human")

    merged_base = pd.concat([human_base, synthetic_base], ignore_index=True)
    merged_mod = pd.concat([human_mod, synthetic_mod], ignore_index=True)
    print(f"  merged_base: {len(merged_base)} rows")
    print(f"  merged_mod:  {len(merged_mod)} rows")

    print("\n=== Building leak ablation (rebuild with repairs 4-5 disabled) ===")
    _, syn_unrep = build_source(synthetic_core, parse_newcode.build_base_rows,
                                apply_leak_repairs=False)
    _, hum_unrep = build_source(human_core, parse_humangen.build_base_rows,
                                apply_leak_repairs=False)
    unrepaired_mod = pd.concat([hum_unrep.assign(source="human"),
                                syn_unrep.assign(source="synthetic")], ignore_index=True)
    leak_ablation = build_leak_ablation(merged_mod, unrepaired_mod)
    print(f"  leak_ablation: {len(leak_ablation)} rows "
          f"({len(_LEAK_AFFECTED)} samples x 4 invalid conditions x 2 variants)")

    report_comment_stats(human_mod, synthetic_mod)

    print("\n--- Running audit on merged_mod ---")
    audit(merged_mod)

    # drop the source column before writing the non-merged files (they're
    # each single-source, so the column would be constant/redundant there)
    synthetic_base = synthetic_base.drop(columns=["source"])
    human_base = human_base.drop(columns=["source"])
    synthetic_mod = synthetic_mod.drop(columns=["source"])
    human_mod = human_mod.drop(columns=["source"])

    outputs = {
        "data/synthetic_base_jul28.csv": synthetic_base,
        "data/human_base_jul28.csv": human_base,
        "data/merged_base_jul28.csv": merged_base,
        "data/synthetic_mod_jul28.csv": synthetic_mod,
        "data/human_mod_jul28.csv": human_mod,
        "data/merged_mod_jul28.csv": merged_mod,
        "data/leak_ablation_jul28.csv": leak_ablation,
    }
    print("\n--- Saving ---")
    for path, df in outputs.items():
        df.to_csv(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    main()
