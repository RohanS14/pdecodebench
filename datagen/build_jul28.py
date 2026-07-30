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


def normalize_source_defects(core_df: pd.DataFrame) -> pd.DataFrame:
    """Repairs that must land before any condition is derived.

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


def build_source(core_rows: list[dict], build_base_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """core_rows: 64 rows (16 gt_samples x 4 core mod_types). Returns
    (base_df [32 rows], mod_df [128 rows, own-16-pool donors])."""
    core_df = _finalize(core_rows, MOD_COL_ORDER)
    core_df = normalize_source_defects(core_df)
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
