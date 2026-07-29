"""
build_v5.py — Build the v5 dataset by combining the existing 16 base problems
(pdedata_clean_v4.xlsx) with Shreya's 16 new base problems (data/newcode_jul28.txt,
parsed by parse_newcode.py), doubling the base-problem count to 32.

Never overwrites any existing dataset file (pdedata_clean.xlsx, _v2, _v3, _v4,
_v4_base.xlsx, physics_code.xlsx). Writes three new files:

  1. data/pdedata_newcode_v5_base.xlsx  — new material only, 16 samples x 4
     core mod_types (Comm_Valid, NoComm_Valid, Comm_InValid, NoComm_InValid)
     = 64 rows. Lets the new material be inspected in isolation.
  2. data/pdedata_clean_v5_base.xlsx    — combined base, 32 samples x 4 core
     mod_types = 128 rows, pre-augmentation.
  3. data/pdedata_clean_v5.xlsx         — final dataset, 32 samples x 8
     mod_types = 256 rows.

Steps:
  1. Require a reviewed data/descriptions/newcode_v5_tag_review.csv (generated
     by parse_newcode.py) -- refuses to proceed if missing.
  2. Build the 4 core mod_type rows for the 16 new gt_samples via
     parse_newcode.get_new_base_rows(). Save as (1).
  3. Load pdedata_clean_v4.xlsx, keep only its 4 core mod_types (drop its
     CorrComm/NoComm_CorrVar/CorrComm_Invalid/NoComm_CorrVar_InValid rows --
     these get regenerated fresh over the full 32-sample pool below, same
     drop-and-regenerate pattern build_v4.py already uses relative to v4_base).
  4. Concatenate old + new core rows, add the invalidity_note column (new,
     purely additive -- NaN for all old rows). Save as (2).
  5. Run the existing, unmodified corrupt_comment.generate_corrcomm_rows()
     and augment_foobar_vars.generate_foobar_rows() over the combined base,
     regenerating CorrComm/CorrComm_Invalid/NoComm_CorrVar/NoComm_CorrVar_InValid
     across all 32 samples (donor pool = full 32, same seed=42 logic).
  6. Assemble the final 256-row dataframe, backfill invalidity_note onto the
     derived invalid mod_types corrupt_comment.py doesn't propagate it to
     (CorrComm_Invalid), run audit_dataset.audit(), save as (3).
"""

import sys

import pandas as pd

sys.path.insert(0, "datagen")
import parse_newcode
from corrupt_comment import generate_corrcomm_rows
from augment_foobar_vars import generate_foobar_rows
from audit_dataset import audit

CORE_MOD_TYPES = ["Comm_Valid", "NoComm_Valid", "Comm_InValid", "NoComm_InValid"]

COL_ORDER = [
    "title", "code", "num_lines", "num_char",
    "pde_class", "phys_process", "phys_valid", "num_method",
    "corruption_source_id", "corruption_source_pde",
    "injected_comments", "delta_comments", "num_comments",
    "gt_sample", "mod_type", "invalidity_note",
]

V4_PATH = "data/pdedata_clean_v4.xlsx"
NEWCODE_BASE_OUT = "data/pdedata_newcode_v5_base.xlsx"
V5_BASE_OUT = "data/pdedata_clean_v5_base.xlsx"
V5_OUT = "data/pdedata_clean_v5.xlsx"


def main():
    # --- Step 1: require reviewed tag CSV ---
    import os
    if not os.path.exists(parse_newcode.TAG_REVIEW_PATH):
        print(f"'{parse_newcode.TAG_REVIEW_PATH}' does not exist yet.")
        print("Generating it now via parse_newcode.write_tag_review_csv() ...")
        parse_newcode.write_tag_review_csv()
        print("\nSTOP: review the tag CSV (phys_process/num_method/invalidity_note/reasoning,")
        print("especially rows flagged 'NEEDS REVIEW') before re-running build_v5.py.")
        return

    print(f"Loading reviewed tags from {parse_newcode.TAG_REVIEW_PATH} ...")
    import csv
    with open(parse_newcode.TAG_REVIEW_PATH, newline="", encoding="utf-8") as f:
        tag_rows = list(csv.DictReader(f))
    print(f"  {len(tag_rows)} reviewed gt_samples loaded")

    # --- Step 2: build new-only core rows ---
    print("\n--- Parsing data/newcode_jul28.txt into core mod_type rows ---")
    new_rows = parse_newcode.get_new_base_rows(tag_rows)
    new_df = pd.DataFrame(new_rows)
    for col in COL_ORDER:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    new_df = new_df[COL_ORDER].sort_values(["gt_sample", "mod_type"]).reset_index(drop=True)
    print(f"  {len(new_df)} rows, {new_df['gt_sample'].nunique()} new gt_samples")

    new_df.to_excel(NEWCODE_BASE_OUT, index=False)
    print(f"Saved {NEWCODE_BASE_OUT}")

    # --- Step 3: load v4, keep only core mod_types ---
    print(f"\n--- Loading {V4_PATH}, keeping core mod_types ---")
    v4_df = pd.read_excel(V4_PATH)
    old_core_df = v4_df[v4_df["mod_type"].isin(CORE_MOD_TYPES)].copy()
    old_core_df["invalidity_note"] = pd.NA
    for col in COL_ORDER:
        if col not in old_core_df.columns:
            old_core_df[col] = pd.NA
    old_core_df = old_core_df[COL_ORDER]
    print(f"  Kept {len(old_core_df)} old core rows ({old_core_df['gt_sample'].nunique()} gt_samples)")

    # --- Step 4: combine, save base ---
    base_df = pd.concat([old_core_df, new_df[COL_ORDER]], ignore_index=True)
    base_df = base_df.sort_values(["gt_sample", "mod_type"]).reset_index(drop=True)
    print(f"\nCombined base: {len(base_df)} rows, {base_df['gt_sample'].nunique()} gt_samples")

    base_df.to_excel(V5_BASE_OUT, index=False)
    print(f"Saved {V5_BASE_OUT}")

    # --- Step 5: regenerate CorrComm/CorrVar over the full 32-sample pool ---
    print("\n--- Generating CorrComm + CorrComm_Invalid (full 32-sample pool) ---")
    corrcomm_rows = generate_corrcomm_rows(base_df)

    print("\n--- Generating NoComm_CorrVar + NoComm_CorrVar_InValid (full 32-sample pool) ---")
    foobar_rows = generate_foobar_rows(base_df)

    new_mod_rows_df = pd.DataFrame(corrcomm_rows + foobar_rows)
    for col in COL_ORDER:
        if col not in new_mod_rows_df.columns:
            new_mod_rows_df[col] = pd.NA

    # --- Step 6: assemble final, backfill invalidity_note, audit, save ---
    final_df = pd.concat([base_df[COL_ORDER], new_mod_rows_df[COL_ORDER]], ignore_index=True)

    note_by_gt = (
        new_df[new_df["invalidity_note"].notna()]
        .drop_duplicates("gt_sample")
        .set_index("gt_sample")["invalidity_note"]
    )
    fill_mask = final_df["invalidity_note"].isna() & (~final_df["phys_valid"].astype(bool))
    final_df.loc[fill_mask, "invalidity_note"] = final_df.loc[fill_mask, "gt_sample"].map(note_by_gt)

    final_df = final_df.sort_values(["gt_sample", "mod_type"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(final_df)} rows")
    print(final_df["mod_type"].value_counts().to_string())

    print("\n--- Running Audit ---")
    audit(final_df)

    final_df.to_excel(V5_OUT, index=False)
    print(f"\nSaved {V5_OUT}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    main()
