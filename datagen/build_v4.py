"""
build_v4.py — Build pdedata_clean_v4.xlsx from pdedata_clean_v4_base.xlsx.

Steps:
  1. Load v4_base (which has 6 mod_types: Comm_Valid, NoComm_Valid, CorrComm,
     NoComm_CorrVar, Comm_InValid, NoComm_InValid).
  2. Drop existing CorrComm and NoComm_CorrVar rows — these will be regenerated
     with improved donor assignment and variable mapping.
  3. Regenerate CorrComm + CorrComm_Invalid (randomized seeded donor per gt_sample).
  4. Regenerate NoComm_CorrVar + NoComm_CorrVar_InValid (shared variable mapping
     across validity conditions).
  5. Run audit and print report.
  6. Save to data/pdedata_clean_v4.xlsx.
"""

import pandas as pd
from corrupt_comment import generate_corrcomm_rows
from augment_foobar_vars import generate_foobar_rows, _normalize
from audit_dataset import audit


def _strip_comments(code: str) -> str:
    """Remove all #-comment lines from code, normalizing encoding first."""
    src = _normalize(code)
    lines = [l for l in src.split("\n") if not l.strip().startswith("#")]
    return "\n".join(lines)


COL_ORDER = [
    "title", "code", "num_lines", "num_char",
    "pde_class", "phys_process", "phys_valid", "num_method",
    "corruption_source_id", "corruption_source_pde",
    "injected_comments", "delta_comments", "num_comments",
    "gt_sample", "mod_type",
]

MOD_TYPES_TO_REGENERATE = {"CorrComm", "NoComm_CorrVar"}


def main():
    print("Loading data/pdedata_clean_v4_base.xlsx ...")
    df = pd.read_excel("data/pdedata_clean_v4_base.xlsx")
    print(f"  Loaded {len(df)} rows, mod_types: {sorted(df['mod_type'].unique())}")

    # Ensure all expected columns exist
    for col in COL_ORDER:
        if col not in df.columns:
            df[col] = pd.NA

    # Fix 1: strip trailing whitespace from categorical fields
    df["num_method"] = df["num_method"].str.strip()
    df["phys_process"] = df["phys_process"].str.strip()

    # Drop mod_types that will be regenerated
    base_df = df[~df["mod_type"].isin(MOD_TYPES_TO_REGENERATE)].copy()

    # Fix 1 cont: fix difffusion typo
    base_df["phys_process"] = base_df["phys_process"].str.replace("difffusion", "diffusion", regex=False)

    # Fix 2: strip comment lines from NoComm_* source rows (Burgers_2 NoComm_InValid had 4 stray comments)
    nocomm_mask = base_df["mod_type"].str.startswith("NoComm")
    base_df.loc[nocomm_mask, "code"] = base_df.loc[nocomm_mask, "code"].apply(_strip_comments)
    base_df.loc[nocomm_mask, "num_comments"] = 0
    base_df.loc[nocomm_mask, "num_lines"] = base_df.loc[nocomm_mask, "code"].apply(lambda c: len(c.split("\n")))
    base_df.loc[nocomm_mask, "num_char"] = base_df.loc[nocomm_mask, "code"].apply(len)

    n_fixed = base_df.loc[nocomm_mask, "num_lines"].count()
    print(f"\nStripped comments from {n_fixed} NoComm_* rows")
    print(f"\nDropped {MOD_TYPES_TO_REGENERATE} rows. Base: {len(base_df)} rows")
    print(f"  mod_types remaining: {sorted(base_df['mod_type'].unique())}")

    # Step 3: CorrComm + CorrComm_Invalid
    print("\n--- Generating CorrComm + CorrComm_Invalid ---")
    corrcomm_rows = generate_corrcomm_rows(base_df)

    # Step 4: NoComm_CorrVar + NoComm_CorrVar_InValid
    print("\n--- Generating NoComm_CorrVar + NoComm_CorrVar_InValid ---")
    foobar_rows = generate_foobar_rows(base_df)

    # Assemble
    new_rows_df = pd.DataFrame(corrcomm_rows + foobar_rows)
    for col in COL_ORDER:
        if col not in new_rows_df.columns:
            new_rows_df[col] = pd.NA

    final_df = pd.concat([base_df[COL_ORDER], new_rows_df[COL_ORDER]], ignore_index=True)
    final_df = final_df.sort_values(["gt_sample", "mod_type"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(final_df)} rows")
    print(final_df["mod_type"].value_counts().to_string())

    # Step 5: Audit
    print("\n--- Running Audit ---")
    audit(final_df)

    # Step 6: Save
    out_path = "data/pdedata_clean_v4.xlsx"
    final_df.to_excel(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    import os
    # Run from repo root so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    main()
