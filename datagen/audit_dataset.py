"""
audit_dataset.py — Schema, typo, completeness, and balance audit for pdedata.

Prints a report with issues and suggestions. Does not modify any files.
Run standalone or import audit(df) for use in build scripts.
"""

import sys
import pandas as pd


ALL_MOD_TYPES = [
    "Comm_Valid",
    "NoComm_Valid",
    "CorrComm",
    "NoComm_CorrVar",
    "Comm_InValid",
    "NoComm_InValid",
    "CorrComm_Invalid",
    "NoComm_CorrVar_InValid",
]

KNOWN_PDE_CLASSES = {"wave", "heat", "burgers", "navier-stokes"}
KNOWN_PROCESSES = {"diffusion", "advection", "oscillation", "restoration"}


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def audit(df: pd.DataFrame) -> bool:
    """Run all checks. Returns True if no issues found."""
    issues = []

    # ------------------------------------------------------------------
    # 1. Typo scan — phys_process
    # ------------------------------------------------------------------
    _section("1. Typo & Casing Audit")

    known_typos = {"difffusion": "diffusion", "Diffusion": "diffusion",
                   "Advection": "advection", "Oscillation": "oscillation",
                   "Restoration": "restoration"}

    for col in ["pde_class", "phys_process"]:
        if col not in df.columns:
            print(f"  MISSING COLUMN: {col}")
            issues.append(f"missing column {col}")
            continue

        vals = df[col].dropna().unique()
        for val in sorted(vals):
            parts = [p.strip() for p in str(val).split("/")]
            for part in parts:
                if col == "pde_class" and part not in KNOWN_PDE_CLASSES:
                    msg = f"  UNEXPECTED {col} value: {part!r}"
                    if part in known_typos:
                        msg += f" -> suggest {known_typos[part]!r}"
                    print(msg)
                    issues.append(msg)
                elif col == "phys_process":
                    if part in known_typos:
                        print(f"  TYPO in phys_process: {part!r} -> suggest {known_typos[part]!r}")
                        issues.append(f"typo: {part}")
                    elif part not in KNOWN_PROCESSES:
                        print(f"  UNEXPECTED phys_process value: {part!r}")
                        issues.append(f"unexpected phys_process: {part}")

    if not issues:
        print("  OK — no typos or casing issues found")

    # ------------------------------------------------------------------
    # 2. phys_valid schema
    # ------------------------------------------------------------------
    _section("2. phys_valid Schema")

    if "phys_valid" not in df.columns:
        print("  MISSING COLUMN: phys_valid")
        issues.append("missing phys_valid")
    else:
        vals = df["phys_valid"].dropna().unique()
        bad = [v for v in vals if v not in {True, False, 1, 0, "True", "False", "1", "0"}]
        if bad:
            print(f"  NON-BOOLEAN values found: {bad}")
            print(f"  Suggestion: cast phys_valid to bool before saving")
            issues.append(f"phys_valid bad values: {bad}")
        else:
            print(f"  OK — values: {sorted(str(v) for v in vals)}")

        # Check alignment: invalid mod_types should have phys_valid=False
        invalid_mod_types = {"Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid"}
        valid_mod_types = {"Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar"}

        for mod in invalid_mod_types:
            subset = df[df["mod_type"] == mod]
            if subset.empty:
                continue
            wrong = subset[subset["phys_valid"].astype(str).isin({"True", "1", "true"})]
            if not wrong.empty:
                print(f"  MISALIGNED: {mod} has {len(wrong)} rows with phys_valid=True")
                issues.append(f"phys_valid misaligned for {mod}")

        for mod in valid_mod_types:
            subset = df[df["mod_type"] == mod]
            if subset.empty:
                continue
            wrong = subset[subset["phys_valid"].astype(str).isin({"False", "0", "false"})]
            if not wrong.empty:
                print(f"  MISALIGNED: {mod} has {len(wrong)} rows with phys_valid=False")
                issues.append(f"phys_valid misaligned for {mod}")

    # ------------------------------------------------------------------
    # 3. Completeness — every gt_sample should have all 8 mod_types
    # ------------------------------------------------------------------
    _section("3. Stratification Check (8 mod_types per gt_sample)")

    if "gt_sample" not in df.columns or "mod_type" not in df.columns:
        print("  MISSING COLUMNS: gt_sample or mod_type")
        issues.append("missing gt_sample/mod_type")
    else:
        gt_samples = sorted(df["gt_sample"].dropna().unique())
        found_mod_types = set(df["mod_type"].unique())
        missing_globally = set(ALL_MOD_TYPES) - found_mod_types
        if missing_globally:
            print(f"  MISSING mod_types entirely: {sorted(missing_globally)}")
            issues.append(f"missing mod_types: {missing_globally}")

        incomplete = []
        for gt in gt_samples:
            subset = df[df["gt_sample"] == gt]
            present = set(subset["mod_type"].unique())
            missing = set(ALL_MOD_TYPES) - present
            if missing:
                incomplete.append((gt, sorted(missing)))

        if incomplete:
            print(f"  {len(incomplete)}/{len(gt_samples)} gt_samples are incomplete:")
            for gt, missing in incomplete:
                print(f"    {gt}: missing {missing}")
            issues.append(f"{len(incomplete)} incomplete gt_samples")
        else:
            print(f"  OK — all {len(gt_samples)} gt_samples have all 8 mod_types")

    # ------------------------------------------------------------------
    # 4. Class imbalance
    # ------------------------------------------------------------------
    _section("4. Class Imbalance")

    total = len(df)
    print(f"\n  Total rows: {total}")

    for col in ["pde_class", "mod_type", "phys_valid"]:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        print(f"\n  {col}:")
        for val, n in counts.items():
            bar = "#" * int(30 * n / total)
            print(f"    {str(val):30s} {n:4d}  {bar}")

    if "phys_process" in df.columns:
        print(f"\n  phys_process (multi-label, counting each tag separately):")
        from collections import Counter
        tag_counts: Counter = Counter()
        for val in df["phys_process"].dropna():
            for tag in str(val).split("/"):
                tag_counts[tag.strip()] += 1
        for tag, n in tag_counts.most_common():
            bar = "#" * int(30 * n / total)
            print(f"    {tag:30s} {n:4d}  {bar}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _section("Summary")
    if issues:
        print(f"  {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        return False
    else:
        print("  All checks passed.")
        return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/merged_mod_jul28.csv"
    df = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_excel(path)
    ok = audit(df)
    sys.exit(0 if ok else 1)
