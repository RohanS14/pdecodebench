"""
corrupt_comment.py — Generate CorrComm rows for pdedata.xlsx.

For each NoComm_Valid sample, find a Comm_Valid donor with a different
pde_class AND different num_method, inject its comments into the
receiver's Comm_Valid code, and append the result as a CorrComm row.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers

def count_comments(code_str: str) -> int:
    return sum(1 for line in code_str.split("\n") if line.strip().startswith("#"))


def extract_comments(code_str: str) -> list[dict]:
    """Return list of {line_no (1-indexed), text} for every #-leading line."""
    comments = []
    for i, line in enumerate(code_str.split("\n"), start=1):
        if line.strip().startswith("#"):
            comments.append({"line_no": i, "text": line})
    return comments


def inject_comments(receiver_code: str, donor_comments: list[dict], n_receiver: int):
    """
    Replace comment texts in receiver_code with donor_comments (cycled/truncated).
    Preserves receiver's leading whitespace on each comment line.
    Returns (new_code, injected_list).
    """
    # Cycle or truncate donor comments to exactly n_receiver entries
    if len(donor_comments) == 0:
        final_donor = []
    else:
        final_donor = [donor_comments[i % len(donor_comments)] for i in range(n_receiver)]

    lines = receiver_code.split("\n")
    injected = []
    donor_idx = 0
    new_lines = []

    for line_no, line in enumerate(lines, start=1):
        if line.strip().startswith("#") and donor_idx < len(final_donor):
            donor_entry = final_donor[donor_idx]
            donor_idx += 1
            # Preserve leading whitespace of the receiver line, use donor text
            leading_ws = line[: len(line) - len(line.lstrip())]
            donor_text_stripped = donor_entry["text"].strip()
            new_line = leading_ws + donor_text_stripped
            new_lines.append(new_line)
            injected.append(
                {
                    "donor_text": donor_entry["text"].strip(),
                    "gt_position": line_no,
                    "source_line_in_donor": donor_entry["line_no"],
                    "line_inserted": line_no,
                }
            )
        else:
            new_lines.append(line)

    return "\n".join(new_lines), injected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_excel("data/pdedata.xlsx")

    # ------------------------------------------------------------------
    # Step 0 — column fixups on existing rows
    # ------------------------------------------------------------------
    if "corruption_source_idx" in df.columns:
        df = df.rename(columns={"corruption_source_idx": "corruption_source_id"})

    # Drop leftover index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Add num_comments for all existing rows
    df["num_comments"] = df["code"].apply(count_comments)

    # Add missing CorrComm-specific columns (NaN for existing rows)
    for col in ["delta_comments"]:
        if col not in df.columns:
            df[col] = pd.NA

    # ------------------------------------------------------------------
    # Step 1–4 — build CorrComm rows
    # ------------------------------------------------------------------
    comm_valid = df[df["mod_type"] == "Comm_Valid"].reset_index(drop=True)
    no_comm_valid = df[df["mod_type"] == "NoComm_Valid"].reset_index(drop=True)

    new_rows = []

    for _, receiver in no_comm_valid.iterrows():
        # Find the Comm_Valid version of this same sample
        comm_match = comm_valid[comm_valid["gt_sample"] == receiver["gt_sample"]]
        if comm_match.empty:
            print(f"WARNING: no Comm_Valid found for {receiver['gt_sample']}, skipping")
            continue
        comm_receiver = comm_match.iloc[0]

        # Find donor: different pde_class AND different num_method
        def qualifies(row):
            return (
                row["pde_class"] != receiver["pde_class"]
                and row["num_method"] != receiver["num_method"]
            )

        donors = comm_valid[comm_valid.apply(qualifies, axis=1)].copy()
        if donors.empty:
            print(f"WARNING: no donor found for {receiver['gt_sample']}, skipping")
            continue

        # Deterministic: alphabetically first gt_sample among qualifiers
        donor = donors.sort_values("gt_sample").iloc[0]

        # Extract comments
        donor_comments = extract_comments(donor["code"])
        receiver_comments = extract_comments(comm_receiver["code"])
        n_receiver = len(receiver_comments)

        if n_receiver == 0:
            print(f"WARNING: receiver {receiver['gt_sample']} has no comments in Comm_Valid, skipping")
            continue

        delta_comments = len(donor_comments) - n_receiver

        # Build corrupted code
        new_code, injected = inject_comments(comm_receiver["code"], donor_comments, n_receiver)

        # Title: e.g. "Wave_CorrComm_Valid_1"
        pde_cap = receiver["pde_class"].capitalize()
        idx = receiver["title"].split("_")[-1]  # last token of NoComm_Valid title
        title = f"{pde_cap}_CorrComm_Valid_{idx}"

        new_row = {
            "title": title,
            "code": new_code,
            "num_lines": len(new_code.split("\n")),
            "num_char": len(new_code),
            "pde_class": receiver["pde_class"],
            "phys_process": receiver["phys_process"],
            "phys_valid": receiver["phys_valid"],
            "num_method": receiver["num_method"],
            "corruption_source_id": donor["title"],
            "corruption_source_pde": donor["pde_class"],
            "injected_comments": str(injected),
            "delta_comments": delta_comments,
            "num_comments": count_comments(new_code),
            "gt_sample": receiver["gt_sample"],
            "mod_type": "CorrComm",
        }
        new_rows.append(new_row)

    print(f"Generated {len(new_rows)} CorrComm rows")

    # ------------------------------------------------------------------
    # Step 5 — assemble final DataFrame
    # ------------------------------------------------------------------
    new_df = pd.DataFrame(new_rows)

    # Align columns before concat
    col_order = [
        "title", "code", "num_lines", "num_char",
        "pde_class", "phys_process", "phys_valid", "num_method",
        "corruption_source_id", "corruption_source_pde",
        "injected_comments", "delta_comments", "num_comments",
        "gt_sample", "mod_type",
    ]

    # Ensure existing df has all columns (fill missing with NaN)
    for col in col_order:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[col_order]
    final_df = pd.concat([df, new_df[col_order]], ignore_index=True)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    n_corr = final_df[final_df["mod_type"] == "CorrComm"].shape[0]
    print(f"Total rows: {final_df.shape[0]}  (expected 64)")
    print(f"CorrComm rows: {n_corr}  (expected 16)")
    assert n_corr == 16, f"Expected 16 CorrComm rows, got {n_corr}"
    assert final_df.shape[0] == 64, f"Expected 64 rows, got {final_df.shape[0]}"
    assert final_df["num_comments"].isna().sum() == 0, "num_comments has NaNs"

    corr_rows = final_df[final_df["mod_type"] == "CorrComm"]
    assert (corr_rows["corruption_source_pde"] != corr_rows["pde_class"]).all(), \
        "Some CorrComm rows have same pde_class as donor"

    print("All assertions passed.")

    final_df.to_excel("../data/pdedata.xlsx", index=False)
    print("Saved data/pdedata.xlsx")


if __name__ == "__main__":
    main()
