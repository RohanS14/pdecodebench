"""
Construct Comm_InValid rows by injecting GT comments from Comm_Valid
into the physically invalid code from NoComm_InValid.

Strategy: for each gt_sample, extract comment lines (and their positions)
from Comm_Valid, then insert them at the same relative positions in NoComm_InValid.
"""
import ast
import pandas as pd
import re


def extract_comments_with_positions(code: str) -> list[dict]:
    """Return list of {line_idx: int, text: str} for every comment line."""
    comments = []
    for i, line in enumerate(code.split("\n")):
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append({"line_idx": i, "text": line})
    return comments


def inject_comments(base_code: str, comments: list[dict]) -> str:
    """Insert comment lines into base_code at the positions they appeared in the donor."""
    lines = base_code.split("\n")
    # Insert in reverse order so earlier insertions don't shift later indices
    for c in sorted(comments, key=lambda x: x["line_idx"], reverse=True):
        idx = min(c["line_idx"], len(lines))
        lines.insert(idx, c["text"])
    return "\n".join(lines)


def build_comm_invalid(df: pd.DataFrame) -> pd.DataFrame:
    comm_valid = df[df["mod_type"] == "Comm_Valid"].set_index("gt_sample")
    nocomm_invalid = df[df["mod_type"] == "NoComm_InValid"].copy()

    rows = []
    for _, row in nocomm_invalid.iterrows():
        gt = row["gt_sample"]
        if gt not in comm_valid.index:
            continue
        donor = comm_valid.loc[gt]
        comments = extract_comments_with_positions(donor["code"])
        if not comments:
            continue

        new_code = inject_comments(row["code"], comments)
        new_row = row.copy()
        new_row["code"] = new_code
        new_row["mod_type"] = "Comm_InValid"
        new_row["num_comments"] = len(comments)
        new_row["num_lines"] = len(new_code.split("\n"))
        new_row["num_char"] = len(new_code)
        # title: replace NoComm_InValid -> Comm_InValid
        new_row["title"] = row["title"].replace("NoComm_InValid", "Comm_InValid")
        rows.append(new_row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_excel("data/pdedata_clean.xlsx")
    comm_invalid = build_comm_invalid(df)

    print(f"Constructed {len(comm_invalid)} Comm_InValid rows")
    print(comm_invalid[["title", "gt_sample", "pde_class", "num_comments"]].to_string())

    df_extended = pd.concat([df, comm_invalid], ignore_index=True)
    df_extended.to_excel("data/pdedata_clean_v2.xlsx", index=False)
    print(f"\nSaved pdedata_clean_v2.xlsx — {len(df_extended)} rows")
    print(df_extended["mod_type"].value_counts())
    print(df_extended["pde_class"].value_counts())
