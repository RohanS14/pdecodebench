"""
corrupt_comment.py — Generate CorrComm and CorrComm_Invalid rows.

For each gt_sample, a single donor is selected randomly (seeded) from
Comm_Valid rows with a different pde_class AND different num_method.
The same donor is used for both:
  - CorrComm:         donor comments injected into Comm_Valid of receiver
  - CorrComm_Invalid: donor comments injected into Comm_InValid of receiver

This prevents the probe from using a deterministic donor fingerprint as a
shortcut to recover pde_class or phys_valid.
"""

import random
import pandas as pd

SEED = 42


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


def build_donor_table(comm_valid: pd.DataFrame, rng: random.Random) -> dict:
    """
    For each gt_sample in comm_valid, pick one donor gt_sample randomly
    (different pde_class AND different num_method). Caps each donor pde_class
    at n_receivers // n_classes uses to prevent one class dominating as donor.
    Receivers are shuffled before assignment so no class is systematically favored.
    Returns {gt_sample: donor_row}.
    """
    n_classes = comm_valid["pde_class"].nunique()
    n_receivers = len(comm_valid)
    per_class_cap = n_receivers // n_classes  # e.g. 16 // 4 = 4

    donor_class_counts: dict[str, int] = {}
    donor_table = {}

    shuffled = comm_valid.sample(frac=1, random_state=rng.randint(0, 2**31))
    for _, receiver in shuffled.iterrows():
        candidates = comm_valid[
            (comm_valid["pde_class"] != receiver["pde_class"])
            & (comm_valid["num_method"] != receiver["num_method"])
            & comm_valid["pde_class"].map(lambda c: donor_class_counts.get(c, 0) < per_class_cap)
        ]
        if candidates.empty:
            # Relax cap if no candidates remain under it
            candidates = comm_valid[
                (comm_valid["pde_class"] != receiver["pde_class"])
                & (comm_valid["num_method"] != receiver["num_method"])
            ]
        if candidates.empty:
            print(f"WARNING: no donor found for {receiver['gt_sample']}, skipping")
            continue
        chosen = candidates.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
        donor_class_counts[chosen["pde_class"]] = donor_class_counts.get(chosen["pde_class"], 0) + 1
        donor_table[receiver["gt_sample"]] = chosen
    return donor_table


def make_corrcomm_rows(
    receivers: pd.DataFrame,
    comm_versions: pd.DataFrame,
    donor_table: dict,
    mod_type_out: str,
    valid_flag: bool,
) -> list[dict]:
    """
    Build CorrComm or CorrComm_Invalid rows.

    receivers      — NoComm_Valid or NoComm_InValid rows
    comm_versions  — Comm_Valid or Comm_InValid rows (defines comment positions)
    donor_table    — {gt_sample: donor_row} from build_donor_table
    mod_type_out   — "CorrComm" or "CorrComm_Invalid"
    valid_flag     — True for CorrComm, False for CorrComm_Invalid
    """
    new_rows = []
    for _, receiver in receivers.iterrows():
        gt = receiver["gt_sample"]

        comm_match = comm_versions[comm_versions["gt_sample"] == gt]
        if comm_match.empty:
            print(f"WARNING: no comm version found for {gt}, skipping")
            continue
        comm_receiver = comm_match.iloc[0]

        if gt not in donor_table:
            continue
        donor = donor_table[gt]

        donor_comments = extract_comments(donor["code"])
        receiver_comments = extract_comments(comm_receiver["code"])
        n_receiver = len(receiver_comments)

        if n_receiver == 0:
            print(f"WARNING: {gt} has no comments in comm version, skipping")
            continue

        delta_comments = len(donor_comments) - n_receiver
        new_code, injected = inject_comments(comm_receiver["code"], donor_comments, n_receiver)

        prefix = receiver["gt_sample"].split("_")[0]
        idx = receiver["title"].split("_")[-1]
        validity_str = "Valid" if valid_flag else "InValid"
        title = f"{prefix}_CorrComm_{validity_str}_{idx}"

        new_rows.append(
            {
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
                "gt_sample": gt,
                "mod_type": mod_type_out,
                # carried from the receiver so CorrComm_Invalid rows describe
                # their failure mode like every other invalid mod_type; NaN on
                # the valid side, matching the rest of the dataset
                "invalidity_note": receiver.get("invalidity_note"),
            }
        )
    return new_rows


def generate_corrcomm_rows(df: pd.DataFrame) -> list[dict]:
    """
    Entry point: returns new CorrComm + CorrComm_Invalid rows to append to df.
    Uses a seeded RNG so donor assignments are reproducible.
    """
    rng = random.Random(SEED)

    comm_valid = df[df["mod_type"] == "Comm_Valid"].reset_index(drop=True)
    comm_invalid = df[df["mod_type"] == "Comm_InValid"].reset_index(drop=True)
    no_comm_valid = df[df["mod_type"] == "NoComm_Valid"].reset_index(drop=True)
    no_comm_invalid = df[df["mod_type"] == "NoComm_InValid"].reset_index(drop=True)

    donor_table = build_donor_table(comm_valid, rng)
    print(f"Donor assignments (seed={SEED}):")
    for gt, donor in sorted(donor_table.items()):
        print(f"  {gt:30s} -> {donor['gt_sample']} ({donor['pde_class']})")

    corrcomm_rows = make_corrcomm_rows(
        no_comm_valid, comm_valid, donor_table, "CorrComm", valid_flag=True
    )
    corrcomm_invalid_rows = make_corrcomm_rows(
        no_comm_invalid, comm_invalid, donor_table, "CorrComm_Invalid", valid_flag=False
    )

    print(f"Generated {len(corrcomm_rows)} CorrComm rows")
    print(f"Generated {len(corrcomm_invalid_rows)} CorrComm_Invalid rows")
    return corrcomm_rows + corrcomm_invalid_rows
