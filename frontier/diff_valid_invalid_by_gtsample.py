"""
diff_valid_invalid_by_gtsample.py — for each of the 32 gt_samples, diff the
Comm_Valid vs Comm_InValid code (real variable names, real comments, so the
diff is maximally readable) to see exactly what was injected as the invalidity
defect. One diff per gt_sample is sufficient -- the same underlying code-level
corruption is shared across all 4 invalid mod_type variants (NoComm_InValid,
CorrComm_Invalid, NoComm_CorrVar_InValid are the same corrupted logic under
different name/comment treatments), so this doesn't need to be repeated per
mod_type.

Deliberately does NOT auto-classify defects (sign flip vs. other) -- two
heuristic attempts both proved unreliable on real examples, so this just
prints the raw, comment-filtered diff plus a plain differing-line count, for
direct manual review rather than a false-confidence label.

Usage: python3 eval/frontier/diff_valid_invalid_by_gtsample.py
Output: agent_docs/valid_invalid_diffs_by_gtsample.txt
"""
import difflib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent


def differing_line_count(diff_lines: list[str]) -> str:
    """No auto-classification -- two attempts at a sign-flip heuristic (a
    literal +/- character swap, then an A-B -> B-A operand-swap regex) both
    proved unreliable on real examples (the regex mis-splits expressions with
    bracketed indices like f_interface[0:n] - f_interface[1:n+1]). Reporting
    a plain differing-line count instead of a false-confidence label; inspect
    each diff directly."""
    removed = [l[1:] for l in diff_lines if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
    if len(removed) != len(added):
        return f"{len(removed)} removed / {len(added)} added line(s) (unequal -- inspect manually)"
    n_diff = sum(1 for r, a in zip(removed, added) if r.strip() != a.strip())
    return f"{n_diff} of {len(removed)} paired line(s) actually differ"


def main() -> None:
    df = pd.read_csv(REPO_ROOT / "data" / "merged_mod_jul28.csv")
    gt_samples = sorted(df["gt_sample"].unique())

    out: list[str] = []
    out.append("Valid vs Invalid code diffs, one per gt_sample (32 total)")
    out.append(
        "Comm_Valid vs Comm_InValid (real names + real comments, most readable pair).\n"
        "Comment-only lines are filtered out before diffing (they get reordered "
        "independent of the actual code defect). No auto-classification -- verify "
        "each one directly."
    )

    for gt_sample in gt_samples:
        valid_row = df[(df["gt_sample"] == gt_sample) & (df["mod_type"] == "Comm_Valid")]
        invalid_row = df[(df["gt_sample"] == gt_sample) & (df["mod_type"] == "Comm_InValid")]
        out.append(f"\n{'=' * 70}\n{gt_sample}\n{'=' * 70}")
        if valid_row.empty or invalid_row.empty:
            out.append("  MISSING Comm_Valid or Comm_InValid row -- skipped.")
            continue

        # Comment lines get shuffled to different positions between Comm_Valid
        # and Comm_InValid independent of the actual code defect (confirmed
        # live -- Burgers_1's real diff, a genuine sign flip, was otherwise
        # buried under this reordering noise). Diff code-only lines so the
        # real defect isn't obscured by comment placement churn.
        def code_only(text: str) -> list[str]:
            lines = []
            for l in text.splitlines(keepends=True):
                stripped = l.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(l)
            return lines

        valid_code = code_only(str(valid_row.iloc[0]["code"]))
        invalid_code = code_only(str(invalid_row.iloc[0]["code"]))
        diff = list(difflib.unified_diff(valid_code, invalid_code, lineterm="",
                                          fromfile="Comm_Valid", tofile="Comm_InValid"))
        if not diff:
            out.append("  (no textual diff found -- identical code?)")
            continue

        out.append(f"  {differing_line_count(diff)}\n")
        out.extend("  " + l for l in diff)

    out_path = REPO_ROOT / "agent_docs" / "valid_invalid_diffs_by_gtsample.txt"
    out_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
