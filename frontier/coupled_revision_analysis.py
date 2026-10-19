"""
coupled_revision_analysis.py — how often does the model revise BOTH its
validity answer AND its PDE-class answer between Stage 1 and Stage 2, and in
which directions (per Rohan's request; see conversation for full discussion).

Validity is binary, so "revised" is exactly captured by s1_valid_match !=
s2_valid_match. PDE class is NOT binary (4 possible classes), so a raw
match-transition would miss a wrong-class -> different-wrong-class revision
(both would show pde_match=0 at both stages despite the stated class
actually changing) -- this script instead reuses parse_score.py's own
ALIASES/"_any_alias_match" keyword logic to independently guess which
canonical class (if any) each stage's raw text maps to, then compares those
guesses directly. This is the same matching rule score_pde() itself uses,
just applied against all 4 candidate classes instead of only the row's
ground-truth class.

Usage: python3 frontier/coupled_revision_analysis.py
Output: agent_docs/coupled_revision_analysis.txt
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "freegen_static_judgments"))

from parse_score import ALIASES, _any_alias_match  # noqa: E402

PDE_CLASSES = list(ALIASES["pde"].keys())  # ["wave", "heat", "burgers", "navier-stokes"]


def guess_pde_class(raw_text: str | None) -> str:
    """Which canonical PDE class (if any) this raw text maps to, using the
    exact same alias keyword logic score_pde() uses -- just checked against
    all 4 classes instead of only the ground-truth one. Returns 'none' if no
    class's aliases appear in the text, or 'ambiguous' if more than one
    class's aliases appear (both are their own distinct categories, not
    lumped into a single miscellaneous bucket, since either could itself be
    a meaningful kind of "revision")."""
    if not raw_text:
        return "none"
    matches = [c for c in PDE_CLASSES if _any_alias_match(raw_text, c, "pde")]
    if len(matches) == 0:
        return "none"
    if len(matches) > 1:
        return "ambiguous(" + "+".join(sorted(matches)) + ")"
    return matches[0]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def pde_direction(s1_guess: str, s2_guess: str, gt_pde: str) -> str:
    """Only called when s1_guess != s2_guess (pde_revised). Classifies the
    transition using the guessed labels against gt_pde -- consistent with
    s1_pde_match/s2_pde_match by construction (same alias-match rule)."""
    s1_right = s1_guess == gt_pde
    s2_right = s2_guess == gt_pde
    if s1_right and not s2_right:
        return "right->wrong"
    if not s1_right and s2_right:
        return "wrong->right"
    return "wrong->wrong (different wrong class)"


def analyze(rows: list[dict], label: str, out: list[str]) -> None:
    out.append(f"\n{'=' * 70}\n{label.upper()} (n={len(rows)})\n{'=' * 70}")

    validity_only = pde_only = both = neither = 0
    joint_table: dict[str, int] = {}
    joint_examples: dict[str, list[str]] = {}

    for row in rows:
        gt_pde = str(row["gt_pde"]).strip().lower()
        s1_pde_raw = row.get("s1_parsed_pde")
        s2_pde_raw = (row.get("s2_submit_args") or {}).get("pde")

        s1_guess = guess_pde_class(s1_pde_raw)
        s2_guess = guess_pde_class(s2_pde_raw)
        pde_revised = s1_guess != s2_guess

        s1_vm, s2_vm = row.get("s1_valid_match"), row.get("s2_valid_match")
        validity_revised = (s1_vm is not None and s2_vm is not None and s1_vm != s2_vm)

        if validity_revised and pde_revised:
            both += 1
            v_dir = "right->wrong" if (s1_vm == 1 and s2_vm == 0) else "wrong->right"
            p_dir = pde_direction(s1_guess, s2_guess, gt_pde)
            key = f"validity {v_dir}  |  pde {p_dir}"
            joint_table[key] = joint_table.get(key, 0) + 1
            joint_examples.setdefault(key, []).append(
                f"{row['title']} (gt_pde={gt_pde}, s1_pde='{s1_pde_raw}' -> s2_pde='{s2_pde_raw}')"
            )
        elif validity_revised:
            validity_only += 1
        elif pde_revised:
            pde_only += 1
        else:
            neither += 1

    out.append(f"\nMarginal counts (out of {len(rows)} rows):")
    out.append(f"  validity revised only (pde unchanged): {validity_only}")
    out.append(f"  pde revised only (validity unchanged):  {pde_only}")
    out.append(f"  BOTH revised:                           {both}")
    out.append(f"  neither revised:                        {neither}")

    out.append(f"\nJoint direction breakdown, rows where BOTH revised (n={both}):")
    if both == 0:
        out.append("  (no rows -- nothing to break down)")
    else:
        for key in sorted(joint_table, key=lambda k: -joint_table[k]):
            count = joint_table[key]
            out.append(f"  {key}: {count}/{both}")
            for ex in joint_examples[key]:
                out.append(f"      - {ex}")

    highlight_key = "validity wrong->right  |  pde wrong->right"
    highlight_count = joint_table.get(highlight_key, 0)
    out.append(
        f"\n*** Most-interesting case (both wrong at S1, both right at S2): "
        f"{highlight_count}/{both if both else len(rows)} of the both-revised rows ***"
    )



# ── Path arguments ───────────────────────────────────────────────────────────
# Defaults point at the completed gemini-2.5-flash 256-row run so existing usage
# is unchanged, but every path is overridable: a new model's sweep lands under a
# different, model-keyed output root, and hand-editing this file per model was
# the previous alternative.
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", default=str(REPO_ROOT / "results/frontier/stratified_256/nothink/gemini25flash__belief_revision_agentic.jsonl"),
                   help="Stage-2 JSONL for the first condition (was: nothink).")
    p.add_argument("--second", default=str(REPO_ROOT / "results/frontier/stratified_256/think/gemini25flash__belief_revision_agentic.jsonl"),
                   help="Stage-2 JSONL for the second condition (was: think).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    nothink = load(args.first)
    think = load(args.second)

    out: list[str] = []
    out.append("Coupled validity + PDE-class revision analysis (256-row scale)")
    out.append(
        "Question: how often does the model revise BOTH its validity answer AND its\n"
        "stated PDE class between Stage 1 (read-only) and Stage 2 (agentic), and in\n"
        "which directions? PDE-class revision is detected by comparing which of the 4\n"
        "canonical classes (if any) each stage's raw text keyword-matches to (same\n"
        "alias rule freegen_static_judgments/parse_score.py's score_pde() uses), not just match/mismatch\n"
        "against ground truth -- this catches wrong-class -> different-wrong-class\n"
        "revisions a naive match-transition would miss."
    )

    analyze(nothink, "nothink", out)
    analyze(think, "think", out)

    out_path = REPO_ROOT / "agent_docs" / "coupled_revision_analysis.txt"
    out_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
