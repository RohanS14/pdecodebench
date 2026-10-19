"""
sign_flip_recall_analysis.py — two checks, both against existing data (no new
API calls):

(1) Recall (does the model call the row invalid at all?) split by defect
    type -- SIGN FLIP / MIXED / OTHER, per agent_docs/sign_flip_classification.txt
    -- and stratified by PDE class, since the sign-flip base rate is not
    uniform across PDE classes (a real confound noted during classification).

(2) Reuse of the existing judge_results.jsonl (already covers all eligible
    rows at full 256-row scale) to check: among rows the model already
    correctly called invalid, does its stated justification's `category`
    verdict amount to "identified the sign flip"? For the 17 gt_samples
    classified as a clean, single-discrepancy SIGN FLIP, category=all (or
    =some, if it occurs) can only be about that one discrepancy -- so this
    is a free re-slice, not a new experiment. The 3 MIXED gt_samples are
    reported as a separate category, not merged in (ambiguous which
    discrepancy a partial-credit verdict refers to).

Usage: python3 eval/frontier/sign_flip_recall_analysis.py
Output: agent_docs/sign_flip_recall_analysis.txt
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent

SIGN_FLIP = [
    "Burgers_1", "Burgers_5", "Burgers_6", "Burgers_7", "Burgers_8",
    "Heat_2", "Heat_3", "Heat_4", "Heat_5", "Heat_7", "Heat_8",
    "NavierStokes_1", "NavierStokes_2", "NavierStokes_5", "NavierStokes_7",
    "Wave_7", "Wave_8",
]
MIXED = ["Heat_1", "NavierStokes_6", "Wave_1"]

INVALID_MOD_TYPES = ["Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid"]


def defect_type(gt_sample: str) -> str:
    if gt_sample in SIGN_FLIP:
        return "SIGN FLIP"
    if gt_sample in MIXED:
        return "MIXED"
    return "OTHER"


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)



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
    p.add_argument("--judge", default=str(REPO_ROOT / "results/frontier/stratified_256/judge/judge_results.jsonl"),
                   help="Judge results JSONL. Must come from the SAME model and "
                        "condition as the Stage-2 files above -- a judge pass from "
                        "another arm silently answers a different question.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = pd.read_csv(REPO_ROOT / "data" / "merged_mod_jul28.csv")
    title_to_gt_sample = dict(zip(dataset["title"], dataset["gt_sample"]))

    nothink = load_jsonl(args.first)
    think = load_jsonl(args.second)
    judge = load_jsonl(args.judge)

    for df in (nothink, think, judge):
        df["gt_sample"] = df["title"].map(title_to_gt_sample)
        df["defect_type"] = df["gt_sample"].map(defect_type)

    out: list[str] = []
    out.append("Sign-flip recall + judge-reuse analysis (256-row scale, existing data only)")
    out.append("=" * 78)

    # ---- (1) Recall by defect type, stratified by PDE class ----
    out.append("\n(1) RECALL (called invalid at all), by defect type and PDE class")
    out.append("-" * 78)
    for label, df in [("nothink", nothink), ("think", think)]:
        out.append(f"\n{label}:")
        sub = df[df["mod_type"].isin(INVALID_MOD_TYPES)]
        for dtype in ["SIGN FLIP", "MIXED", "OTHER"]:
            dsub = sub[sub["defect_type"] == dtype]
            n = len(dsub)
            recall = dsub["s2_valid_match"].mean() * 100 if n else float("nan")
            out.append(f"  {dtype:10s} pooled: {recall:5.1f}%  (n={n})")
            for pde in ["burgers", "heat", "wave", "navier-stokes"]:
                psub = dsub[dsub["gt_pde"] == pde]
                n_p = len(psub)
                if n_p == 0:
                    continue
                r_p = psub["s2_valid_match"].mean() * 100
                out.append(f"      {pde:15s}: {r_p:5.1f}%  (n={n_p})")

    # ---- (2a) Does "some" occur on the 17 pure sign-flip rows? ----
    out.append("\n\n(2a) Does judge category='some' occur on the 17 pure SIGN FLIP rows?")
    out.append("-" * 78)
    for label, tb in [("nothink", 0), ("think", 1536)]:
        sf_judge = judge[(judge["defect_type"] == "SIGN FLIP") & (judge["thinking_budget"] == tb)]
        counts = sf_judge["category"].value_counts()
        out.append(f"  {label} (n={len(sf_judge)}): {dict(counts)}")

    # ---- (2b) Sign-flip-naming rate via existing judge category ----
    out.append("\n\n(2b) Judge-verdict 'named the real discrepancy' rate -- reused from existing")
    out.append("     judge category (scope: only rows the model already correctly called invalid).")
    out.append("     For SIGN FLIP this means 'named the sign flip' (the only discrepancy present).")
    out.append("     For OTHER/ALL this means 'named the actual (non-sign-flip) defect' -- category")
    out.append("     is scored against whatever the real discrepancy is, not sign flips specifically.")
    out.append("-" * 78)
    for dtype in ["SIGN FLIP", "MIXED", "OTHER", "ALL (pooled, any defect type)"]:
        out.append(f"\n{dtype}:")
        for label, tb in [("nothink", 0), ("think", 1536)]:
            if dtype == "ALL (pooled, any defect type)":
                dsub = judge[judge["thinking_budget"] == tb]
            else:
                dsub = judge[(judge["defect_type"] == dtype) & (judge["thinking_budget"] == tb)]
            n = len(dsub)
            if n == 0:
                out.append(f"  {label}: n=0 (no eligible rows)")
                continue
            counts = dsub["category"].value_counts()
            named = dsub["category"].isin(["all", "some"]).sum()
            out.append(f"  {label} (n={n}): category counts = {dict(counts)}  ->  "
                        f"named (all+some)/n = {named}/{n} ({100*named/n:.1f}%)")

    out_path = REPO_ROOT / "agent_docs" / "sign_flip_recall_analysis.txt"
    out_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
