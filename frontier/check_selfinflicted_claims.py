"""
check_selfinflicted_claims.py — find episodes whose final answer attributes an
execution error to the ORIGINAL snippet when the original in fact runs clean.

Observed live (Wave_NoComm_CorrVar_1 @ gemini-3.7-flash/tl_high): the model
rewrote a boundary stencil (`foobar_23 - 1` -> `+ 1`), its own version raised
IndexError, and it reported "The code fails to run and raises an IndexError" as
a fact about the original, scoring it 0. That contradicted its own Stage-1
observation ("The code runs without error") and survived all three existing
guards -- the Stage-2 prompt states the original runs without execution errors,
the submit_final_answer description says an error from your own edit is not
evidence about the original, and the confirmation step re-shows the original
before a second submit is required.

DETECTION, not prevention, deliberately: a fourth prompt warning would risk
changing task difficulty, while a post-hoc check quantifies the problem without
altering what is being measured.

Every snippet in this dataset executes without error -- established
independently, by the trajectory-summary generation for the static experiment
and by the prior agentic sweep. So any claim that the original fails to run is
false by construction. The re-run below therefore is not establishing that
premise; it just confirms it on the few rows where a claim was actually made.

Signals:
  claims_failure   -- final valid_exp asserts the code does not execute.
  any_edit_errored -- some execution INSIDE the episode produced a traceback,
                      read from the stored action_trace (no re-execution).
  v0 verdict       -- solver_v0.py re-run via the harness's own
                      run_python_file(), so cwd/venv/timeout/decoding match the
                      episode. Only for rows that made a claim.

Keyword matching alone would over-flag: "produces NaNs" describes real behaviour
of unstable-but-running code, and "my corrected version crashes" is a TRUE
statement about the model's own edit. Requiring a clean original removes both.

Split of confirmed rows:
  MISATTRIBUTION -- a real traceback existed, but in MODIFIED code.
  FABRICATION    -- no execution in the episode errored at all.

Usage:
  python eval/frontier/check_selfinflicted_claims.py --jsonl <stage2.jsonl>
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

from frontier.agentic_sandbox import run_python_file  # noqa: E402

# Recall-limited shortlist: an unusually phrased claim slips through, so any
# count here is a LOWER BOUND.
CLAIM_PAT = re.compile(
    r"fails? to run|does not run|doesn't run|cannot run|can't run|won't run|"
    r"crash|traceback|errors? out|raises? an?\s|throws? an?\s|"
    r"IndexError|SyntaxError|NameError|TypeError|ValueError|KeyError|"
    r"ZeroDivisionError|AttributeError|RuntimeError|unhandled exception",
    re.I,
)
# Signature of a genuine Python failure. "Warning" is excluded on purpose --
# overflow/invalid-value warnings are normal output for unstable code that runs.
TRACE_PAT = re.compile(r"Traceback \(most recent call last\)|\b\w*Error: ", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no_rerun", action="store_true",
                    help="Trust the premise; skip confirming originals.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    print("=" * 78)
    print(f"SELF-INFLICTED-CLAIM CHECK — {len(rows)} episodes")
    print(f"  {args.jsonl}")
    print("=" * 78)

    flagged, missing, v0_broken = [], [], []
    for r in rows:
        exp = r.get("s2_submit_args", {}).get("valid_exp") or ""
        if not CLAIM_PAT.search(exp):
            continue
        r["_edit_errored"] = any(
            TRACE_PAT.search(str(a.get("result") or "")) for a in (r.get("action_trace") or []))

        ep = Path(r.get("episode_dir") or "")
        if not (ep / "solver_v0.py").exists():
            r["_v0"] = "MISSING"; missing.append(r["title"])
        elif args.no_rerun:
            r["_v0"] = "clean (assumed)"
        else:
            _out, err, timed_out, _ab = run_python_file(
                "solver_v0.py", ep, args.timeout, r["title"], "selfinflicted_check")
            if timed_out:
                r["_v0"] = "TIMEOUT"
            elif TRACE_PAT.search(err or ""):
                r["_v0"] = "ERRORS"; v0_broken.append(r["title"])
            else:
                r["_v0"] = "clean"
        flagged.append(r)

    print(f"\nepisodes claiming an execution error : {len(flagged)}/{len(rows)}")
    if missing:
        print(f"  solver_v0.py not found               : {len(missing)} {missing[:4]}")
    if v0_broken:
        print(f"  originals that genuinely DO error    : {len(v0_broken)} {v0_broken}")
        print("    -> claim legitimate; not counted as a problem "
              "(also contradicts the dataset premise, worth investigating)")

    real = [r for r in flagged if r.get("_v0", "").startswith("clean")]
    mis = [r for r in real if r["_edit_errored"]]
    fab = [r for r in real if not r["_edit_errored"]]

    print(f"\n{'=' * 78}")
    print(f"CONFIRMED — original runs clean, answer claims it errors : {len(real)}")
    print(f"  MISATTRIBUTION (real traceback, but in MODIFIED code)  : {len(mis)}")
    print(f"  FABRICATION    (nothing in the episode errored at all) : {len(fab)}")
    print("=" * 78)

    for label, group in (("MISATTRIBUTION", mis), ("FABRICATION", fab)):
        for r in group:
            m = CLAIM_PAT.search(r["s2_submit_args"]["valid_exp"])
            snip = r["s2_submit_args"]["valid_exp"][max(0, m.start() - 70): m.start() + 110]
            print(f"\n[{label}] {r['title']}")
            print(f"  gt_valid={r['gt_valid']}  mod_type={r['mod_type']}  "
                  f"score={r.get('s2_validity_score')}  "
                  f"valid_match={r.get('s2_valid_match')}  "
                  f"used_edit_source={r.get('s2_used_edit_source')}")
            print(f"  claim: ...{snip.strip()}...")

    if real:
        by_gt = {}
        for r in real:
            by_gt[r["gt_valid"]] = by_gt.get(r["gt_valid"], 0) + 1
        print(f"\nby ground truth: {by_gt}")
        print("  concentration on gt_valid=True matters -- that is where the"
              " specificity claim lives,")
        print("  and a wrongly-scored 0 there looks like a confident false alarm.")

    print("\nCaveat: the claim regex is a recall-limited shortlist, so this is a LOWER BOUND.")


if __name__ == "__main__":
    main()
