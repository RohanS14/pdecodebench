"""
check_hypotheses.py — automated pass/fail/skip checker for the standing
hypotheses in agent_docs/hypotheses.md, per the precise criteria spelled out
in agent_docs/hypothesis_sanity_checks.md.

Reuses viz/report_belief_revision_agentic.py's data loaders and rate-computation helpers
directly rather than duplicating any eval/viz logic -- this script only
computes hypothesis-level comparisons over data that already exists on disk.
It does NOT run the sweep, the judge, or the HTML report; run those first.

Usage:
  python3 frontier/check_hypotheses.py \\
      --nothink results/frontier/stratified_256/nothink/gemini25flash__belief_revision_agentic.jsonl \\
      --think results/frontier/stratified_256/think/gemini25flash__belief_revision_agentic.jsonl \\
      --judge results/frontier/stratified_256/judge/judge_results.jsonl

Note: needs pandas, which lives in system python3's environment in this repo
(same consideration as viz/report_belief_revision_agentic.py), not eval/.venv.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "viz"))

from report_belief_revision_agentic import (  # noqa: E402
    INVALID_MOD_TYPES,
    VALID_MOD_TYPES,
    _rate,
    load_agentic_results,
    load_judge_results,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = {"pass": 0, "fail": 0, "skip": 0}


def report(name: str, verdict: str, detail: str = "") -> None:
    label = {"pass": PASS, "fail": FAIL, "skip": SKIP}[verdict]
    results[verdict] += 1
    print(f"  {label}  {name}" + (f"\n         {detail}" if detail else ""))


def _pooled_rate(df, target_mod_type: str, all_mod_types: list, match_col: str) -> tuple[float, int]:
    others = [m for m in all_mod_types if m != target_mod_type]
    return _rate(df, others, match_col)


def check_h1(df_nothink, df_think) -> None:
    print("\n=== H1a/b/c -- lexical-obfuscation recall/specificity asymmetry ===")

    stages = [("S1", df_nothink, "s1_valid_match"), ("S2 nothink", df_nothink, "s2_valid_match")]
    if not df_think.empty:
        stages.append(("S2 think", df_think, "s2_valid_match"))

    # H1a: specificity(NoComm_CorrVar) > specificity(other 3 valid, pooled),
    # at every available stage.
    spec_target = []
    spec_other = []
    for label, df, col in stages:
        t_rate, t_n = _rate(df, ["NoComm_CorrVar"], col)
        o_rate, o_n = _pooled_rate(df, "NoComm_CorrVar", VALID_MOD_TYPES, col)
        spec_target.append((label, t_rate, t_n))
        spec_other.append((label, o_rate, o_n))
        holds = t_rate > o_rate
        report(
            f"H1a @ {label}: specificity(NoComm_CorrVar)={t_rate:.1f}% (n={t_n}) > "
            f"specificity(other 3 pooled)={o_rate:.1f}% (n={o_n})",
            "pass" if holds else "fail",
        )

    # H1b: recall(NoComm_CorrVar_InValid) < recall(other 3 invalid, pooled) at
    # S1, and the gap shrinks by S2 think (if think data is present).
    rec_target = []
    rec_other = []
    for label, df, col in stages:
        t_rate, t_n = _rate(df, ["NoComm_CorrVar_InValid"], col)
        o_rate, o_n = _pooled_rate(df, "NoComm_CorrVar_InValid", INVALID_MOD_TYPES, col)
        rec_target.append((label, t_rate, t_n))
        rec_other.append((label, o_rate, o_n))

    s1_holds = rec_target[0][1] < rec_other[0][1]
    report(
        f"H1b @ S1: recall(NoComm_CorrVar_InValid)={rec_target[0][1]:.1f}% "
        f"(n={rec_target[0][2]}) < recall(other 3 pooled)={rec_other[0][1]:.1f}% "
        f"(n={rec_other[0][2]})",
        "pass" if s1_holds else "fail",
    )

    if len(stages) == 3:
        s1_gap = rec_other[0][1] - rec_target[0][1]
        think_gap = rec_other[2][1] - rec_target[2][1]
        gap_shrinks = think_gap < s1_gap
        report(
            f"H1b gap shrinks S1 -> S2 think: S1 gap={s1_gap:.1f}pp, "
            f"S2 think gap={think_gap:.1f}pp",
            "pass" if gap_shrinks else "fail",
        )

        # H1c: recall gap shrinks while specificity gap does not, over the
        # same S1 -> S2 think stages.
        spec_s1_gap = spec_target[0][1] - spec_other[0][1]
        spec_think_gap = spec_target[2][1] - spec_other[2][1]
        spec_gap_holds_or_grows = spec_think_gap >= spec_s1_gap - 1e-9
        h1c_holds = gap_shrinks and spec_gap_holds_or_grows
        report(
            f"H1c asymmetry: recall gap shrank ({s1_gap:.1f}pp -> {think_gap:.1f}pp) "
            f"while specificity gap did not ({spec_s1_gap:.1f}pp -> {spec_think_gap:.1f}pp)",
            "pass" if h1c_holds else "fail",
        )
    else:
        report("H1b gap-shrinks / H1c asymmetry", "skip", "no --think data provided")


def check_h2(df_nothink, df_think) -> None:
    print("\n=== H2 -- agentic evidence-gathering improves accuracy over the static baseline ===")
    s1_acc = df_nothink["s1_valid_match"].dropna().mean() * 100
    nothink_acc = df_nothink["s2_valid_match"].dropna().mean() * 100
    report(
        f"accuracy(S2 nothink)={nothink_acc:.1f}% > accuracy(S1)={s1_acc:.1f}%",
        "pass" if nothink_acc > s1_acc else "fail",
    )
    if df_think.empty:
        report("accuracy(S2 think) > accuracy(S1)", "skip", "no --think data provided")
        return
    think_acc = df_think["s2_valid_match"].dropna().mean() * 100
    report(
        f"accuracy(S2 think)={think_acc:.1f}% > accuracy(S1)={s1_acc:.1f}%",
        "pass" if think_acc > s1_acc else "fail",
    )


def check_h3(df_nothink, df_think) -> None:
    print("\n=== H3 -- thinking budget improves accuracy over no-thinking, in aggregate ===")
    if df_think.empty:
        report("accuracy(S2 think) > accuracy(S2 nothink)", "skip", "no --think data provided")
        return
    nothink_acc = df_nothink["s2_valid_match"].dropna().mean() * 100
    think_acc = df_think["s2_valid_match"].dropna().mean() * 100
    holds = think_acc > nothink_acc
    report(
        f"accuracy(S2 think)={think_acc:.1f}% vs. accuracy(S2 nothink)={nothink_acc:.1f}% "
        f"({'higher' if holds else 'NOT higher'} -- reported either way, see "
        f"hypothesis_sanity_checks.md)",
        "pass" if holds else "fail",
    )


def check_h5(df_nothink, df_think) -> None:
    print("\n=== H5 -- voluntary-stop confidence correlates with correctness ===")
    for label, df in [("nothink", df_nothink), ("think", df_think)]:
        if df.empty:
            report(f"H5 @ {label}", "skip", "no data provided")
            continue
        sub = df[df["aborted"] == False].dropna(subset=["actions_remaining_at_submission", "s2_valid_match"])
        voluntary = sub[sub["actions_remaining_at_submission"] > 0]
        forced = sub[sub["actions_remaining_at_submission"] == 0]
        if len(voluntary) == 0 or len(forced) == 0:
            report(f"H5 @ {label}", "skip", f"voluntary n={len(voluntary)}, forced n={len(forced)}")
            continue
        v_acc = voluntary["s2_valid_match"].mean() * 100
        f_acc = forced["s2_valid_match"].mean() * 100
        holds = v_acc > f_acc
        report(
            f"H5 @ {label}: accuracy(voluntary, n={len(voluntary)})={v_acc:.1f}% > "
            f"accuracy(forced, n={len(forced)})={f_acc:.1f}%",
            "pass" if holds else "fail",
        )


def check_j1(df_nothink, df_think, judge_df) -> None:
    print("\n=== J1 -- does thinking improve judge-assessed reasoning quality? ===")
    if df_think.empty:
        report("J1 availability gate", "skip", "no --think data provided")
        return

    eligible_nothink = df_nothink[(df_nothink["gt_valid"] == False) & (df_nothink["s2_valid_match"] == 1)]
    eligible_think = df_think[(df_think["gt_valid"] == False) & (df_think["s2_valid_match"] == 1)]

    if judge_df.empty:
        report(
            "J1 availability gate",
            "skip",
            f"no judge results found (eligible: nothink={len(eligible_nothink)}, "
            f"think={len(eligible_think)})",
        )
        return

    judged_counts = judge_df.groupby("thinking_budget").size()
    judged_nothink = int(judged_counts.get(0, 0))
    judged_think = int(judged_counts.get(1536, 0))

    THRESHOLD = 0.8
    nothink_ok = len(eligible_nothink) > 0 and judged_nothink >= THRESHOLD * len(eligible_nothink)
    think_ok = len(eligible_think) > 0 and judged_think >= THRESHOLD * len(eligible_think)

    if not (nothink_ok and think_ok):
        report(
            "J1 availability gate",
            "skip",
            f"judge results not yet at full scale for both conditions -- "
            f"nothink: judged={judged_nothink}/eligible={len(eligible_nothink)}, "
            f"think: judged={judged_think}/eligible={len(eligible_think)} "
            f"(need >= {int(THRESHOLD*100)}% coverage on both). "
            f"Run the judge scale-up first.",
        )
        return

    category_score = {"none": 0, "some": 1, "all": 2}
    judge_df = judge_df.copy()
    judge_df["_category_score"] = judge_df["category"].map(category_score)

    nothink_j = judge_df[judge_df["thinking_budget"] == 0]
    think_j = judge_df[judge_df["thinking_budget"] == 1536]

    nothink_cat = nothink_j["_category_score"].mean()
    think_cat = think_j["_category_score"].mean()
    nothink_incorrect = nothink_j["contains_incorrect_claims"].mean() * 100
    think_incorrect = think_j["contains_incorrect_claims"].mean() * 100

    cat_holds = think_cat > nothink_cat
    incorrect_holds = think_incorrect < nothink_incorrect

    report(
        f"J1 category: mean(think)={think_cat:.2f} > mean(nothink)={nothink_cat:.2f} "
        f"(0=none,1=some,2=all)",
        "pass" if cat_holds else "fail",
    )
    report(
        f"J1 contains_incorrect_claims: rate(think)={think_incorrect:.1f}% < "
        f"rate(nothink)={nothink_incorrect:.1f}%",
        "pass" if incorrect_holds else "fail",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automated hypothesis sanity checks")
    p.add_argument("--nothink", default=str(
        REPO_ROOT / "results" / "frontier" / "stratified_256" / "nothink" / "gemini25flash__belief_revision_agentic.jsonl"))
    p.add_argument("--think", default=str(
        REPO_ROOT / "results" / "frontier" / "stratified_256" / "think" / "gemini25flash__belief_revision_agentic.jsonl"))
    p.add_argument("--judge", default=str(
        REPO_ROOT / "results" / "frontier" / "stratified_256" / "judge" / "judge_results.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nothink_path = Path(args.nothink)
    if not nothink_path.exists():
        sys.exit(f"[check-hypotheses] ERROR: --nothink file not found: {nothink_path}. "
                  f"Run run_stratified_sweep.py first.")

    df_nothink = load_agentic_results(nothink_path)
    think_path = Path(args.think)
    df_think = load_agentic_results(think_path) if think_path.exists() else __import__("pandas").DataFrame()
    judge_df = load_judge_results(Path(args.judge))

    print(f"[check-hypotheses] nothink: {len(df_nothink)} rows, think: {len(df_think)} rows, "
          f"judge: {len(judge_df)} rows")

    check_h1(df_nothink, df_think)
    check_h2(df_nothink, df_think)
    check_h3(df_nothink, df_think)
    check_h5(df_nothink, df_think)
    check_j1(df_nothink, df_think, judge_df)

    print(f"\n{results['pass']} passed, {results['fail']} failed, {results['skip']} skipped")


if __name__ == "__main__":
    main()
