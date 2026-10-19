"""
score_probe_report.py — gate report for a validity-score sweep.

Computes exactly the numbers the run plan requires before committing to a full
256-row sweep, and nothing else:

  1. Cost per row, extrapolated to 256.
  2. The V0 distribution gate -- is the score graded, or piled onto the endpoint
     bins? A ties-dominated score compresses every rank-based metric, so this
     gates whether any score claim may be reported at all.
  3. Truncation rate. On gemini-2.5-flash at zero thinking, visible output alone
     hit max_output_tokens in 6 of 256 episodes (~1.6%). At HIGH thinking, thinking
     competes for the same budget and thought parts are replayed into context each
     turn, so 1.6% is a fact about that model, not a constant.

Plus first-look-only score diagnostics (AUC, paired-by-gt_sample ranking,
score/binary inconsistency). At probe scale these are UNDERPOWERED and printed
as orientation, not findings -- the pass/fail thresholds in
artifacts/agentic-belief-revision-256-score/hypothesis_sanity_checks.md are
written for the full 256-row run.

Usage:
  python frontier/score_probe_report.py --jsonl <stage2.jsonl> [--stage1 <cache.jsonl>]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frontier.model_pricing import token_cost  # noqa: E402
# Reuse the existing invalid->valid mod_type mapping rather than duplicating it.
from frontier.judge_invalidity_reasoning import INVALID_TO_VALID_MOD_TYPE  # noqa: E402

FULL_SWEEP_ROWS = 256


def load(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def auc(pos_scores: list, neg_scores: list) -> float | None:
    """AUC = P(a random positive outscores a random negative), ties at 0.5.
    Equivalent to a normalized Mann-Whitney U. Threshold-free, and rank-only --
    which is why it is the right lens on a lumpy 0-10 integer score, and the only
    legitimate way to compare across models later."""
    if not pos_scores or not neg_scores:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos_scores for n in neg_scores)
    return wins / (len(pos_scores) * len(neg_scores))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="Stage-2 JSONL for one condition")
    ap.add_argument("--stage1", default=None, help="Stage-1 cache JSONL (for S1 scores)")
    ap.add_argument("--dataset", default=None,
                    help="Dataset CSV, used to backfill gt_sample by title for rows "
                         "written before the harness persisted it.")
    args = ap.parse_args()

    rows = load(args.jsonl)

    # Backfill gt_sample for rows written before the harness persisted it. The
    # pairing is the primary score statistic, so recovering it by title join is
    # worth doing rather than reporting "0 pairs".
    if args.dataset and any(not r.get("gt_sample") for r in rows):
        import csv
        with open(args.dataset) as f:
            by_title = {row["title"]: row.get("gt_sample") for row in csv.DictReader(f)}
        filled = 0
        for r in rows:
            if not r.get("gt_sample") and r.get("title") in by_title:
                r["gt_sample"] = by_title[r["title"]]
                filled += 1
        if filled:
            print(f"(backfilled gt_sample for {filled} row(s) from {args.dataset})")

    live = [r for r in rows if not r.get("aborted")]
    model = rows[0].get("model") if rows else "unknown"
    cond = rows[0].get("condition") if rows else "unknown"

    print("=" * 74)
    print(f"SCORE PROBE REPORT — {model} @ {cond} — {len(rows)} rows "
          f"({len(rows) - len(live)} aborted)")
    print("=" * 74)

    # ── 1. Cost ──────────────────────────────────────────────────────────────
    print("\n[1] COST")
    tot = sum(r.get("total_cost_usd") or 0.0 for r in rows)
    s2_tok = [(r.get("s2_input_tokens") or 0, r.get("s2_output_tokens") or 0,
               r.get("s2_think_tokens") or 0) for r in live]
    per_row = tot / len(rows) if rows else 0.0
    print(f"  measured total            ${tot:.4f} over {len(rows)} rows")
    print(f"  mean cost per row         ${per_row:.4f}")
    print(f"  --> extrapolated to {FULL_SWEEP_ROWS} rows   ${per_row * FULL_SWEEP_ROWS:.2f}")
    if s2_tok:
        mi = sum(t[0] for t in s2_tok) / len(s2_tok)
        mo = sum(t[1] for t in s2_tok) / len(s2_tok)
        mt = sum(t[2] for t in s2_tok) / len(s2_tok)
        print(f"  mean S2 tokens            input {mi:,.0f}  output {mo:,.0f}  thinking {mt:,.0f}")
        in_cost = token_cost(model, int(mi), 0, 0)
        out_cost = token_cost(model, 0, int(mo), int(mt))
        print(f"  cost split                input {100*in_cost/(in_cost+out_cost):.0f}%  "
              f"output+thinking {100*out_cost/(in_cost+out_cost):.0f}%")
    acts = [r.get("s2_action_count") or 0 for r in live]
    if acts:
        print(f"  actions used              mean {sum(acts)/len(acts):.2f}  "
              f"max {max(acts)}  (budget bounds the worst case)")
        print(f"  NOTE: rows that stop early cost less. If the full sweep uses more of the")
        print(f"        budget than this probe did, the extrapolation is an UNDER-estimate.")

    # ── 2. V0 distribution gate ──────────────────────────────────────────────
    print("\n[2] V0 DISTRIBUTION GATE (score must be graded, not degenerate)")
    s1_scores = []
    if args.stage1:
        s1_scores = [r.get("s1_validity_score") for r in load(args.stage1)]
    stages = [("S1", [s for s in s1_scores if s is not None]),
              ("S2", [r.get("s2_validity_score") for r in live
                      if r.get("s2_validity_score") is not None])]
    for label, scores in stages:
        if not scores:
            print(f"  {label}: no scores present")
            continue
        n = len(scores)
        hist = Counter(scores)
        ends = sum(v for k, v in hist.items() if k <= 1 or k >= 9)
        top = max(hist.values())
        distinct = len(hist)
        bar = "  ".join(f"{k}:{hist.get(k,0)}" for k in range(11))
        print(f"  {label} (n={n})  {bar}")
        checks = [("mass in end bins (0-1, 9-10) <= 70%", ends / n <= 0.70, f"{100*ends/n:.0f}%"),
                  ("no single value > 50%",               top / n <= 0.50, f"{100*top/n:.0f}%"),
                  (">= 4 distinct values used",           distinct >= 4,   str(distinct))]
        for name, ok, detail in checks:
            print(f"       {'PASS' if ok else 'FAIL'}  {name}  [{detail}]")
        print(f"       verdict: {'GATE PASSES' if all(c[1] for c in checks) else 'GATE FAILS -> score claims not reportable'}")

    # score compliance
    miss = sum(1 for r in live if r.get("s2_validity_score") is None)
    print(f"  S2 score non-compliance: {miss}/{len(live)} "
          f"({100*miss/len(live) if live else 0:.1f}%)  [plan skips V-checks above 5%]")

    # ── 3. Truncation ────────────────────────────────────────────────────────
    print("\n[3] TRUNCATION (labelled, not counted as the model declining to act)")
    turns = [(r, a) for r in rows for a in (r.get("action_trace") or [])]
    trunc = [(r, a) for r, a in turns if a.get("truncated")]
    eps = {r["title"] for r, _ in trunc}
    fr = Counter(a.get("finish_reason") for _, a in turns)
    print(f"  turns total               {len(turns)}")
    print(f"  truncated turns           {len(trunc)} ({100*len(trunc)/len(turns) if turns else 0:.1f}% of turns)")
    print(f"  episodes affected         {len(eps)}/{len(rows)} "
          f"({100*len(eps)/len(rows) if rows else 0:.1f}%)   [2.5@tb0 baseline: 1.6%]")
    print(f"  finish_reason breakdown   {dict(fr)}")
    nonact = sum(1 for _, a in turns if a.get("text_only") or a.get("empty_response"))
    print(f"  non-action turns          {nonact}  of which truncations: {len(trunc)}")
    print(f"       (any 'model declined to act' statistic must exclude the truncations)")
    maxtok = max((a.get("output_tokens") or 0) + (a.get("think_tokens") or 0)
                 for _, a in turns) if turns else 0
    print(f"  max single-turn out+think {maxtok:,} against the 50,000 cap")

    # ── First-look score diagnostics (UNDERPOWERED at probe scale) ───────────
    print("\n[4] SCORE DIAGNOSTICS — orientation only, underpowered at probe scale")
    scored = [r for r in live if r.get("s2_validity_score") is not None]
    pos = [r["s2_validity_score"] for r in scored if r.get("gt_valid") is True]
    neg = [r["s2_validity_score"] for r in scored if r.get("gt_valid") is False]
    a = auc(pos, neg)
    print(f"  AUC(score vs gt_valid)    {a if a is None else f'{a:.3f}'}   "
          f"(valid n={len(pos)}, invalid n={len(neg)})")
    print(f"  mean score                valid {sum(pos)/len(pos) if pos else float('nan'):.2f}  "
          f"invalid {sum(neg)/len(neg) if neg else float('nan'):.2f}")
    binacc = [r.get("s2_valid_match") for r in live if r.get("s2_valid_match") is not None]
    print(f"  binary accuracy           {sum(binacc)/len(binacc) if binacc else float('nan'):.3f}")

    # Paired: does the clean member outscore its own defective twin?
    #
    # The pair key is (gt_sample, mod_type PAIRING) -- not gt_sample alone. Each
    # gt_sample carries four valid mod_types and four invalid ones, so keying on
    # gt_sample alone silently collapses eight rows into a single pair and throws
    # away most of the comparison. The mapping comes from the judge module.
    pair_of = {inv: val for inv, val in INVALID_TO_VALID_MOD_TYPE.items()}
    by_pair = defaultdict(dict)
    for r in scored:
        gs, mt = r.get("gt_sample"), r.get("mod_type")
        if not gs or not mt:
            continue
        if mt in pair_of:                      # an invalid mod_type
            by_pair[(gs, pair_of[mt])][False] = r["s2_validity_score"]
        else:                                  # its valid counterpart
            by_pair[(gs, mt)][True] = r["s2_validity_score"]
    pairs = [(v[True], v[False]) for v in by_pair.values() if True in v and False in v]
    unpaired = len(scored) - 2 * len(pairs)
    if pairs:
        hi = sum(1 for a_, b_ in pairs if a_ > b_)
        tie = sum(1 for a_, b_ in pairs if a_ == b_)
        inv = sum(1 for a_, b_ in pairs if a_ < b_)
        denom = hi + inv
        print(f"  matched pairs             {len(pairs)}  ({unpaired} scored rows unpaired)")
        print(f"    clean-higher {hi}  tied {tie}  inverted {inv}"
              + (f"   ratio {hi/denom:.2f} (plan threshold 0.65 at full scale)" if denom else ""))
        if tie:
            print(f"    NOTE: {tie} tie(s) -- ties are excluded from the ratio and "
                  f"compress AUC; a high tie count is a V0-adjacent warning.")
    else:
        print("  matched pairs             0 (no valid/invalid twins in this subset)")

    # score/binary inconsistency: flagged invalid yet scored as basically fine
    flagged_fine = [r for r in scored
                    if str(r.get("s2_submit_args", {}).get("valid", "")).lower().startswith("n")
                    and r["s2_validity_score"] >= 8]
    passed_broken = [r for r in scored
                     if str(r.get("s2_submit_args", {}).get("valid", "")).lower().startswith("y")
                     and r["s2_validity_score"] <= 3]
    print(f"  valid=No  but score >= 8  {len(flagged_fine)}  <- 'flagged it, also think it is fine'")
    print(f"  valid=Yes but score <= 3  {len(passed_broken)}  <- watch: may indicate scale misreading")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
