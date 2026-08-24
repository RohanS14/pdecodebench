"""
aggregate_freegen.py — concatenate per-model free-generation JSONLs into one CSV.

Every viz script reads a single flat CSV. That file used to be assembled by hand,
and the hand-assembled one did not match its own inputs: results/pde_llm_eval.csv
holds 992 rows while results_eval_v3/*.jsonl holds 1424, so the published figures
were built from a partial join with no way to notice. This writes the step down
and makes a short model loud.

Usage:
    python freegen/aggregate_freegen.py \
        --results_dir results_jul28 \
        --out         results/pde_llm_eval_jul28.csv \
        --expect_rows 256
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import pandas as pd


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_jul28")
    ap.add_argument("--out",         default="results/pde_llm_eval_jul28.csv")
    ap.add_argument("--expect_items", type=int, default=256,
                    help="ITEMS per model. jul28 = 256 (32 gt_samples x 8 mod_types). "
                         "Rows are items x k, where k is read from the data. Set 0 to "
                         "skip the completeness check (canary subsets).")
    ap.add_argument("--expect_rows", type=int, default=None,
                    help="Deprecated alias for --expect_items, kept so existing "
                         "invocations keep working. A k=3 arm has 3x this many ROWS; "
                         "the check below multiplies by the k it finds in the data "
                         "rather than requiring the caller to know it.")
    ap.add_argument("--allow_incomplete", action="store_true",
                    help="Warn instead of failing when a model is short. Use for "
                         "partial harvests mid-run; never for a final artifact.")
    args = ap.parse_args()
    if args.expect_rows is not None:
        args.expect_items = args.expect_rows

    # RECURSIVE. The one-job-per-model launcher gives each model its own OUTPUT_DIR
    # subdirectory -- not tidiness, but because upload_helper globs a whole results
    # dir and push_dataset_to_hub REPLACES the split, so two concurrent jobs sharing
    # one directory would overwrite each other's uploads. A flat glob finds nothing
    # under that layout, and would find only SOME arms under a half-migrated one,
    # which is the quiet failure: a CSV that looks fine and is missing three models.
    paths = sorted(glob.glob(os.path.join(args.results_dir, "**", "*.jsonl"),
                             recursive=True))
    if not paths:
        sys.exit(f"[aggregate] No *.jsonl under {args.results_dir}")

    rows = [r for p in paths for r in load_jsonl(p)]
    df = pd.DataFrame(rows)
    print(f"[aggregate] {len(df)} rows from {len(paths)} file(s)")
    for p in paths:
        print(f"[aggregate]   {os.path.relpath(p, args.results_dir)}")

    # ── Completeness, per model ───────────────────────────────────────────────
    # Keyed by ARM, not model: two reasoning arms of one model are 2x expect_rows
    # and would both trip the completeness check and the duplicate check below.
    arm = (df["model"] + " [thinking=" + df.get("thinking", "off").astype(str) + "]"
           if "thinking" in df else df["model"])
    # k is READ FROM THE DATA, per arm, not assumed. An arm sampled k=3 has three
    # rows per item and a k=1 arm has one; hardcoding 256 made every k=3 arm look
    # 3x over-complete and failed the run before it could produce a CSV.
    df["_arm"] = arm
    counts = Counter(arm)
    if "sample_idx" in df:
        k_by_arm = df.groupby("_arm")["sample_idx"].nunique().to_dict()
    else:
        k_by_arm = {a: 1 for a in counts}
    short = []
    for model, n in sorted(counts.items()):
        k = int(k_by_arm.get(model, 1)) or 1
        want = args.expect_items * k
        flag = ""
        if args.expect_items and n != want:
            flag = f"  <-- expected {want} ({args.expect_items} items x k={k})"
            short.append((model, n, want))
        else:
            flag = f"  ({args.expect_items} items x k={k})" if k > 1 else ""
        print(f"[aggregate]   {n:>5}  {model}{flag}")

    if short and not args.allow_incomplete:
        sys.exit(f"[aggregate] FAIL: {len(short)} arm(s) short: {short}. "
                 f"Re-run those models (run_eval.py resumes) or pass "
                 f"--allow_incomplete.")

    # ── Health checks that must never be silent ───────────────────────────────
    # sample_idx is part of the row identity under k>1. Without it every legitimate
    # draw 1 and draw 2 counted as a duplicate of draw 0, and the message sent the
    # reader looking for a stale JSONL that was not there.
    dedup_keys = (["title", "mod_type", "model"]
                  + (["thinking"] if "thinking" in df else [])
                  + (["sample_idx"] if "sample_idx" in df else []))
    dupes = df.duplicated(subset=dedup_keys).sum()
    if dupes:
        sys.exit(f"[aggregate] FAIL: {dupes} duplicate {tuple(dedup_keys)} rows — "
                 f"almost always a stale JSONL from a different dataset in {args.results_dir}.")

    # Draws of one item are NOT independent observations. Everything downstream that
    # computes an interval has to pool them first; this states the structure once so
    # a consumer cannot mistake row count for sample size.
    if "sample_idx" in df:
        n_items = df.groupby(["model", "mod_type", "title"]).ngroups
        print(f"[aggregate] {len(df)} rows = {n_items} items x k draws. "
              f"ROWS ARE NOT INDEPENDENT: pool per (title, mod_type, model, "
              f"thinking) before any confidence interval.")
    if "no_verdict" in df:
        nv = int(df["no_verdict"].fillna(False).astype(bool).sum())
        print(f"[aggregate] {nv} row(s) with no verdict "
              f"({nv / max(1, len(df)):.1%}) — these reached no answer and must be "
              f"dropped, not scored.")

    if "finish_reason" in df:
        trunc = df[df["finish_reason"] == "length"]
        if len(trunc):
            print(f"[aggregate] WARNING: {len(trunc)} truncated response(s) — a "
                  f"'length' finish_reason is a failed row, not a datum:")
            for _, r in trunc.head(10).iterrows():
                print(f"[aggregate]   {r['model']}  {r['title']}")

    for col in ("gt_sample", "source", "valid_conf"):
        if col not in df.columns:
            print(f"[aggregate] WARNING: column '{col}' missing — these JSONLs "
                  f"predate the jul28 fields; source/hedge cuts will not work.")
        elif df[col].isna().any():
            print(f"[aggregate] WARNING: {int(df[col].isna().sum())} null '{col}' values")

    if "dataset" in df.columns and df["dataset"].nunique() > 1:
        sys.exit(f"[aggregate] FAIL: rows come from more than one dataset: "
                 f"{sorted(df['dataset'].unique())}. Titles collide between dataset "
                 f"versions, so mixing them silently corrupts every per-condition mean.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.drop(columns=[c for c in ('_arm',) if c in df], inplace=True)
    df.to_csv(args.out, index=False)
    print(f"[aggregate] Wrote {len(df)} rows x {len(df.columns)} cols -> {args.out}")


if __name__ == "__main__":
    main()
