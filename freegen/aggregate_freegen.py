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
    ap.add_argument("--expect_rows", type=int, default=256,
                    help="Rows per model. jul28 = 256 (32 gt_samples x 8 mod_types). "
                         "Set 0 to skip the completeness check (canary subsets).")
    ap.add_argument("--allow_incomplete", action="store_true",
                    help="Warn instead of failing when a model is short. Use for "
                         "partial harvests mid-run; never for a final artifact.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))
    if not paths:
        sys.exit(f"[aggregate] No *.jsonl in {args.results_dir}")

    rows = [r for p in paths for r in load_jsonl(p)]
    df = pd.DataFrame(rows)
    print(f"[aggregate] {len(df)} rows from {len(paths)} file(s)")

    # ── Completeness, per model ───────────────────────────────────────────────
    # Keyed by ARM, not model: two reasoning arms of one model are 2x expect_rows
    # and would both trip the completeness check and the duplicate check below.
    arm = (df["model"] + " [thinking=" + df.get("thinking", "off").astype(str) + "]"
           if "thinking" in df else df["model"])
    counts = Counter(arm)
    short = []
    for model, n in sorted(counts.items()):
        flag = ""
        if args.expect_rows and n != args.expect_rows:
            flag = f"  <-- expected {args.expect_rows}"
            short.append((model, n))
        print(f"[aggregate]   {n:>5}  {model}{flag}")

    if short and not args.allow_incomplete:
        sys.exit(f"[aggregate] FAIL: {len(short)} model(s) short of "
                 f"{args.expect_rows} rows: {short}. Re-run those models "
                 f"(run_eval.py resumes) or pass --allow_incomplete.")

    # ── Health checks that must never be silent ───────────────────────────────
    dedup_keys = ["title", "mod_type", "model"] + (["thinking"] if "thinking" in df else [])
    dupes = df.duplicated(subset=dedup_keys).sum()
    if dupes:
        sys.exit(f"[aggregate] FAIL: {dupes} duplicate {tuple(dedup_keys)} rows — "
                 f"almost always a stale JSONL from a different dataset in {args.results_dir}.")

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
    df.to_csv(args.out, index=False)
    print(f"[aggregate] Wrote {len(df)} rows x {len(df.columns)} cols -> {args.out}")


if __name__ == "__main__":
    main()
