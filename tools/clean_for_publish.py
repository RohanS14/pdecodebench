"""
clean_for_publish.py — drop repair-chain bookkeeping and make one schema per arm.

A results tree that has been through a repair chain carries columns describing HOW a
row was produced rather than WHAT was measured: which pass it came from, how many
tokens a discarded redraw burned, which budget a backfill used. Those are useful
while the campaign is running and misleading afterwards, for one specific reason --
they are present on a MINORITY of rows, so a null does not mean "this did not
happen", it means "the pass that wrote this row did not track it". A reader cannot
tell those apart, and neither can a groupby.

Measured on cross_modal_consistency: source_arm on 24576/24576 rows but naming pass
directories that no longer exist, redraw_* on 1274, backfill_* on 331-1029.

Dropping them is safe because the repair history is preserved in full under
_trash/<date>/ -- this removes it from the PUBLISHED artifact, not from the record.

Every surviving column is then written on every row, in one key order, so the file
is uniform on disk rather than uniform only after a reader normalises it.

Usage:
    python tools/clean_for_publish.py --results_dir <dir> --out_dir <dir> \
        [--drop col ...] [--expect_rows N]
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

# Bookkeeping from the repair chain. Not measurements.
DEFAULT_DROP = [
    "source_arm",
    "redrawn", "redraw_attempt", "redraw_seed",
    "redraw_discarded_tokens", "redraw_still_looping",
    "backfilled", "backfill_skipped",
    "backfill_from_tokens", "backfill_budget", "backfill_seed",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--drop", nargs="*", default=None)
    ap.add_argument("--expect_rows", type=int, default=0)
    ap.add_argument("--key", nargs="*", default=None,
                    help="Identity columns for the duplicate check.")
    args = ap.parse_args()
    drop = set(args.drop if args.drop is not None else DEFAULT_DROP)

    paths = sorted(p for p in glob.glob(
        os.path.join(args.results_dir, "**", "*.jsonl"), recursive=True)
        if not p.endswith((".prerescore", ".pretruncfix", ".prenormalize")))
    if not paths:
        sys.exit(f"[clean] no *.jsonl under {args.results_dir}")

    # One key order for the WHOLE tree, not per file: the arms are concatenated into
    # a single published table, and a per-file order would still produce a ragged
    # union at load time.
    order, seen = [], set()
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    for k in json.loads(line):
                        if k not in drop and k not in seen:
                            seen.add(k); order.append(k)
                    break
    for p in paths:
        for line in open(p):
            if line.strip():
                for k in json.loads(line):
                    if k not in drop and k not in seen:
                        seen.add(k); order.append(k)

    dropped_found = Counter()
    total = 0
    problems = []
    for p in paths:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        for r in rows:
            for k in list(r):
                if k in drop:
                    dropped_found[k] += 1
        out_rows = [{k: r.get(k) for k in order} for r in rows]

        if args.key:
            dupes = [k for k, c in Counter(
                tuple(r.get(x) for x in args.key) for r in out_rows).items() if c > 1]
            if dupes:
                problems.append(f"{os.path.basename(p)}: {len(dupes)} duplicate keys")
        if args.expect_rows and len(out_rows) != args.expect_rows:
            problems.append(f"{os.path.basename(p)}: {len(out_rows)} rows, "
                            f"expected {args.expect_rows}")

        rel = os.path.relpath(p, args.results_dir)
        dest = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        total += len(out_rows)
        print(f"[clean] {rel}: {len(out_rows)} rows -> {len(order)} cols")

    print(f"\n[clean] dropped columns and the rows that carried them:")
    for k in sorted(drop):
        print(f"[clean]   {k:30s} {dropped_found.get(k, 0)}")
    print(f"[clean] {total} rows, {len(order)} uniform columns, {len(paths)} arm(s)")
    if problems:
        print("[clean] PROBLEMS:")
        for x in problems:
            print(f"[clean]   {x}")
        sys.exit(1)


if __name__ == "__main__":
    main()
