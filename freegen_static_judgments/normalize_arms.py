"""
normalize_arms.py — one file per model, one schema, no duplicates.

A truncation rescue leaves an arm RAGGED. Rows written before `max_tokens` and
`verdict_recovered` existed carry 37 columns; rows written by the rescue carry 39.
Concatenated into one table the missing cells become null, and a null `max_tokens`
reads as "no ceiling" when it actually means "written before the column existed" --
which is precisely backwards for the rows whose ceiling is the interesting part.

So the gap is filled with the TRUE value rather than left null:

  max_tokens         the budget this arm was generated under. Rows from a rescue
                     already carry theirs and are never overwritten; everything else
                     gets the model table's entry, which is what it actually ran at.
  verdict_recovered  re-derived from the stored response for every row, so the flag
                     means the same thing across an arm regardless of when the row
                     was written.

Verifies rather than assumes on the way out: uniform key set, no duplicate
(title, mod_type, sample_idx), and the expected row count.

Usage:
    python freegen_static_judgments/normalize_arms.py --results_dir outputs/freegen_xmodal \
        [--expect_rows 768] [--dry_run]
"""
import argparse
import glob
import json
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_score import recovered_loop_verdict  # noqa: E402
from run_eval import arm_max_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--expect_rows", type=int, default=768)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(
        os.path.join(args.results_dir, "**", "*.jsonl"), recursive=True)
        if not p.endswith((".prerescore", ".pretruncfix")))
    if not paths:
        sys.exit(f"[normalize] No *.jsonl in {args.results_dir}")

    failures = []
    for path in paths:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            failures.append(f"{os.path.basename(path)}: empty")
            continue

        model = rows[0]["model"]
        thinking = rows[0].get("thinking", "on")
        table_budget = arm_max_tokens(model, thinking)

        filled = Counter()
        for r in rows:
            if r.get("max_tokens") in (None, 0):
                r["max_tokens"] = int(table_budget)
                filled["max_tokens"] += 1
            rec = recovered_loop_verdict(r.get("model_response") or "",
                                         r.get("finish_reason", ""))
            if r.get("verdict_recovered") != (rec is not None):
                filled["verdict_recovered"] += 1
            r["verdict_recovered"] = rec is not None

        # Rewrite every row through one key order so the file is uniform on disk and
        # not merely uniform once a reader has normalised it.
        key_order = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in key_order:
                    key_order.append(k)
        rows = [{k: r.get(k) for k in key_order} for r in rows]

        keysets = {tuple(r.keys()) for r in rows}
        dupes = [k for k, c in Counter(
            (r["title"], r["mod_type"], r["sample_idx"]) for r in rows).items() if c > 1]

        name = os.path.basename(path)
        status = []
        if len(keysets) != 1:
            failures.append(f"{name}: {len(keysets)} distinct key sets")
            status.append("RAGGED")
        if dupes:
            failures.append(f"{name}: {len(dupes)} duplicate (item, sample_idx)")
            status.append(f"{len(dupes)} DUPES")
        if args.expect_rows and len(rows) != args.expect_rows:
            failures.append(f"{name}: {len(rows)} rows, expected {args.expect_rows}")
            status.append(f"{len(rows)}/{args.expect_rows} rows")

        budgets = dict(Counter(r["max_tokens"] for r in rows))
        recovered = sum(1 for r in rows if r["verdict_recovered"])
        print(f"[normalize] {name}")
        print(f"              rows={len(rows)} cols={len(key_order)} "
              f"budgets={budgets} recovered={recovered} "
              f"filled={dict(filled)}  {' '.join(status) or 'OK'}")

        if not args.dry_run:
            shutil.copy(path, path + ".prenormalize")
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    print()
    if failures:
        print("[normalize] PROBLEMS:")
        for f in failures:
            print(f"[normalize]   {f}")
        sys.exit(1)
    print(f"[normalize] {len(paths)} arm(s): uniform schema, no duplicates, "
          f"{args.expect_rows} rows each")
    if args.dry_run:
        print("[normalize] DRY RUN - nothing written")


if __name__ == "__main__":
    main()
