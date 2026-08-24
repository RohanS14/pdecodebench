"""
validate_rows.py — health check for free-generation JSONL arms.

Written because "768 rows exist" is not the same as "768 usable rows exist". The
checks below are exactly the ones the red-team brief asks for, in one place, so a
partial harvest can be judged without re-deriving them each time:

  completeness  items x k rows, no duplicate (title, mod_type, model, thinking, sample_idx)
  sampling      one and only one (temperature, top_p, top_k, sampling_seed, k_draws) per arm
  truncation    finish_reason == "length" is a FAILED draw, not a short one
  no_verdict    parse produced nothing usable
  diversity     k draws at one temperature must not be textually identical

Usage:
    python freegen/validate_rows.py <dir-or-file> [...] --expect_items 256 --k 3
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "sampling_seed", "k_draws")
ITEM_KEY = ("title", "mod_type", "model", "thinking")


def load(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_arm(path, expect_items, k, thinking):
    rows = list(load(path))
    name = os.path.basename(path)
    fails, warns = [], []

    n_rows = len(rows)
    items = {tuple(r.get(x) for x in ITEM_KEY) for r in rows}
    dupes = [key for key, c in Counter(
        tuple(r.get(x) for x in ITEM_KEY) + (r.get("sample_idx"),) for r in rows
    ).items() if c > 1]

    if dupes:
        fails.append(f"{len(dupes)} duplicate (item, sample_idx) key(s), e.g. {dupes[0]}")
    if expect_items and len(items) != expect_items:
        (fails if len(items) > expect_items else warns).append(
            f"{len(items)} distinct items, expected {expect_items}"
            f" ({expect_items - len(items)} short)")
    if expect_items and n_rows != expect_items * k:
        warns.append(f"{n_rows} rows, expected {expect_items * k}")

    # Sampling parameters must be constant across the arm. A mixed arm means two
    # runs with different settings were concatenated, which no downstream pooling
    # can undo.
    for key in SAMPLING_KEYS:
        vals = {r.get(key) for r in rows}
        if len(vals) != 1:
            fails.append(f"{key} is not constant: {sorted(map(str, vals))}")
    if thinking:
        vals = {r.get("thinking") for r in rows}
        if vals != {thinking}:
            fails.append(f"thinking={sorted(map(str, vals))}, expected {{{thinking}}}")

    idxs = Counter(r.get("sample_idx") for r in rows)
    if set(idxs) != set(range(k)):
        fails.append(f"sample_idx values {sorted(map(str, idxs))}, expected {list(range(k))}")

    trunc = [r for r in rows if r.get("finish_reason") == "length"]
    nv = [r for r in rows if r.get("no_verdict")]
    if trunc:
        fails.append(f"{len(trunc)} truncated draw(s) ({100*len(trunc)/n_rows:.1f}%)"
                     f", e.g. {trunc[0].get('title')} draw {trunc[0].get('sample_idx')}")
    if nv:
        fails.append(f"{len(nv)} no_verdict row(s) ({100*len(nv)/n_rows:.1f}%)")

    # k draws that are byte-identical mean sampling silently collapsed to greedy.
    by_item = defaultdict(list)
    for r in rows:
        by_item[tuple(r.get(x) for x in ITEM_KEY)].append(r.get("model_response") or "")
    full = [v for v in by_item.values() if len(v) == k]
    varied = sum(1 for v in full if len(set(v)) == k)
    if full and varied == 0:
        fails.append(f"no item has {k} textually distinct draws — sampling may be collapsed")
    elif full and varied < 0.5 * len(full):
        warns.append(f"only {varied}/{len(full)} complete items have {k} distinct draws")

    sample = rows[0] if rows else {}
    print(f"\n{name}")
    print(f"  rows={n_rows}  items={len(items)}  k={sample.get('k_draws')}  "
          f"T={sample.get('temperature')} top_p={sample.get('top_p')} "
          f"top_k={sample.get('top_k')} seed={sample.get('sampling_seed')}")
    print(f"  sample_idx counts: {dict(sorted(idxs.items(), key=lambda kv: str(kv[0])))}")
    print(f"  truncated={len(trunc)}  no_verdict={len(nv)}  "
          f"items with {k} distinct draws: {varied}/{len(full)}")
    for w in warns:
        print(f"  WARN  {w}")
    for f in fails:
        print(f"  FAIL  {f}")
    if not fails and not warns:
        print("  OK")
    return fails, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSONL files, or dirs to glob *.jsonl under")
    ap.add_argument("--expect_items", type=int, default=256,
                    help="items per arm (32 gt_samples x 8 mod_types); 0 to skip")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--thinking", default="on", help="expected arm; empty to skip")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
                     if os.path.isdir(p) else [p])
    if not files:
        sys.exit("[validate] no JSONL found")

    n_fail = 0
    for path in files:
        fails, _ = check_arm(path, args.expect_items, args.k, args.thinking)
        n_fail += bool(fails)
    print(f"\n{len(files) - n_fail}/{len(files)} arm(s) clean")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
