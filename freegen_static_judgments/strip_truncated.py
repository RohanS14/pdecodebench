"""
strip_truncated.py — remove censored draws so a rescue run re-generates them.

A draw that stopped at finish_reason "length" produced no verdict and is dropped
downstream, so it is censored data rather than a measurement. run_eval.py re-queues
an item when ANY of its k draws is missing from the output, and skips writing draws
that are already there -- so deleting exactly the censored rows is enough to make a
re-run regenerate them and nothing else.

Writes a .pretruncfix backup beside every file it touches, in the same style as
rescore_jsonl.py, so the censored rows remain inspectable after the rescue.

Usage:
    python freegen_static_judgments/strip_truncated.py --results_dir outputs/freegen_xmodal [--dry_run]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter


def loop_fraction(text: str) -> float:
    """How much of the tail is verbatim-repeated lines.

    A truncated draw has two very different causes and only one of them is worth
    re-running. A trace still reasoning coherently when it hit the ceiling will
    finish if given more room. A trace stuck emitting the same lines forever will
    emit them until ANY ceiling, so re-running it at a larger budget buys nothing
    and costs the compute of generating to the new cap.

    Line-level, not fixed-offset block hashing: measured on this roster, 160-char
    block hashing scored 59 of 68 truncations as "still reasoning" when a line-level
    count puts 63 of 68 in loops. Blocks only match when the loop's period happens to
    align to the block grid, and most do not.
    """
    lines = [re.sub(r"\s+", " ", x).strip() for x in text[-30000:].split("\n")]
    lines = [x for x in lines if len(x) > 25]
    if len(lines) < 20:
        return 0.0
    return 1 - len(Counter(lines)) / len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--loop_threshold", type=float, default=0.5,
                    help="Tail line-repetition at or above which a truncated draw "
                         "is treated as a degenerate loop and LEFT IN PLACE: more "
                         "budget cannot rescue it. Set 1.1 to strip every "
                         "truncation regardless.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "**", "*.jsonl"),
                             recursive=True))
    paths = [p for p in paths if not p.endswith(".prerescore")]
    if not paths:
        sys.exit(f"[strip] No *.jsonl in {args.results_dir}")

    grand = Counter()
    for path in paths:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        keep, drop, looped = [], [], []
        for r in rows:
            if str(r.get("finish_reason", "")).lower() != "length":
                keep.append(r)
            elif loop_fraction(r.get("model_response") or "") >= args.loop_threshold:
                # Still censored and still dropped downstream -- just not worth the
                # GPU time of re-running, so it stays on disk untouched.
                looped.append(r)
                keep.append(r)
            else:
                drop.append(r)
        if not drop:
            print(f"[strip] {os.path.basename(path)}: nothing re-runnable "
                  f"({len(looped)} looped truncation(s) left in place)")
            continue

        # Report the items that will be re-queued, not just the draw count: a
        # re-run regenerates all k draws of an affected item and writes only the
        # missing ones, so items is the number that costs GPU time.
        items = {(r["title"], r["mod_type"]) for r in drop}
        model = drop[0].get("model", "?")
        grand[model] += len(drop)
        print(f"[strip] {os.path.basename(path)}: dropping {len(drop)} re-runnable "
              f"censored draw(s) across {len(items)} item(s); {len(keep)} rows kept "
              f"({len(looped)} looped truncation(s) left in place)")

        if not args.dry_run:
            shutil.copy(path, path + ".pretruncfix")
            with open(path, "w") as f:
                for r in keep:
                    f.write(json.dumps(r) + "\n")

    print(f"\n[strip] censored draws removed per model: {dict(grand)}")
    print(f"[strip] total: {sum(grand.values())}")
    if args.dry_run:
        print("[strip] DRY RUN - nothing written")
    else:
        print("[strip] originals kept alongside as *.pretruncfix")


if __name__ == "__main__":
    main()
