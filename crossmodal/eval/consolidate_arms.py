"""consolidate_arms.py — one file per model, as if the model had been run once.

The repair history is spread across arms: a source run, then one or more backfill
passes that each copied the good rows through and rewrote the failures. Reading a
single arm therefore always loses something -- the source arm has none of the
repairs, and a repair arm killed mid-run is missing draws entirely. That is why the
report kept showing a stale count with an arrow beside it.

This merges them on (item_id, sample_idx), which is the identity of a draw, and
keeps the BEST row for each: a draw that reached a verdict beats one that did not,
and among equals the later pass wins. The result is a full k=3 arm per model that
reads exactly like a single run, with a `source_arm` column recording which pass
each row actually came from so the provenance is never lost.

DELIBERATELY EXCLUDED -- different token budgets are different treatments, not
repairs, and merging them would silently mix decoding regimes inside one arm:
  nemotron-3-nano-30b-64k / -128k   budget sweeps, superseded
  qwen3-8-27b__32k_budget_abandoned  abandoned at 32k, replaced by the 64k run

Repaired rows DID get a larger budget than the originals they replace -- that is
what the repair is. It is recorded per row rather than hidden: a draw that already
terminated would have terminated identically with more room, so the budget only
ever converted a non-answer into an answer.

    python consolidate_arms.py [--out DIR] [--dry_run]
"""
import argparse
import collections
import json
import os

GEN = "/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen"

# Merge order: EARLIEST first, so "later pass wins" falls out of iteration order.
CHAINS = {
    "qwen3-32b":           ["qwen3-32b", "qwen3-32b-backfill", "qwen3-32b-backfill2"],
    "r1-distill-qwen-32b": ["r1-distill-qwen-32b", "r1-distill-qwen-32b-backfill",
                            "r1-distill-qwen-32b-backfill2"],
    "qwq-32b":             ["qwq-32b", "qwq-32b-backfill", "qwq-32b-backfill2",
                            "qwq-32b-backfill3"],
    "qwen3-5-27b":         ["qwen3-5-27b", "qwen3-5-27b-backfill", "qwen3-5-27b-backfill2"],
    "qwen3-6-27b":         ["qwen3-6-27b", "qwen3-6-27b-backfill", "qwen3-6-27b-backfill2",
                            "qwen3-6-27b-backfill3"],
    "glm-4-7-flash":       ["glm-4-7-flash", "glm-4-7-flash-backfill",
                            "glm-4-7-flash-backfill2", "glm-4-7-flash-backfill3"],
    "nemotron-3-nano-30b": ["nemotron-3-nano-30b", "nemotron-3-nano-30b-backfill",
                            "nemotron-3-nano-30b-backfill2"],
    "qwen3-8-27b":         ["qwen3-8-27b", "qwen3-8-27b-backfill", "qwen3-8-27b-backfill2"],
}
N_ITEMS, K = 1024, 3

# LIVE REPAIR ARMS -- discovered rather than listed. A repair pass that is still
# running writes <slug>-rep*, and <slug>-final is the merge of everything before it,
# so appending them in that order lets this be run mid-flight for a partial harvest.
# The torn-tail guard in the read loop already tolerates a file growing underneath
# us, and the "later pass wins only if at least as good" rule means a half-finished
# repair arm can only improve a draw, never demote one.
for _slug in list(CHAINS):
    for _suffix in ("-final", "-rep", "-rep0", "-rep1", "-rep2", "-rep3"):
        _arm = _slug + _suffix
        if os.path.isdir(os.path.join(GEN, _arm)) and _arm not in CHAINS[_slug]:
            CHAINS[_slug].append(_arm)


def has_verdict(r):
    """The one predicate. Same rule the report drops on and the backfill repairs on."""
    return not (str(r.get("finish_reason")) == "length"
                and "</think>" not in str(r.get("response") or ""))


def arm_file(arm):
    d = os.path.join(GEN, arm)
    if not os.path.isdir(d):
        return None
    js = [f for f in sorted(os.listdir(d)) if f.endswith(".jsonl")]
    return os.path.join(d, js[0]) if js else None


def main():
    ap = argparse.ArgumentParser()
    # Written back into xmodal_gen under a -final suffix so the existing upload and
    # report tooling, which globs that one directory, needs no special case.
    ap.add_argument("--out", default="/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen")
    ap.add_argument("--suffix", default="-final")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print(f"{'model':22s} {'rows':>5s} {'items':>5s} {'no-verd':>7s}   provenance")
    print("-" * 92)
    for slug, chain in CHAINS.items():
        best, prov = {}, {}
        for arm in chain:
            p = arm_file(arm)
            if not p:
                continue
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue           # torn tail from a live writer
                key = (r.get("item_id"), r.get("sample_idx"))
                cur = best.get(key)
                # A later pass wins only if it is at least as good. Without the
                # second clause a killed repair job that rewrote a draw badly would
                # overwrite a verdict the earlier pass had already earned.
                if cur is None or (has_verdict(r) >= has_verdict(cur)):
                    best[key] = r
                    prov[key] = arm

        if not best:
            print(f"{slug:22s} (no arms found)")
            continue
        rows = []
        for key in sorted(best, key=lambda k: (str(k[0]), k[1])):
            r = dict(best[key])
            r["source_arm"] = prov[key]
            rows.append(r)

        items = len({r["item_id"] for r in rows})
        nv = sum(1 for r in rows if not has_verdict(r))
        counts = collections.Counter(prov.values())
        pretty = ", ".join(f"{a.replace(slug, '') or 'source'}={n}"
                           for a, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        flag = "" if (len(rows) == N_ITEMS * K and items == N_ITEMS) else "  << SHORT"
        print(f"{slug:22s} {len(rows):5d} {items:5d} {nv:7d}   {pretty}{flag}")

        if not args.dry_run:
            od = os.path.join(args.out, slug + args.suffix)
            os.makedirs(od, exist_ok=True)
            src = os.path.basename(arm_file(chain[0]))
            with open(os.path.join(od, src), "w") as w:
                for r in rows:
                    w.write(json.dumps(r) + "\n")
    if args.dry_run:
        print("\nDRY RUN — nothing written")


if __name__ == "__main__":
    main()
