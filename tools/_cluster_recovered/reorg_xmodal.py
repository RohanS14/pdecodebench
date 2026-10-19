"""
reorg_xmodal.py — one directory per model for cross-modal consistency.

The runner never repairs an arm in place: a pass that left truncated rows was
followed by a NEW directory holding the whole arm with those rows re-run. Eight
results therefore accumulated 45 directories. Measured across all of them
(tools/xm_compare.py): every non-final pass has keys_not_in_final == 0 and
could_fix == 0, so the -final arms are the endpoint of each chain and the rest hold
no unique row.

This promotes the eight -final arms to cross_modal_consistency/<model-slug>/ and
moves everything else to _trash/<date>/, which is reversible. It verifies the row
count and key count of every arm before and after, and refuses to touch a model
whose -final arm does not match what the inventory measured.
"""
import json, glob, os, shutil, sys, argparse

SRC = "/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen"
DST = "/scratch/ehb7466/projects/pde-llm-eval/outputs/cross_modal_consistency"

# -final dir -> canonical model slug, matching the free-generation naming so the two
# trees are navigable the same way.
FINAL = {
    "qwen3-32b-final":            "qwen__qwen3-32b",
    "qwen3-5-27b-final":          "qwen__qwen3-5-27b",
    "qwen3-6-27b-final":          "qwen__qwen3-6-27b",
    "qwen3-8-27b-final":          "qwen__qwen3-8-27b",
    "qwq-32b-final":              "qwen__qwq-32b",
    "r1-distill-qwen-32b-final":  "deepseek-ai__deepseek-r1-distill-qwen-32b",
    "nemotron-3-nano-30b-final":  "nvidia__nvidia-nemotron-3-nano-30b-a3b-bf16",
    "glm-4-7-flash-final":        "zai-org__glm-4-7-flash",
}
EXPECT_ROWS = 3072


def audit(path):
    n = 0
    keys = set()
    trunc = 0
    for line in open(path):
        r = json.loads(line)
        n += 1
        keys.add((r["item_id"], r["condition"], r["sample_idx"], r.get("order_seed")))
        if r.get("finish_reason") == "length":
            trunc += 1
    return n, len(keys), trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trash_dir", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    problems = []
    plan = []
    for d, slug in FINAL.items():
        src = os.path.join(SRC, d)
        files = glob.glob(os.path.join(src, "*.jsonl"))
        if len(files) != 1:
            problems.append(f"{d}: expected 1 jsonl, found {len(files)}")
            continue
        n, uniq, trunc = audit(files[0])
        if n != EXPECT_ROWS or uniq != EXPECT_ROWS:
            problems.append(f"{d}: {n} rows / {uniq} keys, expected {EXPECT_ROWS}")
        plan.append((d, slug, files[0], n, uniq, trunc))

    print(f"{'source -final dir':30s} -> {'model slug':44s} {'rows':>5} {'keys':>5} {'trunc':>5}")
    for d, slug, f, n, uniq, trunc in plan:
        print(f"{d:30s} -> {slug:44s} {n:5d} {uniq:5d} {trunc:5d}")

    retire = sorted(x for x in os.listdir(SRC)
                    if os.path.isdir(os.path.join(SRC, x)) and x not in FINAL)
    print(f"\nretiring {len(retire)} superseded pass dir(s) to {args.trash_dir}")
    for x in retire:
        print(f"  {x}")

    if problems:
        print("\nPROBLEMS — nothing moved:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    if not args.apply:
        print("\nDRY RUN — pass --apply to execute")
        return

    os.makedirs(DST, exist_ok=True)
    os.makedirs(args.trash_dir, exist_ok=True)
    for d, slug, f, n, uniq, trunc in plan:
        out_dir = os.path.join(DST, slug)
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, os.path.basename(f))
        shutil.copy2(f, dest)
        n2, uniq2, trunc2 = audit(dest)
        if (n2, uniq2, trunc2) != (n, uniq, trunc):
            sys.exit(f"FATAL: {slug} copy mismatch {(n2,uniq2,trunc2)} != {(n,uniq,trunc)}")
        print(f"[ok] {slug}: {n2} rows verified at {dest}")

    for x in retire:
        shutil.move(os.path.join(SRC, x), os.path.join(args.trash_dir, x))
    print(f"\n[done] {len(plan)} canonical arms, {len(retire)} dirs retired")


if __name__ == "__main__":
    main()
