import json, glob, os
from collections import Counter

root = "/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen"
def load(d):
    p = glob.glob(os.path.join(root, d, "*.jsonl"))[0]
    out = {}
    dup = 0
    for line in open(p):
        r = json.loads(line)
        k = (r["item_id"], r["condition"], r["sample_idx"], r.get("order_seed"))
        if k in out: dup += 1
        out[k] = r
    return out, dup

FAM = {
 "qwen3-32b":     ["qwen3-32b", "qwen3-32b-backfill", "qwen3-32b-backfill2", "qwen3-32b-final"],
 "qwen3-5-27b":   ["qwen3-5-27b", "qwen3-5-27b-backfill", "qwen3-5-27b-backfill2", "qwen3-5-27b-final"],
 "qwen3-6-27b":   ["qwen3-6-27b", "qwen3-6-27b-backfill", "qwen3-6-27b-backfill2", "qwen3-6-27b-final", "qwen3-6-27b-rep"],
 "qwen3-8-27b":   ["qwen3-8-27b", "qwen3-8-27b-final", "qwen3-8-27b-rep", "qwen3-8-27b-s1", "qwen3-8-27b-s2", "qwen3-8-27b-s3", "qwen3-8-27b-s4", "qwen3-8-27b__32k_budget_abandoned"],
 "qwq-32b":       ["qwq-32b", "qwq-32b-backfill", "qwq-32b-backfill2", "qwq-32b-final", "qwq-32b-rep"],
 "r1-distill":    ["r1-distill-qwen-32b", "r1-distill-qwen-32b-backfill", "r1-distill-qwen-32b-backfill2", "r1-distill-qwen-32b-final"],
 "nemotron":      ["nemotron-3-nano-30b", "nemotron-3-nano-30b-64k", "nemotron-3-nano-30b-128k", "nemotron-3-nano-30b-backfill", "nemotron-3-nano-30b-backfill2", "nemotron-3-nano-30b-final", "nemotron-3-nano-30b-rep0", "nemotron-3-nano-30b-rep1", "nemotron-3-nano-30b-rep2"],
 "glm":           ["glm-4-7-flash", "glm-4-7-flash-backfill", "glm-4-7-flash-backfill3", "glm-4-7-flash-final", "glm-4-7-flash-rep"],
}

for fam, dirs in FAM.items():
    print(f"\n### {fam}")
    final_dir = [d for d in dirs if d.endswith("-final")]
    if not final_dir:
        print("  NO -final"); continue
    F, fdup = load(final_dir[0])
    ftr = sum(1 for r in F.values() if r.get("finish_reason") == "length")
    print(f"  final: {len(F)} keys, {fdup} dup lines, {ftr} truncated")
    for d in dirs:
        if d == final_dir[0]: continue
        O, odup = load(d)
        extra = set(O) - set(F)
        sameresp = sum(1 for k in (set(O) & set(F)) if O[k].get("response") == F[k].get("response"))
        otr = sum(1 for r in O.values() if r.get("finish_reason") == "length")
        # would this pass FIX any row final still has truncated?
        fixes = sum(1 for k in (set(O) & set(F))
                    if F[k].get("finish_reason") == "length"
                    and O[k].get("finish_reason") != "length")
        print(f"    {d:38s} keys={len(O):5d} dupline={odup:3d} trunc={otr:4d} "
              f"keys_not_in_final={len(extra):4d} identical_resp={sameresp:5d} could_fix={fixes}")
