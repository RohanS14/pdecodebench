import json, glob, os, hashlib
from collections import Counter, defaultdict

root = "/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen"
rows_by_dir = {}
for p in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
    d = os.path.basename(os.path.dirname(p))
    n = 0; keys = set(); trunc = 0; nv = 0; sa = Counter()
    conds = set(); items = set(); sidx = Counter(); cols = None
    h = hashlib.md5()
    for line in open(p):
        r = json.loads(line); n += 1
        if cols is None: cols = len(r)
        k = (r.get("item_id"), r.get("condition"), r.get("sample_idx"), r.get("order_seed"))
        keys.add(k); items.add(r.get("item_id")); conds.add(r.get("condition"))
        sidx[r.get("sample_idx")] += 1
        if r.get("finish_reason") == "length": trunc += 1
        if r.get("no_verdict"): nv += 1
        sa[r.get("source_arm")] += 1
        h.update((r.get("response") or "").encode())
    rows_by_dir[d] = dict(path=p, n=n, uniq=len(keys), keys=keys, trunc=trunc, nv=nv,
                          sa=dict(sa), items=len(items), conds=len(conds),
                          cols=cols, rhash=h.hexdigest(), k=dict(sidx),
                          model=os.path.basename(p).split("__")[0])

# group by model
by_model = defaultdict(list)
for d, v in rows_by_dir.items():
    by_model[v["model"]].append((d, v))

print(f"{'pass dir':42s} {'rows':>5} {'uniq':>5} {'items':>5} {'cond':>4} {'trunc':>5} {'cols':>4}  resp-hash")
for m in sorted(by_model):
    print(f"\n### {m}")
    for d, v in sorted(by_model[m]):
        print(f"  {d:40s} {v['n']:5d} {v['uniq']:5d} {v['items']:5d} {v['conds']:4d} "
              f"{v['trunc']:5d} {v['cols']:4d}  {v['rhash'][:10]}")
    # duplicate detection within model
    seen = defaultdict(list)
    for d, v in by_model[m]:
        seen[v["rhash"]].append(d)
    dups = {h: ds for h, ds in seen.items() if len(ds) > 1}
    for h, ds in dups.items():
        print(f"    IDENTICAL responses: {ds}")
    # coverage: union of keys vs best single pass
    union = set()
    for d, v in by_model[m]: union |= v["keys"]
    best = max(by_model[m], key=lambda t: t[1]["uniq"])
    print(f"    union of all passes = {len(union)} keys; largest single pass "
          f"'{best[0]}' = {best[1]['uniq']}; union adds {len(union)-best[1]['uniq']}")
