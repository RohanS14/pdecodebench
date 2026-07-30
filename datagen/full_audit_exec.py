"""
full_audit_exec.py — execution half of the dataset audit.

Three questions the static audit cannot answer:
  1. does every row run?
  2. do the valid rows behave (no NaN/Inf/blow-up), and do the invalid ones misbehave?
  3. do a gt_sample's four surface conditions produce IDENTICAL numbers?

(3) is the strongest integrity check in the suite. Comm_Valid, NoComm_Valid, CorrComm and
NoComm_CorrVar are the same program with different comments and identifiers, so any
numerical difference between them means a surface transform corrupted the physics. Same
for the four invalid conditions.

Each row runs in its own subprocess with a hard timeout, so one hanging simulation cannot
stall the sweep. Comparison uses a rename-invariant fingerprint: the multiset of
(shape, mean, std, has_nan) over every float array left in the namespace.

Usage:
    MPLBACKEND=Agg JAX_PLATFORMS=cpu python datagen/full_audit_exec.py [csv] [timeout_s]
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

HARNESS = textwrap.dedent('''
    import json, runpy, sys, warnings
    warnings.filterwarnings("ignore")
    try:
        import matplotlib; matplotlib.use("Agg")
    except Exception: pass
    import numpy as np

    def fingerprint(ns):
        fps = []
        for v in ns.values():
            try:
                a = np.asarray(v)
            except Exception:
                continue
            if a.dtype.kind not in "fc" or a.size == 0 or a.ndim == 0:
                continue
            with np.errstate(all="ignore"):
                fps.append([list(a.shape),
                            round(float(np.nanmean(a.real)), 8),
                            round(float(np.nanstd(a.real)), 8),
                            bool(np.isnan(a).any() or np.isinf(a).any())])
        return sorted(fps, key=lambda t: (str(t[0]), t[1], t[2]))

    res = {"ok": False, "err": None, "fp": [], "nan": False, "spike": False}
    try:
        ns = runpy.run_path(sys.argv[1], run_name="__main__")
        res["ok"] = True
        res["fp"] = fingerprint(ns)
        for shape, mean, std, hasnan in res["fp"]:
            if hasnan: res["nan"] = True
            if abs(mean) > 1e6 or std > 1e6: res["spike"] = True
    except BaseException as e:
        res["err"] = f"{type(e).__name__}: {e}"[:250]
    print("@@R@@" + json.dumps(res))
''')

VALID_MODS = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar"]
INVALID_MODS = ["Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid"]


def main(csv="data/merged_mod_jul28.csv", timeout=300):
    df = pd.read_csv(csv)
    work = Path("_audit_exec"); work.mkdir(exist_ok=True)
    (work / "_h.py").write_text(HARNESS)

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        f = work / f"{r.title}.py"
        f.write_text(str(r.code).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t"))
        try:
            # cwd is `work`, so the harness must be named relative to it -- passing
            # str(work / "_h.py") resolves to work/work/_h.py and every row fails to start.
            p = subprocess.run([sys.executable, "_h.py", f.name],
                               capture_output=True, text=True, timeout=timeout, cwd=str(work))
            line = [l for l in p.stdout.splitlines() if l.startswith("@@R@@")]
            res = json.loads(line[-1][5:]) if line else {
                "ok": False, "err": "no result: " + p.stderr[-160:], "fp": [], "nan": False, "spike": False}
        except subprocess.TimeoutExpired:
            res = {"ok": False, "err": f"TIMEOUT>{timeout}s", "fp": [], "nan": False, "spike": False}
        res.update(title=r.title, gt_sample=r.gt_sample, mod=r.mod_type,
                   src=r.source, valid=bool(r.phys_valid), note=r.invalidity_note)
        rows.append(res)
        if i % 32 == 0:
            print(f"  ...{i}/{len(df)}", flush=True)

    R = pd.DataFrame(rows)
    R.drop(columns=["fp"]).to_csv("_audit_exec_results.csv", index=False)
    fp = {(r["gt_sample"], r["mod"]): r["fp"] for r in rows}
    ok = {(r["gt_sample"], r["mod"]): r["ok"] for r in rows}

    print("\n" + "=" * 78)
    print("EXECUTION AUDIT")
    print("=" * 78)
    print(f"\n1. EXECUTES: {int(R['ok'].sum())}/{len(R)}   errors: {int((~R['ok']).sum())}")
    if (~R['ok']).any():
        for _, r in R[~R['ok']].iterrows():
            print(f"     FAIL {r['gt_sample']:16s} {r['mod']:24s} {str(r['err'])[:80]}")

    v, iv = R[R['valid'] & R['ok']], R[(~R['valid']) & R['ok']]
    fpos = v[v['nan'] | v['spike']]
    print(f"\n2. VALID rows flagged anomalous (should be 0): {len(fpos)}/{len(v)}")
    for _, r in fpos.iterrows():
        print(f"     {r['gt_sample']:16s} {r['mod']:24s} nan={r['nan']} spike={r['spike']}")
    quiet = iv[~(iv['nan'] | iv['spike'])]
    print(f"   INVALID rows not caught by the NaN/spike heuristic: {len(quiet)}/{len(iv)}")
    per_gt = sorted(quiet['gt_sample'].unique())
    print(f"     affected base problems ({len(per_gt)}): {per_gt}")
    print("     (subtle failure modes are expected here — cross-check invalidity_note)")

    print("\n3. CROSS-CONDITION NUMERICAL IDENTITY")
    for label, mods in [("valid", VALID_MODS), ("invalid", INVALID_MODS)]:
        mismatch, skipped = [], []
        for gt in sorted(df.gt_sample.unique()):
            present = [m for m in mods if ok.get((gt, m))]
            if len(present) < 2:
                skipped.append(gt); continue
            ref = fp[(gt, present[0])]
            for m in present[1:]:
                if fp[(gt, m)] != ref:
                    mismatch.append((gt, present[0], m))
        status = "PASS" if not mismatch else "FAIL"
        print(f"   [{status}] all {label} conditions produce identical numbers"
              f"   (compared {len(df.gt_sample.unique()) - len(skipped)} base problems)")
        for gt, a, b in mismatch[:12]:
            print(f"          {gt:16s} {a} != {b}")
        if skipped:
            print(f"          not comparable (too few executed): {skipped}")

    n_bad = int((~R['ok']).sum()) + len(fpos)
    print("\n" + "=" * 78)
    print("execution audit:", "CLEAN" if n_bad == 0 else f"{n_bad} problem(s)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/merged_mod_jul28.csv"
    to = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    sys.exit(main(csv, to))
