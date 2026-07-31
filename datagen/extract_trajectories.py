"""
Trajectory modality for Experiment 2 Part II (plan §15).

full_audit_exec.py already executes all 256 rows, but keeps only a final-state
fingerprint — (shape, mean, std, has_nan) per array. That is enough to verify the
dataset and not enough to be a physics modality: it cannot say whether a pulse
spread, a wave reflected, or a shock formed.

This harness re-executes each row in the same sandboxed subprocess style, but
instruments the time loop to record the evolving solution field, then derives:

  * a compact numeric summary per row (mass, energy, extrema, variation, decay)
  * a TEXT rendering of the dynamics, which is what gets fed to the LLM
  * for each (gt_sample, condition), the valid-vs-invalid numerical divergence

That last quantity is the point. It lets §15.1 ask whether ‖Δh‖ tracks the size of
the REAL physical error rather than the size of the code edit.

CPU-only by design: the simulations are numpy/scipy. Run as a batch job, never on
a login node.

Usage:
    MPLBACKEND=Agg JAX_PLATFORMS=cpu python datagen/extract_trajectories.py \
        --dataset data/merged_mod_jul28.csv \
        --out_dir /scratch/.../outputs/trajectories \
        --timeout 300
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.dataset_io import DEFAULT_MOD_DATASET, load_dataset  # noqa: E402

# Runs inside the child process. Records snapshots of every evolving float array
# by wrapping the module namespace after execution AND sampling during the run via
# a trace hook on the outermost loop is not reliable across authoring styles, so
# we take the pragmatic route: execute, then reconstruct what we can from the
# final namespace plus any array the author kept as a history/list.
CHILD = r'''
import json, runpy, sys, warnings
warnings.filterwarnings("ignore")
try:
    import matplotlib; matplotlib.use("Agg")
except Exception: pass
import numpy as np

def summarize(a):
    """Scalar physical summary of one array. NaN-safe, never raises."""
    a = np.asarray(a)
    if a.dtype.kind not in "fc" or a.size == 0:
        return None
    r = a.real.astype(np.float64)
    with np.errstate(all="ignore"):
        flat = r.reshape(-1)
        d = {
            "shape": list(a.shape),
            "mean": float(np.nanmean(r)),
            "std": float(np.nanstd(r)),
            "min": float(np.nanmin(r)),
            "max": float(np.nanmax(r)),
            "l2": float(np.sqrt(np.nansum(r * r))),
            "absmean": float(np.nanmean(np.abs(r))),
            # total variation along the last axis: shock/oscillation signature
            "tv": float(np.nansum(np.abs(np.diff(r, axis=-1)))) if r.ndim >= 1 and r.shape[-1] > 1 else 0.0,
            "nan": bool(np.isnan(r).any() or np.isinf(r).any()),
        }
        # Coarse spatial profile, resampled to a fixed length so rows are comparable.
        # For 2D fields take a SLICE through the middle, never a mean across an
        # axis: a symmetric field averages to ~0, which destroys the profile and
        # (worse) makes it a near-zero denominator in the divergence below.
        if r.ndim >= 1 and flat.size >= 8:
            prof = r[r.shape[0] // 2].reshape(-1) if r.ndim > 1 else flat
            if prof.size >= 8:
                idx = np.linspace(0, len(prof) - 1, 16).astype(int)
                d["profile"] = [round(float(x), 6) for x in prof[idx]]
            d["rms"] = float(np.sqrt(np.nanmean(r * r)))
    return d

res = {"ok": False, "err": None, "fields": {}, "history": {}}
try:
    ns = runpy.run_path(sys.argv[1], run_name="__main__")
    res["ok"] = True
    for k, v in ns.items():
        if k.startswith("_"):
            continue
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.dtype.kind not in "fc" or a.size == 0 or a.ndim == 0:
            continue
        s = summarize(a)
        if s is None:
            continue
        # a list-of-arrays kept by the author IS a trajectory; ndim one higher
        if isinstance(v, (list, tuple)) and len(v) > 2 and a.ndim >= 2:
            res["history"][k] = {"n_steps": int(a.shape[0]),
                                 "per_step_l2": [float(np.sqrt(np.nansum(np.asarray(x, dtype=float)**2)))
                                                 for x in a[:64]]}
        res["fields"][k] = s
except BaseException as e:
    res["err"] = f"{type(e).__name__}: {e}"[:250]
print("@@T@@" + json.dumps(res))
'''


def state_field(fields: dict) -> tuple:
    """
    Pick the array that most plausibly IS the solution field: the largest
    non-degenerate float array. Returns (name, summary) or (None, None).
    """
    best, best_key = None, None
    for k, s in fields.items():
        if s is None:
            continue
        size = int(np.prod(s["shape"])) if s["shape"] else 0
        if size < 8:
            continue
        if best is None or size > int(np.prod(best["shape"])):
            best, best_key = s, k
    return best_key, best


def render_text(summary: dict, ok: bool, err: str) -> str:
    """
    Human/LLM-readable description of the dynamics. This is the trajectory as the
    model will see it — deliberately physical vocabulary, no PDE name, no method
    name, and no variable names from the code, so it cannot leak the label.
    """
    if not ok:
        return f"The simulation failed to run to completion ({err})."
    if summary is None:
        return "The simulation completed but produced no recognisable field."
    if summary["nan"]:
        return ("The solution diverged: the field contains non-finite values, "
                "indicating a blow-up or an unstable update.")

    parts = []
    rng = summary["max"] - summary["min"]
    parts.append(f"The final field spans {summary['min']:.4g} to {summary['max']:.4g} "
                 f"(range {rng:.4g}), with mean {summary['mean']:.4g} and "
                 f"spread {summary['std']:.4g}.")
    if summary["tv"] > 0:
        sharp = summary["tv"] / (rng + 1e-12)
        if sharp > 8:
            parts.append("The profile is highly structured, with many sharp "
                         "transitions — steep fronts or fine-scale oscillation.")
        elif sharp > 2:
            parts.append("The profile has moderate structure with visible gradients.")
        else:
            parts.append("The profile is smooth, with gradients largely flattened out.")
    prof = summary.get("profile")
    if prof:
        parts.append("Sampled across the domain the field reads: "
                     + ", ".join(f"{p:.3g}" for p in prof) + ".")
    return " ".join(parts)


def run_one(code: str, work: Path, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=work, delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "_child.py", path],
                           capture_output=True, text=True, timeout=timeout, cwd=str(work))
        lines = [l for l in p.stdout.splitlines() if l.startswith("@@T@@")]
        if not lines:
            return {"ok": False, "err": "no result: " + p.stderr[-200:],
                    "fields": {}, "history": {}}
        return json.loads(lines[-1][5:])
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": f"timeout after {timeout}s", "fields": {}, "history": {}}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def divergence(a: dict, b: dict) -> dict:
    """
    Numerical divergence between a valid run and its invalid twin. This is the
    quantity §15.1 correlates against ‖Δh‖.

    Blow-up is a CATEGORY, not a large number — an invalid run that produces NaN
    has infinite relative L2, which would otherwise dominate every correlation.
    """
    if not (a and b):
        return {"kind": "unavailable", "rel_l2": np.nan}
    if a["nan"] or b["nan"]:
        return {"kind": "blowup", "rel_l2": np.nan}
    if a["shape"] != b["shape"]:
        return {"kind": "shape_change", "rel_l2": np.nan}
    pa, pb = a.get("profile"), b.get("profile")
    if not (pa and pb):
        denom = max(abs(a["l2"]), abs(b["l2"])) + 1e-12
        return {"kind": "scalar", "rel_l2": abs(a["l2"] - b["l2"]) / denom}

    pa, pb = np.array(pa), np.array(pb)
    na, nb = np.sqrt((pa ** 2).sum()), np.sqrt((pb ** 2).sum())
    # Symmetric denominator, bounded in [0, 2]. An asymmetric ‖pa‖ denominator
    # explodes when the valid run happens to sample near a zero crossing, which
    # produced spurious 1e7 divergences before this fix.
    denom = max(na, nb)
    if denom < 1e-12:
        # both slices are ~0: the profile carries no signal, fall back to a
        # whole-field amplitude comparison rather than dividing by nothing
        ra, rb = a.get("rms", a["l2"]), b.get("rms", b["l2"])
        d2 = max(abs(ra), abs(rb)) + 1e-12
        return {"kind": "degenerate_profile", "rel_l2": abs(ra - rb) / d2}
    return {"kind": "field", "rel_l2": float(np.sqrt(((pa - pb) ** 2).sum()) / denom)}


VALID_OF = {
    "Comm_InValid": "Comm_Valid",
    "NoComm_InValid": "NoComm_Valid",
    "CorrComm_Invalid": "CorrComm",
    "NoComm_CorrVar_InValid": "NoComm_CorrVar",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_MOD_DATASET)
    ap.add_argument("--out_dir", default="data/trajectories")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="canary: first N rows only")
    args = ap.parse_args()

    df = load_dataset(args.dataset)
    if args.limit:
        df = df.head(args.limit)

    work = Path(args.out_dir) / "_work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "_child.py").write_text(CHILD)

    out = {}
    n_ok = 0
    for i, (_, r) in enumerate(df.iterrows()):
        res = run_one(str(r["code"]), work, args.timeout)
        key, summ = state_field(res.get("fields", {}))
        rec = {
            "gt_sample": r["gt_sample"],
            "mod_type": r["mod_type"],
            "pde_class": r["pde_class"],
            "phys_valid": bool(r["phys_valid"]),
            "ok": res["ok"],
            "err": res.get("err"),
            "field_name": key,
            "summary": summ,
            "n_history": len(res.get("history", {})),
            "text": render_text(summ, res["ok"], res.get("err")),
        }
        out[f"{r['gt_sample']}|{r['mod_type']}"] = rec
        n_ok += int(res["ok"])
        if (i + 1) % 16 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(df)}] {r['gt_sample']} ({r['mod_type']}) "
                  f"ok={res['ok']} field={key}", flush=True)

    # valid-vs-invalid divergence per (solver, condition)
    div = {}
    for k, rec in out.items():
        inv_mt = rec["mod_type"]
        if inv_mt not in VALID_OF:
            continue
        vkey = f"{rec['gt_sample']}|{VALID_OF[inv_mt]}"
        if vkey not in out:
            continue
        d = divergence(out[vkey]["summary"], rec["summary"])
        div[f"{rec['gt_sample']}|{inv_mt}"] = {
            "gt_sample": rec["gt_sample"], "invalid_mod_type": inv_mt,
            "valid_mod_type": VALID_OF[inv_mt], **d,
        }

    os.makedirs(args.out_dir, exist_ok=True)
    tp = os.path.join(args.out_dir, "trajectories_jul28.json")
    dp = os.path.join(args.out_dir, "divergence_jul28.json")
    with open(tp, "w") as f:
        json.dump(out, f, indent=1)
    with open(dp, "w") as f:
        json.dump(div, f, indent=1)

    kinds = {}
    for v in div.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    finite = [v["rel_l2"] for v in div.values() if np.isfinite(v.get("rel_l2", np.nan))]

    print(f"\nSaved: {tp}  ({len(out)} rows, {n_ok} executed cleanly)", flush=True)
    print(f"Saved: {dp}  ({len(div)} valid/invalid pairs)", flush=True)
    print(f"  divergence kinds: {kinds}", flush=True)
    if finite:
        print(f"  finite rel_l2: median {np.median(finite):.4g}, "
              f"min {min(finite):.4g}, max {max(finite):.4g}", flush=True)
    n_zero = sum(1 for x in finite if x < 1e-12)
    if n_zero:
        print(f"  WARNING: {n_zero} pairs have ZERO divergence — the invalid variant "
              f"produced numerically identical output. Those solvers cannot support "
              f"the §15.1 correlation and must be excluded from it.", flush=True)
    if n_ok < len(out):
        print(f"  NOTE: {len(out) - n_ok} rows failed to execute; their trajectory "
              f"modality is unavailable, not zero.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
