"""
build_exec_trajectories.py — produces the T_exec rung of the corruption ladder.

T_exec is the hardest rung: the actual output of the invalid solver, so the
trajectory is coherent physics of the RIGHT system with the WRONG dynamics. Every
other rung is either structurally empty (T_rand), a rearrangement (T_shuf), or
another system entirely (T_swap), and all three can be built from the CSV. This one
has to be executed.

Only the 32 NoComm_InValid variants are run. The obfuscated invalid variant is
AST-identical and produces the same numbers by construction -- full_audit_exec.py
already asserts that a problem's surface conditions agree -- so its trajectory is
the same array and re-running it would spend CPU reproducing a known result.

Two ways a trajectory is obtained, in order of preference:

  harvest        The solver already keeps a history -- a time-indexed array, a list
                 of frames, or a snapshot dict. 27 of 32 do. Nothing is modified.
  instrument     The solver overwrites in place and keeps nothing useful. Four do.
                 cross_modal_consistency/datagen/instrument_history.py adds a provably write-only recorder
                 to a COPY, verified to leave every computed array bit-identical.

Heat_8 supports neither: it is spectral with no time loop, so there is no history to
harvest and nothing to instrument. It is skipped and recorded, which means the
T_exec condition covers 31 of 32 systems rather than 32.

CPU-only by design -- these are numpy/scipy solvers, so a GPU node would be wasted.
Each row runs in its own subprocess under a timeout, following the sandboxing
pattern extract_trajectories.py and full_audit_exec.py already use: soft isolation,
not a security boundary.

Usage:
    python cross_modal_consistency/datagen/build_exec_trajectories.py --out data/exec_trajectories.npz
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# repo root: this file sits at cross_modal_consistency/<area>/, so three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from cross_modal_consistency.datagen.corrupt_trajectory import DATASET_FRAMES, decimate_frames   # noqa: E402
from cross_modal_consistency.datagen.build_multimodal_items import MULTIMODAL_CSV, canonical     # noqa: E402
from cross_modal_consistency.datagen.instrument_history import (                                 # noqa: E402
    INSTRUMENT_SPEC, NOT_INSTRUMENTABLE, instrument,
)
# Heat_8 has no time loop: it is spectral and evaluates a closed-form solution once
# at t_final. Its solution operator is exact at any t, so sampling it at the same ten
# times the dataset uses IS that system's true trajectory -- not an approximation and
# not a different system. Appended to a COPY, like every other instrumentation here,
# and marked route="analytic" so it is visible in the manifest rather than passing as
# an ordinary harvest.
ANALYTIC_TAIL = """
_HISTORY = []
for _k in range(10):
    _t = t_final * _k / 9.0
    _decay = np.exp(-alpha * K2 * _t)
    _HISTORY.append([np.real(np.fft.ifft2(u0_hat * _decay))])
"""

ANALYTIC = {"Heat_8": ANALYTIC_TAIL}
from cross_modal_consistency.datagen.render_trajectory_table import parse_trajectory             # noqa: E402

csv.field_size_limit(10 ** 9)

DEFAULT_DATASET = "data/merged_mod_jul28.csv"
TARGET_MOD_TYPE = "NoComm_InValid"

# Runs in the child. Harvests a history from the executed namespace, preferring the
# largest candidate that plausibly indexes time on its first axis.
CHILD = r'''
import json, pickle, sys, warnings
warnings.filterwarnings("ignore")
try:
    import matplotlib; matplotlib.use("Agg")
except Exception: pass
import numpy as np

SRC, OUT, MIN_FRAMES, INSTRUMENTED = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
# The (X, Y, C) the dataset's own trajectory for this system has. T_exec must be
# the same physical object as the valid view -- same grid, same components -- or the
# rung compares two different quantities rather than two dynamics.
WANT = tuple(json.loads(sys.argv[5])) if len(sys.argv) > 5 and sys.argv[5] else None

ns = {"__name__": "__main__"}
exec(compile(open(SRC).read(), "<solver>", "exec"), ns)


def orientations(a):
    """Both readings of an array's time axis: first, and last.

    Authors are split on this. Wave_1 stores its history as (101, 101, 800) -- a
    2-D field with time LAST -- and Wave_2 as (151, 40001). Assuming time is axis 0
    silently reads 800 timesteps as an 800-cell spatial axis and harvests the wrong
    object, which is how a 101x101 wave came out looking like a 101-point line.
    Both readings are offered and the one matching the dataset's own shape wins.
    """
    out = []
    if a.shape[0] >= MIN_FRAMES:
        out.append(a)
    if a.ndim >= 2 and a.shape[-1] >= MIN_FRAMES:
        out.append(np.moveaxis(a, -1, 0))
    # A batched generator: (n_samples, T, X, Y, C). NavierStokes_4 is written as a
    # dataset builder rather than a single run -- get_ns2d(n_samples=2, ...) returns
    # two independent trajectories from split PRNG keys, and its only module-level
    # array is that 5-D batch. Reading it as (T=2, ...) fails MIN_FRAMES and the
    # solver was reported "no history found" when in fact it had produced two.
    # Offer the FIRST sample: the keys come from a fixed PRNGKey(0) split, so sample
    # 0 is deterministic and reproducible. Which sample is arbitrary; that it is
    # sample 0 is recorded in the manifest.
    if a.ndim == 5 and a.shape[0] < MIN_FRAMES and a.shape[1] >= MIN_FRAMES:
        out.append(a[0])
    return out


def as_frames(obj):
    """Normalize a history candidate into a list of (T, ...) readings."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fc" and obj.ndim >= 2:
            return orientations(np.asarray(obj.real, dtype=float))
        return None
    if isinstance(obj, dict) and len(obj) >= MIN_FRAMES:
        try:
            keys = sorted(obj)
            arrs = [np.asarray(obj[k], dtype=float) for k in keys]
        except Exception:
            return None
        if arrs and all(a.shape == arrs[0].shape for a in arrs):
            return [np.stack(arrs, axis=0)]
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= MIN_FRAMES:
        try:
            arrs = [np.asarray(a, dtype=float) for a in obj]
        except Exception:
            return None
        if arrs and all(a.shape == arrs[0].shape and a.ndim >= 1 for a in arrs):
            return [np.stack(arrs, axis=0)]
        return None
    return None


if INSTRUMENTED:
    hist = ns.get("_HISTORY")
    frames = None
    if hist:
        arrs = []
        for step in hist:
            fields = [np.asarray(f, dtype=float) for f in step]
            fields = [f[:, None] if f.ndim == 1 else f for f in fields]
            arrs.append(np.stack(fields, axis=-1))
        frames = np.stack(arrs, axis=0)
    best, source = frames, "_HISTORY"
else:
    def spatial(shape):
        """(X, Y, C) after the time axis, in the dataset's 4-D convention."""
        rest = list(shape[1:])
        while len(rest) < 3:
            rest.append(1)
        return tuple(rest[:3])

    def rank(cand):
        """Lower is better. Shape agreement dominates length.

        Picking the longest time axis alone is not enough: several solvers keep a
        1-D diagnostic (a centreline slice, a norm history) that is longer than the
        field they actually evolve, and it wins on length while being the wrong
        physical object. Wave_1 harvested a 101-point line where the dataset has a
        101x101 field, and NavierStokes_8 a 256-point trace where it has 128x128.
        """
        sp = spatial(cand.shape)
        if WANT is None:
            return (0, -cand.shape[0])
        if sp == WANT:
            return (0, -cand.shape[0])
        if np.prod(sp) == np.prod(WANT):
            return (1, -cand.shape[0])          # same cell count, reshaped
        if sp[:2] == WANT[:2]:
            return (2, -cand.shape[0])          # right grid, channels differ
        return (3, abs(int(np.prod(sp)) - int(np.prod(WANT))), -cand.shape[0])

    best, source, best_rank = None, None, None
    for name, value in ns.items():
        if name.startswith("__"):
            continue
        cands = as_frames(value)
        if not cands:
            continue
        for cand in cands:
            if cand.shape[0] < MIN_FRAMES:
                continue
            r = rank(cand)
            if best_rank is None or r < best_rank:
                best, source, best_rank = cand, name, r

if best is None:
    json.dump({"ok": False, "reason": "no history found"}, open(OUT + ".json", "w"))
    sys.exit(0)

np.save(OUT, best)
json.dump({"ok": True, "source": source, "shape": list(best.shape),
           "batched_took_sample_0": bool(
               isinstance(ns.get(source), np.ndarray)
               and getattr(ns.get(source), "ndim", 0) == 5),
           "shape_match": bool(WANT is None or tuple(list(best.shape[1:]) + [1, 1])[:3] == WANT)},
          open(OUT + ".json", "w"))
'''


def coerce_to_want(a, want):
    """Try to reconcile a harvested history's layout with the dataset's.

    A solver may store the same field in a different axis order -- NavierStokes_7
    keeps (components, y, x) where the dataset uses (y, x, components) -- and that
    is a layout difference, not a different quantity. Permuting axes is therefore
    legitimate; reshaping across a different cell count is not, and is left to fail
    so it shows up as a mismatch rather than being silently massaged into shape.
    """
    import itertools

    if want is None:
        return a
    rest = tuple(a.shape[1:])
    if rest == tuple(want):
        return a
    if sorted(rest) != sorted(want):
        return a                    # genuinely a different object; let it be flagged
    for perm in itertools.permutations(range(1, a.ndim)):
        if tuple(a.shape[i] for i in perm) == tuple(want):
            return np.transpose(a, (0,) + perm)
    return a


def to_layout(a):
    """Coerce a harvested history into the dataset's (T, X, Y, C) convention."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:                     # (T, X)
        return a[:, :, None, None]
    if a.ndim == 3:                     # (T, X, Y) or (T, X, C)
        return a[:, :, :, None]
    if a.ndim == 4:
        return a
    if a.ndim > 4:                      # collapse trailing axes into channels
        return a.reshape(a.shape[0], a.shape[1], a.shape[2], -1)
    raise ValueError(f"cannot interpret history of shape {a.shape}")


def run_one(system, code, timeout, workdir, want=None):
    """Execute one solver and return (array | None, status dict)."""
    analytic = system in ANALYTIC
    instrumented = system in INSTRUMENT_SPEC or analytic
    if system in NOT_INSTRUMENTABLE and not analytic:
        return None, {"status": "skipped", "reason": NOT_INSTRUMENTABLE[system]}

    src = code
    if analytic:
        src = code + ANALYTIC[system]
    elif instrumented:
        try:
            src = instrument(code, system)
        except Exception as exc:
            return None, {"status": "instrument_failed", "reason": str(exc)}

    src_path = os.path.join(workdir, f"{system}.py")
    out_path = os.path.join(workdir, f"{system}.npy")
    open(src_path, "w").write(src)
    child_path = os.path.join(workdir, "_child.py")
    open(child_path, "w").write(CHILD)

    try:
        proc = subprocess.run(
            [sys.executable, child_path, src_path, out_path,
             str(DATASET_FRAMES), "1" if instrumented else "0",
             json.dumps(list(want)) if want else ""],
            capture_output=True, text=True, timeout=timeout, cwd=workdir)
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout", "reason": f"exceeded {timeout}s"}

    meta_path = out_path + ".json"
    if not os.path.exists(meta_path):
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
        return None, {"status": "exec_failed", "reason": tail[0][:200]}

    meta = json.load(open(meta_path))
    if not meta.get("ok"):
        return None, {"status": "no_history", "reason": meta.get("reason", "")}

    arr = coerce_to_want(to_layout(np.load(out_path)), want)
    dec = decimate_frames(arr, DATASET_FRAMES)
    got = tuple(dec.shape[1:])
    return dec, {
        "status": "ok",
        "route": "analytic" if analytic else
                 ("instrumented" if instrumented else "harvested"),
        "source": meta.get("source"),
        "raw_frames": int(arr.shape[0]),
        "shape": list(dec.shape),
        "expected_shape": list(want) if want else None,
        "shape_match": bool(want is None or got == tuple(want)),
    }


def main():
    p = argparse.ArgumentParser(description="Build T_exec trajectories")
    p.add_argument("--dataset", default=os.environ.get("DATASET", DEFAULT_DATASET))
    p.add_argument("--out", default=os.environ.get("EXEC_OUT", "data/exec_trajectories.npz"))
    p.add_argument("--timeout", type=int, default=int(os.environ.get("ROW_TIMEOUT", "600")))
    p.add_argument("--systems", default=os.environ.get("SYSTEMS", ""))
    p.add_argument("--allow_shrink", action="store_true",
                   help="Permit the output npz to lose systems that already exist "
                        "in it. Off by default: a partial re-run must not silently "
                        "discard trajectories it did not attempt.")
    p.add_argument("--multimodal", default=os.environ.get("MULTIMODAL", MULTIMODAL_CSV))
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.dataset, newline=""))
            if r["mod_type"] == TARGET_MOD_TYPE]
    if args.systems:
        keep = {s.strip() for s in args.systems.split(",") if s.strip()}
        rows = [r for r in rows if r["gt_sample"] in keep]
    rows.sort(key=lambda r: r["gt_sample"])

    # The dataset's own trajectory for each system fixes what shape T_exec must be.
    want = {}
    for r in csv.DictReader(open(args.multimodal, newline="")):
        name = canonical(r["Example Name"])
        if not name.endswith("_wrong"):
            want[name] = tuple(parse_trajectory(r["Trajectory"]).shape[1:])

    arrays, manifest = {}, {}
    with tempfile.TemporaryDirectory(prefix="exec_traj_") as workdir:
        for r in rows:
            system = r["gt_sample"]
            arr, status = run_one(system, r["code"], args.timeout, workdir,
                                  want.get(system))
            manifest[system] = status
            # Shipped even when the shape disagrees with the stored trajectory.
            # render_trajectory_table.resample puts ANY shape onto the item's own
            # grid -- that is exactly how T_swap works, where 30 of 32 donors differ
            # in shape and 14 flip 1-D <-> 2-D. Dropping T_exec for a shape
            # difference while shipping T_swap with one was an inconsistent rule.
            # The disagreement is real and worth knowing, so it stays as a recorded
            # flag rather than as a reason to lose the system.
            if arr is not None:
                arrays[system] = arr
            note = status.get("route") or status.get("reason", "")
            if status.get("status") == "ok" and not status.get("shape_match", True):
                note += f"  MISMATCH want={status.get('expected_shape')}"
            print(f"[exec] {system:<18}{status['status']:<18}{str(status.get('shape','')):<22}{note[:60]}",
                  flush=True)

    # GUARD. This writes the whole file, so a partial run -- SYSTEMS=NavierStokes_4
    # to fill one gap, say -- would replace a 31-system npz with a 1-system npz and
    # destroy five jobs' worth of executed trajectories. Nearly done exactly that on
    # 2026-08-20. Merge into what is already there instead, and never shrink the file
    # unless asked in as many words.
    if os.path.exists(args.out):
        with np.load(args.out) as existing:
            have = {k: existing[k] for k in existing.files}
        dropped = sorted(set(have) - set(arrays))
        if dropped and not args.allow_shrink:
            print(f"[exec] {len(have)} system(s) already in {args.out}; this run "
                  f"produced {len(arrays)}. MERGING rather than replacing -- keeping "
                  f"{len(dropped)} not re-run this time: {', '.join(dropped[:6])}"
                  f"{' ...' if len(dropped) > 6 else ''}")
            print(f"[exec] pass --allow_shrink to write only this run's systems.")
            merged = dict(have)
            merged.update(arrays)
            arrays = merged
        elif dropped:
            print(f"[exec] --allow_shrink given: DROPPING {len(dropped)} system(s) "
                  f"already on disk: {', '.join(dropped)}")

    np.savez_compressed(args.out, **arrays)
    json.dump(manifest, open(args.out.replace(".npz", "_manifest.json"), "w"), indent=2)

    ok = len(arrays)
    mismatched = [k for k, v in manifest.items()
                  if v["status"] == "ok" and not v.get("shape_match", True)]
    print(f"\n[exec] {ok}/{len(rows)} systems shipped a usable T_exec trajectory -> {args.out}")
    if mismatched:
        print(f"[exec] {len(mismatched)} systems where the solver's grid differs from "
              f"the stored trajectory's: {mismatched}")
        print("[exec] shipped anyway (the renderer normalizes shape, as it does for "
              "T_swap); flagged as a covariate")
    failed = {k: v for k, v in manifest.items() if v["status"] != "ok"}
    if failed:
        print("[exec] not available:")
        for k, v in failed.items():
            print(f"         {k:<18}{v['status']:<18}{v.get('reason','')[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
