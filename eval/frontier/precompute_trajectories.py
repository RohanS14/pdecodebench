"""
precompute_trajectories.py
Run each extracted PDE script and record a numeric trajectory summary.

Output: data/trajectories.jsonl  (one JSON object per script)
Run once before the belief revision eval:
    python eval/frontier/precompute_trajectories.py

The inspector identifies which numpy array is the main solution variable,
stores its name, and computes trajectory statistics over the time axis.
That array name is included in the trajectory block shown to the model so
it can relate the statistics back to the code it read.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent.parent
CODES_DIR  = REPO_ROOT / "data" / "extracted_codes"
OUT_FILE   = REPO_ROOT / "data" / "trajectories.jsonl"
PYTHON_BIN = REPO_ROOT / ".pde_venv" / "bin" / "python"
TIMEOUT    = 180  # generous for NavierStokes JAX compile

# Burgers_1 is now fixed — no exclusions needed.
# To re-exclude a sample, add its base name here e.g. {"Burgers_1"}
EXCLUDED_SAMPLES: set = set()

# ── Inspector script ──────────────────────────────────────────────────────────
# Runs as a subprocess; takes the target script path as argv[1].
# Emits a single JSON line to stdout.
_INSPECTOR = r"""
import sys, json, runpy, numpy as np, os

script_path = sys.argv[1]
title = os.path.splitext(os.path.basename(script_path))[0]
result = {"title": title, "ran_to_completion": False, "exec_error": None}
NULL_FIELDS = [
    "main_array_name", "shape", "has_nan", "has_inf", "finite_fraction",
    "first_nonfinite_time_index", "max_abs_initial", "max_abs_final",
    "max_abs_over_time_sampled", "max_abs_before_nan", "spike_ratio",
    "large_spike_detected",
]

def _candidate_arrays(ns):
    return {
        k: v for k, v in ns.items()
        if isinstance(v, np.ndarray)
        and v.ndim >= 1
        and v.size >= 10
        and not k.startswith("_")
        and np.issubdtype(v.dtype, np.floating)
    }

try:
    ns = runpy.run_path(script_path)
    result["ran_to_completion"] = True
except Exception as exc:
    result["exec_error"] = str(exc)
    for f in NULL_FIELDS:
        result[f] = None
    print(json.dumps(result))
    sys.exit(0)

# ── Find candidate arrays ─────────────────────────────────────────────────────
# Require: numpy float array, at least 10 elements, not a private variable.
# Note: JAX arrays are NOT numpy.ndarray — NavierStokes_4 scripts may return
# no candidates here; ran_to_completion will be True but all fields null.
# Fallback: if no arrays found, retry with run_name="__main__" to capture
# arrays defined only inside if __name__=="__main__": blocks.
arrays = _candidate_arrays(ns)
if not arrays:
    try:
        ns2 = runpy.run_path(script_path, run_name="__main__")
        arrays = _candidate_arrays(ns2)
    except Exception:
        pass

if not arrays:
    for f in NULL_FIELDS:
        result[f] = None
    print(json.dumps(result))
    sys.exit(0)

# ── Solution array selection ──────────────────────────────────────────────────
# Score each candidate; highest score wins, ties broken by size.
#   +2  assigned from odeint / solve_ivp  (detected via AST; works even for
#       obfuscated foobar_N names because those function names are not renamed)
#   +1  shape[0] matches the time array length (3rd arg of odeint, via AST)
#   +1  varies along axis-0  (nanstd > 0 → not a static matrix or coord grid)
import ast as _ast

_odeint_targets, _time_len = set(), None
try:
    with open(script_path) as _f:
        _tree = _ast.parse(_f.read())
    for _node in _ast.walk(_tree):
        if not isinstance(_node, _ast.Assign):
            continue
        _val = _node.value
        if not isinstance(_val, _ast.Call):
            continue
        _fn = _val.func
        _fname = _fn.attr if isinstance(_fn, _ast.Attribute) else (
                 _fn.id  if isinstance(_fn, _ast.Name) else None)
        if _fname not in ("odeint", "solve_ivp"):
            continue
        for _t in _node.targets:
            if isinstance(_t, _ast.Name):
                _odeint_targets.add(_t.id)
        # odeint(func, y0, t, ...) — t is the 3rd positional arg
        if _fname == "odeint" and len(_val.args) >= 3:
            _tname = _val.args[2]
            if isinstance(_tname, _ast.Name) and _tname.id in ns:
                _ta = ns[_tname.id]
                if isinstance(_ta, np.ndarray) and _ta.ndim == 1:
                    _time_len = len(_ta)
except Exception:
    pass

def _sol_score(name, a):
    s = 0
    if name in _odeint_targets:
        s += 2
    if _time_len and a.ndim >= 2 and a.shape[0] == _time_len:
        s += 1
    if a.ndim >= 2 and float(np.nanmean(np.nanstd(a, axis=0))) > 0:
        s += 1
    return s

# Hard overrides: NS_1/2 cavity-flow scripts store only a final spatial
# snapshot — no odeint, no time-series — so the heuristic falls back to
# size/insertion-order and picks the meshgrid X.  Explicitly prefer u
# (primary velocity field) which reflects simulation divergence.
_OVERRIDE = {
    # NavierStokes_1 — readable name / CorrVar foobar
    "NavierStokes_Comm_Valid_1":        "u", "NavierStokes_Comm_InValid_1":        "u",
    "NavierStokes_CorrComm_Valid_1":    "u", "NavierStokes_CorrComm_InValid_1":    "u",
    "NavierStokes_NoComm_Valid_1":      "u", "NavierStokes_NoComm_InValid_1":      "u",
    "NavierStokes_NoComm_CorrVar_1": "foobar_18", "NavierStokes_NoComm_CorrVar_InValid_1": "foobar_18",
    # NavierStokes_2
    "NavierStokes_Comm_Valid_2":        "u", "NavierStokes_Comm_InValid_2":        "u",
    "NavierStokes_CorrComm_Valid_2":    "u", "NavierStokes_CorrComm_InValid_2":    "u",
    "NavierStokes_NoComm_Valid_2":      "u", "NavierStokes_NoComm_InValid_2":      "u",
    "NavierStokes_NoComm_CorrVar_2": "foobar_19", "NavierStokes_NoComm_CorrVar_InValid_2": "foobar_19",
}
if title in _OVERRIDE and _OVERRIDE[title] in arrays:
    main_key, arr = _OVERRIDE[title], arrays[_OVERRIDE[title]]
else:
    main_key, arr = max(arrays.items(), key=lambda kv: (_sol_score(kv[0], kv[1]), kv[1].ndim, kv[1].size))
result["main_array_name"] = main_key
result["shape"] = list(arr.shape)
result["has_nan"]  = bool(np.isnan(arr).any())
result["has_inf"]  = bool(np.isinf(arr).any())
result["finite_fraction"] = round(float(np.isfinite(arr).sum() / arr.size), 6)

# ── Time-axis statistics (axis 0 treated as time) ────────────────────────────
if arr.ndim >= 2:
    ntime = arr.shape[0]

    # First time step with any non-finite value
    result["first_nonfinite_time_index"] = None
    for t in range(ntime):
        if not np.isfinite(arr[t]).all():
            result["first_nonfinite_time_index"] = t
            break

    # 10 evenly-spaced time samples
    tidx = list(dict.fromkeys(
        np.linspace(0, ntime - 1, min(10, ntime), dtype=int).tolist()
    ))
    result["max_abs_over_time_sampled"] = [
        round(float(np.nanmax(np.abs(arr[t]))), 6) for t in tidx
    ]
    result["max_abs_initial"] = round(float(np.nanmax(np.abs(arr[0]))),  6)
    result["max_abs_final"]   = round(float(np.nanmax(np.abs(arr[-1]))), 6)

    # Max abs at the last fully-finite time step
    result["max_abs_before_nan"] = None
    for t in range(ntime - 1, -1, -1):
        if np.isfinite(arr[t]).all():
            result["max_abs_before_nan"] = round(float(np.max(np.abs(arr[t]))), 6)
            break
else:
    # 1-D array — treat as single snapshot
    mx = round(float(np.nanmax(np.abs(arr))), 6)
    result["first_nonfinite_time_index"]  = None
    result["max_abs_initial"]             = mx
    result["max_abs_final"]               = mx
    result["max_abs_over_time_sampled"]   = [mx]
    result["max_abs_before_nan"] = mx if np.isfinite(arr).all() else None

init  = result["max_abs_initial"] or 0.0
final = result["max_abs_final"]   or 0.0
result["spike_ratio"] = round(final / (init + 1e-10), 4)
result["large_spike_detected"] = bool(
    result["has_nan"] or result["has_inf"]
    or result["spike_ratio"] > 1e3
    or final > 1e6
)

print(json.dumps(result))
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_base_sample(title: str) -> str:
    """'Burgers_Comm_InValid_1' → 'Burgers_1'"""
    parts = title.split("_")
    return f"{parts[0]}_{parts[-1]}"


def load_done(path: Path) -> set:
    done: set = set()
    if path.exists():
        for line in open(path):
            line = line.strip()
            if line:
                done.add(json.loads(line)["title"])
    return done


def print_sanity_check(rows: list[dict]) -> None:
    """
    Print a summary table after precompute completes.
    Flags:
      [FAIL]  ran_to_completion=False
      [NULL]  script ran but no float array was found (e.g. JAX scripts)
      [1-D]   only a 1-D array was found — time-axis stats are missing
    """
    print("\n" + "=" * 78)
    print("SANITY CHECK — array selection")
    print("=" * 78)
    print(f"{'Title':<45} {'Array':<20} {'Shape':<18} {'Flag'}")
    print("-" * 78)

    flags = {"FAIL": [], "NULL": [], "1-D": []}

    for r in sorted(rows, key=lambda x: x["title"]):
        title  = r["title"]
        arr    = r.get("main_array_name") or "—"
        shape  = str(r.get("shape") or "—")

        if not r.get("ran_to_completion"):
            flag = "FAIL"
        elif r.get("main_array_name") is None:
            flag = "NULL"
        elif r.get("shape") and len(r["shape"]) == 1:
            flag = "1-D"
        else:
            flag = ""

        if flag:
            flags[flag].append(title)

        marker = f"  ← {flag}" if flag else ""
        print(f"{title:<45} {arr:<20} {shape:<18}{marker}")

    print("=" * 78)
    total = len(rows)
    ok    = sum(1 for r in rows if r.get("ran_to_completion") and r.get("main_array_name") and
                r.get("shape") and len(r["shape"]) >= 2)
    print(f"\nSummary: {total} entries total")
    print(f"  ✓  {ok} — ran OK, 2-D+ array found")
    if flags["FAIL"]:
        print(f"  FAIL ({len(flags['FAIL'])}) — execution error:")
        for t in flags["FAIL"]:
            print(f"       {t}")
    if flags["NULL"]:
        print(f"  NULL ({len(flags['NULL'])}) — ran OK but no float array found (JAX?):")
        for t in flags["NULL"]:
            print(f"       {t}")
    if flags["1-D"]:
        print(f"  1-D  ({len(flags['1-D'])}) — only 1-D array found, no time-axis stats:")
        for t in flags["1-D"]:
            print(f"       {t}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    scripts = sorted(CODES_DIR.glob("*.py"))
    print(f"[precompute] {len(scripts)} scripts in {CODES_DIR}")

    done = load_done(OUT_FILE)
    print(f"[precompute] Already done: {len(done)}")

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, prefix="_pde_insp_"
    ) as f:
        f.write(_INSPECTOR)
        inspector = f.name

    new_rows: list[dict] = []

    try:
        out = open(OUT_FILE, "a")
        processed = skipped = excluded = 0

        for script in scripts:
            title = script.stem
            if get_base_sample(title) in EXCLUDED_SAMPLES:
                excluded += 1
                continue
            if title in done:
                skipped += 1
                continue

            print(f"  {title} ...", end=" ", flush=True)
            env = {**os.environ, "JAX_PLATFORM_NAME": "cpu", "TQDM_DISABLE": "1"}

            try:
                proc = subprocess.run(
                    [str(PYTHON_BIN), inspector, str(script)],
                    capture_output=True, text=True,
                    timeout=TIMEOUT, cwd=str(REPO_ROOT), env=env,
                )
                stdout = proc.stdout.strip()
                if not stdout:
                    raise RuntimeError(
                        f"no output (exit {proc.returncode}): {proc.stderr[:200]}"
                    )
                # Take the last non-empty line — scripts may print to stdout
                # during execution (e.g. "Time = 0.78"), so earlier lines are
                # not our JSON.
                last_line = stdout.splitlines()[-1].strip()
                row = json.loads(last_line)
            except subprocess.TimeoutExpired:
                row = {"title": title, "ran_to_completion": False,
                       "exec_error": "TIMEOUT"}
                for f in ["main_array_name","shape","has_nan","has_inf",
                          "finite_fraction","first_nonfinite_time_index",
                          "max_abs_initial","max_abs_final",
                          "max_abs_over_time_sampled","max_abs_before_nan",
                          "spike_ratio","large_spike_detected"]:
                    row[f] = None
            except Exception as exc:
                print(f"ERROR — {exc}")
                continue

            arr_info = f"{row.get('main_array_name','—')} {row.get('shape','')}"
            status   = "ok" if row.get("ran_to_completion") else f"FAIL({row.get('exec_error','')[:40]})"
            print(f"{status}  [{arr_info}]")

            out.write(json.dumps(row) + "\n")
            out.flush()
            new_rows.append(row)
            processed += 1

        out.close()
    finally:
        os.unlink(inspector)

    print(f"\n[precompute] processed={processed} skipped={skipped} excluded={excluded}")
    print(f"[precompute] → {OUT_FILE}")

    # ── Sanity check over everything written this run ─────────────────────────
    if new_rows:
        print_sanity_check(new_rows)
    else:
        print("[precompute] Nothing new to sanity-check (all already done).")


if __name__ == "__main__":
    main()
