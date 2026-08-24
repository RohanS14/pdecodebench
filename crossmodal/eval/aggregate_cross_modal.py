"""
aggregate_cross_modal.py — Part III results, with the clustering the design requires.

There are 1024 items but only 32 physical systems. Each system contributes 28
corrupted items and 4 clean ones, and items from one system share its code, its
equation, its description and its trajectory — so they are not independent draws.
Pooling them and taking a binomial interval on n=1024 understates the interval by
roughly the square root of the cluster size and will manufacture significance.

Every interval here is a bootstrap that resamples SYSTEMS with replacement, never
items. The A0 (clean) condition is resampled jointly with the corrupted conditions
inside each replicate, because every per-condition d' shares the same false-alarm
term and the seven values are therefore dependent.

Outputs a JSON report and a flat CSV, one row per (model, thinking, condition).

Usage:
    python eval/aggregate_cross_modal.py --results_dir results/xmodal \
        --items data/multimodal_items_v1.csv --out results/xmodal_summary.json
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from crossmodal.eval.parse_consistency import dprime  # noqa: E402

N_BOOT = 2000
SEED = 20260820


def load_rows(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _dprime_from(rows):
    """d' over a set of scored rows, or None when a cell is empty."""
    usable = [r for r in rows if r.get("detection_correct") is not None]
    signal = [r for r in usable if r["is_corrupted"]]
    noise = [r for r in usable if not r["is_corrupted"]]
    if not signal or not noise:
        return None
    n_hit = sum(r["detection_correct"] for r in signal)
    n_fa = len(noise) - sum(r["detection_correct"] for r in noise)
    return dprime(n_hit, len(signal), n_fa, len(noise))


def _rate(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return (sum(vals) / len(vals)) if vals else None


def cluster_bootstrap(by_system, statistic, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for `statistic`, resampling SYSTEMS with replacement.

    by_system: {system: [rows]}. The statistic sees the concatenated rows of one
    replicate, so whatever it needs from other conditions of the same system --
    the shared A0 false-alarm term in particular -- travels with it.
    """
    systems = sorted(by_system)
    if len(systems) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    point = statistic([r for s in systems for r in by_system[s]])
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(len(systems), size=len(systems))
        rows = [r for i in picked for r in by_system[systems[i]]]
        v = statistic(rows)
        if v is not None:
            draws.append(v)
    if len(draws) < n_boot // 2:
        return point, None, None          # too many degenerate replicates to trust
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/xmodal")
    ap.add_argument("--items", default="data/multimodal_items_v1.csv")
    ap.add_argument("--out", default="results/xmodal_summary.json")
    ap.add_argument("--csv_out", default="results/xmodal_summary.csv")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        sys.exit(f"[xmodal-agg] no rows in {args.results_dir}")
    print(f"[xmodal-agg] {len(rows)} scored rows")

    n_sys = len({r["gt_sample"] for r in rows})
    n_parse_fail = sum(1 for r in rows if r.get("parse_route") == "failed")
    print(f"[xmodal-agg] {n_sys} distinct systems -- every interval below is "
          f"bootstrapped over these, not over items")
    print(f"[xmodal-agg] parse failures: {n_parse_fail} "
          f"({100 * n_parse_fail / len(rows):.1f}%) -- scored as null, not as wrong")

    report, flat = {}, []
    arms = sorted({(r["model"], r.get("thinking", "na")) for r in rows})
    for model, thinking in arms:
        arm = [r for r in rows if r["model"] == model and r.get("thinking", "na") == thinking]
        clean = [r for r in arm if not r["is_corrupted"]]
        key = f"{model}|think_{thinking}"
        report[key] = {"n_rows": len(arm), "n_systems": len({r["gt_sample"] for r in arm}),
                       "parse_failure_rate": round(
                           sum(1 for r in arm if r.get("parse_route") == "failed") / len(arm), 4),
                       "conditions": {}}

        for cond in sorted({r["condition"] for r in arm}):
            cond_rows = [r for r in arm if r["condition"] == cond]
            # d' for a corrupted condition needs that condition's hits AND the
            # arm's clean items as its noise trials, so both travel together.
            pool = cond_rows if cond_rows[0]["is_corrupted"] is False else cond_rows + clean
            by_sys = defaultdict(list)
            for r in pool:
                by_sys[r["gt_sample"]].append(r)

            d, d_lo, d_hi = cluster_bootstrap(by_sys, _dprime_from, args.n_boot)
            det, det_lo, det_hi = cluster_bootstrap(
                {s: [r for r in v if r["condition"] == cond] for s, v in by_sys.items()},
                lambda rs: _rate(rs, "detection_correct"), args.n_boot)
            loc, loc_lo, loc_hi = cluster_bootstrap(
                {s: [r for r in v if r["condition"] == cond] for s, v in by_sys.items()},
                lambda rs: _rate(rs, "localization_correct"), args.n_boot)

            entry = {
                "n_items": len(cond_rows),
                "n_systems": len({r["gt_sample"] for r in cond_rows}),
                "detection": det, "detection_ci": [det_lo, det_hi],
                "localization": loc, "localization_ci": [loc_lo, loc_hi],
                "dprime": d, "dprime_ci": [d_lo, d_hi],
                "pde_class": _rate(cond_rows, "pde_class_match"),
                "num_method": _rate(cond_rows, "num_method_match"),
            }
            report[key]["conditions"][cond] = entry
            flat.append({"model": model, "thinking": thinking, "condition": cond,
                         **{k: v for k, v in entry.items() if not isinstance(v, list)},
                         "detection_lo": det_lo, "detection_hi": det_hi,
                         "dprime_lo": d_lo, "dprime_hi": d_hi})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_boot": args.n_boot, "seed": SEED, "clustered_on": "gt_sample",
                   "n_systems": n_sys, "arms": report}, f, indent=2)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"[xmodal-agg] wrote {args.out} and {args.csv_out} "
          f"({len(flat)} model x arm x condition cells)")


if __name__ == "__main__":
    main()
