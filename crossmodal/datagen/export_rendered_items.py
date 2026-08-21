"""
export_rendered_items.py — every item with the text the model actually sees.

data/multimodal_items_v1.csv says which views an item shows and which is corrupted.
It does not say what those views CONTAIN, so auditing an item means running the
prompt builder. This writes the join: one row per item, the four view bodies in
slot order, and optionally the assembled prompt verbatim.

The point is that any claim about an item can be checked by reading a row -- what
the code view said, whether the description named the PDE family, what the numeric
table looked like -- without a GPU, a venv, or trust in this pipeline.

Rows are wide: a rendered trajectory table runs ~9 KB, so the full file is roughly
15 MB with prompts and 12 MB without. Written with csv.QUOTE_ALL and \\n preserved
inside quoted fields; read it with pandas.read_csv, not by splitting on newlines.

Usage:
    python crossmodal/datagen/export_rendered_items.py --out data/multimodal_items_rendered_v1.csv
    python crossmodal/datagen/export_rendered_items.py --no_prompt --limit 32   # quick look
"""
import argparse
import csv
import os
import sys

# repo root: this file sits at crossmodal/<area>/, so three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
csv.field_size_limit(10 ** 9)

from crossmodal.datagen.build_multimodal_items import MOD_DATASET, MULTIMODAL_CSV   # noqa: E402
from crossmodal.eval.consistency_prompts import (                                   # noqa: E402
    ViewSources, build_prompt, load_exec_trajectories, load_items, materialize_views,
)

CARRY = ["item_id", "gt_sample", "pde_class", "num_method", "phys_process", "source",
         "condition", "corrupted_view", "traj_level", "names", "order_seed",
         "slot_1", "slot_2", "slot_3", "slot_4", "outlier_slot",
         "gt_pde_class", "gt_num_method", "invalidity_note",
         "flag_identity_leak", "flag_near_duplicate", "flag_render_hard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/multimodal_items_v1.csv")
    ap.add_argument("--multimodal", default=MULTIMODAL_CSV)
    ap.add_argument("--dataset", default=MOD_DATASET)
    ap.add_argument("--exec_npz", default="data/exec_trajectories.npz")
    ap.add_argument("--out", default="data/multimodal_items_rendered_v1.csv")
    ap.add_argument("--no_prompt", action="store_true",
                    help="Omit the assembled prompt column (smaller file).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exec_traj = load_exec_trajectories(args.exec_npz)
    print(f"[render-export] T_exec trajectories available: {len(exec_traj)}/32")
    sources = ViewSources(args.multimodal, args.dataset, exec_traj)
    items = load_items(args.items)
    if args.limit:
        items = items[:args.limit]

    cols = list(CARRY)
    for i in range(1, 5):
        cols += [f"view_{i}_kind", f"view_{i}_is_outlier", f"view_{i}_text",
                 f"view_{i}_chars"]
    if not args.no_prompt:
        cols += ["prompt", "prompt_chars"]
    cols += ["skipped_reason"]

    n_written = n_skipped = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        w.writeheader()
        for item in items:
            row = {k: item.get(k, "") for k in CARRY}
            try:
                bodies = materialize_views(item, sources)
            except KeyError as e:
                # A system with no trajectory for this rung. Recorded as a row with
                # a reason rather than dropped, so the file's item count always
                # matches the item set and a gap is visible instead of inferred.
                row["skipped_reason"] = f"{type(e).__name__}: {e}"
                w.writerow(row)
                n_skipped += 1
                continue

            slots = [item[f"slot_{i}"] for i in range(1, 5)]
            for i, kind in enumerate(slots, start=1):
                body = bodies[kind]
                row[f"view_{i}_kind"] = kind
                row[f"view_{i}_is_outlier"] = int(
                    item["corrupted_view"] not in ("", "none") and kind == item["corrupted_view"])
                row[f"view_{i}_text"] = body
                row[f"view_{i}_chars"] = len(body)
            if not args.no_prompt:
                prompt = build_prompt(item, sources)
                row["prompt"] = prompt
                row["prompt_chars"] = len(prompt)
            row["skipped_reason"] = ""
            w.writerow(row)
            n_written += 1

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[render-export] {n_written} rows written, {n_skipped} skipped "
          f"-> {args.out} ({size_mb:.1f} MB)")
    if n_skipped:
        print(f"[render-export] skipped rows carry skipped_reason and are still "
              f"present, so the row count matches the item set")


if __name__ == "__main__":
    main()
