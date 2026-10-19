"""
reorg_outputs.py — put every experiment under outputs/ into the one shape.

The cross-modal reorg of 2026-08-25 (tools/reorg_xmodal.py) left two trees in the
canonical layout and eleven pre-August experiments in the old flat one:

    canonical   outputs/<experiment>/<model-slug>/<arm>.<ext>
    legacy      outputs/results_<whatever>/<Model__Name>.jsonl

Same model, two different paths, depending on which month the experiment ran. This
brings the legacy eleven into the canonical shape, and applies the rule the roster
already follows: ONE file per model per experiment. A canary that the full run covers
is retired; a re-run that BACKFILLS is merged into the file it repairs rather than
kept beside it.

What the measurement said (tools/legacy_cmp.py, tools/couldfix.py), and why each
call is what it is:

  results_mc_valid_rerun    MERGE.  QwQ-32B spent its whole budget on all 96
                            phys_valid questions -- finish_reason='length',
                            predicted_letter=None, correct=None. The re-run rescored
                            them by text extraction and recovered 94 of the 96. That
                            repair was never merged back, so results_mc still
                            publishes nulls for a model that answered. Every other
                            model in the re-run is unchanged (0 flips, bar one
                            logprob-noise flip on Coder-7B).
  results_mc_valid_canary   RETIRE. keys_not_in_base=0 and one flip across 192 rows.
  results_var_logprob_canary RETIRE. keys_not_in_base=0; the full run supersedes its
                            numbers outright (224 of 442 logP_diff differ).
  results_old_v1            RETIRE. Its 96 keys are a subset of results_apr2026_v1's
                            144, and the parser moved underneath it.

Retiring is a MOVE to _trash/, never a delete -- the same reason reorg_xmodal.py
moved its 37 passes: "how was this row repaired" is a question a reviewer can ask.

The merge is REPAIR-ONLY and it is deliberately timid: a re-run row replaces a base
row only where the base has no answer at all (correct is None) and the re-run has
one. It never overwrites a scored row, so a re-run that silently changed methodology
cannot rewrite results behind you -- it can only fill holes.

Every merged row is stamped with source_arm. That column is not decoration: the
promotion of the cross-modal -final arms into the canonical tree DROPPED source_arm
(34 columns went in, 33 came out), and .upload_stage/ is now the only place that
lineage survives. Losing it again here would be the same mistake twice.

Usage:
    python tools/reorg_outputs.py             # dry run, prints every move
    APPLY=1 python tools/reorg_outputs.py     # do it
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import date

ROOT = os.environ.get("OUTPUTS_ROOT",
                      "/scratch/ehb7466/projects/pde-llm-eval/outputs")
TRASH = os.environ.get("TRASH_ROOT", "/scratch/ehb7466/_trash")
APPLY = os.environ.get("APPLY", "0") == "1"

# Experiments already in the canonical shape. Named so the report can say "left
# alone" rather than silently skipping them.
ALREADY_CANONICAL = ("free_generation", "cross_modal_consistency")

# legacy directory -> canonical experiment name. Several legacy dirs fold into one
# experiment; that is the point, and it is why outputs/ goes from 14 entries to 7.
FOLD = {
    "results_apr2026_v1":         "eval_v1",
    "results_mc":                 "mc_logprob",
    "results_var_logprob":        "var_logprob",
    "results_layer_probes":       "layer_probes",
    "results_pca":                "layer_probes",
    "hidden_states":              "layer_probes",
    "world_model":                "world_model",
}

# Merged INTO the directory named, never carried across as its own arm.
MERGE_INTO = {"results_mc_valid_rerun": "results_mc"}

# Superseded outright. Moved to _trash, not deleted.
RETIRE = ["results_old_v1", "results_mc_valid_canary", "results_var_logprob_canary"]

# Individual files that the same rule retires. The canary npz is covered by the full
# run's npz for that model; results.csv is a DERIVED aggregate computed before the
# QwQ-32B repair below, so carrying it across would ship a summary that still encodes
# the 96 nulls the merge just fixed. It is regenerable and the desktop holds a copy.
RETIRE_FILES = [
    "hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct_canary.npz",
    "results_mc/results.csv",
]

# The key that identifies one MC row. Same shape the comparison scripts used.
MC_KEY = ("title", "mod_type", "question_type", "correct_letter")


def slug(model_part):
    """`Qwen__Qwen3.8-27B` -> `qwen__qwen3-8-27b`.

    Matches the slugs the two canonical trees already use, so the same model is the
    same directory name in every experiment and a cross-experiment join stays a
    directory name rather than a lookup table. Dots become dashes because a dot in a
    path segment reads as an extension to half the tooling that walks these trees.
    """
    return model_part.lower().replace(".", "-")


def split_name(fname):
    """Return (model_slug, arm) for a legacy results filename.

    The legacy trees name files three different ways and each needs its own read:

        Qwen__QwQ-32B.jsonl                       model only, arm implied
        Qwen__QwQ-32B_hidden.npz                  model + arm suffix
        Qwen_Qwen2.5-Coder-32B-Instruct.npz       SINGLE underscore between org/model
        physics_vs_code_Qwen_Qwen2.5-..._last_tok_Comm_Valid.csv   model buried inside

    Returning None for the arm means "this file has no model dimension" and the
    caller parks it at the experiment root instead of inventing a model directory
    for it -- which is what a combined CSV actually is.
    """
    stem, ext = os.path.splitext(fname)

    # A per-experiment aggregate has no model dimension at all.
    if stem in ("results", "results_combined", "results_mc_combined",
                "results_merged", "pca_repr"):
        return None, None

    # world_model files bury the model between a fixed prefix and a fixed suffix.
    m = re.match(r"^(physics_vs_code|world_model_delta)_(.+?)_"
                 r"(last_tok|mean_pool)(?:_(.+))?$", stem)
    if m:
        kind, model, pool, rest = m.groups()
        arm = "_".join(x for x in (kind, pool, rest) if x)
        return slug(model.replace("_", "__", 1)), arm

    # `Org__Model` (canonical) or `Org_Model` (the npz files) plus an optional arm.
    m = re.match(r"^([A-Za-z0-9.\-]+)__([A-Za-z0-9.\-]+?)(?:_(hidden|probes|canary))?$",
                 stem)
    if not m:
        m2 = re.match(r"^([A-Za-z0-9.\-]+?)_([A-Za-z0-9.\-]+?)(?:_(hidden|probes|canary))?$",
                      stem)
        if not m2:
            return None, None
        m = m2
    org, model, arm = m.groups()
    return slug(f"{org}__{model}"), arm


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def merge_repair(base_path, rerun_path, base_arm, rerun_arm):
    """Fill holes in `base` from `rerun`. Returns (rows, n_repaired).

    Repair-only: a re-run row is taken ONLY where the base has no answer. A base row
    that was scored is never overwritten, so this cannot silently restate results.
    """
    base = read(base_path)
    rerun = {tuple(r.get(k) for k in MC_KEY): r for r in read(rerun_path)}
    out, repaired = [], 0
    for r in base:
        k = tuple(r.get(k_) for k_ in MC_KEY)
        cand = rerun.get(k)
        if cand is not None and r.get("correct") is None and cand.get("correct") is not None:
            cand = dict(cand)
            cand["source_arm"] = rerun_arm
            out.append(cand)
            repaired += 1
        else:
            r = dict(r)
            r.setdefault("source_arm", base_arm)
            out.append(r)
    return out, repaired


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"outputs root not found: {ROOT}")
    trash = os.path.join(TRASH, f"{date.today().isoformat()}-outputs-reorg")
    plan, merges, retires = [], [], []

    for legacy, exp in sorted(FOLD.items()):
        src = os.path.join(ROOT, legacy)
        if not os.path.isdir(src):
            continue
        for fname in sorted(os.listdir(src)):
            p = os.path.join(src, fname)
            if not os.path.isfile(p):
                continue
            if fname.endswith((".prerescore", ".pretruncfix", ".prenormalize")):
                continue  # backups live outside the tree, never inside a results dir
            if f"{legacy}/{fname}" in RETIRE_FILES:
                continue  # retired below rather than carried across
            s, arm = split_name(fname)
            ext = os.path.splitext(fname)[1]
            if s is None:
                dst = os.path.join(ROOT, exp, fname)          # aggregate, no model
            else:
                base = f"{arm}{ext}" if arm else f"{legacy}{ext}"
                dst = os.path.join(ROOT, exp, s, base)
            plan.append((p, dst, fname))

    for rerun_dir, base_dir in MERGE_INTO.items():
        rsrc = os.path.join(ROOT, rerun_dir)
        if not os.path.isdir(rsrc):
            continue
        for fname in sorted(os.listdir(rsrc)):
            bp = os.path.join(ROOT, base_dir, fname)
            if fname.endswith(".jsonl") and os.path.exists(bp):
                merges.append((bp, os.path.join(rsrc, fname), fname, rerun_dir, base_dir))

    for d in RETIRE:
        if os.path.isdir(os.path.join(ROOT, d)):
            retires.append(d)

    print(f"outputs root : {ROOT}")
    print(f"mode         : {'APPLY' if APPLY else 'DRY RUN'}")
    print(f"untouched    : {', '.join(ALREADY_CANONICAL)}\n")

    print(f"=== merge (repair-only, into the file it repairs) ===")
    total_rep = 0
    for bp, rp, fname, rd, bd in merges:
        rows, rep = merge_repair(bp, rp, bd, rd)
        total_rep += rep
        flag = "  <-- repairs" if rep else ""
        print(f"  {fname:46s} {len(rows):5d} rows, {rep:3d} repaired{flag}")
        if APPLY:
            # The pre-merge copy goes straight to _trash, not beside the file it
            # backs up. Left in place it survives the move loop below (which skips
            # backup suffixes by design) and strands a half-empty legacy directory
            # holding nothing but backups -- which is exactly the "one file per
            # model" rule this script exists to enforce, broken by the script.
            os.makedirs(os.path.join(trash, "premerge"), exist_ok=True)
            shutil.copy2(bp, os.path.join(trash, "premerge", fname + ".premerge"))
            with open(bp, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
    print(f"  {total_rep} row(s) repaired across {len(merges)} file(s)")
    if merges:
        print(f"  (re-run dir retired after merge: {list(MERGE_INTO)[0]})")

    print(f"\n=== move into <experiment>/<model-slug>/<arm> ===")
    by_exp = Counter()
    for _, dst, _ in plan:
        by_exp[os.path.relpath(dst, ROOT).split(os.sep)[0]] += 1
    for e, n in sorted(by_exp.items()):
        print(f"  {e:18s} {n:3d} file(s)")
    for src, dst, fname in plan:
        print(f"    {os.path.relpath(src, ROOT):58s} -> {os.path.relpath(dst, ROOT)}")

    print(f"\n=== retire to {trash} ===")
    for d in retires + list(MERGE_INTO):
        print(f"  {d}/")
    for f in RETIRE_FILES:
        if os.path.exists(os.path.join(ROOT, f)):
            print(f"  {f}")

    if not APPLY:
        print("\nDRY RUN. Re-run with APPLY=1.")
        return

    for src, dst, _ in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    os.makedirs(trash, exist_ok=True)
    for f in RETIRE_FILES:
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            dst = os.path.join(trash, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    for d in retires + list(MERGE_INTO):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            shutil.move(p, os.path.join(trash, d))
    for legacy in FOLD:
        p = os.path.join(ROOT, legacy)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)
    print("\napplied.")


if __name__ == "__main__":
    main()
