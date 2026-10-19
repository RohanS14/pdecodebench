"""
rescore_jsonl.py — re-run parsing and scoring over stored model_response text.

Generation is expensive; parsing is not. When the parser or a score definition
changes, the responses on disk are still valid data — only the derived columns are
stale. This re-derives them in place and prints exactly what moved, so a scoring
change is never silently folded into a results file.

It never touches model_response, finish_reason, or any generation-time field.

Usage:
    python freegen_static_judgments/rescore_jsonl.py --results_dir outputs/freegen_jul28_canary \
        --prompt_version v1-valid-compound
"""
import argparse
import glob
import json
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_score import (parse_response, score_row, classify_valid_confidence,  # noqa: E402
                         is_no_verdict, recovered_loop_verdict)

try:
    from parse_score import method_axis  # noqa: E402
except ImportError:
    method_axis = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--prompt_version", default=None,
                    help="Stamp rows with the prompt they were GENERATED under. These "
                         "rows predate the stamp, so it has to be supplied by hand and "
                         "must match the job that produced them.")
    ap.add_argument("--no_embed", action="store_true",
                    help="Skip pde_embed_sim. Leaves the existing value untouched.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    embed_model = None
    if not args.no_embed:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[rescore] embedding model loaded")

    # RECURSIVE, same as aggregate_freegen.py: the one-job-per-model launcher gives
    # each model its own subdirectory, and a flat glob finds nothing under it.
    paths = sorted(glob.glob(os.path.join(args.results_dir, "**", "*.jsonl"),
                             recursive=True))
    if not paths:
        sys.exit(f"[rescore] No *.jsonl in {args.results_dir}")

    changed = Counter()
    rescued = []
    recovered = Counter()
    for path in paths:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for r in rows:
            before = {k: r.get(k) for k in
                      ("parsed_pde", "parsed_method", "parsed_behavior", "parsed_valid",
                       "pde_match", "method_recall", "behavior_recall", "valid_match",
                       "no_verdict", "verdict_recovered")}

            parsed = parse_response(r["model_response"] or "")
            r["parsed_pde"]      = parsed.get("pde")
            r["parsed_method"]   = parsed.get("method")
            r["parsed_behavior"] = parsed.get("behavior")
            r["parsed_valid"]    = parsed.get("valid")

            # score_row reads dataset columns off the row itself; the JSONLs carry
            # them under the same names run_eval used.
            gt = {"pde_class": r["gt_pde"], "num_method": r["gt_method"],
                  "phys_process": r["gt_behavior"], "phys_valid": r["gt_valid"]}
            scored = score_row(parsed, gt, embed_model if not args.no_embed else None)
            if args.no_embed:
                scored.pop("pde_embed_sim", None)
            r.update(scored)
            # no_verdict is DERIVED, so it is re-derived here like every other
            # derived column. Leaving the stored value alone would strand a row that
            # the current rule can now read as answered -- the exact staleness this
            # script exists to remove.
            fr = r.get("finish_reason", "")
            rec = recovered_loop_verdict(r["model_response"] or "", fr)
            r["verdict_recovered"] = rec is not None
            r["no_verdict"] = bool(is_no_verdict(r["model_response"] or "", fr))
            r["valid_conf"] = classify_valid_confidence(parsed.get("valid"))
            if method_axis is not None and "method_axis" not in scored:
                r["method_axis"] = method_axis(parsed.get("method"))
            if args.prompt_version:
                r["prompt_version"] = args.prompt_version

            if rec is not None and not before.get("verdict_recovered"):
                recovered[r["model"]] += 1
            after = {k: r.get(k) for k in before}
            if before != after:
                changed[r["model"]] += 1
                if before["parsed_pde"] is None and after["parsed_pde"] is not None:
                    rescued.append((r["model"], r["title"]))

        if not args.dry_run:
            shutil.copy(path, path + ".prerescore")
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"[rescore] {len(rows):>4} rows  {os.path.basename(path)}")

    print(f"\n[rescore] rows whose derived columns moved: {dict(changed)}")
    print(f"[rescore] rows rescued from a null parse: {len(rescued)}")
    print(f"[rescore] looped-but-answered rows recovered: {dict(recovered)} "
          f"(total {sum(recovered.values())})")
    for m, t in rescued[:12]:
        print(f"[rescore]   {m}  {t}")
    if args.dry_run:
        print("[rescore] DRY RUN - nothing written")
    else:
        print("[rescore] originals kept alongside as *.prerescore")


if __name__ == "__main__":
    main()
