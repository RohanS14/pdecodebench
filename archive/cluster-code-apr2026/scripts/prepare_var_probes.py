"""
Build probe_points.jsonl from pdedata_clean_v3.xlsx.

For each NoComm_CorrVar row:
  - Match to its NoComm_Valid counterpart by title substitution
  - Align NAME tokens to extract {foobar_N: gt_name} mapping
  - Filter to PDE-semantic variables (function params + substantial uses)
  - Emit one probe dict per variable occurrence

For each NoComm_Valid row:
  - Same probe positions as above, but corrupt_var == gt_var (identity)
  - Used as the baseline trajectory

Output: data/probe_points.jsonl

Usage:
    python scripts/prepare_var_probes.py \
        --dataset data/pdedata_clean_v3.xlsx \
        --output  data/probe_points.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from var_probe_utils import get_pde_semantic_map, extract_probe_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="data/pdedata_clean_v3.xlsx")
    parser.add_argument("--output",   default="data/probe_points.jsonl")
    parser.add_argument("--snippets", default=None,
                        help="Comma-separated snippet base names to include "
                             "(e.g. 'Burgers_1,NavierStokes_1'). Matches against "
                             "the part of the title before the condition suffix. "
                             "Default: all snippets.")
    args = parser.parse_args()

    snippet_filter: set[str] | None = None
    if args.snippets:
        snippet_filter = {s.strip() for s in args.snippets.split(",")}

    df = pd.read_excel(args.dataset)

    corr_var_rows  = df[df["mod_type"] == "NoComm_CorrVar"].reset_index(drop=True)
    if snippet_filter is not None:
        import re
        def _base_name(title: str) -> str:
            # Burgers_NoComm_CorrVar_1 → Burgers_1
            m = re.match(r"^([^_]+)_(?:NoComm_CorrVar|NoComm_Valid|[^_]+)_(\d+)$", title)
            return f"{m.group(1)}_{m.group(2)}" if m else title
        corr_var_rows = corr_var_rows[
            corr_var_rows["title"].apply(lambda t: _base_name(t) in snippet_filter)
        ].reset_index(drop=True)
        print(f"[prepare] Snippet filter active: {snippet_filter} → "
              f"{len(corr_var_rows)} rows", flush=True)

    valid_rows_map = {row["title"]: row
                      for _, row in df[df["mod_type"] == "NoComm_Valid"].iterrows()}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_probes = 0
    skipped = 0

    with open(args.output, "w") as out:
        for _, cv_row in corr_var_rows.iterrows():
            cv_title    = cv_row["title"]
            valid_title = cv_title.replace("NoComm_CorrVar", "NoComm_Valid")

            if valid_title not in valid_rows_map:
                print(f"[prepare] WARNING: no NoComm_Valid match for {cv_title!r} "
                      f"(tried {valid_title!r}) — skipping", flush=True)
                skipped += 1
                continue

            nv_row       = valid_rows_map[valid_title]
            corrupt_code = cv_row["code"]
            clean_code   = nv_row["code"]
            pde_class    = cv_row["pde_class"]

            # Build PDE-semantic variable map
            sem_map = get_pde_semantic_map(clean_code, corrupt_code)

            if not sem_map:
                print(f"[prepare] WARNING: no PDE-semantic variables found for "
                      f"{cv_title!r} — skipping", flush=True)
                skipped += 1
                continue

            print(f"[prepare] {cv_title}: {len(sem_map)} semantic vars: "
                  f"{sorted(sem_map.items())}", flush=True)

            # ── NoComm_CorrVar probes ──────────────────────────────────────────
            corr_probes = extract_probe_points(
                code=corrupt_code,
                semantic_map=sem_map,
                title=cv_title,
                mod_type="NoComm_CorrVar",
                pde_class=pde_class,
            )
            for p in corr_probes:
                out.write(json.dumps(p) + "\n")
            total_probes += len(corr_probes)

            # ── NoComm_Valid baseline probes (identity map: gt_name → gt_name) ─
            identity_map = {g: g for g in sem_map.values()}
            valid_probes = extract_probe_points(
                code=clean_code,
                semantic_map=identity_map,
                title=valid_title,
                mod_type="NoComm_Valid",
                pde_class=pde_class,
            )
            for p in valid_probes:
                out.write(json.dumps(p) + "\n")
            total_probes += len(valid_probes)

    print(f"\n[prepare] Done. {total_probes} probe points written to {args.output}",
          flush=True)
    if skipped:
        print(f"[prepare] {skipped} snippets skipped (no match or no semantic vars).",
              flush=True)


if __name__ == "__main__":
    main()
