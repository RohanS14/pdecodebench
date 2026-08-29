"""
render_equation_review.py — one-pass sign-off sheet for data/equations_jul28.csv.

The equation file is the physics ground truth for every cross-modal number in
Experiment 2 Parts II and III. Its `verified_by` column is empty on all 32 rows,
which means no number downstream of it is trustworthy yet, and a wrong equation
corrupts results in a way no downstream statistic can detect.

Signing off means reading each equation against the solver it was derived from.
This puts those two things side by side so that is a single pass rather than 32
file-openings. Nothing here writes to the CSV; sign-off is recorded by filling in
`verified_by`.

Usage:
    python cross_modal_consistency/datagen/render_equation_review.py --out notes/.../equation_review.html
"""
import argparse
import html
import os

import pandas as pd

CSS = """
:root { --bg:#fbfbfa; --fg:#1a1a19; --mut:#6b6b68; --line:#e3e3e0; --card:#fff;
        --flag:#b4530a; --flagbg:#fff4e8; --ok:#2f6f4f; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#151514; --fg:#e8e8e6; --mut:#9a9a96; --line:#2e2e2b; --card:#1d1d1c;
  --flag:#e8935a; --flagbg:#2a1e14; --ok:#7fc0a0; } }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 5rem; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .35rem; letter-spacing:-.01em; }
.sub { color:var(--mut); margin:0 0 1.5rem; font-size:.92rem; }
.bar { display:flex; gap:1.5rem; flex-wrap:wrap; padding:.85rem 1rem; margin-bottom:1.5rem;
  background:var(--card); border:1px solid var(--line); border-radius:10px; }
.bar div { font-size:.85rem; color:var(--mut); }
.bar b { display:block; font-size:1.3rem; color:var(--fg); font-weight:600; }
.sys { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:1rem 1.15rem; margin-bottom:.85rem; }
.sys.flagged { border-color:var(--flag); background:var(--flagbg); }
.hd { display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap; margin-bottom:.6rem; }
.name { font-weight:650; font-size:1.02rem; }
.tag { font-size:.72rem; color:var(--mut); border:1px solid var(--line);
  border-radius:99px; padding:.1rem .5rem; }
.eq { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:1.02rem;
  padding:.6rem .75rem; background:var(--bg); border:1px solid var(--line);
  border-radius:7px; overflow-x:auto; white-space:nowrap; }
.ev { margin-top:.55rem; font-size:.83rem; color:var(--mut); }
.ev code { color:var(--fg); }
.flag { margin-top:.55rem; font-size:.85rem; color:var(--flag); }
details { margin-top:.6rem; }
summary { cursor:pointer; font-size:.85rem; color:var(--mut); }
pre { margin:.55rem 0 0; padding:.75rem; background:var(--bg); border:1px solid var(--line);
  border-radius:7px; overflow-x:auto; font-size:12.5px; line-height:1.45; }
.note { font-size:.85rem; color:var(--mut); border-left:2px solid var(--line);
  padding-left:.8rem; margin:2rem 0 0; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equations", default="data/equations_jul28.csv")
    ap.add_argument("--base",      default="data/merged_base_jul28.csv")
    ap.add_argument("--out",       required=True)
    args = ap.parse_args()

    eq = pd.read_csv(args.equations)
    base = pd.read_csv(args.base)
    valid_code = {r["gt_sample"]: r["code"]
                  for _, r in base[base["phys_valid"] == True].iterrows()}  # noqa: E712

    n_flag = int((eq["needs_review"] == 1).sum())
    n_signed = int(eq["verified_by"].notna().sum())

    parts = [f"<style>{CSS}</style><div class='wrap'>",
             "<h1>Equation sign-off — <code>equations_jul28.csv</code></h1>",
             "<p class='sub'>Physics ground truth for Experiment 2 Parts II and III. "
             "Each equation shown against the valid solver it was derived from. "
             "Record sign-off by filling <code>verified_by</code> in the CSV.</p>",
             "<div class='bar'>"
             f"<div>Systems<b>{len(eq)}</b></div>"
             f"<div>Signed off<b>{n_signed} / {len(eq)}</b></div>"
             f"<div>Flagged by the builder<b>{n_flag}</b></div>"
             f"<div>PDE classes<b>{eq['pde_class'].nunique()}</b></div>"
             "</div>"]

    for _, r in eq.sort_values(["pde_class", "gt_sample"]).iterrows():
        flagged = r["needs_review"] == 1
        parts.append(f"<div class='sys{' flagged' if flagged else ''}'>")
        parts.append(
            f"<div class='hd'><span class='name'>{html.escape(str(r['gt_sample']))}</span>"
            f"<span class='tag'>{html.escape(str(r['pde_class']))}</span>"
            f"<span class='tag'>{html.escape(str(r['dim']))}</span>"
            f"<span class='tag'>{html.escape(str(r['variant']))}</span>"
            f"<span class='tag'>{html.escape(str(r['source']))}</span></div>")
        parts.append(f"<div class='eq'>{html.escape(str(r['equation_unicode']))}</div>")

        ev = []
        for label, col in (("features", "evidence_features"), ("dim", "evidence_dim"),
                           ("method", "evidence_num_method"),
                           ("process", "evidence_phys_process")):
            v = r[col]
            if pd.notna(v):
                ev.append(f"{label} <code>{html.escape(str(v))}</code>")
        if ev:
            parts.append("<div class='ev'>derived from " + " &nbsp;·&nbsp; ".join(ev) + "</div>")
        if flagged:
            parts.append(f"<div class='flag'>⚑ {html.escape(str(r['review_reason']))}</div>")

        code = valid_code.get(r["gt_sample"])
        if code:
            parts.append("<details><summary>valid solver source</summary>"
                         f"<pre>{html.escape(str(code))}</pre></details>")
        else:
            parts.append("<div class='ev'>no valid-variant source found in the base file</div>")
        parts.append("</div>")

    parts.append(
        "<p class='note'>Sign-off is per row. A row is signed when someone has read the "
        "equation against the solver above it and agrees they describe the same system — "
        "not that the equation is a famous one. The single flagged row states its own "
        "doubt; the other 31 are unflagged because the builder found no contradiction, "
        "which is weaker than a human having checked.</p></div>")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"[equation-review] wrote {args.out}  ({len(eq)} systems, {n_flag} flagged, "
          f"{n_signed} already signed)")


if __name__ == "__main__":
    main()
