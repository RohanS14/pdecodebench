"""
Plain-language report: how far each perturbation moves a solver, and whether the
arrangement of solvers survives it.

Hand-rolled SVG/HTML — no plotting library, so the page is small and every mark is
inspectable. Reads the JSON from solver_distances.py.

Usage:
    python probe/viz_solver_distances.py \
        --data probe/results/solver_distances_qwen7b.json \
        --output probe/results/solver_distances.html
"""
import argparse
import json
import os

# dataviz reference palette, slots 1 (blue) and 2 (orange); validated all-pairs.
C_DESC_L, C_DESC_D = "#2a78d6", "#3987e5"     # description changed
C_PHYS_L, C_PHYS_D = "#eb6834", "#d95926"     # physics changed

ORDER = ["NoComm_Valid", "CorrComm", "NoComm_CorrVar",
         "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid",
         "NoComm_CorrVar_InValid"]
SHORT = {
    "NoComm_Valid": "comments removed",
    "CorrComm": "comments made misleading",
    "NoComm_CorrVar": "comments removed + names obfuscated",
    "Comm_InValid": "physics broken · comments kept",
    "NoComm_InValid": "physics broken · no comments",
    "CorrComm_Invalid": "physics broken · misleading comments",
    "NoComm_CorrVar_InValid": "physics broken · names obfuscated",
}


def bars(rows, vmax, width=620, rowh=44, decimals=3, arrow="&rarr;"):
    """
    Horizontal bars where each row names BOTH things being compared.

    rows = [(from_label, to_label, gloss, value, kind)].
    The pair is printed as `from -> to` on the first line and a plain-language
    gloss on the second, so no bar can be read without knowing its two endpoints.
    """
    h = len(rows) * rowh + 6
    barx = 330
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" height="{h}" '
           f'role="img" class="chart">']
    for i, (frm, to, gloss, v, kind) in enumerate(rows):
        y = i * rowh + 4
        w = max(2, (v / vmax) * (width - barx - 62)) if vmax else 2
        fill = f"var(--{'phys' if kind == 'physics' else 'desc'})"
        out.append(
            f'<text x="0" y="{y+15}" class="pair">'
            f'<tspan class="ep">{frm}</tspan>'
            f'<tspan class="arw"> {arrow} </tspan>'
            f'<tspan class="ep">{to}</tspan></text>'
            f'<text x="0" y="{y+30}" class="gloss">{gloss}</text>'
            f'<rect x="{barx}" y="{y+8}" width="{w:.1f}" height="15" rx="4" fill="{fill}">'
            f'<title>{frm} vs {to}: {v:.{decimals}f}</title></rect>'
            f'<text x="{barx+w+7:.1f}" y="{y+20}" class="val">{v:.{decimals}f}</text>')
    out.append("</svg>")
    return "".join(out)


def curves(series, n_layers, width=620, height=250, ymin=-0.05, ymax=1.05):
    """
    One line per condition across ALL layers. series = [(label, values, kind)].
    Exists so no claim rests on a single hand-picked layer.
    """
    pad_l, pad_b, pad_t, pad_r = 34, 26, 8, 150
    pw, ph = width - pad_l - pad_r, height - pad_b - pad_t

    def X(l):
        return pad_l + l / max(1, n_layers - 1) * pw

    def Y(v):
        return pad_t + (1 - (v - ymin) / (ymax - ymin)) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" class="chart">']
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pad_l}" y1="{Y(gv):.1f}" x2="{pad_l+pw}" y2="{Y(gv):.1f}" '
                   f'class="ax"/><text x="{pad_l-6}" y="{Y(gv)+3:.1f}" class="axt" '
                   f'text-anchor="end">{gv:.2f}</text>')
    for l in (0, n_layers // 2, n_layers - 1):
        out.append(f'<text x="{X(l):.1f}" y="{height-8}" class="axt" '
                   f'text-anchor="middle">layer {l}</text>')
    for label, vals, kind in series:
        col = f"var(--{'phys' if kind == 'physics' else 'desc'})"
        pts = " ".join(f"{X(l):.1f},{Y(v):.1f}" for l, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" class="ln" stroke="{col}"/>')
        out.append(f'<text x="{pad_l+pw+8:.1f}" y="{Y(vals[-1])+3:.1f}" class="axt" '
                   f'fill="{col}">{label}</text>')
    out.append("</svg>")
    return "".join(out)


def heat(M, labels, classes, size=250):
    """Solver x solver distance matrix, PDE class labelled on both axes."""
    n = len(M)
    cell = size / n
    PAD = 62                      # gutter for the class names
    lo = min(min(r) for r in M)
    hi = max(max(r) for r in M)
    rng = (hi - lo) or 1.0
    out = [f'<svg viewBox="0 0 {size+PAD+2} {size+PAD+2}" width="100%" '
           f'style="max-width:{size+PAD+2}px" role="img" class="chart">',
           f'<g transform="translate({PAD},{PAD})">']
    for i in range(n):
        for j in range(n):
            t = (M[i][j] - lo) / rng
            # single-hue ramp (blue), light -> dark with magnitude
            l = 97 - 62 * t
            out.append(f'<rect x="{j*cell:.2f}" y="{i*cell:.2f}" width="{cell:.2f}" '
                       f'height="{cell:.2f}" fill="hsl(213 62% {l:.1f}%)">'
                       f'<title>{labels[i]} ↔ {labels[j]}: {M[i][j]:.3f}</title></rect>')
    # class boundaries
    seen, pos = None, 0
    for i, c in enumerate(classes):
        if c != seen:
            if i:
                out.append(f'<line x1="0" y1="{i*cell:.2f}" x2="{size}" y2="{i*cell:.2f}" '
                           f'class="sep"/><line x1="{i*cell:.2f}" y1="0" '
                           f'x2="{i*cell:.2f}" y2="{size}" class="sep"/>')
            seen = c
    out.append("</g>")
    # class names: down the left edge and across the top
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or classes[i] != classes[start]:
            runs.append((classes[start], start, i))
            start = i
    for name, a, b in runs:
        mid = PAD + (a + b) / 2 * cell
        out.append(f'<text x="{PAD-6}" y="{mid+4:.1f}" class="cls" '
                   f'text-anchor="end">{name}</text>')
        out.append(f'<text x="{mid:.1f}" y="{PAD-8}" class="cls" '
                   f'text-anchor="middle" transform="rotate(-32 {mid:.1f} {PAD-8})">'
                   f'{name}</text>')
    out.append("</svg>")
    return "".join(out)


def by_class(pert, solvers, classes, layerless_key="per_solver_at_rsa",
             width=560, rowh=17):
    """
    Valid -> invalid distance for EVERY solver, grouped by PDE class.
    One row per solver; four small multiples, one per surface condition.
    Single hue throughout (this is all the same measurement), so class identity
    is carried by grouping and labels, never by colour.
    """
    conds = [c for c in ORDER if pert[c]["kind"] == "physics"]
    vmax = max(max(pert[c][layerless_key]) for c in conds)
    order = sorted(range(len(solvers)), key=lambda i: (classes[i], solvers[i]))
    out = []
    for c in conds:
        vals = pert[c][layerless_key]
        h = len(order) * rowh + 26
        svg = [f'<svg viewBox="0 0 {width} {h}" width="100%" height="{h}" '
               f'role="img" class="chart">']
        seen = None
        for row, i in enumerate(order):
            y = row * rowh + 20
            if classes[i] != seen:
                seen = classes[i]
                svg.append(f'<text x="0" y="{y+9}" class="clsrow">{seen}</text>')
            w = max(1.5, vals[i] / vmax * (width - 250))
            svg.append(
                f'<text x="104" y="{y+9}" class="lbl2">{solvers[i]}</text>'
                f'<rect x="188" y="{y+2}" width="{w:.1f}" height="11" rx="3" '
                f'fill="var(--phys)"><title>{solvers[i]} ({classes[i]}): '
                f'{vals[i]:.3f}</title></rect>'
                f'<text x="{188+w+6:.1f}" y="{y+11}" class="val2">{vals[i]:.3f}</text>')
        svg.append("</svg>")
        med = sorted(vals)[len(vals) // 2]
        out.append(f'<div class="mult"><p class="cardt">'
                   f'<span class="ep">{pert[c]["vs"]}</span>'
                   f'<span class="arw"> &rarr; </span>'
                   f'<span class="ep">{c}</span><br>'
                   f'<span class="muted">{SHORT[c]} &middot; median {med:.3f}</span></p>'
                   f'{"".join(svg)}</div>')
    return f'<div class="mgrid">{"".join(out)}</div>'


def class_summary(pert, classes, key="per_solver_at_rsa"):
    """Median valid->invalid distance per PDE class, per surface condition."""
    conds = [c for c in ORDER if pert[c]["kind"] == "physics"]
    uniq = sorted(set(classes))
    head = "".join(f"<th class='n'>{u}</th>" for u in uniq)
    body = []
    for c in conds:
        v = pert[c][key]
        cells = []
        for u in uniq:
            xs = sorted(v[i] for i, cl in enumerate(classes) if cl == u)
            cells.append(f"<td class='n'>{xs[len(xs)//2]:.3f}</td>")
        body.append(f"<tr><td><span class='ep'>{pert[c]['vs']}</span>"
                    f"<span class='arw'> &rarr; </span><span class='ep'>{c}</span>"
                    f"<br><span class='muted'>{SHORT[c]}</span></td>"
                    f"{''.join(cells)}</tr>")
    return (f"<table><thead><tr><th>median distance from each solver&rsquo;s "
            f"valid version to its invalid twin</th>{head}"
            f"</tr></thead><tbody>{''.join(body)}</tbody></table>")


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8880;
--line:#e6e5e1;--desc:#2a78d6;--phys:#eb6834;--card:#ffffff;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8a8880;--line:#33322f;
--desc:#3987e5;--phys:#d95926;--card:#222220;color-scheme:dark;}}
:root[data-theme=dark]{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8a8880;
--line:#33322f;--desc:#3987e5;--phys:#d95926;--card:#222220;color-scheme:dark;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:900px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:44px 0 6px;
padding-bottom:7px;border-bottom:1px solid var(--line)}
.sub{color:var(--ink2);margin:0 0 6px}
.q{font-size:15px;color:var(--ink);margin:0 0 14px}
.note{font-size:13.5px;color:var(--ink2);margin:12px 0 0}
.chart{display:block;margin:6px 0 2px;overflow:visible}
.lbl{font-size:12.5px;fill:var(--ink2)}
.val{font-size:12.5px;fill:var(--ink);font-variant-numeric:tabular-nums}
.sep{stroke:var(--surface);stroke-width:1.5}
.key{display:flex;gap:18px;align-items:center;font-size:13px;color:var(--ink2);margin:2px 0 10px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:6px;
vertical-align:-1px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:20px;margin-top:8px}
.cardt{font-size:13px;color:var(--ink2);margin:0 0 4px}
.big{font-size:15px;color:var(--ink);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
th{text-align:left;color:var(--ink2);font-weight:600;border-bottom:2px solid var(--line);padding:7px 9px}
td{border-bottom:1px solid var(--line);padding:7px 9px}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.callout{background:var(--card);border-left:3px solid var(--phys);padding:13px 16px;
margin:16px 0;font-size:14.5px}
code{background:var(--card);padding:1px 5px;border-radius:3px;font-size:13px}
.cls{font-size:9.5px;fill:var(--ink2)}
.pair{font-size:12.5px}
.ep{fill:var(--ink);font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.arw{fill:var(--ink3)}
.gloss{font-size:11px;fill:var(--ink2)}
.ax{stroke:var(--line);stroke-width:1}
.axt{font-size:10px;fill:var(--ink3)}
.ln{fill:none;stroke-width:2}
.clsrow{font-size:12px;fill:var(--ink);font-weight:600}
.lbl2{font-size:11px;fill:var(--ink2)}
.val2{font-size:10.5px;fill:var(--ink2);font-variant-numeric:tabular-nums}
.muted{color:var(--ink3);font-weight:400}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:26px}
.mult{min-width:0}
.defn{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:15px 18px;margin:14px 0 18px;font-size:14.5px}
.defn b{font-weight:600}
.defn .eq{display:block;margin:9px 0 4px;font-size:14px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="probe/results/solver_distances.html")
    ap.add_argument("--pool", default="last_tok")
    args = ap.parse_args()

    D = json.load(open(args.data))
    P = D["pools"][args.pool]
    L = D["rsa_layer"]
    solvers, classes = D["solvers"], D["pde_class"]
    pert = P["perturbations"]

    # Q1 — displacement
    rows1 = [(pert[c]["vs"], c, SHORT[c], pert[c]["mean_by_layer"][L], pert[c]["kind"])
             for c in ORDER if c in pert]
    vmax1 = max(r[3] for r in rows1)

    desc = [r[3] for r in rows1 if r[4] == "surface"]
    phys = [r[3] for r in rows1 if r[4] == "physics"]
    ratio = (sum(phys) / len(phys)) / (sum(desc) / len(desc))

    # Q2 — geometry preservation
    gp = P["geometry_preserved"]
    gpl = P.get("geometry_preserved_by_layer", {})
    rows2 = sorted([(f"{pert[c]['vs']}", c, SHORT[c], gp[c], pert[c]["kind"])
                    for c in ORDER if c in gp], key=lambda r: -r[3])
    curve_series = [(c.replace("NoComm_CorrVar_InValid", "NoCorrVar_Inv")
                      .replace("CorrComm_Invalid", "CorrComm_Inv")
                      .replace("NoComm_InValid", "NoComm_Inv"),
                     gpl[c], pert[c]["kind"]) for c in ORDER if c in gpl]

    # Q3 — matrices
    show = ["Comm_Valid", "CorrComm", "NoComm_CorrVar", "Comm_InValid"]
    titles = {"Comm_Valid": "unperturbed", "CorrComm": "misleading comments",
              "NoComm_CorrVar": "names obfuscated", "Comm_InValid": "physics broken"}
    cards = "".join(
        f'<div><p class="cardt">{titles[c]}</p>{heat(P["matrices"][c], solvers, classes)}</div>'
        for c in show if c in P["matrices"])

    collapsed = [r for r in rows2 if r[3] < 0.6]
    callout = ""
    if collapsed:
        names = "; ".join(f"<b>{t}</b> ({v:.2f})" for _, t, _, v, _ in collapsed)
        callout = (f'<div class="callout">The arrangement survives every change to how the '
                   f'program is <i>described</i> (0.94–1.00) but collapses for {names}. '
                   f'Both collapsing cases have comments present. Breaking the physics with '
                   f'<b>no</b> comments leaves the arrangement intact '
                   f'({dict((t, v) for _, t, _, v, _ in rows2)["NoComm_InValid"]:.2f}), '
                   f'so what moves the geometry is not the physics error on its own. '
                   f'<b>But see the curve below</b> &mdash; this split is already present at '
                   f'layer&nbsp;0, the embedding layer, where nothing but token identity '
                   f'exists. That makes it a token-level effect, not evidence about how the '
                   f'model represents physics.</div>')

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solver distances under perturbation — {D['model'].split('/')[-1]}</title>
<style>{CSS}</style></head><body><main>
<h1>What each perturbation does to the representation</h1>
<p class="sub">{D['model']} · layer {L} of {D['n_layers']-1} · pooling <code>{args.pool}</code>
· {len(solvers)} solvers</p>
<div class="key">
  <span><span class="sw" style="background:var(--desc)"></span>the description changed</span>
  <span><span class="sw" style="background:var(--phys)"></span>the physics changed</span>
</div>

<h2>1 · How far does each change move <em>one solver</em>?</h2>
<p class="q"><b>Every comparison here is one solver against another version of itself</b>,
never against a different solver.
<b>Orange</b> &mdash; the distance between a solver&rsquo;s <b>valid version and its invalid
twin</b>. The two carry the same comments and the same identifier names; the code differs
only by the physics edit.
<b>Blue</b> &mdash; the distance between a solver&rsquo;s unperturbed version and the same
solver re-described, with the implemented physics untouched.
A longer bar means the representation moved further; it does not by itself mean the model
understood what changed.</p>
{bars(rows1, vmax1)}
<p class="note">Formally <code>&#8214;h(solver, perturbed) &minus; h(solver, reference)&#8214; /
&#8214;h(solver, reference)&#8214;</code>, then the <b>mean</b> over the 32 solvers
(the per-class figures in section&nbsp;2 are medians, and are labelled as such).
Each row is labelled <code>reference &rarr; perturbed</code>, so both sides of every
comparison are on the chart. The reference for a
physics edit is that solver's valid twin <i>under the same surface condition</i>, so the
description is held fixed; the reference for a description edit is the unperturbed solver,
so the physics is held fixed. No comparison here crosses between different solvers.
On average, changing the physics moves the representation <b>{ratio:.2f}×</b> as far as
changing the description.</p>

<h2>2 · The valid &rarr; invalid measurement, solver by solver</h2>
<div class="defn">
<b>What is being measured.</b> Every solver exists twice: a <b>valid</b> version and an
<b>invalid</b> version whose physics has been deliberately broken. Under a fixed surface
condition the two share the same comments and the same identifier names &mdash; the physics
edit is the only difference between them. Each is one point in the model&rsquo;s
representation space, and we measure the straight-line distance between those two points,
divided by the size of the valid one so solvers of different magnitude are comparable:
<code class="eq">distance = &#8214;h(invalid) &minus; h(valid twin)&#8214; &divide; &#8214;h(valid twin)&#8214;</code>
0 would mean the model represents a broken solver identically to its working twin.
Larger means the physics edit registered more strongly. Every bar below is one solver&rsquo;s
own valid/invalid pair &mdash; no solver is ever compared to a different solver here.
</div>
{by_class(pert, solvers, classes)}
<p class="q" style="margin-top:22px">Median by PDE class:</p>
{class_summary(pert, classes)}
<p class="note">If the model tracked the physics you would expect this to vary by
PDE class &mdash; a broken diffusion term and a broken wave speed are not the same
intervention. Read across each row to see whether it does.</p>

<h2>3 · Does the arrangement <em>between</em> solvers survive?</h2>
<p class="q"><b>This one is between different solvers</b> &mdash; a different question from
panel 1. Measure how far every solver sits from every other, then ask whether that whole
pattern still holds once the perturbation is applied to all of them. 1.00 = the solvers
keep exactly their relative positions; 0 = the arrangement is unrecognisable.</p>
{bars(rows2, 1.0, decimals=3, arrow="vs")}
<p class="note">Each row correlates the 32&times;32 solver-to-solver distance matrix under
the second condition against the same matrix under the first. Read at layer {L}; the curve
below shows every layer.</p>
{callout}
<p class="q" style="margin-top:22px">The same measurement at <b>every</b> layer &mdash;
so nothing here depends on which layer was picked:</p>
{curves(curve_series, D["n_layers"])}
<p class="note"><b>Why layer {L}?</b> It is simply 40% of depth, chosen before any result
was seen, and used only as a reading point for the matrices and tables. It is not where
anything peaks. As the curve shows, the separation between the comment-bearing invalid
conditions and everything else holds flat across all {D["n_layers"]} layers &mdash;
including layer&nbsp;0, which is the raw token embedding.</p>

<h2>4 · The arrangement itself</h2>
<p class="q">The matrices behind panel 2 &mdash; every solver against every other, ordered and labelled by PDE class
(white lines mark the boundaries).
Darker = further apart. If the model organised solvers by physics you would see
four light blocks on the diagonal.</p>
<div class="grid">{cards}</div>
<p class="note">Within-class distance <b>{P['within_class_dist']:.3f}</b> vs
between-class <b>{P['between_class_dist']:.3f}</b> — solvers of the same PDE class are
only slightly closer to each other than to solvers of a different class.</p>

<h2>The numbers</h2>
<table><thead><tr><th>perturbation</th><th>what changed</th>
<th class="n">vs own reference<br>version (within-solver)</th>
<th class="n">between-solver<br>arrangement</th></tr></thead><tbody>
{"".join(f'<tr><td><span class="ep">{pert[c]["vs"]}</span>'
         f'<span class="arw"> &rarr; </span><span class="ep">{c}</span><br>'
         f'<span class="muted">{SHORT[c]}</span></td>'
         f'<td>{"physics" if pert[c]["kind"]=="physics" else "description"}</td>'
         f'<td class="n">{pert[c]["mean_by_layer"][L]:.3f}</td>'
         f'<td class="n">{gp[c]:+.3f}</td></tr>' for c in ORDER if c in pert)}
</tbody></table>
<p class="note">Column 3 is <b>within-solver</b>: a solver against its own reference version.
Column 4 is <b>between-solver</b>: the Spearman correlation of the 32&times;32
solver-to-solver distance matrix before versus after the perturbation. The two columns
answer different questions and are not comparable to each other.</p>
</main></body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    open(args.output, "w").write(html)
    print(f"Saved: {args.output} ({os.path.getsize(args.output)/1000:.0f} KB)")
    print(f"  physics/description displacement ratio: {ratio:.2f}")
    for frm, to, gloss, v, kind in rows2:
        print(f"  {v:+.3f}  {frm} vs {to}   ({gloss})")


if __name__ == "__main__":
    main()
