"""
Experiment 2 — World Model: interactive report.

Reads every world_model_delta_*.csv in --results_dir and writes one self-contained
HTML file (Plotly embedded, no CDN).

Usage (from repo root):
    python probe/viz_world_model.py \
        --results_dir probe/results/ \
        --output probe/results/world_model_report.html

All cross-model panels are plotted against RELATIVE DEPTH (layer / (L-1)), never
raw layer index — the roster spans 28 to 65 layers and an absolute-index overlay
would be meaningless.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
COND_ORDER = ["S_plain", "S_bare", "S_mislead", "S_obf"]
COND_LABEL = {
    "S_plain": "correct comments",
    "S_bare": "no comments",
    "S_mislead": "misleading comments",
    "S_obf": "obfuscated identifiers",
}


def to_html(fig, first: bool = False) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=first,
                       config={"displayModeBar": False})


def layout(fig, title: str, xt: str, yt: str, height: int = 460):
    fig.update_layout(
        title=title, xaxis_title=xt, yaxis_title=yt, height=height,
        template="plotly_white", hovermode="closest",
        margin=dict(l=70, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def pair_label(a: str, b: str) -> str:
    return f"{a.replace('S_', '')} ↔ {b.replace('S_', '')}"


def load_results(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "world_model_delta_*.csv")))
    if not paths:
        raise SystemExit(f"No world_model_delta_*.csv found in {results_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"Loaded {len(paths)} CSV(s), {len(df)} rows, "
          f"{df.model.nunique()} model(s)", flush=True)
    return df


# --- sections ---------------------------------------------------------------

def build_overview(df: pd.DataFrame) -> str:
    rows = []
    for (model, pool), g in df.groupby(["model", "pool"]):
        nr = g[g.stat == "delta_norm_ratio"]["value"]
        gaps = g[g.stat == "gap"].groupby("layer")["value"].mean()
        best = gaps.idxmax()
        n_layers = g.layer.nunique()
        l0 = gaps.get(0, float("nan"))
        rows.append({
            "model": model, "pool": pool, "layers": n_layers,
            "peak_rel_depth": f"{best / (n_layers - 1):.2f}" if n_layers > 1 else "—",
            "peak_gap": f"{gaps.max():+.4f}",
            "layer0_gap": f"{l0:+.4f}",
            "median_dh_over_h": f"{nr.median():.3g}",
            "min_dh_over_h": f"{nr.min():.3g}",
        })
    tbl = pd.DataFrame(rows)

    warn = ""
    bad = [r for r in rows if float(r["layer0_gap"]) > 0.05]
    if bad:
        warn += (
            "<div class='flag'><b>Layer-0 red flag.</b> "
            + ", ".join(f"{r['model']} / {r['pool']} ({r['layer0_gap']})" for r in bad)
            + " show a consistency gap at the embedding layer. Embeddings can only "
              "see token identity, so a gap there means the statistic is reading "
              "surface token overlap rather than a representation of the physics. "
              "Treat those curves as an artifact until explained.</div>"
        )
    tiny = [r for r in rows if float(r["median_dh_over_h"]) < 1e-3]
    if tiny:
        warn += (
            "<div class='flag'><b>Δh is very small relative to h</b> for "
            + ", ".join(f"{r['model']} / {r['pool']}" for r in tiny)
            + ". Confirm the hidden states were extracted in float32 — in float16 "
              "these cosines would be dominated by rounding error.</div>"
        )

    strata = (df[df.stat == "within_cos"]
              .groupby("stratum")["gt_sample"].nunique().to_dict())
    strat_html = ", ".join(f"<b>{k}</b>: {v} solvers" for k, v in sorted(strata.items()))

    return f"""
<section id="overview"><h2>Overview</h2>
<p class="lede">Δh(s, c, ℓ) = h(invalid) − h(valid) for solver <i>s</i> under surface
condition <i>c</i>. The question is whether the same physical edit moves the
representation the same way under different descriptions of the same program.</p>
<p><b>Surface conditions.</b> {", ".join(f"<code>{c}</code> = {COND_LABEL[c]}" for c in COND_ORDER)}</p>
<p><b>Strata.</b> {strat_html}
<br><span class="note">Pre-registered, not dropped: <code>cadence_leak</code> and
<code>grid_change</code> mark solvers whose invalid variant may be detectable from
something other than the physics. <code>clean_zero_len_delta</code> marks solvers
whose edit changes no code length at all.</span></p>
{tbl.to_html(index=False, classes="tbl", border=0)}
{warn}
<p class="note"><b>Reading this table.</b> <code>peak_gap</code> is the headline
number: mean within-solver cosine minus mean cross-solver cosine. A high
within-solver cosine on its own is not evidence — a single global "this code is
broken" direction produces one while leaving the gap at zero.</p>
</section>"""


def build_consistency(df: pd.DataFrame, first: bool = False) -> str:
    fig = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        col = PALETTE[i % len(PALETTE)]
        dash = "solid" if pool == "mean_pool" else "dot"
        gap = (g[g.stat == "gap"].groupby("rel_depth")["value"].mean().sort_index())
        null_hi = (g[g.stat == "gap_perm_null_hi"]
                   .groupby("rel_depth")["value"].mean().sort_index())
        fig.add_trace(go.Scatter(
            x=null_hi.index, y=null_hi.values, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill=None))
        fig.add_trace(go.Scatter(
            x=null_hi.index, y=np.zeros_like(null_hi.values), mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(150,150,150,0.15)",
            name="permutation null (95%)" if i == 0 else None,
            showlegend=(i == 0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=gap.index, y=gap.values, mode="lines+markers",
            name=f"{model.split('/')[-1]} · {pool}",
            line=dict(color=col, dash=dash, width=2), marker=dict(size=5),
            hovertemplate="rel depth %{x:.2f}<br>gap %{y:+.4f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    layout(fig, "Consistency gap vs. relative depth",
           "relative depth  (layer / (L−1))",
           "mean within-solver cos − mean cross-solver cos")

    fig2 = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        if pool != "mean_pool":
            continue
        col = PALETTE[i % len(PALETTE)]
        w = g[g.stat == "within_cos"].groupby("rel_depth")["value"].mean().sort_index()
        c = g[g.stat == "cross_cos"].groupby("rel_depth")["value"].mean().sort_index()
        short = model.split("/")[-1]
        fig2.add_trace(go.Scatter(x=w.index, y=w.values, mode="lines",
                                  name=f"{short} · within",
                                  line=dict(color=col, width=2)))
        fig2.add_trace(go.Scatter(x=c.index, y=c.values, mode="lines",
                                  name=f"{short} · cross",
                                  line=dict(color=col, width=2, dash="dash")))
    layout(fig2, "Within-solver vs. cross-solver cosine (mean_pool)",
           "relative depth", "cosine similarity")

    return f"""
<section id="consistency"><h2>Consistency</h2>
<p class="lede">The gap, not the raw cosine, is the result. The shaded band is the
95th percentile of a permutation null in which solver identity is shuffled between
conditions; a curve inside the band is indistinguishable from no solver-specific
structure.</p>
{to_html(fig, first=first)}
<p class="note">The second panel shows why. If the solid (within) and dashed (cross)
curves rise together, the model has one generic defect direction rather than a
representation of <i>this</i> solver's physical change.</p>
{to_html(fig2)}
</section>"""


def build_identification(df: pd.DataFrame) -> str:
    fig = go.Figure()
    n_solvers = df[df.stat == "within_cos"]["gt_sample"].nunique()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        col = PALETTE[i % len(PALETTE)]
        dash = "solid" if pool == "mean_pool" else "dot"
        m = g[g.stat == "match_acc"].groupby("rel_depth")["value"].mean().sort_index()
        fig.add_trace(go.Scatter(
            x=m.index, y=m.values, mode="lines+markers",
            name=f"{model.split('/')[-1]} · {pool}",
            line=dict(color=col, dash=dash, width=2), marker=dict(size=5),
            hovertemplate="rel depth %{x:.2f}<br>top-1 %{y:.3f}<extra></extra>"))
    if n_solvers:
        fig.add_hline(y=1.0 / n_solvers, line=dict(color="#E45756", dash="dash"),
                      annotation_text=f"chance = 1/{n_solvers}")
    layout(fig, "Defect identification across descriptions (top-1)",
           "relative depth", "top-1 accuracy")

    # condition-pair heatmap at each model's peak layer
    heat = []
    for (model, pool), g in df.groupby(["model", "pool"]):
        if pool != "mean_pool":
            continue
        gaps = g[g.stat == "gap"].groupby("layer")["value"].mean()
        best = gaps.idxmax()
        sub = g[(g.stat == "within_cos") & (g.layer == best)]
        for (a, b), gg in sub.groupby(["condition_a", "condition_b"]):
            heat.append({"model": model.split("/")[-1],
                         "pair": pair_label(a, b), "value": gg.value.mean()})
    fig2 = go.Figure()
    if heat:
        hd = pd.DataFrame(heat).pivot_table(index="model", columns="pair",
                                            values="value")
        fig2.add_trace(go.Heatmap(
            z=hd.values, x=list(hd.columns), y=list(hd.index),
            colorscale="RdBu", zmid=0, colorbar=dict(title="cos"),
            hovertemplate="%{y}<br>%{x}<br>cos %{z:.3f}<extra></extra>"))
    layout(fig2, "Within-solver cosine by condition pair (peak layer, mean_pool)",
           "condition pair", "model", height=380)

    return f"""
<section id="identification"><h2>Defect identification</h2>
<p class="lede">Given a solver's defect under one description, can we pick that same
solver's defect out of all candidates under a different description? Chance is
1/S, so this is a much sharper test than any 0.5-chance statistic — and it can only
succeed if the defect representation is both solver-specific and stable across
descriptions.</p>
{to_html(fig)}
<p class="note">Expect <code>plain ↔ bare</code> highest (the code is byte-identical,
only comments removed) and <code>plain ↔ obf</code> lowest. Surviving consistency
under identifier obfuscation is the strongest evidence in this experiment.</p>
{to_html(fig2)}
</section>"""


def build_magnitude(df: pd.DataFrame) -> str:
    fig = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        if pool != "mean_pool":
            continue
        col = PALETTE[i % len(PALETTE)]
        ph = g[g.stat == "physics_norm"].groupby("rel_depth")["value"].mean().sort_index()
        su = g[g.stat == "surface_norm"].groupby("rel_depth")["value"].mean().sort_index()
        common = ph.index.intersection(su.index)
        ratio = ph.loc[common] / su.loc[common].replace(0, np.nan)
        fig.add_trace(go.Scatter(
            x=ratio.index, y=ratio.values, mode="lines+markers",
            name=model.split("/")[-1], line=dict(color=col, width=2),
            marker=dict(size=5),
            hovertemplate="rel depth %{x:.2f}<br>ratio %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="#E45756", dash="dash"),
                  annotation_text="physics move = surface move")
    layout(fig, "‖Δ physics‖ / ‖Δ surface‖ vs. relative depth",
           "relative depth", "ratio")

    fig2 = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        if pool != "mean_pool":
            continue
        nr = g[g.stat == "delta_norm_ratio"].groupby("rel_depth")["value"].median()
        fig2.add_trace(go.Scatter(
            x=nr.index, y=nr.values, mode="lines",
            name=model.split("/")[-1],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig2.add_hline(y=1e-3, line=dict(color="#E45756", dash="dash"),
                   annotation_text="float16 danger zone")
    fig2.update_yaxes(type="log")
    layout(fig2, "‖Δh‖ / ‖h‖ (median across solvers)", "relative depth",
           "ratio (log)", height=380)

    return f"""
<section id="magnitude"><h2>Magnitude</h2>
<p class="lede">A ratio below 1 means changing how the program is <i>described</i>
moves the representation more than changing what physics it <i>implements</i>.
That is a result about the model regardless of which way it comes out, and it is
the context every consistency number should be read against.</p>
{to_html(fig)}
<p class="note">The second panel is the precision audit. Δh is a difference between
two forward passes on nearly identical inputs; if ‖Δh‖/‖h‖ approaches 1e-3, float16
storage would leave only one or two significant digits and the cosines would be
measuring rounding error.</p>
{to_html(fig2)}
</section>"""


CSS = """
:root { --fg:#1a1a1a; --muted:#666; --bg:#fff; --line:#e5e5e5; --accent:#4C78A8; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       color:var(--fg); background:var(--bg); }
nav { position:sticky; top:0; z-index:100; background:rgba(255,255,255,.95);
      backdrop-filter:blur(8px); border-bottom:1px solid var(--line); padding:12px 32px; }
nav a { margin-right:22px; text-decoration:none; color:var(--muted); font-weight:500;
        font-size:14px; }
nav a:hover { color:var(--accent); }
main { max-width:1100px; margin:0 auto; padding:0 32px 80px; }
h1 { font-size:28px; margin:36px 0 6px; }
h2 { font-size:21px; margin:52px 0 10px; padding-bottom:8px;
     border-bottom:1px solid var(--line); }
.sub { color:var(--muted); margin:0 0 8px; }
.lede { font-size:15px; color:#333; margin:0 0 18px; }
.note { font-size:13.5px; color:var(--muted); margin:10px 0 24px; }
.flag { background:#fff4f4; border-left:3px solid #E45756; padding:12px 16px;
        margin:18px 0; font-size:14px; }
table.tbl { border-collapse:collapse; width:100%; font-size:13.5px; margin:16px 0; }
table.tbl th { text-align:left; border-bottom:2px solid var(--line); padding:8px 10px;
               color:var(--muted); font-weight:600; }
table.tbl td { border-bottom:1px solid var(--line); padding:8px 10px; }
code { background:#f4f4f6; padding:1px 5px; border-radius:3px; font-size:13px; }
"""



# --- Part II sections -------------------------------------------------------

def _load_optional(results_dir, pattern):
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def build_physics_vs_code(df) -> str:
    """§14 — does pde_class survive partialling out code similarity?"""
    if df is None or df.empty:
        return ""
    fig = go.Figure()
    for i, ((model, pool, cond), g) in enumerate(
            df.groupby(["model", "pool", "condition"])):
        col = PALETTE[i % len(PALETTE)]
        dash = "solid" if cond == "NoComm_CorrVar" else "dot"
        inc = g[g.stat == "pde_class_incremental_r2"].groupby(
            "rel_depth")["value"].mean().sort_index()
        fig.add_trace(go.Scatter(
            x=inc.index, y=inc.values, mode="lines+markers",
            name=f"{model.split('/')[-1]} · {cond}",
            line=dict(color=col, dash=dash, width=2), marker=dict(size=5)))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    layout(fig, "pde_class incremental R² after partialling out code similarity",
           "relative depth", "ΔR² (physics beyond code)")

    fig2 = go.Figure()
    for i, ((model, pool, cond), g) in enumerate(
            df.groupby(["model", "pool", "condition"])):
        if cond != "Comm_Valid":
            continue
        col = PALETTE[i % len(PALETTE)]
        for stat, dash, lbl in [("pde_class_incremental_r2", "solid", "physics"),
                                ("num_method_incremental_r2", "dash", "algorithm"),
                                ("code_only_r2", "dot", "code similarity alone")]:
            v = g[g.stat == stat].groupby("rel_depth")["value"].mean().sort_index()
            if len(v):
                fig2.add_trace(go.Scatter(
                    x=v.index, y=v.values, mode="lines",
                    name=f"{model.split('/')[-1]} · {lbl}",
                    line=dict(color=col, dash=dash, width=2)))
    layout(fig2, "Physics vs. algorithm vs. code similarity (Comm_Valid)",
           "relative depth", "R² / ΔR²")

    # Evaluated PER (model, condition), never as a global max. A single
    # physics-positive model must not suppress the warning for the other three.
    inc = df[df.stat == "pde_class_incremental_r2"]
    negatives = [f"{m.split('/')[-1]} / {c}"
                 for (m, c), g in inc.groupby(["model", "condition"])
                 if g["value"].max() < 0.01]
    verdict = ""
    if negatives:
        verdict = (
            "<div class='flag'><b>Physics-negative for "
            + ", ".join(negatives) +
            ".</b> pde_class adds essentially nothing beyond code similarity at any "
            "layer for these. Report as a negative result, not as a null to be "
            "explained away.</div>")

    return f"""
<section id="physicsvscode"><h2>Physics vs. code</h2>
<p class="lede">The Δh analysis varies how a fixed program is <i>described</i>, so it
cannot separate "represents the physics" from "represents the program". This does.
Representational distance is regressed on <code>same_pde_class</code> against
code-similarity nuisance regressors (token Jaccard, identifier-stripped AST n-grams,
length). Only the <b>incremental</b> R² is a physics claim.</p>
{to_html(fig)}
<p class="note">Significance is a Mantel permutation over <b>solver identity</b>, not
over the 496 pairwise distances — those are not independent observations, and
permuting them directly would inflate every p-value on this page.
The <code>NoComm_CorrVar</code> curves are the stronger evidence: comments removed
and identifiers obfuscated.</p>
{to_html(fig2)}
{verdict}
</section>"""


def build_cross_modal(df) -> str:
    """§15 — equation ↔ code ↔ trajectory retrieval."""
    if df is None or df.empty:
        return ""
    sub = df[(df.stat == "retrieval_top1") & (df.scope == "within_class")]
    fig = go.Figure()
    for i, ((model, mod, cond), g) in enumerate(
            sub.groupby(["model", "modality", "code_condition"])):
        col = PALETTE[i % len(PALETTE)]
        dash = "solid" if cond == "NoComm_CorrVar" else "dot"
        v = g.groupby("rel_depth")["value"].mean().sort_index()
        fig.add_trace(go.Scatter(
            x=v.index, y=v.values, mode="lines+markers",
            name=f"{mod} → {cond}", line=dict(color=col, dash=dash, width=2),
            marker=dict(size=5)))
    ch = df[(df.stat == "retrieval_chance_top1") & (df.scope == "within_class")]
    if len(ch):
        fig.add_hline(y=float(ch["value"].mean()),
                      line=dict(color="#E45756", dash="dash"),
                      annotation_text="within-class chance")
    lb = df[(df.stat == "lexical_baseline_top1") & (df.scope == "within_class")]
    if len(lb):
        fig.add_hline(y=float(lb["value"].mean()),
                      line=dict(color="#F58518", dash="dot"),
                      annotation_text="lexical-only baseline")
    layout(fig, "Cross-modal retrieval, within pde_class (top-1)",
           "relative depth", "top-1 accuracy")

    fig2 = go.Figure()
    for i, ((model, mod, cond), g) in enumerate(
            df[df.stat == "retrieval_top1_lexresid"].groupby(
                ["model", "modality", "code_condition"])):
        if g.empty:
            continue
        v = g[g.scope == "within_class"].groupby("rel_depth")["value"].mean().sort_index()
        if len(v):
            fig2.add_trace(go.Scatter(
                x=v.index, y=v.values, mode="lines",
                name=f"{mod} → {cond}",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    layout(fig2, "Same retrieval after removing lexical overlap",
           "relative depth", "top-1 accuracy", height=380)

    return f"""
<section id="cross_modal_consistency"><h2>Cross-modal alignment</h2>
<p class="lede">Three representations of one physical system — the symbolic
equation, the solver code, and the executed trajectory — two of which contain no
code at all. If the model represents the physics, a representation built from one
modality should locate the same system in another.</p>
{to_html(fig)}
<p class="note"><b>Why within-class.</b> There are only 4 PDE classes, so global
retrieval (chance 1/32) can be solved by 4-way category matching. Scoring within
<code>pde_class</code> (chance 1/8) forces instance-level physics matching. The
headline pairing is equation → <code>NoComm_CorrVar</code>: no comments, obfuscated
identifiers, so almost no lexical overlap with ∂u/∂t = ν ∂²u/∂x².</p>
{to_html(fig2)}
<p class="note">A result that collapses toward chance in this second panel was a
lexical result, not a physics result.</p>
</section>"""


def build_geometry(df) -> str:
    """§16 — representation geometry."""
    if df is None or df.empty:
        return ""
    fig = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        if pool != "mean_pool":
            continue
        for man, dash in [("physics", "solid"), ("code", "dash")]:
            v = g[(g.stat == "participation_ratio") & (g.manifold == man)]
            v = v.groupby("rel_depth")["value"].mean().sort_index()
            if len(v):
                fig.add_trace(go.Scatter(
                    x=v.index, y=v.values, mode="lines",
                    name=f"{model.split('/')[-1]} · {man}",
                    line=dict(color=PALETTE[i % len(PALETTE)], dash=dash, width=2)))
    layout(fig, "Effective dimensionality: physics vs. code manifold",
           "relative depth", "participation ratio")

    fig2 = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        col = PALETTE[i % len(PALETTE)]
        for stat, dash, lbl in [("pde_class_linear_acc", "solid", "linear"),
                                ("pde_class_knn_acc", "dash", "kNN")]:
            v = g[g.stat == stat].groupby("rel_depth")["value"].mean().sort_index()
            if len(v):
                fig2.add_trace(go.Scatter(
                    x=v.index, y=v.values, mode="lines",
                    name=f"{model.split('/')[-1]} · {lbl}",
                    line=dict(color=col, dash=dash, width=2)))
    layout(fig2, "pde_class decodability: linear vs. kNN", "relative depth",
           "LOGO accuracy")

    fig3 = go.Figure()
    for i, ((model, pool), g) in enumerate(df.groupby(["model", "pool"])):
        v = g[g.stat == "validity_cross_class_acc"]
        v = v.groupby("rel_depth")["value"].mean().sort_index()
        if len(v):
            fig3.add_trace(go.Scatter(
                x=v.index, y=v.values, mode="lines+markers",
                name=model.split("/")[-1],
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                marker=dict(size=4)))
    fig3.add_hline(y=0.5, line=dict(color="#E45756", dash="dash"),
                   annotation_text="chance")
    layout(fig3, "Validity direction: train on 3 PDE classes, test on the 4th",
           "relative depth", "held-out class accuracy", height=380)

    gap = df[df.stat == "pde_class_curvature_gap"]["value"]
    note = ""
    if len(gap) and gap.max() > 0.15:
        note = ("<div class='flag'><b>pde_class is substantially more decodable "
                "non-linearly</b> (kNN − linear = "
                f"{gap.max():+.3f}). The structure is present but curved, so every "
                "linear result elsewhere on this page understates it.</div>")

    an = df[df.stat == "mean_pairwise_cos_raw"]["value"]
    aniso = (f"<p class='note'>Anisotropy: median raw pairwise cosine "
             f"{an.median():.3f}. Unlike the Δh analysis, nothing in this section "
             f"differences, so raw similarities are inflated by roughly this much "
             f"— the mean-centred variants are in the CSV.</p>") if len(an) else ""

    return f"""
<section id="geometry"><h2>Geometry</h2>
<p class="lede">Four observations chosen because each can come out negative in an
informative way.</p>
{to_html(fig)}
{to_html(fig2)}
{note}
{to_html(fig3)}
<p class="note">Cross-class transfer of the validity direction is a strong claim
about a general representation of physical correctness. Leak solvers are excluded
here — their invalid variant may be detectable from a sampling-cadence change,
which would let them carry a result that is not about physics at all.</p>
{aniso}
</section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="probe/results/")
    ap.add_argument("--output", default="probe/results/world_model_report.html")
    args = ap.parse_args()

    df = load_results(args.results_dir)
    pvc = _load_optional(args.results_dir, "physics_vs_code_*.csv")
    xm = _load_optional(args.results_dir, "cross_modal_*.csv")
    geo = _load_optional(args.results_dir, "geometry_*.csv")
    for name, d in [("physics_vs_code", pvc), ("cross_modal", xm), ("geometry", geo)]:
        print(f"  {name}: {'-' if d is None else str(len(d)) + ' rows'}", flush=True)

    body = "".join([
        build_overview(df),
        build_consistency(df, first=True),   # embeds plotly.js once
        build_identification(df),
        build_magnitude(df),
        build_physics_vs_code(pvc),
        build_cross_modal(xm),
        build_geometry(geo),
    ])

    models = ", ".join(sorted(df.model.unique()))
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment 2 — World Model</title><style>{CSS}</style></head><body>
<nav>
  <a href="#overview">Overview</a>
  <a href="#consistency">Consistency</a>
  <a href="#identification">Identification</a>
  <a href="#magnitude">Magnitude</a>
  {'<a href="#physicsvscode">Physics vs code</a>' if pvc is not None else ''}
  {'<a href="#crossmodal">Cross-modal</a>' if xm is not None else ''}
  {'<a href="#geometry">Geometry</a>' if geo is not None else ''}
</nav>
<main>
<h1>Experiment 2 — World Model</h1>
<p class="sub">Are physical defects represented consistently across descriptions?</p>
<p class="sub">Models: {models}</p>
{body}
</main></body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
