"""
pde_dual_report.py — one HTML report covering both live PDE experiments.

Experiment 1 (free generation): given a solver, name the PDE, the numerical method,
the dominant behaviour, and whether it is physically valid.

Experiment 2 Part III (cross-modal consistency): four representations of one system,
one corrupted; detect the disagreement and localize it.

They share the same 32 base solvers, so putting them in one document is the point:
Part I asks whether a model can read a solver, Part III asks whether it can tell when
independent representations of that solver disagree.

Both halves degrade gracefully. A panel whose data has not landed yet renders as a
placeholder saying exactly what is missing and which job produces it, rather than
being silently absent -- an empty section is a fact about the run, not a formatting
problem.

Usage:
    python viz/pde_dual_report.py \
        --freegen results/pde_llm_eval_jul28.csv \
        --xmodal_dir results/xmodal \
        --xmodal_summary results/xmodal_summary.json \
        --out viz/pde_dual_report.html
"""
import argparse
import csv
import glob
import math
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# The scoring helpers are imported rather than reimplemented: validity in
# Experiment 1 and detection in Part III are the same kind of asymmetric yes/no
# judgement, and a second local copy of either rule would let the two experiments
# drift apart while still looking comparable in one document.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from freegen.parse_score import valid_intent                       # noqa: E402
from freegen.parse_score import classify_valid_confidence_2x2      # noqa: E402
from crossmodal.eval.parse_consistency import dprime               # noqa: E402
# Same reasoning as the scoring helpers above: imported, not reimplemented. This
# file was written when Experiment 1 was k=1 and every row was one item. The
# roster now runs k=3 to match the consistency arms, so a raw frame holds THREE
# correlated rows per item and every n in this report would be 3x too large --
# point estimates unchanged, every interval about 42% too narrow, in the direction
# that makes a result look real. pool_draws() is a no-op on k=1 input, so this is
# safe for the older CSVs too.
from freegen.report import pool_draws                              # noqa: E402

DARK = "#0d0f18"
PANEL = "#12141e"
GRID = "#1e2130"
FG = "#e0e0e0"
ACCENT = ["#7eb8ff", "#f2a97e", "#8fd694", "#d98fd6", "#e6d17e", "#7ed6d0", "#e67e8f"]

LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(color=FG, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
              size=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=30, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    height=460,
)




def fig_html(fig, first=False, **overrides):
    """Render one figure. plotly.js is NOT embedded here -- the template head owns it.

    Two earlier designs both failed the same way. Threading `first` in from the caller
    meant deleting whichever panel happened to be first silently dropped the library
    and left empty divs behind. Moving the flag in here ("whoever renders first wins")
    fixed deletion but not reordering: panels are built in one order and then sorted
    into lead/appendix in another, so the call that embedded the bundle could easily
    land near the end of the document while ten figures above it had already run
    `Plotly.newPlot` against an undefined global.

    So the bundle is emitted once in <head>, before any panel markup exists, and every
    figure here is a consumer. Build order and document order can now diverge freely,
    which is the whole point of grouping panels by argument rather than by producer.
    `first` is accepted and ignored so existing call sites keep working.
    """
    # LAYOUT is the house style; `overrides` is how one figure asks for more head-
    # room or a taller canvas. Applying LAYOUT unconditionally used to clobber both.
    fig.update_layout(**{**LAYOUT, **overrides})
    # LAYOUT's xaxis/yaxis entries only reach the FIRST axis pair, so a subplot
    # figure would render its second panel on default (white) gridlines.
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return pio.to_html(fig, include_plotlyjs=False,
                       full_html=False, config={"displayModeBar": False})


def placeholder(what, produced_by):
    return (f'<div class="pending"><strong>Not available yet.</strong><br>{what}<br>'
            f'<span class="src">Produced by: {produced_by}</span></div>')



COND_DECODER = [
    ("A0", "&mdash;", "control: all four views agree",
     "The only condition with no corruption, and therefore the only source of the "
     "false-alarm rate. Without it &quot;always say they disagree&quot; would score 7/8."),
    ("X_C", "code", "the solver is swapped for the invalid one",
     "The dataset's delivered <code>_wrong</code> solver for that system. Often a "
     "one-line change &mdash; a flipped flux direction, a wrong power of dx &mdash; in "
     "an otherwise identical, correct-looking program."),
    ("X_M", "math", "the governing equation is swapped",
     "The equation from that system's invalid counterpart, so the stated PDE no longer "
     "matches what the other three views describe."),
    ("X_D", "description", "the prose description is swapped",
     "The natural-language description of the invalid counterpart."),
    ("X_T_rand", "trajectory", "random numbers, correct shape",
     "The grossest rung: noise matched to the real trajectory's shape only. Anything "
     "that reads the numbers at all should catch this."),
    ("X_T_shuf", "trajectory", "the real values, permuted",
     "The sharpest control in the design. The value multiset is exactly the valid "
     "trajectory's &mdash; only the arrangement is destroyed &mdash; so no statistic "
     "blind to position can pass it."),
    ("X_T_swap", "trajectory", "the dataset's delivered wrong trajectory",
     "Used exactly as delivered, never regenerated or reshaped."),
    ("X_T_exec", "trajectory", "what the invalid solver actually printed",
     "The invalid solver re-executed on cpu_short; many diverge to NaN, which is what "
     "makes them invalid. The most realistic rung and the easiest to catch."),
]


def condition_decoder():
    """The eight condition codes, spelled out.

    The codes are compact enough to fit on an axis and opaque enough to be useless
    without this table, so it sits next to the first panel that uses them.
    """
    body = "".join(
        f'<tr><td><code class="ccode">{code}</code></td><td>{view}</td>'
        f'<td>{what}</td><td class="cwhy">{why}</td></tr>'
        for code, view, what, why in COND_DECODER)
    return (f'<table class="tbl decoder"><tr><th>condition</th><th>corrupted view</th>'
            f'<th>what is swapped in</th><th>why it is in the ladder</th></tr>'
            f'{body}</table>')


# ── Figure 1 style: score by condition, and the validity confidence breakdown ──
# Deliberately shaped like Figure 1 of the writeup so the two can be read together.
# Conditions run in the paper's order: the four physically-valid variants first,
# then the four invalid ones, with lexical perturbation increasing inside each half.
COND_ORDER = [
    ("Comm_Valid",              "Clean+Comment",       True),
    ("NoComm_Valid",            "Clean, No Comment",   True),
    ("CorrComm",                "Corrupt Comment",     True),
    ("NoComm_CorrVar",          "Obfuscated Vars",     True),
    ("Comm_InValid",            "Invalid+Comment",     False),
    ("NoComm_InValid",          "Invalid, No Comment", False),
    ("CorrComm_Invalid",        "CorrComment+Invalid", False),
    ("NoComm_CorrVar_InValid",  "Obfuscated+Invalid",     False),
]
CONF_ORDER = [("Confident Yes", "#4fa96a"), ("Uncertain Yes", "#e8c35a"),
              ("Hedged", "#e2913f"), ("Confident No", "#c4574d")]


def _cluster_ci(df, col, n_boot=2000, seed=20260820):
    """Mean and 95% interval, resampling the 32 SYSTEMS rather than the rows.

    Eight conditions and eleven models share each base solver, so rows within a
    system are not independent. Resampling rows would shrink these intervals by
    roughly the square root of that clustering and make every difference look real.
    Same estimator as the cross-modal half, on purpose.
    """
    vals = df[col].dropna()
    if vals.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby("gt_sample")]
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return vals.mean(), float("nan"), float("nan")
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boots = np.array([np.concatenate([groups[j] for j in row]).mean() for row in idx])
    return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fig1_panels(df):
    """The two halves of writeup Figure 1, rebuilt on the jul28 rerun."""
    if df is None or df.empty:
        return []
    present = [(k, lab, v) for k, lab, v in COND_ORDER if (df.mod_type == k).any()]
    labels = [lab for _, lab, _ in present]
    panels = []

    # ── left half: mean task score across models, by condition ────────────────
    metrics = [("pde_match", "PDE type", ACCENT[0]),
               ("method_recall", "Method recall", ACCENT[1]),
               ("behavior_recall", "Behaviour recall", ACCENT[2]),
               ("valid_match", "Validity", ACCENT[3])]
    fig = go.Figure()
    for col, name, colour in metrics:
        if col not in df:
            continue
        mid, lo, hi = [], [], []
        for key, _, _ in present:
            m, l, h = _cluster_ci(df[df.mod_type == key], col)
            mid.append(100 * m); lo.append(100 * (m - l)); hi.append(100 * (h - m))
        fig.add_scatter(x=labels, y=mid, name=name, mode="lines+markers",
                        line=dict(color=colour, width=2), marker=dict(size=8),
                        error_y=dict(type="data", array=hi, arrayminus=lo,
                                     color=colour, thickness=1.2, width=4))
    # The bands are the story: everything left of the first band is clean code,
    # the first band removes lexical cues while leaving the computation intact,
    # and the second band is where the physics is actually broken.
    # Both band captions used to be anchored "top left" of their own rect. The first
    # caption is far wider than its two-category band, so it ran straight over the
    # second one. Captions are centred on their band and parked in headroom added to
    # the y-range, which keeps them clear of the data as well as of each other.
    for x0, x1, text, colour in [
            (1.5, 3.5, "lexical cues removed \u2014 code still valid", "#c4574d"),
            (3.5, 7.5, "physically invalid code", "#8f6fd0")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=colour, opacity=0.09, line_width=0)
        fig.add_annotation(x=(x0 + x1) / 2, y=110, text=text, showarrow=False,
                           xanchor="center", font=dict(color=colour, size=10))
    fig.add_hline(y=50, line=dict(color="#5a6580", dash="dot"),
                  annotation_text="chance on validity", annotation_position="bottom right",
                  annotation_font=dict(color="#7b88a4", size=10))
    fig.update_layout(yaxis_title="score (%)", yaxis_range=[0, 118],
                      xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.38))
    panels.append((
        "Score by condition &mdash; averaged over all models",
        "<b>Yes, this is pooled: every row from every model goes into each point.</b> "
        "It shows the shape of the effect, not a leaderboard &mdash; for per-model "
        "numbers see &quot;Results at a glance&quot; and &quot;Degradation across the "
        "eight conditions&quot;. Four scores over the eight perturbations, in the writeup's "
        "Figure 1 ordering: the four physically-valid variants, then the four "
        "invalid ones. Intervals are 95% bootstrap resampling the <b>32 base "
        "solvers</b>, not the rows &mdash; eight conditions share each solver, so "
        "row-level resampling would understate them. Read the validity line "
        "against the dotted chance line, and the other three against each other.",
        fig_html(fig, height=560, margin=dict(l=65, r=30, t=30, b=170))))

    # ── right half: validity judgements split by confidence ───────────────────
    if "valid_conf" in df:
        fig = go.Figure()
        for bucket, colour in CONF_ORDER:
            pct = []
            for key, _, _ in present:
                sub = df[df.mod_type == key]
                pct.append(100 * (sub["valid_conf"] == bucket).mean() if len(sub) else 0)
            fig.add_bar(x=labels, y=pct, name=bucket, marker_color=colour)
        for x0, x1, text, colour in [
                (-0.5, 3.5, "VALID CODE \u2014 ground truth: yes", "#4fa96a"),
                (3.5, 7.5, "INVALID CODE \u2014 ground truth: no", "#c4574d")]:
            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=colour, opacity=0.05)
            fig.add_annotation(x=(x0 + x1) / 2, y=112, text=text, showarrow=False,
                               xanchor="center", font=dict(color=colour, size=11))
        fig.update_layout(barmode="stack", yaxis_title="% of predictions",
                          yaxis_range=[0, 122], xaxis_tickangle=-30,
                          legend=dict(orientation="h", y=-0.38))
        panels.append((
            "Validity judgements, split by confidence &mdash; averaged over all models",
            "<b>Aggregated across every model</b>, exactly like the panel before it: "
            f"each bar is all ~{len(df) // max(len(present), 1)} responses given under "
            f"that condition &mdash; {df['model'].nunique()} models &times; 32 solvers "
            "&mdash; split into the four confidence buckets. It is a distribution "
            "over responses, not a per-model score, so a bar being 40% green means 40% "
            "of all responses in that condition were a confident &quot;valid&quot; "
            "&mdash; it does not mean any particular model was 40% right.<br><br>"
            "The same judgements as the validity line opposite, but unstacked into "
            "how firmly they were held. Green is the answer &quot;valid&quot; stated "
            "plainly; red is &quot;invalid&quot; stated plainly; the two middle bands "
            "are hedges. On the right-hand half the correct answer is <b>no</b>, so "
            "green there is a confident error. Hedging is measurable only because the "
            "prompt never forces a yes/no &mdash; that is the reason it is left open.",
            fig_html(fig, height=560, margin=dict(l=65, r=30, t=30, b=170))))
    return panels


def validity_dprime_panel(df):
    """Can a model tell valid code from invalid? Asked without a signal-detection stat.

    An earlier version of this panel led with d' and a bar chart of it. d' is the right
    quantity, but naming a statistic is not the same as answering the question, and a
    bar whose units nobody can interpret is not evidence anyone can check. The measured
    thing is a gap between two rates, so the panel now shows that gap directly, with an
    interval, against zero. Same content, no vocabulary to learn.

    The interval resamples the 32 base solvers rather than the rows: each solver appears
    in both a valid and an invalid variant, so the two rates are paired at the solver
    level and row-level resampling would ignore it.
    """
    if df is None or df.empty or "parsed_valid" not in df:
        return []
    d = df.copy()
    d["lean"] = d["parsed_valid"].map(valid_intent)
    u = d[d.lean.notna()].copy()
    if u.empty:
        return []
    u["lean"] = u["lean"].astype(bool)
    # NB: `.gt` is a DataFrame method, so the truth column must not be called that.
    u["truth"] = u["gt_valid"].astype(bool)

    rng = np.random.default_rng(20260820)
    recs = []
    for m, s in u.groupby("model"):
        inv, val = s[~s.truth], s[s.truth]
        if not len(inv) or not len(val):
            continue
        hit, fa = (~inv.lean).mean(), (~val.lean).mean()
        groups = [g for _, g in s.groupby("gt_sample")]
        boots = []
        for _ in range(2000):
            b = pd.concat([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            bi, bv = b[~b.truth], b[b.truth]
            if len(bi) and len(bv):
                boots.append((~bi.lean).mean() - (~bv.lean).mean())
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
        recs.append({"model": m.split("/")[-1], "hit": hit, "fa": fa, "gap": hit - fa,
                     "lo": lo, "hi": hi,
                     "acc": d[d.model == m]["valid_match"].mean(),
                     "abstain": 1 - len(s) / len(d[d.model == m])})
    if not recs:
        return []
    recs.sort(key=lambda r: r["gap"])
    names = [r["model"] for r in recs]

    fig = go.Figure()
    fig.add_bar(x=[100 * r["gap"] for r in recs], y=names, orientation="h",
                marker_color=["#4fa96a" if r["lo"] > 0 else "#6b7a99" for r in recs],
                error_x=dict(type="data", symmetric=False,
                             array=[100 * (r["hi"] - r["gap"]) for r in recs],
                             arrayminus=[100 * (r["gap"] - r["lo"]) for r in recs],
                             color="#8fa6c9", thickness=1.3, width=5),
                showlegend=False)
    fig.add_vline(x=0, line=dict(color="#e67e8f", width=1.5, dash="dash"),
                  annotation_text="no ability to tell them apart",
                  annotation_font=dict(color="#e67e8f", size=10))
    fig.update_layout(xaxis_title="discrimination gap (percentage points)",
                      yaxis_title="")

    # Everything on this panel is in percentage points: the bars, the table, and the
    # prose. It used to mix them -- bars at 22.8, the same number in the table as
    # 0.228, the prose calling it +22.8pp -- so cross-referencing the chart against
    # the row underneath it produced three different numbers for one quantity.
    body = "".join(
        f"<tr><td>{r['model']}</td>"
        f"<td class='num'>{100 * r['hit']:.1f}</td>"
        f"<td class='num'>{100 * r['fa']:.1f}</td>"
        f"<td class='num'>{'<b>' if r['lo'] > 0 else ''}{100 * r['gap']:+.1f}"
        f"{'</b>' if r['lo'] > 0 else ''}</td>"
        f"<td class='num'>[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]</td>"
        f"<td class='num'>{100 * r['abstain']:.0f}</td>"
        f"<td class='num'>{r['acc']:.3f}</td></tr>"
        for r in sorted(recs, key=lambda r: -r["gap"]))
    n_sig = sum(1 for r in recs if r["lo"] > 0)

    # The commentary names models, so it reads them off `recs` rather than repeating
    # them as literals. Hardcoded names survive a rerun that reorders them and quietly
    # start describing the wrong row.
    clear = sorted((r for r in recs if r["lo"] > 0), key=lambda r: -r["gap"])
    dull = [r for r in recs if r["lo"] <= 0]
    lead_txt = ", ".join(f"{r['model']} ({100 * r['gap']:+.1f}pp)" for r in clear) \
        or "no model"
    # The clearest "policy, not perception" case: says invalid most often, yet the
    # interval still includes zero because it says it to valid code just as often.
    pol = max(dull, key=lambda r: r["hit"], default=None)
    pol_txt = ("" if pol is None else
               f"<b>Compare the last two columns.</b> {pol['model']} calls "
               f"{100 * pol['hit']:.0f}% of invalid code invalid, which sounds "
               f"excellent until you see it says the same of {100 * pol['fa']:.0f}% "
               f"of valid code. That is a policy, not a perception.<br><br>")
    # Raw accuracy decoupling: the best-scoring model that cannot discriminate,
    # against the weakest-scoring one that can.
    best_dull = max(dull, key=lambda r: r["acc"], default=None)
    worst_clear = min(clear, key=lambda r: r["acc"], default=None)
    acc_txt = ("" if not (best_dull and worst_clear
                          and best_dull["acc"] >= worst_clear["acc"]) else
               f"Raw accuracy tracks none of this: {best_dull['model']} scores "
               f"{best_dull['acc']:.3f}, above {worst_clear['model']}'s "
               f"{worst_clear['acc']:.3f}, on a gap of {100 * best_dull['gap']:+.1f}pp "
               f"that does not clear zero.<br><br>")
    return [(
        "Can a model tell valid code from invalid?",
        "<b>The gap is the whole panel.</b> Take how often a model calls <i>invalid</i> "
        "code invalid. Subtract how often it calls <i>valid</i> code invalid. What is "
        "left is the only evidence that it can tell the two apart. A model scoring 50% "
        "by calling everything invalid has a gap of zero; a model that genuinely "
        "discriminates has a positive one. Bars are green where the 95% interval "
        "clears zero.<br><br>"
        f"<b>{n_sig} of {len(recs)} models clear zero:</b> {lead_txt}. The remaining "
        f"{len(recs) - n_sig} show no evidence of discriminating at all.<br><br>"
        + pol_txt + acc_txt +
        "The abstain column is why two rows sit near zero from below: a model that "
        "almost never returns a validity verdict has nothing to discriminate with, "
        "which is a different failure from judging and judging wrongly. Intervals "
        "resample the 32 solvers, since each appears in both a valid and an invalid "
        "variant, so the two rates are paired.",
        fig_html(fig, height=520, margin=dict(l=210, r=40, t=30, b=70))
        + f'<table class="tbl" style="margin-top:16px"><tr><th>model</th>'
          f'<th>invalid code<br><span class="thsub">called invalid (%)</span></th>'
          f'<th>valid code<br><span class="thsub">called invalid (%)</span></th>'
          f'<th>gap<br><span class="thsub">pp</span></th>'
          f'<th>95% interval<br><span class="thsub">pp</span></th>'
          f'<th>abstain<br><span class="thsub">no verdict (%)</span></th>'
          f'<th>valid_match<br><span class="thsub">raw accuracy</span></th></tr>'
          f'{body}</table>')]


# ── Workflow schematics ───────────────────────────────────────────────────────


def validity_confidence_panel(df):
    """Per-model 2x2 of the validity answer: {hedged, confident} x {yes, no}.

    Diverging from a centre line because the quantity has a natural sign -- "no" is
    not more or less than "yes", it is the other direction -- and a plain stacked bar
    forces the reader to compare two segment lengths that never share an edge.
    Confident answers sit outside their hedged counterparts, so the distance from the
    centre reads as commitment.

    NOTE the buckets come from classify_valid_confidence_2x2, NOT from the published
    `valid_conf` column. That column's rule has an uncertain-YES bucket and no
    uncertain-no, so every hedged negative is filed as a confident no; the 2x2 this
    panel asks for is not derivable from it.
    """
    if df is None or df.empty or "parsed_valid" not in df:
        return [("Validity answers by confidence and direction", "",
                 placeholder("No free-generation rows loaded.",
                             "freegen/run_eval.py"))]
    d = df.copy()
    d["bucket"] = d["parsed_valid"].map(classify_valid_confidence_2x2)

    ORDER = ["Confident No", "Hedged No", "Hedged Yes", "Confident Yes"]
    COLOR = {"Confident No": "#c1546a", "Hedged No": "#e8a3ae",
             "Hedged Yes": "#a8d3ae", "Confident Yes": "#4fa96a"}
    SIGN = {"Confident No": -1, "Hedged No": -1, "Hedged Yes": 1, "Confident Yes": 1}

    recs = []
    for m, g in d.groupby("model"):
        n = len(g)
        share = {b: (g["bucket"] == b).sum() / n for b in ORDER}
        share["no lean"] = (g["bucket"] == "").sum() / n
        share["model"] = m.split("/")[-1]
        share["n"] = n
        share["net"] = share["Confident Yes"] + share["Hedged Yes"]
        recs.append(share)
    recs.sort(key=lambda r: r["net"])
    names = [r["model"] for r in recs]

    fig = go.Figure()
    for b in ORDER:
        fig.add_bar(
            y=names, x=[SIGN[b] * r[b] for r in recs], orientation="h",
            name=b, marker_color=COLOR[b],
            marker_line=dict(color=PANEL, width=0.8),
            customdata=[[r[b], r["n"]] for r in recs],
            hovertemplate="%{y}<br>" + b + ": %{customdata[0]:.1%} of %{customdata[1]}<extra></extra>")
    # Abstentions straddle the centre: a row that never leaned has no direction, and
    # putting it on either side would invent one.
    fig.add_bar(y=names, x=[-r["no lean"] / 2 for r in recs], orientation="h",
                name="no lean", marker_color="#5a6274",
                marker_line=dict(color=PANEL, width=0.8),
                hovertemplate="%{y}<br>no lean<extra></extra>")
    fig.add_bar(y=names, x=[r["no lean"] / 2 for r in recs], orientation="h",
                name="no lean", marker_color="#5a6274", showlegend=False,
                marker_line=dict(color=PANEL, width=0.8), hoverinfo="skip")

    fig.add_vline(x=0, line=dict(color="#8592ae", width=1.2))
    fig.update_layout(
        barmode="relative",
        xaxis=dict(title="share of the model's 256 answers   "
                         "\u2190 says invalid   |   says valid \u2192",
                   tickformat=".0%", gridcolor=GRID, zerolinecolor=GRID,
                   range=[-1.02, 1.02]),
        yaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"))

    body = "".join(
        f"<tr><td>{r['model']}</td>"
        + "".join(f"<td class='num'>{100 * r[b]:.1f}</td>" for b in ORDER)
        + f"<td class='num'>{100 * r['no lean']:.1f}</td>"
        + f"<td class='num'>{100 * (r['Hedged No'] + r['Hedged Yes']):.1f}</td></tr>"
        for r in sorted(recs, key=lambda r: -r["net"]))

    return [(
        "Validity answers by confidence and direction",
        "Every model answers the same validity question 256 times. This is what those "
        "answers look like before any of them are scored: which way the model leaned, "
        "and whether it committed.<br><br>"
        "<b>Read outward from the centre.</b> Left of the line the model called the "
        "code invalid, right of it valid. The pale band on each side is the hedged "
        "form of that same answer &mdash; a lean with a qualifier attached. Grey at "
        "the centre is a row that took no direction at all.<br><br>"
        "<b>The buckets are not the published <code>valid_conf</code> column.</b> That "
        "rule has an uncertain-yes bucket and no uncertain-no, so a hedged &quot;no, "
        "though it might be fine for small dt&quot; is filed as a <i>confident</i> no "
        "while &quot;yes, but the boundary conditions are wrong&quot; keeps its hedge. "
        "The two directions are therefore not comparable under it, and the 2&times;2 "
        "asked for here cannot be derived from it. These buckets come from a symmetric "
        "rule applied at report time; the stored column is untouched.",
        fig_html(fig, height=520, margin=dict(l=210, r=40, t=30, b=90))
        + f'<table class="tbl" style="margin-top:16px"><tr><th>model</th>'
          f'<th>confident<br><span class="thsub">no (%)</span></th>'
          f'<th>hedged<br><span class="thsub">no (%)</span></th>'
          f'<th>hedged<br><span class="thsub">yes (%)</span></th>'
          f'<th>confident<br><span class="thsub">yes (%)</span></th>'
          f'<th>no lean<br><span class="thsub">(%)</span></th>'
          f'<th>hedged<br><span class="thsub">either way (%)</span></th></tr>'
          f'{body}</table>')]




def consistency_figure_panels(xmodal_rows=None):
    """The four cross-representation figures, embedded straight into this report.

    They are matplotlib rather than plotly because they are the paper figures --
    the same functions that write figures/*.pdf for submission. Rendering them here
    as PNG means the report shows exactly what the paper will show, instead of a
    plotly restatement that could drift from it.

    Reads whatever CSV it is given. While that CSV is the synthetic generator's
    output, every panel says so in its own banner: an unlabelled synthetic figure
    sitting beside real results is the single most misleading thing this report
    could contain.
    """
    import base64
    import io

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from viz.consistency import figures as CF, style as CS
        # Match the report's ground. The paper build still renders light; this is the
        # same figure code with a different palette, not a second set of figures.
        CS.apply("dark")
    except Exception as exc:                                    # noqa: BLE001
        return [("Cross-representation figures", "",
                 placeholder(f"Could not import the plotting module ({exc}).",
                             "viz/consistency/"))]

    # Real results first. The generator exists to develop the figures before data
    # lands; once it has landed, showing synthetic panels beside real ones is the
    # most misleading thing this report could do.
    synthetic, note = False, ""
    d = None
    try:
        from viz.consistency.adapter import from_xmodal
        if xmodal_rows:
            d = from_xmodal(pd.DataFrame(xmodal_rows))
    except Exception as exc:                                    # noqa: BLE001
        note = f" (adapter failed: {exc})"
        d = None
    if d is None or d.empty:
        from viz.consistency.synth import Effects, generate
        d = generate(Effects())
        synthetic = True

    banner = ("" if not synthetic else
              '<div class="synthbanner"><b>SYNTHETIC DATA.</b> No cross-modal rows '
              'reached this report' + note + ', so these panels are rendered from '
              '<code>viz/consistency/synth.py</code> with known injected effects. '
              'Do not read them as results.</div>')
    if not synthetic:
        banner = ('<div class="realbanner"><b>Real data</b> &mdash; '
                  f'{len(d):,} rows from <code>pde-llm-eval-xmodal-consistency</code>, '
                  'mapped through <code>viz/consistency/adapter.py</code>. '
                  '<b>One field is missing:</b> <code>judge_correct</code> has no '
                  'counterpart in the run &mdash; no LLM-judge pass has been made over '
                  'the justifications &mdash; so the judge half of the last panel is '
                  'absent rather than invented.</div>')

    CAPTIONS = {
        "fig1_blame_matrix": (
            "Blame matrix",
            "Rows are conditions, columns are the view the model accused. "
            "Row-normalized, so each row asks: given THIS was corrupted, where did "
            "the blame land? The four trajectory rungs get their own rows &mdash; "
            "collapsing them averages a shape-matched noise field together with the "
            "invalid solver's own output."),
        "fig2_trust_scatter": (
            "Which view does it trust, and is that trust lexical?",
            "Detection rate against false-blame rate, one point per modality per "
            "naming condition. The arrow is the finding: if obfuscating identifiers "
            "moves a view, the model's trust in it was keyed to names rather than to "
            "physics. The side panel takes the trajectory apart by corruption rung."),
        "fig3_outcome_stack": (
            "What actually happened, per condition",
            "Every item lands in exactly one outcome, so the bars sum to 1. Faceted "
            "by reasoning setting. This is the panel that separates &quot;never "
            "noticed&quot; from &quot;noticed and blamed the wrong view&quot;."),
        "fig4_justification_gap": (
            "Does the stated reason match the picked slot?",
            "Localization accuracy against the rate at which a judge finds the "
            "justification names the real defect. The annotated gap is how much of "
            "the localization score the model cannot explain. A0 has no corrupted "
            "view, so localization is undefined there and reads n/a rather than 0."),
    }

    has_judge = "judge_correct" in d.columns and d["judge_correct"].notna().any()

    out = []
    for name, fn in CF.FIGURES.items():
        title, question = CAPTIONS.get(name, (name, ""))
        if name == "fig4_justification_gap" and not has_judge:
            # Rendering it anyway would leave a chart showing only localization
            # accuracy -- already in F.1 and F.3 -- under a title promising a
            # comparison the data cannot support.
            out.append((title, question, banner + placeholder(
                "This panel compares two things: how often the model picked the "
                "right view, and how often its stated reason actually named the "
                "real defect. The second has never been measured &mdash; there is "
                "no <code>judge_correct</code> field in the run, because no "
                "LLM-judge pass has been made over the 4,096 justifications. "
                "Without it the panel would show only localization accuracy, which "
                "F.1 and F.3 already show.",
                "an LLM-judge pass over the justification column; "
                "viz/consistency/figures.py::fig4_justification_gap renders it "
                "automatically once that column exists")))
            continue
        try:
            fig = fn(d)
            buf = io.BytesIO()
            # Must be the FIGURE's own colour, not a literal. Forcing white put a
            # white frame around a dark figure AND left every label drawn outside
            # the axes -- titles, ticks, colorbar text -- as light grey on white.
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            img = (f'<div class="figwrap"><img alt="{title}" '
                   f'src="data:image/png;base64,{b64}"></div>')
        except Exception as exc:                                # noqa: BLE001
            img = placeholder(f"{name} failed to render: {exc}",
                              "viz/consistency/figures.py")
        out.append((title, question, banner + img))
    return out




# ── The confidence breakdown, pooled and per model ────────────────────────────
# Shared by the two panels below so they cannot disagree about what a bucket is.
# NOTE the buckets come from classify_valid_confidence_2x2, NOT from the published
# `valid_conf` column, which is what freegen/report.py's panel ② uses. That column's
# rule has an uncertain-YES bucket and no uncertain-no, so every hedged negative is
# filed as a confident no -- an asymmetry that would sit badly next to the two
# direction-and-confidence panels in this document, which are symmetric by
# construction. Same figure, symmetric rule.
# "no lean" is a REAL bucket and sits between the two directions, not off the chart.
# classify_valid_confidence_2x2 returns "" for an answer with no direction at all --
# an abstention is not a lean -- and on this roster that is 92 of 5,426 answers
# (1.7%), things like "depends on the time step and grid resolution". Leaving it out
# of the stack made every bar fall short of 100% by a different amount with nothing
# saying why, which reads as a rendering fault rather than as data.
# DIRECTION only. The confident/hedged split was four legend entries here and it
# overstated what the verdict field can carry: the prompt asks for a terse
# fill-in-the-blank answer ("Be concise. Output only: ... valid: ____"), so 85% of
# answers are a bare yes or no and the hedged bands were slivers that invited reading
# a confidence signal off a format artefact. Real confidence needs resampling, not a
# lexicon -- see the k=3 flip rate.
HEDGE_ORDER = ["predicts invalid", "predicts valid"]
HEDGE_COLOR = {"predicts invalid": "#c1546a", "predicts valid": "#4fa96a"}
_DIRECTION = {"Confident No": "predicts invalid", "Hedged No": "predicts invalid",
              "Confident Yes": "predicts valid", "Hedged Yes": "predicts valid"}


def _hedge_frame(df):
    """(present conditions, frame with a `bucket` column) or (None, None)."""
    if df is None or df.empty or "parsed_valid" not in df or "mod_type" not in df:
        return None, None
    d = df.copy()
    d["bucket"] = d["parsed_valid"].map(classify_valid_confidence_2x2)
    # An answer with no direction at all is dropped, not drawn: after the run-on
    # parser fix that is 4 draws in 5,426, and a legend entry for it implied a
    # behaviour the models do not have. The bars remain a share of answers that
    # stated a direction, so they still sum to 100.
    d["bucket"] = d["bucket"].map(_DIRECTION)
    d = d[d["bucket"].notna()]
    present = [(k, lab, gt) for k, lab, gt in COND_ORDER if (d["mod_type"] == k).any()]
    return (present, d) if present else (None, None)


def _hedge_labels(present):
    """Condition labels carrying the answer that is actually correct for each.

    Without the marker a reader takes "more green" for "better", which is exactly
    backwards on the four invalid rows.
    """
    return [(lab + " \u2713") if gt else ("\u26a0 " + lab) for _, lab, gt in present]


def _hedge_split(present):
    """Index of the first invalid condition, for the divider and the shading."""
    return sum(1 for _, _, gt in present if gt)


def hedge_breakdown_panel(df):
    """Stacked confidence shares per perturbation, every model pooled.

    The overview the per-model grid below deviates from. Two of the eight
    perturbations -- a corrupted comment and corrupted variable names -- change the
    ANNOTATION and not the physics, so a model reading the code should answer them
    exactly as it answers the clean variant. Movement between those bars is the
    model reading the label.
    """
    present, d = _hedge_frame(df)
    if present is None:
        return [("Validity confidence breakdown", "",
                 placeholder("No free-generation rows loaded.", "freegen/run_eval.py"))]

    labels, split = _hedge_labels(present), _hedge_split(present)
    fig = go.Figure()
    for bucket in HEDGE_ORDER:
        means, hi_err, lo_err = [], [], []
        for code, _, _ in present:
            sub = d[d["mod_type"].eq(code)].copy()
            sub["_ind"] = (sub["bucket"] == bucket).astype(float)
            # Resampling the 32 SYSTEMS, not the rows: eight conditions and eight
            # models share each base solver, so rows within a system are not
            # independent and a row bootstrap would shrink these intervals by
            # roughly the square root of that clustering.
            mn, lo, hi = _cluster_ci(sub, "_ind")
            means.append(mn * 100)
            lo_err.append((mn - lo) * 100 if lo == lo else 0.0)
            hi_err.append((hi - mn) * 100 if hi == hi else 0.0)
        fig.add_bar(
            name=bucket, x=labels, y=means, marker_color=HEDGE_COLOR[bucket],
            marker_line=dict(color=PANEL, width=0.6),
            error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                         color="rgba(224,224,224,0.5)", thickness=1.1, width=4),
            hovertemplate=bucket + " | %{x}: %{y:.1f}%<extra></extra>")

    boundary = split - 0.5
    fig.add_vline(x=boundary, line=dict(color="#a878d8", width=1.5, dash="dash"))
    fig.add_vrect(x0=boundary, x1=len(present) - 0.5, fillcolor="#7d3c98",
                  opacity=0.10, line_width=0)
    fig.add_annotation(x=0, y=107, text="VALID CODE \u2014 correct answer is Yes",
                       showarrow=False, xanchor="left",
                       font=dict(size=10, color="#8fd694"))
    fig.add_annotation(x=len(present) - 1, y=107,
                       text="INVALID CODE \u2014 correct answer is No",
                       showarrow=False, xanchor="right",
                       font=dict(size=10, color="#d98fd6"))
    fig.update_layout(
        barmode="stack", height=470,
        yaxis=dict(title="% of answers", range=[0, 112], gridcolor=GRID),
        xaxis=dict(tickangle=-25),
        legend=dict(orientation="h", y=-0.42, x=0.5, xanchor="center"))

    return [(
        "What the models predict, all models pooled",
        "Every answer to the validity question, bucketed by which way the model "
        "leaned and whether it committed. Each bar is one perturbation; the shaded "
        "half is the four where the code really is broken.<br><br>"
        "<b>Green over the shaded half is the failure of interest</b> &mdash; the "
        "model asserting that physically invalid code is fine. Red over the unshaded "
        "half is the opposite failure, and the one that has grown: a false alarm on "
        "working code.<br><br>"
        "<b>Corrupt Comment and Obfuscated Variables are the controls.</b> Both leave "
        "the physics untouched and change only the naming, so a model reading the "
        "code should answer them exactly as it answers Clean. Any movement is the "
        "model reading the label instead.<br><br>"
        "Intervals are 95% bootstrap over the <b>32 base systems</b>, not over rows: "
        "eight conditions and eight models share each solver, so a row bootstrap "
        "would narrow every one of them. Each interval is for that bucket's own "
        "share, not for the cumulative height it is drawn at. Answers that stated no "
        "direction at all are excluded from the denominator, so the bars still sum "
        "to 100 &mdash; after the run-on parser fix that is 4 draws in 5,426.<br><br>"
        "<b>This is direction, not confidence.</b> The confident/hedged split used to "
        "be four bands here and it has been dropped: the prompt asks for a terse "
        "fill-in-the-blank verdict, so 85% of answers are a bare <code>yes</code> or "
        "<code>no</code> and the hedged bands were slivers of a format artefact. "
        "Confidence is measured by resampling instead &mdash; see the k=3 flip rate.",
        fig_html(fig, height=520, margin=dict(l=70, r=30, t=40, b=190)))]


def hedge_breakdown_by_model_panel(df):
    """The same breakdown, one small multiple per model.

    The pooled panel above averages across the roster, and on this measure the
    roster does not agree: the newer checkpoints answer the invalid half almost
    perfectly and the valid half barely better than chance, and the older ones do
    the reverse. A single pooled bar shows the average of two opposite behaviours
    and looks like moderate competence at both.
    """
    present, d = _hedge_frame(df)
    if present is None or "model" not in d:
        return []

    models = sorted(d["model"].unique(),
                    key=lambda m: (d[d["model"].eq(m)]["bucket"]
                                   .isin(["Hedged Yes", "Confident Yes"]).mean()))
    if len(models) < 2:
        return []

    labels, split = _hedge_labels(present), _hedge_split(present)
    ncols = 4
    nrows = (len(models) + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows, cols=ncols, shared_yaxes=True,
        subplot_titles=[m.split("/")[-1] for m in models],
        vertical_spacing=0.09, horizontal_spacing=0.045)

    for i, model in enumerate(models):
        r, c = i // ncols + 1, i % ncols + 1
        g = d[d["model"].eq(model)]
        for bucket in HEDGE_ORDER:
            ys = []
            for code, _, _ in present:
                sub = g[g["mod_type"].eq(code)]
                ys.append(100 * (sub["bucket"] == bucket).mean() if len(sub) else 0.0)
            fig.add_bar(
                x=labels, y=ys, name=bucket, marker_color=HEDGE_COLOR[bucket],
                marker_line=dict(color=PANEL, width=0.4),
                legendgroup=bucket, showlegend=(i == 0),
                hovertemplate=("<b>" + model.split("/")[-1]
                               + "</b><br>" + bucket
                               + " | %{x}: %{y:.1f}%<extra></extra>"),
                row=r, col=c)
        fig.add_vrect(x0=split - 0.5, x1=len(present) - 0.5, fillcolor="#7d3c98",
                      opacity=0.10, line_width=0, row=r, col=c)
        fig.add_vline(x=split - 0.5, line=dict(color="#a878d8", width=1, dash="dash"),
                      row=r, col=c)

    # Legend ABOVE the panels: at the bottom it shares a strip with the vertical
    # tick labels and clips the middle columns.
    fig.update_layout(barmode="stack", height=360 * nrows,
                      legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                                  yanchor="bottom"))
    fig.update_yaxes(range=[0, 100], gridcolor=GRID, title_text="")
    # Axis title on the LEFT COLUMN only -- shared_yaxes hides the ticks on the
    # inner panels, so repeating the title there would label an axis with no scale.
    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text="% of answers", row=r, col=1)
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=8), showticklabels=False)
    # Tick labels on the BOTTOM row only. Eight labels at -55 degrees are taller than
    # the gap between rows, so every one of them ran into the subplot title beneath
    # it. The columns share one category axis, so a label under the bottom panel
    # names its whole column and nothing is lost by hiding the repeats.
    last_row_of = {}
    for i in range(len(models)):
        last_row_of[i % ncols] = i // ncols + 1
    for c, r in last_row_of.items():
        fig.update_xaxes(showticklabels=True, row=r, col=c + 1)
    for ann in fig.layout.annotations[:len(models)]:
        ann.font = dict(size=11, color=FG)

    return [(
        "What each model predicts",
        "Ordered by how often the model says the code is valid, least to most. The "
        "shaded half of each panel is the four perturbations where the code really "
        "is broken.<br><br>"
        "<b>This is the panel the pooled bar hides.</b> Read the shaded half across "
        "the row and the newer checkpoints go almost solid red &mdash; they call "
        "broken code broken nearly every time. Then read the unshaded half of the "
        "same panels: the red does not go away. They are not detecting the fault, "
        "they are readier to call anything faulty, and the pooled average of that "
        "against the older models' opposite bias reads as moderate competence at "
        "both.<br><br>"
        "No intervals here: each bar is one model on one perturbation, 32 items, and "
        "an interval on 32 would be wide enough to swamp the panel. The pooled figure "
        "above carries the uncertainty; this one carries the shape.<br><br>"
        "<b>What this panel cannot tell you is how sure any of them are.</b> Every "
        "model here is a reasoning model answering with thinking on, and the "
        "deliberation happens inside <code>&lt;think&gt;</code>, which is stripped "
        "before parsing. A model can reason for twenty thousand tokens about whether "
        "the CFL condition holds and then write <code>valid: no</code>, and this "
        "panel records that identically to a reflexive answer. Pooled over the "
        "roster, 90.1% of items have every draw giving an unqualified verdict "
        "&mdash; but only 67.4% have all three draws agreeing.",
        fig_html(fig, height=360 * nrows + 220,
                 margin=dict(l=55, r=30, t=95, b=250)))]


def perturbation_confidence_panel(df):
    """Validity answers per perturbation, split by direction AND confidence.

    The same 2x2 as the per-model panel, but the rows are the eight perturbations.
    That is the comparison that matters for Experiment 1: two of the perturbations
    -- a corrupted comment and corrupted variable names -- change the ANNOTATION and
    not the physics, so a model reading the code should answer them exactly as it
    answers the clean variant. Any movement between those rows is the model reading
    the label instead.

    Each row is marked with the answer that is actually correct for it, because the
    same lean is right on the valid rows and wrong on the invalid ones, and a bar
    chart with no truth marker invites reading "more green" as "better".
    """
    if df is None or df.empty or "parsed_valid" not in df or "mod_type" not in df:
        return [("Validity answers by perturbation", "",
                 placeholder("No free-generation rows loaded.",
                             "freegen/run_eval.py"))]
    d = df.copy()
    d["bucket"] = d["parsed_valid"].map(classify_valid_confidence_2x2)

    ORDER = ["Confident No", "Hedged No", "Hedged Yes", "Confident Yes"]
    COLOR = {"Confident No": "#c1546a", "Hedged No": "#e8a3ae",
             "Hedged Yes": "#a8d3ae", "Confident Yes": "#4fa96a"}
    SIGN = {"Confident No": -1, "Hedged No": -1, "Hedged Yes": 1, "Confident Yes": 1}

    recs = []
    for code, label, gt_valid in COND_ORDER:
        g = d[d["mod_type"].eq(code)]
        if g.empty:
            continue
        n = len(g)
        r = {b: (g["bucket"] == b).sum() / n for b in ORDER}
        r["no lean"] = (g["bucket"] == "").sum() / n
        r.update(code=code, label=label, gt_valid=gt_valid, n=n)
        # "Correct" = leaned the way the ground truth points, confident or hedged.
        side = ["Hedged Yes", "Confident Yes"] if gt_valid else \
               ["Hedged No", "Confident No"]
        r["correct_lean"] = sum(r[b] for b in side)
        recs.append(r)
    if not recs:
        return []
    recs.reverse()                       # paper order reads top-down
    names = [r["label"] for r in recs]

    fig = go.Figure()
    for b in ORDER:
        fig.add_bar(y=names, x=[SIGN[b] * r[b] for r in recs], orientation="h",
                    name=b, marker_color=COLOR[b],
                    marker_line=dict(color=PANEL, width=0.8),
                    customdata=[[r[b], r["n"]] for r in recs],
                    hovertemplate="%{y}<br>" + b +
                                  ": %{customdata[0]:.1%} of %{customdata[1]}<extra></extra>")
    fig.add_bar(y=names, x=[-r["no lean"] / 2 for r in recs], orientation="h",
                name="no lean", marker_color="#5a6274",
                marker_line=dict(color=PANEL, width=0.8),
                hovertemplate="%{y}<br>no lean<extra></extra>")
    fig.add_bar(y=names, x=[r["no lean"] / 2 for r in recs], orientation="h",
                name="no lean", marker_color="#5a6274", showlegend=False,
                marker_line=dict(color=PANEL, width=0.8), hoverinfo="skip")
    fig.add_vline(x=0, line=dict(color="#8592ae", width=1.2))

    # Truth markers: an arrow head on the side that is correct for that row.
    for r in recs:
        fig.add_annotation(x=(0.97 if r["gt_valid"] else -0.97), y=r["label"],
                           text=("code IS valid \u2192" if r["gt_valid"]
                                 else "\u2190 code is NOT valid"),
                           showarrow=False, font=dict(size=9, color="#7b88a4"),
                           xanchor="right" if r["gt_valid"] else "left")

    fig.update_layout(
        barmode="relative",
        xaxis=dict(title="share of the 352 answers per perturbation   "
                         "\u2190 says invalid   |   says valid \u2192",
                   tickformat=".0%", gridcolor=GRID, zerolinecolor=GRID,
                   range=[-1.02, 1.02]),
        yaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center"))

    body = "".join(
        f"<tr><td>{r['label']}</td>"
        f"<td class='num'>{'valid' if r['gt_valid'] else 'invalid'}</td>"
        + "".join(f"<td class='num'>{100 * r[b]:.1f}</td>" for b in ORDER)
        + f"<td class='num'>{100 * r['no lean']:.1f}</td>"
        + f"<td class='{'high' if r['correct_lean'] >= 0.5 else 'low'}'>"
          f"{100 * r['correct_lean']:.1f}</td></tr>"
        for r in reversed(recs))

    # The comparison the panel exists for, computed rather than asserted.
    by_code = {r["code"]: r for r in recs}
    notes = []
    for clean, perturbed, what in (
            ("NoComm_Valid", "CorrComm", "a corrupted comment"),
            ("NoComm_Valid", "NoComm_CorrVar", "corrupted variable names")):
        if clean in by_code and perturbed in by_code:
            a, b = by_code[clean]["correct_lean"], by_code[perturbed]["correct_lean"]
            notes.append(f"Adding {what} to code that is still physically valid moves "
                         f"the correct lean from {100 * a:.1f}% to {100 * b:.1f}% "
                         f"({100 * (b - a):+.1f}pp).")
    note_html = ("<br><br><b>" + " ".join(notes) + "</b>") if notes else ""

    return [(
        "Validity answers by perturbation",
        "The same four buckets as the previous panel, but split by which "
        "perturbation was applied rather than by which model answered. All eleven "
        "models are pooled; each row is 352 answers.<br><br>"
        "<b>Read against the truth marker.</b> The four upper rows are code that is "
        "physically valid, so leaning right is correct; the four lower rows are "
        "genuinely invalid code, so leaning left is correct. The marker on each row "
        "says which side that is.<br><br>"
        "<b>Two of these perturbations change nothing physical.</b> A corrupted "
        "comment and corrupted variable names leave the code's behaviour identical "
        "&mdash; a model judging the physics should answer them exactly as it "
        "answers the clean variant." + note_html,
        fig_html(fig, height=560, margin=dict(l=200, r=40, t=30, b=100))
        + f'<table class="tbl" style="margin-top:16px"><tr><th>perturbation</th>'
          f'<th>ground truth</th>'
          f'<th>confident<br><span class="thsub">no (%)</span></th>'
          f'<th>hedged<br><span class="thsub">no (%)</span></th>'
          f'<th>hedged<br><span class="thsub">yes (%)</span></th>'
          f'<th>confident<br><span class="thsub">yes (%)</span></th>'
          f'<th>no lean<br><span class="thsub">(%)</span></th>'
          f'<th>leaned correctly<br><span class="thsub">(%)</span></th></tr>'
          f'{body}</table>')]


def flow(stages, accent="e1"):
    """Render a pipeline schematic from a list of stages.

    Plain HTML/CSS rather than SVG: the stage text is real, selectable, wrapping
    text, and it stays readable when the window is narrow. Each stage is
    (label, title, detail, chips); chips render as a fan-out row and are what make
    the condition grid visible as a grid instead of as prose.
    """
    out = [f'<div class="flow flow-{accent}">']
    for i, (label, title, detail, chips) in enumerate(stages):
        if i:
            out.append('<div class="farrow"></div>')
        chip_html = ""
        if chips:
            chip_html = '<div class="fchips">' + "".join(
                f'<span class="fchip{" fchip-hi" if str(c).startswith("*") else ""}">'
                f'{str(c).lstrip("*")}</span>' for c in chips) + "</div>"
        out.append(
            f'<div class="fstage"><div class="fnum">{label}</div>'
            f'<div class="fbody"><div class="ftitle">{title}</div>'
            f'<div class="fdet">{detail}</div>{chip_html}</div></div>')
    out.append("</div>")
    return "\n".join(out)


def exp1_schematic(df):
    n = len(df) if df is not None else 0
    nm = df["model"].nunique() if df is not None and len(df) else 0
    return flow([
        ("data", "32 PDE solvers",
         "<code>data/merged_mod_jul28.csv</code> — one valid reference solver per "
         "system, each with a hand-checked PDE class, numerical method, dominant "
         "behaviour and validity label.", []),
        ("×8", "Eight conditions per solver",
         "Comments, identifiers and physical correctness are crossed, so a model that "
         "reads surface text rather than computation degrades on a predictable subset. "
         "32 systems × 8 = <b>256 items</b>.",
         ["Comm_Valid", "NoComm_Valid", "CorrComm_Valid", "NoComm_CorrVar_Valid",
          "Comm_InValid", "NoComm_InValid", "CorrComm_InValid", "NoComm_CorrVar_InValid"]),
        ("ask", "One open prompt, four fields",
         "The model sees source only — no equation, no trajectory. Answer format is "
         "left unconstrained on purpose so that hedging stays visible instead of being "
         "forced into a yes/no.",
         ["*pde", "*method", "*behavior", "*valid"]),
        ("run", f"{nm or 11} models, vLLM batch",
         "<code>freegen/run_eval.py</code> on h200 nodes. Generation length is the "
         "model's own maximum; a truncated row is a failed row, not a datum. Every "
         "response is stored whole.", []),
        ("parse", "Deterministic field parser",
         "<code>freegen/parse_score.py</code>. Tolerant of markdown emphasis, bullets "
         "and numbered lists — a formatting difference must never read as a refusal. "
         "An unparsed field scores <b>null</b>, never zero.", []),
        ("score", "Four scores plus two axes",
         "Alias tables shared with Experiment 2 so &quot;identify the PDE&quot; means "
         "the same thing in both.",
         ["pde_match", "method_recall", "behavior_recall", "valid_match",
          "method_axis", "valid_conf"]),
        ("out", f"{n} rows on HuggingFace",
         "<code>bermaneh/pde-llm-eval-results-jul28</code> — every row carries its "
         "prompt version, finish reason and full response text.", []),
    ], "e1")


def exp2_workflow(rows):
    """The experimental workflow as a diagram, not a list of stages.

    The previous version was a vertical stack of prose blocks. It described the
    pipeline accurately and showed nothing: the shape of this experiment is a
    fan-out (one system becomes four views) followed by a single corruption and a
    shuffle, and none of that is visible in a column of paragraphs.
    """
    n = len(rows) if rows else 0
    views = [("code", "the solver, comments stripped", "#e8a25a"),
             ("equation", "the governing PDE", "#8fd694"),
             ("trajectory", "10 frames of the solution", "#7eb8ff"),
             ("description", "prose account of the physics", "#d98fd6")]
    view_boxes = "".join(
        f'<div class="wf-view" style="border-top-color:{c}">'
        f'<div class="wf-vname" style="color:{c}">{name}</div>'
        f'<div class="wf-vdet">{det}</div></div>' for name, det, c in views)
    conds = "".join(
        f'<span class="fchip{" fchip-hi" if k != "A0" else ""}">{k}</span>'
        for k, _, _, _ in COND_DECODER)
    outs = "".join(f'<span class="wf-out">{o}</span>' for o in
                   ("agree", "outlier", "system_pde_class", "system_num_method",
                    "justification"))
    return f"""
<div class="wf">

  <div class="wf-row"><div class="wf-box wf-src">
    <b>32 PDE systems</b><div class="wf-sub">Burgers · Heat · Wave · Navier&ndash;Stokes.
    Each has a verified valid solver and an invalid counterpart.</div></div></div>
  <div class="wf-arrow"><span>each system is rendered four independent ways</span></div>

  <div class="wf-views">{view_boxes}</div>
  <div class="wf-note">All four describe the same system. Any one of them is enough to
    identify it &mdash; that is what makes them independent representations rather than
    four parts of one description.</div>
  <div class="wf-arrow"><span>corrupt <b>exactly one</b> view, or none &rarr; 8 conditions</span></div>

  <div class="wf-box wf-cond">{conds}
    <div class="wf-sub">A0 is the control: nothing corrupted. It is the only place a
      false alarm can be measured. The trajectory gets four rungs because its
      corruption can be made much grosser or much subtler than the others'.</div></div>
  <div class="wf-arrow"><span>&times; 2 identifier settings (real | obfuscated)
    &nbsp;&times;&nbsp; 2 slot orders</span></div>

  <div class="wf-box wf-prompt">
    <b>one prompt &mdash; View 1, View 2, View 3, View 4</b>
    <div class="wf-sub">Presented in a seeded random order behind a neutral legend, so
      position carries no information and the model is never told which kind of view
      sits in which slot. It is told at most one may not belong.</div>
    <div class="wf-math">32 systems &times; 8 conditions &times; 2 namings &times; 2 orders
      = <b>1,024 items</b> per arm</div></div>
  <div class="wf-arrow"><span>run on every arm</span></div>

  <div class="wf-box wf-models"><b>4 arms</b> &nbsp;
    <span class="fchip">Qwen3-32B · thinking off</span>
    <span class="fchip">Qwen3-32B · thinking on</span>
    <span class="fchip">QwQ-32B</span>
    <span class="fchip">DeepSeek-R1-Distill-32B</span>
    <div class="wf-math">4 arms &times; 1,024 = <b>{n:,} responses</b></div></div>
  <div class="wf-arrow"><span>structured output</span></div>

  <div class="wf-box wf-out-row">{outs}</div>
  <div class="wf-arrow"><span>scored three ways</span></div>

  <div class="wf-views wf-metrics">
    <div class="wf-view"><div class="wf-vname">did it flag?</div>
      <div class="wf-vdet">said the views disagree. Compared against its own A0 rate,
        because seven of eight conditions are corrupted.</div></div>
    <div class="wf-view"><div class="wf-vname">did it name the right view?</div>
      <div class="wf-vdet">scored only where it flagged. Chance is 1 in 4.</div></div>
    <div class="wf-view"><div class="wf-vname">who did it blame instead?</div>
      <div class="wf-vdet">the accusation resolved back through that item's own slot
        order &mdash; the panel that turned out to matter most.</div></div>
  </div>
</div>"""


def exp2_schematic(rows):
    return flow([
        ("data", "One system, four independent representations",
         "Built from the same 32 solvers. The four views are generated separately and "
         "each is sufficient on its own to identify the system.",
         ["*code (comments stripped)", "*governing equation", "*trajectory (10 frames)",
          "*natural-language description"]),
        ("×8", "Corrupt exactly one view — eight conditions",
         "A0 is the all-agree control and the only source of the false-alarm rate. The "
         "four trajectory rungs are a ladder: shape-matched noise, a permutation of the "
         "trajectory's own values, the dataset's delivered wrong trajectory, and the "
         "actual output of the invalid solver.",
         ["A0 none", "X_C code", "X_M math", "X_D description",
          "X_T_rand", "X_T_shuf", "X_T_swap", "X_T_exec"]),
        ("mask", "Randomize slot order, vary identifiers",
         "Views are presented as View 1–4 behind a neutral legend, in an order seeded "
         "per item, so position carries no information. Identifiers are either the "
         "solver's real names or obfuscated — the lexical-cue control.",
         ["order_seed", "names = real | obfuscated"]),
        ("ask", "Structured output, five fields",
         "The three uncorrupted views form a majority that determines the answer. "
         "Localization is asked separately from detection so the two can come apart.",
         ["*agree", "*outlier", "system_pde_class", "system_num_method",
          "*justification"]),
        ("run", "3 models × {thinking on, off}",
         "<code>crossmodal/eval/run_cross_modal_consistency.py</code>. Context length is "
         "derived per model from its own config rather than assumed. Reasoning arms get "
         "32k output tokens.",
         ["Qwen3-32B on/off", "QwQ-32B", "DeepSeek-R1-Distill-32B"]),
        ("parse", "Parse route recorded, never guessed",
         "<code>crossmodal/eval/parse_consistency.py</code>. JSON, fenced JSON, embedded "
         "JSON, then a regex cascade — and the route that succeeded is stored per row. An "
         "unclosed <code>&lt;think&gt;</code> parses as a failure, not as whatever the "
         "trace ended on.", []),
        ("score", "d′, not accuracy",
         "Seven of eight conditions are corrupted, so &quot;always say no&quot; scores "
         "0.875 while knowing nothing. Intervals bootstrap the <b>32 systems</b>, not the "
         "items, because items within a system are not independent.",
         ["d′ (Hautus)", "localization | detected", "degeneracy flag"]),
        ("out", f"{len(rows)} rows on HuggingFace",
         "<code>bermaneh/pde-llm-eval-xmodal-consistency</code> — plus the rendered-view "
         "CSV, which stores each item's four view bodies and the assembled prompt "
         "verbatim.", []),
    ], "e2")


def exp1_headline(df):
    """The numbers, before any chart."""
    if df is None or df.empty:
        return placeholder("No free-generation rows loaded.",
                           "sbatch/run_freegen_jul28.sbatch")
    order = df.groupby("model")["pde_match"].mean().sort_values(ascending=False).index.tolist()
    body = []
    for m in order:
        s = df[df.model == m]
        on = s[s.method_axis == "on"] if "method_axis" in s else s.iloc[0:0]
        hedge = (s["valid_conf"] == "Hedged").mean() if "valid_conf" in s else float("nan")
        body.append(
            f"<tr><td>{m}</td><td>{len(s)}</td>"
            f"<td class='num'>{s['pde_match'].mean():.3f}</td>"
            f"<td class='num'>{on['method_recall'].mean() if len(on) else float('nan'):.3f}</td>"
            f"<td class='num'>{s['behavior_recall'].mean():.3f}</td>"
            f"<td class='num'>{s['valid_match'].mean():.3f}</td>"
            f"<td class='num'>{hedge:.2f}</td></tr>")
    best = df.groupby("model")["pde_match"].mean().max()
    vmean = df["valid_match"].mean()
    return (
        f'<div class="kpis">'
        f'<div class="kpi"><div class="kv">{df["pde_match"].mean():.3f}</div>'
        f'<div class="kl">mean pde_match, all models</div></div>'
        f'<div class="kpi"><div class="kv">{best:.3f}</div>'
        f'<div class="kl">best model, pde_match</div></div>'
        f'<div class="kpi"><div class="kv">{vmean:.3f}</div>'
        f'<div class="kl">mean valid_match &nbsp;<span class="ksub">chance = 0.5</span></div></div>'
        f'<div class="kpi"><div class="kv">'
        f'{int((df.get("finish_reason", pd.Series(dtype=str)) == "length").sum())}</div>'
        f'<div class="kl">truncated rows</div></div></div>'
        f'<table class="tbl"><tr><th>model</th><th>rows</th><th>pde</th>'
        f'<th>method (on-axis)</th><th>behaviour</th><th>validity</th>'
        f'<th>hedge rate</th></tr>{"".join(body)}</table>')


def exp2_headline(rows, summary):
    if not rows:
        return placeholder("No cross-modal rows loaded.",
                           "sbatch/run_cross_modal_consistency.sbatch")
    if not summary or "arms" not in summary:
        return placeholder("Rows are present but the aggregate has not been rebuilt.",
                           "crossmodal/eval/aggregate_cross_modal.py")
    body = []
    for arm, a in sorted(summary["arms"].items()):
        c = a["conditions"]
        corr = {k: v for k, v in c.items() if k != "A0"}
        det = sum(v["detection"] * v["n_items"] for v in corr.values()) / \
              sum(v["n_items"] for v in corr.values())
        fa = 1 - c["A0"]["detection"] if "A0" in c else float("nan")
        loc = [v["localization"] for v in corr.values() if v["localization"] is not None]
        dps = {k: v["dprime"] for k, v in corr.items() if v["dprime"] is not None}
        worst = min(dps, key=dps.get) if dps else "—"
        best = max(dps, key=dps.get) if dps else "—"
        model, think = arm.split("|")
        body.append(
            f"<tr><td>{model}</td><td>{think.replace('think_','')}</td>"
            f"<td class='num'>{a['n_rows']}</td>"
            f"<td class='num'>{det:.3f}</td><td class='num'>{fa:.3f}</td>"
            f"<td class='num'>{(sum(loc)/len(loc)) if loc else float('nan'):.3f}</td>"
            f"<td class='low'>{worst}</td><td class='high'>{best}</td></tr>")
    n_sys = summary.get("n_systems", 32)
    return (
        f'<div class="kpis">'
        f'<div class="kpi"><div class="kv">{len(rows)}</div>'
        f'<div class="kl">rows &nbsp;<span class="ksub">4 arms × 8 conditions</span></div></div>'
        f'<div class="kpi"><div class="kv">{n_sys}</div>'
        f'<div class="kl">systems resampled per interval</div></div>'
        f'<div class="kpi"><div class="kv">0.25</div>'
        f'<div class="kl">chance on localization</div></div>'
        f'<div class="kpi"><div class="kv">0.875</div>'
        f'<div class="kl">accuracy of &quot;always say no&quot; &nbsp;'
        f'<span class="ksub">why d′</span></div></div></div>'
        f'<table class="tbl"><tr><th>model</th><th>thinking</th><th>rows</th>'
        f'<th>hit rate<br><span class="thsub">corrupted items</span></th>'
        f'<th>false alarms<br><span class="thsub">A0 only</span></th>'
        f'<th>localization<br><span class="thsub">given detected</span></th>'
        f'<th>hardest</th><th>easiest</th></tr>{"".join(body)}</table>')


def exp1_panels(df, first_flag):
    panels = []
    if df is None or df.empty:
        return [("Experiment 1 — free generation",
                 "Given only solver source, name the PDE, the method, the behaviour, "
                 "and judge physical validity.",
                 placeholder("No free-generation rows loaded.",
                             "sbatch/run_freegen_jul28.sbatch, groups a/b/c"))]

    models = sorted(df["model"].unique())
    order = df.groupby("model")["pde_match"].mean().sort_values(ascending=False).index.tolist()

    # V1 was "Identification vs judgement": a per-model grouped bar of pde_match,
    # behavior_recall and valid_match. Removed rather than fixed -- three bars whose
    # chance levels are ~0, ~0 and 0.5 cannot be compared by height, which is exactly
    # how a grouped bar chart invites you to read them. "Score by condition" carries
    # the same numbers, and "Can a model tell valid code from invalid?" asks the
    # validity half properly.

    # V2 — method axis
    if "method_axis" in df:
        pooled, onaxis, n_on = [], [], []
        for m in order:
            sub = df[df.model == m]
            on = sub[sub.method_axis == "on"]
            pooled.append(sub["method_recall"].mean())
            onaxis.append(on["method_recall"].mean() if len(on) else float("nan"))
            n_on.append(len(on))
        fig = go.Figure()
        fig.add_bar(name="method_recall, pooled", x=order, y=pooled, marker_color=ACCENT[1])
        fig.add_bar(name="method_recall, on-axis only", x=order, y=onaxis, marker_color=ACCENT[2])
        fig.add_scatter(name="n on-axis (right axis)", x=order, y=n_on, yaxis="y2",
                        mode="markers+lines", marker=dict(color=ACCENT[0], size=9))
        fig.update_layout(barmode="group", yaxis_title="recall", xaxis_tickangle=-30,
                          yaxis2=dict(title="n rows on-axis", overlaying="y", side="right",
                                      gridcolor="rgba(0,0,0,0)"))
        panels.append((
            "The method axis",
            "Ground truth labels time integration (explicit/implicit) and, when the "
            "solver is FFT-based, a spectral basis. A response naming only a spatial "
            "discretization answers an axis the ground truth leaves blank — an "
            "abstention, not an error. Pooled recall mostly measures how often a "
            "model lands on the labelled axis; on-axis recall measures whether it "
            "knows the method.",
            fig_html(fig)))

    # V3 — condition degradation
    if "mod_type" in df:
        conds = sorted(df["mod_type"].unique())
        fig = go.Figure()
        for i, m in enumerate(order):
            sub = df[df.model == m]
            fig.add_scatter(name=m.split("/")[-1], x=conds,
                            y=[sub[sub.mod_type == c]["pde_match"].mean() for c in conds],
                            mode="lines+markers", marker=dict(size=7),
                            line=dict(color=ACCENT[i % len(ACCENT)]))
        fig.update_layout(yaxis_title="pde_match", xaxis_tickangle=-25)
        panels.append((
            "Degradation across the eight conditions",
            "Comment corruption, identifier obfuscation and physical invalidity, "
            "crossed. A model reading physics rather than surface text should be "
            "flat across conditions that leave the computation unchanged.",
            fig_html(fig)))

    # V4 was "Hedging on validity": raw stacked counts per model. Superseded by
    # "Validity judgements, split by confidence", which shows the same four buckets as
    # percentages against the conditions that move them.

    # V5 — health
    trunc = int((df.get("finish_reason", pd.Series(dtype=str)) == "length").sum())
    nulls = int(df["parsed_pde"].isna().sum()) if "parsed_pde" in df else 0
    rows = "".join(
        f"<tr><td>{m}</td><td>{len(df[df.model == m])}</td>"
        f"<td>{int((df[(df.model == m)].get('finish_reason', pd.Series(dtype=str)) == 'length').sum())}</td>"
        f"<td>{int(df[df.model == m]['parsed_pde'].isna().sum()) if 'parsed_pde' in df else 0}</td>"
        f"<td>{int(df[df.model == m]['model_response'].astype(str).str.len().max())}</td></tr>"
        for m in order)
    panels.append((
        "Raw responses",
        "The scores above are worth exactly as much as these traces. Sampling is "
        "stratified over model and outcome, so the browser always contains wrong "
        "answers, hedges and unparsed fields — not a highlight reel. Ground truth "
        "and the parsed fields sit above each response so you can check the scoring "
        "yourself. Responses are stored and shown in full.",
        response_browser(sample_responses(df), "sbatch/run_freegen_jul28.sbatch",
                         prefix="fg")))

    panels.append((
        "Run health",
        f"Truncation is a failed row, not a datum. Total truncated: <b>{trunc}</b>. "
        f"Total unparsed pde field: <b>{nulls}</b>.",
        f'<table class="tbl"><tr><th>model</th><th>rows</th><th>truncated</th>'
        f'<th>null pde</th><th>longest response (chars)</th></tr>{rows}</table>'))
    return panels


def sample_responses(df, per_cell=3, max_chars=60000):
    """A spread of raw rows to read, not a cherry-picked highlight reel.

    Sampling is stratified over (model x outcome) so the browser always contains
    wrong answers and hedges, not just the flattering ones. Within a cell rows are
    taken at evenly spaced indices rather than the first N, which would cluster on
    whatever solver happens to sort first.
    """
    rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df.model == model]
        cells = {
            "correct on validity": sub[sub.valid_match == 1],
            "wrong on validity": sub[sub.valid_match == 0],
            "hedged": sub[sub.get("valid_conf", pd.Series(dtype=str)) == "Hedged"],
            "unparsed field": sub[sub.parsed_pde.isna()] if "parsed_pde" in sub else sub.iloc[0:0],
        }
        for label, cell in cells.items():
            if not len(cell):
                continue
            idx = [int(i) for i in
                   pd.Series(range(len(cell))).sample(
                       min(per_cell, len(cell)), random_state=20260820).sort_values()]
            for i in idx:
                r = cell.iloc[i]
                text = str(r.get("model_response", ""))
                rows.append({
                    "model": model, "cell": label,
                    "title": str(r.get("title", "")),
                    "gt_sample": str(r.get("gt_sample", "")),
                    "mod_type": str(r.get("mod_type", "")),
                    "source": str(r.get("source", "")),
                    "gt": {k: str(r.get(f"gt_{k}", "")) for k in
                           ("pde", "method", "behavior", "valid")},
                    "parsed": {k: ("" if pd.isna(r.get(f"parsed_{k}")) else
                                   str(r.get(f"parsed_{k}", ""))) for k in
                               ("pde", "method", "behavior", "valid")},
                    "scores": {k: (None if pd.isna(r.get(k)) else round(float(r.get(k)), 3))
                               for k in ("pde_match", "method_recall", "behavior_recall",
                                         "valid_match") if k in r},
                    "axis": str(r.get("method_axis", "")),
                    "conf": str(r.get("valid_conf", "")),
                    "finish": str(r.get("finish_reason", "")),
                    "justification": "",
                    "chars": len(text),
                    # Long reasoning traces are kept whole up to a generous cap; when
                    # a trace is cut, the row says so rather than ending silently.
                    "text": text if len(text) <= max_chars else
                            text[:max_chars] + f"\n\n[... {len(text) - max_chars} more "
                                               f"characters in the stored row; this viewer "
                                               f"caps at {max_chars} for page weight ...]",
                })
    return rows




def response_browser(rows, kind, prefix="rb"):
    """A self-contained reader over sampled raw rows.

    Every id and every JS name is namespaced by `prefix`, and the script body runs
    inside an IIFE. The page carries TWO of these browsers -- one per experiment --
    and the first version declared `const RB` at global scope in both, so the second
    <script> was a duplicate-declaration SyntaxError and never ran, while
    getElementById kept resolving the shared ids to the first browser's DOM. The
    cross-modal panel rendered as an empty shell with no error visible on the page.
    """
    if not rows:
        return placeholder("No rows to display yet.", kind)
    payload = json.dumps(rows).replace("</", "<\\/")
    P = prefix
    return f"""
<div class="browser">
  <div class="controls">
    <select id="{P}-model" onchange="{P}Filter()"></select>
    <select id="{P}-cell" onchange="{P}Filter()"></select>
    <button onclick="{P}Step(-1)">&larr; prev</button>
    <span id="{P}-pos"></span>
    <button onclick="{P}Step(1)">next &rarr;</button>
  </div>
  <div id="{P}-meta" class="rb-meta"></div>
  <div id="{P}-just" class="rb-just"></div>
  <pre id="{P}-text" class="rb-text"></pre>
</div>
<script>
(function() {{
const RB = {payload};
let rbIdx = 0, rbView = RB;
const $ = id => document.getElementById('{P}-' + id);
const esc = s => String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function rbInit() {{
  const ms = [...new Set(RB.map(r=>r.model))].sort();
  const cs = [...new Set(RB.map(r=>r.cell))];
  $('model').innerHTML =
    '<option value="">all models</option>' + ms.map(m=>`<option>${{esc(m)}}</option>`).join('');
  $('cell').innerHTML =
    '<option value="">all outcomes</option>' + cs.map(c=>`<option>${{esc(c)}}</option>`).join('');
  rbFilter();
}}
function rbFilter() {{
  const m = $('model').value, c = $('cell').value;
  rbView = RB.filter(r => (!m || r.model===m) && (!c || r.cell===c));
  rbIdx = 0; rbShow();
}}
function rbStep(d) {{ if(!rbView.length) return;
  rbIdx = (rbIdx + d + rbView.length) % rbView.length; rbShow(); }}
function rbShow() {{
  const t = $('text'), meta = $('meta'), just = $('just');
  $('pos').textContent = rbView.length ? `${{rbIdx+1}} / ${{rbView.length}}` : '0 / 0';
  if(!rbView.length) {{ meta.innerHTML=''; just.innerHTML=''; t.textContent=''; return; }}
  const r = rbView[rbIdx];
  const gt = Object.entries(r.gt).map(([k,v])=>`<b>${{esc(k)}}</b> ${{esc(v)||'&mdash;'}}`).join(' &nbsp;&middot;&nbsp; ');
  const pa = Object.entries(r.parsed).map(([k,v])=>`<b>${{esc(k)}}</b> ${{esc(v)||'<i>null</i>'}}`).join(' &nbsp;&middot;&nbsp; ');
  const sc = Object.entries(r.scores).map(([k,v])=>`${{esc(k)}}=${{v===null?'null':esc(v)}}`).join('  ');
  meta.innerHTML =
    `<div class="rb-line"><span class="rb-tag">${{esc(r.cell)}}</span> ${{esc(r.model)}} &nbsp;&middot;&nbsp; `
    + `${{esc(r.title)}} &nbsp;&middot;&nbsp; ${{esc(r.mod_type)}} &nbsp;&middot;&nbsp; ${{esc(r.source)}} &nbsp;&middot;&nbsp; `
    + `${{esc(r.chars)}} chars &nbsp;&middot;&nbsp; finish=${{esc(r.finish)}}</div>`
    + `<div class="rb-line rb-gt">ground truth &nbsp; ${{gt}}</div>`
    + `<div class="rb-line rb-pa">parsed &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ${{pa}}</div>`
    + `<div class="rb-line rb-sc">${{sc}} &nbsp;&middot;&nbsp; ${{esc(r.axis)}} &nbsp;&middot;&nbsp; ${{esc(r.conf)}}</div>`;
  // The justification is pulled out of the trace on purpose: in the reasoning arms
  // it sits at the end of tens of thousands of characters, and it is the field that
  // shows a right answer reached for the wrong reason.
  just.innerHTML = r.justification
    ? `<div class="rb-jlabel">justification (parsed)</div><div>${{esc(r.justification)}}</div>` : '';
  t.textContent = r.text || '(this row stored no response text)';
}}
window['{P}Filter'] = rbFilter; window['{P}Step'] = rbStep;
rbInit();
}})();
</script>"""


def worked_example_panel(path="data/multimodal_items_rendered_v1.csv",
                         system="Burgers_1", condition="X_C", seed="0"):
    """One real item, shown exactly as the model received it.

    Every number in Part III rests on the claim that the four views are genuinely
    independent descriptions of one system and that only one of them was tampered
    with. That claim is checkable only by reading an actual item, so the report
    carries one rather than asking anyone to take it on trust. The corrupted view is
    marked here for the reader; the model saw no such marking, and no legend beyond
    "View 1..4".
    """
    if not os.path.exists(path):
        return []
    csv.field_size_limit(sys.maxsize)
    found = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("gt_sample") == system and r.get("condition") == condition
                    and str(r.get("order_seed")) == str(seed)):
                found[r["names"]] = {
                    "item_id": r.get("item_id", ""),
                    "pde": r.get("gt_pde_class", ""), "method": r.get("gt_num_method", ""),
                    "views": [{"kind": r[f"view_{i}_kind"],
                               "outlier": r[f"view_{i}_is_outlier"] == "1",
                               "text": r[f"view_{i}_text"]} for i in range(1, 5)],
                }
            if len(found) == 2:
                break
    if not found:
        return []
    payload = json.dumps(found).replace("</", "<\\/")
    kinds = [v["kind"] for v in next(iter(found.values()))["views"]]
    outlier_kind = next((v["kind"] for v in next(iter(found.values()))["views"]
                         if v["outlier"]), "?")

    body = f"""
<div class="controls">
  <select id="we-names" onchange="weShow()">
    <option value="real">real identifiers</option>
    <option value="obfuscated">obfuscated identifiers</option>
  </select>
  {''.join(f'<button class="we-tab" id="we-tab-{i}" onclick="weTab({i})">View {i+1}</button>'
           for i in range(4))}
</div>
<div id="we-meta" class="rb-meta"></div>
<pre id="we-text" class="rb-text"></pre>
<script>
(function() {{
const WE = {payload};
let weIdx = 0;
function weTab(i) {{ weIdx = i; weShow(); }}
function weShow() {{
  const n = document.getElementById('we-names').value;
  const item = WE[n] || WE[Object.keys(WE)[0]];
  const v = item.views[weIdx];
  for (let i = 0; i < 4; i++) {{
    const b = document.getElementById('we-tab-' + i);
    b.classList.toggle('active', i === weIdx);
    b.classList.toggle('is-outlier', item.views[i].outlier);
  }}
  document.getElementById('we-meta').innerHTML =
    `<div class="rb-line"><span class="rb-tag">View ${{weIdx + 1}} of 4</span> `
    + `this slot holds the <b>${{v.kind}}</b> view &nbsp;·&nbsp; ${{v.text.length}} chars</div>`
    + (v.outlier
        ? `<div class="rb-line" style="color:#e69090">THIS IS THE CORRUPTED VIEW &mdash; `
          + `swapped for the invalid solver's version. The model was not told this.</div>`
        : `<div class="rb-line rb-gt">uncorrupted &mdash; agrees with the other two `
          + `uncorrupted views</div>`);
  document.getElementById('we-text').textContent = v.text;
}}
window.weTab = weTab; window.weShow = weShow;
weShow();
}})();
</script>"""
    return [(
        "One item, exactly as the model saw it",
        f"System <code>{system}</code>, condition <code>{condition}</code>. The four "
        f"views arrive in a randomized order behind a neutral legend &mdash; in this "
        f"item the slots hold <b>{', '.join(kinds)}</b> &mdash; and the model is told "
        f"only that at most one of them may not belong. <b>The corrupted view here is "
        f"the {outlier_kind}</b>, marked in red for you and marked in no way at all for "
        f"the model.<br><br>"
        f"Ground truth: <b>{next(iter(found.values()))['pde']}</b>, "
        f"<b>{next(iter(found.values()))['method']}</b>. Switch the dropdown to see the "
        f"identifier-obfuscation condition &mdash; note that only the code view's text "
        f"changes, which is what makes it so striking that obfuscation degrades "
        f"detection of the other three views too.<br><br>"
        f"Read the corrupted solver against the equation and the trajectory: the "
        f"difference from the valid version is usually a single line. That is the task, "
        f"and it is why the code view is the hardest of the four to catch.",
        body)]


# ── Which representation carries weight? ──────────────────────────────────────
# The design gives every item a 3-against-1 vote: three views agree, one dissents.
# So each view's standing can be read two ways, and they are not the same thing:
#
#   how readily is it SUSPECTED   -- how often the model accuses it on A0, where
#                                    nothing is wrong and the accusation is therefore
#                                    pure prior, uninformed by any evidence
#   how reliably is it CHECKED    -- how often the model names it when it really is
#                                    the dissenter and the other three outvote it
#
# A view the model treats as authoritative is rarely suspected. A view it can
# actually verify is reliably caught. Those come apart badly, and the gap is the
# result.
TRUST_VIEWS = ["math", "description", "trajectory", "code"]
VIEW_COLOUR = {"code": "#e8a25a", "trajectory": "#7eb8ff",
               "math": "#8fd694", "description": "#d98fd6"}


def trust_panels(d):
    if d is None or d.empty or "blamed" not in d:
        return []
    arms = sorted(d["arm"].unique())
    panels = []

    # ── 1. suspicion vs verification, per view ───────────────────────────────
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15,
                        subplot_titles=(
                            "accused when NOTHING is wrong  (pure prior)",
                            "correctly named when it IS the dissenter"))
    for i, v in enumerate(TRUST_VIEWS):
        a0 = [100 * (d[(d.arm == a) & (d.condition == "A0")].blamed == v).mean()
              for a in arms]
        caught = [100 * (d[(d.arm == a) & (d.corrupted_view == v)].blamed == v).mean()
                  for a in arms]
        fig.add_bar(x=arms, y=a0, name=v, marker_color=VIEW_COLOUR[v],
                    legendgroup=v, row=1, col=1)
        fig.add_bar(x=arms, y=caught, name=v, marker_color=VIEW_COLOUR[v],
                    legendgroup=v, showlegend=False, row=1, col=2)
    fig.update_yaxes(title_text="% of A0 items", row=1, col=1)
    fig.update_yaxes(title_text="% of items where this view was corrupted", row=1, col=2)
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.3))
    panels.append((
        "Which representation does the model trust?",
        "Each item is a three-against-one vote, so a view's standing shows up twice.<br><br>"
        "<b>Left &mdash; how readily it is suspected.</b> On A0 all four views genuinely "
        "agree, so any accusation is pure prior. <b>The equation is accused on 0.8&ndash;"
        "1.6% of items and the description on 0&ndash;4%. The code is accused on "
        "6&ndash;33%.</b> The model treats the governing equation and the prose as very "
        "nearly beyond question, and the solver as the natural suspect.<br><br>"
        "<b>Right &mdash; how reliably it is actually checked.</b> This does not follow "
        "the same order. The equation is both the most trusted and among the best "
        "caught when it really is wrong; the code is the least trusted and the worst "
        "caught. So the low standing of the code is not the model knowing something "
        "about it &mdash; the model suspects the code most and verifies it least.<br><br>"
        "Being cheap to check and being treated as authoritative are different "
        "properties, and for the equation they happen to coincide: comparing a written "
        "PDE against a prose description is a text-to-text comparison, while checking a "
        "solver against a trajectory means running it in your head.",
        fig_html(fig, height=500, margin=dict(l=65, r=30, t=60, b=140))))

    # ── 2. where the blame actually flows ────────────────────────────────────
    mis = d[(d.agree == "no") & d.blamed.notna()
            & (d.blamed != d.corrupted_view) & (d.corrupted_view != "none")]
    if len(mis):
        order = ["code", "trajectory", "description", "math"]
        z, txt = [], []
        for v in order:
            s = mis[mis.corrupted_view == v]
            share = s.blamed.value_counts(normalize=True)
            z.append([share.get(x, 0) for x in order])
            txt.append([f"{share.get(x, 0):.0%}" if x != v else "—" for x in order])
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"blamed the {x}" for x in order],
            y=[f"{v} was corrupted" for v in order],
            colorscale="Blues", zmin=0, zmax=1, text=txt, texttemplate="%{text}",
            showscale=False))
        fig.update_layout(height=400)
        tot = mis.blamed.value_counts(normalize=True)
        panels.append((
            "When it blames the wrong view, it blames the code",
            "Only the items where the model flagged a disagreement and then named the "
            "wrong view. Rows are what was actually corrupted; the diagonal is empty by "
            "construction.<br><br>"
            f"<b>Blame flows almost entirely along one axis: code &harr; trajectory.</b> "
            f"When the <b>trajectory</b> is corrupted and the model misattributes, it "
            f"blames the <b>code {100 * z[order.index('trajectory')][0]:.0f}%</b> of the "
            f"time. When the <b>code</b> is corrupted and it misattributes, it blames the "
            f"<b>trajectory {100 * z[0][order.index('trajectory')]:.0f}%</b> of the time. "
            f"Across every misattribution in the experiment, "
            f"{100 * tot.get('code', 0):.0f}% land on the code and "
            f"{100 * tot.get('trajectory', 0):.0f}% on the trajectory &mdash; leaving "
            f"{100 * (tot.get('math', 0) + tot.get('description', 0)):.0f}% for the "
            "equation and the description combined.<br><br>"
            "<b>This is the clearest statement of what the model's world model is "
            "doing.</b> It knows the trajectory is supposed to be what the code "
            "produces, so when the numbers and the program disagree it correctly "
            "concludes that one of those two is at fault &mdash; and then cannot tell "
            "which end is broken. Meanwhile the equation and the description are treated "
            "as axioms rather than as evidence that could itself be wrong. The model has "
            "the causal structure and not the ability to adjudicate inside it.",
            fig_html(fig, height=420, margin=dict(l=170, r=30, t=30, b=90))))

    # ── 3. the trajectory question, asked directly ───────────────────────────
    rungs = ["T_rand", "T_shuf", "T_swap", "T_exec"]
    rung_label = {"T_rand": "T_rand<br>random numbers",
                  "T_shuf": "T_shuf<br>own values, shuffled",
                  "T_swap": "T_swap<br>another system's",
                  "T_exec": "T_exec<br>invalid solver's output"}
    if (d.traj_level.isin(rungs)).any():
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                            subplot_titles=("flagged that something disagrees",
                                            "and correctly named the trajectory"))
        xs = [rung_label[r] for r in rungs]
        for i, a in enumerate(arms):
            sub = d[(d.arm == a) & (d.corrupted_view == "trajectory")]
            fig.add_scatter(x=xs, name=a, legendgroup=a, mode="lines+markers",
                            marker=dict(size=9), line=dict(color=ACCENT[i % len(ACCENT)]),
                            y=[100 * (sub[sub.traj_level == r].agree == "no").mean()
                               for r in rungs], row=1, col=1)
            fig.add_scatter(x=xs, name=a, legendgroup=a, showlegend=False,
                            mode="lines+markers", marker=dict(size=9),
                            line=dict(color=ACCENT[i % len(ACCENT)]),
                            y=[100 * (sub[sub.traj_level == r].blamed == "trajectory").mean()
                               for r in rungs], row=1, col=2)
        fig.update_yaxes(range=[0, 100], title_text="% of items", row=1, col=1)
        fig.update_yaxes(range=[0, 100], row=1, col=2)
        fig.update_layout(legend=dict(orientation="h", y=-0.32))
        panels.append((
            "If the trajectory is wrong, does it say so?",
            "No &mdash; not reliably, and the gap between the two panels is where the "
            "answer lives. Left is noticing that something disagrees; right is naming "
            "the trajectory as the thing that disagrees. The four rungs run from "
            "structurally empty to physically subtle: random numbers of the right shape, "
            "the trajectory's own values permuted, another system's trajectory, and the "
            "output the invalid solver actually produced.<br><br>"
            "<b>Qwen3-32B without thinking flags shape-matched random numbers on 28% of "
            "items and names the trajectory on 8.6%.</b> Even QwQ-32B, which flags "
            "T_exec on 99% of items, names the trajectory on only 50% &mdash; the rest "
            "of the time it has noticed the disagreement and blamed the solver.<br><br>"
            "The two panels also come apart in a way worth noticing: T_exec is flagged "
            "the most often but is <i>not</i> the best-attributed rung for two of the "
            "arms. A trajectory full of NaNs is unmistakably wrong; deciding that the "
            "<i>trajectory</i> rather than the <i>code</i> is the thing to blame for it "
            "is a separate judgement, and one they get wrong.",
            fig_html(fig, height=480, margin=dict(l=65, r=30, t=60, b=150))))
    return panels


# ── The research question, answered directly ──────────────────────────────────
# "When independent representations of the same physical system disagree, can the
#  model detect the disagreement and identify which representation disagrees? Does
#  that judgement track physics or lexical cues?"
#
# These panels lead Part III because everything after them is supporting detail.
# The organising idea is that a raw flag rate cannot answer any part of it: the
# model's willingness to cry foul has to be measured against what it does when
# nothing is wrong, which is the entire reason the A0 control exists.
RQ_VIEWS = ["code", "math", "description", "trajectory"]


def _blamed_view(r):
    """Resolve the model's `view_N` answer back to the representation it names.

    Slot order is randomized per item, so `view_2` means nothing until it is read
    through that item's own legend.
    """
    try:
        i = int(str(r["outlier"]).replace("view_", "")) - 1
        slots = r["slots"]
        return slots[i] if slots is not None and 0 <= i < len(slots) else None
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def rq_panels(df):
    if df is None or df.empty or "condition" not in df:
        return []
    d = df.copy()
    d["arm"] = (d["model"].astype(str).str.split("/").str[-1] + " · think "
                + d.get("thinking", pd.Series("na", index=d.index))
                   .astype(str).str.replace("think_", "", regex=False))
    d["blamed"] = d.apply(_blamed_view, axis=1)
    arms = sorted(d["arm"].unique())
    panels = []

    # ── 1. the question in the user's own words ───────────────────────────────
    xc = d[d.condition == "X_C"]
    if len(xc):
        outcomes = [
            ("thinks everything agrees", "#c4574d",
             lambda s: (s.agree == "yes").mean()),
            ("flags it, blames an innocent view", "#e8c35a",
             lambda s: ((s.agree == "no") & s.blamed.notna() & (s.blamed != "code")).mean()),
            ("flags it and blames the code", "#4fa96a",
             lambda s: ((s.agree == "no") & (s.blamed == "code")).mean()),
        ]
        fig = go.Figure()
        for label, colour, fn in outcomes:
            fig.add_bar(name=label, x=arms, y=[100 * fn(xc[xc.arm == a]) for a in arms],
                        marker_color=colour,
                        text=[f"{100 * fn(xc[xc.arm == a]):.0f}%" for a in arms],
                        textposition="inside")
        fig.update_layout(barmode="stack", yaxis_title="% of X_C items",
                          yaxis_range=[0, 100], xaxis_tickangle=-15,
                          legend=dict(orientation="h", y=-0.22))
        worst = max(arms, key=lambda a: (xc[xc.arm == a].agree == "yes").mean())
        worst_pct = 100 * (xc[xc.arm == worst].agree == "yes").mean()
        panels.append((
            "When the code is wrong, does the model notice?",
            "The sharpest form of the question. These are the <b>X_C</b> items only: "
            "the solver has been swapped for its invalid counterpart while the "
            "equation, the description and the trajectory all still describe the real "
            "system. Three views agree with each other and the code contradicts them. "
            "Red is the model reporting that everything is consistent.<br><br>"
            f"<b>{worst.split(' ·')[0]} says everything is fine on "
            f"{worst_pct:.0f}% of them.</b> Two of the four arms miss more than two "
            "thirds. The corruption is usually a single line &mdash; a flipped flux "
            "direction, a wrong power of dx &mdash; inside a program that still looks "
            "entirely reasonable, and the three other views are right there "
            "contradicting it.",
            fig_html(fig, height=470, margin=dict(l=65, r=30, t=30, b=110))))

    # ── 2. signal vs policy — the honest version of the detection rate ────────
    fig = go.Figure()
    base = {a: (d[(d.arm == a) & (d.condition == "A0")].agree == "no").mean()
            for a in arms}
    for i, v in enumerate(RQ_VIEWS):
        lifts = []
        for a in arms:
            sub = d[(d.arm == a) & (d.corrupted_view == v)]
            lifts.append(100 * ((sub.agree == "no").mean() - base[a]) if len(sub) else None)
        fig.add_bar(name=v, x=arms, y=lifts, marker_color=ACCENT[i])
    fig.add_hline(y=0, line=dict(color="#e67e8f", width=1.5, dash="dash"),
                  annotation_text="no signal: flags corrupted items no more often than clean ones",
                  annotation_font=dict(color="#e67e8f", size=10))
    fig.update_layout(barmode="group", xaxis_tickangle=-15,
                      yaxis_title="extra flagging vs the model's own clean-item rate (pp)",
                      legend=dict(orientation="h", y=-0.22))
    base_txt = " &nbsp;·&nbsp; ".join(f"{a.split(' ·')[0]} {100 * b:.0f}%"
                                      for a, b in base.items())
    panels.append((
        "Is it detecting anything, or just flagging everything?",
        "A raw detection rate cannot answer the research question, because seven of "
        "eight conditions are corrupted &mdash; a model that always says &quot;they "
        "disagree&quot; scores 87.5%. What matters is whether it flags a corrupted "
        "item <i>more often than it flags a clean one</i>. This chart is that "
        "difference: detection rate on each corrupted view minus the same model's "
        "false-alarm rate on A0, where nothing is wrong.<br><br>"
        f"<b>False-alarm rates on A0:</b> {base_txt}. QwQ-32B flags most clean items "
        "too, which is why its high raw detection numbers overstate it so badly.<br><br>"
        "<b>The code bar is the shortest in every arm.</b> For DeepSeek-R1 it is "
        "roughly two points &mdash; corrupted code is very nearly invisible to it. "
        "Corrupted equations, by contrast, lift flagging by 23 to 39 points in every "
        "arm. Whatever the models are checking, it is not the solver.",
        fig_html(fig, height=500, margin=dict(l=75, r=30, t=30, b=110))))

    # ── 3. the contradiction: most-blamed and least-detected are the same view ─
    a0 = d[(d.condition == "A0") & (d.agree == "no") & d.blamed.notna()]
    if len(a0) and len(xc):
        fig = make_subplots(
            rows=1, cols=2, horizontal_spacing=0.16,
            subplot_titles=("nothing is wrong — what does it blame anyway?",
                            "the code IS wrong — does it say so?"))
        for i, v in enumerate(RQ_VIEWS):
            fig.add_bar(x=arms, name=v, marker_color=ACCENT[i], legendgroup=v,
                        y=[100 * (a0[a0.arm == a].blamed == v).mean() if len(a0[a0.arm == a])
                           else 0 for a in arms], row=1, col=1)
        fig.add_bar(x=arms, name="blames the code, correctly", marker_color="#4fa96a",
                    showlegend=True,
                    y=[100 * ((xc[xc.arm == a].agree == "no")
                              & (xc[xc.arm == a].blamed == "code")).mean() for a in arms],
                    row=1, col=2)
        fig.update_yaxes(title_text="% of false alarms", row=1, col=1)
        fig.update_yaxes(title_text="% of X_C items", range=[0, 100], row=1, col=2)
        fig.update_xaxes(tickangle=-20)
        fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.28))
        share = {a: 100 * (a0[a0.arm == a].blamed == "code").mean() for a in arms
                 if len(a0[a0.arm == a])}
        panels.append((
            "The contradiction: code is the first thing blamed and the last thing caught",
            "Left: the items where all four views genuinely agree, restricted to the "
            "ones the model wrongly flagged, broken down by what it accused. Right: "
            "how often it correctly blames the code when the code really is the "
            "corrupted view.<br><br>"
            f"<b>Code takes {min(share.values()):.0f}&ndash;{max(share.values()):.0f}% "
            "of the blame when nothing is wrong at all</b> &mdash; it is the default "
            "suspect for every arm. Yet it is the view models are least able to catch "
            "when it genuinely is the odd one out.<br><br>"
            "<b>The sharpest way to state it.</b> Take how often a model accuses a view "
            "when that view really is corrupted, and divide by how often it accuses the "
            "same view when nothing is corrupted. That ratio is how much the evidence "
            "moves the accusation. For the <b>equation</b> it is <b>37&times; to "
            "81&times;</b>. For the <b>code</b> it is <b>1.5&times; to 3.0&times;</b>. "
            "Corrupting the code barely changes how often the code gets blamed &mdash; "
            "the accusation was already going to be made. A judgement that accuses the code "
            "most often when there is nothing to find, and least often when there is, "
            "is not being driven by the evidence in front of it. This is the clearest "
            "sign in the experiment that the flag is a prior about code being "
            "untrustworthy rather than a comparison across representations.",
            fig_html(fig, height=520, margin=dict(l=65, r=30, t=60, b=140))))

    # ── 4. is it reading the views, or their positions? ───────────────────────
    cor = d[(d.corrupted_view != "none") & d.outlier.notna() & (d.agree == "no")]
    if len(cor):
        slots = ["view_1", "view_2", "view_3", "view_4"]
        fig = go.Figure()
        for i, sl in enumerate(slots):
            fig.add_bar(name=sl, x=arms, marker_color=ACCENT[i],
                        y=[100 * (cor[cor.arm == a].outlier == sl).mean() for a in arms])
        fig.add_hline(y=25, line=dict(color="#e67e8f", dash="dash"),
                      annotation_text="25% — what a position-blind model would give each slot",
                      annotation_font=dict(color="#e67e8f", size=10))
        fig.update_layout(barmode="group", yaxis_title="% of accusations",
                          xaxis_tickangle=-15, legend=dict(orientation="h", y=-0.22))
        panels.append((
            "Is it comparing representations, or just picking a slot?",
            "Slot order is randomized per item and the views are presented behind a "
            "neutral legend, so the corrupted view lands in each of the four positions "
            "equally often &mdash; in this run 24.8 / 25.8 / 23.4 / 26.0%. A model "
            "reasoning about content should therefore accuse the four <i>positions</i> "
            "at roughly 25% each, and any departure is a positional habit rather than "
            "a judgement about physics.<br><br>"
            "<b>Every arm over-accuses the last slot.</b> Qwen3-32B without thinking "
            "is the extreme case: 38% of its accusations land on view_4 and 15% on "
            "view_1, against a truth that is essentially uniform. Part of what looks "
            "like cross-modal reasoning is recency.<br><br>"
            "The cleaner statistic is the rate of accusing slot <i>k</i> on items where "
            "<i>k</i> is <b>not</b> the corrupted view, which removes any confound with "
            "what happened to be corrupted where. It rises monotonically with position "
            "for both Qwen3 arms (0.019 / 0.040 / 0.060 / 0.094 without thinking) and "
            "for DeepSeek-R1. QwQ-32B is the exception: its false accusations are high "
            "everywhere and not ordered by position (0.100 / 0.115 / 0.080 / 0.135), "
            "consistent with it accusing near-indiscriminately rather than having a "
            "positional habit.",
            fig_html(fig, height=470, margin=dict(l=65, r=30, t=30, b=110))))

    # ── 5. the world-model dissociation ──────────────────────────────────────
    if "pde_class_match" in d and d.pde_class_match.notna().any():
        conds = ["A0", "X_C", "X_M", "X_D",
                 "X_T_rand", "X_T_shuf", "X_T_swap", "X_T_exec"]
        conds = [c for c in conds if (d.condition == c).any()]
        fig = go.Figure()
        for i, a in enumerate(arms):
            sub = d[(d.arm == a) & d.pde_class_match.notna()]
            fig.add_scatter(name=a, x=conds, mode="lines+markers",
                            marker=dict(size=8), line=dict(color=ACCENT[i % len(ACCENT)]),
                            y=[sub[sub.condition == c]["pde_class_match"].mean()
                               for c in conds])
        fig.update_layout(yaxis_title="names the PDE class correctly",
                          yaxis_range=[0, 1], xaxis_tickangle=-25,
                          legend=dict(orientation="h", y=-0.3))
        panels.append((
            "What it knows vs what it can check",
            "The same responses also name the physical system. This asks whether that "
            "identification survives having one of the four representations corrupted "
            "&mdash; and it is flat. Every arm names the PDE class about as well when "
            "the code is wrong, or the equation, or the trajectory, as it does when "
            "all four views agree.<br><br>"
            "So the model holds a stable read of <i>which system this is</i>, "
            "recovered from the representations that agree, at the same time as it "
            "fails to report that one of them dissents. Identification is robust; "
            "disagreement-detection is not. If cross-modal consistency is a probe of "
            "an underlying world model, this is the shape of the answer: enough "
            "representation to identify the system, not enough to audit it.",
            fig_html(fig, height=490, margin=dict(l=65, r=30, t=30, b=130))))
    panels += obfuscation_panels(d)
    return panels


def crossmodal_frame(rows):
    """The one prepared frame both the lead and appendix panels read from."""
    if not rows:
        return None
    d = pd.DataFrame(rows)
    if d.empty or "condition" not in d:
        return None
    d = d.copy()
    d["arm"] = (d["model"].astype(str).str.split("/").str[-1]
                .str.replace("-Distill-Qwen-32B", "", regex=False) + " · think "
                + d.get("thinking", pd.Series("na", index=d.index))
                   .astype(str).str.replace("think_", "", regex=False))
    d["blamed"] = d.apply(_blamed_view, axis=1)
    return d


def _mcnemar_p(b, c):
    """Exact two-sided McNemar on the discordant pairs.

    The correct test for a paired binary outcome, and the off-diagonal counts ARE
    the test: b items the model got right with real names and wrong when they were
    obfuscated, c the other way. An unpaired two-proportion test on the same data
    throws away the pairing and overstates its uncertainty.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def obfuscation_panels(d):
    """Physics or lexical cues, analysed as the paired comparison it actually is.

    Every item exists twice -- once with the solver's real identifiers, once
    obfuscated -- matched on system, condition, slot order, model and arm. Comparing
    the two marginal rates throws that pairing away. Worse, the marginal delta hides
    offsetting flips: a model can lose one item and gain another and look unchanged
    while being unreliable on both. `consistency` (correct under BOTH namings) is the
    number that exposes it, following the contrast-set convention.
    """
    key = ["arm", "gt_sample", "condition", "order_seed"]
    if not set(key + ["names"]).issubset(d.columns):
        return []
    p = d.pivot_table(index=key, columns="names", values="detection_correct",
                      aggfunc="first").dropna()
    if p.empty or "real" not in p or "obfuscated" not in p:
        return []
    panels = []

    recs = []
    for arm, s in p.groupby(level=0):
        r, o = s["real"].astype(int), s["obfuscated"].astype(int)
        b = int(((r == 1) & (o == 0)).sum())
        c = int(((r == 0) & (o == 1)).sum())
        recs.append({"arm": arm, "real": r.mean(), "obf": o.mean(),
                     "both": ((r == 1) & (o == 1)).mean(),
                     "b": b, "c": c, "n": len(s), "p": _mcnemar_p(b, c)})

    # ── identity-line scatter: the field's idiom for a paired condition change ──
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                    line=dict(color="#5a6580", dash="dash"),
                    hovertemplate="no effect of obfuscation<extra></extra>")
    fig.add_annotation(x=0.72, y=0.78, text="line of no effect", showarrow=False,
                       textangle=-45, font=dict(color="#7b88a4", size=10))
    for i, r in enumerate(recs):
        sig = "" if r["p"] >= 0.05 else "  ✱"
        fig.add_scatter(x=[r["real"]], y=[r["obf"]], mode="markers+text",
                        name=r["arm"], marker=dict(size=15, color=ACCENT[i % len(ACCENT)]),
                        text=[f"  {r['arm'].split('·')[0][:14]}{sig}"], textposition="middle right",
                        textfont=dict(size=10, color=ACCENT[i % len(ACCENT)]))
    fig.update_layout(xaxis_title="detection rate, real identifiers",
                      yaxis_title="detection rate, obfuscated identifiers",
                      xaxis_range=[0, 1.15], yaxis_range=[0, 1], showlegend=False)

    body = "".join(
        f"<tr><td>{r['arm']}</td><td class='num'>{r['n']}</td>"
        f"<td class='num'>{r['real']:.3f}</td>"
        f"<td class='num'>{r['obf']:.3f} <span class='{'low' if r['obf'] < r['real'] else 'high'}'>"
        f"({r['obf'] - r['real']:+.3f})</span></td>"
        f"<td class='num'>{r['both']:.3f}</td>"
        f"<td class='num'>{r['b']} / {r['c']}</td>"
        f"<td class='num'>{'<b>' if r['p'] < 0.05 else ''}{r['p']:.4f}"
        f"{'</b>' if r['p'] < 0.05 else ''}</td></tr>" for r in recs)
    panels.append((
        "Physics or lexical cues? The paired obfuscation test",
        "Every item was run twice &mdash; once with the solver's real variable and "
        "function names, once with those names replaced by uninformative ones &mdash; "
        "matched on system, condition, slot order and model. If the judgement tracked "
        "physics, the points would sit on the dashed line. <b>✱ marks a model whose "
        "change is significant by an exact McNemar test on the paired items.</b><br><br>"
        "<b>Read the consistency column, not the delta.</b> Qwen3-32B with thinking "
        "goes 0.671 &rarr; 0.606, which looks like a six-point scratch; but it is "
        "correct under <i>both</i> namings on only 0.500 of items, because it loses 86 "
        "items and gains 53 different ones. The marginal rates cancel those flips out "
        "and understate the instability badly.<br><br>"
        "The two arms that move significantly are the two Qwen3-32B arms. QwQ and "
        "DeepSeek do not move &mdash; but neither is discriminating much to begin "
        "with, so there is little for obfuscation to remove.",
        fig_html(fig, height=470, margin=dict(l=70, r=40, t=30, b=60))
        + f'<table class="tbl" style="margin-top:16px"><tr><th>model · arm</th>'
          f'<th>paired items</th><th>real names</th><th>obfuscated (Δ)</th>'
          f'<th>consistency<br><span class="thsub">correct under BOTH</span></th>'
          f'<th>flips<br><span class="thsub">real✓obf✗ / real✗obf✓</span></th>'
          f'<th>McNemar p</th></tr>{body}</table>'))

    # ── per-view, signed, zero-centred; negatives are real and must be visible ──
    p2 = d[d.corrupted_view != "none"].pivot_table(
        index=key + ["corrupted_view"], columns="names",
        values="detection_correct", aggfunc="first").dropna()
    if not p2.empty:
        views = ["code", "math", "description", "trajectory"]
        arms = sorted({i[0] for i in p2.index})
        fig = go.Figure()
        for i, v in enumerate(views):
            ys, txt = [], []
            for a in arms:
                s = p2[(p2.index.get_level_values(0) == a)
                       & (p2.index.get_level_values(4) == v)]
                if not len(s):
                    ys.append(None); txt.append(""); continue
                r, o = s["real"].astype(int), s["obfuscated"].astype(int)
                ys.append(100 * (o.mean() - r.mean()))
                pv = _mcnemar_p(int(((r == 1) & (o == 0)).sum()),
                                int(((r == 0) & (o == 1)).sum()))
                txt.append("✱" if pv < 0.05 else "")
            fig.add_bar(name=v, x=arms, y=ys, marker_color=ACCENT[i],
                        text=txt, textposition="outside")
        fig.add_hline(y=0, line=dict(color="#5a6580", width=1.5))
        fig.update_layout(barmode="group", xaxis_tickangle=-15,
                          yaxis_title="Δ detection, obfuscated − real (pp)",
                          legend=dict(orientation="h", y=-0.24))
        panels.append((
            "Which view does obfuscation damage?",
            "The same paired comparison, split by which view was the corrupted one. "
            "Bars are signed and the axis is centred on zero, so an improvement would "
            "be drawn rather than clipped &mdash; obfuscation genuinely helps "
            "occasionally, and a chart that cannot show that is misleading. ✱ marks "
            "significance by McNemar within that cell.<br><br>"
            "<b>Correction &mdash; read this together with the next panel.</b> The raw "
            "rates do fall on almost every view, and an earlier version of this report "
            "read that as obfuscation degrading the models' ability to discriminate. "
            "That reading does not survive checking it against the A0 control: "
            "obfuscation also makes three of the four arms flag <i>less of everything</i>, "
            "including items where nothing is wrong. Their false-alarm rates fall by "
            "6&ndash;9 points. A large part of the drop shown here is that shift in "
            "willingness to commit, not a loss of discrimination.<br><br>"
            "The largest raw losses for Qwen3-32B with thinking are on <b>math</b> "
            "(−14.5pp, p=0.049) and <b>trajectory</b> (−8.3pp, p=0.011), while code "
            "itself barely moves (−3.2pp, p=0.80) &mdash; it was already near the floor, "
            "so there was little left to lose.",
            fig_html(fig, height=470, margin=dict(l=70, r=30, t=30, b=110))))

        # Lift = detection on that view minus the SAME naming condition's own
        # false-alarm rate. Comparing raw rates across namings silently compares two
        # different response criteria.
        fig2 = go.Figure()
        for i, v in enumerate(views):
            ys = []
            for a in arms:
                cell = []
                for nm in ("real", "obfuscated"):
                    sub = d[(d.arm == a) & (d.names == nm)]
                    fa = (sub[sub.condition == "A0"].agree == "no").mean()
                    hit = (sub[sub.corrupted_view == v].agree == "no").mean()
                    cell.append(hit - fa)
                ys.append(100 * (cell[1] - cell[0]))
            fig2.add_bar(name=v, x=arms, y=ys, marker_color=ACCENT[i])
        fig2.add_hline(y=0, line=dict(color="#5a6580", width=1.5))
        fig2.update_layout(barmode="group", xaxis_tickangle=-15,
                           yaxis_title="change in lift over own false-alarm rate (pp)",
                           legend=dict(orientation="h", y=-0.24))
        panels.append((
            "Or does it just stop committing? Obfuscation with the false alarms removed",
            "The same paired comparison, but each bar is now detection on that view "
            "<i>minus that arm's own false-alarm rate under the same naming</i>. If "
            "obfuscation destroyed the models' ability to discriminate, these bars "
            "would shrink toward zero. Mostly they do not.<br><br>"
            "<b>What obfuscation actually does to three of the four arms is make them "
            "reluctant to flag anything.</b> False-alarm rates: DeepSeek-R1 0.317 → "
            "0.246, Qwen3-32B off 0.156 → 0.094, Qwen3-32B on 0.422 → 0.333. Once that "
            "is subtracted, several cells get <i>better</i> under obfuscation &mdash; "
            "Qwen3-32B on goes from +5.4 to +11.1 points on code, DeepSeek from +1.0 to "
            "+3.5. QwQ-32B is the exception in both directions: its false-alarm rate "
            "<i>rises</i> (0.694 → 0.726) and its lift falls on every view, which is the "
            "one arm where a genuine loss of discrimination is the better reading.<br><br>"
            "This matters for the research question. Stripping meaningful identifiers "
            "does not mainly remove the models' grip on the physics &mdash; it moves how "
            "readily they will accuse anything. That is a shift in a <i>prior</i>, which "
            "is the same mechanism the blame-lift panel points at.",
            fig_html(fig2, height=470, margin=dict(l=70, r=30, t=30, b=110))))
    return panels


# ── Part III panels ───────────────────────────────────────────────────────────
def exp3_panels(rows, summary):
    panels = []
    if not rows:
        return [("Part III — cross-modal consistency",
                 "Four representations of one physical system, one corrupted. Detect "
                 "the disagreement, name the outlier, identify the system.",
                 placeholder("No cross-modal rows yet.",
                             "sbatch/run_cross_modal_consistency.sbatch"))]

    df = pd.DataFrame(rows)
    conds = [c for c in ["A0", "X_C", "X_D", "X_M", "X_T_rand", "X_T_shuf",
                         "X_T_swap", "X_T_exec"] if c in set(df["condition"])]

    # detection by condition, per arm
    fig = go.Figure()
    for i, (model, arm) in enumerate(sorted({(r["model"], r.get("thinking", "na"))
                                             for r in rows})):
        sub = df[(df.model == model) & (df.get("thinking", "na") == arm)]
        fig.add_bar(name=f"{model.split('/')[-1]} · think {arm}", x=conds,
                    y=[sub[sub.condition == c]["detection_correct"].mean() for c in conds],
                    marker_color=ACCENT[i % len(ACCENT)])
    fig.update_layout(barmode="group", yaxis_title="detection accuracy")
    panels.append((
        "The raw numbers &mdash; how often it said &quot;these disagree&quot;, per condition",
        "<b>What each bar is.</b> One bar per condition, per model-arm. The height is "
        "the fraction of those items the model got <i>right</i> on the yes/no question "
        "&mdash; and because the correct answer differs between A0 and the rest, the "
        "two halves of this chart mean different things.<br><br>"
        "<b>On the seven corrupted conditions</b> (everything except A0) exactly one "
        "view was tampered with, so the correct answer is &quot;they disagree&quot;. A "
        "tall bar means it caught the corruption. See the decoder in 2.2 for what each "
        "code swaps in.<br><br>"
        "<b>On A0 nothing was corrupted</b>, so the correct answer is &quot;they "
        "agree&quot;. A tall A0 bar means the model correctly kept quiet; a short one "
        "means it cried foul over four views that genuinely matched. <b>A0 is where "
        "the false-alarm rate comes from</b>, and it is the only condition that "
        "supplies it.<br><br>"
        "<b>Why you cannot read this chart alone.</b> Seven of eight conditions are "
        "corrupted, so a model that answers &quot;they disagree&quot; every time gets "
        "seven bars at 1.0, an A0 bar at 0.0, and 87.5% overall while knowing nothing. "
        "QwQ-32B is close to exactly that. The next panels take it apart: 2.4 asks "
        "which <i>view</i> is catchable, 2.5 whether it can then name it, and 2.9 "
        "subtracts the A0 guessing to leave the actual skill.",
        fig_html(fig, False)))

    # ── Which REPRESENTATION is detectable — the research question itself ──────
    VIEWS = ["code", "description", "math", "trajectory"]
    # The trajectory used to appear as ONE pooled bar beside the other three, which
    # buried the whole point of the ladder: T_rand and T_exec are not the same task,
    # and averaging them makes the trajectory look uniformly easy. Each rung gets its
    # own category, so "code is hardest" is a claim against every rung individually
    # rather than against their mean.
    ROWS_BY_VIEW = ([("code", "code", None), ("description", "description", None),
                     ("math", "math", None)]
                    + [(f"trajectory · {lvl}", "trajectory", lvl)
                       for lvl in ("T_rand", "T_shuf", "T_swap", "T_exec")])
    XCATS = [lab for lab, _, _ in ROWS_BY_VIEW]

    def _view_cell(sub, view, level, col):
        cell = sub[(sub.corrupted_view == view) & sub[col].notna()]
        if level is not None:
            cell = cell[cell.traj_level == level]
        return cell[col].mean() if len(cell) else None

    fig = go.Figure()
    for i, (model, arm) in enumerate(sorted({(r["model"], r.get("thinking", "na"))
                                             for r in rows})):
        sub = df[(df.model == model) & (df.get("thinking", "na") == arm)]
        fig.add_bar(name=f"{model.split('/')[-1]} · think {arm}", x=XCATS,
                    y=[_view_cell(sub, v, lv, "detection_correct")
                       for _, v, lv in ROWS_BY_VIEW],
                    marker_color=ACCENT[i % len(ACCENT)])
    fig.add_vline(x=2.5, line=dict(color="#3a4258", width=1, dash="dot"))
    fig.update_layout(barmode="group", yaxis_title="detection rate",
                      yaxis_range=[0, 1.05], xaxis_tickangle=-30,
                      legend=dict(orientation="h", y=-0.42))
    panels.append((
        "Step 1 &mdash; DID it notice? Detection rate by which view was corrupted",
        "The first half of the research question. Every item has exactly one corrupted "
        "view; this asks how often the model said &quot;these do not agree&quot; at all, "
        "broken out by <i>which</i> view was the corrupted one. It says nothing about "
        "whether the model then pointed at the right view &mdash; that is the next "
        "panel.<br><br>"
        "The trajectory is broken out into all four of its rungs rather than pooled, "
        "because they are not one task: <b>T_rand</b> is shape-matched noise, "
        "<b>T_shuf</b> is the trajectory's own values permuted, <b>T_swap</b> is the "
        "dataset's delivered wrong trajectory, and <b>T_exec</b> is what the invalid "
        "solver actually printed. Pooling them averaged a floor with a ceiling and made "
        "the numeric view look uniformly easy. The dotted line separates the three "
        "single-condition views from the ladder.<br><br>"
        "<b>Code is the hardest in every arm</b> — and now against each rung "
        "individually, not just their mean. When the solver is the view that "
        "contradicts the equation, the description and the numbers, models are least "
        "able to say so.",
        fig_html(fig, height=560, margin=dict(l=65, r=30, t=30, b=175))))

    fig = go.Figure()
    for i, (model, arm) in enumerate(sorted({(r["model"], r.get("thinking", "na"))
                                             for r in rows})):
        sub = df[(df.model == model) & (df.get("thinking", "na") == arm)]
        fig.add_bar(name=f"{model.split('/')[-1]} · think {arm}", x=XCATS,
                    y=[_view_cell(sub, v, lv, "localization_correct")
                       for _, v, lv in ROWS_BY_VIEW],
                    marker_color=ACCENT[i % len(ACCENT)])
    fig.add_vline(x=2.5, line=dict(color="#3a4258", width=1, dash="dot"))
    fig.add_hline(y=0.25, line=dict(color="#e67e8f", dash="dash"),
                  annotation_text="chance (1 of 4 slots)",
                  annotation_font=dict(color="#e67e8f", size=10))
    fig.update_layout(barmode="group", yaxis_title="localization | detected",
                      yaxis_range=[0, 1.05], xaxis_tickangle=-30,
                      legend=dict(orientation="h", y=-0.42))
    panels.append((
        "Step 2 &mdash; having noticed, did it point at the RIGHT view?",
        "The second half, and a different question from the panel before it. Scored "
        "<b>only on the items the previous panel counted as detected</b>, so the two "
        "are not two views of one number: a model can notice a disagreement and then "
        "blame the wrong representation. Chance here is 0.25. "
        "Localization scored only where detection succeeded, so this is the second "
        "half of the question and not a rerun of the first. Note the dissociation: "
        "code is the hardest view to <i>notice</i>, but once noticed it is named "
        "correctly as often as any other. The failure is in seeing the disagreement, "
        "not in attributing it. Broken out by trajectory rung as well, so a rung that "
        "is easy to notice but hard to attribute is visible instead of averaged away.",
        fig_html(fig, height=560, margin=dict(l=65, r=30, t=30, b=175))))

    # The old "Physics or lexical cues?" panel compared two marginal rates and
    # plotted their difference. Every item exists in BOTH namings, so that discarded
    # the pairing -- and the marginal delta cancels offsetting flips, which is exactly
    # where the instability lives. Replaced by obfuscation_panels(), which pairs the
    # items, reports consistency, and tests with McNemar.

    conf_rows = []
    for actual in VIEWS:
        sub = df[(df.corrupted_view == actual) & df.outlier.notna()
                 & (df.agree == "no")]
        if not len(sub):
            continue
        counts = {v: 0 for v in VIEWS}
        for _, r in sub.iterrows():
            slots = r.get("slots")
            try:
                idx = int(str(r["outlier"]).replace("view_", "")) - 1
                named = slots[idx] if slots is not None and 0 <= idx < len(slots) else None
            except (ValueError, TypeError, IndexError):
                named = None
            if named in counts:
                counts[named] += 1
        tot = sum(counts.values()) or 1
        conf_rows.append((actual, {k: v / tot for k, v in counts.items()}, tot))
    if conf_rows:
        fig = go.Figure(go.Heatmap(
            z=[[c[1][v] for v in VIEWS] for c in conf_rows],
            x=[f"blamed {v}" for v in VIEWS],
            y=[f"actually {c[0]}" for c in conf_rows],
            colorscale="Blues", zmin=0, zmax=1,
            text=[[f"{c[1][v]:.2f}" for v in VIEWS] for c in conf_rows],
            texttemplate="%{text}", showscale=True))
        fig.update_layout(height=420)
        panels.append((
            "Confusion matrix &mdash; what it blamed vs what was actually corrupted",
            "Rows are the view that was actually corrupted; columns are the view the "
            "model named, resolved through each item's randomized slot order. Pooled "
            "over all arms and restricted to items where the model said the views "
            "disagree. The diagonal is correct attribution; off-diagonal mass shows "
            "which representation gets blamed in its place.",
            fig_html(fig, False)))

    # the corruption ladder
    ladder = [c for c in ["X_T_rand", "X_T_shuf", "X_T_swap", "X_T_exec"] if c in conds]
    if ladder:
        fig = go.Figure()
        for i, (model, arm) in enumerate(sorted({(r["model"], r.get("thinking", "na"))
                                                 for r in rows})):
            sub = df[(df.model == model) & (df.get("thinking", "na") == arm)]
            fig.add_scatter(name=f"{model.split('/')[-1]} · think {arm}", x=ladder,
                            y=[sub[sub.condition == c]["detection_correct"].mean()
                               for c in ladder],
                            mode="lines+markers", marker=dict(size=11),
                            line=dict(color=ACCENT[i % len(ACCENT)], width=3))
        fig.update_layout(yaxis_title="detection accuracy")
        panels.append((
            "The corruption ladder",
            "Structurally empty to physically subtle. <b>T_rand</b>: shape-matched "
            "noise — a floor, detectable from smoothness alone. <b>T_shuf</b>: a "
            "permutation of the trajectory's own values, so every marginal statistic "
            "is identical and only arrangement differs. <b>T_swap</b>: the dataset's "
            "own swapped trajectory — coherent physics, wrong referent. <b>T_exec</b>: "
            "the invalid solver's real output. Reading the curve across the four is "
            "the measurement: passing T_rand but failing T_shuf means only summary "
            "statistics are being read.",
            fig_html(fig, False)))

    # d' with clustered intervals
    if summary:
        fig = go.Figure()
        for i, (arm, body) in enumerate(summary.get("arms", {}).items()):
            xs, ys, lo, hi = [], [], [], []
            for c, e in body.get("conditions", {}).items():
                if e.get("dprime") is None or c == "A0":
                    continue
                xs.append(c); ys.append(e["dprime"])
                ci = e.get("dprime_ci") or [None, None]
                lo.append((e["dprime"] - ci[0]) if ci[0] is not None else 0)
                hi.append((ci[1] - e["dprime"]) if ci[1] is not None else 0)
            fig.add_bar(name=arm, x=xs, y=ys, marker_color=ACCENT[i % len(ACCENT)],
                        error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                                     color="#888"))
        fig.update_layout(yaxis_title="d′")
        # The d' panel lived here. It answered the right question -- is the
        # detection real or is the model just flagging everything -- but stated it
        # as a signal-detection statistic with a clustered bootstrap, which is not
        # how anyone reads a result. "Is it detecting anything, or just flagging
        # everything?" makes the same comparison in percentage points against each
        # model's own false-alarm rate, and the summary table carries both raw rates.

    # parse health + degeneracy
    n_fail = int((df.get("parse_route", pd.Series(dtype=str)) == "failed").sum())
    body = []
    for (model, arm) in sorted({(r["model"], r.get("thinking", "na")) for r in rows}):
        sub = df[(df.model == model) & (df.get("thinking", "na") == arm)]
        agree = Counter(sub.get("agree", pd.Series(dtype=str)).dropna())
        degen = "YES — quote no d′ from this arm" if len(agree) == 1 else "no"
        body.append(f"<tr><td>{model}</td><td>{arm}</td><td>{len(sub)}</td>"
                    f"<td>{int((sub.get('parse_route', pd.Series(dtype=str)) == 'failed').sum())}</td>"
                    f"<td>{dict(agree)}</td><td>{degen}</td></tr>")
    # rows[:400] took whichever model's file was read first -- one arm, no contrast.
    # Stratify over (model x thinking x corrupted_view x detection outcome) so the
    # browser always contains misses next to hits.
    buckets = {}
    for r in rows:
        key = (r.get("model"), r.get("thinking"), r.get("corrupted_view"),
               r.get("detection_correct"))
        buckets.setdefault(key, []).append(r)
    x_sample = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        b = buckets[key]
        step = max(1, len(b) // 3)
        x_sample.extend(b[::step][:3])

    x_rows = []
    for r in x_sample:
        x_rows.append({
            "model": r.get("model", ""),
            "cell": f"{r.get('condition','')} / "
                    f"{ {1:'detected',0:'missed'}.get(r.get('detection_correct'),'unscored') }",
            "title": r.get("item_id", ""), "gt_sample": r.get("gt_sample", ""),
            "mod_type": f"think={r.get('thinking','na')}",
            "source": f"slots={r.get('slots','')}",
            "gt": {"corrupted view": r.get("corrupted_view", ""),
                   "outlier slot": str(r.get("outlier_slot", "")),
                   "pde class": r.get("gt_pde_class", ""),
                   "method": r.get("gt_num_method", "")},
            "parsed": {"agree": r.get("agree", ""), "outlier": r.get("outlier", ""),
                       "pde class": r.get("system_pde_class", ""),
                       "method": r.get("system_num_method", "")},
            "scores": {k: r.get(k) for k in
                       ("detection_correct", "localization_correct",
                        "pde_class_match", "num_method_match") if k in r},
            "axis": r.get("parse_route", ""), "conf": r.get("traj_level", ""),
            "finish": r.get("finish_reason", ""),
            "justification": str(r.get("justification") or ""),
            "chars": len(str(r.get("response", ""))),
            "text": str(r.get("response", "")),
        })
    panels.append((
        "Raw responses and justifications",
        "Each row shows what the model answered against what the item actually was: "
        "which view was corrupted, which slot held it, and the model's own account of "
        "what it thought was inconsistent. The justification is where a right answer "
        "for the wrong reason becomes visible — a model can name the correct outlier "
        "while explaining it by formatting rather than physics.",
        response_browser(x_rows, "sbatch/run_cross_modal_consistency.sbatch", prefix="xm")))

    panels.append((
        "Parse health and degeneracy",
        f"Parse failures score <b>null</b>, never as a wrong answer — otherwise "
        f"identification accuracy is deflated by exactly the parse-failure rate, "
        f"invisibly. Total failures: <b>{n_fail}</b>.",
        f'<table class="tbl"><tr><th>model</th><th>thinking</th><th>rows</th>'
        f'<th>parse failures</th><th>agree responses</th><th>degenerate?</th></tr>'
        f'{"".join(body)}</table>'))
    return panels


def from_hub(repo):
    """Rows from an HF dataset, or None. Lets the report be rebuilt from the
    published artifact alone -- no cluster, no local results directory."""
    try:
        from datasets import load_dataset
        d = load_dataset(repo, split="train", download_mode="force_redownload")
        print(f"[report] {repo}: {len(d)} rows")
        return d.to_pandas()
    except Exception as e:
        print(f"[report] could not load {repo}: {type(e).__name__}: {e}")
        return None


def build(freegen_csv, xmodal_dir, xmodal_summary, out,
          freegen_hf=None, xmodal_hf=None):
    df = None
    if freegen_hf:
        df = from_hub(freegen_hf)
    if df is None and freegen_csv and os.path.exists(freegen_csv):
        df = pd.read_csv(freegen_csv)
        print(f"[report] free-gen rows from CSV: {len(df)}")

    if df is not None and len(df):
        # A draw that ran out of budget mid-reasoning reached no answer. Scoring it
        # counts a model that said nothing as a model that said something false, so
        # it is dropped here rather than carried into the panels.
        if "no_verdict" in df.columns:
            bad = df["no_verdict"].astype(str).str.lower().isin(("true", "1"))
            if bad.any():
                print(f"[report] dropping {int(bad.sum())} row(s) with no verdict "
                      f"({100 * bad.mean():.1f}%) — these reached no answer")
                df = df[~bad]
        n_raw = len(df)
        df = pool_draws(df)
        if len(df) != n_raw:
            k = n_raw / max(len(df), 1)
            print(f"[report] pooled {n_raw} draws -> {len(df)} items (k={k:.1f}); "
                  f"every Experiment 1 n below counts ITEMS, not draws")

    rows = []
    if xmodal_hf:
        xdf = from_hub(xmodal_hf)
        if xdf is not None:
            rows = xdf.to_dict("records")
    if not rows:
        for p in sorted(glob.glob(os.path.join(xmodal_dir or "", "*.jsonl"))):
            with open(p) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    print(f"[report] cross-modal rows: {len(rows)}")
    summary = None
    if xmodal_summary and os.path.exists(xmodal_summary):
        summary = json.load(open(xmodal_summary))

    first = [True]
    xframe = crossmodal_frame(rows)
    # Experiment 1 keeps the panels that answer a question; the diagnostics that
    # support them move to the appendix. The report had grown to 28 panels, which is
    # a filing cabinet, not an argument.
    KEEP_E1 = ("Score by condition", "Can a model tell", "Raw responses")
    exp1_panels_all = exp1_panels(df, first)

    # The cross-modal half is ordered by the experiment's own question rather than by
    # whichever function happened to build the panel. The RQ has three moves -- did it
    # DETECT the disagreement, did it LOCALIZE which view dissents, and is that
    # judgement physical or lexical -- so the lead reads in that order, closing on what
    # the model still knows when one view is corrupted. Panels that re-cut those same
    # rows a fourth way are real results, not filler, but they are supporting detail;
    # they drop to the appendix so the lead is an argument instead of a filing cabinet.
    XMODAL_LEAD = (
        "Is it detecting anything",              # detect: signal, or a flag-everything bias
        "Step 1",                                # detect: by which view was corrupted
        "Step 2",                                # localize: having noticed, point correctly
        "Which representation does the model trust",   # the modality preference baseline
        "Physics or lexical cues",               # does that preference survive obfuscation
        "Is it comparing representations",       # position-effect control on shuffled order
        "What it knows vs what it can check",    # the `system` field: identify vs audit
        "Raw responses and justifications",      # the `justification` field, unabridged
    )

    xmodal_all = (trust_panels(xframe)
                  + rq_panels(pd.DataFrame(rows) if rows else None)
                  + exp3_panels(rows, summary))

    def take(prefixes, pool):
        """Split `pool` into the named panels, in the order named, and the remainder.

        Matching is by title prefix so the panel titles stay editable without this
        list having to track them character for character. A prefix that matches
        nothing is skipped silently: when a job has not landed, the producing
        function returns a placeholder instead, and a missing lead panel should
        shorten the report, not raise.
        """
        chosen, used = [], set()
        for pref in prefixes:
            for i, panel in enumerate(pool):
                if i not in used and panel[0].startswith(pref):
                    chosen.append(panel)
                    used.add(i)
                    break
        return chosen, [p for i, p in enumerate(pool) if i not in used]

    xmodal_lead, xmodal_rest = take(XMODAL_LEAD, xmodal_all)

    # Grouped, not one flat run of sixteen. Each experiment opens with its own
    # workflow schematic and its headline numbers, THEN its charts -- the earlier
    # layout interleaved two unrelated experiments in a single V0..V15 list, which
    # is what made the report unreadable.
    groups = [
        ("Both experiments", "start", [
            ("Overview", "", overview_html(df, rows)),
        ]),
        ("Experiment 1 &mdash; free generation", "e1",
         [("Experimental workflow", "Source in, four fields out. Every stage below is a file "
           "you can open; the eight condition chips are the actual mod_type values in "
           "the dataset.", exp1_schematic(df)),
          ("Results at a glance", "The whole experiment in one table, before any "
           "chart. Method recall is reported on-axis only: a response naming a spatial "
           "discretization when ground truth labels time integration has abstained, "
           "not erred.", exp1_headline(df))]
         + fig1_panels(df)[:1] + validity_dprime_panel(df)
         + validity_confidence_panel(df)
         + perturbation_confidence_panel(df)
         # Pooled first as the overview, then the per-model grid it deviates from.
         # The pooled bar on its own is the average of two opposite behaviours --
         # newer checkpoints near-perfect on the invalid half and near chance on the
         # valid half, older ones the reverse -- which reads as moderate competence
         # at both, and is the one thing this figure must not say.
         + hedge_breakdown_panel(df)
         + hedge_breakdown_by_model_panel(df)
         + [p for p in exp1_panels_all
            if any(p[0].startswith(k) for k in KEEP_E1)]),
        ("Experiment 2 &mdash; cross-modal consistency", "e2",
         [("Experimental workflow", "One physical system, rendered four ways. Exactly "
           "one rendering is corrupted, the four are shuffled behind a neutral legend, "
           "and the model is asked which &mdash; if any &mdash; does not belong. Read "
           "top to bottom.", exp2_workflow(rows)),
          ("Results at a glance", "Hit rate is over corrupted items; the false-alarm "
           "rate comes from A0 alone, which is why A0 exists. &quot;Hardest&quot; and "
           "&quot;easiest&quot; name the corruption this arm detected least and most "
           "often, above its own false-alarm rate.", exp2_headline(rows, summary))]
         + worked_example_panel()
         + xmodal_lead),
        ("Appendix &mdash; supporting detail", "ap",
         fig1_panels(df)[1:]
         + [p for p in exp1_panels_all
            if not any(p[0].startswith(k) for k in KEEP_E1)]
         + xmodal_rest),
    ]

    nav, secs, idx = [], [], 0
    for gname, gcls, gpanels in groups:
        # The appendix ships collapsed. It is the longest group in the report and
        # listing it open put thirty-odd buttons in the sidebar, which buried the
        # argument the lead groups are making; the count stays visible so it reads as
        # folded rather than missing.
        fold = gcls == "ap"
        if fold:
            nav.append(f'<div class="nav-group nav-{gcls} foldable" onclick="fold()">'
                       f'<span id="foldcaret">&#9656;</span> {gname} '
                       f'<span class="foldn">{len(gpanels)}</span></div>')
        else:
            nav.append(f'<div class="nav-group nav-{gcls}">{gname}</div>')
        for j, (title, q, bodyhtml) in enumerate(gpanels):
            prefix = {"e1": "1", "e2": "2", "cf": "F", "ap": "A"}.get(gcls, "")
            tag = "" if not prefix else f'{prefix}.{j + 1}&nbsp;&nbsp;'
            nav.append(
                f'<button class="nav-btn nav-{gcls}{" folded" if fold else ""} '
                f'{"active" if idx == 0 else ""}" '
                f'onclick="show({idx})">{tag}{title}</button>')
            eyebrow = ("" if gcls == "start" else
                       f'<div class="eyebrow eb-{gcls}">{gname}</div>')
            secs.append(
                f'<div class="section {"active" if idx == 0 else ""}">'
                f'{eyebrow}<div class="chart-header"><h2>{title}</h2></div>'
                f'{f"<div class=question>{q}</div>" if q else ""}{bodyhtml}</div>')
            idx += 1
    nav, secs = "\n".join(nav), "\n".join(secs)

    html = TEMPLATE.format(nav=nav, sections=secs, plotlyjs=get_plotlyjs(),
                           n_free=(len(df) if df is not None else 0), n_x=len(rows))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"[report] wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, {idx} panels)")


def overview_html(df, rows):
    n_free = len(df) if df is not None else 0
    n_models = df["model"].nunique() if df is not None and len(df) else 0
    return f"""
<div class="question" style="max-width:900px">
Two experiments over the same 32 PDE solvers.<br><br>
<b>Experiment 1 — free generation.</b> Show a model one solver's source and ask it to
name the PDE, the numerical method, the dominant physical behaviour, and whether the
code is physically valid. Eight conditions cross comment corruption, identifier
obfuscation and injected physical invalidity.
Currently <b>{n_free}</b> rows across <b>{n_models}</b> models.<br><br>
<b>Experiment 2 Part III — cross-modal consistency.</b> Show four independent
representations of one system — solver code, governing equation, numerical trajectory,
natural-language description — in randomized slot order, with exactly one corrupted.
Ask which, if any, disagrees. The three uncorrupted views form a majority that
determines the answer. Currently <b>{len(rows)}</b> rows.<br><br>
The pairing is the point: Experiment 1 asks whether a model can read a solver;
Part III asks whether it can tell when independent descriptions of that solver
disagree, and whether that judgement tracks physics or lexical cues.
</div>"""


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<script>{plotlyjs}</script>
<title>PDE LLM Eval — free generation and cross-modal consistency</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d0f18; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 290px; min-width: 290px; background: #12141e;
              border-right: 1px solid #1e2130; overflow-y: auto; }}
  #sidebar h1 {{ font-size: 0.85rem; color: #fff; padding: 14px 12px 10px;
                 border-bottom: 1px solid #1e2130; line-height: 1.4; }}
  .meta {{ font-size: 0.7rem; color: #555; padding: 8px 12px 12px;
           border-bottom: 1px solid #1e2130; line-height: 1.6; }}
  .meta strong {{ color: #888; }}
  .nav-btn {{ display: block; width: 100%; background: none; border: none;
              border-left: 3px solid transparent; color: #777; padding: 7px 12px 7px 9px;
              cursor: pointer; font-size: 0.76rem; text-align: left; line-height: 1.4; }}
  .nav-btn:hover {{ background: #191c2a; color: #ccc; }}
  .nav-btn.active {{ background: #171d30; color: #7eb8ff; border-left-color: #3a7bdd; }}
  #main {{ flex: 1; overflow-y: auto; padding: 22px 26px; }}
  .section {{ display: none; }} .section.active {{ display: block; }}
  .chart-header h2 {{ margin: 5px 0 10px; font-size: 1rem; color: #ddd; font-weight: 500; }}
  .question {{ font-size: 0.8rem; color: #5a80b0; max-width: 860px;
               border-left: 3px solid #1e3560; padding-left: 10px;
               line-height: 1.7; margin-bottom: 16px; }}
  .pending {{ border: 1px dashed #2a3450; color: #6b7a99; padding: 26px;
              border-radius: 6px; font-size: 0.85rem; line-height: 1.8; max-width: 700px; }}
  .pending .src {{ color: #46527a; font-family: ui-monospace, monospace; font-size: 0.78rem; }}
  .tbl {{ border-collapse: collapse; font-size: 0.78rem; margin-top: 8px; }}
  .tbl th, .tbl td {{ border: 1px solid #1e2130; padding: 5px 11px; text-align: left; }}
  .tbl th {{ color: #8fa6c9; font-weight: 500; background: #141826; }}
  .browser {{ max-width: 1080px; }}
  .controls {{ display: flex; gap: 9px; align-items: center; margin-bottom: 12px; }}
  .controls select, .controls button {{ background: #171d30; color: #cfd8e8;
      border: 1px solid #26304a; border-radius: 4px; padding: 5px 10px;
      font-size: 0.78rem; cursor: pointer; font-family: inherit; }}
  .controls button:hover {{ background: #1d2540; }}
  .controls span {{ font-size: 0.75rem; color: #6b7a99; min-width: 66px; text-align: center;
             font-family: ui-monospace, monospace; }}
  .rb-meta {{ background: #141826; border: 1px solid #1e2130; border-radius: 6px 6px 0 0;
              padding: 10px 13px; font-size: 0.75rem; line-height: 1.85; }}
  .rb-line b {{ color: #8fa6c9; font-weight: 500; }}
  .rb-tag {{ background: #1d2540; color: #7eb8ff; border-radius: 3px;
             padding: 1px 7px; font-size: 0.7rem; }}
  .rb-gt {{ color: #8fd694; }} .rb-pa {{ color: #f2a97e; }}
  .rb-sc {{ color: #6b7a99; font-family: ui-monospace, monospace; font-size: 0.72rem; }}
  .rb-just {{ background: #12182a; border: 1px solid #1e2130; border-top: none;
              padding: 11px 14px; font-size: 0.78rem; line-height: 1.7; color: #cfe0ff; }}
  .rb-just:empty {{ display: none; }}
  .rb-jlabel {{ color: #7eb8ff; font-size: 0.68rem; letter-spacing: 0.08em;
                text-transform: uppercase; margin-bottom: 5px; }}
  .rb-text {{ background: #0a0c14; border: 1px solid #1e2130; border-top: none;
              border-radius: 0 0 6px 6px; padding: 14px 16px; font-size: 0.76rem;
              line-height: 1.65; color: #c8cddb; white-space: pre-wrap;
              word-wrap: break-word; max-height: 620px; overflow-y: auto;
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}

  /* ── experiment colour identity ─────────────────────────────────────────── */
  .nav-group {{ font-size: 0.63rem; letter-spacing: 0.11em; text-transform: uppercase;
                color: #4d5875; padding: 15px 12px 6px; border-top: 1px solid #1e2130;
                margin-top: 4px; }}
  .nav-group.nav-e1 {{ color: #4d7fbe; }} .nav-group.nav-e2 {{ color: #b98a52; }}
  .nav-group.nav-ap {{ color: #5d6478; }}
  .nav-group.nav-cf {{ color: #8a7fb8; }}
  .nav-btn.nav-cf.active {{ color: #c3b8f0; border-left-color: #6d5fa8;
                            background: #1c1a2b; }}
  .eyebrow.eb-cf {{ color: #8a7fb8; }}
  /* The embedded figures render on the dark palette, so the card has to be the
     panel colour. A white card behind a dark figure was the "colors are messed up"
     bug: the figure's own background became a floating dark rectangle. */
  .figwrap {{ background: #12141e; border: 1px solid #1e2130; border-radius: 6px;
              padding: 10px; overflow-x: auto; }}
  .figwrap img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
  .realbanner {{ border: 1px solid #24403a; background: #131f1c; color: #8fd6b8;
                 padding: 12px 15px; border-radius: 6px; margin-bottom: 14px;
                 font-size: 0.78rem; line-height: 1.6; }}
  .realbanner code {{ color: #b8e6cf; }}
  .synthbanner {{ border: 1px dashed #7a5c2a; background: #241d10; color: #d9b877;
                  padding: 12px 15px; border-radius: 6px; margin-bottom: 14px;
                  font-size: 0.78rem; line-height: 1.6; }}
  .synthbanner code {{ color: #f0c98a; }}
  .nav-group.foldable {{ cursor: pointer; user-select: none; }}
  .nav-group.foldable:hover {{ color: #99a3bb; }}
  #foldcaret {{ display: inline-block; transition: transform 0.12s; font-size: 0.7rem; }}
  #foldcaret.open {{ transform: rotate(90deg); }}
  .foldn {{ color: #46527a; }}
  .nav-btn.folded {{ display: none; }}
  .nav-btn.folded.shown {{ display: block; }}
  .nav-btn.nav-ap.active {{ color: #b9c2d6; border-left-color: #5d6478; background: #191c26; }}
  .nav-btn.nav-e1.active {{ color: #7eb8ff; border-left-color: #3a7bdd; background: #161d30; }}
  .nav-btn.nav-e2.active {{ color: #f0b878; border-left-color: #c98a3c; background: #241d13; }}
  .eyebrow {{ font-size: 0.64rem; letter-spacing: 0.12em; text-transform: uppercase;
              margin-bottom: 3px; }}
  .eb-e1 {{ color: #4d7fbe; }} .eb-e2 {{ color: #b98a52; }} .eb-ap {{ color: #5d6478; }}

  /* ── workflow schematic ─────────────────────────────────────────────────── */
  .flow {{ max-width: 880px; --fc: #3a7bdd; --fct: #7eb8ff; --fbg: #141c2e; }}
  .flow-e2 {{ --fc: #c98a3c; --fct: #f0b878; --fbg: #221b12; }}
  .fstage {{ display: flex; gap: 13px; align-items: flex-start;
             background: var(--fbg); border: 1px solid #232838;
             border-left: 3px solid var(--fc); border-radius: 6px; padding: 12px 15px; }}
  .fnum {{ flex: 0 0 46px; font-size: 0.63rem; letter-spacing: 0.06em;
           text-transform: uppercase; color: var(--fct); padding-top: 2px;
           font-family: ui-monospace, monospace; }}
  .ftitle {{ font-size: 0.87rem; color: #e6e9f2; margin-bottom: 4px; font-weight: 500; }}
  .fdet {{ font-size: 0.77rem; color: #97a3bd; line-height: 1.65; }}
  .fdet code {{ color: var(--fct); font-size: 0.74rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .fdet b {{ color: #ccd4e6; font-weight: 500; }}
  .fchips {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }}
  .fchip {{ background: #0e1220; border: 1px solid #262c40; border-radius: 3px;
            padding: 2px 8px; font-size: 0.7rem; color: #8592ae;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .fchip-hi {{ border-color: var(--fc); color: var(--fct); }}
  .farrow {{ width: 2px; height: 16px; background: #262c40; margin: 0 0 0 25px; }}

  .we-tab {{ background: #171d30; color: #9aa6bd; border: 1px solid #26304a;
             border-radius: 4px; padding: 5px 13px; font-size: 0.78rem; cursor: pointer;
             font-family: inherit; }}
  .we-tab:hover {{ background: #1d2540; }}
  .we-tab.active {{ background: #1d2540; color: #7eb8ff; border-color: #3a7bdd; }}
  .we-tab.is-outlier {{ border-color: #7d3c3c; }}
  .we-tab.is-outlier::after {{ content: " \\2022"; color: #e69090; }}

  /* ── experimental workflow diagram ─────────────────────────────────────── */
  .wf {{ max-width: 900px; }}
  .wf-box {{ background: #171c2b; border: 1px solid #262c40; border-radius: 7px;
             padding: 13px 17px; font-size: 0.83rem; color: #dbe1ee; }}
  .wf-src {{ border-left: 3px solid #b98a52; }}
  .wf-sub {{ font-size: 0.75rem; color: #8592ae; line-height: 1.6; margin-top: 6px; }}
  .wf-math {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: 0.75rem; color: #f0b878; margin-top: 9px; }}
  .wf-arrow {{ display: flex; align-items: center; gap: 11px; margin: 3px 0 3px 28px;
               font-size: 0.73rem; color: #7b88a4; }}
  .wf-arrow::before {{ content: ""; width: 2px; height: 34px; background: #2f3750;
                       flex: 0 0 2px; }}
  .wf-arrow b {{ color: #cbd3e4; font-weight: 500; }}
  .wf-views {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 9px; }}
  .wf-metrics {{ grid-template-columns: repeat(3, 1fr); }}
  .wf-view {{ background: #141826; border: 1px solid #232838; border-top: 3px solid #4d5875;
              border-radius: 6px; padding: 11px 12px; }}
  .wf-vname {{ font-size: 0.8rem; color: #dbe1ee; margin-bottom: 5px; }}
  .wf-vdet {{ font-size: 0.72rem; color: #8592ae; line-height: 1.55; }}
  .wf-note {{ font-size: 0.73rem; color: #6f7c98; margin: 9px 0 0; padding-left: 2px;
              line-height: 1.6; }}
  .wf-cond {{ border-left: 3px solid #c98a3c; }}
  .wf-cond .fchip {{ margin: 0 4px 5px 0; display: inline-block; }}
  .wf-prompt {{ border-left: 3px solid #3a7bdd; }}
  .wf-models .fchip {{ margin: 0 4px 0 0; }}
  .wf-out-row {{ border-left: 3px solid #4fa96a; }}
  .wf-out {{ display: inline-block; background: #102018; border: 1px solid #24503a;
             color: #8fd694; border-radius: 3px; padding: 2px 9px; margin: 0 5px 0 0;
             font-size: 0.74rem;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}

  /* ── condition decoder ─────────────────────────────────────────────────── */
  .dechead {{ font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase;
              color: #b98a52; margin: 26px 0 7px; }}
  .decoder {{ max-width: 1040px; }}
  .decoder td {{ vertical-align: top; }}
  .ccode {{ color: #f0b878; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.75rem; }}
  .cwhy {{ color: #8592ae; font-size: 0.74rem; line-height: 1.6; max-width: 430px; }}
  .cwhy code {{ color: #b0bcd6; font-size: 0.72rem; }}

  /* ── headline numbers ───────────────────────────────────────────────────── */
  .kpis {{ display: flex; flex-wrap: wrap; gap: 11px; margin-bottom: 18px; }}
  .kpi {{ background: #141826; border: 1px solid #1e2130; border-radius: 6px;
          padding: 12px 18px; min-width: 155px; }}
  .kv {{ font-size: 1.5rem; color: #e6e9f2; font-weight: 300; line-height: 1.2;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .kl {{ font-size: 0.71rem; color: #7b88a4; margin-top: 4px; line-height: 1.5; }}
  .ksub {{ color: #4d5875; }}
  .thsub {{ color: #55607d; font-weight: 400; font-size: 0.68rem; }}
  .tbl td.num {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                 text-align: right; }}
  .tbl td.low {{ color: #e69090; font-family: ui-monospace, monospace; }}
  .tbl td.high {{ color: #8fd694; font-family: ui-monospace, monospace; }}
</style></head><body>
<div id="sidebar">
  <h1>PDE LLM Eval — two experiments</h1>
  <div class="meta">
    <strong>Dataset:</strong> merged_mod_jul28, 32 solvers<br>
    <strong>Free-gen rows:</strong> {n_free}<br>
    <strong>Cross-modal rows:</strong> {n_x}
  </div>
  {nav}
</div>
<div id="main">{sections}</div>
<script>
function fold(open) {{
  var btns = document.querySelectorAll('.nav-btn.folded');
  // Explicit `open` wins so show() can reveal the group; a bare click toggles.
  var want = (open === undefined)
      ? !(btns.length && btns[0].classList.contains('shown'))
      : open;
  btns.forEach(function(b) {{ b.classList.toggle('shown', want); }});
  document.getElementById('foldcaret').classList.toggle('open', want);
}}
function show(i) {{
  document.querySelectorAll('.section').forEach((s,j)=>s.classList.toggle('active',i===j));
  document.querySelectorAll('.nav-btn').forEach((b,j)=>b.classList.toggle('active',i===j));
  // Landing on an appendix panel with the group folded would leave the sidebar
  // showing no selection at all, so unfold whenever the target lives in there.
  var t = document.querySelectorAll('.nav-btn')[i];
  if (t && t.classList.contains('folded')) fold(true);
  document.getElementById('main').scrollTop = 0;
  window.dispatchEvent(new Event('resize'));
}}
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freegen", default="results/pde_llm_eval_jul28.csv")
    ap.add_argument("--xmodal_dir", default="results/xmodal")
    ap.add_argument("--xmodal_summary", default="results/xmodal_summary.json")
    ap.add_argument("--out", default="viz/pde_dual_report.html")
    ap.add_argument("--freegen_hf", default=None,
                    help="HF repo for free-generation rows; overrides --freegen.")
    ap.add_argument("--xmodal_hf", default=None,
                    help="HF repo for cross-modal rows; overrides --xmodal_dir.")
    a = ap.parse_args()
    build(a.freegen, a.xmodal_dir, a.xmodal_summary, a.out,
          freegen_hf=a.freegen_hf, xmodal_hf=a.xmodal_hf)
