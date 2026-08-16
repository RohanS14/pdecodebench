"""
visualize_frontier.py — Belief revision experiment visualizations.

Reads: results/frontier/<slug>__belief_revision.jsonl
Produces 5 figures (saved as HTML + PNG):
  V1  — Accuracy by mod_type (valid_match + pde_match), S1 vs S2, two panels
  V2  — Transition table counts + delta-accuracy bar by mod_type
  V3  — Hedging stacked bar S1 vs S2, split by gt_valid
  V4  — Accuracy by PDE class (valid_match), S1 vs S2
  V5  — traj_signal stratified transition table

Usage:
  python viz/visualize_frontier.py                        # auto-detect latest result
  python viz/visualize_frontier.py --input results/frontier/gemini25flashpreview0417__belief_revision.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

MOD_ORDER = [
    "Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]
MOD_SHORT = {
    "Comm_Valid":              "Clean+Comment",
    "NoComm_Valid":            "Clean, No Comment",
    "CorrComm":                "Corrupt Comment",
    "NoComm_CorrVar":          "Corrupt Variable",
    "Comm_InValid":            "Invalid+Comment",
    "NoComm_InValid":          "Invalid, No Comment",
    "CorrComm_Invalid":        "CorrComment+Invalid",
    "NoComm_CorrVar_InValid":  "CorrVar+Invalid",
}

PDE_ORDER = ["burgers", "heat", "wave", "navier-stokes"]
PDE_LABEL = {"burgers": "Burgers", "heat": "Heat", "wave": "Wave", "navier-stokes": "NavierStokes"}

HEDGE_ORDER  = ["Confident Yes", "Uncertain Yes", "Hedged", "Confident No"]
HEDGE_COLORS = {
    "Confident Yes": "#2ecc71",
    "Uncertain Yes": "#a9dfbf",
    "Hedged":        "#f0b27a",
    "Confident No":  "#c0392b",
}

TRAJ_ORDER  = ["clear_invalid", "ambiguous_invalid", "clear_valid"]
TRANS_ORDER = ["right->right", "right->wrong", "wrong->right", "wrong->wrong"]
TRANS_SHORT = {"right->right": "R→R", "right->wrong": "R→W",
               "wrong->right": "W→R", "wrong->wrong": "W→W"}
TRANS_COLORS = {
    "right->right": "#27ae60",
    "right->wrong": "#e74c3c",
    "wrong->right": "#2980b9",
    "wrong->wrong": "#7f8c8d",
}

MARGIN   = dict(l=70, r=200, t=70, b=80)
CMARGIN  = dict(l=60, r=160, t=55, b=70)
FONT     = dict(family="Arial, sans-serif", size=13, color="#111")
LEGEND   = dict(x=1.02, y=1, xanchor="left", yanchor="top",
                bgcolor="rgba(240,242,255,0.95)", bordercolor="#aaa",
                borderwidth=1, font=dict(color="#111"))


# ── Utilities ─────────────────────────────────────────────────────────────────

def bootstrap_ci(arr, n=1000, ci=95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0], arr[0]
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, ci + (100 - ci) / 2)
    return lo, np.mean(arr), hi


def load_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    # Normalise mod_type spelling (dataset uses CorrComm_Invalid, not CorrComm_InValid)
    df["mod_type"] = df["mod_type"].str.strip()
    df["gt_pde"]   = df["gt_pde"].str.strip()
    return df


_FIGS: list[tuple[str, str, str, go.Figure, str]] = []  # (nav_title, plot_title, description, fig, extra_html)
_VIZ_IDX = [0]  # mutable counter for Vi labels


def save(fig, name: str, out_dir: Path, title: str = "", description: str = "", extra_html: str = "") -> None:
    """extra_html is optional raw HTML appended after the figure within the
    same section -- e.g. plain <pre>-formatted tables that belong in the same
    numbered V-section as the figure rather than getting their own section.
    Existing callers that don't pass it are unaffected (defaults to "").

    fig may be None for a section that is pure extra_html (e.g. a hand-drawn
    SVG schematic) with no Plotly chart at all -- write_combined_html skips
    the figure entirely for such sections."""
    _VIZ_IDX[0] += 1
    nav_title = f"V{_VIZ_IDX[0]}  {title or name}"
    _FIGS.append((nav_title, title or name, description, fig, extra_html))
    if fig is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        png_path = out_dir / f"{name}.png"
        fig.write_image(str(png_path), scale=2)
        print(f"  saved {png_path}")
    except Exception:
        pass  # kaleido optional


def write_combined_html(out_dir: Path, slug: str) -> None:
    nav_html = content_html = ""
    first_fig_idx = next((i for i, entry in enumerate(_FIGS) if entry[3] is not None), None)
    for idx, (nav_title, plot_title, desc, fig, extra_html) in enumerate(_FIGS):
        active = "active" if idx == 0 else ""
        nav_html += f'<button class="nav-btn {active}" onclick="show({idx})">{nav_title}</button>\n'
        fig_html = (
            fig.to_html(full_html=False, include_plotlyjs=(idx == first_fig_idx), div_id=f"fig{idx}")
            if fig is not None else ""
        )
        content_html += f'''
<div class="section {active}" id="sec{idx}">
  <div class="chart-header">
    <h2>{plot_title}</h2>
    <p class="question">{desc}</p>
  </div>
  {fig_html}
  {extra_html}
</div>'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Belief Revision Results — {slug}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0d0f18; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 260px; min-width: 260px; background: #12141e;
              border-right: 1px solid #1e2130; overflow-y: auto; display: flex;
              flex-direction: column; }}
  #sidebar h1 {{ font-size: 0.85rem; color: #fff; padding: 14px 12px 10px;
                 border-bottom: 1px solid #1e2130; line-height: 1.4; }}
  .meta {{ font-size: 0.7rem; color: #555; padding: 8px 12px 12px;
           border-bottom: 1px solid #1e2130; line-height: 1.6; }}
  .meta strong {{ color: #888; }}
  .nav-btn {{ display: block; width: 100%; background: none; border: none;
              border-left: 3px solid transparent; color: #777; padding: 7px 12px 7px 9px;
              cursor: pointer; font-size: 0.76rem; text-align: left; line-height: 1.4;
              transition: background 0.1s; }}
  .nav-btn:hover {{ background: #191c2a; color: #ccc; }}
  .nav-btn.active {{ background: #171d30; color: #7eb8ff; border-left-color: #3a7bdd; }}
  #main {{ flex: 1; overflow-y: auto; padding: 22px 26px; }}
  .section {{ display: none; }}
  .section.active {{ display: block; }}
  .chart-header h2 {{ margin: 5px 0 10px; font-size: 1rem; color: #ddd; font-weight: 500; }}
  .question {{ font-size: 0.8rem; color: #5a80b0; max-width: 820px;
               border-left: 3px solid #1e3560; padding-left: 10px;
               line-height: 1.6; margin-bottom: 16px; }}
</style>
</head>
<body>
<div id="sidebar">
  <h1>Belief Revision — {slug}</h1>
  <div class="meta">
    <strong>Model:</strong> {slug}<br>
    <strong>Experiment:</strong> Two-stage code + trajectory eval<br>
    <strong>Dataset:</strong> PDEcodeBench v4 (128 scripts, 8 mod_types)
  </div>
  {nav_html}
</div>
<div id="main">{content_html}</div>
<script>
function show(i) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('sec'+i).classList.add('active');
  document.querySelectorAll('.nav-btn')[i].classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""

    out_path = out_dir / f"belief_revision_{slug}.html"
    out_path.write_text(html)
    print(f"  saved {out_path}")


# ── V1: Accuracy by mod_type, S1 vs S2, two panels ───────────────────────────

def v1_accuracy_by_modtype(df: pd.DataFrame, out_dir: Path) -> None:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Panel A — Validity Prediction (valid_match)",
                        "Panel B — PDE Identification (pde_match) [control]"],
        horizontal_spacing=0.12,
    )

    for col_idx, (metric, s1_col, s2_col) in enumerate([
        ("valid_match", "s1_valid_match", "s2_valid_match"),
        ("pde_match",   "s1_pde_match",   "s2_pde_match"),
    ], start=1):

        conds   = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
        labels  = [MOD_SHORT.get(c, c) for c in conds]

        for stage, col, color, dash in [
            ("S1 (code only)",   s1_col, "#2980b9", "solid"),
            ("S2 (code+traj)",   s2_col, "#e74c3c", "dash"),
        ]:
            ys, lo_errs, hi_errs = [], [], []
            for cond in conds:
                sub = df[df["mod_type"] == cond][col].dropna()
                lo, mean, hi = bootstrap_ci(sub)
                ys.append(mean * 100)
                lo_errs.append((mean - lo) * 100)
                hi_errs.append((hi - mean) * 100)

            fig.add_trace(go.Scatter(
                x=labels, y=ys,
                mode="lines+markers",
                name=stage,
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=8),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=hi_errs,
                    arrayminus=lo_errs,
                    thickness=1.5,
                    width=4,
                ),
                legendgroup=stage,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        fig.update_yaxes(range=[0, 105], title_text="Accuracy (%)",
                         row=1, col=col_idx)
        fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                      annotation_text="chance", annotation_position="bottom right",
                      row=1, col=col_idx)

    fig.update_layout(
        title="Accuracy by Mod Type: Stage 1 vs Stage 2",
        font=FONT, legend=LEGEND, margin=MARGIN,
        width=1300, height=520,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=35)
    save(fig, "V1_accuracy_modtype", out_dir,
         title="Accuracy by Mod Type: S1 vs S2",
         description="Panel A: validity prediction accuracy. Panel B: PDE identification (control — should be flat). "
                     "S2 gain on invalid mod_types = trajectory evidence helps. S2 gain on valid = model corrected "
                     "over-critical prior. RW=0 means Stage 2 never degraded a correct answer.")


# ── V2: Transition stacked bar + delta annotations (single chart) ─────────────

def v2_transitions(df: pd.DataFrame, out_dir: Path) -> None:
    conds  = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
    labels = [MOD_SHORT.get(c, c) for c in conds]

    fig = go.Figure()

    # Stacked bars — omit R→W from bars (n=0), keep in legend
    for trans in TRANS_ORDER:
        ys = []
        for cond in conds:
            sub = df[df["mod_type"] == cond]
            ys.append((sub["transition"] == trans).sum())
        total_n = sum(ys)
        legend_name = f"{TRANS_SHORT[trans]} (n=0)" if total_n == 0 else TRANS_SHORT[trans]
        fig.add_trace(go.Bar(
            name=legend_name,
            x=labels, y=ys,
            marker_color=TRANS_COLORS[trans],
            visible=True if total_n > 0 else "legendonly",
        ))

    # Delta annotations above each bar group
    for i, cond in enumerate(conds):
        sub = df[df["mod_type"] == cond]
        d = (sub["s2_valid_match"] - sub["s1_valid_match"]).dropna()
        _, mean, _ = bootstrap_ci(d)
        color = "#1a7a3c" if mean >= 0 else "#a93226"
        fig.add_annotation(
            x=labels[i], y=len(sub) + 0.5,
            text=f"Δ{mean*100:+.0f}pp",
            showarrow=False,
            font=dict(size=11, color=color),
            yanchor="bottom",
        )

    fig.update_layout(
        title="Transition Counts by Mod Type",
        barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
        width=1100, height=520,
        yaxis_title="Count",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=35,
    )
    save(fig, "V2_transitions", out_dir,
         title="Transitions: Counts by Mod Type",
         description="Stacked transition counts per mod_type. R→R: correct both stages. "
                     "R→W: degraded (n=0 — Stage 2 never hurt a correct answer). "
                     "W→R: improved. W→W: wrong both stages. "
                     "Δ labels show S2−S1 accuracy change in percentage points.")


# ── V3: Hedging stacked bar S1 vs S2, single combined chart ──────────────────

def v3_hedging(df: pd.DataFrame, out_dir: Path) -> None:
    conds  = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
    labels = [MOD_SHORT.get(c, c) for c in conds]

    # Build x-axis: S1/S2 tight within each mod_type, gap between mod_types.
    # We use a numeric x with custom ticktext so we can control bar spacing.
    # Each mod_type occupies positions [i*3, i*3+1]; gap of 1 between groups.
    x_s1 = [i * 3     for i in range(len(conds))]
    x_s2 = [i * 3 + 1 for i in range(len(conds))]
    tick_vals = [i * 3 + 0.5 for i in range(len(conds))]

    # Subtitle bands: add a shape rectangle behind each group, colored by gt_valid
    # valid (first 4) = light green tint, invalid (last 4) = light red tint
    shapes = []
    for i in range(len(conds)):
        color = "rgba(39,174,96,0.06)" if i < 4 else "rgba(192,57,43,0.06)"
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=i*3 - 0.6, x1=i*3 + 1.6, y0=0, y1=1,
            fillcolor=color, line_width=0, layer="below",
        ))

    fig = go.Figure()

    for hedge in HEDGE_ORDER:
        ys1, ys2 = [], []
        for cond in conds:
            sub = df[df["mod_type"] == cond]
            n = len(sub)
            ys1.append((sub["s1_hedge_class"] == hedge).sum() / n * 100 if n > 0 else 0)
            ys2.append((sub["s2_hedge_class"] == hedge).sum() / n * 100 if n > 0 else 0)

        fig.add_trace(go.Bar(
            name=hedge, x=x_s1, y=ys1,
            marker_color=HEDGE_COLORS[hedge],
            legendgroup=hedge, showlegend=True,
            width=0.8,
        ))
        fig.add_trace(go.Bar(
            name=hedge, x=x_s2, y=ys2,
            marker_color=HEDGE_COLORS[hedge],
            legendgroup=hedge, showlegend=False,
            width=0.8,
            marker_pattern_shape="",
        ))

    # S1/S2 labels below each pair
    for i, lbl in enumerate(labels):
        for offset, stage_lbl in [(0, "S1"), (1, "S2")]:
            fig.add_annotation(
                x=i*3 + offset, y=-8, text=stage_lbl,
                showarrow=False, font=dict(size=9, color="#666"),
                xref="x", yref="y",
            )

    # Annotation: gt=valid / gt=invalid band labels
    fig.add_annotation(x=4.5, y=108, text="← gt = Valid →",
                       showarrow=False, font=dict(size=11, color="#27ae60"), xref="x", yref="y")
    fig.add_annotation(x=16.5, y=108, text="← gt = Invalid →",
                       showarrow=False, font=dict(size=11, color="#c0392b"), xref="x", yref="y")

    fig.update_layout(
        title="Hedging Distribution: S1 vs S2",
        barmode="stack", font=FONT, legend=LEGEND,
        margin=dict(l=70, r=200, t=70, b=100),
        width=1400, height=540,
        shapes=shapes,
        xaxis=dict(
            tickvals=tick_vals, ticktext=labels,
            tickangle=35, tickfont=dict(size=11),
        ),
        yaxis=dict(title="% of scripts", range=[-15, 115]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    save(fig, "V3_hedging", out_dir,
         title="Hedging Distribution: S1 vs S2",
         description="Confidence class breakdown per mod_type. Solid = S1 (code only), hatched = S2 (code+traj). "
                     "Green-tinted: gt=valid code (correct answer = Yes). Red-tinted: gt=invalid (correct = No). "
                     "Near-zero hedging reflects Gemini 2.5 Flash's strict instruction-following at temperature=0.")


# ── V4: Accuracy by PDE class ─────────────────────────────────────────────────

def v4_accuracy_by_pde(df: pd.DataFrame, out_dir: Path) -> None:
    pdes   = [p for p in PDE_ORDER if p in df["gt_pde"].unique()]
    labels = [PDE_LABEL.get(p, p) for p in pdes]

    fig = go.Figure()

    for stage, s1_col, color, dash in [
        ("S1 (code only)", "s1_valid_match", "#2980b9", "solid"),
        ("S2 (code+traj)", "s2_valid_match", "#e74c3c", "dash"),
    ]:
        ys, lo_errs, hi_errs = [], [], []
        for pde in pdes:
            sub = df[df["gt_pde"] == pde][s1_col].dropna()
            lo, mean, hi = bootstrap_ci(sub)
            ys.append(mean * 100)
            lo_errs.append((mean - lo) * 100)
            hi_errs.append((hi - mean) * 100)

        fig.add_trace(go.Scatter(
            x=labels, y=ys,
            mode="lines+markers",
            name=stage,
            line=dict(color=color, dash=dash, width=2),
            marker=dict(size=10),
            error_y=dict(type="data", symmetric=False,
                         array=hi_errs, arrayminus=lo_errs,
                         thickness=1.5, width=5),
        ))

    fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                  annotation_text="chance", annotation_position="bottom right")

    fig.update_layout(
        title="Validity Accuracy by PDE Class: Stage 1 vs Stage 2",
        xaxis_title="PDE Class",
        yaxis=dict(title="Accuracy (%)", range=[0, 105]),
        font=FONT, legend=LEGEND, margin=CMARGIN,
        width=800, height=480,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    save(fig, "V4_accuracy_pde", out_dir,
         title="Validity Accuracy by PDE Class: S1 vs S2",
         description="S1 vs S2 valid_match accuracy per PDE family (32 scripts each). "
                     "NavierStokes expected smallest S2 gain (no time-series for NS_1/2). "
                     "Burgers expected largest gain (simple 1D, clear NaN signals). "
                     "Low S1 accuracy on valid scripts across all classes = systematic over-critical prior.")


# ── V5: traj_signal stratified transition table ───────────────────────────────

def v5_traj_signal_transitions(df: pd.DataFrame, out_dir: Path) -> None:
    sigs   = [s for s in TRAJ_ORDER if s in df["traj_signal"].unique()]
    n_sigs = len(sigs)

    fig = make_subplots(
        rows=1, cols=n_sigs,
        subplot_titles=[f"{s}\n(n={len(df[df['traj_signal']==s])})" for s in sigs],
        horizontal_spacing=0.08,
    )

    for col_idx, sig in enumerate(sigs, start=1):
        sub = df[df["traj_signal"] == sig]

        for trans in TRANS_ORDER:
            count = (sub["transition"] == trans).sum()
            pct   = count / len(sub) * 100 if len(sub) > 0 else 0
            total_across_sigs = (df["transition"] == trans).sum()
            legend_name = f"{TRANS_SHORT[trans]} (n=0)" if total_across_sigs == 0 else TRANS_SHORT[trans]
            fig.add_trace(go.Bar(
                name=legend_name,
                x=[TRANS_SHORT[trans]] if total_across_sigs > 0 else [f"{TRANS_SHORT[trans]} (n=0)"],
                y=[pct] if total_across_sigs > 0 else [0],
                marker_color=TRANS_COLORS[trans],
                legendgroup=trans,
                showlegend=(col_idx == 1),
                visible=True if total_across_sigs > 0 else "legendonly",
                text=[f"{count}"] if total_across_sigs > 0 else [""],
                textposition="inside",
            ), row=1, col=col_idx)

        # delta annotation
        d = (sub["s2_valid_match"] - sub["s1_valid_match"]).dropna()
        lo, mean, hi = bootstrap_ci(d)
        xref = "x domain" if col_idx == 1 else f"x{col_idx} domain"
        yref = "y domain" if col_idx == 1 else f"y{col_idx} domain"
        fig.add_annotation(
            text=f"Δ={mean*100:+.1f}pp",
            x=0.5, y=-0.18, xref=xref, yref=yref,
            showarrow=False, font=dict(size=12, color="#333"),
        )
        fig.update_yaxes(title_text="% of scripts", range=[0, 105],
                         row=1, col=col_idx)

    fig.update_layout(
        title="Transitions Stratified by Trajectory Signal Strength",
        barmode="group", font=FONT, legend=LEGEND,
        margin=dict(l=70, r=200, t=70, b=110),
        width=1100, height=520,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    save(fig, "V5_traj_signal_transitions", out_dir,
         title="Transitions by Trajectory Signal Strength",
         description="Transition counts stratified by traj_signal. "
                     "clear_invalid: trajectory has NaN/spike — model should update. "
                     "ambiguous_invalid: subtle bug, clean trajectory — no update expected. "
                     "clear_valid: valid code, clean trajectory — model should stay correct. "
                     "Key contrast: WR rate for clear_invalid vs ambiguous_invalid tests whether updates are signal-driven.")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def latex_transition_table(df: pd.DataFrame, out_dir: Path) -> None:
    conds = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Belief Revision Transitions by Modification Type}",
        r"\label{tab:transitions}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Mod Type} & \textbf{R$\to$R (\%)} & \textbf{R$\to$W (\%)} & "
        r"\textbf{W$\to$R (\%)} & \textbf{W$\to$W (\%)} & $\boldsymbol{\Delta}$ \textbf{Acc (pp)} \\",
        r"\midrule",
    ]

    all_rr = all_rw = all_wr = all_ww = 0
    for cond in conds:
        sub = df[df["mod_type"] == cond]
        n   = len(sub)
        if n == 0:
            continue
        rr = (sub["transition"] == "right->right").sum()
        rw = (sub["transition"] == "right->wrong").sum()
        wr = (sub["transition"] == "wrong->right").sum()
        ww = (sub["transition"] == "wrong->wrong").sum()
        delta = (sub["s2_valid_match"] - sub["s1_valid_match"]).mean() * 100
        all_rr += rr; all_rw += rw; all_wr += wr; all_ww += ww
        label = r"\texttt{" + cond.replace("_", r"\_") + "}"
        lines.append(
            f"{label} & {rr/n*100:.0f} & {rw/n*100:.0f} & "
            f"{wr/n*100:.0f} & {ww/n*100:.0f} & {delta:+.1f} \\\\"
        )

    total = all_rr + all_rw + all_wr + all_ww
    if total > 0:
        lines += [
            r"\midrule",
            r"\textbf{Overall} & "
            f"{all_rr/total*100:.0f} & {all_rw/total*100:.0f} & "
            f"{all_wr/total*100:.0f} & {all_ww/total*100:.0f} & "
            f"{(df['s2_valid_match']-df['s1_valid_match']).mean()*100:+.1f} \\\\",
        ]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item R$\to$R: correct both stages. R$\to$W: degraded. "
        r"W$\to$R: improved. W$\to$W: wrong both stages.",
        r"\item $\Delta$ Acc = S2 $-$ S1 accuracy (percentage points). $N=16$ per mod type.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]

    out_path = out_dir / "transition_table.tex"
    out_path.write_text("\n".join(lines))
    print(f"  saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="")
    p.add_argument("--out_dir", default="results/frontier/viz")
    return p.parse_args()


def main():
    args    = parse_args()
    results_dir = Path("results/frontier")

    if args.input:
        result_path = Path(args.input)
    else:
        candidates = sorted(results_dir.glob("*__belief_revision.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No belief_revision JSONL found in {results_dir}")
        result_path = candidates[-1]

    print(f"[viz] Loading {result_path}")
    df = load_results(result_path)
    print(f"[viz] {len(df)} rows")

    slug    = result_path.stem
    out_dir = Path(args.out_dir) / slug
    print(f"[viz] Output → {out_dir}\n")

    v1_accuracy_by_modtype(df, out_dir)
    v2_transitions(df, out_dir)
    v3_hedging(df, out_dir)
    v4_accuracy_by_pde(df, out_dir)
    v5_traj_signal_transitions(df, out_dir)
    latex_transition_table(df, out_dir)
    write_combined_html(out_dir, slug)

    print("\n[viz] Done.")


if __name__ == "__main__":
    main()
