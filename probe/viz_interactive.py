"""
Interactive HTML report for PDE probe experiment.

Runnable before probe CSVs exist (RSA-only) and again once probes finish.
Same script, same output path — just re-run to update.

Usage:
    python probe/viz_interactive.py \
        --hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --results_dir probe/results/ \
        --output probe/results/report.html
"""
import argparse
import os
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_utils import (
    PDE_CLASSES, MOD_TYPES, BINARY_PROCESS_LABELS, BINARY_METHOD_LABELS,
    load_data, extract_label_arrays,
)

POOLS = ["mean_pool", "last_tok"]
LAYERS_TO_PLOT = [0, 7, 14, 21, 28]

LABEL_ORDER = (
    ["pde_class"]
    + [f"process_{p}" for p in BINARY_PROCESS_LABELS]
    + [f"method_{m}" for m in BINARY_METHOD_LABELS]
    + ["phys_valid"]
)
CHANCE = {
    "pde_class": 0.25,
    **{f"process_{p}": 0.5 for p in BINARY_PROCESS_LABELS},
    **{f"method_{m}": 0.5 for m in BINARY_METHOD_LABELS},
    "phys_valid": 0.5,
}

MT_COLORS = {mt: c for mt, c in zip(
    MOD_TYPES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
)}
POOL_COLORS = {"mean_pool": "#1f77b4", "last_tok": "#d62728"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_rdm(reps: np.ndarray) -> np.ndarray:
    reps_f = reps.astype(np.float32)
    norms = np.linalg.norm(reps_f, axis=1, keepdims=True) + 1e-8
    return 1.0 - (reps_f / norms) @ (reps_f / norms).T


def block_rdm_score(rdm: np.ndarray, pde_labels) -> float:
    within, between = [], []
    for i in range(len(rdm)):
        for j in range(i + 1, len(rdm)):
            if pde_labels[i] == pde_labels[j]:
                within.append(rdm[i, j])
            else:
                between.append(rdm[i, j])
    return float(np.mean(within) / (np.mean(between) + 1e-8)) if between else float("nan")


def sort_order(pde_labels, gt_samples, mod_types):
    pde_ord = {c: i for i, c in enumerate(PDE_CLASSES)}
    mt_ord = {m: i for i, m in enumerate(MOD_TYPES)}
    N = len(pde_labels)
    return np.array(sorted(range(N), key=lambda i: (
        pde_ord.get(str(pde_labels[i]).lower(), 99),
        str(gt_samples[i]),
        mt_ord.get(str(mod_types[i]), 99),
    )))


def wilson_ci(k: int, n: int) -> tuple:
    """Wilson binomial 95% CI."""
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def to_html(fig, first: bool = False) -> str:
    return pio.to_html(fig, full_html=False,
                       include_plotlyjs=first,
                       config={"responsive": True})


# ── Page 1 — Overview ────────────────────────────────────────────────────────

def build_overview(data: dict) -> str:
    N, L, D = data["mean_pool"].shape
    pde_labels = data["pde_classes"]
    mod_types = data["mod_types"]

    summary_rows = [
        ("Model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        ("Examples (N)", N),
        ("Transformer layers", f"{L - 1} (index 0 = embedding, 1–{L-1} = transformer)"),
        ("Hidden dim (D)", D),
        ("Pooling strategies", ", ".join(POOLS)),
        ("GT samples (unique base problems)", int(len(np.unique(data["gt_samples"])))),
        ("Mod-types per GT sample", len(MOD_TYPES)),
    ]
    summary_html = "<table><tr><th>Parameter</th><th>Value</th></tr>"
    for k, v in summary_rows:
        summary_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    summary_html += "</table>"

    pde_lower = np.array([p.lower() for p in pde_labels])
    dist_html = "<table><tr><th>mod_type \\ pde_class</th>"
    for pc in PDE_CLASSES:
        dist_html += f"<th>{pc}</th>"
    dist_html += "<th>Total</th></tr>"
    for mt in MOD_TYPES:
        dist_html += f"<tr><td><b>{mt}</b></td>"
        row_total = 0
        for pc in PDE_CLASSES:
            cnt = int(np.sum((mod_types == mt) & (pde_lower == pc)))
            dist_html += f"<td>{cnt}</td>"
            row_total += cnt
        dist_html += f"<td><b>{row_total}</b></td></tr>"
    dist_html += "</table>"

    label_rows = [("pde_class", "4-class", "wave / heat / burgers / navier-stokes", "0.25")]
    for p in BINARY_PROCESS_LABELS:
        label_rows.append((f"process_{p}", "binary", p, "0.50"))
    for m in BINARY_METHOD_LABELS:
        label_rows.append((f"method_{m}", "binary", m, "0.50"))
    label_rows.append(("phys_valid", "binary", "physically valid implementation", "0.50"))
    label_html = "<table><tr><th>Label</th><th>Type</th><th>Meaning</th><th>Chance</th></tr>"
    for name, typ, meaning, chance in label_rows:
        label_html += f"<tr><td><code>{name}</code></td><td>{typ}</td><td>{meaning}</td><td>{chance}</td></tr>"
    label_html += "</table>"

    return f"""
<div class="page" id="page-overview">
<h2>Overview</h2>
<h3>Experiment Summary</h3>
{summary_html}
<h3>Dataset Distribution (mod_type × pde_class)</h3>
{dist_html}
<h3>Labels Probed</h3>
{label_html}
</div>
"""


# ── Page 2 — Pilot Study (Pooled Probe) ──────────────────────────────────────

def build_pilot(dfs: dict, first_plotly: bool = False, majority: dict = None) -> str:
    if majority is None:
        majority = {}
    available = [p for p in POOLS if dfs.get(p) is not None]
    if not available:
        return f"""
<div class="page" id="page-pilot">
<h2>Pilot Study</h2>
<p><em>Probe results not yet available. Re-run
<code>probe/slurm/probe_pooled.slurm</code> once hidden states are extracted.</em></p>
</div>
"""

    first_label = LABEL_ORDER[0]

    # ── Accuracy vs layer ──
    acc_fig = go.Figure()
    acc_traces: list = []

    for label in LABEL_ORDER:
        visible = (label == first_label)
        for pool in available:
            df = dfs[pool]
            sub = df[(df["label"] == label) & (df["layer"] != "bow")].copy()
            sub["layer"] = sub["layer"].astype(int)
            sub = sub.sort_values("layer")
            layers = sub["layer"].tolist()
            accs = sub["accuracy"].tolist()
            ci_lo = sub["ci_low"].tolist()
            ci_hi = sub["ci_high"].tolist()
            color = POOL_COLORS[pool]
            rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"

            acc_fig.add_trace(go.Scatter(
                x=layers, y=ci_lo, mode="lines",
                line=dict(width=0), showlegend=False,
                visible=visible, hoverinfo="skip",
                legendgroup=f"{label}_{pool}", legendgrouptitle_text="",
            ))
            acc_traces.append(label)
            acc_fig.add_trace(go.Scatter(
                x=layers, y=ci_hi, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=rgba,
                showlegend=False, visible=visible, hoverinfo="skip",
                legendgroup=f"{label}_{pool}",
            ))
            acc_traces.append(label)
            hover = [f"Layer {l}: acc={a:.3f} [{lo:.3f}–{hi:.3f}] (16 folds, bootstrap n=10,000)"
                     for l, a, lo, hi in zip(layers, accs, ci_lo, ci_hi)]
            acc_fig.add_trace(go.Scatter(
                x=layers, y=accs, mode="lines+markers",
                name=pool, line=dict(color=color), marker=dict(size=5),
                hovertext=hover, hoverinfo="text",
                visible=visible, showlegend=True,
                legendgroup=f"{label}_{pool}",
            ))
            acc_traces.append(label)

            if pool == "mean_pool":
                bow_rows = df[(df["label"] == label) & (df["layer"] == "bow")]
                if not bow_rows.empty:
                    bv  = float(bow_rows["accuracy"].iloc[0])
                    blo = bow_rows["ci_low"].iloc[0]
                    bhi = bow_rows["ci_high"].iloc[0]
                    if not (np.isnan(float(blo)) or np.isnan(float(bhi))):
                        blo, bhi = float(blo), float(bhi)
                        acc_fig.add_trace(go.Scatter(
                            x=[layers[0], layers[-1], layers[-1], layers[0]],
                            y=[blo, blo, bhi, bhi],
                            fill="toself", fillcolor="rgba(255,127,14,0.15)",
                            line=dict(width=0), mode="lines",
                            name="BoW CI", hoverinfo="skip",
                            visible=visible, showlegend=False,
                        ))
                        acc_traces.append(label)
                    acc_fig.add_trace(go.Scatter(
                        x=[layers[0], layers[-1]], y=[bv, bv],
                        mode="lines", line=dict(dash="dash", color="#ff7f0e", width=1.5),
                        name="BoW", hoverinfo="skip",
                        visible=visible, showlegend=True,
                    ))
                    acc_traces.append(label)

        chance = CHANCE.get(label, 0.5)
        acc_fig.add_trace(go.Scatter(
            x=[0, 28], y=[chance, chance],
            mode="lines", line=dict(dash="dot", color="gray", width=1),
            name=f"chance={chance:.2f}", hoverinfo="skip",
            visible=visible, showlegend=True,
        ))
        acc_traces.append(label)
        maj = majority.get(label, chance)
        if abs(maj - chance) > 0.005:
            acc_fig.add_trace(go.Scatter(
                x=[0, 28], y=[maj, maj],
                mode="lines", line=dict(dash="dash", color="#9b59b6", width=1.2),
                name=f"majority={maj:.2f}", hoverinfo="skip",
                visible=visible, showlegend=True,
            ))
            acc_traces.append(label)

    acc_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [t == lbl for t in acc_traces]},
                   {"title": f"Pooled probe: accuracy vs layer — {lbl}"}])
        for lbl in LABEL_ORDER
    ]
    acc_fig.update_layout(
        title=f"Pooled probe: accuracy vs layer — {first_label}",
        xaxis_title="Layer (0 = embedding)",
        yaxis_title="LOGO-CV Accuracy",
        yaxis=dict(range=[0, 1.05]),
        height=450,
        updatemenus=[dict(buttons=acc_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
    )
    acc_html = to_html(acc_fig, first=first_plotly)

    # ── AUROC vs layer ──
    auroc_fig = go.Figure()
    auroc_traces: list = []

    for label in LABEL_ORDER:
        visible = (label == first_label)
        for pool in available:
            df = dfs[pool]
            sub = df[(df["label"] == label) & (df["layer"] != "bow")].copy()
            sub["layer"] = sub["layer"].astype(int)
            sub = sub.sort_values("layer")
            if "auroc" not in sub.columns:
                continue
            sub_valid = sub[sub["auroc"].notna() & (sub["auroc"] != float("nan"))]
            # drop rows where auroc is NaN (undefined for 4-class without OvR)
            sub_valid = sub.dropna(subset=["auroc"])
            if sub_valid.empty:
                continue
            layers = sub_valid["layer"].tolist()
            aurocs = sub_valid["auroc"].tolist()
            ci_lo = (sub_valid["auroc_ci_low"].tolist() if "auroc_ci_low" in sub_valid.columns
                     else sub_valid["auroc"].tolist())
            ci_hi = (sub_valid["auroc_ci_high"].tolist() if "auroc_ci_high" in sub_valid.columns
                     else sub_valid["auroc"].tolist())
            color = POOL_COLORS[pool]
            rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"
            auroc_fig.add_trace(go.Scatter(
                x=layers, y=ci_lo, mode="lines", line=dict(width=0),
                showlegend=False, visible=visible, hoverinfo="skip",
                legendgroup=f"{label}_{pool}", legendgrouptitle_text="",
            ))
            auroc_traces.append(label)
            auroc_fig.add_trace(go.Scatter(
                x=layers, y=ci_hi, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=rgba,
                showlegend=False, visible=visible, hoverinfo="skip",
                legendgroup=f"{label}_{pool}",
            ))
            auroc_traces.append(label)
            hover = [f"Layer {l}: AUROC={a:.3f} [{lo:.3f}–{hi:.3f}]"
                     for l, a, lo, hi in zip(layers, aurocs, ci_lo, ci_hi)]
            auroc_fig.add_trace(go.Scatter(
                x=layers, y=aurocs, mode="lines+markers",
                name=pool, line=dict(color=color), marker=dict(size=5),
                hovertext=hover, hoverinfo="text",
                visible=visible, showlegend=True,
                legendgroup=f"{label}_{pool}",
            ))
            auroc_traces.append(label)

        auroc_fig.add_trace(go.Scatter(
            x=[0, 28], y=[0.5, 0.5],
            mode="lines", line=dict(dash="dot", color="gray", width=1),
            name="chance=0.50", hoverinfo="skip",
            visible=visible, showlegend=True,
        ))
        auroc_traces.append(label)

        if "mean_pool" in available:
            bow_auroc_row = dfs["mean_pool"][
                (dfs["mean_pool"]["label"] == label) & (dfs["mean_pool"]["layer"] == "bow")
            ]
            if not bow_auroc_row.empty and "auroc" in bow_auroc_row.columns:
                bav = bow_auroc_row["auroc"].iloc[0]
                if not (isinstance(bav, float) and np.isnan(bav)):
                    bav = float(bav)
                    balo = bow_auroc_row["auroc_ci_low"].iloc[0]
                    bahi = bow_auroc_row["auroc_ci_high"].iloc[0]
                    if not (np.isnan(float(balo)) or np.isnan(float(bahi))):
                        balo, bahi = float(balo), float(bahi)
                        auroc_fig.add_trace(go.Scatter(
                            x=[0, 28, 28, 0],
                            y=[balo, balo, bahi, bahi],
                            fill="toself", fillcolor="rgba(255,127,14,0.15)",
                            line=dict(width=0), mode="lines",
                            name="BoW AUROC CI", hoverinfo="skip",
                            visible=visible, showlegend=False,
                        ))
                        auroc_traces.append(label)
                    auroc_fig.add_trace(go.Scatter(
                        x=[0, 28], y=[bav, bav],
                        mode="lines", line=dict(dash="dash", color="#ff7f0e", width=1.5),
                        name="BoW", hoverinfo="skip",
                        visible=visible, showlegend=True,
                    ))
                    auroc_traces.append(label)

    auroc_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [t == lbl for t in auroc_traces]},
                   {"title": f"Pooled probe: AUROC vs layer — {lbl}"}])
        for lbl in LABEL_ORDER
    ]
    auroc_fig.update_layout(
        title=f"Pooled probe: AUROC vs layer — {first_label}",
        xaxis_title="Layer (0 = embedding)",
        yaxis_title="LOGO-CV AUROC",
        yaxis=dict(range=[0.4, 1.05]),
        height=420,
        updatemenus=[dict(buttons=auroc_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
    )
    auroc_html = to_html(auroc_fig)

    # Pre-extract BoW per-mod-type values (same in all pools; use first available)
    bow_mt_vals: dict = {}  # label → {mt: acc}
    _bow_pool = available[0]
    for label in LABEL_ORDER:
        bow_row = dfs[_bow_pool][(dfs[_bow_pool]["label"] == label) &
                                  (dfs[_bow_pool]["layer"] == "bow")]
        if not bow_row.empty:
            bow_mt_vals[label] = {
                mt: float(bow_row.iloc[0].get(f"mt_{mt}", float("nan")))
                for mt in MOD_TYPES
            }
        else:
            bow_mt_vals[label] = {mt: float("nan") for mt in MOD_TYPES}

    # ── Mod-type breakdown at best layer ──
    mt_fig = go.Figure()
    mt_traces: list = []

    for label in LABEL_ORDER:
        visible = (label == first_label)
        for pool in available:
            df = dfs[pool]
            sub = df[(df["label"] == label) & (df["layer"] != "bow")].copy()
            sub["layer"] = sub["layer"].astype(int)
            if sub.empty:
                continue
            best_row = sub.loc[sub["accuracy"].idxmax()]
            mt_accs, err_lo, err_hi, hover_mt = [], [], [], []
            for mt in MOD_TYPES:
                val = float(best_row.get(f"mt_{mt}", float("nan")))
                mt_accs.append(val)
                if not np.isnan(val):
                    k = round(val * 16)
                    lo, hi = wilson_ci(k, 16)
                    err_lo.append(val - lo)
                    err_hi.append(hi - val)
                    hover_mt.append(f"{mt}<br>acc={val:.3f} [{lo:.3f}–{hi:.3f}]<br>"
                                    f"Wilson CI (n=16 folds)")
                else:
                    err_lo.append(0); err_hi.append(0)
                    hover_mt.append(f"{mt}<br>N/A")
            mt_fig.add_trace(go.Bar(
                x=MOD_TYPES, y=mt_accs, name=pool,
                marker_color=POOL_COLORS[pool],
                error_y=dict(type="data", symmetric=False,
                             array=err_hi, arrayminus=err_lo),
                hovertext=hover_mt, hoverinfo="text",
                visible=visible, showlegend=True,
            ))
            mt_traces.append(label)

        # BoW bar group
        bow_vals = [bow_mt_vals[label].get(mt, float("nan")) for mt in MOD_TYPES]
        bow_err_lo, bow_err_hi, bow_hover = [], [], []
        for mt, v in zip(MOD_TYPES, bow_vals):
            if not np.isnan(v):
                k = round(v * 16)
                lo, hi = wilson_ci(k, 16)
                bow_err_lo.append(v - lo)
                bow_err_hi.append(hi - v)
                bow_hover.append(f"{mt}<br>BoW acc={v:.3f} [{lo:.3f}–{hi:.3f}]<br>Wilson CI (n=16)")
            else:
                bow_err_lo.append(0); bow_err_hi.append(0)
                bow_hover.append(f"{mt}<br>N/A")
        mt_fig.add_trace(go.Bar(
            x=MOD_TYPES, y=bow_vals, name="BoW",
            marker_color="#ff7f0e",
            error_y=dict(type="data", symmetric=False, array=bow_err_hi, arrayminus=bow_err_lo),
            hovertext=bow_hover, hoverinfo="text",
            visible=visible, showlegend=True,
        ))
        mt_traces.append(label)

        chance = CHANCE.get(label, 0.5)
        mt_fig.add_trace(go.Scatter(
            x=MOD_TYPES, y=[chance] * len(MOD_TYPES),
            mode="lines", line=dict(dash="dot", color="gray", width=1),
            name=f"chance={chance:.2f}", hoverinfo="skip",
            visible=visible, showlegend=True,
        ))
        mt_traces.append(label)
        maj = majority.get(label, chance)
        if abs(maj - chance) > 0.005:
            mt_fig.add_trace(go.Scatter(
                x=MOD_TYPES, y=[maj] * len(MOD_TYPES),
                mode="lines", line=dict(dash="dash", color="#9b59b6", width=1.2),
                name=f"majority={maj:.2f}", hoverinfo="skip",
                visible=visible, showlegend=True,
            ))
            mt_traces.append(label)

    mt_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [t == lbl for t in mt_traces]},
                   {"title": f"Pooled probe: mod-type breakdown — {lbl} (best layer)"}])
        for lbl in LABEL_ORDER
    ]
    mt_fig.update_layout(
        title=f"Pooled probe: mod-type breakdown — {first_label} (best layer)",
        xaxis_title="mod_type",
        yaxis_title="Accuracy at best layer",
        yaxis=dict(range=[0, 1.05]),
        barmode="group", height=420,
        updatemenus=[dict(buttons=mt_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
    )
    mt_html = to_html(mt_fig)

    # ── Summary tables: one per label, best-layer accuracy per mod_type ──
    # Pre-extract best-row values for each (label, pool)
    best_vals: dict = {}  # (label, pool) → {mt: acc}
    best_layers: dict = {}  # (label, pool) → layer index
    for pool in available:
        df = dfs[pool]
        for label in LABEL_ORDER:
            sub = df[(df["label"] == label) & (df["layer"] != "bow")].copy()
            sub["layer"] = sub["layer"].astype(int)
            if sub.empty:
                best_vals[(label, pool)] = {}
                best_layers[(label, pool)] = "—"
                continue
            best_row = sub.loc[sub["accuracy"].idxmax()]
            best_layers[(label, pool)] = int(best_row["layer"])
            best_vals[(label, pool)] = {
                mt: float(best_row.get(f"mt_{mt}", float("nan"))) for mt in MOD_TYPES
            }

    # Extract BoW accuracy per (label, pool)
    bow_acc: dict = {}
    for pool in available:
        df = dfs[pool]
        for label in LABEL_ORDER:
            bow_row = df[(df["label"] == label) & (df["layer"] == "bow")]
            bow_acc[(label, pool)] = (
                float(bow_row["accuracy"].iloc[0]) if not bow_row.empty else float("nan")
            )

    tables_html = ""
    for label in LABEL_ORDER:
        layer_note = " / ".join(
            f"{pool}: layer {best_layers.get((label, pool), '—')}"
            for pool in available
        )
        bow_note = " / ".join(
            f"{pool} BoW: {'—' if np.isnan(bow_acc.get((label, pool), float('nan'))) else f'{bow_acc[(label, pool)]:.3f}'}"
            for pool in available
        )
        chance = CHANCE.get(label, 0.5)
        maj = majority.get(label, chance)
        maj_note = f"majority={maj:.3f}"
        # Header: one group of 4 columns per pool, plus BoW at end
        pool_headers = "".join(
            f'<th colspan="4" style="text-align:center;background:#e8eef4;">{pool}</th>'
            for pool in available
        ) + '<th style="text-align:center;background:#fde8cc;">BoW</th>'
        subheaders = "".join(
            "<th>acc</th><th>CI low</th><th>CI high</th><th>variance</th>"
            for _ in available
        ) + "<th>acc</th>"
        hdr = f"<tr><th rowspan='2'>mod_type</th>{pool_headers}</tr><tr>{subheaders}</tr>"

        rows_html = ""
        for mt in MOD_TYPES:
            cells = f"<td><b>{mt}</b></td>"
            for pool in available:
                val = best_vals.get((label, pool), {}).get(mt, float("nan"))
                if np.isnan(val):
                    cells += "<td>—</td><td>—</td><td>—</td><td>—</td>"
                else:
                    k = round(val * 16)
                    lo, hi = wilson_ci(k, 16)
                    var = val * (1 - val)
                    cells += (f"<td>{val:.3f}</td><td>{lo:.3f}</td>"
                               f"<td>{hi:.3f}</td><td>{var:.4f}</td>")
            bv = bow_mt_vals.get(label, {}).get(mt, float("nan"))
            cells += f"<td>{'—' if np.isnan(bv) else f'{bv:.3f}'}</td>"
            rows_html += f"<tr>{cells}</tr>"

        # Majority reference row (same value across all mod_types; CI/variance not applicable)
        maj_cells = "<td><i>majority</i></td>"
        for _ in available:
            maj_cells += f"<td>{maj:.3f}</td><td>—</td><td>—</td><td>—</td>"
        maj_cells += "<td>—</td>"  # BoW column
        rows_html += f'<tr style="background:#fff8f0;color:#888;">{maj_cells}</tr>'

        tables_html += f"""
<div style="margin-bottom:24px;">
  <h4 style="margin-bottom:6px;">{label} <small style="font-weight:normal;color:#888;">({layer_note} &nbsp;|&nbsp; {bow_note} &nbsp;|&nbsp; {maj_note})</small></h4>
  <table>{hdr}{rows_html}</table>
</div>"""

    return f"""
<div class="page" id="page-pilot">
<h2>Pilot Study</h2>
<p>LOGO-CV grouped by gt_sample (16 folds). All 128 rows used for training and testing.
Bootstrap 95% CI (n=10,000 resamples) over per-fold accuracies.
Available poolings: {", ".join(available)}.</p>
<h3>Accuracy vs Layer</h3>
{acc_html}
<h3>AUROC vs Layer</h3>
<p><small>AUROC undefined for 4-class pde_class (shown as empty). Binary labels only.</small></p>
{auroc_html}
<h3>Mod-type Breakdown (best layer per label)</h3>
<p><small>Error bars: Wilson binomial 95% CI (n=16, one binary outcome per LOGO fold per mod_type).</small></p>
{mt_html}
<h3>Summary Tables</h3>
<p><small>Accuracy at the best layer per label (same as bar chart). CI: Wilson 95% binomial (n=16 folds).
Variance: p(1−p) of binary fold outcomes.</small></p>
{tables_html}
</div>
"""


# ── Page 3 — RSA Analysis ────────────────────────────────────────────────────

def _class_dist_matrix(reps: np.ndarray, pde_labels: np.ndarray) -> np.ndarray:
    """4×4 mean pairwise cosine distance matrix between pde_class groups."""
    classes = PDE_CLASSES
    mat = np.full((len(classes), len(classes)), float("nan"))
    reps_f = reps.astype(np.float32)
    norms = np.linalg.norm(reps_f, axis=1, keepdims=True) + 1e-8
    reps_n = reps_f / norms
    for i, ci in enumerate(classes):
        idx_i = np.where(np.array([p.lower() for p in pde_labels]) == ci)[0]
        for j, cj in enumerate(classes):
            idx_j = np.where(np.array([p.lower() for p in pde_labels]) == cj)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            sims = reps_n[idx_i] @ reps_n[idx_j].T
            dists = 1.0 - sims
            if i == j:
                # within-class: exclude diagonal (self-distance = 0)
                np.fill_diagonal(dists, float("nan"))
            mat[i, j] = float(np.nanmean(dists))
    return mat


def build_rsa(data: dict, first_plotly: bool = False) -> str:
    pde_labels = data["pde_classes"]
    phys_valid_labels = np.array(["valid" if v else "invalid" for v in data["phys_valid"]])
    mod_type_labels = data["mod_types"]
    gt_samples = data["gt_samples"]
    N, L, D = data["mean_pool"].shape

    print("  Computing block scores and class distance matrices...", flush=True)
    block_scores: dict = {p: [] for p in POOLS}
    valid_block_scores: dict = {p: [] for p in POOLS}
    # class dist matrices at selected layers only (expensive to store all 29)
    class_dists: dict = {p: {} for p in POOLS}

    for pool in POOLS:
        reps_all = data[pool].astype(np.float32)
        for l in range(L):
            rdm = compute_rdm(reps_all[:, l, :])
            block_scores[pool].append(block_rdm_score(rdm, pde_labels))
            valid_block_scores[pool].append(block_rdm_score(rdm, phys_valid_labels))
        # class dist at every layer for the dropdown
        for l in range(L):
            class_dists[pool][l] = _class_dist_matrix(reps_all[:, l, :], pde_labels)
        print(f"    {pool}: pde best={min(block_scores[pool]):.4f} @ layer "
              f"{int(np.argmin(block_scores[pool]))}  |  "
              f"valid best={min(valid_block_scores[pool]):.4f} @ layer "
              f"{int(np.argmin(valid_block_scores[pool]))}", flush=True)

    # ── Figure 1: 4×4 class distance matrix with layer dropdown ──
    cdm_fig = go.Figure()
    cdm_labels = []
    for pool in POOLS:
        for l in range(L):
            mat = class_dists[pool][l]
            hover = [[f"{PDE_CLASSES[i]} → {PDE_CLASSES[j]}<br>mean dist={mat[i,j]:.3f}"
                      if not np.isnan(mat[i,j]) else f"{PDE_CLASSES[i]} → {PDE_CLASSES[j]}<br>N/A"
                      for j in range(len(PDE_CLASSES))]
                     for i in range(len(PDE_CLASSES))]
            cdm_fig.add_trace(go.Heatmap(
                z=mat,
                x=PDE_CLASSES, y=PDE_CLASSES,
                colorscale="RdYlBu_r", zmin=0, zmax=1,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                colorbar=dict(title="Mean cosine dist"),
                visible=False,
                texttemplate="%{z:.3f}",
            ))
            cdm_labels.append(f"Layer {l} [{pool}]")

    cdm_fig.data[0].visible = True
    n_cdm = len(cdm_labels)
    cdm_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [j == i for j in range(n_cdm)]},
                   {"title": f"PDE class distance matrix — {lbl}"}])
        for i, lbl in enumerate(cdm_labels)
    ]
    cdm_fig.update_layout(
        title=f"PDE class distance matrix — {cdm_labels[0]}",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=520, height=480,
        updatemenus=[dict(buttons=cdm_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.14, yanchor="top")],
    )
    cdm_html = to_html(cdm_fig, first=first_plotly)

    # ── Figure 2: Block score vs layer — pde_class + phys_valid ──
    block_fig = go.Figure()
    for pool in POOLS:
        color = POOL_COLORS[pool]
        block_fig.add_trace(go.Scatter(
            x=list(range(L)), y=block_scores[pool],
            mode="lines+markers", marker=dict(size=4),
            name=f"pde_class [{pool}]",
            line=dict(color=color, width=2),
        ))
        # phys_valid as dashed version of same color
        block_fig.add_trace(go.Scatter(
            x=list(range(L)), y=valid_block_scores[pool],
            mode="lines+markers", marker=dict(size=4, symbol="diamond"),
            name=f"phys_valid [{pool}]",
            line=dict(color=color, width=1.5, dash="dash"),
        ))
    block_fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                        annotation_text="ratio=1 (no clustering)",
                        annotation_position="bottom right")
    block_fig.update_layout(
        title="RSA block score vs layer — pde_class and phys_valid",
        xaxis_title="Layer (0 = embedding)",
        yaxis_title="Within / between ratio  (lower = better clustering)",
        height=400, legend=dict(orientation="h", y=-0.2),
    )
    block_html = to_html(block_fig)

    # ── Figure 3: Mod-type drift from Comm_Valid (same gt_sample) ──
    # For each gt_sample and each non-Comm_Valid mod_type, compute cosine distance
    # from the Comm_Valid representation at the best pde_class layer.
    drift_htmls = []
    for pool in POOLS:
        best_l = int(np.argmin(block_scores[pool]))
        reps_best = data[pool].astype(np.float32)[:, best_l, :]
        norms = np.linalg.norm(reps_best, axis=1, keepdims=True) + 1e-8
        reps_n = reps_best / norms

        drifts: dict = {mt: [] for mt in MOD_TYPES if mt != "Comm_Valid"}
        for gt in np.unique(gt_samples):
            gt_mask = gt_samples == gt
            cv_idx = np.where(gt_mask & (mod_type_labels == "Comm_Valid"))[0]
            if len(cv_idx) == 0:
                continue
            cv_rep = reps_n[cv_idx[0]]
            for mt in MOD_TYPES:
                if mt == "Comm_Valid":
                    continue
                mt_idx = np.where(gt_mask & (mod_type_labels == mt))[0]
                if len(mt_idx) == 0:
                    continue
                dist = float(1.0 - float(cv_rep @ reps_n[mt_idx[0]]))
                drifts[mt].append(dist)

        mt_order = [mt for mt in MOD_TYPES if mt != "Comm_Valid"]
        means = [float(np.mean(drifts[mt])) if drifts[mt] else float("nan") for mt in mt_order]
        sems  = [float(np.std(drifts[mt]) / np.sqrt(len(drifts[mt]))) if len(drifts[mt]) > 1 else 0.0
                 for mt in mt_order]
        hover_drift = [
            f"{mt}<br>mean dist from Comm_Valid={m:.4f}<br>SE={s:.4f}<br>n={len(drifts[mt])} gt_samples"
            for mt, m, s in zip(mt_order, means, sems)
        ]
        colors_drift = [MT_COLORS.get(mt, "#888") for mt in mt_order]

        drift_fig = go.Figure()
        drift_fig.add_trace(go.Bar(
            x=mt_order, y=means,
            error_y=dict(type="data", array=sems, visible=True),
            marker_color=colors_drift,
            hovertext=hover_drift, hoverinfo="text",
            showlegend=False,
        ))
        drift_fig.update_layout(
            title=f"Mod-type drift from Comm_Valid — layer {best_l} [{pool}]",
            xaxis_title="mod_type",
            yaxis_title="Mean cosine distance",
            height=400,
        )
        drift_htmls.append(
            "<p><small>Higher = representation moved further from the GT-comment baseline "
            "(Comm_Valid at same gt_sample). Cosine distance at the best pde_class layer.</small></p>"
            + to_html(drift_fig)
        )

    return f"""
<div class="page" id="page-rsa">
<h2>RSA Analysis</h2>

<h3>PDE Class Distance Matrix</h3>
<p><small>Mean pairwise cosine distance between pde_class groups (averaged over all rows in each class).
Diagonal = within-class mean distance (self-pairs excluded). Lower diagonal = tighter within-class geometry.
Use the dropdown to scan across layers.</small></p>
{cdm_html}

<h3>Block Score vs Layer</h3>
<p><small>Within/between cosine distance ratio for pde_class (solid) and phys_valid (dashed) groupings.
Values below 1.0 indicate within-group representations are closer than between-group.
Lower = better clustering.</small></p>
{block_html}

<h3>Mod-type Drift from Comm_Valid</h3>
<p><small>For each gt_sample, mean cosine distance from the Comm_Valid representation to each other
mod_type at the best pde_class layer. Quantifies how much each manipulation shifts the representation.
Error bars = SE across 16 gt_samples.</small></p>
{"".join(drift_htmls)}
</div>
"""


# ── Page 4 — Hyperparam Sweep ────────────────────────────────────────────────

C_COLORS = {0.01: "#1f77b4", 0.1: "#2ca02c", 1.0: "#d62728", 10.0: "#9467bd"}
REPR_DASH = {"raw": "solid", "pca20": "dash"}


def build_hyperparam(hp_dfs: dict, first_plotly: bool = False) -> str:
    """
    hp_dfs keyed by (representation, pool) → DataFrame
    e.g. ("raw", "mean_pool") → df
    """
    available_keys = [k for k, v in hp_dfs.items() if v is not None]
    if not available_keys:
        return """
<div class="page" id="page-hyperparam">
<h2>Hyperparam Sweep</h2>
<p><em>Results not yet available. Re-run once probe_raw_grid and probe_pca_grid jobs complete.</em></p>
</div>
"""

    reprs = sorted({k[0] for k in available_keys})
    pools_avail = sorted({k[1] for k in available_keys})
    first_label = LABEL_ORDER[0]
    first_pool = pools_avail[0]

    # ── Figure 1: C sensitivity — accuracy vs layer, 4 C curves per repr ──
    # One figure per pool (pool selector via button group outside figure)
    # Traces: (label, C, repr) — label dropdown controls visibility
    c_figs_html = []
    for pool in pools_avail:
        fig = go.Figure()
        traces_labels = []
        all_Cs = sorted(C_COLORS.keys())

        for label in LABEL_ORDER:
            visible = (label == first_label)
            chance = CHANCE.get(label, 0.5)
            for repr_ in reprs:
                df = hp_dfs.get((repr_, pool))
                if df is None:
                    continue
                sub = df[df["label"] == label].copy()
                sub["layer"] = sub["layer"].astype(int)
                for C in sorted(sub["C"].unique()):
                    c_sub = sub[sub["C"] == C].sort_values("layer")
                    if c_sub.empty:
                        continue
                    layers = c_sub["layer"].tolist()
                    accs = c_sub["accuracy"].tolist()
                    ci_lo = c_sub["ci_low"].tolist()
                    ci_hi = c_sub["ci_high"].tolist()
                    color = C_COLORS.get(C, "#888")
                    dash = REPR_DASH.get(repr_, "solid")
                    rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)"

                    grp = f"{label}_C{C}_{repr_}_{pool}"
                    fig.add_trace(go.Scatter(
                        x=layers, y=ci_lo, mode="lines", line=dict(width=0),
                        showlegend=False, visible=visible, hoverinfo="skip",
                        legendgroup=grp, legendgrouptitle_text="",
                    ))
                    traces_labels.append(label)
                    fig.add_trace(go.Scatter(
                        x=layers, y=ci_hi, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=rgba,
                        showlegend=False, visible=visible, hoverinfo="skip",
                        legendgroup=grp,
                    ))
                    traces_labels.append(label)
                    hover = [f"Layer {l}: acc={a:.3f} [{lo:.3f}–{hi:.3f}]<br>C={C} repr={repr_}"
                             for l, a, lo, hi in zip(layers, accs, ci_lo, ci_hi)]
                    fig.add_trace(go.Scatter(
                        x=layers, y=accs, mode="lines+markers",
                        name=f"C={C} [{repr_}]",
                        line=dict(color=color, dash=dash, width=2),
                        marker=dict(size=4),
                        hovertext=hover, hoverinfo="text",
                        visible=visible, showlegend=True,
                        legendgroup=grp,
                    ))
                    traces_labels.append(label)

            fig.add_trace(go.Scatter(
                x=[0, 28], y=[chance, chance],
                mode="lines", line=dict(dash="dot", color="gray", width=1),
                name=f"chance={chance:.2f}", hoverinfo="skip",
                visible=visible, showlegend=True,
            ))
            traces_labels.append(label)

        buttons = [
            dict(label=lbl, method="update",
                 args=[{"visible": [t == lbl for t in traces_labels]},
                       {"title": f"C sensitivity — {lbl} [{pool}]"}])
            for lbl in LABEL_ORDER
        ]
        fig.update_layout(
            title=f"C sensitivity — {first_label} [{pool}]",
            xaxis_title="Layer (0 = embedding)",
            yaxis_title="LOGO-CV Accuracy",
            yaxis=dict(range=[0, 1.05]),
            height=450,
            legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
            updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                              x=0.0, xanchor="left", y=1.12, yanchor="top")],
        )
        is_first = first_plotly and (pool == pools_avail[0])
        c_figs_html.append(
            f"<h4>{pool}</h4>"
            "<p><small>Solid = raw (D=3584) · Dashed = PCA-20 · Color = C value</small></p>"
            + to_html(fig, first=is_first)
        )

    # ── Figure 2: raw vs PCA-20 at best layer — bar chart per C ──
    repr_fig = go.Figure()
    repr_traces = []
    REPR_COLORS = {"raw": "#1f77b4", "pca20": "#e377c2"}

    for label in LABEL_ORDER:
        visible = (label == first_label)
        for pool in pools_avail:
            for repr_ in reprs:
                df = hp_dfs.get((repr_, pool))
                if df is None:
                    continue
                sub = df[df["label"] == label].copy()
                sub["layer"] = sub["layer"].astype(int)
                Cs = sorted(sub["C"].unique())
                best_accs, err_lo, err_hi, hovers = [], [], [], []
                for C in Cs:
                    c_sub = sub[sub["C"] == C]
                    if c_sub.empty:
                        best_accs.append(float("nan")); err_lo.append(0); err_hi.append(0)
                        hovers.append("N/A")
                        continue
                    best_row = c_sub.loc[c_sub["accuracy"].idxmax()]
                    acc = float(best_row["accuracy"])
                    lo = float(best_row["ci_low"])
                    hi = float(best_row["ci_high"])
                    best_accs.append(acc)
                    err_lo.append(acc - lo)
                    err_hi.append(hi - acc)
                    hovers.append(f"C={C} repr={repr_} [{pool}]<br>acc={acc:.3f} [{lo:.3f}–{hi:.3f}]<br>best layer={int(best_row['layer'])}")

                repr_fig.add_trace(go.Bar(
                    x=[f"C={C}" for C in Cs],
                    y=best_accs,
                    name=f"{repr_} [{pool}]",
                    marker_color=REPR_COLORS.get(repr_, "#888"),
                    opacity=0.85 if pool == "mean_pool" else 0.5,
                    error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo),
                    hovertext=hovers, hoverinfo="text",
                    visible=visible, showlegend=True,
                ))
                repr_traces.append(label)

        chance = CHANCE.get(label, 0.5)
        repr_fig.add_trace(go.Scatter(
            x=["C=0.01", "C=0.1", "C=1.0", "C=10.0"], y=[chance] * 4,
            mode="lines", line=dict(dash="dot", color="gray", width=1),
            name=f"chance={chance:.2f}", hoverinfo="skip",
            visible=visible, showlegend=True,
        ))
        repr_traces.append(label)

    repr_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [t == lbl for t in repr_traces]},
                   {"title": f"raw vs PCA-20 at best layer — {lbl}"}])
        for lbl in LABEL_ORDER
    ]
    repr_fig.update_layout(
        title=f"raw vs PCA-20 at best layer — {first_label}",
        xaxis_title="C (regularization)",
        yaxis_title="Best-layer accuracy",
        yaxis=dict(range=[0, 1.05]),
        barmode="group", height=420,
        updatemenus=[dict(buttons=repr_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
    )
    repr_html = to_html(repr_fig)

    # ── Figure 3: phys_valid AUROC vs layer (only label with non-NaN AUROC) ──
    auroc_fig = go.Figure()
    auroc_traces_labels = []
    for pool in pools_avail:
        for repr_ in reprs:
            df = hp_dfs.get((repr_, pool))
            if df is None:
                continue
            sub = df[df["label"] == "phys_valid"].copy()
            sub["layer"] = sub["layer"].astype(int)
            for C in sorted(sub["C"].unique()):
                c_sub = sub[(sub["C"] == C)].dropna(subset=["auroc"]).sort_values("layer")
                if c_sub.empty:
                    continue
                color = C_COLORS.get(C, "#888")
                dash = REPR_DASH.get(repr_, "solid")
                layers = c_sub["layer"].tolist()
                aurocs = c_sub["auroc"].tolist()
                ci_lo = c_sub["auroc_ci_low"].tolist()
                ci_hi = c_sub["auroc_ci_high"].tolist()
                rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)"
                grp_hp = f"C{C}_{repr_}_{pool}"
                auroc_fig.add_trace(go.Scatter(
                    x=layers, y=ci_lo, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=grp_hp, legendgrouptitle_text="",
                ))
                auroc_fig.add_trace(go.Scatter(
                    x=layers, y=ci_hi, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=rgba,
                    showlegend=False, hoverinfo="skip",
                    legendgroup=grp_hp,
                ))
                hover = [f"Layer {l}: AUROC={a:.3f} [{lo:.3f}–{hi:.3f}]<br>C={C} repr={repr_} [{pool}]"
                         for l, a, lo, hi in zip(layers, aurocs, ci_lo, ci_hi)]
                auroc_fig.add_trace(go.Scatter(
                    x=layers, y=aurocs, mode="lines+markers",
                    name=f"C={C} {repr_} [{pool}]",
                    line=dict(color=color, dash=dash, width=2),
                    marker=dict(size=4),
                    hovertext=hover, hoverinfo="text",
                    legendgroup=grp_hp,
                ))

    auroc_fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                        annotation_text="chance", annotation_position="bottom right")
    auroc_fig.update_layout(
        title="phys_valid AUROC vs layer — all C × repr × pool",
        xaxis_title="Layer (0 = embedding)",
        yaxis_title="LOGO-CV AUROC (phys_valid only)",
        yaxis=dict(range=[0.4, 1.05]),
        height=450,
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
    )
    auroc_html = (
        "<p><small>AUROC defined only for phys_valid — all other labels are constant "
        "within LOGO fold and yield undefined AUROC.</small></p>"
        + to_html(auroc_fig)
    )

    return f"""
<div class="page" id="page-hyperparam">
<h2>Hyperparam Sweep</h2>
<p>Grid: C ∈ {{0.01, 0.1, 1.0, 10.0}} × representation ∈ {{raw D=3584, PCA-20}} × pooling ∈ {{mean_pool, last_tok}}.
PCA fit inside each LOGO fold on training data only. LOGO-CV grouped by gt_sample (16 folds).
Solid lines = raw; dashed lines = PCA-20. Color encodes C value.</p>

<h3>C Sensitivity — Accuracy vs Layer</h3>
<p><small>Each curve is one (C, representation) combination. Use label dropdown to switch target.
If layer-ordering conclusions are robust, curves should maintain the same relative ranking across C values.</small></p>
{"".join(c_figs_html)}

<h3>raw vs PCA-20 at Best Layer</h3>
<p><small>For each C, the best accuracy achievable at any layer. Compares raw vs PCA-20 compression.
Opacity: solid fill = mean_pool, faded = last_tok.</small></p>
{repr_html}

<h3>phys_valid AUROC vs Layer</h3>
<p><small>Only phys_valid has non-NaN AUROC in the LOGO-CV setup (it varies across mod_types within a fold).
All C × repr × pool combinations overlaid.</small></p>
{auroc_html}
</div>
"""


# ── Page 5 — Validity Split ───────────────────────────────────────────────────

SPLIT_STYLES = {
    "full":   dict(color="#2c3e50", dash="solid",  width=2.5),
    "comm":   dict(color="#27ae60", dash="solid",  width=2.0),
    "nocomm": dict(color="#e74c3c", dash="dash",   width=2.0),
}
SPLIT_LABELS = {"full": "Full (n=128)", "comm": "Comm (n=64)", "nocomm": "NoComm (n=64)"}


def build_validity_split(vs_dfs: dict, first_plotly: bool = False) -> str:
    """
    vs_dfs keyed by pool → DataFrame with columns:
    split, layer, pool, C, n_rows, accuracy, ci_low, ci_high, auroc, auroc_ci_low, auroc_ci_high
    """
    available = [p for p in POOLS if vs_dfs.get(p) is not None]
    if not available:
        return """
<div class="page" id="page-validity-split">
<h2>Validity Split</h2>
<p><em>Results not yet available. Re-run once probe_validity_split job completes.</em></p>
</div>
"""

    first_pool = available[0]

    def _build_metric_fig(metric: str, yrange: list, ylabel: str,
                          threshold_lines: list, title_prefix: str,
                          is_first: bool = False) -> str:
        fig = go.Figure()
        for pool in available:
            df = vs_dfs[pool]
            for split in ["full", "comm", "nocomm"]:
                sub = df[df["split"] == split].copy()
                sub["layer"] = sub["layer"].astype(int)
                sub = sub.sort_values("layer")
                if sub.empty or sub[metric].isna().all():
                    continue
                ci_lo_col = "ci_low" if metric == "accuracy" else "auroc_ci_low"
                ci_hi_col = "ci_high" if metric == "accuracy" else "auroc_ci_high"
                layers = sub["layer"].tolist()
                vals = sub[metric].tolist()
                ci_lo = sub[ci_lo_col].tolist() if ci_lo_col in sub.columns else vals
                ci_hi = sub[ci_hi_col].tolist() if ci_hi_col in sub.columns else vals
                style = SPLIT_STYLES[split]
                color = style["color"]
                rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)"
                label = SPLIT_LABELS[split]
                pool_suffix = f" [{pool}]" if len(available) > 1 else ""

                grp_vs = f"{split}_{pool}"
                fig.add_trace(go.Scatter(
                    x=layers, y=ci_lo, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=grp_vs, legendgrouptitle_text="",
                ))
                fig.add_trace(go.Scatter(
                    x=layers, y=ci_hi, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=rgba,
                    showlegend=False, hoverinfo="skip",
                    legendgroup=grp_vs,
                ))
                hover = [f"Layer {l}: {metric}={v:.3f} [{lo:.3f}–{hi:.3f}]<br>{label}{pool_suffix}"
                         for l, v, lo, hi in zip(layers, vals, ci_lo, ci_hi)]
                fig.add_trace(go.Scatter(
                    x=layers, y=vals, mode="lines+markers",
                    name=f"{label}{pool_suffix}",
                    line=dict(color=color, dash=style["dash"] if len(available) == 1
                              else ("solid" if pool == "mean_pool" else "dot"),
                              width=style["width"]),
                    marker=dict(size=5),
                    hovertext=hover, hoverinfo="text",
                    legendgroup=grp_vs,
                ))

        for y_val, label, color in threshold_lines:
            fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_position="bottom right")

        fig.update_layout(
            title=f"{title_prefix} — phys_valid",
            xaxis_title="Layer (0 = embedding)",
            yaxis_title=ylabel,
            yaxis=dict(range=yrange),
            height=430,
            legend=dict(orientation="h", y=-0.22),
        )
        return to_html(fig, first=is_first)

    auroc_html = _build_metric_fig(
        metric="auroc",
        yrange=[0.35, 1.05],
        ylabel="LOGO-CV AUROC (phys_valid)",
        threshold_lines=[
            (0.5,  "chance (0.50)",             "gray"),
            (0.65, "confound threshold (0.65)",  "#e74c3c"),
            (0.75, "strong signal (0.75)",        "#27ae60"),
        ],
        title_prefix="AUROC vs layer by comment split",
        is_first=first_plotly,
    )

    acc_html = _build_metric_fig(
        metric="accuracy",
        yrange=[0.35, 1.05],
        ylabel="LOGO-CV Accuracy (phys_valid)",
        threshold_lines=[
            (0.5, "chance (0.50)", "gray"),
        ],
        title_prefix="Accuracy vs layer by comment split",
    )

    # Pre-registered interpretation table
    interp_rows = [
        ("NoComm AUROC < 0.65", "Comment confound confirmed — model encodes validity via comments, not physics"),
        ("NoComm AUROC 0.65–0.75", "Ambiguous — weak physics signal or comment bleed-through"),
        ("NoComm AUROC ≥ 0.75", "Model encodes physical validity independent of surface form (stronger finding)"),
        ("Comm AUROC ≥ 0.75", "Comments carry validity signal (expected)"),
        ("Full AUROC ≥ 0.80", "Replication of pilot result (internal check)"),
    ]
    interp_html = "<table><tr><th>Criterion</th><th>Interpretation</th></tr>"
    for crit, interp in interp_rows:
        interp_html += f"<tr><td><code>{crit}</code></td><td>{interp}</td></tr>"
    interp_html += "</table>"

    return f"""
<div class="page" id="page-validity-split">
<h2>Validity Split</h2>
<p>Tests whether phys_valid probe signal survives when docstring comments are absent.
<b>NoComm</b> = {{NoComm_Valid, NoComm_InValid, NoComm_CorrVar, NoComm_CorrVar_InValid}} (n=64).
<b>Comm</b> = remaining 4 mod_types (n=64). C=1.0, LOGO-CV grouped by gt_sample.</p>
<p><small>⚠ NoComm folds have only 4 test rows each → AUROC coarse (5 possible values per fold).
  CIs are wide by construction. Use the mean as the primary signal, not the CI boundary.</small></p>

<h3>AUROC vs Layer</h3>
<p><small>Pre-registered threshold: NoComm AUROC &lt; 0.65 → comment confound.
Dashed red = 0.65 boundary. Green = 0.75 strong-signal line.</small></p>
{auroc_html}

<h3>Accuracy vs Layer</h3>
<p><small>Complementary to AUROC. More stable given small test folds.</small></p>
{acc_html}

<h3>Pre-registered Interpretation</h3>
{interp_html}
</div>
"""


# ── Page 4 — Transfer Probe ──────────────────────────────────────────────────

def build_transfer(dfs: dict) -> str:
    available = [p for p in POOLS if dfs.get(p) is not None]
    if not available:
        return """
<div class="page" id="page-transfer">
<h2>Transfer Probe</h2>
<p><em>Transfer probe results not yet available.
Re-run <code>probe/slurm/probe_transfer_mean_pool.slurm</code> and
<code>probe/slurm/probe_transfer_last_tok.slurm</code> once those jobs complete,
then regenerate this report.</em></p>
</div>
"""

    first_label = LABEL_ORDER[0]

    # ── Figure 1: Label × mod_type heatmap ──
    hm_fig = go.Figure()
    hm_traces: list = []
    for pool in available:
        df = dfs[pool]
        z = np.full((len(LABEL_ORDER), len(MOD_TYPES)), float("nan"))
        hover_hm = [[""]*len(MOD_TYPES) for _ in range(len(LABEL_ORDER))]
        for i, label in enumerate(LABEL_ORDER):
            sub = df[df["label"] == label]
            if sub.empty:
                continue
            best_row = sub.loc[sub["overall_acc"].idxmax()]
            for j, mt in enumerate(MOD_TYPES):
                val = float(best_row.get(f"mt_{mt}", float("nan")))
                z[i, j] = val
                hover_hm[i][j] = f"{label} | {mt}<br>acc={val:.3f}" if not np.isnan(val) else "N/A"
        hm_fig.add_trace(go.Heatmap(
            z=z, x=MOD_TYPES, y=LABEL_ORDER,
            colorscale="RdYlGn", zmin=0, zmax=1,
            text=hover_hm,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Accuracy"),
            visible=(pool == available[0]),
        ))
        hm_traces.append(pool)

    hm_buttons = (
        [dict(label=p, method="update",
              args=[{"visible": [t == p for t in hm_traces]},
                    {"title": f"Transfer probe: label × mod_type [{p}]"}])
         for p in available]
        if len(available) > 1 else []
    )
    cv_j = MOD_TYPES.index("Comm_Valid")
    hm_fig.update_layout(
        title=f"Transfer probe: label × mod_type [{available[0]}]",
        height=max(350, 32 * len(LABEL_ORDER) + 160),
        yaxis=dict(autorange="reversed"),
        shapes=[dict(
            type="rect",
            x0=cv_j - 0.5, x1=cv_j + 0.5,
            y0=-0.5, y1=len(LABEL_ORDER) - 0.5,
            line=dict(color="blue", width=2.5),
            fillcolor="rgba(0,0,0,0)",
        )],
        updatemenus=([dict(buttons=hm_buttons, direction="down", showactive=True,
                           x=0.0, xanchor="left", y=1.12, yanchor="top")]
                     if hm_buttons else []),
    )
    hm_html = to_html(hm_fig)

    # ── Figure 2: Per-label transfer curves ──
    curve_fig = go.Figure()
    curve_traces: list = []

    for label in LABEL_ORDER:
        visible = (label == first_label)
        for pool in available:
            df = dfs[pool]
            sub = df[df["label"] == label].sort_values("layer")
            if sub.empty:
                continue
            layers = sub["layer"].tolist()
            overall = sub["overall_acc"].tolist()
            ci_lo = sub["ci_low"].tolist()
            ci_hi = sub["ci_high"].tolist()
            color = POOL_COLORS[pool]
            rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)"

            curve_fig.add_trace(go.Scatter(
                x=layers, y=ci_lo, mode="lines", line=dict(width=0),
                showlegend=False, visible=visible, hoverinfo="skip",
            ))
            curve_traces.append(label)
            curve_fig.add_trace(go.Scatter(
                x=layers, y=ci_hi, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=rgba,
                showlegend=False, visible=visible, hoverinfo="skip",
            ))
            curve_traces.append(label)

            hover_ov = [f"Layer {l} | overall [{pool}] | acc={a:.3f} [{lo:.3f}–{hi:.3f}]"
                        for l, a, lo, hi in zip(layers, overall, ci_lo, ci_hi)]
            curve_fig.add_trace(go.Scatter(
                x=layers, y=overall, mode="lines+markers",
                name=f"overall [{pool}]",
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
                hovertext=hover_ov, hoverinfo="text",
                visible=visible, showlegend=(label == first_label),
            ))
            curve_traces.append(label)

            for mt in MOD_TYPES:
                col = f"mt_{mt}"
                if col not in sub.columns:
                    continue
                vals = sub[col].tolist()
                ls = "solid" if mt == "Comm_Valid" else "dash"
                hover_mt = [f"Layer {l} | {mt} [{pool}] | acc={v:.3f}"
                            for l, v in zip(layers, vals)]
                curve_fig.add_trace(go.Scatter(
                    x=layers, y=vals, mode="lines",
                    name=f"{mt} [{pool}]",
                    line=dict(dash=ls, color=MT_COLORS.get(mt, "#888"), width=1.2),
                    hovertext=hover_mt, hoverinfo="text",
                    visible=visible, showlegend=(label == first_label),
                ))
                curve_traces.append(label)

        chance = CHANCE.get(label, 0.5)
        curve_fig.add_trace(go.Scatter(
            x=list(range(29)), y=[chance] * 29,
            mode="lines", line=dict(dash="dot", color="gray", width=1),
            name=f"chance={chance:.2f}", hoverinfo="skip",
            visible=visible, showlegend=(label == first_label),
        ))
        curve_traces.append(label)

    curve_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [t == lbl for t in curve_traces]},
                   {"title": f"Transfer probe curves: {lbl}"}])
        for lbl in LABEL_ORDER
    ]
    curve_fig.update_layout(
        title=f"Transfer probe curves: {first_label}",
        xaxis_title="Layer (0 = embedding)",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05]),
        height=480,
        updatemenus=[dict(buttons=curve_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
        annotations=[dict(
            text=("CI shading: bootstrap 95% CI over per-fold overall accuracies (n=10,000). "
                  "Transfer probe trains on Comm_Valid only (15 examples/fold); "
                  "wide CIs are expected with N=16."),
            xref="paper", yref="paper", x=0, y=-0.14,
            showarrow=False, font=dict(size=10, color="#666"), align="left",
        )],
    )
    curve_html = to_html(curve_fig)

    return f"""
<div class="page" id="page-transfer">
<h2>Transfer Probe</h2>
<p>Trained on <code>Comm_Valid</code> only (15 examples/fold), tested on all 8 mod_types
of the held-out gt_sample. Blue border = training condition. Available poolings: {", ".join(available)}.</p>
<h3>Label × Mod-type Heatmap (best layer per label)</h3>
{hm_html}
<h3>Per-label Transfer Curves</h3>
{curve_html}
</div>
"""


# ── CSS + JS ─────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     display:flex;min-height:100vh;background:#f5f5f5;}
#sidebar{
    position:fixed;top:0;left:0;width:180px;height:100vh;
    background:#2c3e50;padding:24px 0;overflow-y:auto;z-index:999;
    display:flex;flex-direction:column;gap:4px;
}
#sidebar h1{
    color:#ecf0f1;font-size:13px;font-weight:700;letter-spacing:.5px;
    padding:0 18px 16px;border-bottom:1px solid #3d5166;margin-bottom:8px;
    text-transform:uppercase;
}
.nav-item{
    color:#bdc3c7;padding:10px 18px;cursor:pointer;font-size:13px;
    font-weight:500;border-left:3px solid transparent;transition:all .15s;
    user-select:none;
}
.nav-item:hover{color:#ecf0f1;background:rgba(255,255,255,.06);}
.nav-item.active{color:#ecf0f1;background:rgba(255,255,255,.1);
                 border-left-color:#3498db;}
#content{
    margin-left:180px;flex:1;overflow-y:auto;
}
.page{
    display:none;padding:28px 40px 40px;background:white;
    min-height:100vh;
}
.page.active{display:block;}
h2{color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;margin-bottom:16px;
   margin-top:0;}
h3{color:#555;margin-top:24px;margin-bottom:8px;}
table{border-collapse:collapse;margin:10px 0;font-size:13px;}
th,td{border:1px solid #ddd;padding:6px 14px;}
th{background:#f0f0f0;font-weight:600;}
code{background:#f4f4f4;padding:1px 5px;border-radius:3px;font-size:12px;}
p,li{color:#444;line-height:1.6;margin-bottom:8px;}
small{color:#777;}
"""

JS = """
function showPage(name) {
  document.querySelectorAll('.page').forEach(function(p) {
    p.classList.toggle('active', p.id === 'page-' + name);
  });
  document.querySelectorAll('.nav-item').forEach(function(i) {
    i.classList.toggle('active', i.dataset.page === name);
  });
}
document.addEventListener('DOMContentLoaded', function() { showPage('overview'); });
"""

PAGES = [
    ("overview",        "Overview"),
    ("pilot",           "Pilot Study"),
    ("rsa",             "RSA Analysis"),
    ("hyperparam",      "Hyperparam Sweep"),
    ("validity-split",  "Validity Split"),
    ("transfer",        "Transfer Probe"),
    ("multimodel",      "Multi-Model Probe"),
    ("rsa-dimcheck",    "Multimodel RSA"),
    ("multimodel-v2",   "Multi-Model Probe v2"),
]

# Model display config — slug (results subdir) → display name, total layers, color
MULTIMODEL_CONFIGS = {
    "coder7b":   {"label": "Coder-7B",   "n_layers": 28, "color": "#1f77b4"},
    "coder32b":  {"label": "Coder-32B",  "n_layers": 64, "color": "#ff7f0e"},
    "qwq32b":    {"label": "QwQ-32B",    "n_layers": 64, "color": "#2ca02c"},
}


def _mm_layout(title: str, xaxis_title: str, yaxis_title: str,
               yrange: list, height: int = 420) -> dict:
    return dict(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(range=yrange),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                    font=dict(size=11)),
        height=height,
        margin=dict(l=60, r=20, t=50, b=110),
    )


def build_rsa_dimcheck(dimcheck_dfs: dict) -> str:
    """Multimodel RSA dimensionality control page.

    dimcheck_dfs: {key: DataFrame} where key is e.g. "raw_coder7b", "rp20_coder7b", etc.
    Each DataFrame has columns: mod_type, mean_d_task, sem_d_task,
      r_sym, r_sym_ci_lo, r_sym_ci_hi, r_asym, r_asym_ci_lo, r_asym_ci_hi
    """
    if not dimcheck_dfs:
        return """
<div class="page" id="page-rsa-dimcheck">
<h2>Multimodel RSA</h2>
<p><em>No dimensionality control results yet.</em></p>
</div>
"""

    slugs_ordered = ["coder7b", "coder32b", "qwq32b"]
    conditions = [
        ("raw",  "Raw"),
        ("rp20", "RP k=20"),
        ("rp50", "RP k=50"),
        ("rp100","RP k=100"),
    ]
    MT_DISPLAY = [m for m in MOD_TYPES if m != "Comm_Valid"]

    def _drift_fig(condition_key: str, condition_label: str, metric: str,
                   ci_lo_col: str, ci_hi_col: str, ylabel: str, yrange: list,
                   hline: float = None) -> go.Figure:
        fig = go.Figure()
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            key = f"{condition_key}_{slug}"
            df = dimcheck_dfs.get(key)
            if df is None:
                continue
            sub = df[df["mod_type"].isin(MT_DISPLAY)].set_index("mod_type").reindex(MT_DISPLAY)
            vals   = sub[metric].tolist()
            err_lo = (sub[metric] - sub[ci_lo_col]).tolist()
            err_hi = (sub[ci_hi_col] - sub[metric]).tolist()
            hover  = [f"{mt}<br>{metric}={v:.3f} [{lo:.3f}–{hi:.3f}]"
                      for mt, v, lo, hi in zip(MT_DISPLAY, vals,
                                               sub[ci_lo_col].tolist(),
                                               sub[ci_hi_col].tolist())]
            fig.add_trace(go.Bar(
                x=MT_DISPLAY, y=vals, name=cfg["label"],
                marker_color=cfg["color"],
                error_y=dict(type="data", symmetric=False,
                             array=err_hi, arrayminus=err_lo),
                hovertext=hover, hoverinfo="text",
            ))
        if hline is not None:
            fig.add_hline(y=hline, line_dash="dot", line_color="gray",
                          annotation_text=f"r={hline}", annotation_position="top right")
        fig.update_layout(
            **_mm_layout(f"{condition_label} — {ylabel}", "mod_type", ylabel, yrange),
            barmode="group",
        )
        fig.update_xaxes(tickangle=25)
        return fig

    panels_html = []

    # Panel A: raw d_task
    fig_a = _drift_fig("raw", "Raw", "mean_d_task", "mean_d_task", "mean_d_task",
                       "Mean cosine distance (d_task)", [0, None])
    # compute ymax from data
    max_dt = max(
        dimcheck_dfs[f"raw_{s}"]["mean_d_task"].max()
        for s in slugs_ordered if f"raw_{s}" in dimcheck_dfs
    )
    fig_a.update_layout(yaxis=dict(range=[0, max_dt * 1.25]))
    panels_html.append(f"""
<h3>Panel A — Raw Drift (mean cosine distance from Comm_Valid)</h3>
<p><small>Direct cosine distance d_task(g,m) = 1 – cos(h(g,GT), h(g,m)), averaged over 16 gt_samples.
Error bars = SEM. Not dimensionality-normalized — larger D may compress distances.</small></p>
{to_html(fig_a)}""")

    # Panel B: r_sym (main) for raw + each projection
    for ckey, clabel in conditions:
        fig_b = _drift_fig(ckey, clabel, "r_sym", "r_sym_ci_lo", "r_sym_ci_hi",
                           "r_sym (normalized drift)", [0, 1.4], hline=1.0)
        is_main = (ckey == "raw")
        tag = " <em>(main result)</em>" if is_main else ""
        panels_html.append(f"""
<h3>Panel B{' ' if is_main else ' (' + clabel + ')'} — Normalized Drift r_sym{tag}</h3>
<p><small>r_sym(g,m) = d_task(g,m) / d_control_sym(g), where d_control_sym(g) is the mean distance
from h(g,GT) to GT representations of <em>other-class</em> problems (12 per g). Ratio is
dimensionality-agnostic. r&lt;1 = stable relative to cross-class baseline. r=1 = as disruptive
as switching PDE. Error bars = bootstrap 95% CI (n_boot=10,000, resample over 16 gt_samples).
{('Random projection: R ~ N(0,1/k), L2-normalize, 10 repeats averaged.' if ckey != 'raw' else '')}</small></p>
{to_html(fig_b)}""")

    # Panel C: r_asym (secondary) — raw only
    fig_c = _drift_fig("raw", "Raw", "r_asym", "r_asym_ci_lo", "r_asym_ci_hi",
                       "r_asym (normalized drift, secondary)", [0, 1.4], hline=1.0)
    panels_html.append(f"""
<h3>Panel C — Asymmetric Normalized Drift r_asym (secondary)</h3>
<p><small>r_asym(g,m) = d_task(g,m) / d_control_asym(g,m), where d_control_asym uses
h(g,m) (the modified rep) as anchor to cross-class GT — varies with mod_type.
Compare with Panel B (raw) to assess anchor sensitivity.</small></p>
{to_html(fig_c)}""")

    return f"""
<div class="page" id="page-rsa-dimcheck">
<h2>Multimodel RSA — Dimensionality Control</h2>
<p><strong>Question:</strong> Is the CorrVar drift reduction in larger models (7B→32B) a genuine
robustness effect, or a geometric artifact of higher hidden dimensionality (D=3584 vs D=5120)?</p>
<p><strong>Decision rule:</strong> If bootstrap 95% CIs of r_sym_CorrVar are non-overlapping across
models in the raw condition and at least 2 of 3 random projection dimensions → robustness is real.
If CIs overlap in all conditions → geometric artifact cannot be ruled out.</p>
{"".join(panels_html)}
</div>
"""


def build_multimodel(mm_dfs: dict, rsa_dfs: dict = None, first_plotly: bool = False) -> str:
    """Multi-model comparison: 4 figures addressing the two hypotheses.

    mm_dfs: {(slug, pool): DataFrame}  (probe_hyperparam format, C=1.0, PCA-20)
    """
    if not mm_dfs:
        return """
<div class="page" id="page-multimodel">
<h2>Multi-Model Probe</h2>
<p><em>No multi-model results yet.</em></p>
</div>
"""

    pool_dash = {"mean_pool": "solid", "last_tok": "dash"}
    POOL_LABEL = {"mean_pool": "mean-pool", "last_tok": "last-tok"}
    LABELS_ALL = list(LABEL_ORDER)
    slugs_ordered = ["coder7b", "coder32b", "qwq32b"]

    def _get_sub(slug, pool, label):
        df = mm_dfs.get((slug, pool))
        if df is None:
            return None
        sub = df[df["label"] == label].copy()
        if "C" in sub.columns:
            sub = sub[sub["C"] == 1.0].copy()
        if sub.empty or "layer" not in sub.columns:
            return None
        sub = sub[sub["layer"].apply(lambda x: str(x).lstrip("-").isdigit())].copy()
        if sub.empty:
            return None
        sub["layer"] = sub["layer"].astype(int)
        return sub.sort_values("layer")

    # ── Figure 1: Accuracy vs relative layer depth (all labels, dropdown) ──
    acc_fig = go.Figure()
    acc_traces = []
    first_label_acc = "pde_class"
    for label in LABELS_ALL:
        visible = (label == first_label_acc)
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            for pool in POOLS:
                sub = _get_sub(slug, pool, label)
                if sub is None or sub.empty:
                    continue
                n_layers = cfg["n_layers"]
                rel = (sub["layer"] / n_layers).tolist()
                accs = sub["accuracy"].tolist()
                ci_lo = sub["ci_low"].tolist() if "ci_low" in sub.columns else accs
                ci_hi = sub["ci_high"].tolist() if "ci_high" in sub.columns else accs
                color = cfg["color"]
                rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)"
                grp = f"{label}_{slug}_{pool}"
                name = f"{cfg['label']} [{POOL_LABEL[pool]}]"
                hover = [f"{name}<br>rel={r:.2f} (layer {l})<br>acc={a:.3f} [{lo:.3f}–{hi:.3f}]"
                         for r, l, a, lo, hi in zip(rel, sub["layer"].tolist(), accs, ci_lo, ci_hi)]
                acc_fig.add_trace(go.Scatter(
                    x=rel, y=ci_lo, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip", visible=visible,
                    legendgroup=grp, legendgrouptitle_text="",
                ))
                acc_traces.append(label)
                acc_fig.add_trace(go.Scatter(
                    x=rel, y=ci_hi, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=rgba,
                    showlegend=False, hoverinfo="skip", visible=visible,
                    legendgroup=grp,
                ))
                acc_traces.append(label)
                acc_fig.add_trace(go.Scatter(
                    x=rel, y=accs, mode="lines+markers",
                    name=name,
                    line=dict(color=color, dash=pool_dash[pool], width=2),
                    marker=dict(size=4),
                    hovertext=hover, hoverinfo="text",
                    visible=visible, showlegend=True,
                    legendgroup=grp,
                ))
                acc_traces.append(label)

    acc_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [t == lbl for t in acc_traces]},
                   {"title": f"Accuracy vs Relative Layer Depth — {lbl}"}])
        for lbl in LABELS_ALL
    ]
    acc_fig.update_layout(
        **_mm_layout("Accuracy vs Relative Layer Depth — pde_class",
                     "Relative layer depth (layer / n_layers)",
                     "LOGO-CV Accuracy", [0, 1.05]),
        updatemenus=[dict(buttons=acc_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
    )
    acc_html = to_html(acc_fig, first=first_plotly)

    # ── Figure 2: phys_valid AUROC vs relative layer depth ──
    auroc_fig = go.Figure()
    for slug in slugs_ordered:
        cfg = MULTIMODEL_CONFIGS.get(slug)
        if cfg is None:
            continue
        for pool in POOLS:
            sub = _get_sub(slug, pool, "phys_valid")
            if sub is None or sub.empty:
                continue
            sub = sub.dropna(subset=["auroc"])
            if sub.empty:
                continue
            n_layers = cfg["n_layers"]
            rel = (sub["layer"] / n_layers).tolist()
            aurocs = sub["auroc"].tolist()
            ci_lo = sub["auroc_ci_low"].tolist() if "auroc_ci_low" in sub.columns else aurocs
            ci_hi = sub["auroc_ci_high"].tolist() if "auroc_ci_high" in sub.columns else aurocs
            color = cfg["color"]
            rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)"
            grp = f"auroc_{slug}_{pool}"
            name = f"{cfg['label']} [{POOL_LABEL[pool]}]"
            hover = [f"{name}<br>rel={r:.2f} (layer {l})<br>AUROC={a:.3f} [{lo:.3f}–{hi:.3f}]"
                     for r, l, a, lo, hi in zip(rel, sub["layer"].tolist(), aurocs, ci_lo, ci_hi)]
            auroc_fig.add_trace(go.Scatter(
                x=rel, y=ci_lo, mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
                legendgroup=grp, legendgrouptitle_text="",
            ))
            auroc_fig.add_trace(go.Scatter(
                x=rel, y=ci_hi, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=rgba,
                showlegend=False, hoverinfo="skip",
                legendgroup=grp,
            ))
            auroc_fig.add_trace(go.Scatter(
                x=rel, y=aurocs, mode="lines+markers",
                name=name,
                line=dict(color=color, dash=pool_dash[pool], width=2),
                marker=dict(size=4),
                hovertext=hover, hoverinfo="text",
                showlegend=True, legendgroup=grp,
            ))

    auroc_fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                        annotation_text="chance", annotation_position="bottom right")
    auroc_fig.update_layout(
        **_mm_layout("phys_valid AUROC vs Relative Layer Depth",
                     "Relative layer depth (layer / n_layers)",
                     "AUROC", [0.4, 1.05]),
    )
    auroc_html = to_html(auroc_fig)

    # ── Figure 3: Best-layer summary bar chart ──
    # Two side-by-side subplots: pde_class accuracy | phys_valid AUROC
    from plotly.subplots import make_subplots
    bar_fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["pde_class — best-layer Accuracy",
                                            "phys_valid — best-layer AUROC"])
    bar_seen = set()
    for col, (label, metric, ci_lo_col, ci_hi_col) in enumerate([
        ("pde_class",  "accuracy", "ci_low",        "ci_high"),
        ("phys_valid", "auroc",    "auroc_ci_low",   "auroc_ci_high"),
    ], start=1):
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            # use mean_pool only for summary bar
            sub = _get_sub(slug, "mean_pool", label)
            if sub is None or sub.empty:
                continue
            if metric == "auroc":
                sub = sub.dropna(subset=["auroc"])
            if sub.empty:
                continue
            best_idx = sub[metric].idxmax()
            best = sub.loc[best_idx]
            err_lo = float(best[metric]) - float(best.get(ci_lo_col, best[metric]))
            err_hi = float(best.get(ci_hi_col, best[metric])) - float(best[metric])
            show_legend = slug not in bar_seen
            bar_seen.add(slug)
            bar_fig.add_trace(go.Bar(
                x=[cfg["label"]], y=[float(best[metric])],
                error_y=dict(type="data", symmetric=False,
                             array=[err_hi], arrayminus=[err_lo]),
                name=cfg["label"],
                marker_color=cfg["color"],
                showlegend=show_legend,
                legendgroup=slug,
                hovertemplate=(
                    f"{cfg['label']}<br>{metric}={best[metric]:.3f}<br>"
                    f"layer {int(best['layer'])} (rel={int(best['layer'])/cfg['n_layers']:.2f})"
                    "<extra></extra>"
                ),
            ), row=1, col=col)

    bar_fig.update_layout(
        height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=50, b=110),
        barmode="group",
    )
    bar_html = to_html(bar_fig)

    # ── Figure 4: phys_valid per-mod_type at best layer (mean_pool) ──
    mt_fig = go.Figure()
    mt_col_map = {mt: f"mt_{mt}" for mt in MOD_TYPES}
    for slug in slugs_ordered:
        cfg = MULTIMODEL_CONFIGS.get(slug)
        if cfg is None:
            continue
        sub = _get_sub(slug, "mean_pool", "phys_valid")
        if sub is None or sub.empty:
            continue
        best_idx = sub["auroc"].dropna().idxmax() if sub["auroc"].notna().any() else sub["accuracy"].idxmax()
        best = sub.loc[best_idx]
        mt_vals = [float(best.get(mt_col_map[mt], float("nan"))) for mt in MOD_TYPES]
        hover = [f"{cfg['label']}<br>{mt}: acc={v:.3f}<br>layer {int(best['layer'])}"
                 for mt, v in zip(MOD_TYPES, mt_vals)]
        mt_fig.add_trace(go.Bar(
            x=MOD_TYPES, y=mt_vals,
            name=cfg["label"],
            marker_color=cfg["color"],
            hovertext=hover, hoverinfo="text",
            legendgroup=slug,
        ))

    mt_fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                     annotation_text="chance", annotation_position="bottom right")
    mt_fig.update_layout(
        **_mm_layout("phys_valid Accuracy by mod_type — at best AUROC layer (mean_pool)",
                     "mod_type", "Accuracy", [0, 1.05], height=440),
        barmode="group",
    )
    mt_fig.update_xaxes(tickangle=20)
    mt_html = to_html(mt_fig)

    models_present = sorted({slug for slug, _ in mm_dfs})
    models_str = ", ".join(
        MULTIMODEL_CONFIGS[s]["label"] for s in slugs_ordered if s in models_present
    )

    # ── RSA figures (optional — only if rsa_dfs provided) ──
    rsa_html = ""
    if rsa_dfs:
        # Fig 5: Block score vs relative layer depth
        block_fig = go.Figure()
        label_dash = {"pde_class": "solid", "phys_valid": "dash"}
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            df = rsa_dfs.get(f"block_{slug}")
            if df is None:
                continue
            n_layers = cfg["n_layers"]
            for pool in ["mean_pool"]:  # mean_pool only for clarity
                sub = df[df["pool"] == pool].sort_values("layer")
                rel = (sub["layer"] / n_layers).tolist()
                for metric, dash in label_dash.items():
                    col = "pde_block_score" if metric == "pde_class" else "valid_block_score"
                    vals = sub[col].tolist()
                    block_fig.add_trace(go.Scatter(
                        x=rel, y=vals, mode="lines+markers",
                        name=f"{cfg['label']} {metric}",
                        line=dict(color=cfg["color"], dash=dash, width=2),
                        marker=dict(size=3),
                        hovertemplate=f"{cfg['label']} {metric}<br>rel=%{{x:.2f}}<br>score=%{{y:.3f}}<extra></extra>",
                    ))

        block_fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                            annotation_text="ratio=1 (no clustering)",
                            annotation_position="bottom right")
        block_fig.update_layout(
            **_mm_layout("RSA Block Score vs Relative Layer Depth (mean_pool)",
                         "Relative layer depth", "Within / between ratio (lower = better)", [0.3, 1.3]),
        )
        block_rsa_html = (
            "<p><small>Solid = pde_class clustering, dashed = phys_valid clustering. "
            "Lower = within-group representations are tighter relative to between-group. "
            "Mean_pool only for clarity.</small></p>"
            + to_html(block_fig)
        )

        # Fig 6: Mod-type drift grouped bars
        drift_fig = go.Figure()
        mt_order = [mt for mt in MOD_TYPES if mt != "Comm_Valid"]
        drift_seen = set()
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            df = rsa_dfs.get(f"drift_{slug}")
            if df is None:
                continue
            sub = df[df["pool"] == "mean_pool"]
            vals, errs, hover = [], [], []
            for mt in mt_order:
                row = sub[sub["mod_type"] == mt]
                if row.empty:
                    vals.append(float("nan")); errs.append(0); hover.append(mt)
                else:
                    r = row.iloc[0]
                    vals.append(float(r["mean_dist"]))
                    errs.append(float(r["sem"]))
                    hover.append(f"{cfg['label']}<br>{mt}<br>dist={r['mean_dist']:.4f} ± {r['sem']:.4f}<br>layer {int(r['best_layer'])}")
            drift_fig.add_trace(go.Bar(
                x=mt_order, y=vals,
                error_y=dict(type="data", array=errs),
                name=cfg["label"],
                marker_color=cfg["color"],
                hovertext=hover, hoverinfo="text",
            ))
        all_drift_vals = [
            float(rsa_dfs[f"drift_{s}"][rsa_dfs[f"drift_{s}"]["pool"] == "mean_pool"]["mean_dist"].max())
            for s in slugs_ordered if f"drift_{s}" in rsa_dfs
        ]
        drift_ymax = max(all_drift_vals) * 1.25 if all_drift_vals else 0.15
        drift_fig.update_layout(
            **_mm_layout("Mod-type Drift from Comm_Valid at Best pde_class Layer (mean_pool)",
                         "mod_type", "Mean cosine distance", [0, drift_ymax], height=420),
            barmode="group",
        )
        drift_fig.update_xaxes(tickangle=20)
        drift_rsa_html = (
            "<p><small>Mean cosine distance from Comm_Valid (same gt_sample) at each model's best "
            "pde_class layer. Higher = mod_type manipulation moves representation further from the "
            "GT baseline. Error bars = SE across 16 gt_samples.</small></p>"
            + to_html(drift_fig)
        )

        # Fig 7: PCA-2 scatter — one subplot per model, mean_pool only
        PDE_SYMBOLS = {"burgers": "circle", "heat": "square", "navier-stokes": "diamond", "wave": "cross"}
        pca_figs_html = []
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            df = rsa_dfs.get(f"pca2_{slug}")
            if df is None:
                continue
            sub = df[df["pool"] == "mean_pool"]
            if sub.empty:
                continue
            layer = int(sub["layer"].iloc[0])
            var1 = float(sub["var_pc1"].iloc[0])
            var2 = float(sub["var_pc2"].iloc[0])
            pca_fig = go.Figure()
            mt_seen_pca = set()
            for mt in MOD_TYPES:
                mt_sub = sub[sub["mod_type"] == mt]
                if mt_sub.empty:
                    continue
                for pde in PDE_SYMBOLS:
                    pde_sub = mt_sub[mt_sub["pde_class"] == pde]
                    if pde_sub.empty:
                        continue
                    show_leg = mt not in mt_seen_pca
                    mt_seen_pca.add(mt)
                    hover = [f"{row['title']}<br>{pde} / {mt}" for _, row in pde_sub.iterrows()]
                    pca_fig.add_trace(go.Scatter(
                        x=pde_sub["pc1"].tolist(), y=pde_sub["pc2"].tolist(),
                        mode="markers",
                        marker=dict(
                            color=MT_COLORS.get(mt, "#888"),
                            symbol=PDE_SYMBOLS[pde],
                            size=9, line=dict(width=0.5, color="white"),
                        ),
                        name=mt,
                        legendgroup=mt,
                        showlegend=show_leg,
                        hovertext=hover, hoverinfo="text",
                    ))
            pca_fig.update_layout(
                title=f"{cfg['label']} — PCA-2 at layer {layer} (mean_pool, {var1*100:.0f}%+{var2*100:.0f}% var)",
                xaxis_title=f"PC1 ({var1*100:.1f}% var)",
                yaxis_title=f"PC2 ({var2*100:.1f}% var)",
                height=440,
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=10)),
                margin=dict(l=60, r=20, t=50, b=130),
            )
            pca_figs_html.append(f"<h4>{cfg['label']}</h4>" + to_html(pca_fig))

        pca_html_joined = "\n".join(pca_figs_html) if pca_figs_html else "<p><em>PCA-2 data not yet available.</em></p>"

        rsa_html = f"""
<h3>5. RSA Block Score vs Layer</h3>
<p><small>Do models with better MCQ accuracy also show more structured representation geometry?
Block score measures how tightly representations cluster by label relative to other labels.</small></p>
{block_rsa_html}

<h3>6. Mod-type Drift from Comm_Valid</h3>
<p><small>How far does each manipulation shift representations from the fully-commented valid baseline?
Consistent ordering across models suggests dataset-driven structure, not model-specific.</small></p>
{drift_rsa_html}

<h3>7. PCA-2 Scatter at Best pde_class Layer</h3>
<p><small>First two principal components at each model's best pde_class layer (mean_pool).
<strong>Shape</strong> = PDE class &nbsp;|&nbsp; <strong>Color</strong> = mod_type.
Separation by shape = PDE class structure; separation by color = comment/validity sensitivity.</small></p>
{pca_html_joined}
"""

    return f"""
<div class="page" id="page-multimodel">
<h2>Multi-Model Probe</h2>
<p>Linear probe (PCA-20, C=1.0, LOGO-CV grouped by gt_sample, 16 folds) across model scale and
reasoning type. X-axis uses relative layer depth so models with different layer counts align.
<strong>Solid lines</strong> = mean-pool &nbsp;|&nbsp; <strong>Dashed lines</strong> = last-tok.
Models: {models_str}.</p>

<h3>1. Accuracy vs Layer — all labels</h3>
<p><small>Use dropdown to switch label. Scale hypothesis: pde_class accuracy should increase
from Coder-7B → Coder-32B.</small></p>
{acc_html}

<h3>2. phys_valid AUROC vs Layer</h3>
<p><small>AUROC is defined only for phys_valid (only label that varies within a LOGO fold).
Reasoning hypothesis: QwQ-32B AUROC should exceed Coder-32B on NoComm mod_types.</small></p>
{auroc_html}

<h3>3. Best-layer Summary</h3>
<p><small>Best achievable value at any layer (mean_pool). Error bars = 95% CI at that layer.
Direct answer to both hypotheses.</small></p>
{bar_html}

<h3>4. phys_valid by mod_type at Best AUROC Layer</h3>
<p><small>Per-mod_type accuracy at each model's best phys_valid AUROC layer (mean_pool).
NoComm mod_types (NoComm_Valid, NoComm_InValid, NoComm_CorrVar, NoComm_CorrVar_InValid)
reveal whether reasoning pretraining helps without comment scaffolding.</small></p>
{mt_html}
{rsa_html}
</div>
"""


def build_multimodel_v2(mm_v2_dfs: dict) -> str:
    """Multi-Model Probe v2: apples-to-apples PCA-20 comparison across all 3 models.

    mm_v2_dfs: {(slug, pool): DataFrame} — probe_hyperparam_pca20 CSVs for all 3 models.
    Figures:
      1. pde_class accuracy vs relative depth — C=1.0, both pools, CI bands
      2. pde_class accuracy vs relative depth — best-C per (slug, pool), CI bands
      3. phys_valid AUROC vs relative depth — C=1.0, mean_pool only
      4. Best-layer summary bars — acc (C=1.0 vs best-C) and AUROC (C=1.0)
      5. Mod-type breakdown at best pde_class layer — C=1.0, mean_pool
    """
    if not mm_v2_dfs:
        return """
<div class="page" id="page-multimodel-v2">
<h2>Multi-Model Probe v2</h2>
<p><em>No data yet — CSVs not found.</em></p>
</div>
"""

    slugs_ordered = ["coder7b", "coder32b", "qwq32b"]
    pool_dash = {"mean_pool": "solid", "last_tok": "dash"}
    POOL_LABEL = {"mean_pool": "mean-pool", "last_tok": "last-tok"}

    def _get_c1(slug, pool, label):
        """Return rows for C=1.0, numeric layers only, sorted."""
        df = mm_v2_dfs.get((slug, pool))
        if df is None:
            return None
        sub = df[(df["label"] == label) & (df["C"] == 1.0)].copy()
        sub = sub[sub["layer"].apply(lambda x: str(x).lstrip("-").isdigit())].copy()
        if sub.empty:
            return None
        sub["layer"] = sub["layer"].astype(int)
        return sub.sort_values("layer")

    def _get_best_c(slug, pool, label):
        """Return rows for the best C (max accuracy over all layers), numeric layers only."""
        df = mm_v2_dfs.get((slug, pool))
        if df is None:
            return None
        sub = df[df["label"] == label].copy()
        sub = sub[sub["layer"].apply(lambda x: str(x).lstrip("-").isdigit())].copy()
        if sub.empty or "C" not in sub.columns:
            return None
        sub["layer"] = sub["layer"].astype(int)
        best_c = sub.loc[sub["accuracy"].idxmax(), "C"]
        result = sub[sub["C"] == best_c].sort_values("layer")
        return result, best_c

    def _rgba(hex_color, alpha=0.12):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def _add_acc_traces(fig, slug, pool, sub, cfg, name):
        """Add CI band + line for an accuracy trace."""
        n_layers = cfg["n_layers"]
        rel = (sub["layer"] / n_layers).tolist()
        accs = sub["accuracy"].tolist()
        ci_lo = sub["ci_low"].tolist() if "ci_low" in sub.columns else accs
        ci_hi = sub["ci_high"].tolist() if "ci_high" in sub.columns else accs
        color = cfg["color"]
        rgba = _rgba(color)
        grp = f"{slug}_{pool}"
        hover = [f"{name}<br>rel={r:.2f} (layer {l})<br>acc={a:.3f} [{lo:.3f}–{hi:.3f}]"
                 for r, l, a, lo, hi in zip(rel, sub["layer"].tolist(), accs, ci_lo, ci_hi)]
        fig.add_trace(go.Scatter(x=rel, y=ci_lo, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", legendgroup=grp))
        fig.add_trace(go.Scatter(x=rel, y=ci_hi, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba,
                                 showlegend=False, hoverinfo="skip", legendgroup=grp))
        fig.add_trace(go.Scatter(x=rel, y=accs, mode="lines+markers",
                                 name=name,
                                 line=dict(color=color, dash=pool_dash[pool], width=2),
                                 marker=dict(size=4),
                                 hovertext=hover, hoverinfo="text",
                                 showlegend=True, legendgroup=grp))

    # ── Figure 1: pde_class accuracy C=1.0, both pools ──
    fig1 = go.Figure()
    for slug in slugs_ordered:
        cfg = MULTIMODEL_CONFIGS.get(slug)
        if cfg is None:
            continue
        for pool in POOLS:
            sub = _get_c1(slug, pool, "pde_class")
            if sub is None or sub.empty:
                continue
            name = f"{cfg['label']} [{POOL_LABEL[pool]}]"
            _add_acc_traces(fig1, slug, pool, sub, cfg, name)
    fig1.add_hline(y=0.25, line_dash="dot", line_color="gray",
                   annotation_text="chance (0.25)", annotation_position="bottom right")
    fig1.update_layout(
        **_mm_layout("pde_class Accuracy vs Relative Layer Depth — C=1.0 (all models)",
                     "Relative layer depth (layer / n_layers)",
                     "LOGO-CV Accuracy", [0, 1.05]),
    )
    fig1_html = to_html(fig1, first=True)

    # ── Figure 2: pde_class accuracy best-C per (slug, pool) ──
    fig2 = go.Figure()
    best_c_labels = {}  # track which C was used
    for slug in slugs_ordered:
        cfg = MULTIMODEL_CONFIGS.get(slug)
        if cfg is None:
            continue
        for pool in POOLS:
            result = _get_best_c(slug, pool, "pde_class")
            if result is None:
                continue
            sub, best_c = result
            best_c_labels[(slug, pool)] = best_c
            name = f"{cfg['label']} [{POOL_LABEL[pool]}] C={best_c:.4g}"
            _add_acc_traces(fig2, slug, pool, sub, cfg, name)
    fig2.add_hline(y=0.25, line_dash="dot", line_color="gray",
                   annotation_text="chance (0.25)", annotation_position="bottom right")
    fig2.update_layout(
        **_mm_layout("pde_class Accuracy vs Relative Layer Depth — Best C per model/pool",
                     "Relative layer depth (layer / n_layers)",
                     "LOGO-CV Accuracy", [0, 1.05]),
    )
    fig2_html = to_html(fig2)

    # ── Figure 3: phys_valid AUROC C=1.0, mean_pool only ──
    fig3 = go.Figure()
    for slug in slugs_ordered:
        cfg = MULTIMODEL_CONFIGS.get(slug)
        if cfg is None:
            continue
        sub = _get_c1(slug, "mean_pool", "phys_valid")
        if sub is None or sub.empty:
            continue
        sub = sub.dropna(subset=["auroc"])
        if sub.empty:
            continue
        n_layers = cfg["n_layers"]
        rel = (sub["layer"] / n_layers).tolist()
        aurocs = sub["auroc"].tolist()
        ci_lo = sub["auroc_ci_low"].tolist() if "auroc_ci_low" in sub.columns else aurocs
        ci_hi = sub["auroc_ci_high"].tolist() if "auroc_ci_high" in sub.columns else aurocs
        color = cfg["color"]
        rgba = _rgba(color)
        grp = f"auroc_{slug}"
        name = f"{cfg['label']}"
        hover = [f"{name}<br>rel={r:.2f} (layer {l})<br>AUROC={a:.3f} [{lo:.3f}–{hi:.3f}]"
                 for r, l, a, lo, hi in zip(rel, sub["layer"].tolist(), aurocs, ci_lo, ci_hi)]
        fig3.add_trace(go.Scatter(x=rel, y=ci_lo, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip", legendgroup=grp))
        fig3.add_trace(go.Scatter(x=rel, y=ci_hi, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor=rgba,
                                  showlegend=False, hoverinfo="skip", legendgroup=grp))
        fig3.add_trace(go.Scatter(x=rel, y=aurocs, mode="lines+markers",
                                  name=name,
                                  line=dict(color=color, width=2),
                                  marker=dict(size=4),
                                  hovertext=hover, hoverinfo="text",
                                  showlegend=True, legendgroup=grp))
    fig3.add_hline(y=0.5, line_dash="dot", line_color="gray",
                   annotation_text="chance (0.5)", annotation_position="bottom right")
    fig3.update_layout(
        **_mm_layout("phys_valid AUROC vs Relative Layer Depth — C=1.0, mean_pool",
                     "Relative layer depth (layer / n_layers)",
                     "AUROC", [0.4, 1.05]),
    )
    fig3_html = to_html(fig3)

    # ── Figure 4: Best-layer summary bars — acc (C=1.0 vs best-C) and AUROC ──
    from plotly.subplots import make_subplots as _msp
    fig4 = _msp(rows=1, cols=3,
                subplot_titles=["pde_class Acc — C=1.0",
                                "pde_class Acc — best C",
                                "phys_valid AUROC — C=1.0"])
    bar4_seen = set()
    for col_idx, (c_mode, label, metric, ci_lo_col, ci_hi_col) in enumerate([
        ("c1",     "pde_class",  "accuracy", "ci_low",      "ci_high"),
        ("best_c", "pde_class",  "accuracy", "ci_low",      "ci_high"),
        ("c1",     "phys_valid", "auroc",    "auroc_ci_low","auroc_ci_high"),
    ], start=1):
        for slug in slugs_ordered:
            cfg = MULTIMODEL_CONFIGS.get(slug)
            if cfg is None:
                continue
            if c_mode == "c1":
                sub = _get_c1(slug, "mean_pool", label)
            else:
                result = _get_best_c(slug, "mean_pool", label)
                sub = result[0] if result else None
            if sub is None or sub.empty:
                continue
            if metric == "auroc":
                sub = sub.dropna(subset=["auroc"])
            if sub.empty:
                continue
            best_idx = sub[metric].idxmax()
            best = sub.loc[best_idx]
            val = float(best[metric])
            err_lo = val - float(best.get(ci_lo_col, val))
            err_hi = float(best.get(ci_hi_col, val)) - val
            show_legend = (col_idx == 1) and (slug not in bar4_seen)
            bar4_seen.add(slug)
            fig4.add_trace(go.Bar(
                x=[cfg["label"]], y=[val],
                error_y=dict(type="data", symmetric=False, array=[err_hi], arrayminus=[err_lo]),
                name=cfg["label"],
                marker_color=cfg["color"],
                showlegend=show_legend,
                legendgroup=slug,
                hovertemplate=(
                    f"{cfg['label']}<br>{metric}={val:.3f}<br>"
                    f"layer {int(best['layer'])} (rel={int(best['layer'])/cfg['n_layers']:.2f})"
                    "<extra></extra>"
                ),
            ), row=1, col=col_idx)
    fig4.update_layout(
        height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=110),
        barmode="group",
    )
    fig4_html = to_html(fig4)

    # ── Figure 5: Mod-type breakdown at best pde_class layer — C=1.0, mean_pool ──
    fig5 = go.Figure()
    for slug in slugs_ordered:
        cfg = MULTIMODEL_CONFIGS.get(slug)
        if cfg is None:
            continue
        sub = _get_c1(slug, "mean_pool", "pde_class")
        if sub is None or sub.empty:
            continue
        best_idx = sub["accuracy"].idxmax()
        best = sub.loc[best_idx]
        mt_vals = [float(best.get(f"mt_{mt}", float("nan"))) for mt in MOD_TYPES]
        hover = [f"{cfg['label']}<br>{mt}: acc={v:.3f}<br>layer {int(best['layer'])}"
                 for mt, v in zip(MOD_TYPES, mt_vals)]
        fig5.add_trace(go.Bar(
            x=MOD_TYPES, y=mt_vals,
            name=cfg["label"],
            marker_color=cfg["color"],
            hovertext=hover, hoverinfo="text",
        ))
    fig5.add_hline(y=0.25, line_dash="dot", line_color="gray",
                   annotation_text="chance (0.25)", annotation_position="bottom right")
    fig5.update_layout(
        **_mm_layout("pde_class Mod-type Breakdown at Best Layer — C=1.0, mean_pool",
                     "mod_type", "Accuracy", [0, 1.05], height=440),
        barmode="group",
    )
    fig5.update_xaxes(tickangle=20)
    fig5_html = to_html(fig5)

    models_present = sorted({slug for slug, _ in mm_v2_dfs})
    models_str = ", ".join(
        MULTIMODEL_CONFIGS[s]["label"] for s in slugs_ordered if s in models_present
    )

    return f"""
<div class="page" id="page-multimodel-v2">
<h2>Multi-Model Probe v2 — Apples-to-Apples (PCA-20)</h2>
<p>Linear probe with PCA-20 representations and LOGO-CV (16 folds, grouped by gt_sample) across all
three models. Unlike the original multi-model page, all models use the same PCA-20 dimensionality
reduction, making accuracy numbers directly comparable. X-axis = relative layer depth (layer /
n_layers) to align models. <strong>Solid lines</strong> = mean-pool &nbsp;|&nbsp;
<strong>Dashed lines</strong> = last-tok. Models: {models_str}.</p>

<h3>1. pde_class Accuracy vs Layer — C=1.0 (consistent regularisation)</h3>
<p><small>Same C across all models for a fair comparison. CI bands = 95% bootstrap.</small></p>
{fig1_html}

<h3>2. pde_class Accuracy vs Layer — Best C per model/pool</h3>
<p><small>Best-C selected independently per (model, pool) by max accuracy over all layers.
Legend shows which C was selected for each curve.</small></p>
{fig2_html}

<h3>3. phys_valid AUROC vs Layer — C=1.0, mean_pool</h3>
<p><small>AUROC is only defined for phys_valid (binary label). Mean_pool only for clarity.</small></p>
{fig3_html}

<h3>4. Best-layer Summary</h3>
<p><small>Peak value at any layer (mean_pool). Left: pde_class acc at C=1.0; Centre: pde_class acc
at best C; Right: phys_valid AUROC at C=1.0. Error bars = 95% CI at that layer.</small></p>
{fig4_html}

<h3>5. pde_class Mod-type Breakdown at Best Layer — C=1.0, mean_pool</h3>
<p><small>Per mod_type accuracy at each model's best pde_class layer (C=1.0, mean_pool).
All mod_types should be above chance (0.25) for a well-calibrated classifier.</small></p>
{fig5_html}
</div>
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden",
                        default="probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz")
    parser.add_argument("--results_dir", default="probe/results/")
    parser.add_argument("--output", default="probe/results/report.html")
    args = parser.parse_args()

    print("Loading NPZ...", flush=True)
    data = load_data(args.hidden)
    _label_arrays = extract_label_arrays(data)
    majority = {}
    for lbl, arr in _label_arrays.items():
        valid = arr[arr >= 0]
        counts = np.bincount(valid)
        majority[lbl] = float(counts.max() / len(valid))
    print("  Majority baselines:", {k: f"{v:.3f}" for k, v in majority.items()}, flush=True)

    results = Path(args.results_dir)
    pooled_dfs, transfer_dfs = {}, {}
    for pool in POOLS:
        p = results / f"probe_pooled_{pool}.csv"
        if p.exists():
            pooled_dfs[pool] = pd.read_csv(p)
            print(f"  Loaded {p}", flush=True)
        p = results / f"probe_transfer_{pool}.csv"
        if p.exists():
            transfer_dfs[pool] = pd.read_csv(p)
            print(f"  Loaded {p}", flush=True)

    # Hyperparam sweep CSVs — from results_dir or results_dir/canary_hp/
    hp_dfs: dict = {}
    for repr_ in ("raw", "pca20"):
        for pool in POOLS:
            fname = f"probe_hyperparam_{repr_}_{pool}.csv"
            for search_dir in [results, results / "canary_hp"]:
                p = search_dir / fname
                if p.exists():
                    hp_dfs[(repr_, pool)] = pd.read_csv(p)
                    print(f"  Loaded {p}", flush=True)
                    break

    # Validity split CSVs
    vs_dfs: dict = {}
    vs_dir = results / "validity_split"
    for pool in POOLS:
        p = vs_dir / f"validity_split_{pool}.csv"
        if p.exists():
            vs_dfs[pool] = pd.read_csv(p)
            print(f"  Loaded {p}", flush=True)

    print("Building Overview...", flush=True)
    overview_html = build_overview(data)

    print("Building Pilot Study...", flush=True)
    pilot_needs_plotly = bool(pooled_dfs)
    pilot_html = build_pilot(pooled_dfs, first_plotly=pilot_needs_plotly, majority=majority)

    print("Building RSA...", flush=True)
    rsa_html = build_rsa(data, first_plotly=(not pilot_needs_plotly))

    print("Building Hyperparam Sweep...", flush=True)
    hp_needs_plotly = not pilot_needs_plotly and True
    hyperparam_html = build_hyperparam(hp_dfs, first_plotly=False)

    print("Building Validity Split...", flush=True)
    validity_split_html = build_validity_split(vs_dfs, first_plotly=False)

    print("Building Transfer Probe...", flush=True)
    transfer_html = build_transfer(transfer_dfs)

    # Multi-model CSVs — one subdir per model slug, same probe_hyperparam_{repr}_{pool}.csv format
    # Coder-7B baseline comes from results_dir directly (slug "coder7b")
    mm_dfs: dict = {}
    multimodel_slug_dirs = {
        "coder7b":  results,
        "coder32b": results / "coder32b",
        "qwq32b":   results / "qwq32b",
    }
    for slug, slug_dir in multimodel_slug_dirs.items():
        for pool in POOLS:
            p = slug_dir / f"probe_hyperparam_pca20_{pool}.csv"
            if p.exists():
                mm_dfs[(slug, pool)] = pd.read_csv(p)
                print(f"  Loaded multimodel {slug}/{pool}: {p}", flush=True)

    # RSA CSVs for multi-model comparison
    rsa_dfs: dict = {}
    rsa_dir = results / "rsa"
    for slug in ("coder7b", "coder32b", "qwq32b"):
        for kind in ("block", "drift", "pca2"):
            p = rsa_dir / f"rsa_{kind}_{slug}.csv" if kind != "pca2" else rsa_dir / f"pca2_{slug}.csv"
            if p.exists():
                rsa_dfs[f"{kind}_{slug}"] = pd.read_csv(p)
                print(f"  Loaded RSA {kind}/{slug}: {p}", flush=True)

    print("Building Multi-Model Probe...", flush=True)
    multimodel_html = build_multimodel(mm_dfs, rsa_dfs=rsa_dfs or None, first_plotly=False)

    # RSA dimensionality control CSVs
    dimcheck_dfs: dict = {}
    dimcheck_dir = results / "rsa_dimcheck"
    for slug in ("coder7b", "coder32b", "qwq32b"):
        for ckey in ("raw", "rp20", "rp50", "rp100"):
            suffix = "" if ckey == "raw" else f"_{ckey}"
            p = dimcheck_dir / f"drift_{slug}{suffix}.csv"
            if p.exists():
                dimcheck_dfs[f"{ckey}_{slug}"] = pd.read_csv(p)
                print(f"  Loaded dimcheck {ckey}/{slug}: {p}", flush=True)
    print("Building Multimodel RSA...", flush=True)
    rsa_dimcheck_html = build_rsa_dimcheck(dimcheck_dfs or {})

    # Multi-model v2 CSVs — same files as mm_dfs (probe_hyperparam_pca20) but all C values
    # mm_dfs already loaded above; reuse directly for v2 page
    print("Building Multi-Model Probe v2...", flush=True)
    multimodel_v2_html = build_multimodel_v2(mm_dfs)

    nav_items = "".join(
        f'<div class="nav-item" data-page="{pid}" onclick="showPage(\'{pid}\')">{label}</div>'
        for pid, label in PAGES
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PDE Probe — {data['mean_pool'].shape[0]} examples, {data['mean_pool'].shape[1]} layers</title>
  <style>{CSS}</style>
</head>
<body>
  <nav id="sidebar">
    <h1>PDE Probe</h1>
    {nav_items}
  </nav>
  <div id="content">
    {overview_html}
    {pilot_html}
    {rsa_html}
    {hyperparam_html}
    {validity_split_html}
    {transfer_html}
    {multimodel_html}
    {rsa_dimcheck_html}
    {multimodel_v2_html}
  </div>
  <script>{JS}</script>
</body>
</html>"""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nReport saved: {args.output}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
