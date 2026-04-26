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
    MOD_TYPES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
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
            ))
            acc_traces.append(label)
            acc_fig.add_trace(go.Scatter(
                x=layers, y=ci_hi, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=rgba,
                showlegend=False, visible=visible, hoverinfo="skip",
            ))
            acc_traces.append(label)
            hover = [f"Layer {l}: acc={a:.3f} [{lo:.3f}–{hi:.3f}] (16 folds, bootstrap n=10,000)"
                     for l, a, lo, hi in zip(layers, accs, ci_lo, ci_hi)]
            acc_fig.add_trace(go.Scatter(
                x=layers, y=accs, mode="lines+markers",
                name=pool, line=dict(color=color), marker=dict(size=5),
                hovertext=hover, hoverinfo="text",
                visible=visible, showlegend=True,
            ))
            acc_traces.append(label)

            bow_rows = df[(df["label"] == label) & (df["layer"] == "bow")]
            if not bow_rows.empty:
                bv = float(bow_rows["accuracy"].iloc[0])
                acc_fig.add_trace(go.Scatter(
                    x=[layers[0], layers[-1]], y=[bv, bv],
                    mode="lines", line=dict(dash="dash", color="#ff7f0e", width=1.5),
                    name=f"BoW [{pool}]", hoverinfo="skip",
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
        bow_hover = [f"{mt}<br>BoW acc={v:.3f}" if not np.isnan(v) else f"{mt}<br>N/A"
                     for mt, v in zip(MOD_TYPES, bow_vals)]
        mt_fig.add_trace(go.Bar(
            x=MOD_TYPES, y=bow_vals, name="BoW",
            marker_color="#ff7f0e",
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
<p>LOGO-CV grouped by gt_sample (16 folds). All 96 rows used for training and testing.
Bootstrap 95% CI (n=10,000 resamples) over per-fold accuracies.
Available poolings: {", ".join(available)}.</p>
<h3>Accuracy vs Layer</h3>
{acc_html}
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

def build_rsa(data: dict, first_plotly: bool = False) -> str:
    pde_labels = data["pde_classes"]
    mod_type_labels = data["mod_types"]
    gt_samples = data["gt_samples"]
    titles = data["titles"]
    N, L, D = data["mean_pool"].shape

    order = sort_order(pde_labels, gt_samples, mod_type_labels)
    pde_sorted = np.array([pde_labels[i] for i in order])
    titles_sorted = np.array([titles[i] for i in order])
    mt_sorted = np.array([mod_type_labels[i] for i in order])

    print("  Computing RDMs for both poolings...", flush=True)
    block_scores: dict = {p: [] for p in POOLS}
    rdms_selected: dict = {p: {} for p in POOLS}

    for pool in POOLS:
        reps_all = data[pool].astype(np.float32)
        for l in range(L):
            rdm = compute_rdm(reps_all[:, l, :])
            block_scores[pool].append(block_rdm_score(rdm, pde_labels))
            if l in LAYERS_TO_PLOT:
                rdms_selected[pool][l] = rdm[np.ix_(order, order)]
        print(f"    {pool} done (best block score: {min(block_scores[pool]):.4f} "
              f"at layer {int(np.argmin(block_scores[pool]))})", flush=True)

    hover_mat = [
        [f"Row: {titles_sorted[i]} [{mt_sorted[i]}]<br>"
         f"Col: {titles_sorted[j]} [{mt_sorted[j]}]"
         for j in range(N)]
        for i in range(N)
    ]

    class_positions: dict = {}
    for i, p in enumerate(pde_sorted):
        class_positions.setdefault(p.lower(), []).append(i)
    tick_pos = [int(np.mean(v)) for v in class_positions.values()]
    tick_labels = list(class_positions.keys())

    boundary_shapes = []
    prev = pde_sorted[0].lower()
    for i, p in enumerate(pde_sorted):
        if p.lower() != prev:
            for axis in ("x", "y"):
                boundary_shapes.append(dict(
                    type="line",
                    **({f"{axis}0": i - 0.5, f"{axis}1": i - 0.5,
                        f"{'y' if axis=='x' else 'x'}0": -0.5,
                        f"{'y' if axis=='x' else 'x'}1": N - 0.5}),
                    line=dict(color="white", width=1.5),
                ))
            prev = p.lower()

    # ── Figure 1: Layer heatmap ──
    heatmap_fig = go.Figure()
    combo_labels = []
    for pool in POOLS:
        for l in LAYERS_TO_PLOT:
            heatmap_fig.add_trace(go.Heatmap(
                z=rdms_selected[pool][l],
                colorscale="Viridis", zmin=0, zmax=1,
                text=hover_mat,
                hovertemplate="%{text}<br>dist=%{z:.3f}<extra></extra>",
                colorbar=dict(title="Cosine dist"),
                visible=False,
            ))
            combo_labels.append(f"Layer {l} [{pool}]")

    heatmap_fig.data[0].visible = True
    n_combos = len(combo_labels)
    hm_buttons = [
        dict(label=lbl, method="update",
             args=[{"visible": [j == i for j in range(n_combos)]},
                   {"title": f"RSA heatmap — {lbl}"}])
        for i, lbl in enumerate(combo_labels)
    ]
    heatmap_fig.update_layout(
        title=f"RSA heatmap — {combo_labels[0]}",
        xaxis=dict(tickvals=tick_pos, ticktext=tick_labels, tickangle=30),
        yaxis=dict(tickvals=tick_pos, ticktext=tick_labels, autorange="reversed"),
        width=680, height=640,
        shapes=boundary_shapes,
        updatemenus=[dict(buttons=hm_buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.14, yanchor="top")],
    )
    heatmap_html = to_html(heatmap_fig, first=first_plotly)

    # ── Figure 2: Block score vs layer ──
    block_fig = go.Figure()
    for pool in POOLS:
        block_fig.add_trace(go.Scatter(
            x=list(range(L)), y=block_scores[pool],
            mode="lines+markers", marker=dict(size=4),
            name=pool, line=dict(color=POOL_COLORS[pool]),
        ))
    for l in LAYERS_TO_PLOT:
        block_fig.add_vline(x=l, line_dash="dot", line_color="gray", opacity=0.35)
    block_fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                        annotation_text="ratio=1 (no clustering)",
                        annotation_position="bottom right")
    block_fig.update_layout(
        title="RSA block score vs layer (within/between cosine distance ratio)",
        xaxis_title="Layer (0 = embedding)",
        yaxis_title="Within / between ratio  (lower = better PDE clustering)",
        height=380, legend=dict(orientation="h"),
    )
    block_html = to_html(block_fig)

    # ── Figure 3: Mod-type comparison at best-clustering layer ──
    compare_mod_types = ["Comm_Valid", "CorrComm", "NoComm_CorrVar"]
    mt_htmls = []
    for pool in POOLS:
        best_l = int(np.argmin(block_scores[pool]))
        reps_best = data[pool].astype(np.float32)[:, best_l, :]
        fig_mt = make_subplots(
            rows=1, cols=len(compare_mod_types),
            subplot_titles=[f"{mt} (N={int(np.sum(mod_type_labels == mt))})"
                            for mt in compare_mod_types],
        )
        for col_i, mt in enumerate(compare_mod_types, 1):
            mask = mod_type_labels == mt
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            pde_sub = np.array([pde_labels[i] for i in idx])
            sub_order = np.array(sorted(
                range(len(idx)),
                key=lambda i: PDE_CLASSES.index(pde_sub[i].lower())
                if pde_sub[i].lower() in PDE_CLASSES else 99,
            ))
            idx_s = idx[sub_order]
            rdm_sub = compute_rdm(reps_best[idx_s])
            pde_sub_s = np.array([pde_labels[i] for i in idx_s])
            titles_sub = np.array([titles[i] for i in idx_s])
            score = block_rdm_score(rdm_sub, pde_sub_s)
            hover_sub = [[f"{titles_sub[i]}<br>{titles_sub[j]}<br>dist={rdm_sub[i,j]:.3f}"
                          for j in range(len(idx_s))] for i in range(len(idx_s))]
            fig_mt.add_trace(go.Heatmap(
                z=rdm_sub, colorscale="Viridis", zmin=0, zmax=1,
                text=hover_sub,
                hovertemplate="%{text}<extra></extra>",
                showscale=(col_i == len(compare_mod_types)),
            ), row=1, col=col_i)
            fig_mt.layout.annotations[col_i - 1].text += f"<br>block={score:.3f}"
        fig_mt.update_layout(
            title=f"RSA mod-type comparison — best layer {best_l} [{pool}]",
            height=370,
        )
        mt_htmls.append(to_html(fig_mt))

    return f"""
<div class="page" id="page-rsa">
<h2>RSA Analysis</h2>
<p>Pairwise cosine distance matrices from hidden states, rows/columns sorted by
(pde_class, gt_sample, mod_type). Lower within/between ratio = better PDE-class clustering.</p>
<h3>Layer Heatmap</h3>
{heatmap_html}
<h3>Block Score vs Layer</h3>
{block_html}
<h3>Mod-type Comparison (best clustering layer per pooling)</h3>
{"".join(mt_htmls)}
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
<p>Trained on <code>Comm_Valid</code> only (15 examples/fold), tested on all 6 mod_types
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
    ("overview", "Overview"),
    ("pilot",    "Pilot Study"),
    ("rsa",      "RSA Analysis"),
    ("transfer", "Transfer Probe"),
]


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

    print("Building Overview...", flush=True)
    overview_html = build_overview(data)

    print("Building Pilot Study...", flush=True)
    # First plotly embed goes in pilot (or rsa if no pilot data)
    pilot_needs_plotly = bool(pooled_dfs)
    pilot_html = build_pilot(pooled_dfs, first_plotly=pilot_needs_plotly, majority=majority)

    print("Building RSA...", flush=True)
    rsa_html = build_rsa(data, first_plotly=(not pilot_needs_plotly))

    print("Building Transfer Probe...", flush=True)
    transfer_html = build_transfer(transfer_dfs)

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
    {transfer_html}
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
