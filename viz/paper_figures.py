"""
paper_figures.py — Generate paper-ready static PNG figures.

Run from the viz/ directory:
    python paper_figures.py

Produces:
    paper_fig1.png  — MC: Δ confidence and Δ entropy, two panels side by side
    paper_fig2.png  — Accuracy profile + Validity predictions, two panels side by side
    paper_fig3.png  — Reasoning vs non-reasoning model robustness
"""
import re as _re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Data loading ──────────────────────────────────────────────────────────────
# Free-gen results. Defaults to the v3 file these figures were published from;
# point PDE_FREEGEN_CSV at results/pde_llm_eval_jul28.csv to rerun them on jul28.
import os as _os
_FREEGEN_CSV = _os.environ.get("PDE_FREEGEN_CSV", "../results/pde_llm_eval.csv")
print(f"[viz] free-gen input: {_FREEGEN_CSV}")
df_llm = pd.read_csv(_FREEGEN_CSV)
df_mc  = pd.read_csv("../results/pde_mc_logprob.csv")

for col in ["pde_match","method_any_match","behavior_any_match","valid_match",
            "pde_embed_sim","method_recall","behavior_recall"]:
    if col in df_llm.columns:
        df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")

df_mc["correct"] = df_mc["correct"].astype(str).str.lower().map(
    {"true": True, "false": False, "1": True, "0": False})
for col in ["logprob_correct", "margin", "entropy"]:
    df_mc[col] = pd.to_numeric(df_mc[col], errors="coerce")

MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct":          "Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct":          "Llama-3.3-70B",
    "Qwen/QwQ-32B":                               "QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct":             "Qwen2.5-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct":            "Qwen2.5-32B",
    "Qwen/Qwen3-32B":                             "Qwen3-32B",
    "google/gemma-3-27b-it":                      "Gemma-3-27B",
    "mistralai/Mistral-Nemo-Instruct-2407":       "Mistral-12B",
    "microsoft/phi-4":                            "phi-4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":  "DeepSeek-R1-32B",
}
df_llm["model_short"] = df_llm["model"].map(MODEL_SHORT).fillna(df_llm["model"])
df_mc["model_short"]  = df_mc["model"].map(MODEL_SHORT).fillna(df_mc["model"])

ALL_CONDS = [
    "Comm_Valid", "NoComm_Valid",
    "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid",
    "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]
COND_SHORT = {
    "Comm_Valid":             "Clean+Comment",
    "NoComm_Valid":           "Clean, No Comment",
    "CorrComm":               "Corrupt Comment",
    "NoComm_CorrVar":         "Corrupt Variable",
    "Comm_InValid":           "Invalid+Comment",
    "NoComm_InValid":         "Invalid, No Comment",
    "CorrComm_Invalid":       "CorrComment+Invalid",
    "NoComm_CorrVar_InValid": "CorrVar+Invalid",
}
COND_COLOR = {
    "Comm_Valid":             "#2ecc71",
    "NoComm_Valid":           "#27ae60",
    "CorrComm":               "#e67e22",
    "NoComm_CorrVar":         "#e74c3c",
    "Comm_InValid":           "#9b59b6",
    "NoComm_InValid":         "#7d3c98",
    "CorrComm_Invalid":       "#c0392b",
    "NoComm_CorrVar_InValid": "#8e44ad",
}
LLM_METRICS = [
    ("pde_match",       "PDE Type"),
    ("method_recall",   "Method Recall"),
    ("behavior_recall", "Behavior Recall"),
    ("valid_match",     "Validity"),
]
QTYPE_LABEL = {
    "pde_class":    "PDE Class",
    "phys_process": "Phys. Process",
    "num_method":   "Num. Method",
}

TEMPLATE = "plotly_white"
METRIC_COLORS = px.colors.qualitative.Plotly

# Shared right-side legend (used by fig1)
RIGHT_LEGEND = dict(
    x=1.02, y=1, xanchor="left", yanchor="top",
    bgcolor="rgba(255,255,255,0.95)", bordercolor="#aaa", borderwidth=1,
    font=dict(size=11),
)


# ── Utilities ─────────────────────────────────────────────────────────────────
def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    data = data.dropna().values
    if len(data) < 1:
        return np.nan, np.nan, np.nan, np.nan
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data), 0.0
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, ci + (100 - ci) / 2)
    return lower, upper, np.mean(data), np.var(data)


def delta_vs_baseline(df, metric, baseline="Comm_Valid"):
    bl = df[df["mod_type"] == baseline].groupby("question_type")[metric].mean()
    grp = df.groupby(["mod_type", "question_type"])[metric].mean().reset_index()
    grp["delta"] = grp.apply(lambda r: r[metric] - bl.get(r["question_type"], np.nan), axis=1)
    return grp


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: MC Δ confidence and Δ entropy — two panels SIDE BY SIDE
#           Legend: vertical, right side
# ═══════════════════════════════════════════════════════════════════════════════
BASELINE = "Comm_Valid"
non_bl = [c for c in ALL_CONDS if c != BASELINE]
qtypes_no_valid = ["pde_class", "phys_process", "num_method"]

lp_delta  = delta_vs_baseline(df_mc, "logprob_correct")
ent_delta = delta_vs_baseline(df_mc, "entropy")

fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Δ log P(correct answer)<br>confidence drop vs clean baseline",
        "Δ Entropy<br>uncertainty increase vs clean baseline",
    ],
    horizontal_spacing=0.12,
)

for cond in non_bl:
    color  = COND_COLOR[cond]
    label  = COND_SHORT[cond]
    xlabels = [QTYPE_LABEL[q] for q in qtypes_no_valid]

    sub_lp = (lp_delta[(lp_delta["mod_type"] == cond) &
                       (lp_delta["question_type"].isin(qtypes_no_valid))]
              .set_index("question_type").reindex(qtypes_no_valid))
    fig1.add_trace(go.Bar(
        name=label, x=xlabels,
        y=sub_lp["delta"].values.astype(float).round(3),
        marker_color=color, legendgroup=cond, showlegend=True,
        hovertemplate=label + " | %{x}: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    sub_ent = (ent_delta[(ent_delta["mod_type"] == cond) &
                         (ent_delta["question_type"].isin(qtypes_no_valid))]
               .set_index("question_type").reindex(qtypes_no_valid))
    fig1.add_trace(go.Bar(
        name=label, x=xlabels,
        y=sub_ent["delta"].values.astype(float).round(3),
        marker_color=color, legendgroup=cond, showlegend=False,
        hovertemplate=label + " | %{x}: %{y:.3f}<extra></extra>",
    ), row=1, col=2)

fig1.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.25)", row=1, col=1)
fig1.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.25)", row=1, col=2)

fig1.update_layout(
    barmode="group",
    height=460, width=1100,
    template=TEMPLATE,
    legend=RIGHT_LEGEND,
    margin=dict(l=65, r=195, t=100, b=55),
    font=dict(size=12),
)
fig1.update_yaxes(title_text="Δ log P(correct)  ← more negative = less confident",
                  title_font=dict(size=11), row=1, col=1)
fig1.update_yaxes(title_text="Δ Entropy  ← positive = more spread",
                  title_font=dict(size=11), row=1, col=2)

fig1.write_image("paper_fig1.png", scale=2)
print("✓ paper_fig1.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Accuracy profile + Validity predictions — SIDE BY SIDE
#           Two separate horizontal legends: one below each panel
# ═══════════════════════════════════════════════════════════════════════════════

# The hedge rule is canonical in freegen/parse_score.py — it used to be copy-pasted
# here and in two sibling viz scripts, which drifted. NOTE: the shared rule has an
# explicit hedge lexicon the old local copy lacked ("possibly", "cannot determine",
# "depends", ...), so bucket shares are NOT comparable to writeup.pdf Figure 1.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "eval"))
from parse_score import classify_valid_confidence as _classify_conf, VALID_CONF_CLASSES

_CONF_ORDER  = list(VALID_CONF_CLASSES)
_CONF_COLORS = {"Confident Yes": "#27ae60", "Uncertain Yes": "#f1c40f",
                "Hedged": "#e67e22", "Confident No": "#c0392b"}

# run_eval.py now stores valid_conf per row, so the artifact and the figure carry
# the same label. Recompute only for older results that predate that column.
if "valid_conf" not in df_llm.columns:
    df_llm["valid_conf"] = df_llm["parsed_valid"].apply(_classify_conf)

_valid_conds   = [c for c in ALL_CONDS
                  if "InValid" not in c and "Invalid" not in c
                  and c in df_llm["mod_type"].unique()]
_invalid_conds = [c for c in ALL_CONDS
                  if ("InValid" in c or "Invalid" in c)
                  and c in df_llm["mod_type"].unique()]
_ordered_conds  = _valid_conds + _invalid_conds
_ordered_labels = (
    [COND_SHORT.get(c, c) + " ✓" for c in _valid_conds] +
    ["⚠ " + COND_SHORT.get(c, c) for c in _invalid_conds]
)
n_valid = len(_valid_conds)
n_total = len(_ordered_conds)

# Panel domain centers in paper coordinates (width=1200, l=65, r=20, h_spacing=0.11)
# plot_frac = (1200-65-20)/1200 = 0.929; col1=[0,0.445]*0.929+l/w; col2=[0.555,1]*0.929+l/w
_L2_CENT = 0.054 + 0.445 / 2 * 0.929   # ≈ 0.261  center of left panel
_R2_CENT = 0.054 + (0.555 + 1.0) / 2 * 0.929  # ≈ 0.776  center of right panel

fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Score by Condition", "Validity Confidence Breakdown"],
    horizontal_spacing=0.11,
)

# ── Left panel: accuracy lines — assigned to legend (default) ─────────────────
x_labels_left = [COND_SHORT[c] for c in ALL_CONDS]
for i, (metric, mlabel) in enumerate(LLM_METRICS):
    means, lowers, uppers = [], [], []
    for cond in ALL_CONDS:
        lo, hi, mn, _ = bootstrap_ci(df_llm[df_llm["mod_type"] == cond][metric])
        means.append(mn * 100); lowers.append(lo * 100); uppers.append(hi * 100)
    fig2.add_trace(go.Scatter(
        x=x_labels_left, y=means,
        error_y=dict(type="data", symmetric=False,
                     array=[u - m for u, m in zip(uppers, means)],
                     arrayminus=[m - l for m, l in zip(means, lowers)],
                     color="rgba(0,0,0,0.35)", thickness=1.2, width=4),
        mode="lines+markers", name=mlabel,
        line=dict(color=METRIC_COLORS[i], width=2.5), marker=dict(size=7),
        legend="legend", showlegend=True,
    ), row=1, col=1)

# Zone shading left panel
for x0, x1, color, label in [
    ("Clean+Comment",   "Clean, No Comment",   "#2ecc71", "Clean valid"),
    ("Corrupt Comment", "Corrupt Variable",    "#e74c3c", "Corrupted"),
    ("Invalid+Comment", "Invalid, No Comment", "#9b59b6", "Invalid"),
]:
    fig2.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.07, line_width=0,
                   annotation_text=label, annotation_position="top left",
                   annotation_font=dict(size=9, color="#aaa"),
                   annotation_yshift=-18, row=1, col=1)

# ── Right panel: stacked bars — assigned to legend2 ──────────────────────────
for cat in _CONF_ORDER:
    means_r, err_lo_r, err_hi_r = [], [], []
    for cond in _ordered_conds:
        gt_val = ("InValid" not in cond and "Invalid" not in cond)
        sub = df_llm[df_llm["gt_valid"] == gt_val]
        indicator = (sub[sub["mod_type"] == cond]["valid_conf"] == cat).astype(float)
        lo, hi, mn, _ = bootstrap_ci(indicator)
        means_r.append(mn * 100)
        err_lo_r.append((mn - lo) * 100)
        err_hi_r.append((hi - mn) * 100)
    fig2.add_trace(go.Bar(
        name=cat, x=_ordered_labels, y=means_r,
        error_y=dict(type="data", symmetric=False,
                     array=err_hi_r, arrayminus=err_lo_r,
                     color="rgba(0,0,0,0.45)", thickness=1.2, width=4),
        marker_color=_CONF_COLORS[cat],
        legend="legend2", showlegend=True,
    ), row=1, col=2)

# Shade + divider for invalid section
_boundary_x = n_valid - 0.5
fig2.add_vrect(x0=_boundary_x, x1=n_total - 0.5,
               fillcolor="#7d3c98", opacity=0.10, line_width=0, row=1, col=2)
fig2.add_vline(x=_boundary_x, line_color="rgba(130,60,180,0.7)",
               line_width=1.5, line_dash="dash", row=1, col=2)

# VALID / INVALID CODE labels inside the chart (transparent vrect + annotation_text)
fig2.add_vrect(x0=-0.5, x1=_boundary_x, fillcolor="rgba(0,0,0,0)", line_width=0,
               annotation_text="VALID CODE — GT: Yes", annotation_position="top left",
               annotation_font=dict(size=9, color="#1a8040"),
               annotation_bgcolor="rgba(220,250,230,0.8)",
               annotation_bordercolor="#1a8040", annotation_borderwidth=1,
               row=1, col=2)
fig2.add_vrect(x0=_boundary_x, x1=n_total - 0.5, fillcolor="rgba(0,0,0,0)", line_width=0,
               annotation_text="INVALID CODE — GT: No", annotation_position="top right",
               annotation_font=dict(size=9, color="#7d3c98"),
               annotation_bgcolor="rgba(230,215,250,0.8)",
               annotation_bordercolor="#7d3c98", annotation_borderwidth=1,
               row=1, col=2)

# Two horizontal legends — centered under their respective panel
_LEG_STYLE = dict(orientation="h", yanchor="top",
                  bgcolor="rgba(255,255,255,0.95)", bordercolor="#ccc",
                  borderwidth=1, font=dict(size=10))

fig2.update_layout(
    barmode="stack",
    height=510, width=1200,
    template=TEMPLATE,
    legend=dict(**_LEG_STYLE,  x=_L2_CENT, y=-0.45, xanchor="center"),
    legend2=dict(**_LEG_STYLE, x=_R2_CENT, y=-0.45, xanchor="center"),
    margin=dict(l=65, r=20, t=60, b=265),
    font=dict(size=12),
)
fig2.update_yaxes(title_text="Score (%)", title_font=dict(size=11), row=1, col=1)
fig2.update_yaxes(title_text="% of predictions", range=[0, 112],
                  title_font=dict(size=11), row=1, col=2)
fig2.update_xaxes(tickangle=-30, tickfont=dict(size=10), row=1, col=1)
fig2.update_xaxes(tickangle=-25, tickfont=dict(size=10), row=1, col=2)

fig2.write_image("paper_fig2.png", scale=2)
print("✓ paper_fig2.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Reasoning vs Non-reasoning — PDE task score robustness
#           Legend: horizontal, below x-axis labels
# ═══════════════════════════════════════════════════════════════════════════════
REASONING_MODELS = {"QwQ-32B", "DeepSeek-R1-32B", "Qwen3-32B"}

for col in ["pde_match", "method_any_match", "behavior_any_match", "valid_match"]:
    df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")
df_llm["overall_acc"] = df_llm[["pde_match","method_any_match",
                                 "behavior_any_match","valid_match"]].mean(axis=1)
df_llm["model_type"] = df_llm["model_short"].apply(
    lambda m: "Reasoning" if m in REASONING_MODELS else "Non-reasoning"
)

_r_conds  = [c for c in ALL_CONDS if c in df_llm["mod_type"].unique()]
_r_labels = [COND_SHORT.get(c, c) for c in _r_conds]

# Compute max bar + CI value across both panels to set a clean y-axis ceiling
_bar_highs = []
for mtype in ["Reasoning", "Non-reasoning"]:
    for cond in _r_conds:
        data = df_llm[(df_llm["model_type"]==mtype) & (df_llm["mod_type"]==cond)]["overall_acc"]
        _, hi, _, _ = bootstrap_ci(data)
        _bar_highs.append(hi * 100)
for m in df_llm["model_short"].unique():
    _, hi, _, _ = bootstrap_ci(df_llm[df_llm["model_short"]==m]["overall_acc"])
    _bar_highs.append(hi * 100)
_y_ceil = int(np.ceil(max(_bar_highs) / 10) * 10) + 5   # next multiple of 10 + 5 buffer

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "PDE Task Score by Condition",
        "PDE Task Score by Model",
    ],
    horizontal_spacing=0.10,
)

# Left panel
for mtype, color in [("Reasoning", "#e67e22"), ("Non-reasoning", "#3498db")]:
    sub_type = df_llm[df_llm["model_type"] == mtype]
    means, err_lo, err_hi = [], [], []
    for cond in _r_conds:
        data = sub_type[sub_type["mod_type"] == cond]["overall_acc"]
        lo, hi, mn, _ = bootstrap_ci(data)
        means.append(mn * 100); err_lo.append((mn - lo) * 100); err_hi.append((hi - mn) * 100)
    fig3.add_trace(go.Bar(
        name=mtype, x=_r_labels, y=means,
        error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                     color="rgba(0,0,0,0.4)", thickness=1.2, width=4),
        marker_color=color, legendgroup=mtype, showlegend=True,
        hovertemplate=mtype + " | %{x}: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

for x0, x1, color in [
    ("Corrupt Comment", "Corrupt Variable",    "#e74c3c"),
    ("Invalid+Comment", "Invalid, No Comment", "#9b59b6"),
]:
    fig3.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.07,
                   line_width=0, row=1, col=1)

# Right panel
_model_order = sorted(df_llm["model_short"].unique(),
                      key=lambda m: df_llm[df_llm["model_short"]==m]["overall_acc"].mean())
for model in _model_order:
    is_r = model in REASONING_MODELS
    data = df_llm[df_llm["model_short"] == model]["overall_acc"]
    lo, hi, mn, _ = bootstrap_ci(data)
    fig3.add_trace(go.Bar(
        name=model, x=[model], y=[mn * 100],
        error_y=dict(type="data", symmetric=False,
                     array=[(hi - mn) * 100], arrayminus=[(mn - lo) * 100],
                     color="rgba(0,0,0,0.4)", thickness=1.2, width=4),
        marker_color="#e67e22" if is_r else "#3498db",
        marker_pattern_shape="x" if is_r else "",
        legendgroup="Reasoning" if is_r else "Non-reasoning",
        showlegend=False,
        hovertemplate=model + ": %{y:.1f}%<extra></extra>",
    ), row=1, col=2)

fig3.update_layout(
    title=dict(text="Reasoning Models Show Greater Robustness to Code Perturbation",
               font=dict(size=14)),
    barmode="group",
    height=540, width=1100,
    template=TEMPLATE,
    legend=dict(
        orientation="h", x=0.5, y=-0.48, xanchor="center", yanchor="top",
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#ccc", borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=65, r=20, t=80, b=260),
    font=dict(size=12),
)
# Shared y-axis label that names the task; both panels use same scale
_y_title = "Score (avg. PDE type, method, behavior, validity) %"
fig3.update_yaxes(title_text=_y_title, title_font=dict(size=10),
                  range=[0, _y_ceil], row=1, col=1)
fig3.update_yaxes(title_text=_y_title, title_font=dict(size=10),
                  range=[0, _y_ceil], row=1, col=2)
fig3.update_xaxes(tickangle=-25, tickfont=dict(size=10), row=1, col=1)
fig3.update_xaxes(tickangle=-35, tickfont=dict(size=10), row=1, col=2)

fig3.write_image("paper_fig3.png", scale=2)
print("✓ paper_fig3.png")
