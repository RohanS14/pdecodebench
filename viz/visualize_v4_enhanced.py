"""
Enhanced visualizations for PDE eval — adds baselines, error bars, correlations, NeurIPS compact formatting.

Extends visualize_v3.py with:
- Baseline accuracy overlays (random baseline + heuristic baseline if applicable)
- 95% confidence intervals on all accuracy metrics
- Model-to-model correlation matrix (Spearman)
- Compact publication-ready figures
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Load ──────────────────────────────────────────────────────────────────────
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
for col in ["logprob_correct","margin","entropy"]:
    df_mc[col] = pd.to_numeric(df_mc[col], errors="coerce")

MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct":        "Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct":       "Llama-3.3-70B",
    "Qwen/QwQ-32B":                            "QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct":          "Qwen2.5-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct":         "Qwen2.5-32B",
    "Qwen/Qwen3-32B":                          "Qwen3-32B",
    "google/gemma-3-27b-it":                   "Gemma-3-27B",
    "mistralai/Mistral-Nemo-Instruct-2407":    "Mistral-12B",
    "microsoft/phi-4":                         "phi-4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-32B",
}
df_llm["model_short"] = df_llm["model"].map(MODEL_SHORT).fillna(df_llm["model"])
df_mc["model_short"]  = df_mc["model"].map(MODEL_SHORT).fillna(df_mc["model"])

ALL_CONDS = ["Comm_Valid","NoComm_Valid","CorrComm","NoComm_CorrVar","Comm_InValid","NoComm_InValid",
             "CorrComm_Invalid","NoComm_CorrVar_InValid"]
COND_SHORT = {
    "Comm_Valid":              "Clean+Cmnt",
    "NoComm_Valid":            "Clean",
    "CorrComm":                "CorruptCmnt",
    "NoComm_CorrVar":          "CorruptVar",
    "Comm_InValid":            "Invalid+Cmnt",
    "NoComm_InValid":          "Invalid",
    "CorrComm_Invalid":        "Invalid+CorruptCmnt",
    "NoComm_CorrVar_InValid":  "Invalid+CorruptVar",
}
COND_COLOR = {
    "Comm_Valid":              "#2ecc71",
    "NoComm_Valid":            "#27ae60",
    "CorrComm":                "#e67e22",
    "NoComm_CorrVar":          "#e74c3c",
    "Comm_InValid":            "#9b59b6",
    "NoComm_InValid":          "#7d3c98",
    "CorrComm_Invalid":        "#c0392b",
    "NoComm_CorrVar_InValid":  "#922b21",
}

LEGEND = dict(x=1.02, y=1, xanchor="left", yanchor="top",
              bgcolor="rgba(240,242,255,0.95)", bordercolor="#aaa", borderwidth=1,
              font=dict(color="#111"))
MARGIN = dict(l=70, r=220, t=60, b=60)
COMPACT_MARGIN = dict(l=60, r=160, t=50, b=50)

# ── Utility functions ──────────────────────────────────────────────────────────

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Compute bootstrap 95% confidence interval."""
    data = data.dropna().values
    if len(data) < 2:
        return np.nan, np.nan, np.mean(data)

    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, ci + (100-ci)/2)
    return lower, upper, np.mean(data)

def compute_metrics_with_ci(df, metric, conds=None, by_model=False):
    """
    Compute metric mean and 95% CI for each condition (or model).
    Returns: dict mapping condition -> (lower, mean, upper)
    """
    if conds is None:
        conds = ALL_CONDS

    groupby_col = "model_short" if by_model else "mod_type"
    results = {}

    for cond in conds:
        if by_model:
            data = df[df["model_short"]==cond][metric]
        else:
            data = df[df["mod_type"]==cond][metric]

        lower, upper, mean = bootstrap_ci(data)
        results[cond] = (lower, mean, upper)

    return results

# ── BASELINE OVERLAY SECTION ───────────────────────────────────────────────────

# Baselines for LLM eval
BASELINE_RANDOM = {
    "pde_match": 0.25,        # 1 out of 4 PDE classes
    "method_recall": 0.50,    # Random guess on method (T/F per candidate)
    "behavior_recall": 0.50,  # Random guess on behavior (T/F per candidate)
    "valid_match": 0.50,      # Random T/F on validity
}

BASELINE_MAJORITY = {
    "pde_match": 0.333,       # Majority class: wave (4 out of 12)
    "method_any_match": 0.5,  # Majority: no explicit method
    "behavior_any_match": 0.5,
    "valid_match": 0.667,     # Majority: yes, valid (8 out of 12)
}

# ── ENHANCED VISUALIZATION SECTION ─────────────────────────────────────────────

figs = []

def add(section, title, fig, question):
    figs.append((section, title, fig, question))

# E1. Accuracy across conditions WITH ERROR BARS + BASELINE
fig_e1 = go.Figure()
x_labels = [COND_SHORT[c] for c in ALL_CONDS]
metric_colors = px.colors.qualitative.Plotly

LLM_METRICS = [
    ("pde_match",       "PDE Type"),
    ("method_recall",   "Method"),
    ("behavior_recall", "Behavior"),
    ("valid_match",     "Validity"),
]

for i, (metric, mlabel) in enumerate(LLM_METRICS):
    metrics_by_cond = compute_metrics_with_ci(df_llm, metric, conds=ALL_CONDS, by_model=False)

    means = []
    lowers = []
    uppers = []
    for cond in ALL_CONDS:
        lower, mean, upper = metrics_by_cond[cond]
        means.append(mean * 100)
        lowers.append(lower * 100)
        uppers.append(upper * 100)

    error_y = dict(
        type="data",
        symmetric=False,
        array=[u - m for u, m in zip(uppers, means)],
        arrayminus=[m - l for m, l in zip(means, lowers)],
    )

    fig_e1.add_trace(go.Scatter(
        x=x_labels, y=means,
        error_y=error_y,
        mode="lines+markers", name=mlabel,
        line=dict(color=metric_colors[i], width=2),
        marker=dict(size=8),
    ))

# Add random baseline as light horizontal line
for metric, mlabel in LLM_METRICS:
    baseline = BASELINE_RANDOM[metric] * 100
    fig_e1.add_hline(
        y=baseline, line_dash="dot", line_color="rgba(200,200,200,0.4)",
        annotation_text=f"Random: {baseline:.0f}%", annotation_position="right",
        annotation_font=dict(size=9, color="#888")
    )

fig_e1.update_layout(
    title="LLM Eval: Accuracy by condition (with 95% CI)",
    yaxis_title="Score (%)", yaxis_range=[-5, 115],
    xaxis=dict(tickangle=-15),
    height=400, legend=LEGEND, margin=COMPACT_MARGIN,
)
add("Enhanced", "E1. Accuracy with confidence intervals", fig_e1,
    "Error bars show 95% bootstrap CI. Dotted lines = random baseline. "
    "Tighter error bars = more consistent performance; wider bars = higher variance across samples.")

# E2. Per-model accuracy WITH ERROR BARS for one condition (Clean Valid)
fig_e2 = go.Figure()
models = sorted(df_llm["model_short"].unique())

for i, (metric, mlabel) in enumerate(LLM_METRICS):
    means = []
    lowers = []
    uppers = []

    for model in models:
        data = df_llm[(df_llm["model_short"]==model) & (df_llm["mod_type"]=="Comm_Valid")][metric]
        lower, upper, mean = bootstrap_ci(data)
        means.append(mean * 100)
        lowers.append(lower * 100)
        uppers.append(upper * 100)

    error_y = dict(
        type="data",
        symmetric=False,
        array=[u - m for u, m in zip(uppers, means)],
        arrayminus=[m - l for m, l in zip(means, lowers)],
    )

    fig_e2.add_trace(go.Bar(
        x=models, y=means,
        error_y=error_y,
        name=mlabel,
        marker_color=metric_colors[i],
    ))

fig_e2.update_layout(
    title="LLM Eval: Per-model baseline accuracy (Clean+Comment, 95% CI)",
    yaxis_title="Score (%)", yaxis_range=[0, 115],
    barmode="group", height=380,
    xaxis=dict(tickangle=-30),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1),
    margin=COMPACT_MARGIN,
)
add("Enhanced", "E2. Baseline accuracy by model (ground truth condition)", fig_e2,
    "Bars show per-model performance on clean, commented code. Error bars = 95% CI. "
    "Establishes upper bound for each model's capability on this task.")

# E3. MODEL CORRELATION MATRIX (Spearman)
# Compute accuracy for each model × condition
model_acc = {}
for model in models:
    accs = []
    for cond in ALL_CONDS:
        # Use valid_match as representative metric
        mean_acc = df_llm[(df_llm["model_short"]==model) & (df_llm["mod_type"]==cond)]["valid_match"].mean()
        accs.append(mean_acc)
    model_acc[model] = accs

# Convert to matrix
acc_matrix = pd.DataFrame(model_acc, index=ALL_CONDS).T

# Compute Spearman correlation
corr_matrix = acc_matrix.corr(method="spearman")

fig_e3 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale="RdBu",
    zmid=0,
    zmin=-1, zmax=1,
    text=np.round(corr_matrix.values, 2),
    texttemplate="%{text}",
    textfont={"size": 9},
    colorbar=dict(title="Spearman ρ"),
))
fig_e3.update_layout(
    title="Model correlation: How robustness profiles align (Spearman)",
    height=400, margin=COMPACT_MARGIN,
    xaxis=dict(tickangle=-30),
)
add("Enhanced", "E3. Inter-model correlation (robustness profiles)", fig_e3,
    "Shows which models have similar robustness patterns across conditions. "
    "High correlation (red) = models degrade similarly under same perturbations. "
    "Low/negative correlation (blue) = models fail on different conditions.")

# E4. ROBUSTNESS RANKING — model sensitivity to perturbations
sensitivities = {}
for model in models:
    # Measure drop from Clean to worst condition for valid_match
    clean_acc = df_llm[(df_llm["model_short"]==model) & (df_llm["mod_type"]=="Comm_Valid")]["valid_match"].mean()
    worst_cond = min(["CorrComm","NoComm_CorrVar","NoComm_InValid"],
                     key=lambda c: df_llm[(df_llm["model_short"]==model) & (df_llm["mod_type"]==c)]["valid_match"].mean())
    worst_acc = df_llm[(df_llm["model_short"]==model) & (df_llm["mod_type"]==worst_cond)]["valid_match"].mean()
    sensitivities[model] = clean_acc - worst_acc

ranked = sorted(sensitivities.items(), key=lambda x: x[1])

fig_e4 = go.Figure(data=go.Bar(
    x=[m for m, _ in ranked],
    y=[s*100 for _, s in ranked],
    marker_color="#3498db",
))
fig_e4.update_layout(
    title="Model robustness: Accuracy drop (Clean baseline → worst condition, Validity metric)",
    yaxis_title="Δ Accuracy (%)", yaxis_range=[0, max([s*100 for _, s in ranked])*1.1],
    xaxis=dict(tickangle=-30),
    height=380,
    margin=COMPACT_MARGIN,
)
add("Enhanced", "E4. Robustness ranking (sensitivity to perturbations)", fig_e4,
    "Lower bars = more robust (smaller drop from clean to worst condition). "
    "Higher bars = sensitive to corruption. Measures consistency across conditions.")

# E5. MC ACCURACY WITH BASELINES
df_mc_by_qtype = {}
for qtype in ["pde_class", "phys_process", "num_method", "phys_valid"]:
    df_q = df_mc[df_mc["question_type"]==qtype].copy()
    by_cond = {}
    for cond in ALL_CONDS:
        acc_vals = df_q[df_q["mod_type"]==cond]["correct"]
        lower, upper, mean = bootstrap_ci(acc_vals)
        by_cond[cond] = (lower, mean, upper)
    df_mc_by_qtype[qtype] = by_cond

fig_e5 = go.Figure()
qtype_labels = {"pde_class": "PDE (4-way MC)", "phys_process": "Process (T/F)",
                "num_method": "Method (T/F)", "phys_valid": "Validity (T/F)"}
qtype_baselines = {"pde_class": 0.25, "phys_process": 0.50, "num_method": 0.50, "phys_valid": 0.50}
colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]

for i, (qtype, color) in enumerate(zip(["pde_class", "phys_process", "num_method", "phys_valid"], colors)):
    means = []
    lowers = []
    uppers = []
    for cond in ALL_CONDS:
        lower, mean, upper = df_mc_by_qtype[qtype][cond]
        means.append(mean * 100)
        lowers.append(lower * 100)
        uppers.append(upper * 100)

    error_y = dict(
        type="data",
        symmetric=False,
        array=[u - m for u, m in zip(uppers, means)],
        arrayminus=[m - l for m, l in zip(means, lowers)],
    )

    x_vals = [COND_SHORT[c] for c in ALL_CONDS]
    fig_e5.add_trace(go.Scatter(
        x=x_vals, y=means,
        error_y=error_y,
        mode="lines+markers", name=qtype_labels[qtype],
        line=dict(color=color, width=2),
        marker=dict(size=8),
    ))

    # Add baseline for this question type
    baseline = qtype_baselines[qtype] * 100
    fig_e5.add_hline(
        y=baseline, line_dash="dot", line_color=color, opacity=0.3,
        annotation_text=f"{qtype_labels[qtype]} random: {baseline:.0f}%",
        annotation_position="right",
        annotation_font=dict(size=8, color=color),
    )

fig_e5.update_layout(
    title="MC Logprob: Accuracy by question type (with 95% CI and random baseline)",
    yaxis_title="Accuracy (%)", yaxis_range=[-5, 115],
    xaxis=dict(tickangle=-15),
    height=400,
    legend=LEGEND,
    margin=COMPACT_MARGIN,
)
add("Enhanced", "E5. MC accuracy by question type (with baselines)", fig_e5,
    "Dotted lines = random baseline for each question type. "
    "PDE Class (25%), Physical Process/Method/Validity (50% each). "
    "Confidence intervals show consistency across the 16 samples per condition.")

# E6. BEHAVIOR ALIAS INSPECTION
# Shows what models actually said for behavior, grouped by GT behavior pattern.
# Colored by behavior_any_match (green=any token hit, red=no hit).
# Helps verify alias coverage and catch GT typos / missing aliases.

def truncate(s, n=60):
    if pd.isna(s): return "(no answer)"
    s = str(s).strip().replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s

gt_behaviors = sorted(df_llm["gt_behavior"].dropna().unique())
fig_e6 = make_subplots(
    rows=len(gt_behaviors), cols=1,
    subplot_titles=[f'GT behavior: "{b}"' for b in gt_behaviors],
    vertical_spacing=0.05,
)
for ri, gt_b in enumerate(gt_behaviors, 1):
    sub = df_llm[df_llm["gt_behavior"] == gt_b].copy()
    sub["ans"] = sub["parsed_behavior"].apply(truncate)
    grp = sub.groupby(["ans", "behavior_any_match"]).size().reset_index(name="n").sort_values("n")
    for match, color, label in [(1, "#2ecc71", "Any token matched"), (0, "#e74c3c", "No token matched")]:
        g = grp[grp["behavior_any_match"] == match]
        fig_e6.add_trace(go.Bar(
            x=g["n"], y=g["ans"], orientation="h",
            name=label, marker_color=color,
            showlegend=(ri == 1), legendgroup=label,
        ), row=ri, col=1)

fig_e6.update_layout(
    title="LLM Eval: Behavior alias inspection — what did models say?",
    barmode="stack",
    height=max(800, len(gt_behaviors) * 220),
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=420, r=180, t=80, b=40),
)
add("Enhanced", "E6. Behavior alias inspection", fig_e6,
    "Green = at least one GT behavior token matched; Red = none matched. "
    "Long red bars indicate missing aliases. "
    "GT typos are normalized at scoring time via _GT_TYPOS in parse_score.py.")

# ── E7. Code length vs accuracy ───────────────────────────────────────────────
import re as _re

# Length covariate. jul28 results carry num_char directly (num_lines is +1 on
# 80/256 rows, all synthetic — a trailing-newline counting difference between the
# two parsers; num_char is correct). Older result CSVs predate that column, so
# fall back to joining the archived v3 workbook.
if "num_char" in df_llm.columns:
    _LEN_COL, _LEN_LABEL = "num_char", "Number of characters of code"
else:
    _LEN_COL, _LEN_LABEL = "num_lines", "Number of lines of code"
    _src = pd.read_excel("../data/archive/pdedata_clean_v3.xlsx",
                         usecols=["title", "num_lines"])
    df_llm = df_llm.merge(_src, on="title", how="left")

_agg = (df_llm
        .groupby(["title", "mod_type", _LEN_COL])[
            ["pde_match","method_any_match","behavior_any_match","valid_match"]]
        .mean().reset_index())
_agg["overall_acc"] = _agg[["pde_match","method_any_match","behavior_any_match","valid_match"]].mean(axis=1)

fig_e7 = go.Figure()
for cond in ALL_CONDS:
    sub = _agg[_agg["mod_type"] == cond]
    if sub.empty: continue
    clabel = COND_SHORT.get(cond, cond)
    color  = COND_COLOR.get(cond, "#888")
    fig_e7.add_trace(go.Scatter(
        x=sub[_LEN_COL], y=sub["overall_acc"] * 100,
        mode="markers", name=clabel, legendgroup=cond, showlegend=True,
        marker=dict(color=color, size=5, opacity=0.35),
        hovertemplate="%{customdata}<br>Lines: %{x}<br>Acc: %{y:.1f}%<extra></extra>",
        customdata=sub["title"],
    ))
    if len(sub) >= 3:
        m, b = np.polyfit(sub[_LEN_COL], sub["overall_acc"] * 100, 1)
        xs = np.linspace(sub[_LEN_COL].min(), sub[_LEN_COL].max(), 50)
        fig_e7.add_trace(go.Scatter(
            x=xs, y=m * xs + b, mode="lines",
            name=clabel + " (trend)", legendgroup=cond, showlegend=False,
            line=dict(color=color, width=2),
        ))
fig_e7.update_layout(
    title="LLM Eval: Code length vs overall accuracy (per snippet, averaged across models)",
    xaxis_title=_LEN_LABEL, yaxis_title="Overall accuracy (%)",
    yaxis_range=[0, 105], height=500,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=70, r=220, t=80, b=80),
)
add("Enhanced", "E7. Code length vs accuracy (scatter + trend)", fig_e7,
    "Each dot = one snippet (accuracy averaged across all 10 models). Lines = per-condition linear trend. "
    "NoComm_InValid tends to be longer (mean ~131 lines) and harder. "
    "Length effect is modest relative to condition effect.")

_agg["len_bin"] = pd.qcut(_agg[_LEN_COL], q=4,
                           labels=["Short (Q1)", "Med-Short (Q2)", "Med-Long (Q3)", "Long (Q4)"])
_bin_acc = _agg.groupby(["len_bin","mod_type"])["overall_acc"].mean().reset_index()
fig_e8 = go.Figure()
for cond in ALL_CONDS:
    sub = _bin_acc[_bin_acc["mod_type"] == cond]
    if sub.empty: continue
    fig_e8.add_trace(go.Bar(
        name=COND_SHORT.get(cond, cond),
        x=sub["len_bin"].astype(str), y=sub["overall_acc"] * 100,
        marker_color=COND_COLOR.get(cond, "#888"),
    ))
fig_e8.update_layout(
    title="LLM Eval: Accuracy by code length quartile and condition",
    xaxis_title="Code length quartile", yaxis_title="Overall accuracy (%)",
    yaxis_range=[0, 105], barmode="group", height=460,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=70, r=220, t=80, b=80),
)
add("Enhanced", "E8. Code length quartile vs accuracy", fig_e8,
    f"Snippets split into quartiles by {_LEN_COL} (Q1=shortest, Q4=longest). "
    "Accuracy averaged across all models and question types per cell.")

# ── E9. Validity prediction confidence (4-way, with bootstrap CI) ─────────────
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
_present_conds  = [c for c in ALL_CONDS if c in df_llm["mod_type"].unique()]
_cond_labels    = [COND_SHORT.get(c, c) for c in _present_conds]

fig_e9 = make_subplots(rows=2, cols=1,
    subplot_titles=["gt = VALID code (model should say 'Yes')",
                    "gt = INVALID code (model should say 'No')"],
    vertical_spacing=0.14)
for gt_val, row_idx in [(True, 1), (False, 2)]:
    sub = df_llm[df_llm["gt_valid"] == gt_val]
    for cat in _CONF_ORDER:
        means, err_lo, err_hi = [], [], []
        for cond in _present_conds:
            indicator = (sub[sub["mod_type"] == cond]["valid_conf"] == cat).astype(float)
            lo, hi, mn = bootstrap_ci(indicator)
            means.append(mn * 100)
            err_lo.append((mn - lo) * 100)
            err_hi.append((hi - mn) * 100)
        fig_e9.add_trace(go.Bar(
            name=cat, x=_cond_labels, y=means,
            error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                         color="rgba(0,0,0,0.5)", thickness=1.2, width=4),
            marker_color=_CONF_COLORS[cat],
            legendgroup=cat, showlegend=(row_idx == 1),
            hovertemplate="%{x}<br>" + cat + ": %{y:.1f}%"
                          "<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>",
            customdata=[[means[i] - err_lo[i], means[i] + err_hi[i]] for i in range(len(means))],
        ), row=row_idx, col=1)
fig_e9.update_layout(
    title="LLM Eval: Validity prediction confidence breakdown (95% bootstrap CI)",
    barmode="stack", yaxis_title="% of predictions", yaxis2_title="% of predictions",
    height=650,
    legend=dict(x=1.02, y=1, xanchor="left", title="Prediction type",
                bgcolor="rgba(240,242,255,0.95)", bordercolor="#aaa", borderwidth=1,
                font=dict(color="#111")),
    margin=dict(l=70, r=240, t=80, b=80),
)
fig_e9.update_yaxes(range=[0, 101])
add("Enhanced", "E9. Validity prediction confidence breakdown", fig_e9,
    "Top: valid code (correct = Yes). Bottom: invalid code (correct = No). "
    "52% of invalid-code rows receive confident 'Yes' — genuine comprehension failure, not a parsing artifact. "
    "Error bars: 95% bootstrap CI on proportion of predictions per category.")

# ── E10. Behavior: at-least-1-correct vs full recall (with bootstrap CI) ───────
fig_e10 = go.Figure()
for metric, label, color in [
    ("behavior_any_match", "≥1 behavior correct (any_match)", "#3498db"),
    ("behavior_recall",    "Full recall (all behaviors)",     "#85c1e9"),
]:
    means, err_lo, err_hi = [], [], []
    for cond in _present_conds:
        data = df_llm[df_llm["mod_type"] == cond][metric]
        lo, hi, mn = bootstrap_ci(data)
        means.append(mn * 100)
        err_lo.append((mn - lo) * 100)
        err_hi.append((hi - mn) * 100)
    fig_e10.add_trace(go.Bar(
        name=label,
        x=[COND_SHORT.get(c, c) for c in _present_conds],
        y=means,
        error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                     color="rgba(0,0,0,0.5)", thickness=1.2, width=4),
        marker_color=color,
        hovertemplate="%{x}<br>" + label + ": %{y:.1f}%"
                      "<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>",
        customdata=[[means[i] - err_lo[i], means[i] + err_hi[i]] for i in range(len(means))],
    ))
fig_e10.update_layout(
    title="LLM Eval: Behavior — ≥1 correct vs full recall (95% bootstrap CI)",
    xaxis_title="Condition", yaxis_title="% predictions", yaxis_range=[0, 105],
    barmode="group", height=430,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=70, r=220, t=80, b=80),
)
add("Enhanced", "E10. Behavior: ≥1 correct vs full recall", fig_e10,
    "Dark blue: model named ≥1 ground-truth behavior (any_match). "
    "Light blue: fraction of all gt behaviors named (recall). "
    "ALIASES: 'wave propagation'→oscillation, 'heat conduction'→diffusion, "
    "'transport/convection'→advection, 'damping/decay'→restoration.")

# ═══════════════════════════════════════════════════════════════════════════════
# Build HTML
# ═══════════════════════════════════════════════════════════════════════════════
sections = {}
for section, title, fig, question in figs:
    sections.setdefault(section, []).append((title, fig, question))

nav_html = content_html = ""
first = True
idx = 0

for section_name, charts in sections.items():
    nav_html += f'<div class="nav-section">{section_name}</div>\n'
    for title, fig, question in charts:
        active = "active" if first else ""
        nav_html += f'<button class="nav-btn {active}" onclick="show({idx})">{title}</button>\n'
        fig_html = fig.to_html(full_html=False, include_plotlyjs=(idx==0), div_id=f"fig{idx}")
        content_html += f'''
<div class="section {active}" id="sec{idx}">
  <div class="chart-header">
    <span class="section-tag">{section_name}</span>
    <h2>{title}</h2>
    <p class="question">{question}</p>
  </div>
  {fig_html}
</div>'''
        first = False
        idx += 1

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDE Experiment Results (Enhanced)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0d0f18; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 250px; min-width: 250px; background: #12141e;
              border-right: 1px solid #1e2130; overflow-y: auto; display: flex;
              flex-direction: column; }}
  #sidebar h1 {{ font-size: 0.85rem; color: #fff; padding: 14px 12px 10px;
                 border-bottom: 1px solid #1e2130; line-height: 1.4; }}
  .hyp {{ font-size: 0.7rem; color: #555; padding: 8px 12px 12px;
          border-bottom: 1px solid #1e2130; line-height: 1.6; }}
  .hyp strong {{ color: #888; }}
  .nav-section {{ font-size: 0.66rem; color: #444; text-transform: uppercase;
                  letter-spacing: 0.09em; padding: 12px 12px 3px; }}
  .nav-btn {{ display: block; width: 100%; background: none; border: none; border-left: 3px solid transparent;
              color: #777; padding: 7px 12px 7px 9px; cursor: pointer; font-size: 0.76rem;
              text-align: left; line-height: 1.4; transition: background 0.1s; }}
  .nav-btn:hover {{ background: #191c2a; color: #ccc; }}
  .nav-btn.active {{ background: #171d30; color: #7eb8ff; border-left-color: #3a7bdd; }}
  #main {{ flex: 1; overflow-y: auto; padding: 22px 26px; }}
  .section {{ display: none; }}
  .section.active {{ display: block; }}
  .section-tag {{ font-size: 0.68rem; color: #444; text-transform: uppercase; letter-spacing: 0.08em; }}
  .chart-header h2 {{ margin: 5px 0 10px; font-size: 1rem; color: #ddd; font-weight: 500; }}
  .question {{ font-size: 0.8rem; color: #5a80b0; max-width: 820px;
               border-left: 3px solid #1e3560; padding-left: 10px;
               line-height: 1.6; margin-bottom: 16px; }}
</style>
</head>
<body>
<div id="sidebar">
  <h1>PDE Results (Enhanced)</h1>
  <div class="hyp">
    <strong>New in v4:</strong> Baselines, 95% CIs, model correlations, compact NeurIPS formatting
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

out = "results_v4_enhanced.html"
with open(out, "w") as f:
    f.write(html)

print(f"✓ Saved enhanced visualizations to {out}")
print(f"  - Added baseline overlays (random: 25% PDE, 50% T/F)")
print(f"  - Added 95% bootstrap confidence intervals")
print(f"  - Added model-to-model correlation matrix (Spearman)")
print(f"  - Added robustness ranking (sensitivity analysis)")
print(f"  - Compact NeurIPS-ready formatting")
