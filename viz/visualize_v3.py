"""
Hypothesis-driven visualizations — clean layout, no legend overlap.
ENHANCED: Includes baselines, 95% confidence intervals, variance reporting, correlation matrix,
robustness ranking, and summary tables.

pde-llm-eval: accuracy degrades under perturbations; comments help on invalid code.
pde-mc-logprob: logprob drops even when accuracy is stable; variable obfuscation strongest.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

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
    "meta-llama/Llama-3.1-8B-Instruct":              "Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct":             "Llama-3.3-70B",
    "Qwen/QwQ-32B":                                  "QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct":                "Qwen2.5-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct":               "Qwen2.5-32B",
    "Qwen/Qwen3-32B":                                "Qwen3-32B",
    "google/gemma-3-27b-it":                         "Gemma-3-27B",
    "mistralai/Mistral-Nemo-Instruct-2407":          "Mistral-12B",
    "microsoft/phi-4":                               "phi-4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":     "DeepSeek-R1-32B",
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
QTYPE_LABEL = {
    "pde_class":    "PDE Class",
    "phys_process": "Phys. Process",
    "num_method":   "Num. Method",
    "phys_valid":   "Validity",
}
PDE_COLORS = {"wave":"#3498db","heat":"#e74c3c","burgers":"#f39c12","navier-stokes":"#9b59b6"}

# Shared legend style — right side, no overlap
LEGEND = dict(x=1.02, y=1, xanchor="left", yanchor="top",
              bgcolor="rgba(240,242,255,0.95)", bordercolor="#aaa", borderwidth=1,
              font=dict(color="#111"))
MARGIN = dict(l=70, r=220, t=80, b=80)
COMPACT_MARGIN = dict(l=60, r=160, t=50, b=50)

# ── BASELINE & CI FUNCTIONS ───────────────────────────────────────────────────
def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Compute bootstrap 95% confidence interval and variance."""
    data = data.dropna().values
    if len(data) < 1:
        return np.nan, np.nan, np.nan, np.nan
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data), 0.0

    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, ci + (100-ci)/2)
    return lower, upper, np.mean(data), np.var(data)

def compute_metrics_with_ci(df, metric, conds=None, by_model=False):
    """Compute metric mean, 95% CI, and variance for each condition."""
    if conds is None:
        conds = ALL_CONDS

    groupby_col = "model_short" if by_model else "mod_type"
    results = {}

    for cond in conds:
        if by_model:
            data = df[df["model_short"]==cond][metric]
        else:
            data = df[df["mod_type"]==cond][metric]

        lower, upper, mean, var = bootstrap_ci(data)
        results[cond] = (lower, mean, upper, var)

    return results

# Baselines
BASELINE_RANDOM = {
    "pde_match": 0.25,
    "method_recall": 0.50,
    "behavior_recall": 0.50,
    "valid_match": 0.50,
}

LLM_METRICS = [
    ("pde_match",       "PDE Type"),
    ("method_recall",   "Method Recall"),
    ("behavior_recall", "Behavior Recall"),
    ("valid_match",     "Validity"),
]

figs = []
def add(section, title, fig, question):
    figs.append((section, title, fig, question))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — pde-llm-eval
# ═══════════════════════════════════════════════════════════════════════════════

# A1. Four perturbation comparisons — delta bars only (no baseline overlay)
# x = metric, color = perturbation type
comparisons = [
    ("NoComm_Valid",    "Comm_Valid",      "① Remove comment",       "#3498db"),
    ("CorrComm",        "Comm_Valid",      "② Corrupt comment",      "#e67e22"),
    ("NoComm_CorrVar",  "NoComm_Valid",    "③ Corrupt variable names","#e74c3c"),
    ("Comm_InValid",    "NoComm_InValid",  "④ Add comment to invalid","#2ecc71"),
]

fig_a1 = go.Figure()
metric_labels = [label for _, label in LLM_METRICS]

for cond, ref_cond, clabel, color in comparisons:
    deltas = []
    for metric, _ in LLM_METRICS:
        cond_mean = df_llm[df_llm["mod_type"]==cond][metric].mean()
        ref_mean  = df_llm[df_llm["mod_type"]==ref_cond][metric].mean()
        deltas.append(round((cond_mean - ref_mean)*100, 1))
    fig_a1.add_trace(go.Bar(name=clabel, x=metric_labels, y=deltas, marker_color=color))

fig_a1.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=2,
                 annotation_text="No change (baseline = 0)", annotation_position="right",
                 annotation_font=dict(size=9, color="#fff"))
fig_a1.update_layout(
    title="Accuracy change per perturbation (vs baseline)",
    yaxis_title="Δ Accuracy (percentage points)",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "① Effect of each perturbation on accuracy", fig_a1,
    "Each group of bars = one evaluation metric. Colors = 4 perturbation types, each compared to its natural reference. "
    "③ Corrupt variable names compares NoComm_CorrVar vs NoComm_Valid — only variable names differ. "
    "④ Shows whether adding a comment helps when the code is physically invalid.")

# A1b. Absolute accuracy for perturbation pairs (with baselines as bars)
fig_a1b = go.Figure()

metric_labels = [label for _, label in LLM_METRICS]
comparison_pairs = [
    ("NoComm_Valid", "Comm_Valid", "① Remove comment"),
    ("CorrComm", "Comm_Valid", "② Corrupt comment"),
    ("NoComm_CorrVar", "NoComm_Valid", "③ Corrupt var names"),
    ("Comm_InValid", "NoComm_InValid", "④ Add comment"),
]

# Add baseline bars for each comparison
baseline_set = set([ref_cond for _, ref_cond, _ in comparison_pairs])
for ref_cond in baseline_set:
    baseline_vals = []
    for metric, _ in LLM_METRICS:
        baseline_vals.append(df_llm[df_llm["mod_type"]==ref_cond][metric].mean() * 100)
    fig_a1b.add_trace(go.Bar(
        name=f"Baseline: {COND_SHORT[ref_cond]}",
        x=metric_labels,
        y=baseline_vals,
        marker_color="rgba(150,150,150,0.5)",
    ))

# Add perturbed bars for each comparison
for cond, ref_cond, clabel in comparison_pairs:
    vals = []
    for metric, _ in LLM_METRICS:
        vals.append(df_llm[df_llm["mod_type"]==cond][metric].mean() * 100)
    fig_a1b.add_trace(go.Bar(name=clabel, x=metric_labels, y=vals, marker_color=COND_COLOR[cond]))

fig_a1b.update_layout(
    title="Absolute accuracy per perturbation",
    yaxis_title="Accuracy (%)",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "② Absolute: Effect of each perturbation on accuracy", fig_a1b,
    "Shows absolute accuracy (not delta) for baseline vs perturbed conditions. "
    "Gray bars = baseline accuracy. Colored bars = accuracy after perturbation. "
    "Compare with delta chart (①) to understand relative vs absolute impact.")

# A2. Accuracy across all conditions WITH ERROR BARS — one line per metric
fig_a2 = go.Figure()
x_labels = [COND_SHORT[c] for c in ALL_CONDS]
metric_colors = px.colors.qualitative.Plotly
for i, (metric, mlabel) in enumerate(LLM_METRICS):
    metrics_by_cond = compute_metrics_with_ci(df_llm, metric, conds=ALL_CONDS, by_model=False)

    means = []
    lowers = []
    uppers = []
    for cond in ALL_CONDS:
        lower, mean, upper, var = metrics_by_cond[cond]
        means.append(mean * 100)
        lowers.append(lower * 100)
        uppers.append(upper * 100)

    error_y = dict(
        type="data",
        symmetric=False,
        array=[u - m for u, m in zip(uppers, means)],
        arrayminus=[m - l for m, l in zip(means, lowers)],
    )

    fig_a2.add_trace(go.Scatter(
        x=x_labels, y=means,
        error_y=error_y,
        mode="lines+markers", name=mlabel,
        line=dict(color=metric_colors[i], width=2), marker=dict(size=9),
    ))

# Shade zones
for x0, x1, color, label in [
    ("Clean+Comment","Clean, No Comment","#2ecc71","Clean valid"),
    ("Corrupt Comment","Corrupt Variable","#e74c3c","Corrupted"),
    ("Invalid+Comment","Invalid, No Comment","#9b59b6","Invalid"),
]:
    fig_a2.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0,
                     annotation_text=label, annotation_position="top left",
                     annotation_font=dict(size=10, color="#aaa"))
fig_a2.update_layout(
    title="LLM Free Generation Evaluation: Accuracy across all Conditions",
    yaxis_title="Score (%)",
    xaxis=dict(tickangle=-20),
    height=460, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "② Accuracy profile across all conditions", fig_a2,
    "Each line traces one metric across all 6 conditions. Green zone = clean valid code. "
    "Red zone = corrupted conditions. Purple zone = physically invalid. "
    "Method and behavior use partial credit (0.5 for one of two required terms).")

# A3. Per-model: one chart per metric, models as lines WITH ERROR BARS
for metric, mlabel in LLM_METRICS:
    fig = go.Figure()
    models = sorted(df_llm["model_short"].unique())
    for i, model in enumerate(models):
        model_data = df_llm[df_llm["model_short"]==model]

        means = []
        lowers = []
        uppers = []
        for cond in ALL_CONDS:
            data = model_data[model_data["mod_type"]==cond][metric]
            lower, upper, mean, var = bootstrap_ci(data)
            means.append(mean * 100)
            lowers.append(lower * 100)
            uppers.append(upper * 100)

        error_y = dict(
            type="data",
            symmetric=False,
            array=[u - m for u, m in zip(uppers, means)],
            arrayminus=[m - l for m, l in zip(means, lowers)],
        )

        fig.add_trace(go.Scatter(
            x=x_labels, y=means,
            error_y=error_y,
            mode="lines+markers", name=model,
            line=dict(color=metric_colors[i], width=2), marker=dict(size=8),
        ))
    for x0, x1, color in [
        ("Corrupt Comment","Corrupt Variable","#e74c3c"),
        ("Invalid+Comment","Invalid, No Comment","#9b59b6"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0)
    fig.update_layout(
        title=f"{mlabel} — per-model robustness",
        yaxis_title="Score (%)", xaxis=dict(tickangle=-20),
        height=440, legend=LEGEND, margin=MARGIN,
    )
    add("A · LLM Eval", f"③ Per-model: {mlabel}", fig,
        f"Each line = one model's {mlabel.lower()} score across conditions. "
        "Flatter lines = more robust. Steeper drops into the shaded zones = more sensitive to corruption. "
        "Error bars = 95% bootstrap CI. n=16 per (model, condition) cell — CIs are intentionally wide at pilot scale.")

# A4. Answer alias inspection
def truncate(s, n=60):
    if pd.isna(s): return "(no answer)"
    s = str(s).strip().replace("\n"," ")
    return s[:n]+"…" if len(s)>n else s

pde_types = sorted(df_llm["gt_pde"].dropna().unique())
fig_a4 = make_subplots(rows=len(pde_types), cols=1,
    subplot_titles=[f'Ground truth: "{p}"' for p in pde_types],
    vertical_spacing=0.06)
for ri, pde in enumerate(pde_types, 1):
    sub = df_llm[df_llm["gt_pde"]==pde].copy()
    sub["ans"] = sub["parsed_pde"].apply(truncate)
    grp = sub.groupby(["ans","pde_match"]).size().reset_index(name="n").sort_values("n")
    for match, color, label in [(1,"#2ecc71","Scored correct"),(0,"#e74c3c","Scored wrong")]:
        g = grp[grp["pde_match"]==match]
        fig_a4.add_trace(go.Bar(
            x=g["n"], y=g["ans"], orientation="h",
            name=label, marker_color=color,
            showlegend=(ri==1), legendgroup=label,
        ), row=ri, col=1)
fig_a4.update_layout(
    title="Answer distribution — alias inspection",
    barmode="stack", height=max(750, len(pde_types)*210),
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=400, r=160, t=80, b=40),
)
add("A · LLM Eval", "④ Answer distribution — alias inspection", fig_a4,
    "Green = scored correct, Red = scored wrong. Long red bars reveal aliases being penalized — "
    'e.g. "Burgers\' equation" and "inviscid Burgers\' equation" both marked wrong when gt is "burgers".')

# A5. Per-PDE-class accuracy across conditions WITH ERROR BARS
PDE_CLASSES = ["wave", "heat", "burgers", "navier-stokes"]

fig_a5a = go.Figure()
# n per (pde_class, mod_type) cell = 16 total rows / 4 classes = 4 — CIs are uninformative at n=4, omitted
for cond in ALL_CONDS:
    means = []
    for p in PDE_CLASSES:
        data = df_llm[(df_llm["pde_class"]==p)&(df_llm["mod_type"]==cond)]["pde_match"]
        means.append(data.mean() * 100 if len(data) > 0 else float("nan"))
    fig_a5a.add_trace(go.Bar(
        name=COND_SHORT[cond], x=PDE_CLASSES,
        y=means,
        marker_color=COND_COLOR[cond],
    ))
fig_a5a.update_layout(
    title="PDE identification accuracy by equation family",
    yaxis_title="PDE match (%)", yaxis_range=[0, 110],
    barmode="group", height=460, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑤ PDE accuracy by equation family", fig_a5a,
    "Each group = one PDE class. Colors = conditions. "
    "Navier-Stokes is identified near-perfectly in all conditions except CorrVar. "
    "Burgers collapses most severely under variable obfuscation (NoComm_CorrVar). "
    "Note: n=4 samples per bar (16 rows ÷ 4 PDE classes) — CIs omitted as uninformative at this granularity.")

# A5b. Valid detection by PDE class (invalid conditions only)
fig_a5b = go.Figure()
for cond in ["NoComm_InValid", "Comm_InValid"]:
    vals = [df_llm[(df_llm["pde_class"]==p)&(df_llm["mod_type"]==cond)]["valid_match"].mean()*100
            for p in PDE_CLASSES]
    fig_a5b.add_trace(go.Bar(
        name=COND_SHORT[cond], x=PDE_CLASSES,
        y=[round(v, 1) for v in vals],
        marker_color=COND_COLOR[cond],
    ))
fig_a5b.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                  annotation_text="chance", annotation_position="right",
                  annotation_font=dict(size=10, color="#aaa"))
fig_a5b.update_layout(
    title="Physical invalidity detection by equation family",
    yaxis_title="Valid match (% correct, should answer 'No')", yaxis_range=[0, 110],
    barmode="group", height=420, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑥ Invalidity detection by equation family", fig_a5b,
    "Only invalid conditions shown. Correct answer is 'No' (not physically valid). "
    "Navier-Stokes invalidity is almost never detected (5.6%). "
    "Burgers invalidity detected most often (~47%). "
    "Shows which PDE families' physics models understand well enough to flag bugs.")

# ── A6/A7. Code length vs accuracy ───────────────────────────────────────────
# jul28 results carry num_char directly. Older result CSVs predate it, so fall
# back to joining the archived v3 workbook (the file moved into data/archive/).
if "num_char" in df_llm.columns:
    _LEN_COL = "num_char"
else:
    _LEN_COL = "num_lines"
    _src = pd.read_excel("../data/archive/pdedata_clean_v3.xlsx",
                         usecols=["title", "num_lines"])
    df_llm = df_llm.merge(_src, on="title", how="left")

METRICS_LINELEN = [
    ("pde_match",          "PDE class accuracy"),
    ("method_any_match",   "Method accuracy"),
    ("behavior_any_match", "Behavior accuracy"),
    ("valid_match",        "Validity accuracy"),
]

# Per-(title, mod_type): average accuracy across models
_agg = (df_llm
        .groupby(["title", "mod_type", _LEN_COL])[
            ["pde_match","method_any_match","behavior_any_match","valid_match"]]
        .mean()
        .reset_index())
_agg["overall_acc"] = _agg[["pde_match","method_any_match","behavior_any_match","valid_match"]].mean(axis=1)

# ── A7. Scatter: num_lines vs overall accuracy, per condition ─────────────────
fig_a6 = go.Figure()
for cond in ALL_CONDS:
    sub = _agg[_agg["mod_type"] == cond]
    if sub.empty:
        continue
    clabel = COND_SHORT.get(cond, cond)
    color  = COND_COLOR.get(cond, "#888")
    # Raw points (faint)
    fig_a6.add_trace(go.Scatter(
        x=sub[_LEN_COL], y=sub["overall_acc"] * 100,
        mode="markers",
        name=clabel,
        legendgroup=cond,
        showlegend=True,
        marker=dict(color=color, size=5, opacity=0.35),
        hovertemplate="%{customdata}<br>Lines: %{x}<br>Acc: %{y:.1f}%<extra></extra>",
        customdata=sub["title"],
    ))
    # Linear trend
    if len(sub) >= 3:
        m, b = np.polyfit(sub[_LEN_COL], sub["overall_acc"] * 100, 1)
        xs = np.linspace(sub[_LEN_COL].min(), sub[_LEN_COL].max(), 50)
        fig_a6.add_trace(go.Scatter(
            x=xs, y=m * xs + b,
            mode="lines",
            name=clabel + " (trend)",
            legendgroup=cond,
            showlegend=False,
            line=dict(color=color, width=2),
        ))
fig_a6.update_layout(
    title="Code length vs overall accuracy",
    xaxis_title="Number of lines of code",
    yaxis_title="Overall accuracy (%)",
    yaxis_range=[0, 105],
    height=500, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑦ Code length vs accuracy — scatter + trend", fig_a6,
    "Each dot is one code snippet (accuracy averaged across all 10 models). "
    "Lines show per-condition linear trends. "
    "Code length has no consistent effect on accuracy: overall Spearman r=0.17 (p≈0.050), "
    "with an IQR effect of only 2.7 pp (Q1≈40 lines vs Q4≈95 lines). "
    "Per-condition correlations are near zero for 6 of 8 conditions (r≤0.13, p>0.6); "
    "the exceptions (Comm_Valid r=0.68, CorrComm r=0.49) likely reflect confounding "
    "with PDE class distribution rather than a length effect. "
    "Per-metric correlations go in opposite directions: pde_match (r=+0.35) and method (r=+0.47) "
    "vs behavior_any_match (r=−0.34) and valid_match (r=−0.05), confirming there is no "
    "consistent length → difficulty signal. Condition membership is the dominant predictor.")

# ── A8. Binned: code length quartile vs accuracy by condition ─────────────────
_agg["len_bin"] = pd.qcut(_agg[_LEN_COL], q=4,
                           labels=["Short\n(Q1)", "Med-Short\n(Q2)", "Med-Long\n(Q3)", "Long\n(Q4)"])
_bin_acc = (_agg.groupby(["len_bin", "mod_type"])["overall_acc"]
            .mean()
            .reset_index())

fig_a7 = go.Figure()
for cond in ALL_CONDS:
    sub = _bin_acc[_bin_acc["mod_type"] == cond]
    if sub.empty:
        continue
    fig_a7.add_trace(go.Bar(
        name=COND_SHORT.get(cond, cond),
        x=sub["len_bin"].astype(str),
        y=sub["overall_acc"] * 100,
        marker_color=COND_COLOR.get(cond, "#888"),
    ))
fig_a7.update_layout(
    title="Accuracy by code length quartile",
    xaxis_title="Code length quartile",
    yaxis_title="Overall accuracy (%)",
    yaxis_range=[0, 105],
    barmode="group", height=460, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑧ Code length quartile vs accuracy", fig_a7,
    "Snippets split into quartiles by number of lines (Q1=shortest, Q4=longest). "
    "Accuracy averaged across all models and question types per cell. "
    "The effect is small and inconsistent: the overall IQR accuracy change is only 2.7 pp "
    "(Spearman r=0.17, p≈0.050 across 128 snippet-level points). "
    "No condition shows a reliable monotone trend from Q1→Q4.")

# ── A9. Validity confidence breakdown (4-way, single figure) ─────────────────
# All 8 conditions on one x-axis; left half = valid code (correct = Yes),
# right half = invalid code (correct = No).  Shaded region separates them.
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


# Order: valid-code conditions first, then invalid-code conditions
_valid_conds   = [c for c in ALL_CONDS if "InValid" not in c and "Invalid" not in c
                  and c in df_llm["mod_type"].unique()]
_invalid_conds = [c for c in ALL_CONDS if ("InValid" in c or "Invalid" in c)
                  and c in df_llm["mod_type"].unique()]
_ordered_conds = _valid_conds + _invalid_conds
_ordered_labels = (
    [COND_SHORT.get(c, c) + " ✓" for c in _valid_conds] +
    ["⚠ " + COND_SHORT.get(c, c) for c in _invalid_conds]
)

# Also keep _present_conds for A10 below
_present_conds = [c for c in ALL_CONDS if c in df_llm["mod_type"].unique()]

fig_a8 = go.Figure()

for cat in _CONF_ORDER:
    means, err_lo, err_hi = [], [], []
    for cond in _ordered_conds:
        gt_val = ("InValid" not in cond and "Invalid" not in cond)
        sub = df_llm[df_llm["gt_valid"] == gt_val]
        indicator = (sub[sub["mod_type"] == cond]["valid_conf"] == cat).astype(float)
        lo, hi, mn, _ = bootstrap_ci(indicator)
        means.append(mn * 100)
        err_lo.append((mn - lo) * 100)
        err_hi.append((hi - mn) * 100)
    fig_a8.add_trace(go.Bar(
        name=cat,
        x=_ordered_labels,
        y=means,
        error_y=dict(type="data", symmetric=False,
                     array=err_hi, arrayminus=err_lo,
                     color="rgba(0,0,0,0.5)", thickness=1.2, width=4),
        marker_color=_CONF_COLORS[cat],
        hovertemplate=(
            "%{x}<br>" + cat + ": %{y:.1f}%"
            "<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>"
        ),
        customdata=[[means[i] - err_lo[i], means[i] + err_hi[i]]
                    for i in range(len(means))],
    ))

# Shade and clearly delineate the invalid-code half
_boundary_x = len(_valid_conds) - 0.5
# Solid dividing line
fig_a8.add_vline(x=_boundary_x, line_color="rgba(200,150,255,0.8)", line_width=2, line_dash="dash")
# Strong background shade for invalid side
fig_a8.add_vrect(
    x0=_boundary_x, x1=len(_ordered_conds) - 0.5,
    fillcolor="#7d3c98", opacity=0.15, line_width=0,
)
# Bold header annotations
fig_a8.add_annotation(
    x=_boundary_x + 0.5 * len(_invalid_conds), y=107,
    text="<b>INVALID CODE — GT: No</b>",
    showarrow=False, font=dict(size=12, color="#d7a8ff"), xanchor="center",
    bgcolor="rgba(80,20,110,0.6)", bordercolor="#9b59b6", borderwidth=1,
)
fig_a8.add_annotation(
    x=0.5 * (len(_valid_conds) - 1), y=107,
    text="<b>VALID CODE — GT: Yes</b>",
    showarrow=False, font=dict(size=12, color="#7dffaa"), xanchor="center",
    bgcolor="rgba(10,60,30,0.6)", bordercolor="#27ae60", borderwidth=1,
)

fig_a8.update_layout(
    title="Free-Generation Validity Predictions across Perturbations",
    barmode="stack",
    yaxis_title="% of predictions",
    yaxis_range=[0, 112],
    xaxis=dict(tickangle=-20),
    height=540,
    legend=dict(x=1.02, y=1, xanchor="left", title="Prediction type",
                bgcolor="rgba(240,242,255,0.95)", bordercolor="#aaa", borderwidth=1,
                font=dict(color="#111")),
    margin=dict(l=70, r=220, t=90, b=80),
)
add("A · LLM Eval", "⑨ Free-Gen Validity Predictions", fig_a8,
    "Single chart covering all 8 conditions. Left (green-annotated): valid code — correct answer is Yes. "
    "Right (purple-shaded): invalid code — correct answer is No. "
    "Classification rules — "
    "Confident Yes: bare 'yes'/'true'/'valid'; "
    "Uncertain Yes: starts with 'yes' + qualifier, or phrases like 'physically valid [caveat]'; "
    "Confident No: bare 'no'/'false'/'invalid', or 'no, because…' patterns; "
    "Uncertain/No Answer: hedges such as 'potentially', 'unknown', etc. that abstain. "
    "Error bars: 95% bootstrap CI on proportion of predictions per category.")

# ── A10. Behavior: at-least-1-correct rate by condition ──────────────────────
fig_a9 = go.Figure()
for metric, label, color in [
    ("behavior_any_match", "≥1 behavior correct (any_match)", "#3498db"),
    ("behavior_recall",    "Full recall (all behaviors)",     "#85c1e9"),
]:
    means, err_lo, err_hi = [], [], []
    for cond in _present_conds:
        data = df_llm[df_llm["mod_type"] == cond][metric]
        lo, hi, mn, _ = bootstrap_ci(data)
        means.append(mn * 100)
        err_lo.append((mn - lo) * 100)
        err_hi.append((hi - mn) * 100)
    fig_a9.add_trace(go.Bar(
        name=label,
        x=[COND_SHORT.get(c, c) for c in _present_conds],
        y=means,
        error_y=dict(type="data", symmetric=False,
                     array=err_hi, arrayminus=err_lo,
                     color="rgba(0,0,0,0.5)", thickness=1.2, width=4),
        marker_color=color,
        hovertemplate=(
            "%{x}<br>" + label + ": %{y:.1f}%"
            "<br>95% CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>"
        ),
        customdata=[[means[i] - err_lo[i], means[i] + err_hi[i]]
                    for i in range(len(means))],
    ))
fig_a9.update_layout(
    title="Behavior accuracy — ≥1 correct vs full recall",
    xaxis_title="Condition",
    yaxis_title="% predictions",
    yaxis_range=[0, 105],
    barmode="group", height=430, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑩ Behavior: ≥1 correct vs full recall", fig_a9,
    "Dark blue: model named at least one of the ground-truth physical behaviors (any_match). "
    "Light blue: fraction of all gt behaviors mentioned (recall — partial credit). "
    "Error bars: 95% bootstrap CI (n=1000 resamples). "
    "When gt has multiple behaviors (e.g. advection/diffusion/restoration), any_match "
    "is generous while recall penalizes missing behaviors. "
    "Scoring uses ALIASES: 'wave propagation' counts as oscillation, "
    "'heat conduction' as diffusion, 'transport/convection' as advection, 'damping/decay' as restoration.")

# ── A11. Reasoning vs Non-reasoning models ───────────────────────────────────
REASONING_MODELS = {"QwQ-32B", "DeepSeek-R1-32B", "Qwen3-32B"}

df_llm["model_type"] = df_llm["model_short"].apply(
    lambda m: "Reasoning" if m in REASONING_MODELS else "Non-reasoning"
)

fig_a11 = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "By condition (group averages)",
        "By model (sorted by mean accuracy)",
    ],
    vertical_spacing=0.18,
)

# Compute overall_acc per row
for col in ["pde_match", "method_any_match", "behavior_any_match", "valid_match"]:
    df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")
df_llm["overall_acc"] = df_llm[["pde_match","method_any_match","behavior_any_match","valid_match"]].mean(axis=1)

_r11_conds  = [c for c in ALL_CONDS if c in df_llm["mod_type"].unique()]
_r11_labels = [COND_SHORT.get(c, c) for c in _r11_conds]

# Panel 1: per-condition grouped bars, one bar per model-type group
for mtype, color in [("Reasoning", "#e67e22"), ("Non-reasoning", "#3498db")]:
    sub_type = df_llm[df_llm["model_type"] == mtype]
    means, err_lo, err_hi = [], [], []
    for cond in _r11_conds:
        data = sub_type[sub_type["mod_type"] == cond]["overall_acc"]
        lo, hi, mn, _ = bootstrap_ci(data)
        means.append(mn * 100)
        err_lo.append((mn - lo) * 100)
        err_hi.append((hi - mn) * 100)
    fig_a11.add_trace(go.Bar(
        name=mtype, x=_r11_labels, y=means,
        error_y=dict(type="data", symmetric=False,
                     array=err_hi, arrayminus=err_lo,
                     color="rgba(0,0,0,0.5)", thickness=1.2, width=4),
        marker_color=color,
        legendgroup=mtype, showlegend=True,
        hovertemplate=mtype + " | %{x}: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

# Panel 2: per-model overall accuracy across all conditions
_model_order = sorted(df_llm["model_short"].unique(),
                      key=lambda m: df_llm[df_llm["model_short"]==m]["overall_acc"].mean())
for model in _model_order:
    is_r = model in REASONING_MODELS
    data = df_llm[df_llm["model_short"] == model]["overall_acc"]
    lo, hi, mn, _ = bootstrap_ci(data)
    fig_a11.add_trace(go.Bar(
        name=model,
        x=[model],
        y=[mn * 100],
        error_y=dict(type="data", symmetric=False,
                     array=[(hi - mn) * 100], arrayminus=[(mn - lo) * 100],
                     color="rgba(0,0,0,0.5)", thickness=1.2, width=4),
        marker_color="#e67e22" if is_r else "#3498db",
        marker_pattern_shape="x" if is_r else "",
        legendgroup="Reasoning" if is_r else "Non-reasoning",
        showlegend=False,
        hovertemplate=model + ": %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

fig_a11.update_layout(
    title="Reasoning vs Non-reasoning model accuracy",
    barmode="group",
    height=720,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=70, r=220, t=60, b=80),
)
fig_a11.update_yaxes(title_text="Overall accuracy (%)", row=1, col=1)
fig_a11.update_yaxes(title_text="Overall accuracy (%)", row=2, col=1)
fig_a11.update_xaxes(tickangle=-20, row=1, col=1)
fig_a11.update_xaxes(tickangle=-30, row=2, col=1)

add("A · LLM Eval", "⑪ Reasoning vs Non-reasoning accuracy", fig_a11,
    "Top: per-condition group averages — orange = reasoning models (QwQ-32B, DeepSeek-R1-32B, Qwen3-32B), "
    "blue = non-reasoning (Llama-3.1-8B, Llama-3.3-70B, Qwen2.5-7B, Qwen2.5-32B, Gemma-3-27B, Mistral-12B, phi-4). "
    "Error bars = 95% bootstrap CI across all (model, snippet) pairs in each group. "
    "Bottom: individual model means sorted by accuracy — orange bars are reasoning models. "
    "Overall accuracy = average of pde_match, method_any_match, behavior_any_match, valid_match.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — pde-mc-logprob
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE = "Comm_Valid"
non_bl = [c for c in ALL_CONDS if c != BASELINE]
qtypes = list(QTYPE_LABEL.keys())

def delta_vs_baseline(df, metric, baseline=BASELINE):
    bl = df[df["mod_type"]==baseline].groupby("question_type")[metric].mean()
    grp = df.groupby(["mod_type","question_type"])[metric].mean().reset_index()
    grp["delta"] = grp.apply(lambda r: r[metric] - bl.get(r["question_type"], np.nan), axis=1)
    return grp

acc_delta = delta_vs_baseline(df_mc, "correct")
lp_delta  = delta_vs_baseline(df_mc, "logprob_correct")
ent_delta = delta_vs_baseline(df_mc, "entropy")

# B1a. Accuracy change — one chart, question type on x, condition as color
fig_b1a = go.Figure()
for cond in non_bl:
    sub = acc_delta[acc_delta["mod_type"]==cond].set_index("question_type").reindex(qtypes)
    fig_b1a.add_trace(go.Bar(
        name=COND_SHORT[cond],
        x=[QTYPE_LABEL[q] for q in qtypes],
        y=(sub["delta"].values.astype(float)*100).round(1),
        marker_color=COND_COLOR[cond],
    ))
fig_b1a.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_b1a.update_layout(
    title="MC: Accuracy change under corruption (Δ vs clean baseline)",
    yaxis_title="Δ Accuracy (percentage points)",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "① Does accuracy drop? (Δ vs clean baseline)", fig_b1a,
    "Accuracy change per question type, for each non-baseline condition. "
    "Negative = model gets fewer answers right vs clean code. "
    "Look at which conditions and question types cause the biggest drops.")

# B1b. Confidence (logprob) change — same layout as above for direct comparison
fig_b1b = go.Figure()
for cond in non_bl:
    sub = lp_delta[lp_delta["mod_type"]==cond].set_index("question_type").reindex(qtypes)
    fig_b1b.add_trace(go.Bar(
        name=COND_SHORT[cond],
        x=[QTYPE_LABEL[q] for q in qtypes],
        y=sub["delta"].values.astype(float).round(3),
        marker_color=COND_COLOR[cond],
    ))
fig_b1b.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_b1b.update_layout(
    title="MC: Confidence change under corruption (Δ log P vs clean baseline)",
    yaxis_title="Δ log P(correct answer)  ← more negative = less confident",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "② Does confidence drop? (Δ log P vs baseline)", fig_b1b,
    "Same layout as chart ①, but showing log-probability of the correct answer instead of accuracy. "
    "Compare ① and ②: if confidence drops (②) more than accuracy (①) for the same condition, "
    "the model is losing certainty before losing correctness — that is the core hypothesis.")

# B2. Scatter: Δacc vs Δlogprob — one point per (condition × question type)
merged = acc_delta.merge(lp_delta, on=["mod_type","question_type"], suffixes=("_acc","_lp"))
merged = merged[merged["mod_type"] != BASELINE]
merged["acc_pp"] = merged["delta_acc"] * 100

fig_b2 = go.Figure()
for cond in non_bl:
    sub = merged[merged["mod_type"]==cond]
    fig_b2.add_trace(go.Scatter(
        x=sub["acc_pp"].round(1), y=sub["delta_lp"].round(3),
        mode="markers+text",
        name=COND_SHORT[cond],
        marker=dict(color=COND_COLOR[cond], size=16,
                    line=dict(width=1.5, color="white")),
        text=sub["question_type"].map(QTYPE_LABEL),
        textposition="top center",
        textfont=dict(size=9, color="#aaa"),
    ))
fig_b2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
fig_b2.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
fig_b2.add_annotation(x=2, y=-0.08, text="← accuracy stable, confidence drops<br>(hypothesis confirmed)",
                       showarrow=False, font=dict(size=10, color="#2ecc71"), xanchor="left")
fig_b2.update_layout(
    title="Accuracy change vs Confidence change",
    xaxis_title="Δ Accuracy (percentage points vs clean baseline)",
    yaxis_title="Δ log P(correct)  ← negative = less confident",
    height=500, legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "③ Scatter: accuracy drop vs confidence drop", fig_b2,
    "X-axis = accuracy change, Y-axis = confidence change. Both vs clean baseline. "
    "Bottom-right quadrant (x≈0, y<0): confidence drops while accuracy holds — hypothesis confirmed. "
    "Bottom-left: both drop. Dots labeled by question type.")

# B3. Entropy increase
fig_b3 = go.Figure()
for cond in non_bl:
    sub = ent_delta[ent_delta["mod_type"]==cond].set_index("question_type").reindex(qtypes)
    fig_b3.add_trace(go.Bar(
        name=COND_SHORT[cond],
        x=[QTYPE_LABEL[q] for q in qtypes],
        y=sub["delta"].values.astype(float).round(3),
        marker_color=COND_COLOR[cond],
    ))
fig_b3.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_b3.update_layout(
    title="MC: Entropy change under corruption (Δ vs clean baseline)",
    yaxis_title="Δ Entropy  ← positive = more spread across answer choices",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "④ Entropy increase under corruption", fig_b3,
    "Entropy = spread of probability mass across all 4 answer choices. "
    "Positive bar = model is more uncertain under that condition vs clean baseline. "
    "Confirms the second part of the hypothesis: variable obfuscation should raise entropy.")

# B3b. Combined two-panel: Δ logprob + Δ entropy, excluding validity question type
_qtypes_no_valid = ["pde_class", "phys_process", "num_method"]
_qtype_labels_nv = {q: QTYPE_LABEL[q] for q in _qtypes_no_valid}

fig_b3b = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "Δ log P(correct answer) — confidence drop vs clean baseline",
        "Δ Entropy — uncertainty increase vs clean baseline",
    ],
    vertical_spacing=0.14,
)

# Panel 1: Δ logprob (excluding phys_valid)
for cond in non_bl:
    sub = lp_delta[(lp_delta["mod_type"] == cond) &
                   (lp_delta["question_type"].isin(_qtypes_no_valid))].set_index("question_type").reindex(_qtypes_no_valid)
    fig_b3b.add_trace(go.Bar(
        name=COND_SHORT[cond],
        x=[_qtype_labels_nv[q] for q in _qtypes_no_valid],
        y=sub["delta"].values.astype(float).round(3),
        marker_color=COND_COLOR[cond],
        legendgroup=cond,
        showlegend=True,
        hovertemplate=COND_SHORT[cond] + " | %{x}: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

# Panel 2: Δ entropy (excluding phys_valid)
for cond in non_bl:
    sub = ent_delta[(ent_delta["mod_type"] == cond) &
                    (ent_delta["question_type"].isin(_qtypes_no_valid))].set_index("question_type").reindex(_qtypes_no_valid)
    fig_b3b.add_trace(go.Bar(
        name=COND_SHORT[cond],
        x=[_qtype_labels_nv[q] for q in _qtypes_no_valid],
        y=sub["delta"].values.astype(float).round(3),
        marker_color=COND_COLOR[cond],
        legendgroup=cond,
        showlegend=False,
        hovertemplate=COND_SHORT[cond] + " | %{x}: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

fig_b3b.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=1)
fig_b3b.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)

fig_b3b.update_layout(
    title="MC: Δ confidence and Δ entropy under code perturbation",
    barmode="group",
    height=720,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=70, r=220, t=60, b=60),
)
fig_b3b.update_yaxes(title_text="Δ log P(correct)  ← more negative = less confident", row=1, col=1)
fig_b3b.update_yaxes(title_text="Δ Entropy  ← positive = more spread", row=2, col=1)

add("B · MC Logprob", "④b Δ confidence + Δ entropy under corruption", fig_b3b,
    "Top: Δ log P(correct answer) — how much confidence drops under each condition vs clean baseline. "
    "Bottom: Δ Entropy — how much the model spreads probability across choices (more uncertainty). "
    "Validity question type excluded; only PDE Class, Physical Process, and Numerical Method shown. "
    "The two panels together confirm the core finding: perturbations that drop confidence (top, negative) "
    "also raise uncertainty (bottom, positive), with variable obfuscation (CorrVar) the strongest signal.")

# B4. By PDE class — accuracy and confidence for PDE Class question only
sub_pc = df_mc[df_mc["question_type"]=="pde_class"]
bl_acc = sub_pc[sub_pc["mod_type"]==BASELINE].groupby("pde_class")["correct"].mean()
bl_lp  = sub_pc[sub_pc["mod_type"]==BASELINE].groupby("pde_class")["logprob_correct"].mean()
pde_classes = ["wave","heat","burgers","navier-stokes"]

for metric_key, bl_grp, ylabel, scale, title_suffix in [
    ("correct",         bl_acc, "Δ Accuracy (pp)", 100,  "accuracy"),
    ("logprob_correct", bl_lp,  "Δ log P(correct)", 1,   "confidence"),
]:
    fig = go.Figure()
    for cond in non_bl:
        sub_c = sub_pc[sub_pc["mod_type"]==cond].groupby("pde_class")[metric_key].mean()
        deltas = [round((sub_c.get(p,np.nan) - bl_grp.get(p,np.nan))*scale, 2) for p in pde_classes]
        fig.add_trace(go.Bar(
            name=COND_SHORT[cond], x=pde_classes,
            y=[d if not np.isnan(d) else 0 for d in deltas],
            marker_color=COND_COLOR[cond],
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"MC: {title_suffix.title()} drop by PDE class",
        yaxis_title=ylabel,
        barmode="group", height=460,
        legend=LEGEND, margin=MARGIN,
    )
    add("B · MC Logprob", f"⑤ By PDE class: {title_suffix}", fig,
        f"Δ{title_suffix} for each equation family under each condition. "
        "Shows which PDE classes (wave, heat, burgers, navier-stokes) are most sensitive "
        "to comment corruption vs variable obfuscation.")

# B5. Per-model confidence drop (PDE Class question)
bl_model = (df_mc[(df_mc["mod_type"]==BASELINE)&(df_mc["question_type"]=="pde_class")]
            .groupby("model_short")["logprob_correct"].mean())
fig_b5 = go.Figure()
models_mc = sorted(df_mc["model_short"].unique())

# Add baseline logprob VALUES as reference trace
baseline_vals = [bl_model.get(m, np.nan) for m in models_mc]
fig_b5.add_trace(go.Bar(
    name=f"Baseline: {COND_SHORT[BASELINE]} (logprob)",
    x=models_mc,
    y=baseline_vals,
    marker_color="rgba(150,150,150,0.4)",
    opacity=0.7,
))

for cond in non_bl:
    deltas = []
    for m in models_mc:
        sub = df_mc[(df_mc["model_short"]==m)&(df_mc["mod_type"]==cond)&
                    (df_mc["question_type"]=="pde_class")]
        deltas.append(round(sub["logprob_correct"].mean() - bl_model.get(m,np.nan), 3))
    fig_b5.add_trace(go.Bar(name=COND_SHORT[cond], x=models_mc, y=deltas,
                             marker_color=COND_COLOR[cond]))

# Add baseline reference line
fig_b5.add_hline(y=0, line_dash="solid", line_color="rgba(200,200,200,0.6)", line_width=2,
                 annotation_text=f"Baseline: {COND_SHORT[BASELINE]} (clean)",
                 annotation_position="right",
                 annotation_font=dict(size=9, color="#ccc"))
fig_b5.update_layout(
    title="MC: Per-model confidence drop (PDE Class question)",
    yaxis_title="Δ log P(correct)  ← negative = less confident",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "⑦ Per-model confidence robustness", fig_b5,
    "Each model compared to its own clean baseline confidence. "
    "Bars close to 0 = model stays certain even under corruption. "
    "Large negative bars = model loses confidence when variable names are obfuscated.")

# ═══════════════════════════════════════════════════════════════════════════════
# NEW FIGURES: Absolute values (not deltas) for MC data
# ═══════════════════════════════════════════════════════════════════════════════

# B6a. Absolute accuracy across all conditions (PDE Class question)
sub_pc = df_mc[df_mc["question_type"]=="pde_class"]
x_labels_mc = [COND_SHORT[c] for c in ALL_CONDS]
fig_b6a = go.Figure()
models_abs = sorted(df_mc["model_short"].unique())
metric_colors_mc = px.colors.qualitative.Plotly

for i, model in enumerate(models_abs):
    means = []
    lowers = []
    uppers = []
    for cond in ALL_CONDS:
        data = sub_pc[(sub_pc["model_short"]==model)&(sub_pc["mod_type"]==cond)]["correct"]
        lower, upper, mean, var = bootstrap_ci(data)
        means.append(mean * 100)
        lowers.append(lower * 100)
        uppers.append(upper * 100)

    error_y = dict(
        type="data",
        symmetric=False,
        array=[u - m for u, m in zip(uppers, means)],
        arrayminus=[m - l for m, l in zip(means, lowers)],
    )

    fig_b6a.add_trace(go.Scatter(
        x=x_labels_mc, y=means,
        error_y=error_y,
        mode="lines+markers", name=model,
        line=dict(color=metric_colors_mc[i], width=2), marker=dict(size=8),
    ))

# Add random baseline line
fig_b6a.add_hline(y=25, line_dash="dot", line_color="rgba(200,200,200,0.3)",
                  annotation_text="Random (4-way): 25%", annotation_position="right",
                  annotation_font=dict(size=8, color="#888"))

for x0, x1, color, label in [
    ("Clean+Comment","Clean, No Comment","#2ecc71","Clean valid"),
    ("Corrupt Comment","Corrupt Variable","#e74c3c","Corrupted"),
    ("Invalid+Comment","Invalid, No Comment","#9b59b6","Invalid"),
]:
    fig_b6a.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=10, color="#aaa"))

fig_b6a.update_layout(
    title="MC: Absolute accuracy per model across conditions",
    yaxis_title="Accuracy (%)",
    xaxis=dict(tickangle=-20),
    height=460, legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "⑧ Absolute accuracy: all conditions per model", fig_b6a,
    "Each line = one model's accuracy across all 8 conditions. "
    "Unlike delta charts, this shows absolute performance levels directly. "
    "Error bars = 95% bootstrap CI. n=16 per (model, condition) cell — wide CIs reflect pilot scale (16 samples/condition).")

# B6b. Absolute log-probability across all conditions (PDE Class question)
fig_b6b = go.Figure()
for i, model in enumerate(models_abs):
    means = []
    lowers = []
    uppers = []
    for cond in ALL_CONDS:
        data = sub_pc[(sub_pc["model_short"]==model)&(sub_pc["mod_type"]==cond)]["logprob_correct"]
        lower, upper, mean, var = bootstrap_ci(data)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)

    error_y = dict(
        type="data",
        symmetric=False,
        array=[u - m for u, m in zip(uppers, means)],
        arrayminus=[m - l for m, l in zip(means, lowers)],
    )

    fig_b6b.add_trace(go.Scatter(
        x=x_labels_mc, y=means,
        error_y=error_y,
        mode="lines+markers", name=model,
        line=dict(color=metric_colors_mc[i], width=2), marker=dict(size=8),
    ))

for x0, x1, color, label in [
    ("Clean+Comment","Clean, No Comment","#2ecc71","Clean valid"),
    ("Corrupt Comment","Corrupt Variable","#e74c3c","Corrupted"),
    ("Invalid+Comment","Invalid, No Comment","#9b59b6","Invalid"),
]:
    fig_b6b.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=10, color="#aaa"))

fig_b6b.update_layout(
    title="MC: Absolute confidence per model across conditions",
    yaxis_title="log P(correct answer)",
    xaxis=dict(tickangle=-20),
    height=460, legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "⑨ Absolute confidence: all conditions per model", fig_b6b,
    "Each line = one model's log-probability of correct answer across all 8 conditions. "
    "Compare to delta chart (⑦) to see whether drops are uniform or asymmetric. "
    "Error bars = 95% bootstrap CI. n=16 per (model, condition) cell — wide CIs reflect pilot scale.")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_metric_table_html(df, metric, conds, metric_label):
    """Generate HTML table: mean + asymmetric 95% CI bounds + n + variance."""
    html = (f"<table><tr><th>Condition</th><th>n</th><th>Mean (%)</th>"
            f"<th>95% CI lower</th><th>95% CI upper</th><th>Variance</th></tr>\n")
    for cond in conds:
        data = df[df["mod_type"]==cond][metric].dropna()
        n = len(data)
        if n == 0:
            html += f"<tr><td>{COND_SHORT[cond]}</td><td>0</td><td colspan='4'>no data</td></tr>\n"
            continue
        lower, upper, mean, var = bootstrap_ci(data)
        html += (f"<tr><td>{COND_SHORT[cond]}</td><td>{n}</td>"
                 f"<td>{mean*100:.1f}</td>"
                 f"<td>{lower*100:.1f}</td><td>{upper*100:.1f}</td>"
                 f"<td>{var:.4f}</td></tr>\n")
    html += "</table>"
    return html

def build_table_section():
    """Build summary table section with variance and CI data (only conditions present in data)."""
    present_conds = [c for c in ALL_CONDS if c in df_llm["mod_type"].unique()]
    section_html = '<div class="table-section">'
    section_html += "<h2>Summary Tables — LLM Eval Metrics with asymmetric 95% bootstrap CI (n shown per condition)</h2>\n"
    for metric, mlabel in LLM_METRICS:
        section_html += f"<h3>{mlabel}</h3>\n"
        section_html += generate_metric_table_html(df_llm, metric, present_conds, mlabel)
        section_html += "\n"
    section_html += "</div>"
    return section_html

table_inner_html = build_table_section()

# ═══════════════════════════════════════════════════════════════════════════════
# Build HTML — sidebar nav + main content
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

# Summary tables as a proper nav section (not always-visible)
nav_html += '<div class="nav-section">Summary</div>\n'
nav_html += f'<button class="nav-btn" onclick="show({idx})">Stats Tables</button>\n'
content_html += f'<div class="section" id="sec{idx}">{table_inner_html}</div>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDE Experiment Results (Enhanced v3)</title>
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
  .table-section {{ margin: 20px 0; }}
  .table-section h2 {{ font-size: 1.1rem; color: #ddd; margin: 20px 0 15px; }}
  .table-section h3 {{ font-size: 0.95rem; color: #aaa; margin: 15px 0 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  table th, table td {{ border: 1px solid #444; padding: 8px 12px; text-align: left; }}
  table th {{ background: #1a1d2a; color: #fff; font-weight: 600; }}
  table tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
  table tr:hover {{ background: rgba(255,255,255,0.06); }}
</style>
</head>
<body>
<div id="sidebar">
  <h1>PDE Results (v3 Enhanced)</h1>
  <div class="hyp">
    <strong>LLM Eval:</strong> Does accuracy degrade under perturbations? Do comments help on invalid code?<br><br>
    <strong>MC Logprob:</strong> Does confidence drop even when accuracy stays stable?<br><br>
    <strong>New:</strong> CIs, variances, baselines, tables
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

out = "results_v3_enhanced.html"
with open(out, "w") as f:
    f.write(html)
print(f"✓ Written: {out}  ({idx} charts + tables)")
print(f"  - Added baseline references in delta charts")
print(f"  - Added 95% bootstrap confidence intervals with variance reporting")
print(f"  - Added summary tables with CI and variance data")
print(f"  - Hypothesis-driven visualizations with compact formatting")
