"""
Focused visualizations addressing:
1. pde-llm-eval: answer alias distribution (what did models actually say?)
2. pde-mc-logprob: confidence under corruption (logprob delta from clean baseline)
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
        df_llm[col] = df_llm[col].astype(float)

df_mc["correct"] = df_mc["correct"].astype(str).str.lower().map(
    {"true": True, "false": False, "1": True, "0": False}
)
for col in ["logprob_correct","margin","entropy"]:
    df_mc[col] = pd.to_numeric(df_mc[col], errors="coerce")

MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct":       "Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct":       "Llama-3.3-70B",
    "Qwen/QwQ-32B":                            "QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct":          "Qwen2.5-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct":         "Qwen2.5-32B",
    "Qwen/Qwen3-32B":                          "Qwen3-32B",
    "mistralai/Mistral-Nemo-Instruct-2407":    "Mistral-12B",
    "microsoft/phi-4":                         "phi-4",
    "google/gemma-3-27b-it":                   "Gemma-3-27B",
}
df_llm["model_short"] = df_llm["model"].map(MODEL_SHORT).fillna(df_llm["model"])
df_mc["model_short"]  = df_mc["model"].map(MODEL_SHORT).fillna(df_mc["model"])

CONDITION_ORDER  = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar", "Comm_InValid", "NoComm_InValid"]
CONDITION_LABELS = {
    "Comm_Valid":      "Clean + Comment",
    "NoComm_Valid":    "Clean, No Comment",
    "CorrComm":        "Corrupt Comment",
    "NoComm_CorrVar":  "Corrupt Variable",
    "Comm_InValid":    "Invalid + Comment",
    "NoComm_InValid":  "Invalid, No Comment",
}
CONDITION_COLOR = {
    "Comm_Valid":      "#2ecc71",
    "NoComm_Valid":    "#27ae60",
    "CorrComm":        "#e67e22",
    "NoComm_CorrVar":  "#e74c3c",
    "Comm_InValid":    "#9b59b6",
    "NoComm_InValid":  "#8e44ad",
}
QTYPE_LABELS = {
    "pde_class":    "PDE Class",
    "phys_process": "Physical Process",
    "num_method":   "Numerical Method",
    "phys_valid":   "Physical Validity",
}

figs = []  # (title, fig, description)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — pde-llm-eval: Answer alias explorer
# ═══════════════════════════════════════════════════════════════════════════════

def truncate(s, n=55):
    if pd.isna(s): return "(no answer)"
    s = str(s).strip().replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s

# 1A. For each gt_pde: horizontal bar showing answer counts, colored by match
pde_types = sorted(df_llm["gt_pde"].dropna().unique())
fig1a = make_subplots(
    rows=len(pde_types), cols=1,
    subplot_titles=[f"gt: {p}" for p in pde_types],
    vertical_spacing=0.04,
)
for row_i, pde in enumerate(pde_types, 1):
    sub = df_llm[df_llm["gt_pde"] == pde].copy()
    sub["answer_trunc"] = sub["parsed_pde"].apply(truncate)
    grp = sub.groupby(["answer_trunc","pde_match"]).size().reset_index(name="count")
    grp = grp.sort_values("count", ascending=True)
    for match_val, color, label in [(1, "#2ecc71", "Correct"), (0, "#e74c3c", "Incorrect")]:
        g = grp[grp["pde_match"] == match_val]
        fig1a.add_trace(go.Bar(
            x=g["count"], y=g["answer_trunc"],
            orientation="h",
            name=label,
            marker_color=color,
            showlegend=(row_i == 1),
            legendgroup=label,
        ), row=row_i, col=1)

fig1a.update_layout(
    title="PDE Type Answers: What Did Models Actually Say? (green=matched, red=not matched)",
    barmode="stack",
    height=max(600, len(pde_types) * 180),
    legend=dict(orientation="h", y=1.02),
)
fig1a.update_xaxes(title_text="Count")
figs.append(("PDE Answers: Alias Distribution", fig1a,
    "Each bar shows a unique answer string the model gave. Green = scored correct, "
    "Red = scored incorrect. Visually reveals aliases being penalized — e.g. 'heat equation' vs '1D Heat Equation'."))

# 1B. Method answers: same view but for method_any_match
method_types = sorted(df_llm["gt_method"].dropna().astype(str).unique())[:8]  # top 8
fig1b = make_subplots(
    rows=len(method_types), cols=1,
    subplot_titles=[f"gt: {m}" for m in method_types],
    vertical_spacing=0.05,
)
for row_i, method in enumerate(method_types, 1):
    sub = df_llm[df_llm["gt_method"].astype(str) == method].copy()
    sub["answer_trunc"] = sub["parsed_method"].apply(truncate)
    grp = sub.groupby(["answer_trunc","method_any_match"]).size().reset_index(name="count")
    grp = grp.sort_values("count", ascending=True)
    for match_val, color, label in [(1, "#2ecc71", "Correct"), (0, "#e74c3c", "Incorrect")]:
        g = grp[grp["method_any_match"] == match_val]
        fig1b.add_trace(go.Bar(
            x=g["count"], y=g["answer_trunc"],
            orientation="h",
            name=label,
            marker_color=color,
            showlegend=(row_i == 1),
            legendgroup=label,
        ), row=row_i, col=1)
fig1b.update_layout(
    title="Method Answers: What Did Models Actually Say?",
    barmode="stack",
    height=max(600, len(method_types) * 160),
    legend=dict(orientation="h", y=1.02),
)
figs.append(("Method Answers: Alias Distribution", fig1b,
    "Same view for numerical methods. Reveals e.g. 'finite difference, explicit Euler' being matched to 'explicit'."))

# 1C. Match rate by model × pde_type — how consistent is each model?
pivot_pde = df_llm.pivot_table(values="pde_match", index="model_short",
                                columns="gt_pde", aggfunc="mean")
fig1c = go.Figure(go.Heatmap(
    z=(pivot_pde.values.astype(float) * 100).round(0),
    x=pivot_pde.columns.tolist(),
    y=pivot_pde.index.tolist(),
    colorscale="RdYlGn", zmin=0, zmax=100,
    text=(pivot_pde.values.astype(float) * 100).round(0),
    texttemplate="%{text:.0f}%",
    colorbar=dict(title="Match %"),
))
fig1c.update_layout(
    title="PDE Match Rate by Model × PDE Type (note: includes alias penalties)",
    height=350,
)
figs.append(("PDE Match by Model × Type", fig1c,
    "Match rates per model per PDE class. Low scores may reflect alias issues, not model failures."))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — pde-mc-logprob: Confidence under corruption
# ═══════════════════════════════════════════════════════════════════════════════

# 2A. The core story: Δlogprob from clean baseline (Comm_Valid) per condition × question type
baseline = df_mc[df_mc["mod_type"] == "Comm_Valid"].groupby("question_type")["logprob_correct"].mean()
cond_means = df_mc.groupby(["mod_type","question_type"])["logprob_correct"].mean().reset_index()
cond_means["delta"] = cond_means.apply(
    lambda r: r["logprob_correct"] - baseline.get(r["question_type"], np.nan), axis=1
)
cond_means["cond_label"] = cond_means["mod_type"].map(CONDITION_LABELS)
cond_means["qtype_label"] = cond_means["question_type"].map(QTYPE_LABELS)

fig2a = go.Figure()
for cond in CONDITION_ORDER[1:]:  # skip baseline itself
    sub = cond_means[cond_means["mod_type"] == cond].sort_values("question_type")
    fig2a.add_trace(go.Bar(
        name=CONDITION_LABELS[cond],
        x=sub["qtype_label"],
        y=sub["delta"].round(3),
        marker_color=CONDITION_COLOR[cond],
    ))
fig2a.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4,
                annotation_text="Clean baseline (Comment + Valid)", annotation_position="top right")
fig2a.update_layout(
    title="Confidence Change vs. Clean Baseline — Δ log P(correct answer)",
    yaxis_title="Δ log P (negative = less confident than baseline)",
    barmode="group",
    height=480,
    legend_title="Condition",
)
figs.append(("Confidence Drop vs Clean Baseline", fig2a,
    "Bars show how much model confidence (log-probability of the correct MC answer) changes relative to "
    "the clean baseline (code with correct comment, valid simulation). "
    "Negative = model is less certain under that condition. "
    "NoComm_CorrVar (variable obfuscation) shows the largest confidence drop for PDE Class."))

# 2B. Per-model confidence story for each question type — line plot across conditions
for qtype in ["pde_class", "phys_valid"]:  # the two most informative
    fig = go.Figure()
    models = sorted(df_mc["model_short"].unique())
    colors = px.colors.qualitative.Plotly
    for i, model in enumerate(models):
        sub = df_mc[(df_mc["model_short"] == model) & (df_mc["question_type"] == qtype)]
        grp = sub.groupby("mod_type")["logprob_correct"].mean().reindex(CONDITION_ORDER)
        fig.add_trace(go.Scatter(
            x=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
            y=grp.values.round(3),
            mode="lines+markers",
            name=model,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
        ))
    # shade corrupted zone
    fig.add_vrect(x0="Corrupt Comment", x1="Corrupt Variable",
                  fillcolor="red", opacity=0.07, line_width=0,
                  annotation_text="Corruption zone", annotation_position="top left")
    fig.update_layout(
        title=f"Confidence (log P correct) by Condition — {QTYPE_LABELS[qtype]}",
        yaxis_title="Mean log P(correct answer)",
        xaxis_title="Condition (left = clean, right = invalid/corrupted)",
        height=420,
        legend_title="Model",
    )
    figs.append((f"Confidence by Condition: {QTYPE_LABELS[qtype]}", fig,
        f"Each line traces one model's confidence across conditions for '{QTYPE_LABELS[qtype]}'. "
        "The shaded region marks corrupted conditions. A dip there means the model is less sure "
        "of its answer when the code is obfuscated or comments are misleading."))

# 2C. Accuracy + confidence side-by-side per condition (dual-axis)
for qtype in ["pde_class", "phys_valid"]:
    acc_grp   = df_mc[df_mc["question_type"]==qtype].groupby("mod_type")["correct"].mean().reindex(CONDITION_ORDER)
    lp_grp    = df_mc[df_mc["question_type"]==qtype].groupby("mod_type")["logprob_correct"].mean().reindex(CONDITION_ORDER)
    cond_lbls = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    colors_bar = [CONDITION_COLOR[c] for c in CONDITION_ORDER]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=cond_lbls, y=np.round(acc_grp.values.astype(float) * 100, 1),
        name="Accuracy (%)",
        marker_color=colors_bar, opacity=0.75,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cond_lbls, y=lp_grp.values.round(3),
        name="Mean log P(correct)",
        mode="lines+markers",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=9, color="white"),
    ), secondary_y=True)
    fig.update_layout(
        title=f"Accuracy & Confidence by Condition — {QTYPE_LABELS[qtype]}",
        height=420,
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="Accuracy (%)", range=[0,110], secondary_y=False)
    fig.update_yaxes(title_text="log P(correct)", secondary_y=True)
    figs.append((f"Accuracy + Confidence: {QTYPE_LABELS[qtype]}", fig,
        "Bars = accuracy (left axis). Dotted line = log-probability of the correct token (right axis, more negative = less confident). "
        "When the line drops further than the bars, the model is losing confidence before losing accuracy — a leading signal."))

# 2D. Confidence drop: per-model breakdown for pde_class (the most dramatic)
baseline_per_model = (
    df_mc[(df_mc["question_type"]=="pde_class") & (df_mc["mod_type"]=="Comm_Valid")]
    .groupby("model_short")["logprob_correct"].mean()
)
fig2d = go.Figure()
models = sorted(df_mc["model_short"].unique())
colors_list = px.colors.qualitative.Plotly
for i, cond in enumerate(CONDITION_ORDER[1:]):
    deltas, model_labels = [], []
    for m in models:
        sub = df_mc[(df_mc["model_short"]==m) & (df_mc["question_type"]=="pde_class") & (df_mc["mod_type"]==cond)]
        bl  = baseline_per_model.get(m, np.nan)
        delta = sub["logprob_correct"].mean() - bl
        deltas.append(round(delta, 3))
        model_labels.append(m)
    fig2d.add_trace(go.Bar(
        name=CONDITION_LABELS[cond],
        x=model_labels, y=deltas,
        marker_color=CONDITION_COLOR[cond],
    ))
fig2d.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
fig2d.update_layout(
    title="PDE Class — Per-Model Confidence Drop vs Clean Baseline",
    yaxis_title="Δ log P(correct) vs clean baseline",
    barmode="group",
    height=450,
)
figs.append(("Per-Model Confidence Drop: PDE Class", fig2d,
    "How much each model's confidence in its PDE class answer drops under each condition, "
    "relative to its own clean baseline. Reveals whether some models are more robust to obfuscation."))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2B — MC Logprob: breakdown by PDE class (wave/heat/burgers/navier-stokes)
# ═══════════════════════════════════════════════════════════════════════════════

PDE_CLASSES = ["wave", "heat", "burgers", "navier-stokes"]
PDE_COLORS  = {"wave": "#3498db", "heat": "#e74c3c", "burgers": "#f39c12", "navier-stokes": "#9b59b6"}

# 2E. Δlogprob from baseline by PDE class × condition (for pde_class question type)
sub_pde = df_mc[df_mc["question_type"] == "pde_class"].copy()
bl_pde  = sub_pde[sub_pde["mod_type"] == "Comm_Valid"].groupby("pde_class")["logprob_correct"].mean()
grp_pde = sub_pde.groupby(["pde_class", "mod_type"])["logprob_correct"].mean().reset_index()
grp_pde["delta"] = grp_pde.apply(lambda r: r["logprob_correct"] - bl_pde.get(r["pde_class"], np.nan), axis=1)

fig2e = go.Figure()
for pde in PDE_CLASSES:
    sub = grp_pde[grp_pde["pde_class"] == pde].set_index("mod_type").reindex(CONDITION_ORDER).reset_index()
    fig2e.add_trace(go.Bar(
        name=pde,
        x=[CONDITION_LABELS[c] for c in CONDITION_ORDER[1:]],
        y=sub[sub["mod_type"].isin(CONDITION_ORDER[1:])].set_index("mod_type").reindex(CONDITION_ORDER[1:])["delta"].round(3).values,
        marker_color=PDE_COLORS[pde],
    ))
fig2e.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
fig2e.update_layout(
    title="Confidence Drop by PDE Class — Δ log P(correct) vs Clean Baseline (PDE Class question)",
    yaxis_title="Δ log P(correct answer)",
    barmode="group",
    height=460,
    legend_title="PDE Class",
)
figs.append(("Confidence Drop by PDE Class", fig2e,
    "Each group of bars is one condition (non-baseline). Colors = PDE class. "
    "Reveals which equation families are most sensitive to code corruption. "
    "Wave equations and navier-stokes show the largest confidence drops under variable obfuscation."))

# 2F. Line plot: confidence across conditions, one line per PDE class
fig2f = go.Figure()
for pde in PDE_CLASSES:
    sub = sub_pde[sub_pde["pde_class"] == pde].groupby("mod_type")["logprob_correct"].mean().reindex(CONDITION_ORDER)
    fig2f.add_trace(go.Scatter(
        x=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
        y=sub.values.round(3),
        mode="lines+markers",
        name=pde,
        line=dict(color=PDE_COLORS[pde], width=2),
        marker=dict(size=9),
    ))
fig2f.add_vrect(x0="Corrupt Comment", x1="Corrupt Variable",
                fillcolor="red", opacity=0.07, line_width=0,
                annotation_text="Corruption zone", annotation_position="top left")
fig2f.update_layout(
    title="Confidence by Condition & PDE Class (PDE Class question)",
    yaxis_title="Mean log P(correct answer)",
    height=430,
    legend_title="PDE Class",
)
figs.append(("Confidence Traces by PDE Class", fig2f,
    "Each line traces one PDE class's mean model confidence across conditions. "
    "The shaded zone = corrupted conditions. Steeper dips = that PDE class is more "
    "recognizable from structure than from comments/variable names."))

# 2G. Heatmap: pde_class × condition (absolute logprob, pde_class question)
pivot_pde_cond = sub_pde.groupby(["pde_class", "mod_type"])["logprob_correct"].mean().unstack()
pivot_pde_cond = pivot_pde_cond.reindex(columns=CONDITION_ORDER)
pivot_pde_cond.columns = [CONDITION_LABELS[c] for c in CONDITION_ORDER]

fig2g = go.Figure(go.Heatmap(
    z=pivot_pde_cond.values.astype(float).round(3),
    x=pivot_pde_cond.columns.tolist(),
    y=pivot_pde_cond.index.tolist(),
    colorscale="RdYlGn",
    text=pivot_pde_cond.values.astype(float).round(2),
    texttemplate="%{text:.2f}",
    colorbar=dict(title="log P"),
))
fig2g.update_layout(
    title="Mean log P(correct) by PDE Class × Condition (PDE Class question)",
    height=340,
)
figs.append(("Confidence Heatmap: PDE Class × Condition", fig2g,
    "Absolute log-probability (more negative = less confident). "
    "Shows both the baseline confidence (how easy each PDE class is to identify) "
    "and how corruption affects each class differently."))

# 2H. Accuracy by PDE class × condition (pde_class question)
pivot_acc = sub_pde.groupby(["pde_class", "mod_type"])["correct"].mean().unstack()
pivot_acc = pivot_acc.reindex(columns=CONDITION_ORDER)
pivot_acc.columns = [CONDITION_LABELS[c] for c in CONDITION_ORDER]

fig2h = go.Figure(go.Heatmap(
    z=(pivot_acc.values.astype(float) * 100).round(0),
    x=pivot_acc.columns.tolist(),
    y=pivot_acc.index.tolist(),
    colorscale="RdYlGn",
    zmin=0, zmax=100,
    text=(pivot_acc.values.astype(float) * 100).round(0),
    texttemplate="%{text:.0f}%",
    colorbar=dict(title="Acc %"),
))
fig2h.update_layout(
    title="MC Accuracy by PDE Class × Condition (PDE Class question)",
    height=340,
)
figs.append(("Accuracy Heatmap: PDE Class × Condition", fig2h,
    "Accuracy companion to the confidence heatmap above. "
    "Compare the two: where confidence drops faster than accuracy, the model is losing certainty before losing correctness."))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — pde-llm-eval: Partial credit scoring (method_recall, behavior_recall)
# ═══════════════════════════════════════════════════════════════════════════════

# 3A. Partial credit distribution — how often do models get 0, 0.5, 1.0 on method?
for metric, label in [("method_recall", "Method"), ("behavior_recall", "Behavior")]:
    df_llm[metric] = df_llm[metric].astype(float)
    credit_counts = df_llm.groupby(["model_short", metric]).size().reset_index(name="count")
    fig = go.Figure()
    credit_vals = sorted(credit_counts[metric].dropna().unique())
    colors_credit = {0.0: "#e74c3c", 0.5: "#f39c12", 1.0: "#2ecc71"}
    for val in credit_vals:
        sub = credit_counts[credit_counts[metric] == val]
        sub = sub.set_index("model_short").reindex(sorted(df_llm["model_short"].unique())).reset_index()
        color = colors_credit.get(round(val, 1), "#95a5a6")
        fig.add_trace(go.Bar(
            name=f"{val:.1f} credit",
            x=sub["model_short"],
            y=sub["count"].fillna(0),
            marker_color=color,
        ))
    fig.update_layout(
        title=f"{label} Recall — Credit Distribution by Model (0 = miss, 0.5 = partial, 1.0 = full)",
        barmode="stack",
        yaxis_title="Count",
        height=420,
        legend_title="Credit",
    )
    figs.append((f"Partial Credit: {label}", fig,
        f"Stacked bars show how many answers got no credit (red), half credit (orange, one of two methods mentioned), "
        f"or full credit (green). Half-credit cases are where gt has two required {label.lower()}s and model named exactly one."))

# 3B. Mean recall score by model (vs binary any_match) — shows impact of partial credit
fig3b = go.Figure()
models_llm = sorted(df_llm["model_short"].unique())
metrics_compare = [
    ("method_any_match", "method_recall", "Method"),
    ("behavior_any_match", "behavior_recall", "Behavior"),
]
for metric_bin, metric_recall, label in metrics_compare:
    df_llm[metric_bin] = df_llm[metric_bin].astype(float)
    bin_means    = df_llm.groupby("model_short")[metric_bin].mean().reindex(models_llm)
    recall_means = df_llm.groupby("model_short")[metric_recall].mean().reindex(models_llm)
    fig3b.add_trace(go.Bar(
        name=f"{label} (binary any-match)",
        x=models_llm,
        y=(bin_means.values.astype(float) * 100).round(1),
        opacity=0.5,
        marker_pattern_shape="/",
    ))
    fig3b.add_trace(go.Bar(
        name=f"{label} (partial credit recall)",
        x=models_llm,
        y=(recall_means.values.astype(float) * 100).round(1),
    ))
fig3b.update_layout(
    title="Method & Behavior: Binary Score vs Partial Credit Recall",
    barmode="group",
    yaxis=dict(title="Score (%)", range=[0, 110]),
    height=450,
    legend_title="Metric",
)
figs.append(("Binary vs Partial Credit", fig3b,
    "Hatched bars = old binary scoring (any match = 1, else 0). "
    "Solid bars = new partial credit recall (half credit for naming one of two required methods/behaviors). "
    "Difference shows how much the binary scorer over- or under-rewards models."))

# 3C. Partial credit recall by model × condition (method)
df_llm["mod_label"] = df_llm["mod_type"].map(CONDITION_LABELS).fillna(df_llm["mod_type"])
pivot_recall = df_llm.pivot_table(values="method_recall", index="model_short",
                                   columns="mod_label", aggfunc="mean")
col_order = [CONDITION_LABELS[m] for m in CONDITION_ORDER if CONDITION_LABELS[m] in pivot_recall.columns]
pivot_recall = pivot_recall.reindex(columns=col_order)
fig3c = go.Figure(go.Heatmap(
    z=(pivot_recall.values.astype(float) * 100).round(0),
    x=pivot_recall.columns.tolist(),
    y=pivot_recall.index.tolist(),
    colorscale="RdYlGn", zmin=0, zmax=100,
    text=(pivot_recall.values.astype(float) * 100).round(0),
    texttemplate="%{text:.0f}%",
    colorbar=dict(title="Recall %"),
))
fig3c.update_layout(
    title="Method Partial Credit Recall by Model × Condition",
    height=350,
)
figs.append(("Method Recall Heatmap", fig3c,
    "Using partial credit recall (not binary). Each cell = mean recall score across samples. "
    "100% = model named all required methods. 50% = named half. 0% = named none."))

# ═══════════════════════════════════════════════════════════════════════════════
# Build HTML
# ═══════════════════════════════════════════════════════════════════════════════
nav_items = ""
sections_html = ""

for i, item in enumerate(figs):
    title, fig = item[0], item[1]
    desc = item[2] if len(item) > 2 else ""
    fig_html = fig.to_html(full_html=False, include_plotlyjs=(i == 0), div_id=f"fig{i}")
    active = "active" if i == 0 else ""
    section_label = "🔵 LLM Eval" if "LLM" in title or "PDE Answers" in title or "Method Answers" in title or "PDE Match" in title else "🟠 MC Logprob"
    nav_items += f'<button class="nav-btn {active}" onclick="show({i})" title="{title}">{title}</button>\n'
    desc_html = f'<p class="desc">{desc}</p>' if desc else ""
    sections_html += f'<div class="section {active}" id="sec{i}"><h2>{title}</h2>{desc_html}{fig_html}</div>\n'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDE Results — Alias & Confidence Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f1117; color: #e0e0e0; margin: 0; }}
  header {{ background: #1a1d27; padding: 18px 28px; border-bottom: 1px solid #2a2d3a; }}
  header h1 {{ margin: 0; font-size: 1.3rem; color: #fff; }}
  header p  {{ margin: 4px 0 0; font-size: 0.82rem; color: #888; }}
  #nav {{ display: flex; flex-wrap: wrap; gap: 5px; padding: 12px 28px;
          background: #13151f; border-bottom: 1px solid #2a2d3a; }}
  .nav-btn {{ background: #1e2130; border: 1px solid #2a2d3a; color: #aaa;
              padding: 5px 11px; border-radius: 6px; cursor: pointer; font-size: 0.75rem; text-align:left; }}
  .nav-btn:hover {{ background: #252840; color: #fff; }}
  .nav-btn.active {{ background: #3a5ccc; border-color: #3a5ccc; color: #fff; }}
  .section {{ display: none; padding: 20px 28px; }}
  .section.active {{ display: block; }}
  .section h2 {{ font-size: 1rem; color: #aaa; margin: 0 0 6px; font-weight: 500; }}
  .desc {{ font-size: 0.82rem; color: #777; margin: 0 0 14px; max-width: 860px;
           border-left: 3px solid #2a2d3a; padding-left: 10px; line-height: 1.5; }}
</style>
</head>
<body>
<header>
  <h1>PDE Experiment — Alias &amp; Confidence Analysis</h1>
  <p>
    <strong>Left section</strong>: pde-llm-eval answer distribution — see actual model answers &amp; alias penalties &nbsp;·&nbsp;
    <strong>Right section</strong>: pde-mc-logprob — confidence (log P) under code corruption
  </p>
</header>
<div id="nav">{nav_items}</div>
{sections_html}
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

out = "results_alias_confidence.html"
with open(out, "w") as f:
    f.write(html)
print(f"Written: {out}  ({len(figs)} charts)")
