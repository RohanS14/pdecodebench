"""
Hypothesis-driven visualizations — clean layout, no legend overlap.

pde-llm-eval: accuracy degrades under perturbations; comments help on invalid code.
pde-mc-logprob: logprob drops even when accuracy is stable; variable obfuscation strongest.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Load ──────────────────────────────────────────────────────────────────────
df_llm = pd.read_csv("../results/pde_llm_eval.csv")
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
}
df_llm["model_short"] = df_llm["model"].map(MODEL_SHORT).fillna(df_llm["model"])
df_mc["model_short"]  = df_mc["model"].map(MODEL_SHORT).fillna(df_mc["model"])

ALL_CONDS = ["Comm_Valid","NoComm_Valid","CorrComm","NoComm_CorrVar","Comm_InValid","NoComm_InValid"]
COND_SHORT = {
    "Comm_Valid":      "Clean+Comment",
    "NoComm_Valid":    "Clean, No Comment",
    "CorrComm":        "Corrupt Comment",
    "NoComm_CorrVar":  "Corrupt Variable",
    "Comm_InValid":    "Invalid+Comment",
    "NoComm_InValid":  "Invalid, No Comment",
}
COND_COLOR = {
    "Comm_Valid":      "#2ecc71",
    "NoComm_Valid":    "#27ae60",
    "CorrComm":        "#e67e22",
    "NoComm_CorrVar":  "#e74c3c",
    "Comm_InValid":    "#9b59b6",
    "NoComm_InValid":  "#7d3c98",
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

LLM_METRICS = [
    ("pde_match",       "PDE Type  [0 or 100%]"),
    ("method_recall",   "Method  [0, 50, or 100%]"),
    ("behavior_recall", "Behavior  [0, 50, or 100%]"),
    ("valid_match",     "Validity  [0 or 100%]"),
]

figs = []
def add(section, title, fig, question):
    figs.append((section, title, fig, question))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — pde-llm-eval
# ═══════════════════════════════════════════════════════════════════════════════

# A1. Four perturbation comparisons — one grouped bar per metric
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

fig_a1.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_a1.update_layout(
    title="LLM Eval: Accuracy change for each perturbation (vs appropriate baseline)",
    yaxis_title="Δ Accuracy (percentage points)",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "① Effect of each perturbation on accuracy", fig_a1,
    "Each group of bars = one evaluation metric. Colors = 4 perturbation types, each compared to its natural reference. "
    "③ Corrupt variable names compares NoComm_CorrVar vs NoComm_Valid — only variable names differ. "
    "④ Shows whether adding a comment helps when the code is physically invalid.")

# A2. Accuracy across all conditions — one line per metric
fig_a2 = go.Figure()
x_labels = [COND_SHORT[c] for c in ALL_CONDS]
metric_colors = px.colors.qualitative.Plotly
for i, (metric, mlabel) in enumerate(LLM_METRICS):
    vals = df_llm.groupby("mod_type")[metric].mean().reindex(ALL_CONDS)
    fig_a2.add_trace(go.Scatter(
        x=x_labels, y=(vals.values.astype(float)*100).round(1),
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
    title="LLM Eval: Accuracy across all 6 conditions",
    yaxis_title="Score (%)",
    xaxis=dict(tickangle=-20),
    height=460, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "② Accuracy profile across all conditions", fig_a2,
    "Each line traces one metric across all 6 conditions. Green zone = clean valid code. "
    "Red zone = corrupted conditions. Purple zone = physically invalid. "
    "Method and behavior use partial credit (0.5 for one of two required terms).")

# A3. Per-model: one chart per metric, models as lines
for metric, mlabel in LLM_METRICS:
    fig = go.Figure()
    for i, model in enumerate(sorted(df_llm["model_short"].unique())):
        vals = df_llm[df_llm["model_short"]==model].groupby("mod_type")[metric].mean().reindex(ALL_CONDS)
        fig.add_trace(go.Scatter(
            x=x_labels, y=(vals.values.astype(float)*100).round(1),
            mode="lines+markers", name=model,
            line=dict(color=metric_colors[i], width=2), marker=dict(size=8),
        ))
    for x0, x1, color in [
        ("Corrupt Comment","Corrupt Variable","#e74c3c"),
        ("Invalid+Comment","Invalid, No Comment","#9b59b6"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0)
    fig.update_layout(
        title=f"LLM Eval: {mlabel} — per-model robustness",
        yaxis_title="Score (%)", xaxis=dict(tickangle=-20),
        height=440, legend=LEGEND, margin=MARGIN,
    )
    add("A · LLM Eval", f"③ Per-model: {mlabel}", fig,
        f"Each line = one model's {mlabel.lower()} score across conditions. "
        "Flatter lines = more robust. Steeper drops into the shaded zones = more sensitive to corruption.")

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
    title="LLM Eval: What did models actually say? (alias inspection)",
    barmode="stack", height=max(750, len(pde_types)*210),
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=400, r=160, t=80, b=40),
)
add("A · LLM Eval", "④ Answer distribution — alias inspection", fig_a4,
    "Green = scored correct, Red = scored wrong. Long red bars reveal aliases being penalized — "
    'e.g. "Burgers\' equation" and "inviscid Burgers\' equation" both marked wrong when gt is "burgers".')

# A5. Per-PDE-class accuracy across conditions
PDE_CLASSES = ["wave", "heat", "burgers", "navier-stokes"]

fig_a5a = go.Figure()
for cond in ALL_CONDS:
    vals = [df_llm[(df_llm["pde_class"]==p)&(df_llm["mod_type"]==cond)]["pde_match"].mean()*100
            for p in PDE_CLASSES]
    fig_a5a.add_trace(go.Bar(
        name=COND_SHORT[cond], x=PDE_CLASSES,
        y=[round(v, 1) for v in vals],
        marker_color=COND_COLOR[cond],
    ))
fig_a5a.update_layout(
    title="LLM Eval: PDE identification accuracy by equation family and condition",
    yaxis_title="PDE match (%)", yaxis_range=[0, 110],
    barmode="group", height=460, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑤ PDE accuracy by equation family", fig_a5a,
    "Each group = one PDE class. Colors = conditions. "
    "Navier-Stokes is identified near-perfectly in all conditions except CorrVar. "
    "Burgers collapses most severely under variable obfuscation (NoComm_CorrVar). "
    "Shows which equation families the models understand most robustly.")

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
    title="LLM Eval: Physical invalidity detection by equation family",
    yaxis_title="Valid match (% correct, should answer 'No')", yaxis_range=[0, 110],
    barmode="group", height=420, legend=LEGEND, margin=MARGIN,
)
add("A · LLM Eval", "⑥ Invalidity detection by equation family", fig_a5b,
    "Only invalid conditions shown. Correct answer is 'No' (not physically valid). "
    "Navier-Stokes invalidity is almost never detected (5.6%). "
    "Burgers invalidity detected most often (~47%). "
    "Shows which PDE families' physics models understand well enough to flag bugs.")

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
    title="MC Logprob: Does accuracy drop under corruption? (Δ vs clean baseline)",
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
    title="MC Logprob: Does confidence drop under corruption? (Δ log P vs clean baseline)",
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
    title="Scatter: Accuracy change vs Confidence change (each dot = one condition × question type)",
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
    title="MC Logprob: Does the model become more uncertain under corruption? (Δ Entropy)",
    yaxis_title="Δ Entropy  ← positive = more spread across answer choices",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "④ Entropy increase under corruption", fig_b3,
    "Entropy = spread of probability mass across all 4 answer choices. "
    "Positive bar = model is more uncertain under that condition vs clean baseline. "
    "Confirms the second part of the hypothesis: variable obfuscation should raise entropy.")

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
        title=f"MC Logprob: {title_suffix.title()} drop by PDE class (PDE Class question)",
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
for cond in non_bl:
    deltas = []
    for m in models_mc:
        sub = df_mc[(df_mc["model_short"]==m)&(df_mc["mod_type"]==cond)&
                    (df_mc["question_type"]=="pde_class")]
        deltas.append(round(sub["logprob_correct"].mean() - bl_model.get(m,np.nan), 3))
    fig_b5.add_trace(go.Bar(name=COND_SHORT[cond], x=models_mc, y=deltas,
                             marker_color=COND_COLOR[cond]))
fig_b5.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_b5.update_layout(
    title="MC Logprob: Per-model confidence drop (PDE Class question, vs own clean baseline)",
    yaxis_title="Δ log P(correct)  ← negative = less confident",
    barmode="group", height=460,
    legend=LEGEND, margin=MARGIN,
)
add("B · MC Logprob", "⑦ Per-model confidence robustness", fig_b5,
    "Each model compared to its own clean baseline confidence. "
    "Bars close to 0 = model stays certain even under corruption. "
    "Large negative bars = model loses confidence when variable names are obfuscated.")

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

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDE Experiment Results</title>
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
  <h1>PDE Experiment Results</h1>
  <div class="hyp">
    <strong>LLM Eval:</strong> Does accuracy degrade under perturbations? Do comments help on invalid code?<br><br>
    <strong>MC Logprob:</strong> Does confidence drop even when accuracy stays stable?
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

out = "results_v3.html"
with open(out, "w") as f:
    f.write(html)
print(f"Written: {out}  ({idx} charts)")
