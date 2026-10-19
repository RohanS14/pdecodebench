"""
Visualizations for pde-var-logprob-evolution experiment.

Loads bermaneh/pde-var-logprob-evolution-results-v1 from HuggingFace and produces
seven interactive Plotly charts showing how model token probability for corrupt vs
ground-truth variable names evolves as code is consumed (code_fraction).
"""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load data ─────────────────────────────────────────────────────────────────

from datasets import load_dataset
print("Loading dataset from HuggingFace...")
ds = load_dataset("bermaneh/pde-var-logprob-evolution-results-v1", split="train")
df = ds.to_pandas()
print(f"  {len(df)} rows loaded")

# ── Model metadata ────────────────────────────────────────────────────────────

MODEL_SHORT = {
    "Qwen/Qwen2.5-Coder-7B-Instruct":  "Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B",
    "Qwen/Qwen3-32B":                  "Qwen3-32B",
}
MODEL_TYPE = {
    "Qwen/Qwen2.5-Coder-7B-Instruct":  "Code",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Code",
    "Qwen/Qwen3-32B":                  "Reasoning",
}
MODEL_COLOR = {
    "Qwen/Qwen2.5-Coder-7B-Instruct":  "#4a9eff",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "#1a6dd4",
    "Qwen/Qwen3-32B":                  "#9b3dca",
}
TYPE_COLOR  = {"Code": "#3a8fd4", "Reasoning": "#9b3dca"}
PDE_COLOR   = {
    "burgers":       "#e74c3c",
    "heat":          "#e67e22",
    "navier-stokes": "#2ecc71",
    "wave":          "#3498db",
}
PDE_LABEL   = {
    "burgers":       "Burgers",
    "heat":          "Heat",
    "navier-stokes": "Navier-Stokes",
    "wave":          "Wave",
}

df["model_short"] = df["model"].map(MODEL_SHORT).fillna(df["model"])
df["model_type"]  = df["model"].map(MODEL_TYPE).fillna("Unknown")

MODELS    = list(MODEL_SHORT.keys())
PDE_CLASSES = ["burgers", "heat", "navier-stokes", "wave"]
N_BINS    = 20

LEGEND = dict(x=1.02, y=1, xanchor="left", yanchor="top",
              bgcolor="rgba(240,242,255,0.95)", bordercolor="#aaa", borderwidth=1,
              font=dict(color="#111"))
MARGIN = dict(l=70, r=220, t=60, b=60)

# ── Helpers ───────────────────────────────────────────────────────────────────

def bootstrap_ci(data, n=1000, ci=95):
    data = np.asarray(data.dropna() if hasattr(data, "dropna") else data)
    if len(data) < 2:
        m = float(np.mean(data)) if len(data) else np.nan
        return m, m, m
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, ci + (100 - ci) / 2)
    return lo, hi, float(np.mean(data))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def bin_by_fraction(sub, n_bins=N_BINS):
    """Bin code_fraction into n_bins equal-width bins; return bin midpoints."""
    edges = np.linspace(0, 1, n_bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    sub   = sub.copy()
    sub["bin"] = pd.cut(sub["code_fraction"], bins=edges, labels=list(range(n_bins)),
                        include_lowest=True)
    return sub, mids


def model_adaptation_traces(sub, mids, models=MODELS, dash="solid", show_ci=True):
    """Return list of Scatter traces: one line per model for logP_diff vs bin."""
    traces = []
    for m in models:
        msub = sub[sub["model"] == m]
        if msub.empty:
            continue
        grp = msub.groupby("bin")["logP_diff"]
        ys, los, his = [], [], []
        for b in range(len(mids)):
            vals = grp.get_group(b) if b in grp.groups else pd.Series([], dtype=float)
            lo, hi, mean = bootstrap_ci(vals)
            ys.append(mean); los.append(lo); his.append(hi)

        color = MODEL_COLOR[m]
        short = MODEL_SHORT[m]

        if show_ci:
            traces.append(go.Scatter(
                x=np.concatenate([mids, mids[::-1]]),
                y=np.concatenate([his, los[::-1]]),
                fill="toself", fillcolor=hex_to_rgba(color, 0.12),
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        traces.append(go.Scatter(
            x=mids, y=ys,
            mode="lines+markers",
            name=short,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{short}</b><br>Position: %{{x:.2f}}<br>logP_diff: %{{y:.2f}}<extra></extra>",
        ))
    return traces


# ── Figures ───────────────────────────────────────────────────────────────────

figs = []

def add(section, title, fig, question):
    figs.append((section, title, fig, question))


# ── V1. Adaptation curve by model (CorrVar only) ──────────────────────────────
corrvar = df[df["mod_type"] == "NoComm_CorrVar"]
corrvar_binned, mids = bin_by_fraction(corrvar)

fig_v1 = go.Figure()
for t in model_adaptation_traces(corrvar_binned, mids):
    fig_v1.add_trace(t)

fig_v1.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)")
fig_v1.update_layout(
    title="logP_diff vs code position — corrupt variable condition (all models)",
    xaxis_title="Code fraction (position in file)",
    yaxis_title="logP_diff = logP(gt var) − logP(corrupt var)",
    height=460, legend=LEGEND, margin=MARGIN,
    xaxis=dict(gridcolor="#e5e5e5", zeroline=False),
    yaxis=dict(gridcolor="#e5e5e5", zeroline=False),
)
add("LogProb Evolution", "V1. Adaptation curve by model", fig_v1,
    "Methodology: 32 PDE code snippets (Burgers, Heat, Navier-Stokes, Wave; 4 snippets each) were "
    "corrupted by replacing every variable name with a sequential foobar_N label (NoComm_CorrVar condition). "
    "All snippets contain physically valid PDE code; no invalid-code variants are present in this dataset. "
    "<br><br>"
    "<b>Variable selection:</b> Variables are classified by a whitelist + pattern classifier into four classes; "
    "only pde_state and pde_param are included as probe targets. "
    "Excluded classes: (1) loop_index — single-character loop variables i, j, k, l, m; "
    "(2) grid_size — array dimension names n, N, nx, ny, nz, nt, Nx, Ny, etc.; "
    "(3) other — names that appear in the code but match none of the above rules. "
    "Within pde_state: solution-field quantities such as u, v, w, p, T, phi, psi and their "
    "time-stepped or stencil variants (u_n, u_np1, u_jm1, u_hat, …). "
    "Within pde_param: physical and numerical constants such as dx, dy, dt, nu, mu, alpha, Re, CFL, "
    "and flux intermediates such as f, flux, fL, fR. "
    "Classification uses a fixed name whitelist with pattern-based fallbacks "
    "(names matching d[xyztrs].* → pde_param; names ending in _n, _np1, _hat, _bar, _init, stencil suffixes → pde_state). "
    "Function definition names are excluded regardless of their spelling. "
    "<br><br>"
    "<b>Probe coverage:</b> Every occurrence of every selected variable in the snippet is probed — "
    "no subsampling or stride. Each occurrence produces one probe point. "
    "For each probe, the prefix is all code from the file start up to (but not including) the variable token. "
    "Using vLLM prompt_logprobs, we compute "
    "logP_gt = Σ log P(gt token_i | prefix) and logP_corrupt = Σ log P(foobar_N token_i | prefix), "
    "summing log-probs across all BPE tokens of the variable name with BPE boundary detection. "
    "logP_diff = logP_gt − logP_corrupt: negative = model assigns higher probability to the corrupt name. "
    "code_fraction = character offset of the occurrence / total characters in the snippet. "
    "3 models × 32 snippets × ~96 probe points per snippet (varies by snippet length and variable density) "
    "= 9,270 total probe scorings. Shaded bands = 95% bootstrap CI.")


# ── V2. Code vs reasoning model types ─────────────────────────────────────────
fig_v2 = go.Figure()

thin = model_adaptation_traces(corrvar_binned, mids, show_ci=False)
for t in thin:
    t.update(line=dict(width=1, dash="dot"), opacity=0.4, showlegend=False)
    fig_v2.add_trace(t)

for mtype, color in TYPE_COLOR.items():
    sub = corrvar_binned[corrvar_binned["model_type"] == mtype]
    if sub.empty:
        continue
    grp = sub.groupby("bin")["logP_diff"]
    ys, los, his = [], [], []
    for b in range(len(mids)):
        vals = grp.get_group(b) if b in grp.groups else pd.Series([], dtype=float)
        lo, hi, mean = bootstrap_ci(vals)
        ys.append(mean); los.append(lo); his.append(hi)

    fig_v2.add_trace(go.Scatter(
        x=np.concatenate([mids, mids[::-1]]),
        y=np.concatenate([his, los[::-1]]),
        fill="toself", fillcolor=hex_to_rgba(color, 0.12),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_v2.add_trace(go.Scatter(
        x=mids, y=ys, mode="lines+markers",
        name=mtype,
        line=dict(color=color, width=3),
        marker=dict(size=7, color=color),
        hovertemplate=f"<b>{mtype}</b><br>Position: %{{x:.2f}}<br>logP_diff: %{{y:.2f}}<extra></extra>",
    ))

fig_v2.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)")
fig_v2.update_layout(
    title="logP_diff vs code position — Code vs Reasoning model type",
    xaxis_title="Code fraction", yaxis_title="logP_diff (mean ± 95% CI)",
    height=460, legend=LEGEND, margin=MARGIN,
    xaxis=dict(gridcolor="#e5e5e5", zeroline=False),
    yaxis=dict(gridcolor="#e5e5e5", zeroline=False),
)
add("LogProb Evolution", "V2. Code vs Reasoning adaptation", fig_v2,
    "Bold lines = model-type average (Code = both Qwen2.5-Coder models averaged, Reasoning = Qwen3-32B). "
    "Thin dotted lines = individual models. Shaded bands = 95% bootstrap CI.")


# ── V3. State vs param variables ──────────────────────────────────────────────
var_classes = ["pde_state", "pde_param"]
vc_colors   = {"pde_state": "#e74c3c", "pde_param": "#e67e22"}
vc_labels   = {"pde_state": "State vars (u, v, …)", "pde_param": "Param vars (dx, nu, …)"}

fig_v3 = make_subplots(
    rows=1, cols=len(MODELS),
    subplot_titles=[MODEL_SHORT[m] for m in MODELS],
    shared_yaxes=True,
)
for col_idx, m in enumerate(MODELS, 1):
    for vc in var_classes:
        sub = corrvar_binned[(corrvar_binned["model"] == m) & (corrvar_binned["var_class"] == vc)]
        if sub.empty:
            continue
        grp = sub.groupby("bin")["logP_diff"]
        ys, los, his = [], [], []
        for b in range(len(mids)):
            vals = grp.get_group(b) if b in grp.groups else pd.Series([], dtype=float)
            lo, hi, mean = bootstrap_ci(vals)
            ys.append(mean); los.append(lo); his.append(hi)

        color = vc_colors[vc]
        fig_v3.add_trace(go.Scatter(
            x=np.concatenate([mids, mids[::-1]]),
            y=np.concatenate([his, los[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, 0.12),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=col_idx)
        fig_v3.add_trace(go.Scatter(
            x=mids, y=ys, mode="lines+markers",
            name=vc_labels[vc],
            line=dict(color=color, width=2),
            marker=dict(size=5),
            legendgroup=vc,
            showlegend=(col_idx == 1),
            hovertemplate=f"<b>{vc}</b><br>Position: %{{x:.2f}}<br>%{{y:.2f}}<extra></extra>",
        ), row=1, col=col_idx)

fig_v3.update_layout(
    title="logP_diff by variable class — state (u, v) vs param (dx, nu) per model",
    height=440,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=MARGIN,
)
fig_v3.update_xaxes(gridcolor="#e5e5e5", zeroline=False)
fig_v3.update_yaxes(gridcolor="#e5e5e5", zeroline=False)
add("LogProb Evolution", "V3. State vs param variable adaptation", fig_v3,
    "Red = PDE state variables (u, v — the solution field). Orange = PDE parameter variables "
    "(dx, nu — grid spacing and physics constants). Shaded bands = 95% bootstrap CI.")


# ── V4. Early vs late adaptation delta ────────────────────────────────────────
EARLY = (0.0, 0.2)
LATE  = (0.8, 1.0)

model_shorts  = [MODEL_SHORT[m] for m in MODELS]
early_means, late_means = [], []
for m in MODELS:
    msub = corrvar[corrvar["model"] == m]
    _, _, em = bootstrap_ci(msub[msub["code_fraction"].between(*EARLY)]["logP_diff"])
    _, _, lm = bootstrap_ci(msub[msub["code_fraction"].between(*LATE)]["logP_diff"])
    early_means.append(em)
    late_means.append(lm)

fig_v4 = go.Figure()
fig_v4.add_trace(go.Bar(name="Early (0–20%)",   x=model_shorts, y=early_means, marker_color="#3a8fd4"))
fig_v4.add_trace(go.Bar(name="Late (80–100%)",  x=model_shorts, y=late_means,  marker_color="#e74c3c"))

for em, lm, ms in zip(early_means, late_means, model_shorts):
    fig_v4.add_annotation(x=ms, y=min(em, lm) - 0.5,
                          text=f"Δ {lm - em:.1f}", showarrow=False,
                          font=dict(size=10, color="#555"))

fig_v4.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)")
fig_v4.update_layout(
    title="Adaptation acceleration: early (0–20%) vs late (80–100%) code position",
    xaxis_title="Model", yaxis_title="Mean logP_diff",
    barmode="group", height=420, legend=LEGEND, margin=MARGIN,
    xaxis=dict(gridcolor="#e5e5e5"),
    yaxis=dict(gridcolor="#e5e5e5"),
)
add("LogProb Evolution", "V4. Early vs late adaptation delta", fig_v4,
    "Blue = mean logP_diff at the start of the file (first 20%). Red = mean logP_diff at the end (last 20%). "
    "Δ annotation shows how much more negative the preference for foobar_N becomes as context accumulates.")


# ── V5. Valid condition control ────────────────────────────────────────────────
valid        = df[df["mod_type"] == "NoComm_Valid"]
valid_binned, _ = bin_by_fraction(valid)

fig_v5 = go.Figure()
for t in model_adaptation_traces(valid_binned, mids):
    fig_v5.add_trace(t)

fig_v5.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)",
                 annotation_text="logP_diff = 0 (expected)", annotation_position="right",
                 annotation_font=dict(size=9, color="#888"))
fig_v5.update_layout(
    title="Control: logP_diff vs code position — valid (non-corrupt) condition",
    xaxis_title="Code fraction", yaxis_title="logP_diff",
    height=420, legend=LEGEND, margin=MARGIN,
    xaxis=dict(gridcolor="#e5e5e5", zeroline=False),
    yaxis=dict(gridcolor="#e5e5e5", zeroline=False),
)
add("LogProb Evolution", "V5. Valid condition control (sanity check)", fig_v5,
    "In the NoComm_Valid condition, gt_var == corrupt_var (no corruption), so logP_diff = 0 by definition. "
    "Any systematic deviation indicates a bug in the scoring pipeline. All models should be flat at zero.")


# ── V6. logP_diff distribution by quintile (box plot) ─────────────────────────
corrvar_q = corrvar.copy()
Q_LABELS = ["Q1 (0–20%)", "Q2 (20–40%)", "Q3 (40–60%)", "Q4 (60–80%)", "Q5 (80–100%)"]
corrvar_q["quintile"] = pd.qcut(corrvar_q["code_fraction"], q=5, labels=Q_LABELS)

fig_v6 = go.Figure()
for m in MODELS:
    msub  = corrvar_q[corrvar_q["model"] == m]
    color = MODEL_COLOR[m]
    short = MODEL_SHORT[m]
    for q in Q_LABELS:
        qvals = msub[msub["quintile"] == q]["logP_diff"]
        fig_v6.add_trace(go.Box(
            y=qvals.values, x=[q] * len(qvals),
            name=short, legendgroup=short,
            showlegend=(q == Q_LABELS[0]),
            marker_color=color, line_color=color, boxmean=True,
            hovertemplate=f"<b>{short}</b><br>{q}<br>%{{y:.2f}}<extra></extra>",
        ))

fig_v6.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)")
fig_v6.update_layout(
    title="Distribution of logP_diff by code-fraction quintile (corrupt condition)",
    xaxis_title="Code fraction quintile", yaxis_title="logP_diff",
    boxmode="group", height=480, legend=LEGEND, margin=MARGIN,
    xaxis=dict(gridcolor="#e5e5e5"),
    yaxis=dict(gridcolor="#e5e5e5"),
)
add("LogProb Evolution", "V6. logP_diff distribution by quintile", fig_v6,
    "Box plots of raw logP_diff values across five equal-quantile code-position bins. "
    "Dot inside each box = mean. If the trend is outlier-driven, the box stays near 0 while the mean drops. "
    "A consistent shift means the entire distribution moves negative as quintile increases.")


# ── V7. Adaptation by PDE class ───────────────────────────────────────────────
fig_v7 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[PDE_LABEL[p] for p in PDE_CLASSES],
    shared_yaxes=True, shared_xaxes=True,
    vertical_spacing=0.12, horizontal_spacing=0.06,
)
rc = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (row, col), pde in zip(rc, PDE_CLASSES):
    pde_sub = corrvar_binned[corrvar_binned["pde_class"] == pde]
    for m in MODELS:
        msub  = pde_sub[pde_sub["model"] == m]
        if msub.empty:
            continue
        grp = msub.groupby("bin")["logP_diff"]
        ys, los, his = [], [], []
        for b in range(len(mids)):
            vals = grp.get_group(b) if b in grp.groups else pd.Series([], dtype=float)
            lo, hi, mean = bootstrap_ci(vals)
            ys.append(mean); los.append(lo); his.append(hi)

        color = MODEL_COLOR[m]
        short = MODEL_SHORT[m]
        fig_v7.add_trace(go.Scatter(
            x=np.concatenate([mids, mids[::-1]]),
            y=np.concatenate([his, los[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, 0.10),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig_v7.add_trace(go.Scatter(
            x=mids, y=ys, mode="lines+markers",
            name=short,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            legendgroup=short,
            showlegend=(row == 1 and col == 1),
            hovertemplate=f"<b>{short}</b><br>{PDE_LABEL[pde]}<br>Position: %{{x:.2f}}<br>%{{y:.2f}}<extra></extra>",
        ), row=row, col=col)

fig_v7.update_layout(
    title="logP_diff vs code position by PDE class (corrupt condition, all models)",
    height=600,
    legend=dict(x=1.02, y=1, xanchor="left", bgcolor="rgba(240,242,255,0.95)",
                bordercolor="#aaa", borderwidth=1, font=dict(color="#111")),
    margin=dict(l=70, r=200, t=70, b=60),
)
fig_v7.update_xaxes(gridcolor="#e5e5e5", zeroline=False, title_text="Code fraction")
fig_v7.update_yaxes(gridcolor="#e5e5e5", zeroline=False, title_text="logP_diff")
fig_v7.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.3)")
add("LogProb Evolution", "V7. Adaptation by PDE class", fig_v7,
    "Each subplot shows the logP_diff adaptation curve for one PDE type (4 snippets × all models). "
    "Burgers (1D, convection-diffusion), Heat (parabolic diffusion), Navier-Stokes (incompressible flow), "
    "Wave (hyperbolic, second-order). Shaded bands = 95% bootstrap CI. "
    "Different PDE types have different variable vocabularies and structural complexity — "
    "does adaptation strength or rate vary across equation families?")


# ── V8. State vs param vars — averaged across models ──────────────────────────
# Only two var_class values in this dataset: pde_state and pde_param.
# pde_state: solution-field variables (u, v, p, …)
# pde_param: grid/physics constants (dx, dt, nu, Re, …)
fig_v8 = go.Figure()

# Thin per-model lines behind (one per model × var_class combination)
for vc, vc_color in vc_colors.items():
    for m in MODELS:
        msub = corrvar_binned[(corrvar_binned["var_class"] == vc) & (corrvar_binned["model"] == m)]
        if msub.empty:
            continue
        grp = msub.groupby("bin")["logP_diff"]
        ys = []
        for b in range(len(mids)):
            vals = grp.get_group(b) if b in grp.groups else pd.Series([], dtype=float)
            _, _, mean = bootstrap_ci(vals)
            ys.append(mean)
        fig_v8.add_trace(go.Scatter(
            x=mids, y=ys, mode="lines",
            name=f"{MODEL_SHORT[m]} / {vc_labels[vc]}",
            line=dict(color=vc_color, width=1, dash="dot"),
            opacity=0.35, showlegend=False,
            hovertemplate=f"<b>{MODEL_SHORT[m]}</b> — {vc}<br>Position: %{{x:.2f}}<br>%{{y:.2f}}<extra></extra>",
        ))

# Bold aggregate lines (averaged across all models)
for vc, vc_color in vc_colors.items():
    sub = corrvar_binned[corrvar_binned["var_class"] == vc]
    grp = sub.groupby("bin")["logP_diff"]
    ys, los, his = [], [], []
    for b in range(len(mids)):
        vals = grp.get_group(b) if b in grp.groups else pd.Series([], dtype=float)
        lo, hi, mean = bootstrap_ci(vals)
        ys.append(mean); los.append(lo); his.append(hi)

    fig_v8.add_trace(go.Scatter(
        x=np.concatenate([mids, mids[::-1]]),
        y=np.concatenate([his, los[::-1]]),
        fill="toself", fillcolor=hex_to_rgba(vc_color, 0.12),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_v8.add_trace(go.Scatter(
        x=mids, y=ys, mode="lines+markers",
        name=vc_labels[vc],
        line=dict(color=vc_color, width=3),
        marker=dict(size=7, color=vc_color),
        hovertemplate=f"<b>{vc_labels[vc]}</b><br>Position: %{{x:.2f}}<br>logP_diff: %{{y:.2f}}<extra></extra>",
    ))

fig_v8.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.4)")
fig_v8.update_layout(
    title="logP_diff by variable class — state vs param (averaged across all models)",
    xaxis_title="Code fraction (position in file)",
    yaxis_title="logP_diff = logP(gt var) − logP(corrupt var)",
    height=460, legend=LEGEND, margin=MARGIN,
    xaxis=dict(gridcolor="#e5e5e5", zeroline=False),
    yaxis=dict(gridcolor="#e5e5e5", zeroline=False),
)
add("LogProb Evolution", "V8. State vs param vars (aggregated)", fig_v8,
    "Bold lines = mean logP_diff averaged across all three models. "
    "Thin dotted lines = individual model contributions per class. Shaded bands = 95% bootstrap CI. "
    "State variables (red): solution-field quantities — u, v, w, p, T, phi, psi and their "
    "time-stepped / stencil variants (u_n, u_np1, u_jm1, …). "
    "Param variables (orange): physical and numerical constants — dx, dy, dt, nu, mu, alpha, Re, CFL, "
    "and flux intermediates (f, flux, fL, fR). "
    "These are the only two classes included as probe targets. "
    "Excluded from probing: loop indices (i, j, k, l, m), grid-size names (nx, ny, nt, Nx, …), "
    "function definition names, and anything else unclassified ('other'). "
    "If state variables adapt more strongly than param variables, it suggests models encode semantic roles "
    "(solution field vs. numerical scaffolding) when updating their token expectations under corrupt naming.")


# ── Assemble HTML ─────────────────────────────────────────────────────────────

sections: dict = {}
for section, title, fig, question in figs:
    sections.setdefault(section, []).append((title, fig, question))

nav_html = content_html = ""
first = True
idx   = 0

for section_name, charts in sections.items():
    nav_html += f'<div class="nav-section">{section_name}</div>\n'
    for title, fig, question in charts:
        active    = "active" if first else ""
        nav_html += f'<button class="nav-btn {active}" onclick="show({idx})">{title}</button>\n'
        fig_html  = fig.to_html(full_html=False, include_plotlyjs=(idx == 0), div_id=f"fig{idx}")
        content_html += f"""
<div class="section {active}" id="sec{idx}">
  <div class="chart-header">
    <span class="section-tag">{section_name}</span>
    <h2>{title}</h2>
    <p class="question">{question}</p>
  </div>
  {fig_html}
</div>"""
        first = False
        idx  += 1

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LogProb Evolution — Variable Naming Adaptation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0d0f18; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 260px; min-width: 260px; background: #12141e;
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
  .question {{ font-size: 0.8rem; color: #5a80b0; max-width: 860px;
               border-left: 3px solid #1e3560; padding-left: 10px;
               line-height: 1.6; margin-bottom: 16px; }}
</style>
</head>
<body>
<div id="sidebar">
  <h1>LogProb Evolution<br>Variable Naming Adaptation</h1>
  <div class="hyp">
    <strong>Hypothesis:</strong> logP_diff = logP(gt) − logP(corrupt) trends negative as models
    read corrupted code, showing adaptation to foobar_N naming.
    <br><br>
    <strong>Models:</strong> Qwen2.5-Coder-7B · Qwen2.5-Coder-32B · Qwen3-32B (reasoning)
    <br><br>
    <strong>Data:</strong> 9,270 probe points · 32 PDE snippets · 2 conditions
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

out = "var_logprob_evolution.html"
with open(out, "w") as f:
    f.write(html)

print(f"✓ Saved to {out}")
for label in ["V1: Adaptation curve by model", "V2: Code vs Reasoning",
              "V3: State vs param variables (per model)", "V4: Early vs late delta",
              "V5: Valid condition control", "V6: Distribution by quintile",
              "V7: Adaptation by PDE class", "V8: State vs param vars (aggregated)"]:
    print(f"  {label}")
