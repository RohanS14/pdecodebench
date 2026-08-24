"""
freegen_jul28_report.py — Experiment 1 (free generation) on the jul28 dataset.

Self-contained interactive HTML report. Reproduces the two cuts from writeup.pdf
Figure 1 (score by condition; validity confidence breakdown) on the 256-row
merged_mod_jul28 dataset, and adds the two cuts the jul28 release makes possible:

  * source — human-written vs synthetic base solvers (128 / 128)
  * hedge calibration — is a hedged answer actually less likely to be right?

Usage:
    python freegen/report.py \
        --input  results/pde_llm_eval_jul28.csv \
        --output viz/freegen_jul28_report.html
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# This file's own directory FIRST, eval/ appended after. Both hold a parse_score.py on
# the cluster and eval/'s is the stale pre-split copy; inserting eval/ at position 0
# shadowed freegen/parse_score.py, which is what killed the first canary submission.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from parse_score import classify_valid_confidence, valid_intent, VALID_CONF_CLASSES  # noqa: E402

TEMPLATE = "plotly_white"
METRIC_COLORS = px.colors.qualitative.Plotly

ALL_CONDS = [
    "Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
    "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid",
]
COND_SHORT = {
    "Comm_Valid":             "Clean+Comment",
    "NoComm_Valid":           "Clean, No Comment",
    "CorrComm":               "Corrupt Comment",
    "NoComm_CorrVar":         "Obfuscated Variables",
    "Comm_InValid":           "Invalid+Comment",
    "NoComm_InValid":         "Invalid, No Comment",
    "CorrComm_Invalid":       "CorrComment+Invalid",
    "NoComm_CorrVar_InValid": "Obfuscated+Invalid",
}
LLM_METRICS = [
    ("pde_match",       "PDE Type"),
    ("method_recall",   "Method Recall"),
    ("behavior_recall", "Behavior Recall"),
    ("valid_match",     "Validity"),
]
CONF_COLORS = {"Confident Yes": "#27ae60", "Uncertain Yes": "#f1c40f",
               "Hedged": "#e67e22", "Confident No": "#c0392b"}
SOURCE_COLORS = {"human": "#2980b9", "synthetic": "#c0392b"}

MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct":          "Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct":         "Llama-3.3-70B",
    "Qwen/QwQ-32B":                              "QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct":            "Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct":           "Qwen2.5-Coder-32B",
    "Qwen/Qwen3-32B":                            "Qwen3-32B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct":         "Qwen3-Coder-30B",
    "google/gemma-3-27b-it":                     "Gemma-3-27B",
    "mistralai/Mistral-Nemo-Instruct-2407":      "Mistral-12B",
    "microsoft/phi-4":                           "phi-4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":  "DeepSeek-R1-32B",
}
REASONING_MODELS = {"QwQ-32B", "DeepSeek-R1-32B", "Qwen3-32B"}


ITEM_KEYS = ["model", "thinking", "mod_type", "title"]


def pool_draws(df):
    """Collapse k sampled draws of one item into ONE observation.

    Under k>1 the frame holds three rows per item, and they are three samples of one
    model on one prompt -- correlated by construction, not three independent items.
    Bootstrapping the raw rows resamples 3n values with the same mean, so the point
    estimate is unchanged and every interval narrows by about sqrt(k): a 42% shrink
    at k=3, in the direction that makes a result look real. Numeric scores are
    averaged over the draws, which is the per-item expected score under the sampling
    distribution; categorical fields take the item's modal value.

    A k=1 frame passes through unchanged, so callers need not know which they hold.
    """
    if "sample_idx" not in df.columns or df["sample_idx"].nunique() <= 1:
        return df
    keys = [k for k in ITEM_KEYS if k in df.columns]
    num = df.select_dtypes(include=[np.number]).columns.difference(
        keys + ["sample_idx", "k_draws"])

    def _one(g):
        row = g.iloc[0].copy()
        for c in num:
            row[c] = g[c].mean()
        for c in ("valid_conf", "valid_lean", "parsed_valid", "finish_reason"):
            if c in g.columns:
                m = g[c].mode(dropna=True)
                row[c] = m.iloc[0] if len(m) else np.nan
        row["n_draws_pooled"] = len(g)
        return row

    out = df.groupby(keys, dropna=False, sort=False).apply(
        _one, include_groups=False).reset_index()
    return out


def bootstrap_ci(data, n_bootstrap=1000, ci=95, seed=42):
    """Mean with a percentile bootstrap CI over ITEMS. Returns (lo, hi, mean, n).

    Callers must hand this pooled items, never raw draws -- see pool_draws(). load()
    pools before returning, so every figure in this file gets item-level input.
    """
    data = pd.Series(data).dropna().values
    if len(data) < 1:
        return np.nan, np.nan, np.nan, 0
    if len(data) < 2:
        return float(data[0]), float(data[0]), float(data[0]), 1
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n_bootstrap, len(data)), replace=True).mean(axis=1)
    return (float(np.percentile(means, (100 - ci) / 2)),
            float(np.percentile(means, ci + (100 - ci) / 2)),
            float(np.mean(data)), len(data))


# ═════════════════════════════════════════════════════════════════════════════
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["pde_match", "method_any_match", "behavior_any_match", "valid_match",
                "pde_embed_sim", "method_recall", "behavior_recall", "num_char"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gt_valid"] = df["gt_valid"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    df["model_short"] = df["model"].map(MODEL_SHORT).fillna(df["model"])
    df["model_type"] = np.where(df["model_short"].isin(REASONING_MODELS),
                                "Reasoning", "Non-reasoning")

    # valid_conf is written by run_eval.py; recompute only for older results.
    if "valid_conf" not in df.columns:
        df["valid_conf"] = df["parsed_valid"].apply(classify_valid_confidence)

    # The directional lean, which exists even inside the Hedged bucket.
    # score_valid abstains on a hedge (valid_match = 0), so hedge-conditioned
    # accuracy computed from valid_match would be 0 by construction.
    df["valid_lean"] = df["parsed_valid"].apply(valid_intent)
    df["lean_correct"] = np.where(df["valid_lean"].isna(), np.nan,
                                  (df["valid_lean"] == df["gt_valid"]).astype(float))

    df["overall_acc"] = df[["pde_match", "method_any_match",
                            "behavior_any_match", "valid_match"]].mean(axis=1)

    # A run that reached no answer is DROPPED, not scored. Left in, score_valid gives
    # it valid_match=0 -- counting a model that said nothing as one that said
    # something false -- and classify_valid_confidence files it as "Hedged", which
    # inflates a bucket this report plots. The count is printed rather than absorbed.
    if "no_verdict" in df.columns:
        nv = df["no_verdict"].fillna(False).astype(bool)
        if nv.any():
            print(f"[report] dropping {int(nv.sum())} row(s) with no verdict "
                  f"({nv.mean():.1%}) — these reached no answer")
            df = df[~nv].copy()

    # Pool k draws to one observation per item BEFORE anything computes an interval.
    n_before = len(df)
    df = pool_draws(df)
    if len(df) != n_before:
        print(f"[report] pooled {n_before} draws -> {len(df)} items "
              f"(k={n_before / max(1, len(df)):.1f}); every CI below is over ITEMS")
    return df


FIGS = []


def add(fig, title, caption):
    FIGS.append((title, fig, caption))


def conds_present(df):
    return [c for c in ALL_CONDS if c in set(df["mod_type"])]


# ── 1. Score by condition ────────────────────────────────────────────────────
def fig_score_by_condition(df):
    conds = conds_present(df)
    fig = go.Figure()
    for i, (metric, label) in enumerate(LLM_METRICS):
        means, lo_err, hi_err, ns = [], [], [], []
        for cond in conds:
            lo, hi, mn, n = bootstrap_ci(df[df["mod_type"] == cond][metric])
            means.append(mn * 100); lo_err.append((mn - lo) * 100)
            hi_err.append((hi - mn) * 100); ns.append(n)
        fig.add_trace(go.Scatter(
            x=[COND_SHORT[c] for c in conds], y=means, mode="lines+markers", name=label,
            line=dict(color=METRIC_COLORS[i], width=2.5), marker=dict(size=8),
            error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                         color="rgba(0,0,0,0.35)", thickness=1.2, width=4),
            customdata=ns,
            hovertemplate=label + " | %{x}: %{y:.1f}%  (n=%{customdata})<extra></extra>",
        ))
    for x0, x1, color, note in [
        ("Corrupt Comment", "Obfuscated Variables", "#e74c3c", "lexical cues removed/misleading"),
        ("Invalid+Comment", "Obfuscated+Invalid",  "#9b59b6", "physically invalid code"),
    ]:
        if x0 in [COND_SHORT[c] for c in conds] and x1 in [COND_SHORT[c] for c in conds]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.07, line_width=0,
                          annotation_text=note, annotation_position="top left",
                          annotation_font=dict(size=9, color="#999"))
    fig.update_layout(
        title="① Score by condition — all models pooled (95% bootstrap CI)",
        yaxis_title="Score (%)", yaxis_range=[0, 100], template=TEMPLATE,
        height=460, margin=dict(l=70, r=200, t=70, b=110),
        legend=dict(x=1.02, y=1, xanchor="left"))
    fig.update_xaxes(tickangle=-25)
    add(fig, "Score by condition",
        "The writeup.pdf Figure 1 (left) cut, recomputed on the 256-row jul28 dataset "
        "(32 base solvers instead of 16). Method and behavior are scored by partial "
        "recall over the ground-truth label set; PDE type and validity are binary. "
        "A hedged validity answer abstains and scores 0 — see panel ⑤ for what those "
        "rows actually said.")
    return fig


# ── 2. Validity confidence (hedging) ─────────────────────────────────────────
def fig_hedge_breakdown(df):
    conds = conds_present(df)
    valid_c   = [c for c in conds if "InValid" not in c and "Invalid" not in c]
    invalid_c = [c for c in conds if "InValid" in c or "Invalid" in c]
    ordered = valid_c + invalid_c
    labels = ([COND_SHORT[c] + " ✓" for c in valid_c] +
              ["⚠ " + COND_SHORT[c] for c in invalid_c])

    fig = go.Figure()
    for cat in VALID_CONF_CLASSES:
        means, lo_err, hi_err = [], [], []
        for cond in ordered:
            sub = df[df["mod_type"] == cond]
            lo, hi, mn, _ = bootstrap_ci((sub["valid_conf"] == cat).astype(float))
            means.append(mn * 100); lo_err.append((mn - lo) * 100); hi_err.append((hi - mn) * 100)
        fig.add_trace(go.Bar(
            name=cat, x=labels, y=means, marker_color=CONF_COLORS[cat],
            error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                         color="rgba(0,0,0,0.45)", thickness=1.2, width=4),
            hovertemplate=cat + " | %{x}: %{y:.1f}%<extra></extra>"))

    boundary = len(valid_c) - 0.5
    fig.add_vline(x=boundary, line_color="rgba(130,60,180,0.7)", line_width=1.5, line_dash="dash")
    fig.add_vrect(x0=boundary, x1=len(ordered) - 0.5, fillcolor="#7d3c98", opacity=0.08, line_width=0)
    fig.add_annotation(x=0, y=107, text="VALID CODE — correct answer is Yes",
                       showarrow=False, font=dict(size=10, color="#1a8040"), xanchor="left")
    fig.add_annotation(x=len(ordered) - 1, y=107, text="INVALID CODE — correct answer is No",
                       showarrow=False, font=dict(size=10, color="#7d3c98"), xanchor="right")
    fig.update_layout(
        title="② Validity confidence breakdown — how models hedge (95% bootstrap CI)",
        barmode="stack", yaxis_title="% of predictions", yaxis_range=[0, 112],
        template=TEMPLATE, height=500, margin=dict(l=70, r=200, t=70, b=140),
        legend=dict(x=1.02, y=1, xanchor="left", title="Stated confidence"))
    fig.update_xaxes(tickangle=-25)
    add(fig, "Validity confidence breakdown",
        "Bucket shares are NOT comparable to writeup.pdf Figure 1 (right). That figure "
        "used a looser rule with no hedge lexicon, so phrasings like \"possibly valid\" "
        "fell into a catch-all; the canonical rule in freegen/parse_score.py routes them to "
        "<b>Hedged</b>. Confident-Yes bars over the shaded region are the failure of "
        "interest: the model asserts physically invalid code is fine.")
    return fig


# ── 3. Source: human vs synthetic ────────────────────────────────────────────
def fig_by_source(df):
    if "source" not in df.columns or df["source"].isna().all():
        return None
    conds = conds_present(df)
    sources = [s for s in ("human", "synthetic") if s in set(df["source"])]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11, subplot_titles=[
        "Overall score by condition", "Validity accuracy by condition"])
    for src in sources:
        for col, (metric, _) in enumerate([("overall_acc", ""), ("valid_match", "")], start=1):
            means, lo_err, hi_err, ns = [], [], [], []
            for cond in conds:
                sub = df[(df["mod_type"] == cond) & (df["source"] == src)]
                lo, hi, mn, n = bootstrap_ci(sub[metric])
                means.append(mn * 100); lo_err.append((mn - lo) * 100)
                hi_err.append((hi - mn) * 100); ns.append(n)
            fig.add_trace(go.Scatter(
                x=[COND_SHORT[c] for c in conds], y=means, mode="lines+markers",
                name=src, legendgroup=src, showlegend=(col == 1),
                line=dict(color=SOURCE_COLORS[src], width=2.5), marker=dict(size=7),
                error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                             color="rgba(0,0,0,0.3)", thickness=1.2, width=4),
                customdata=ns,
                hovertemplate=src + " | %{x}: %{y:.1f}%  (n=%{customdata})<extra></extra>",
            ), row=1, col=col)

    fig.update_layout(
        title="③ Human-written vs synthetic base solvers (95% bootstrap CI)",
        template=TEMPLATE, height=470, margin=dict(l=70, r=180, t=80, b=130),
        legend=dict(x=1.02, y=1, xanchor="left", title="source"))
    fig.update_yaxes(title_text="Score (%)", range=[0, 100])
    fig.update_xaxes(tickangle=-25)

    lens = df.groupby("source")["num_char"].mean() if "num_char" in df.columns else None
    len_note = ""
    if lens is not None and not lens.isna().all():
        len_note = (" <b>Read with care:</b> source is confounded with code length — "
                    + ", ".join(f"{s} solvers average {lens[s]:,.0f} characters"
                                for s in sources if s in lens.index)
                    + ". Any human/synthetic gap may be a length effect; panel ④ separates them.")
    add(fig, "Human vs synthetic",
        "New at jul28: the 32 base problems come from two independently-tracked pools "
        "of 16 — hand-written solvers and LLM-generated ones (both human-verified). "
        "Donors for the corrupted-comment and obfuscated-variable conditions never "
        "cross the source boundary, so the two halves are independent." + len_note)
    return fig


# ── 4. Length control for the source cut ─────────────────────────────────────
def fig_length_control(df):
    if "num_char" not in df.columns or df["num_char"].isna().all():
        return None
    per_snippet = (df.groupby(["title", "source", "num_char"])["overall_acc"]
                     .mean().reset_index())
    if per_snippet.empty:
        return None
    # Shared length bins so human and synthetic are compared like for like.
    per_snippet["len_bin"] = pd.qcut(per_snippet["num_char"], q=4, duplicates="drop",
                                     labels=None)
    bins = [b for b in per_snippet["len_bin"].cat.categories]
    bin_labels = [f"{int(b.left):,}–{int(b.right):,}" for b in bins]

    fig = go.Figure()
    for src in ("human", "synthetic"):
        means, lo_err, hi_err, ns = [], [], [], []
        for b in bins:
            sub = per_snippet[(per_snippet["source"] == src) & (per_snippet["len_bin"] == b)]
            lo, hi, mn, n = bootstrap_ci(sub["overall_acc"])
            means.append(mn * 100 if n else None)
            lo_err.append((mn - lo) * 100 if n else 0)
            hi_err.append((hi - mn) * 100 if n else 0)
            ns.append(n)
        fig.add_trace(go.Bar(
            name=src, x=bin_labels, y=means, marker_color=SOURCE_COLORS[src],
            error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                         color="rgba(0,0,0,0.4)", thickness=1.2, width=4),
            customdata=ns,
            hovertemplate=src + " | %{x} chars: %{y:.1f}%  (n=%{customdata} snippets)<extra></extra>"))
    fig.update_layout(
        title="④ Source gap within matched code-length bins",
        barmode="group", xaxis_title="Code length (characters, shared quartile bins)",
        yaxis_title="Overall score (%)", template=TEMPLATE, height=430,
        margin=dict(l=70, r=180, t=70, b=80),
        legend=dict(x=1.02, y=1, xanchor="left", title="source"))
    add(fig, "Source gap, length-controlled",
        "Human solvers are roughly 2.5x longer than synthetic ones, so the raw source "
        "gap in panel ③ is partly a length effect. Bins are quartiles of the pooled "
        "character count, so a bar pair compares snippets of similar size. Empty bars "
        "mean one source has no snippets in that bin — that imbalance is itself the "
        "finding, and the comparison should not be read across it. "
        "Length uses num_char, not num_lines: num_lines is +1 on 80/256 rows "
        "(all synthetic, a parser artifact).")
    return fig


# ── 5. Hedge calibration ─────────────────────────────────────────────────────
def fig_hedge_calibration(df):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13, subplot_titles=[
        "Is the stated lean right, given how it was stated?",
        "How often does the model hedge, given it was right / wrong?"])

    # Left: accuracy of the directional lean within each confidence bucket.
    means, lo_err, hi_err, ns, no_lean = [], [], [], [], []
    for cat in VALID_CONF_CLASSES:
        sub = df[df["valid_conf"] == cat]
        lo, hi, mn, n = bootstrap_ci(sub["lean_correct"])
        means.append(mn * 100); lo_err.append((mn - lo) * 100)
        hi_err.append((hi - mn) * 100); ns.append(n)
        no_lean.append(int(sub["valid_lean"].isna().sum()))
    fig.add_trace(go.Bar(
        x=list(VALID_CONF_CLASSES), y=means,
        marker_color=[CONF_COLORS[c] for c in VALID_CONF_CLASSES],
        error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                     color="rgba(0,0,0,0.45)", thickness=1.2, width=4),
        customdata=np.stack([ns, no_lean], axis=-1), showlegend=False,
        hovertemplate="%{x}: %{y:.1f}%  (n=%{customdata[0]} with a readable lean, "
                      "%{customdata[1]} with none)<extra></extra>"), row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(0,0,0,0.35)",
                  annotation_text="chance (binary)", annotation_font=dict(size=9),
                  row=1, col=1)

    # Right: hedge rate conditioned on whether the lean was right.
    for cat in VALID_CONF_CLASSES:
        ys, err_lo, err_hi = [], [], []
        for correct, _ in [(1.0, "right"), (0.0, "wrong")]:
            sub = df[df["lean_correct"] == correct]
            lo, hi, mn, _ = bootstrap_ci((sub["valid_conf"] == cat).astype(float))
            ys.append(mn * 100); err_lo.append((mn - lo) * 100); err_hi.append((hi - mn) * 100)
        fig.add_trace(go.Bar(
            name=cat, x=["lean was right", "lean was wrong"], y=ys,
            marker_color=CONF_COLORS[cat],
            error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                         color="rgba(0,0,0,0.45)", thickness=1.2, width=4),
            hovertemplate=cat + " | %{x}: %{y:.1f}%<extra></extra>"), row=1, col=2)

    fig.update_layout(
        title="⑤ Hedge calibration — does hedging track being wrong?",
        barmode="stack", template=TEMPLATE, height=470,
        margin=dict(l=70, r=190, t=80, b=90),
        legend=dict(x=1.02, y=1, xanchor="left", title="Stated confidence"))
    fig.update_yaxes(title_text="Lean matches ground truth (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="% of predictions", range=[0, 105], row=1, col=2)
    fig.update_xaxes(tickangle=-15, row=1, col=1)
    add(fig, "Hedge calibration",
        "The scorer treats a hedge as an abstention worth 0, so accuracy within the "
        "Hedged bucket is 0 by construction if read off valid_match. Both panels use "
        "the <i>directional lean</i> instead (parse_score.valid_intent): the yes/no the "
        "response leans toward, even when wrapped in \"possibly\" or \"it depends\". "
        "Left: if the model is calibrated, Confident buckets sit above Hedged. Right: "
        "if hedging is informative, the Hedged share is larger over wrong answers than "
        "right ones. Rows with no readable lean are excluded and counted in the tooltip.")
    return fig


# ── 6. Reasoning vs non-reasoning ────────────────────────────────────────────
def fig_reasoning(df):
    conds = conds_present(df)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=["Score by condition", "Score by model"])
    for mtype, color in [("Reasoning", "#e67e22"), ("Non-reasoning", "#3498db")]:
        sub_t = df[df["model_type"] == mtype]
        if sub_t.empty:
            continue
        means, lo_err, hi_err = [], [], []
        for cond in conds:
            lo, hi, mn, _ = bootstrap_ci(sub_t[sub_t["mod_type"] == cond]["overall_acc"])
            means.append(mn * 100); lo_err.append((mn - lo) * 100); hi_err.append((hi - mn) * 100)
        fig.add_trace(go.Bar(
            name=mtype, x=[COND_SHORT[c] for c in conds], y=means, marker_color=color,
            error_y=dict(type="data", symmetric=False, array=hi_err, arrayminus=lo_err,
                         color="rgba(0,0,0,0.4)", thickness=1.2, width=4),
            legendgroup=mtype,
            hovertemplate=mtype + " | %{x}: %{y:.1f}%<extra></extra>"), row=1, col=1)

    order = sorted(df["model_short"].unique(),
                   key=lambda m: df[df["model_short"] == m]["overall_acc"].mean())
    for m in order:
        is_r = m in REASONING_MODELS
        lo, hi, mn, _ = bootstrap_ci(df[df["model_short"] == m]["overall_acc"])
        fig.add_trace(go.Bar(
            name=m, x=[m], y=[mn * 100], showlegend=False,
            marker_color="#e67e22" if is_r else "#3498db",
            marker_pattern_shape="x" if is_r else "",
            error_y=dict(type="data", symmetric=False, array=[(hi - mn) * 100],
                         arrayminus=[(mn - lo) * 100],
                         color="rgba(0,0,0,0.4)", thickness=1.2, width=4),
            hovertemplate=m + ": %{y:.1f}%<extra></extra>"), row=1, col=2)

    fig.update_layout(
        title="⑥ Reasoning vs non-reasoning robustness",
        barmode="group", template=TEMPLATE, height=520,
        margin=dict(l=70, r=180, t=80, b=160),
        legend=dict(x=1.02, y=1, xanchor="left"))
    ytitle = "Score (avg. PDE type, method, behavior, validity) %"
    fig.update_yaxes(title_text=ytitle, title_font=dict(size=10), range=[0, 100])
    fig.update_xaxes(tickangle=-30)
    add(fig, "Reasoning vs non-reasoning",
        "Model type is assigned from the roster, not measured: QwQ-32B, "
        "DeepSeek-R1-Distill-32B and Qwen3-32B are the reasoning models. Thinking is "
        "suppressed for Qwen3/QwQ via enable_thinking=False; DeepSeek-R1 always emits "
        "&lt;think&gt; blocks, which parse_score strips before scoring.")
    return fig


# ── 7. Per-condition table ───────────────────────────────────────────────────
def fig_table(df):
    rows = []
    for cond in conds_present(df):
        sub = df[df["mod_type"] == cond]
        row = {"Condition": COND_SHORT[cond], "n": len(sub)}
        for metric, label in LLM_METRICS:
            lo, hi, mn, _ = bootstrap_ci(sub[metric])
            row[label] = f"{mn*100:.1f}  [{lo*100:.1f}, {hi*100:.1f}]"
        row["Hedged %"] = f"{(sub['valid_conf'] == 'Hedged').mean()*100:.1f}"
        row["No lean %"] = f"{sub['valid_lean'].isna().mean()*100:.1f}"
        rows.append(row)
    tbl = pd.DataFrame(rows)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(tbl.columns), fill_color="#34495e",
                    font=dict(color="white", size=11), align="left"),
        cells=dict(values=[tbl[c] for c in tbl.columns],
                   fill_color="#f7f9fa", align="left", font=dict(size=11), height=24))])
    fig.update_layout(title="⑦ Per-condition summary (mean [95% CI])",
                      height=80 + 28 * len(tbl), margin=dict(l=20, r=20, t=60, b=20),
                      template=TEMPLATE)
    add(fig, "Summary table",
        "\"No lean %\" is the share of responses from which no yes/no could be read at "
        "all — those rows are excluded from panel ⑤ rather than counted as wrong.")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
def write_html(path, df, src_path):
    head = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Free generation on jul28 — pde-llm-eval</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        margin: 0; padding: 32px 40px; background: #fbfcfd; color: #1c2833; }}
 h1 {{ font-size: 26px; margin: 0 0 4px; }}
 .sub {{ color: #5d6d7e; font-size: 13px; margin-bottom: 26px; }}
 .card {{ background: #fff; border: 1px solid #e4e9ee; border-radius: 8px;
          padding: 14px 18px 6px; margin-bottom: 26px; }}
 .cap {{ font-size: 12.5px; color: #4a5b6b; line-height: 1.55;
         border-left: 3px solid #d6dde4; padding: 4px 0 4px 12px; margin: 2px 0 14px; }}
 code {{ background: #eef2f5; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
</style></head><body>
<h1>Experiment 1 — free generation on <code>merged_mod_jul28</code></h1>
<div class="sub">
 {len(df):,} rows &middot; {df['model'].nunique()} models &middot;
 {df['title'].nunique()} snippets &middot; {df['gt_sample'].nunique() if 'gt_sample' in df else '?'} base solvers &middot;
 input <code>{os.path.basename(src_path)}</code>
</div>
"""
    parts = [head]
    first = True
    for title, fig, caption in FIGS:
        parts.append('<div class="card">')
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
        parts.append(f'<div class="cap"><b>{title}.</b> {caption}</div>')
        parts.append("</div>")
        first = False
    parts.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="results/pde_llm_eval_jul28.csv")
    ap.add_argument("--output", default="viz/freegen_jul28_report.html")
    args = ap.parse_args()

    df = load(args.input)
    print(f"[report] {len(df)} rows, {df['model'].nunique()} models from {args.input}")
    if "finish_reason" in df.columns:
        n_trunc = int((df["finish_reason"] == "length").sum())
        if n_trunc:
            print(f"[report] WARNING: {n_trunc} truncated response(s) in this input — "
                  f"a 'length' finish_reason is a failed row, not a datum.")

    fig_score_by_condition(df)
    fig_hedge_breakdown(df)
    fig_by_source(df)
    fig_length_control(df)
    fig_hedge_calibration(df)
    fig_reasoning(df)
    fig_table(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_html(args.output, df, args.input)
    print(f"[report] Wrote {len([f for f in FIGS])} panels -> {args.output}")


if __name__ == "__main__":
    main()
