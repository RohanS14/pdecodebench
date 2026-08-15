"""
visualize_agentic.py — Agentic belief-revision experiment visualizations.

Reads:
  results/frontier/stratified_64/nothink/<slug>__belief_revision_agentic.jsonl
  results/frontier/stratified_64/think/<slug>__belief_revision_agentic.jsonl
  results/frontier/stratified_64/judge/judge_results.jsonl

Produces (Results 1-3 from the stratified-run plan; Result 4 is manual trace
inspection, not part of this script):
  Result 1 — 4-line accuracy-by-mod_type (S1/S2 x thinking off/on)
  Result 2 — transition stacked bars, one per thinking condition
  Result 3a — turns used by mod_type, one per thinking condition
  Result 3b — judge none/some/all stacked bars by mod_type and by pde_class,
              plus a separate contains_incorrect_claims rate table

Reuses viz/visualize_frontier.py's bootstrap_ci/save/write_combined_html and
its MOD_ORDER/MOD_SHORT/PDE_ORDER/PDE_LABEL/FONT/LEGEND/MARGIN constants
directly rather than duplicating them.

Usage:
  python viz/visualize_agentic.py \\
      --nothink results/frontier/stratified_64/nothink/gemini25flash__belief_revision_agentic.jsonl \\
      --think results/frontier/stratified_64/think/gemini25flash__belief_revision_agentic.jsonl \\
      --judge results/frontier/stratified_64/judge/judge_results.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualize_frontier import (  # noqa: E402
    FONT,
    LEGEND,
    MARGIN,
    MOD_ORDER,
    MOD_SHORT,
    PDE_LABEL,
    PDE_ORDER,
    TRANS_COLORS,
    TRANS_ORDER,
    TRANS_SHORT,
    bootstrap_ci,
    save,
    write_combined_html,
)


def load_agentic_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    df["mod_type"] = df["mod_type"].str.strip()
    df["gt_pde"] = df["gt_pde"].str.strip()
    return df


def load_judge_results(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["title", "thinking_budget", "mod_type", "pde_class",
                                      "category", "contains_incorrect_claims", "explanation"])
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


# ── Result 1: 4-line accuracy-by-mod_type ────────────────────────────────────

def v1_accuracy_by_modtype_4line(df_nothink: pd.DataFrame, df_think: pd.DataFrame, out_dir: Path) -> None:
    """4 lines per panel: S1 (thinking off), S2 (thinking off), S1 (thinking
    on), S2 (thinking on). The two S1 lines will be numerically identical --
    Stage 1 is 100% independent of thinking_budget (confirmed by reading the
    harness code) -- plotted anyway, exactly as requested, since the overlap
    is itself a correct, honest confirmation that Stage 1 is unaffected by
    the sweep variable, not a redundant chart."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Panel A — Validity Prediction (valid_match)",
                        "Panel B — PDE Identification (pde_match) [control]"],
        horizontal_spacing=0.12,
    )

    series = [
        ("S1 (thinking off)", "s1_valid_match", "s1_pde_match", df_nothink, "#2980b9", "solid"),
        ("S2 (thinking off)", "s2_valid_match", "s2_pde_match", df_nothink, "#e74c3c", "dash"),
        ("S1 (thinking on)",  "s1_valid_match", "s1_pde_match", df_think,   "#8e44ad", "solid"),
        ("S2 (thinking on)",  "s2_valid_match", "s2_pde_match", df_think,   "#f39c12", "dash"),
    ]

    for col_idx, metric_kind in enumerate(["valid_match", "pde_match"], start=1):
        conds = None
        for name, valid_col, pde_col, df, color, dash in series:
            col = valid_col if metric_kind == "valid_match" else pde_col
            if conds is None:
                conds = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
            labels = [MOD_SHORT.get(c, c) for c in conds]
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
                name=name,
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=8),
                error_y=dict(type="data", symmetric=False, array=hi_errs, arrayminus=lo_errs,
                             thickness=1.5, width=4),
                legendgroup=name,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        fig.update_yaxes(range=[0, 105], title_text="Accuracy (%)", row=1, col=col_idx)
        fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                      annotation_text="chance", annotation_position="bottom right",
                      row=1, col=col_idx)

    fig.update_layout(
        title="Accuracy by Mod Type: Stage 1 vs Stage 2, Thinking Off vs On",
        font=FONT, legend=LEGEND, margin=MARGIN,
        width=1300, height=520,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=35)
    save(fig, "V1_accuracy_modtype_4line", out_dir,
         title="Accuracy by Mod Type: S1 vs S2, thinking off vs on",
         description="4 lines per panel. The two S1 lines are expected to overlap exactly -- "
                     "Stage 1 doesn't depend on thinking_budget at all, so this confirms that "
                     "invariant rather than being a redundant chart. Panel B (PDE identification) "
                     "is a control and should stay flat across all 4 lines.")


# ── Result 2: transition stacked bars, one per thinking condition ───────────

def v2_transitions_by_condition(df: pd.DataFrame, out_dir: Path, condition_label: str, name_suffix: str) -> None:
    conds = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
    labels = [MOD_SHORT.get(c, c) for c in conds]

    fig = go.Figure()
    for trans in TRANS_ORDER:
        ys = []
        for cond in conds:
            sub = df[df["mod_type"] == cond]
            ys.append((sub["transition"] == trans).sum())
        total_n = sum(ys)
        legend_name = f"{TRANS_SHORT[trans]} (n=0)" if total_n == 0 else TRANS_SHORT[trans]
        fig.add_trace(go.Bar(
            name=legend_name, x=labels, y=ys,
            marker_color=TRANS_COLORS[trans],
            visible=True if total_n > 0 else "legendonly",
        ))

    for i, cond in enumerate(conds):
        sub = df[df["mod_type"] == cond]
        d = (sub["s2_valid_match"] - sub["s1_valid_match"]).dropna()
        _, mean, _ = bootstrap_ci(d)
        color = "#1a7a3c" if mean >= 0 else "#a93226"
        fig.add_annotation(
            x=labels[i], y=len(sub) + 0.5,
            text=f"Δ{mean*100:+.0f}pp", showarrow=False,
            font=dict(size=11, color=color), yanchor="bottom",
        )

    fig.update_layout(
        title=f"Transition Counts by Mod Type — {condition_label}",
        barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
        width=1100, height=520, yaxis_title="Count",
        plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=35,
    )
    save(fig, f"V2_transitions_{name_suffix}", out_dir,
         title=f"Transitions — {condition_label}: Counts by Mod Type",
         description="Stacked transition counts per mod_type for this thinking condition. "
                     "R→R: correct both stages. R→W: degraded. W→R: improved. W→W: wrong both "
                     "stages. Δ labels show S2−S1 accuracy change in percentage points.")


# ── Result 3a: turns used by condition ───────────────────────────────────────

def v3a_turns_by_condition(df: pd.DataFrame, out_dir: Path, condition_label: str, name_suffix: str) -> None:
    conds = [m for m in MOD_ORDER if m in df["mod_type"].unique()]
    labels = [MOD_SHORT.get(c, c) for c in conds]

    fig = go.Figure()
    ys, lo_errs, hi_errs = [], [], []
    for cond in conds:
        sub = df[df["mod_type"] == cond]["s2_action_count"].dropna()
        lo, mean, hi = bootstrap_ci(sub)
        ys.append(mean)
        lo_errs.append(mean - lo)
        hi_errs.append(hi - mean)
    fig.add_trace(go.Bar(
        x=labels, y=ys, marker_color="#3a7bdd",
        error_y=dict(type="data", symmetric=False, array=hi_errs, arrayminus=lo_errs),
    ))
    fig.update_layout(
        title=f"Investigative Actions Used by Mod Type — {condition_label}",
        font=FONT, margin=MARGIN, width=1000, height=480,
        yaxis_title="Mean s2_action_count", plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=35,
    )
    save(fig, f"V3a_turns_{name_suffix}", out_dir,
         title=f"Turns used — {condition_label}",
         description="Mean investigative action count (edit_source/run_diagnostic; "
                     "submit_final_answer never counts) per mod_type, with bootstrap 95% CI.")


# ── Result 3b: judge none/some/all stacked bars ──────────────────────────────

CATEGORY_ORDER = ["none", "some", "all"]
CATEGORY_COLORS = {"none": "#c0392b", "some": "#f39c12", "all": "#27ae60"}
INVALID_MOD_TYPES = ["Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid"]


def v3b_judge_by_modtype(judge_df: pd.DataFrame, out_dir: Path) -> None:
    if judge_df.empty:
        print("  [viz-agentic] No judge results -- skipping V3b mod_type chart.")
        return
    for thinking_budget in sorted(judge_df["thinking_budget"].unique()):
        sub_tb = judge_df[judge_df["thinking_budget"] == thinking_budget]
        condition_label = "thinking off" if thinking_budget == 0 else f"thinking on (budget={thinking_budget})"
        conds = [m for m in INVALID_MOD_TYPES if m in sub_tb["mod_type"].unique()]
        labels = [MOD_SHORT.get(c, c) for c in conds]

        fig = go.Figure()
        for cat in CATEGORY_ORDER:
            ys = [(sub_tb[sub_tb["mod_type"] == c]["category"] == cat).sum() for c in conds]
            fig.add_trace(go.Bar(name=cat, x=labels, y=ys, marker_color=CATEGORY_COLORS[cat]))

        fig.update_layout(
            title=f"Invalidity-Reasoning Correctness by Mod Type — {condition_label}",
            barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
            width=1000, height=480, yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=35,
        )
        suffix = "nothink" if thinking_budget == 0 else "think"
        save(fig, f"V3b_judge_modtype_{suffix}", out_dir,
             title=f"Invalidity-reasoning correctness by mod_type — {condition_label}",
             description="LLM-judge (gemini-2.5-pro) classification of the model's valid_exp "
                         "justification, restricted to rows where the model correctly classified "
                         "the code as invalid (gt_valid=False AND s2_valid_match=1). 'none': no "
                         "genuine reason identified. 'some': at least one real, verifiable reason "
                         "(not necessarily all). 'all': every distinct injected discrepancy "
                         "accounted for. This is a recall-only metric -- see contains_incorrect_claims "
                         "separately for precision.")


def v3b_judge_by_pdeclass(judge_df: pd.DataFrame, out_dir: Path) -> None:
    if judge_df.empty:
        print("  [viz-agentic] No judge results -- skipping V3b pde_class chart.")
        return
    for thinking_budget in sorted(judge_df["thinking_budget"].unique()):
        sub_tb = judge_df[judge_df["thinking_budget"] == thinking_budget]
        condition_label = "thinking off" if thinking_budget == 0 else f"thinking on (budget={thinking_budget})"
        conds = [p for p in PDE_ORDER if p in sub_tb["pde_class"].unique()]
        labels = [PDE_LABEL.get(c, c) for c in conds]

        fig = go.Figure()
        for cat in CATEGORY_ORDER:
            ys = [(sub_tb[sub_tb["pde_class"] == c]["category"] == cat).sum() for c in conds]
            fig.add_trace(go.Bar(name=cat, x=labels, y=ys, marker_color=CATEGORY_COLORS[cat]))

        fig.update_layout(
            title=f"Invalidity-Reasoning Correctness by PDE Class — {condition_label}",
            barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
            width=800, height=480, yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        suffix = "nothink" if thinking_budget == 0 else "think"
        save(fig, f"V3b_judge_pdeclass_{suffix}", out_dir,
             title=f"Invalidity-reasoning correctness by PDE class — {condition_label}",
             description="Same none/some/all judge classification as the mod_type view, "
                         "aggregated by PDE class instead.")


def contains_incorrect_claims_table(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Separate reportable rate, NOT folded into the none/some/all charts --
    category is a recall-only metric with no precision sensitivity, this
    table is the only place the precision signal (shotgun-reasoning rate)
    is visible."""
    if judge_df.empty:
        return pd.DataFrame(columns=["thinking_budget", "n", "contains_incorrect_claims_rate"])
    rows = []
    for thinking_budget, sub in judge_df.groupby("thinking_budget"):
        rate = sub["contains_incorrect_claims"].mean()
        rows.append({"thinking_budget": thinking_budget, "n": len(sub),
                     "contains_incorrect_claims_rate": round(rate, 3)})
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic belief-revision visualizations")
    p.add_argument("--nothink", required=True, help="Path to the thinking_budget=0 Stage-2 JSONL")
    p.add_argument("--think", required=True, help="Path to the thinking_budget=1536 Stage-2 JSONL")
    p.add_argument("--judge", required=True, help="Path to judge_results.jsonl")
    p.add_argument("--out_dir", default="results/frontier/stratified_64/viz")
    return p.parse_args()


def main():
    args = parse_args()

    df_nothink = load_agentic_results(Path(args.nothink))
    df_think = load_agentic_results(Path(args.think))
    judge_df = load_judge_results(Path(args.judge))
    print(f"[viz-agentic] nothink: {len(df_nothink)} rows, think: {len(df_think)} rows, "
          f"judge: {len(judge_df)} rows")

    out_dir = Path(args.out_dir)
    print(f"[viz-agentic] Output → {out_dir}\n")

    v1_accuracy_by_modtype_4line(df_nothink, df_think, out_dir)
    v2_transitions_by_condition(df_nothink, out_dir, "thinking off", "nothink")
    v2_transitions_by_condition(df_think, out_dir, "thinking on", "think")
    v3a_turns_by_condition(df_nothink, out_dir, "thinking off", "nothink")
    v3a_turns_by_condition(df_think, out_dir, "thinking on", "think")
    v3b_judge_by_modtype(judge_df, out_dir)
    v3b_judge_by_pdeclass(judge_df, out_dir)

    incorrect_claims = contains_incorrect_claims_table(judge_df)
    if not incorrect_claims.empty:
        print("\n[viz-agentic] contains_incorrect_claims rate by thinking condition:")
        print(incorrect_claims.to_string(index=False))
        incorrect_claims.to_csv(out_dir / "contains_incorrect_claims_rate.csv", index=False)

    write_combined_html(out_dir, "agentic_stratified_64")

    print("\n[viz-agentic] Done.")


if __name__ == "__main__":
    main()
