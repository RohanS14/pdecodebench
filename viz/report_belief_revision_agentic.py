"""
report_belief_revision_agentic.py — Agentic belief-revision experiment visualizations.

Reads:
  results/frontier/stratified_64/nothink/<slug>__belief_revision_agentic.jsonl
  results/frontier/stratified_64/think/<slug>__belief_revision_agentic.jsonl
  results/frontier/stratified_64/judge/judge_results.jsonl

Produces (Results 1-3 from the stratified-run plan; Result 4 is manual trace
inspection, not part of this script):
  Result 1 — 3-line accuracy-by-mod_type (S1 shared, S2 x thinking off/on)
  Result 2 — transition stacked bars, one per thinking condition
  Result 3a — turns used by mod_type, one per thinking condition
  Result 3b — judge none/some/all stacked bars by mod_type and by pde_class,
              plus a separate contains_incorrect_claims rate table
  Result 10 — recall vs specificity, aggregate and by mod_type, for S1 and
              both S2 thinking conditions (see agent_docs/hypotheses.md H1)

Reuses viz/report_belief_revision.py's bootstrap_ci/save/write_combined_html and
its MOD_ORDER/MOD_SHORT/PDE_ORDER/PDE_LABEL/FONT/LEGEND/MARGIN constants
directly rather than duplicating them.

Usage (both conditions):
  python viz/report_belief_revision_agentic.py \\
      --nothink results/frontier/stratified_64/nothink/gemini25flash__belief_revision_agentic.jsonl \\
      --think results/frontier/stratified_64/think/gemini25flash__belief_revision_agentic.jsonl \\
      --judge results/frontier/stratified_64/judge/judge_results.jsonl

Usage (nothink-only, e.g. before the think-condition sweep has run -- --think
and --judge are both optional; the report renders whatever's available):
  python viz/report_belief_revision_agentic.py \\
      --nothink results/frontier/stratified_64/nothink/gemini25flash__belief_revision_agentic.jsonl
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

import report_belief_revision as _vf  # noqa: E402 -- needed for the _VIZ_IDX rewind in v0_architecture_diagram
from report_belief_revision import (  # noqa: E402
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


# ── Result 0: system architecture schematic ──────────────────────────────────
# Hand-drawn SVG, not a Plotly chart -- box-and-arrow diagrams are precise
# layout problems Plotly's shapes/annotations API fights rather than helps
# with. Passed to save() as extra_html with fig=None (see report_belief_revision.py
# save()/write_combined_html(), extended to tolerate a figure-less section).
# Content approved in chat before drawing; see docs/superpowers/specs/
# 2026-08-16-agentic-belief-revision-design-v4.md for the full written design
# this diagram summarizes.
#
# Layout is cursor-based (box heights computed from content via _box_h(),
# never hand-guessed) and every line is width-checked at build time via
# _check_fits() -- an overflowing line raises immediately instead of silently
# rendering badly, so layout bugs surface here, not as a "looks bad" report.

_CHAR_W = 0.58  # average Arial character width as a fraction of font-size

def _check_fits(line: str, box_w: float, font_size: float, pad: float = 20) -> None:
    est_w = len(line) * font_size * _CHAR_W
    if est_w > box_w - pad:
        raise ValueError(
            f"V0 diagram line too wide for its box: {est_w:.0f}px estimated > "
            f"{box_w - pad:.0f}px available (box_w={box_w}, font={font_size}): {line!r}"
        )


def _box_h(has_title: bool, n_lines: int, title_size=14, body_size=12, pad=13) -> float:
    h = pad * 2
    if has_title:
        h += title_size + 6
    h += n_lines * (body_size + 5)
    return h


def _svg_box(x, y, w, title, lines, fill, stroke, dashed=False, title_size=14, body_size=12, pad=13):
    h = _box_h(bool(title), len(lines), title_size, body_size, pad)
    cx = x + w / 2
    if title:
        _check_fits(title, w, title_size, pad=16)
    for line in lines:
        _check_fits(line, w, body_size, pad=16)
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" ry="9" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"'
        + (' stroke-dasharray="6,4"' if dashed else '') + '/>'
    ]
    ty = y + pad + (title_size if title else body_size) * 0.8
    if title:
        parts.append(
            f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{title_size}" '
            f'font-weight="600" fill="#1a1a1a">{title}</text>'
        )
        ty += title_size + 6
    for line in lines:
        ty += body_size + 5
        parts.append(
            f'<text x="{cx}" y="{ty - (body_size + 5) + body_size}" text-anchor="middle" '
            f'font-size="{body_size}" fill="#3a3a3a">{line}</text>'
        )
    return "\n".join(parts), h


def _svg_arrow(x1, y1, x2, y2):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#777" '
        f'stroke-width="1.5" marker-end="url(#arrowhead)"/>'
    )


def _svg_note(cx, y, text, font_size=11.5):
    return (
        f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="{font_size}" '
        f'fill="#777" font-style="italic">{text}</text>'
    )


def v0_architecture_diagram(out_dir: Path) -> None:
    W = 900
    CX = W / 2
    GAP = 20         # vertical gap between stacked elements
    ARROW_LEN = 22
    y = 20           # layout cursor

    svg_parts = []

    def place(x, w, title, lines, fill, stroke, **kw):
        nonlocal y
        block, h = _svg_box(x, y, w, title, lines, fill, stroke, **kw)
        svg_parts.append(block)
        top = y
        y += h
        return top, h

    def arrow_down(gap=GAP):
        nonlocal y
        svg_parts.append(_svg_arrow(CX, y, CX, y + gap))
        y += gap

    # INPUT
    place(CX - 190, 380, "INPUT",
          ["PDE solver snippet + ground truth",
           "(pde class, method, process, validity)"],
          "#f4f4f6", "#666")
    arrow_down()

    # STAGE 1
    place(CX - 350, 700, "STAGE 1: static read-only judgment",
          ["Single turn, no tools, temperature 0",
           "Answers pde / method / behavior / valid (+ 1-line explanation each)",
           "Cached once per row, reused by both Stage 2 runs below"],
          "#eaf3fb", "#2980b9")
    arrow_down()

    # STAGE 2 outer container -- height computed from its own children below,
    # so the container is drawn LAST for this block (after we know the total),
    # but its top y is fixed now.
    stage2_top = y
    inner_x = CX - 370
    inner_w = 740
    y += 46  # room for container title + subtitle line

    turn_note_top, turn_note_h = place(
        inner_x, inner_w, "",
        ["Each turn, 1st attempt: model may reply with text and/or a tool call",
         "(encourages reasoning before acting). No tool call yet: empty reply",
         "gets one free retry, then the 2nd attempt must produce a tool call."],
        "#ffffff", "#aaaaaa", dashed=True, body_size=11, pad=9)
    y += GAP * 0.7

    tool_w = (inner_w - 2 * 16) / 3
    tools_top = y
    tool_specs = [
        ("edit_source", ["diff -> new file version,", "reruns full simulation", "counts vs budget"],
         "#eaf3fb", "#2980b9"),
        ("run_diagnostic", ["read-only script,", "no simulation rerun", "counts vs budget"],
         "#eaf3fb", "#2980b9"),
        ("submit_final_answer", ["always available,", "never counts vs budget"],
         "#f5eef8", "#8e44ad"),
    ]
    max_lines = max(len(t_lines) for _, t_lines, _, _ in tool_specs)
    tool_h = 0
    for i, (t_title, t_lines, t_fill, t_stroke) in enumerate(tool_specs):
        t_lines = t_lines + [""] * (max_lines - len(t_lines))  # pad so every tool box is the same height
        tx = inner_x + i * (tool_w + 16)
        block, tool_h = _svg_box(tx, tools_top, tool_w, t_title, t_lines, t_fill, t_stroke,
                                  title_size=12.5, body_size=10.8)
        svg_parts.append(block)
    y = tools_top + tool_h + GAP * 0.7

    svg_parts.append(_svg_note(CX, y, "Investigative budget: 6 actions total (edit_source + run_diagnostic)"))
    y += 17
    svg_parts.append(_svg_note(CX, y, "Also capped: per-turn output length, per-episode cost, disk-safety limits"))
    y += 15
    svg_parts.append(_svg_note(CX, y, "(disk-safety limit hit, rare: episode ends early, no answer scored)"))
    y += GAP

    stage2_h = y - stage2_top
    svg_parts.insert(0, _svg_box(CX - 380, stage2_top, 760, "STAGE 2: agentic investigation loop",
                                  [], "#f8f8fa", "#666")[0])
    svg_parts.insert(1, _svg_note(
        CX, stage2_top + 38,
        "run twice per row, thinking budget off/on, scored independently", font_size=11))
    # manually pad the container to the computed height (title box was drawn
    # with lines=[] so _svg_box gave it a minimal height; redraw at full size)
    svg_parts[0] = _svg_box(CX - 380, stage2_top, 760, "STAGE 2: agentic investigation loop",
                             [], "#f8f8fa", "#666")[0].replace(
        f'height="{_box_h(True, 0)}"', f'height="{stage2_h}"')

    arrow_down()

    # SUBMISSION + CONFIRMATION
    sub_top = y
    y += 40
    col_w = 350
    voluntary_top, voluntary_h = place(
        CX - 370, col_w, "Voluntary (budget remaining)",
        ["1st submit call: intercepted, not final yet",
         "Shown original code + Stage 1 answer, asked",
         "to reconfirm. 2nd submit call is final."],
        "#ffffff", "#8e44ad", body_size=10.8, title_size=12)
    y = voluntary_top  # reset cursor to draw the second column alongside
    forced_top, forced_h = place(
        CX + 20, col_w, "Forced (budget exhausted)",
        ["Only submit_final_answer remains",
         "Original code + Stage 1 answer shown first,",
         "before the model's one attempt. That call is final."],
        "#ffffff", "#8e44ad", body_size=10.8, title_size=12)
    y = max(voluntary_top + voluntary_h, forced_top + forced_h) + 24
    svg_parts.append(_svg_note(
        CX, y, "(guards against the model judging its own edited code instead of the original)"))
    y += 14
    sub_h = y - sub_top + 6
    svg_parts.insert(2, _svg_box(CX - 400, sub_top, 800, "SUBMISSION AND CONFIRMATION",
                                  [], "#f5eef8", "#8e44ad")[0].replace(
        f'height="{_box_h(True, 0)}"', f'height="{sub_h}"'))
    y += GAP
    arrow_down()

    # SCORING
    place(CX - 350, 700, "SCORING: against ground truth (same scorer as Stage 1)",
          ["valid_match / pde_match / method_any_match / behavior_any_match",
           "Recall on truly-invalid rows, specificity on truly-valid rows,",
           "per mod_type, compared across S1, S2 no-thinking, S2 thinking"],
          "#eafaf1", "#27ae60")
    arrow_down(gap=38)
    svg_parts.append(
        f'<text x="{CX + 16}" y="{y - 24}" text-anchor="start" font-size="11.5" '
        f'fill="#777" font-style="italic">only rows correctly called invalid</text>'
    )
    y += 6

    # JUDGE
    place(CX - 350, 700, "JUDGE: invalidity-reasoning quality (separate, stronger model)",
          ["Sees this row's code plus its valid counterpart",
           "(and a ground-truth name/comment reference for 3 obscured mod_types)",
           "Scores the model's own explanation: category none/some/all,",
           "and whether it also contains an incorrect claim"],
          "#fef5e7", "#f39c12", body_size=11)

    H = y + 20
    svg = "\n".join([
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;max-width:{W}px;background:#fff;">',
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#777"/></marker></defs>',
        *svg_parts,
        '</svg>',
    ])
    html = f'<div style="overflow-x:auto;padding:8px 0;">{svg}</div>'

    # Deliberately V0: this section is added before any other save() call, so
    # temporarily rewinding the shared _VIZ_IDX counter to -1 makes it "V0"
    # without shifting every subsequent section's number.
    _vf._VIZ_IDX[0] = -1
    save(None, "V0_architecture", out_dir,
         title="System Architecture",
         description="High-level schematic of the two-stage agentic pipeline: Stage 1 "
                      "(static judgment) feeds Stage 2 (agentic investigation, run under "
                      "both thinking conditions), which submits through a confirmation "
                      "step, gets scored against ground truth, and for correctly-flagged "
                      "invalid rows has its reasoning quality separately judged.",
         extra_html=html)


# ── Result 1: 3-line accuracy-by-mod_type ────────────────────────────────────

# Method/behavior (Panels C/D) turned out to be a negative, fairly flat
# result -- S1/S2-nothink/S2-think track each other closely for both metrics,
# unlike Panel A (validity), where the stages diverge a lot. Distracting more
# than informative as a default view, so hidden unless explicitly requested.
SHOW_METHOD_BEHAVIOR_PANELS = False


def v1_accuracy_by_modtype(df_nothink: pd.DataFrame, df_think: pd.DataFrame, out_dir: Path,
                            show_method_behavior: bool = SHOW_METHOD_BEHAVIOR_PANELS) -> None:
    """3 lines per panel: S1 (shared), S2 (thinking off), S2 (thinking on).
    Stage 1 is 100% independent of thinking_budget (confirmed by reading the
    harness code) -- it's cached once per row and reused for both Stage-2
    sweeps, not re-run under a "thinking on" condition. A separate "S1
    (thinking on)" line would therefore be misleading: it would imply Stage 1
    was ever executed with thinking enabled, when it never is regardless of
    which sweep it's paired with. Plotting one shared S1 line makes that
    structural fact explicit instead of implying something false.

    df_think may be empty (nothink-only run, no --think data yet) -- the "S2
    (thinking on)" series is simply dropped from the plot in that case (2
    lines instead of 3), rather than erroring.

    show_method_behavior=True adds Panels C/D (method/behavior any_match,
    2x2 grid); default False shows just Panels A/B (validity/pde, 1x2)."""
    # (subplot title, metric suffix used to build s1_<suffix>/s2_<suffix> column names)
    metrics = [
        ("Panel A — Validity Prediction (valid_match)", "valid_match"),
        ("Panel B — PDE Identification (pde_match) [control]", "pde_match"),
    ]
    if show_method_behavior:
        metrics += [
            ("Panel C — Method Identification (method_any_match)", "method_any_match"),
            ("Panel D — Behavior Identification (behavior_any_match)", "behavior_any_match"),
        ]
    n_rows = 2 if show_method_behavior else 1
    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=[t for t, _ in metrics],
        horizontal_spacing=0.12, vertical_spacing=0.16,
    )

    series = [
        ("S1 (shared)",       "s1", df_nothink, "#2980b9", "solid"),
        ("S2 (thinking off)", "s2", df_nothink, "#e74c3c", "dash"),
        ("S2 (thinking on)",  "s2", df_think,   "#f39c12", "dash"),
    ]
    series = [s for s in series if not s[2].empty]

    for panel_idx, (_, metric) in enumerate(metrics):
        row_idx, col_idx = panel_idx // 2 + 1, panel_idx % 2 + 1
        conds = None
        for name, stage_prefix, df, color, dash in series:
            col = f"{stage_prefix}_{metric}"
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
                showlegend=(panel_idx == 0),
            ), row=row_idx, col=col_idx)

        fig.update_yaxes(range=[0, 105], title_text="Accuracy (%)", row=row_idx, col=col_idx)
        fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                      annotation_text="chance", annotation_position="bottom right",
                      row=row_idx, col=col_idx)

    think_present = not df_think.empty
    title_suffix = "S1 (shared) vs S2, Thinking Off vs On" if think_present else "S1 (shared) vs S2, Thinking Off only (no --think data yet)"
    fig.update_layout(
        title=f"Accuracy by Mod Type: {title_suffix}",
        font=FONT, legend=LEGEND, margin=MARGIN,
        width=1300, height=(980 if show_method_behavior else 520),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=35)
    n_lines = 3 if think_present else 2
    desc = (
        f"{n_lines} lines per panel: S1 is a single shared line (Stage 1 is cached once per "
        "row and reused for both Stage-2 sweeps -- it never runs under a thinking "
        "condition, so there is deliberately no separate 'S1 (thinking on)' line). "
        f"Panel B (PDE identification) is a control and should stay flat across all {n_lines} lines."
    )
    if show_method_behavior:
        desc += (
            " Panels C and D (method/behavior) use partial-credit matching (correct if "
            "any predicted label matches any ground-truth label, since the model may "
            "name up to 3 candidates) -- not the same exact-match criterion as Panels A "
            "and B. Note: both turned out close to flat across all stages -- execution "
            "evidence barely moves method/behavior identification, unlike validity."
        )
    save(fig, "V1_accuracy_modtype_3line", out_dir,
         title=f"Accuracy by Mod Type: S1 (shared) vs S2, {'thinking off vs on' if think_present else 'thinking off only'}",
         description=desc)


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
VALID_MOD_TYPES = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar"]


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
             description="LLM-judge (gemini-3.1-pro-preview) classification of the model's valid_exp "
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


# ── Result 10: recall / specificity, aggregate and by mod_type ──────────────
# recall  = fraction of truly-invalid rows (gt_valid=False) the model correctly
#           calls invalid -- match_col==1 on those rows already means "correctly
#           identified as invalid", since match is scored against ground truth.
# specificity = fraction of truly-valid rows (gt_valid=True) the model correctly
#           calls valid -- match_col==1 on those rows already means "correctly
#           identified as valid". Each mod_type is homogeneous in gt_valid by
#           construction, so no extra valid/invalid-call parsing is needed here;
#           the existing s1_valid_match/s2_valid_match columns already encode
#           correctness against whichever ground truth that row has.

def _rate(df: pd.DataFrame, mod_types: list, match_col: str) -> tuple[float, int]:
    sub = df[df["mod_type"].isin(mod_types)][match_col].dropna()
    if len(sub) == 0:
        return float("nan"), 0
    return sub.mean() * 100, len(sub)


def v10_recall_specificity(df_nothink: pd.DataFrame, df_think: pd.DataFrame, out_dir: Path) -> None:
    """Aggregate (all 4 mod_types pooled) and mod_type-stratified recall/
    specificity, for S1 (shared), S2 nothink, and S2 think (if available).
    See agent_docs/hypotheses.md (H1) for the reasoning this panel visualizes:
    recall converges toward ceiling under agentic investigation regardless of
    mod_type, while specificity does not move (and can worsen) under the same
    investigation -- because a specific injected defect is a discrete target
    evidence can converge to, while "this code is fine" has no equivalent
    target for evidence to confirm."""
    think_present = not df_think.empty
    stages = [("S1 (shared)", df_nothink, "s1_valid_match", "#2980b9")]
    stages.append(("S2 (thinking off)", df_nothink, "s2_valid_match", "#e74c3c"))
    if think_present:
        stages.append(("S2 (thinking on)", df_think, "s2_valid_match", "#f39c12"))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Panel A — Recall (truly-invalid rows correctly called invalid)",
                        "Panel B — Specificity (truly-valid rows correctly called valid)"],
        horizontal_spacing=0.12,
    )

    for col_idx, mod_types in enumerate([INVALID_MOD_TYPES, VALID_MOD_TYPES], start=1):
        x_labels = ["Aggregate"] + [MOD_SHORT.get(m, m) for m in mod_types]
        for name, df, match_col, color in stages:
            ys, ns = [], []
            agg_rate, agg_n = _rate(df, mod_types, match_col)
            ys.append(agg_rate)
            ns.append(agg_n)
            for m in mod_types:
                rate, n = _rate(df, [m], match_col)
                ys.append(rate)
                ns.append(n)
            fig.add_trace(go.Bar(
                name=name, x=x_labels, y=ys,
                marker_color=color,
                text=[f"n={n}" for n in ns], textposition="outside",
                legendgroup=name, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        fig.update_yaxes(range=[0, 115], title_text="Rate (%)", row=1, col=col_idx)
        fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                      annotation_text="chance", annotation_position="bottom right",
                      row=1, col=col_idx)

    fig.update_layout(
        title="Recall vs Specificity: Aggregate and by Mod Type",
        barmode="group", font=FONT, legend=LEGEND, margin=MARGIN,
        width=1400, height=560,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=20)

    extra_html = (
        _pre_table(["Stage", "Recall", "Specificity"], _table1_rows(df_nothink, df_think),
                   caption="Table 1 — Aggregate recall / specificity by stage")
        + _pre_table(["Stage"] + [MOD_SHORT.get(m, m) for m in INVALID_MOD_TYPES],
                     _table2_rows(df_nothink, df_think, INVALID_MOD_TYPES),
                     caption="Table 2 — Recall by mod_type (truly-invalid rows only)")
        + _pre_table(["Stage"] + [MOD_SHORT.get(m, m) for m in VALID_MOD_TYPES],
                     _table2_rows(df_nothink, df_think, VALID_MOD_TYPES),
                     caption="Table 3 — Specificity by mod_type (truly-valid rows only)")
    )

    save(fig, "V10_recall_specificity", out_dir,
         title="Recall vs Specificity: Aggregate and by Mod Type",
         description="Recall (Panel A) = fraction of truly-invalid rows (gt_valid=False) "
                     "correctly called invalid, computed only over the 4 invalid mod_types. "
                     "Specificity (Panel B) = fraction of truly-valid rows (gt_valid=True) "
                     "correctly called valid, computed only over the 4 valid mod_types. "
                     "'Aggregate' pools all 4 mod_types on that panel's side; the remaining "
                     "bars split by individual mod_type (n=8 each in the 64-row stratified "
                     "sample -- a single row is worth ~12.5 percentage points, so per-mod_type "
                     "differences smaller than that shouldn't be over-read). See "
                     "agent_docs/hypotheses.md (H1) for the hypothesis this panel tests: "
                     "recall converges toward 100% under agentic investigation across every "
                     "mod_type, while specificity does not improve under the same investigation "
                     "and stays far lower specifically in the GT-variable-name conditions "
                     "(Comm_Valid, NoComm_Valid, CorrComm) than in the obfuscated-name "
                     "condition (NoComm_CorrVar). The three tables below the chart give the "
                     "exact N/n values behind every bar.",
         extra_html=extra_html)


def _fmt_rate(rate: float, n: int) -> str:
    if n == 0 or rate != rate:  # rate != rate catches NaN
        return "n/a"
    k = round(rate / 100 * n)
    return f"{k}/{n} ({rate:.1f}%)"


def _stage_rows(df_nothink: pd.DataFrame, df_think: pd.DataFrame) -> list:
    """(stage_label, df, match_col) for whichever stages are available."""
    stages = [("S1", df_nothink, "s1_valid_match"), ("S2 nothink", df_nothink, "s2_valid_match")]
    if not df_think.empty:
        stages.append(("S2 think", df_think, "s2_valid_match"))
    return stages


def _table1_rows(df_nothink: pd.DataFrame, df_think: pd.DataFrame) -> list:
    rows = []
    for stage_label, df, match_col in _stage_rows(df_nothink, df_think):
        rec_rate, rec_n = _rate(df, INVALID_MOD_TYPES, match_col)
        spec_rate, spec_n = _rate(df, VALID_MOD_TYPES, match_col)
        rows.append([stage_label, _fmt_rate(rec_rate, rec_n), _fmt_rate(spec_rate, spec_n)])
    return rows


def _table2_rows(df_nothink: pd.DataFrame, df_think: pd.DataFrame, mod_types: list) -> list:
    rows = []
    for stage_label, df, match_col in _stage_rows(df_nothink, df_think):
        row = [stage_label]
        for m in mod_types:
            rate, n = _rate(df, [m], match_col)
            row.append(_fmt_rate(rate, n))
        rows.append(row)
    return rows


def _pre_table(header: list, rows: list, caption: str = "") -> str:
    """Simple, well-formatted plain-text table (fixed-width columns), rendered
    as a monospace <pre> block -- deliberately not an interactive plotly
    Table widget, so it reads like the tables already discussed in text."""
    all_rows = [header] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header))]
    def fmt_row(r):
        return "  ".join(str(v).ljust(w) for v, w in zip(r, widths))
    lines = [fmt_row(header), "  ".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in rows]
    body = "\n".join(lines)
    cap_html = f'<div style="font-weight:600;margin:18px 0 4px;">{caption}</div>' if caption else ""
    return f'{cap_html}<pre style="font-family:ui-monospace,monospace;font-size:13px;line-height:1.5;">{body}</pre>'


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic belief-revision visualizations")
    p.add_argument("--nothink", required=True, help="Path to the thinking_budget=0 Stage-2 JSONL")
    p.add_argument("--think", default=None,
                   help="Path to the thinking_budget=1536 Stage-2 JSONL. Optional -- if omitted, "
                        "the report renders nothink-only versions of every result (2-line "
                        "accuracy plot instead of 4, no think-condition transitions/turns/judge "
                        "sections) rather than erroring.")
    p.add_argument("--judge", default="results/frontier/stratified_64/judge/judge_results.jsonl",
                   help="Path to judge_results.jsonl. Optional -- if the file doesn't exist yet, "
                        "judge sections are skipped (load_judge_results already handles this).")
    p.add_argument("--out_dir", default="results/frontier/stratified_64/viz")
    return p.parse_args()


def main():
    args = parse_args()

    df_nothink = load_agentic_results(Path(args.nothink))
    df_think = load_agentic_results(Path(args.think)) if args.think else pd.DataFrame()
    judge_df = load_judge_results(Path(args.judge))
    print(f"[viz-agentic] nothink: {len(df_nothink)} rows, think: {len(df_think)} rows, "
          f"judge: {len(judge_df)} rows")
    if df_think.empty:
        print("[viz-agentic] No --think data provided -- rendering nothink-only "
              "versions of every result (2-line accuracy plot, no think-condition "
              "transitions/turns sections).")

    out_dir = Path(args.out_dir)
    print(f"[viz-agentic] Output → {out_dir}\n")

    v0_architecture_diagram(out_dir)
    v1_accuracy_by_modtype(df_nothink, df_think, out_dir)
    v2_transitions_by_condition(df_nothink, out_dir, "thinking off", "nothink")
    v3a_turns_by_condition(df_nothink, out_dir, "thinking off", "nothink")
    if not df_think.empty:
        v2_transitions_by_condition(df_think, out_dir, "thinking on", "think")
        v3a_turns_by_condition(df_think, out_dir, "thinking on", "think")
    v3b_judge_by_modtype(judge_df, out_dir)
    v3b_judge_by_pdeclass(judge_df, out_dir)
    v10_recall_specificity(df_nothink, df_think, out_dir)

    incorrect_claims = contains_incorrect_claims_table(judge_df)
    if not incorrect_claims.empty:
        print("\n[viz-agentic] contains_incorrect_claims rate by thinking condition:")
        print(incorrect_claims.to_string(index=False))
        incorrect_claims.to_csv(out_dir / "contains_incorrect_claims_rate.csv", index=False)

    # Derive the report slug from the output directory's parent (e.g.
    # results/frontier/stratified_256/viz -> "stratified_256") rather than a
    # hardcoded "stratified_64", so the report filename tracks whichever
    # dataset scale it was actually generated from.
    slug = f"agentic_{out_dir.resolve().parent.name}"
    write_combined_html(out_dir, slug)

    print("\n[viz-agentic] Done.")


if __name__ == "__main__":
    main()
