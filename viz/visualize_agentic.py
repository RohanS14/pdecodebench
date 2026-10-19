"""
visualize_agentic.py — Agentic belief-revision experiment visualizations.

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

Reuses viz/visualize_frontier.py's bootstrap_ci/save/write_combined_html and
its MOD_ORDER/MOD_SHORT/PDE_ORDER/PDE_LABEL/FONT/LEGEND/MARGIN constants
directly rather than duplicating them.

Usage (both conditions):
  python viz/visualize_agentic.py \\
      --nothink results/frontier/stratified_64/nothink/gemini25flash__belief_revision_agentic.jsonl \\
      --think results/frontier/stratified_64/think/gemini25flash__belief_revision_agentic.jsonl \\
      --judge results/frontier/stratified_64/judge/judge_results.jsonl

Usage (nothink-only, e.g. before the think-condition sweep has run -- --think
and --judge are both optional; the report renders whatever's available):
  python viz/visualize_agentic.py \\
      --nothink results/frontier/stratified_64/nothink/gemini25flash__belief_revision_agentic.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval" / "frontier"))

REPO_ROOT = Path(__file__).parent.parent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import visualize_frontier as _vf  # noqa: E402 -- needed for the _VIZ_IDX rewind in v0_architecture_diagram
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
    return with_condition_column(df)


def load_judge_results(path: Path) -> pd.DataFrame:
    base_cols = ["title", "condition", "thinking_budget", "mod_type",
                 "pde_class", "category", "explanation"]
    if not Path(path).exists():
        return pd.DataFrame(columns=base_cols + [c for c, _ in PRECISION_FIELDS])
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = with_condition_column(pd.DataFrame(rows))
    # Guarantee every precision column exists, so a judge run from either side
    # of the field split does not KeyError downstream. Absent ones stay all-NA
    # and are dropped from the table by the notna() check.
    for col, _ in PRECISION_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


# ── Condition identity ───────────────────────────────────────────────────────
# Every label in this report used to be hardcoded to the original two-condition
# design ("thinking off" / "thinking on"), which silently mislabels any other
# condition -- a thinking_level=high run would be captioned "thinking off".
# Labels are now derived from the data instead.
#
# `condition` is the field going forward (the only one that can name a Gemini 3
# condition); older 2.5 rows predate it and carry only a numeric
# thinking_budget, so that is the fallback. Same precedence as
# judge_invalidity_reasoning._condition_of().

def with_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a non-null `condition` column, deriving it from
    thinking_budget for rows written before `condition` existed.

    Non-null matters: several judge panels groupby() this column, and pandas
    silently DROPS None/NaN group keys -- which made the whole
    contains_incorrect_claims section vanish from the report rather than error.
    """
    if df.empty:
        return df
    df = df.copy()
    if "condition" not in df.columns:
        df["condition"] = None
    def _fill(row):
        c = row.get("condition")
        if c is not None and c == c and str(c).strip():
            return str(c)
        tb = row.get("thinking_budget")
        if tb is not None and tb == tb:
            return f"tb{int(tb)}"
        return "unknown"
    df["condition"] = df.apply(_fill, axis=1)
    return df


def condition_label(cond: str) -> str:
    """Human-readable form of a condition slug, for titles and captions."""
    if cond.startswith("tl_"):
        return f"thinking_level={cond[3:]}"
    if cond == "tb_dynamic":
        return "thinking_budget=dynamic"
    if cond.startswith("tb"):
        return f"thinking_budget={cond[2:]}"
    return cond


def df_condition_label(df: pd.DataFrame) -> str:
    """Label for a single-condition Stage-2 frame. Falls back to a generic name
    rather than guessing if a frame somehow mixes conditions."""
    if df.empty:
        return "no data"
    conds = sorted(set(df["condition"])) if "condition" in df.columns else []
    if len(conds) == 1:
        return condition_label(conds[0])
    return "mixed conditions" if conds else "unknown condition"


# ── Result 0: system architecture schematic ──────────────────────────────────
# Hand-drawn SVG, not a Plotly chart -- box-and-arrow diagrams are precise
# layout problems Plotly's shapes/annotations API fights rather than helps
# with. Passed to save() as extra_html with fig=None (see visualize_frontier.py
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


def v1_accuracy_by_modtype(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path,
                            show_method_behavior: bool = SHOW_METHOD_BEHAVIOR_PANELS) -> None:
    """One S1 line per condition plus one S2 line per condition.

    IMPORTANT -- this changed. Under the original design Stage 1 was run once at
    thinking_budget=0 and reused across both Stage-2 sweeps, so a single shared
    S1 line was correct and a per-condition S1 line would have implied something
    false. Stage 1 now runs at the SAME thinking setting as the Stage 2 it feeds,
    so S1 and S2 differ only in tool access -- which is the whole point, since
    otherwise "does investigation help" would confound tool access with thinking.
    S1 is therefore labelled with its condition, not as "shared".

    Caching Stage 1 still matters, for a different reason than before: on Gemini
    3 nothing is deterministic (temperature is stripped and is not honoured
    anyway), so the cache is what guarantees every Stage-2 row sees the same
    Stage-1 draw, rather than being reuse of an identical answer.

    df_b may be empty (single-condition run) -- its series are dropped from the
    plot rather than erroring.

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

    lab_a, lab_b = df_condition_label(df_a), df_condition_label(df_b)
    series = [
        (f"S1 ({lab_a})", "s1", df_a, "#2980b9", "solid"),
        (f"S2 ({lab_a})", "s2", df_a, "#e74c3c", "dash"),
    ]
    if not df_b.empty:
        series += [(f"S1 ({lab_b})", "s1", df_b, "#7f8c8d", "solid"),
                   (f"S2 ({lab_b})", "s2", df_b, "#f39c12", "dash")]
    series = [x for x in series if not x[2].empty]

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

    think_present = not df_b.empty
    title_suffix = (f"S1 vs S2, {lab_a} vs {lab_b}" if think_present
                    else f"S1 vs S2, {lab_a}")
    fig.update_layout(
        title=f"Accuracy by Mod Type: {title_suffix}",
        font=FONT, legend=LEGEND, margin=MARGIN,
        width=1300, height=(980 if show_method_behavior else 520),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=35)
    n_lines = len(series)
    desc = (
        f"{n_lines} lines per panel. Stage 1 runs at the SAME thinking setting as the Stage 2 "
        "it feeds, so S1 and S2 differ only in tool access -- that is deliberate: it keeps "
        "'does investigation help' from confounding tool access with thinking level. (Under the "
        "original design Stage 1 always ran at thinking_budget=0 and was shared across both "
        "Stage-2 sweeps; that is no longer the case, so S1 is labelled per condition.) "
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
    save(fig, "V1_accuracy_modtype", out_dir,
         title=f"Accuracy by Mod Type: {title_suffix}",
         description=desc)


# ── Result 2: transition stacked bars, one per thinking condition ───────────

def v2_transitions_by_condition(df: pd.DataFrame, out_dir: Path, cond_label: str, name_suffix: str) -> None:
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
        title=f"Transition Counts by Mod Type — {cond_label}",
        barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
        width=1100, height=520, yaxis_title="Count",
        plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=35,
    )
    save(fig, f"V2_transitions_{name_suffix}", out_dir,
         title=f"Transitions — {cond_label}: Counts by Mod Type",
         description="Stacked transition counts per mod_type for this thinking condition. "
                     "R→R: correct both stages. R→W: degraded. W→R: improved. W→W: wrong both "
                     "stages. Δ labels show S2−S1 accuracy change in percentage points.")


# ── Result 3a: turns used by condition ───────────────────────────────────────

def v3a_turns_by_condition(df: pd.DataFrame, out_dir: Path, cond_label: str, name_suffix: str) -> None:
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
        title=f"Investigative Actions Used by Mod Type — {cond_label}",
        font=FONT, margin=MARGIN, width=1000, height=480,
        yaxis_title="Mean s2_action_count", plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=35,
    )
    save(fig, f"V3a_turns_{name_suffix}", out_dir,
         title=f"Turns used — {cond_label}",
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
    for cond in sorted(judge_df["condition"].unique()):
        sub_tb = judge_df[judge_df["condition"] == cond]
        cond_label = condition_label(cond)
        conds = [m for m in INVALID_MOD_TYPES if m in sub_tb["mod_type"].unique()]
        labels = [MOD_SHORT.get(c, c) for c in conds]

        fig = go.Figure()
        for cat in CATEGORY_ORDER:
            ys = [(sub_tb[sub_tb["mod_type"] == c]["category"] == cat).sum() for c in conds]
            fig.add_trace(go.Bar(name=cat, x=labels, y=ys, marker_color=CATEGORY_COLORS[cat]))

        fig.update_layout(
            title=f"Invalidity-Reasoning Correctness by Mod Type — {cond_label}",
            barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
            width=1000, height=480, yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=35,
        )
        suffix = cond
        save(fig, f"V3b_judge_modtype_{suffix}", out_dir,
             title=f"Invalidity-reasoning correctness by mod_type — {cond_label}",
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
    for cond in sorted(judge_df["condition"].unique()):
        sub_tb = judge_df[judge_df["condition"] == cond]
        cond_label = condition_label(cond)
        conds = [p for p in PDE_ORDER if p in sub_tb["pde_class"].unique()]
        labels = [PDE_LABEL.get(c, c) for c in conds]

        fig = go.Figure()
        for cat in CATEGORY_ORDER:
            ys = [(sub_tb[sub_tb["pde_class"] == c]["category"] == cat).sum() for c in conds]
            fig.add_trace(go.Bar(name=cat, x=labels, y=ys, marker_color=CATEGORY_COLORS[cat]))

        fig.update_layout(
            title=f"Invalidity-Reasoning Correctness by PDE Class — {cond_label}",
            barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
            width=800, height=480, yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        suffix = cond
        save(fig, f"V3b_judge_pdeclass_{suffix}", out_dir,
             title=f"Invalidity-reasoning correctness by PDE class — {cond_label}",
             description="Same none/some/all judge classification as the mod_type view, "
                         "aggregated by PDE class instead.")


PRECISION_FIELDS = [
    # (column, output name). The judge's precision signal was split: a single
    # `contains_incorrect_claims` defined as "extraneous, speculative, or
    # factually wrong" fired mostly on runtime figures the judge could not
    # check, not on anything wrong about the code. Both new fields are reported
    # separately; the legacy merged field is still read so older judge runs
    # render rather than showing a blank table.
    ("contains_incorrect_code_claims", "bad_code_claims_rate"),
    ("contains_unverifiable_execution_claims", "unverifiable_exec_claims_rate"),
    ("contains_incorrect_claims", "legacy_merged_rate"),
]


def contains_incorrect_claims_table(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Separate reportable rates, NOT folded into the none/some/all charts --
    category is a recall-only metric with no precision sensitivity, this
    table is the only place the precision signal (shotgun-reasoning rate)
    is visible."""
    if judge_df.empty:
        return pd.DataFrame(columns=["condition", "n"] + [o for _, o in PRECISION_FIELDS])
    # Grouped on `condition`, not thinking_budget: the latter is None on Gemini 3
    # rows, and pandas groupby DROPS None keys, so this table silently came back
    # empty -- the precision signal vanished from the report without erroring.
    # with_condition_column() guarantees the key is non-null.
    rows = []
    for cond, sub in judge_df.groupby("condition"):
        entry = {"condition": cond, "n": len(sub)}
        for col, out in PRECISION_FIELDS:
            if col in sub.columns and sub[col].notna().any():
                entry[out] = round(sub[col].mean(), 3)
        rows.append(entry)
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


def v10_recall_specificity(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path) -> None:
    """Aggregate (all 4 mod_types pooled) and mod_type-stratified recall/
    specificity, for S1 (shared), S2 nothink, and S2 think (if available).
    See agent_docs/hypotheses.md (H1) for the reasoning this panel visualizes:
    recall converges toward ceiling under agentic investigation regardless of
    mod_type, while specificity does not move (and can worsen) under the same
    investigation -- because a specific injected defect is a discrete target
    evidence can converge to, while "this code is fine" has no equivalent
    target for evidence to confirm."""
    think_present = not df_b.empty
    lab_a, lab_b = df_condition_label(df_a), df_condition_label(df_b)
    stages = [(f"S1 ({lab_a})", df_a, "s1_valid_match", "#2980b9"),
              (f"S2 ({lab_a})", df_a, "s2_valid_match", "#e74c3c")]
    if think_present:
        stages += [(f"S1 ({lab_b})", df_b, "s1_valid_match", "#7f8c8d"),
                   (f"S2 ({lab_b})", df_b, "s2_valid_match", "#f39c12")]

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
        _pre_table(["Stage", "Recall", "Specificity"], _table1_rows(df_a, df_b),
                   caption="Table 1 — Aggregate recall / specificity by stage")
        + _pre_table(["Stage"] + [MOD_SHORT.get(m, m) for m in INVALID_MOD_TYPES],
                     _table2_rows(df_a, df_b, INVALID_MOD_TYPES),
                     caption="Table 2 — Recall by mod_type (truly-invalid rows only)")
        + _pre_table(["Stage"] + [MOD_SHORT.get(m, m) for m in VALID_MOD_TYPES],
                     _table2_rows(df_a, df_b, VALID_MOD_TYPES),
                     caption="Table 3 — Specificity by mod_type (truly-valid rows only)")
    )

    # Per-mod_type n is DERIVED, never hardcoded: this caption previously stated
    # the 64-row figures ("n=8 each ... ~12.5 percentage points") and silently
    # went stale when the same script was pointed at the 256-row sample, telling
    # the reader to ignore differences 4x larger than the real resolution.
    _mod_ns = sorted({n for mods in (INVALID_MOD_TYPES, VALID_MOD_TYPES)
                      for m in mods for _, n in [_rate(df_a, [m], "s2_valid_match")] if n})
    if not _mod_ns:
        _res = "per-mod_type sample sizes vary -- see the tables below for exact n"
    elif len(set(_mod_ns)) == 1:
        _n = _mod_ns[0]
        _res = (f"n={_n} each in this sample -- a single row is worth "
                f"~{100/_n:.1f} percentage points, so per-mod_type differences smaller "
                f"than that shouldn't be over-read")
    else:
        _res = (f"n ranges {min(_mod_ns)}-{max(_mod_ns)} across mod_types -- a single row is "
                f"worth ~{100/max(_mod_ns):.1f}-{100/min(_mod_ns):.1f} percentage points, so "
                f"per-mod_type differences smaller than that shouldn't be over-read")

    save(fig, "V10_recall_specificity", out_dir,
         title="Recall vs Specificity: Aggregate and by Mod Type",
         description="Recall (Panel A) = fraction of truly-invalid rows (gt_valid=False) "
                     "correctly called invalid, computed only over the 4 invalid mod_types. "
                     "Specificity (Panel B) = fraction of truly-valid rows (gt_valid=True) "
                     "correctly called valid, computed only over the 4 valid mod_types. "
                     "'Aggregate' pools all 4 mod_types on that panel's side; the remaining "
                     f"bars split by individual mod_type ({_res}). See "
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


def _stage_rows(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list:
    """(stage_label, df, match_col) for whichever stages are available. Labels
    carry the actual condition rather than the historical nothink/think names."""
    lab_a = df_condition_label(df_a)
    stages = [(f"S1 ({lab_a})", df_a, "s1_valid_match"),
              (f"S2 ({lab_a})", df_a, "s2_valid_match")]
    if not df_b.empty:
        lab_b = df_condition_label(df_b)
        stages += [(f"S1 ({lab_b})", df_b, "s1_valid_match"),
                   (f"S2 ({lab_b})", df_b, "s2_valid_match")]
    return stages


def _table1_rows(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list:
    rows = []
    for stage_label, df, match_col in _stage_rows(df_a, df_b):
        rec_rate, rec_n = _rate(df, INVALID_MOD_TYPES, match_col)
        spec_rate, spec_n = _rate(df, VALID_MOD_TYPES, match_col)
        rows.append([stage_label, _fmt_rate(rec_rate, rec_n), _fmt_rate(spec_rate, spec_n)])
    return rows


def _table2_rows(df_a: pd.DataFrame, df_b: pd.DataFrame, mod_types: list) -> list:
    rows = []
    for stage_label, df, match_col in _stage_rows(df_a, df_b):
        row = [stage_label]
        for m in mod_types:
            rate, n = _rate(df, [m], match_col)
            row.append(_fmt_rate(rate, n))
        rows.append(row)
    return rows


def v11_sign_flip_naming_rate(judge_df: pd.DataFrame, out_dir: Path) -> None:
    """Reuses eval/frontier/sign_flip_detail_csv.py's manual, diff-based
    CLASSIFICATION (see agent_docs/sign_flip_classification.txt) rather than
    duplicating it -- this panel does not re-derive the ground-truth defect
    labels, only re-slices the existing judge verdicts by them."""
    if judge_df.empty:
        return
    from sign_flip_detail_csv import CLASSIFICATION  # noqa: E402

    dataset = pd.read_csv(REPO_ROOT / "data" / "merged_mod_jul28.csv")
    title_to_gt_sample = dict(zip(dataset["title"], dataset["gt_sample"]))

    df = judge_df.copy()
    df["gt_sample"] = df["title"].map(title_to_gt_sample)
    df["defect_type"] = df["gt_sample"].map(lambda g: CLASSIFICATION.get(g, ("UNKNOWN", ""))[0])

    categories = ["SIGN FLIP", "MIXED", "OTHER"]
    x_labels = categories + ["ALL (pooled)"]
    # Derived from the data, not hardcoded to the original (nothink=0, think=1536)
    # pair -- a Gemini 3 run has neither of those thinking_budget values.
    _palette = ["#e74c3c", "#f39c12", "#7f8c8d", "#2980b9"]
    conditions = [(condition_label(c), c, _palette[i % len(_palette)])
                  for i, c in enumerate(sorted(df["condition"].unique()))]

    def named_rate(sub: pd.DataFrame) -> tuple[float, int, int]:
        n = len(sub)
        named = int(sub["category"].isin(["all", "some"]).sum())
        return (100 * named / n if n else float("nan")), named, n

    fig = go.Figure()
    table_rows = []
    per_cond_stats = {}
    for cond_label, tb, color in conditions:
        ys = []
        stats = []
        for cat in categories:
            sub = df[(df["defect_type"] == cat) & (df["condition"] == tb)]
            rate, named, n = named_rate(sub)
            ys.append(rate)
            stats.append((rate, named, n))
        sub_all = df[df["condition"] == tb]
        rate_all, named_all, n_all = named_rate(sub_all)
        ys.append(rate_all)
        stats.append((rate_all, named_all, n_all))
        per_cond_stats[cond_label] = stats
        fig.add_trace(go.Bar(name=cond_label, x=x_labels, y=ys, marker_color=color))

    for i, label in enumerate(x_labels):
        row = [label]
        for cond_label, _, _ in conditions:
            rate, named, n = per_cond_stats[cond_label][i]
            row.append(_fmt_rate(rate, n))
        table_rows.append(row)

    fig.update_layout(
        barmode="group",
        title="Did the justification name the real defect? (reused judge category, by defect type)",
        yaxis=dict(title="category in {all, some} rate (%)", range=[0, 105]),
        font=FONT, legend=LEGEND, margin=MARGIN,
        width=800, height=480,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.add_hline(y=50, line_dash="dot", line_color="#aaa")

    extra_html = _pre_table(
        ["Defect type"] + [c[0] for c in conditions], table_rows,
        caption="category in {all, some} rate, scope: rows already correctly called invalid",
    )

    save(fig, "V11_sign_flip_naming_rate", out_dir,
         title="Sign-Flip Defect Naming Rate (reused judge verdicts)",
         description="For SIGN FLIP gt_samples (17, a single clean discrepancy each -- see "
                     "agent_docs/sign_flip_classification.txt), the judge's category=all/some "
                     "verdict is exactly 'did the justification name the sign flip'. For MIXED/"
                     "OTHER/ALL, the same category field is scored against whichever real "
                     "discrepancy is actually present, not sign flips specifically. Scope for "
                     "every bar: only rows the model already correctly called invalid (recall "
                     "itself is separately near-ceiling for SIGN FLIP -- see "
                     "agent_docs/sign_flip_recall_analysis.txt). On the original two-condition "
                     "2.5 data, SIGN FLIP had the LOWEST naming rate of any category in both "
                     "conditions despite being the easiest defect type to detect at all, and "
                     "extended thinking did not close that gap -- check whether that still holds "
                     "here rather than assuming it, since this report may be rendering a "
                     "different model or a single condition.",
         extra_html=extra_html)


# ── Results 12-16: the validity score ────────────────────────────────────────
# The score is a 0-10 physical-validity rating collected alongside (never
# instead of) the binary valid/invalid call, at both stages. Its purpose is to
# separate DISCRIMINATION (can the model rank correct code above defective
# code?) from CALIBRATION (does it put its yes/no threshold in the right
# place?) -- two things the binary field fuses together and cannot distinguish.
#
# Panel order is deliberate: V12 (distribution) comes first because the score is
# bimodal BY GENUINE CLASS SEPARATION, and that fact is what justifies using a
# rank-based metric (V14) over a mean (V13). Reading V13 or V14 without V12
# invites treating a mean over a two-humped ordinal as meaningful.

SCORE_MIN, SCORE_MAX = 0, 10

# Valid <-> invalid mod_type pairings, for the paired comparisons. Sourced from
# the judge module rather than re-derived, so there is one definition.
try:
    from judge_invalidity_reasoning import INVALID_TO_VALID_MOD_TYPE  # noqa: E402
except ImportError:  # pragma: no cover -- viz can render without eval/ on path
    INVALID_TO_VALID_MOD_TYPE = {
        "Comm_InValid": "Comm_Valid", "NoComm_InValid": "NoComm_Valid",
        "NoComm_CorrVar_InValid": "NoComm_CorrVar", "CorrComm_Invalid": "CorrComm",
    }
PAIRING_ORDER = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar"]


def _has_scores(df: pd.DataFrame) -> bool:
    return (not df.empty and "s2_validity_score" in df.columns
            and df["s2_validity_score"].notna().any())


def score_auc(pos: list, neg: list) -> float:
    """P(a random truly-valid row outscores a random truly-invalid row), ties at
    0.5 -- a normalised Mann-Whitney U. Threshold-free and rank-only, which is
    why it is the right lens on a lumpy 0-10 integer: it is unaffected by the
    score's absolute calibration or by clustering, and it is the only measure
    that stays comparable across models (a "7" from one model is not a "7" from
    another, but a ranking is a ranking)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _scores(df: pd.DataFrame, stage: str, gt_valid: bool) -> list:
    col = f"{stage}_validity_score"
    if df.empty or col not in df.columns:
        return []
    sub = df[df["gt_valid"] == gt_valid][col].dropna()
    return [int(x) for x in sub]


def _best_threshold(df: pd.DataFrame, stage: str) -> tuple:
    """Accuracy achievable from the score alone at its best cutoff, vs the
    model's own binary call. The gap between them IS the calibration loss: how
    much accuracy is lost purely to where the model puts its yes/no line,
    rather than to what it can tell apart."""
    col = f"{stage}_validity_score"
    sub = df[df[col].notna()]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    best_t, best_acc = None, -1.0
    for t in range(SCORE_MIN, SCORE_MAX + 2):
        acc = ((sub[col] >= t) == (sub["gt_valid"] == True)).mean()  # noqa: E712
        if acc > best_acc:
            best_t, best_acc = t, acc
    own = sub[f"{stage}_valid_match"].dropna().mean()
    return best_t, best_acc, own


def v12_score_distribution(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path) -> None:
    frames = [(df_condition_label(df_a), df_a)] + ([(df_condition_label(df_b), df_b)] if not df_b.empty else [])
    frames = [(lab, d) for lab, d in frames if _has_scores(d)]
    if not frames:
        print("  [viz-agentic] No validity_score data -- skipping V12 distribution.")
        return

    fig = make_subplots(rows=len(frames), cols=2,
                        subplot_titles=[f"{lab} — {st}" for lab, _ in frames for st in ("Stage 1", "Stage 2")],
                        horizontal_spacing=0.10, vertical_spacing=0.18)
    xs = list(range(SCORE_MIN, SCORE_MAX + 1))
    for r, (lab, d) in enumerate(frames, start=1):
        for c, stage in enumerate(("s1", "s2"), start=1):
            for gt, name, color in [(True, "truly valid", "#27ae60"), (False, "truly invalid", "#c0392b")]:
                vals = _scores(d, stage, gt)
                counts = [vals.count(x) for x in xs]
                fig.add_trace(go.Bar(x=xs, y=counts, name=name, marker_color=color,
                                     legendgroup=name, showlegend=(r == 1 and c == 1)),
                              row=r, col=c)
            fig.update_xaxes(title_text="validity_score", dtick=1, row=r, col=c)
            fig.update_yaxes(title_text="Rows", row=r, col=c)

    fig.update_layout(title="Validity-Score Distribution by Ground Truth",
                      barmode="group", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1300, height=380 * len(frames),
                      plot_bgcolor="white", paper_bgcolor="white")

    rows = []
    for lab, d in frames:
        for stage in ("s1", "s2"):
            allv = _scores(d, stage, True) + _scores(d, stage, False)
            col = f"{stage}_validity_score"
            missing = int(d[col].isna().sum()) if col in d.columns else 0
            ends = sum(1 for v in allv if v <= 1 or v >= 9)
            rows.append([f"{lab} {stage.upper()}", str(len(allv)), str(len(set(allv))),
                         f"{100*ends/len(allv):.0f}%" if allv else "n/a", str(missing)])
    extra = _pre_table(["Condition/stage", "n scored", "distinct values", "mass in end bins (0-1, 9-10)", "missing"],
                       rows, caption="Table — scale usage and compliance")

    save(fig, "V12_score_distribution", out_dir,
         title="Validity-score distribution by ground truth",
         description="Counts of each 0-10 score, split by ground truth, per stage. Read this "
                     "panel BEFORE V13/V14. The distribution is expected to be bimodal, but the "
                     "cause matters: if the two humps sit on opposite sides (truly-valid high, "
                     "truly-invalid low) that is genuine class separation, and a mean over it is "
                     "a weak summary -- use the rank-based AUC in V14 instead. If instead both "
                     "classes pile onto the same end bins, the model is not using the scale and "
                     "no score claim is trustworthy. 'distinct values' and 'missing' in the table "
                     "are the compliance check: a score the model often declines to give, or one "
                     "that only ever takes two values, is a schema problem rather than a "
                     "measurement.",
         extra_html=extra)


def v13_mean_score_by_modtype(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path) -> None:
    frames = [(df_condition_label(df_a), df_a)] + ([(df_condition_label(df_b), df_b)] if not df_b.empty else [])
    frames = [(lab, d) for lab, d in frames if _has_scores(d)]
    if not frames:
        print("  [viz-agentic] No validity_score data -- skipping V13 mean score.")
        return

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Panel A — truly-invalid mod_types (score should be LOW)",
                                        "Panel B — truly-valid mod_types (score should be HIGH)"],
                        horizontal_spacing=0.12)
    palette = ["#2980b9", "#e74c3c", "#7f8c8d", "#f39c12"]
    idx = 0
    for lab, d in frames:
        for stage in ("s1", "s2"):
            color = palette[idx % len(palette)]
            idx += 1
            for col_idx, mods in enumerate([INVALID_MOD_TYPES, VALID_MOD_TYPES], start=1):
                mods = [m for m in mods if m in set(d["mod_type"])]
                ys, los, his = [], [], []
                for m in mods:
                    sub = d[d["mod_type"] == m][f"{stage}_validity_score"].dropna()
                    lo, mean, hi = bootstrap_ci(sub)
                    ys.append(mean); los.append(mean - lo); his.append(hi - mean)
                fig.add_trace(go.Bar(
                    name=f"{stage.upper()} ({lab})", x=[MOD_SHORT.get(m, m) for m in mods], y=ys,
                    marker_color=color,
                    error_y=dict(type="data", symmetric=False, array=his, arrayminus=los),
                    legendgroup=f"{stage}{lab}", showlegend=(col_idx == 1),
                ), row=1, col=col_idx)
    for c in (1, 2):
        fig.update_yaxes(range=[0, 10.6], title_text="Mean validity_score", row=1, col=c)
    fig.update_layout(title="Mean Validity Score by Mod Type",
                      barmode="group", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1400, height=560, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(tickangle=20)

    save(fig, "V13_mean_score_modtype", out_dir,
         title="Mean validity score by mod_type",
         description="Mean 0-10 score per mod_type with bootstrap 95% CIs, split by whether the "
                     "mod_type is truly-invalid (Panel A) or truly-valid (Panel B). CAVEAT: a "
                     "mean over a bimodal ordinal is a weak summary -- see V12 for the shape it "
                     "is summarising, and prefer V14's AUC, which is rank-based and therefore "
                     "unaffected by clustering. This panel is useful mainly for seeing whether "
                     "the valid/invalid gap is uniform across the naming conditions, not for the "
                     "absolute values.")


def v14_score_auc(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path) -> None:
    frames = [(df_condition_label(df_a), df_a)] + ([(df_condition_label(df_b), df_b)] if not df_b.empty else [])
    frames = [(lab, d) for lab, d in frames if _has_scores(d)]
    if not frames:
        print("  [viz-agentic] No validity_score data -- skipping V14 AUC.")
        return

    x_labels = ["Aggregate"] + [MOD_SHORT.get(m, m) for m in PAIRING_ORDER]
    fig = go.Figure()
    palette = ["#2980b9", "#e74c3c", "#7f8c8d", "#f39c12"]
    idx = 0
    table_rows = []
    for lab, d in frames:
        for stage in ("s1", "s2"):
            color = palette[idx % len(palette)]; idx += 1
            ys = [score_auc(_scores(d, stage, True), _scores(d, stage, False))]
            for valid_mt in PAIRING_ORDER:
                inv_mt = next(k for k, v in INVALID_TO_VALID_MOD_TYPE.items() if v == valid_mt)
                col = f"{stage}_validity_score"
                pos = [int(x) for x in d[d["mod_type"] == valid_mt][col].dropna()]
                neg = [int(x) for x in d[d["mod_type"] == inv_mt][col].dropna()]
                ys.append(score_auc(pos, neg))
            fig.add_trace(go.Bar(name=f"{stage.upper()} ({lab})", x=x_labels, y=ys,
                                 marker_color=color,
                                 text=[f"{y:.3f}" if y == y else "n/a" for y in ys],
                                 textposition="outside"))
            best_t, best_acc, own = _best_threshold(d, stage)
            table_rows.append([f"{lab} {stage.upper()}", f"{ys[0]:.3f}",
                               f">= {best_t}", f"{best_acc*100:.1f}%", f"{own*100:.1f}%",
                               f"{(best_acc-own)*100:+.1f} pp"])

    fig.add_hline(y=0.5, line_dash="dot", line_color="#aaa",
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1.08], title_text="AUC (score vs gt_valid)")
    fig.update_layout(title="Score Discrimination (AUC): Aggregate and by Mod-Type Pairing",
                      barmode="group", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1400, height=540, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(tickangle=20)

    extra = _pre_table(
        ["Condition/stage", "AUC", "best cutoff", "acc @ best cutoff", "model's own binary acc", "calibration loss"],
        table_rows,
        caption="Table — discrimination vs calibration. 'Calibration loss' is how much accuracy "
                "is recoverable purely by moving the yes/no threshold, with the model's ranking "
                "held fixed.")

    save(fig, "V14_score_auc", out_dir,
         title="Score discrimination (AUC), aggregate and by mod-type pairing",
         description="AUC = P(a random truly-valid row outscores a random truly-invalid row), "
                     "ties counted 0.5 -- a normalised Mann-Whitney U. Threshold-free and "
                     "rank-only, so it is unaffected by the score's absolute calibration or by "
                     "the clustering visible in V12, and it is the only score measure that stays "
                     "comparable across models. Per-pairing bars compare each valid mod_type "
                     "against its own invalid counterpart. The KEY read is the table: a high AUC "
                     "alongside a large calibration loss means the model CAN rank correct code "
                     "above defective code and is simply putting its yes/no cutoff in the wrong "
                     "place -- a calibration failure, not a knowledge failure. A low AUC means "
                     "the opposite, and the score adds nothing the binary call did not already "
                     "say. Per-pairing bars rest on 32 rows per side here, so read directions "
                     "rather than magnitudes.")


def v15_paired_ranking(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path) -> None:
    """Does the clean member of a matched pair outscore its OWN defective twin?

    Pairs are keyed on (gt_sample, mod_type pairing) -- NOT gt_sample alone.
    Each gt_sample carries four valid mod_types and four invalid ones, so keying
    on gt_sample alone silently collapses eight rows into one pair and discards
    most of the comparison. Pairing removes between-sample difficulty variance,
    which makes this better powered than the unpaired AUC."""
    frames = [(df_condition_label(df_a), df_a)] + ([(df_condition_label(df_b), df_b)] if not df_b.empty else [])
    frames = [(lab, d) for lab, d in frames if _has_scores(d)]
    if not frames or "gt_sample" not in df_a.columns or df_a["gt_sample"].isna().all():
        print("  [viz-agentic] No gt_sample/score data -- skipping V15 paired ranking.")
        return

    OUT = ["clean higher", "tied", "inverted"]
    COLORS = {"clean higher": "#27ae60", "tied": "#95a5a6", "inverted": "#c0392b"}
    # S1 hatched, S2 solid. Colour has to keep encoding the OUTCOME, so stage is
    # carried by fill pattern instead -- reusing colour for stage would cost the
    # clean/tied/inverted distinction that is the whole point of the panel.
    STAGE_PATTERN = {"s1": "/", "s2": ""}

    def _paired_counts(d: pd.DataFrame, stage: str) -> tuple[dict, dict]:
        """Pair up each row with its own twin and bucket the comparison.

        Same pairing key at both stages, so the S1 and S2 bars are strictly
        comparable. A pair whose row is present but UNSCORED (a blank forced
        answer) is bucketed as "tied" rather than dropped. No score exists, so
        the model expressed no ranking either way, and _ratio() already excludes
        ties from its denominator -- exactly where an uninformative pair
        belongs. Dropping it would make one arm's pair total differ from the
        rest for a reason unrelated to ranking; calling it "inverted" would
        assert a ranking that was never made."""
        col = f"{stage}_validity_score"
        pairs = {}
        for _, r in d.iterrows():
            gs, mt = r.get("gt_sample"), r.get("mod_type")
            if not gs:
                continue
            sc = r.get(col)
            key = (gs, INVALID_TO_VALID_MOD_TYPE.get(mt, mt))
            # None marks PRESENT BUT UNSCORED, distinct from an absent twin.
            pairs.setdefault(key, {})[bool(r["gt_valid"])] = (
                int(sc) if sc == sc and sc is not None else None)
        counts = {k: 0 for k in OUT}
        per_pairing = {p: {k: 0 for k in OUT} for p in PAIRING_ORDER}
        for (gs, pairing), v in pairs.items():
            if True not in v or False not in v:
                continue
            if v[True] is None or v[False] is None:
                k = "tied"
            else:
                k = "clean higher" if v[True] > v[False] else ("tied" if v[True] == v[False] else "inverted")
            counts[k] += 1
            if pairing in per_pairing:
                per_pairing[pairing][k] += 1
        return counts, per_pairing

    def _ratio(c: dict) -> float:
        denom = c["clean higher"] + c["inverted"]
        return c["clean higher"] / denom if denom else float("nan")

    fig = go.Figure()
    table_rows = []
    first = True
    for lab, d in frames:
        stage_counts = {}
        for stage in ("s1", "s2"):
            counts, per_pairing = _paired_counts(d, stage)
            stage_counts[stage] = counts
            for k in OUT:
                fig.add_trace(go.Bar(
                    name=k, x=["Aggregate"] + [MOD_SHORT.get(p, p) for p in PAIRING_ORDER],
                    y=[counts[k]] + [per_pairing[p][k] for p in PAIRING_ORDER],
                    marker_color=COLORS[k],
                    marker_pattern_shape=STAGE_PATTERN[stage],
                    # offsetgroup puts S1 and S2 side by side while barmode=stack
                    # still stacks the three outcomes WITHIN each bar.
                    offsetgroup=f"{lab}-{stage}",
                    legendgroup=k, showlegend=(first and stage == "s1"),
                    hovertemplate=f"{stage.upper()} ({lab})<br>{k}: %{{y}}<extra></extra>",
                ))
            table_rows.append([f"{lab} {stage.upper()}", str(sum(counts.values())),
                               str(counts["clean higher"]), str(counts["tied"]),
                               str(counts["inverted"]),
                               f"{_ratio(counts):.2f}" if _ratio(counts) == _ratio(counts) else "n/a"])
        # Delta row: the tie collapse is the finding, so spell it out rather than
        # leaving the reader to subtract two rows.
        a, b = stage_counts["s1"], stage_counts["s2"]
        d_ratio = _ratio(b) - _ratio(a)
        table_rows.append([f"  {lab} delta S1->S2",
                           f"{sum(b.values()) - sum(a.values()):+d}",
                           f"{b['clean higher'] - a['clean higher']:+d}",
                           f"{b['tied'] - a['tied']:+d}",
                           f"{b['inverted'] - a['inverted']:+d}",
                           f"{d_ratio:+.2f}" if d_ratio == d_ratio else "n/a"])
        first = False

    fig.update_layout(title="Paired Ranking: Clean vs Its Own Defective Twin (S1 vs S2)",
                      barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1300, height=520, yaxis_title="Matched pairs",
                      plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=20)

    extra = _pre_table(["Condition/stage", "pairs", "clean higher", "tied", "inverted", "ratio (higher / (higher+inverted))"],
                       table_rows,
                       caption="Table — paired ranking, per stage with the S1->S2 delta. Ties are "
                               "EXCLUDED from the ratio and separately compress AUC, so a high tie "
                               "count is itself a warning that the scale is being used too coarsely.")

    save(fig, "V15_paired_ranking", out_dir,
         title="Paired ranking: clean vs its own defective twin",
         description="Valid and invalid rows come in matched pairs that differ by ONE injected "
                     "defect on otherwise identical code, so this asks the sharpest available "
                     "question: per pair, did the clean version outscore its own corrupted twin? "
                     "Pairing removes between-sample difficulty variance, making this better "
                     "powered than the unpaired AUC in V14. Pairs are keyed on (gt_sample, "
                     "mod_type pairing), not gt_sample alone -- each gt_sample has four valid and "
                     "four invalid mod_types, so keying on the sample alone would collapse eight "
                     "rows into a single pair. Inverted pairs are the highest-value rows to "
                     "inspect by hand. Both stages are shown -- S1 hatched, S2 solid -- because "
                     "the S1->S2 movement is the point: Stage 1 is the model's static ranking and "
                     "Stage 2 is what execution did to it. Read the delta row in the table first. "
                     "Tie resolution is usually the largest effect: a model that scores a clean "
                     "script and its defective twin identically has told you nothing, and evidence "
                     "is what breaks the tie. A condition that arrives at S1 with few ties and no "
                     "inversions has little left for execution to fix, so a small delta there is "
                     "a ceiling effect rather than a failure of investigation.",
         # Was missing: `extra` was built and then dropped, so this panel's table
         # had never rendered in ANY report. Every other table-bearing panel
         # passes extra_html; this one silently did not.
         extra_html=extra)


def v16_score_vs_judge(df_a: pd.DataFrame, judge_df: pd.DataFrame, out_dir: Path) -> None:
    """Groundedness cross-check: a good AUC proves the score RANKS correctly, not
    that it is grounded in anything the model actually found. The judge's
    none/some/all verdict on the same explanation text is the external check."""
    if judge_df.empty or not _has_scores(df_a):
        print("  [viz-agentic] No judge/score overlap -- skipping V16.")
        return
    scores = df_a.set_index("title")["s2_validity_score"].to_dict()
    rows_by_cat = {c: [] for c in CATEGORY_ORDER}
    for _, r in judge_df.iterrows():
        sc = scores.get(r["title"])
        if sc is not None and sc == sc and r["category"] in rows_by_cat:
            rows_by_cat[r["category"]].append(int(sc))

    fig = go.Figure()
    ys, los, his, ns = [], [], [], []
    for c in CATEGORY_ORDER:
        vals = rows_by_cat[c]
        lo, mean, hi = bootstrap_ci(pd.Series(vals, dtype=float)) if vals else (float("nan"),) * 3
        ys.append(mean); los.append(mean - lo); his.append(hi - mean); ns.append(len(vals))
    fig.add_trace(go.Bar(x=CATEGORY_ORDER, y=ys,
                         marker_color=[CATEGORY_COLORS[c] for c in CATEGORY_ORDER],
                         text=[f"n={n}" for n in ns], textposition="outside",
                         error_y=dict(type="data", symmetric=False, array=his, arrayminus=los)))
    fig.update_layout(title="Validity Score vs Judge-Assessed Reasoning Quality",
                      font=FONT, margin=MARGIN, width=800, height=460,
                      yaxis_title="Mean validity_score", plot_bgcolor="white",
                      paper_bgcolor="white")
    fig.update_yaxes(range=[0, 10.6])

    save(fig, "V16_score_vs_judge", out_dir,
         title="Validity score vs judge-assessed reasoning quality",
         description="Mean score for rows the judge graded none / some / all, over "
                     "correctly-called-invalid rows only. This is a GROUNDEDNESS check, not a "
                     "performance measure: a high AUC shows the score ranks correctly, but not "
                     "that it reflects anything the model actually found -- if the score were "
                     "confabulated to agree with the binary call, it would rank fine while "
                     "meaning nothing. Expect little separation here: on the data this was built "
                     "for, all three categories sat near the floor of the scale, so the score "
                     "does NOT predict whether the explanation was right. That null is the "
                     "result; do not read a small gap between bars as a finding.")


def v17_precision_rates(judge_df: pd.DataFrame, out_dir: Path) -> None:
    """Judge precision rates as tables only -- no chart. `category` (V3b) is a
    RECALL-only metric: it asks whether the justification named the real defect,
    and says nothing about whether it also asserted something false. These rates
    are the only place that shows up.

    Reports whichever precision columns the judge file actually populated, so a
    pre-split run (merged `contains_incorrect_claims`) and a post-split run
    (code-claims + execution-claims) both render."""
    if judge_df.empty:
        print("  [viz-agentic] No judge results -- skipping V17 precision rates.")
        return
    cols = [(c, o) for c, o in PRECISION_FIELDS
            if c in judge_df.columns and judge_df[c].notna().any()]
    if not cols:
        print("  [viz-agentic] No precision fields populated -- skipping V17.")
        return

    blocks = [_pre_table(
        ["Condition", "n"] + [o for _, o in cols],
        [[condition_label(cond), str(len(sub))]
         + [_fmt_rate(sub[c].mean() * 100, len(sub)) for c, _ in cols]
         for cond, sub in judge_df.groupby("condition")],
        caption="Table 1 — precision rate, global (all judged rows pooled)")]

    for cond, sub in judge_df.groupby("condition"):
        rows = []
        for m in INVALID_MOD_TYPES:
            s = sub[sub["mod_type"] == m]
            if s.empty:
                continue
            rows.append([MOD_SHORT.get(m, m), str(len(s))]
                        + [_fmt_rate(s[c].mean() * 100, len(s)) for c, _ in cols])
        if rows:
            blocks.append(_pre_table(
                ["mod_type", "n"] + [o for _, o in cols], rows,
                caption=f"Table 2 — precision rate by mod_type — {condition_label(cond)}"))

    save(None, "V17_precision_rates", out_dir,
         title="Judge precision rates (global and by mod_type)",
         description="Rate at which the model's justification contained a faulty claim, "
                     "over rows it already correctly called invalid. This is the PRECISION "
                     "counterpart to V3b's category, which is recall-only -- a justification "
                     "can name every real defect ('all') and still assert something false. "
                     "bad_code_claims_rate is the one that matters: it fires only when the "
                     "justification says something factually WRONG about the code. "
                     "unverifiable_exec_claims_rate fires when it cites runtime figures the "
                     "judge cannot check from the code alone (step counts, magnitudes, NaN "
                     "timing); that is NOT an error -- the model could execute the code -- and "
                     "it runs so high that it is near-constant, so read it as a description of "
                     "how the model argues rather than as a comparative metric. "
                     "legacy_merged_rate appears only for judge runs predating the split, where "
                     "one field conflated 'extraneous, speculative, or factually wrong'; on the "
                     "3.7 high sweep 20 of its 24 flagged rows were objections to runtime "
                     "figures rather than to anything wrong about the code, so legacy figures "
                     "overstate the error rate roughly 6x and are not comparable to "
                     "bad_code_claims_rate.",
         extra_html="".join(blocks))


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
    p.add_argument("--stage2", action="append", default=[],
                   help="Stage-2 JSONL. Repeatable (once per thinking condition). Condition "
                        "labels are read from each file's `condition` field, falling back to "
                        "tb<thinking_budget> for rows written before that field existed, so "
                        "both the old 2.5 two-condition data and a single-condition Gemini 3 "
                        "run render correctly.")
    p.add_argument("--nothink", default=None,
                   help="DEPRECATED alias for the first --stage2 path, kept so existing "
                        "two-condition invocations keep working.")
    p.add_argument("--think", default=None,
                   help="DEPRECATED alias for the second --stage2 path.")
    p.add_argument("--judge", default="results/frontier/stratified_64/judge/judge_results.jsonl",
                   help="Path to judge_results.jsonl. Optional -- if the file doesn't exist yet, "
                        "judge sections are skipped (load_judge_results already handles this).")
    p.add_argument("--out_dir", default="results/frontier/stratified_64/viz")
    return p.parse_args()


def main():
    args = parse_args()

    # --stage2 is the current flag; --nothink/--think are kept as positional
    # aliases so existing two-condition invocations keep working unchanged.
    paths = list(args.stage2) or [p for p in (args.nothink, args.think) if p]
    if not paths:
        sys.exit("[viz-agentic] ERROR: pass at least one --stage2 <path> "
                 "(or the deprecated --nothink).")
    if len(paths) > 2:
        sys.exit(f"[viz-agentic] ERROR: {len(paths)} conditions given; this report "
                 "renders at most 2. Run it once per pair.")

    df_a = load_agentic_results(Path(paths[0]))
    df_b = load_agentic_results(Path(paths[1])) if len(paths) > 1 else pd.DataFrame()
    judge_df = load_judge_results(Path(args.judge))
    lab_a, lab_b = df_condition_label(df_a), df_condition_label(df_b)
    print(f"[viz-agentic] condition A ({lab_a}): {len(df_a)} rows")
    if not df_b.empty:
        print(f"[viz-agentic] condition B ({lab_b}): {len(df_b)} rows")
    else:
        print("[viz-agentic] single condition -- per-condition transitions/turns "
              "sections render once, and the accuracy plot shows 2 lines.")
    print(f"[viz-agentic] judge: {len(judge_df)} rows")

    out_dir = Path(args.out_dir)
    print(f"[viz-agentic] Output → {out_dir}\n")

    v0_architecture_diagram(out_dir)
    cond_a = sorted(set(df_a["condition"]))[0]
    v1_accuracy_by_modtype(df_a, df_b, out_dir)
    v2_transitions_by_condition(df_a, out_dir, lab_a, cond_a)
    v3a_turns_by_condition(df_a, out_dir, lab_a, cond_a)
    if not df_b.empty:
        cond_b = sorted(set(df_b["condition"]))[0]
        v2_transitions_by_condition(df_b, out_dir, lab_b, cond_b)
        v3a_turns_by_condition(df_b, out_dir, lab_b, cond_b)
    v3b_judge_by_modtype(judge_df, out_dir)
    v3b_judge_by_pdeclass(judge_df, out_dir)
    v10_recall_specificity(df_a, df_b, out_dir)
    # Score panels. V12 first by design -- V13/V14 are hard to read without
    # knowing the shape of the distribution they summarise.
    v12_score_distribution(df_a, df_b, out_dir)
    v13_mean_score_by_modtype(df_a, df_b, out_dir)
    v14_score_auc(df_a, df_b, out_dir)
    v15_paired_ranking(df_a, df_b, out_dir)
    v16_score_vs_judge(df_a, judge_df, out_dir)
    v11_sign_flip_naming_rate(judge_df, out_dir)
    v17_precision_rates(judge_df, out_dir)

    incorrect_claims = contains_incorrect_claims_table(judge_df)
    if not incorrect_claims.empty:
        print("\n[viz-agentic] contains_incorrect_claims rate by condition:")
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
