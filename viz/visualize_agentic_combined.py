"""
visualize_agentic_combined.py — one HTML report covering ALL FOUR arms of the
validity/PDE benchmark side by side.

Deliberately a SEPARATE script from visualize_agentic.py, which stays untouched
and keeps producing the per-arm reports. Only *pure* helpers are imported from
it -- data loaders, rate/AUC computation, table formatting, the mod_type
constants. In particular `save()` from visualize_frontier is NOT used: it
appends to a module-global figure list there, so calling it would entangle the
two scripts' state whenever both are imported in one process. This file builds
its figures directly and writes its own HTML.

Layout:
  - Combined single-figure panels, all four arms on one chart: accuracy by
    mod_type (8 lines), PDE identification (control, 8 lines), the summary
    accuracy/AUC bars, and Agent Choices.
  - Everything else renders as FOUR stacked figures, one per arm, in fixed
    order, each titled with its arm name. No toggles: with only four arms,
    stacking is simpler than JS show/hide and lets you compare by scrolling.

Usage:
  python3 viz/visualize_agentic_combined.py \
      --arm "2.5-flash-tb0=<stage2.jsonl>=<judge.jsonl>" \
      --arm "2.5-flash-tb8192=<stage2.jsonl>=<judge.jsonl>" \
      --arm "3.7-flash-low=<stage2.jsonl>=<judge.jsonl>" \
      --arm "3.7-flash-high=<stage2.jsonl>=<judge.jsonl>" \
      --out_dir results/frontier/combined_viz

Needs pandas + plotly, which live in system python3 here, not eval/.venv.
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualize_agentic import (  # noqa: E402  -- pure helpers only
    INVALID_MOD_TYPES,
    VALID_MOD_TYPES,
    PAIRING_ORDER,
    INVALID_TO_VALID_MOD_TYPE,
    SCORE_MIN,
    SCORE_MAX,
    _best_threshold,
    _pre_table,
    _rate,
    load_agentic_results,
    load_judge_results,
    score_auc,
)
from visualize_frontier import (  # noqa: E402
    FONT, LEGEND, MARGIN, MOD_ORDER, MOD_SHORT, PDE_ORDER, PDE_LABEL,
    TRANS_COLORS, TRANS_ORDER, TRANS_SHORT, bootstrap_ci,
)

# One hue family per arm, two shades: light = Stage 1, dark = Stage 2. Colour
# carries the ARM, shade carries the STAGE, so an 8-line chart stays readable.
ARM_COLORS = {
    0: ("#f5b041", "#c0560b"),   # orange
    1: ("#bb8fce", "#6c3483"),   # purple
    2: ("#85c1e9", "#1a5276"),   # blue
    3: ("#7dcea0", "#196f3d"),   # green
}


# Report panel order. Panels not listed keep their build order and go last.
PANEL_ORDER = [
    "Summary — accuracy and AUC across arms",
    "Agent choices",
    "Transitions by mod_type",
    "Transitions by mod_type (reorganized)",
    "Reasoning quality across arms",
    "Paired ranking: clean vs its own defective twin",
    "Validity accuracy by mod_type — all 8 lines",
    "Score ranking AUC by mod_type",
    "PDE identification by mod_type (control)",
]

# Panels whose HTML figure is exported as-is to a paper PDF, keyed by panel
# title. agent_fig1 is NOT here: it is built separately by p_paper_fig1() and has
# no corresponding HTML panel, which is why export takes explicit (figure, name)
# pairs rather than deriving everything from the panel list.
#
# The top-level figure title is STRIPPED in every PDF -- a LaTeX float supplies
# its own caption, and a duplicated title reads as an error. Subplot titles (the
# arm names) are kept, since they label the panels.
# NOTE: "Reasoning quality across arms" is deliberately NOT here. Entries in this
# map export the SAME figure object the HTML shows, so leaving it would push the
# HTML's new error bars into agent_fig3.pdf. It is exported via extra_pdfs
# instead, rebuilt with show_ci=False.
PDF_FROM_PANEL = {
    "Transitions by mod_type": "agent_fig2",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", default=[], required=True,
                   help='"label=stage2_jsonl=judge_jsonl[=judge_s1_jsonl]". Repeat per arm. '
                        'The judge path may be empty ("label=stage2="). The optional 4th '
                        'component is a Stage-1 judge pass (--stage s1 output); supplying it '
                        'adds the paired S1-vs-S2 reasoning-quality table for that arm. '
                        'Explicit rather than derived, so an arm with several judge files '
                        'picks the right one.')
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_ci_cache", action="store_true",
                   help="Disable the bootstrap CI cache entirely (compute, do not read or write).")
    p.add_argument("--recompute_ci", action="store_true",
                   help="Ignore cached intervals, recompute them all, and rewrite the cache.")
    p.add_argument("--no_png", action="store_true",
                   help="Skip PNG/table export and write only the HTML (faster iteration).")
    return p.parse_args()


def load_arms(specs):
    arms = []
    for spec in specs:
        parts = spec.split("=")
        if len(parts) not in (2, 3, 4):
            sys.exit(f"[combined] bad --arm {spec!r}: expected label=stage2=judge[=judge_s1]")
        label, s2p = parts[0], parts[1]
        jp = parts[2] if len(parts) >= 3 else ""
        j1p = parts[3] if len(parts) == 4 else ""
        df = load_agentic_results(Path(s2p))
        jd = load_judge_results(Path(jp)) if jp else load_judge_results(Path("/nonexistent"))
        j1 = load_judge_results(Path(j1p)) if j1p else load_judge_results(Path("/nonexistent"))
        arms.append({"label": label, "df": df, "judge": jd, "judge_s1": j1})
        extra = f", judge_s1 {len(j1)} rows" if len(j1) else ""
        print(f"[combined] {label:20s} {len(df)} rows, judge {len(jd)} rows{extra}")
    return arms


# ── metric helpers ───────────────────────────────────────────────────────────

def _acc(df, stage):
    v = df[f"{stage}_valid_match"].dropna()
    return 100 * v.mean() if len(v) else float("nan")


def _auc(df, stage):
    col = f"{stage}_validity_score"
    if col not in df.columns:
        return float("nan")
    pos = [int(x) for x in df[df["gt_valid"] == True][col].dropna()]   # noqa: E712
    neg = [int(x) for x in df[df["gt_valid"] == False][col].dropna()]  # noqa: E712
    return score_auc(pos, neg)


# ── cluster bootstrap ────────────────────────────────────────────────────────
#
# The design is 32 ground-truth scripts x 8 lexical perturbations = 256 rows.
# The eight rows sharing a gt_sample are the SAME physics and the same solver
# code, differing only lexically, so they are NOT independent draws: measured on
# 2.5-flash-tb0, the sd of per-script accuracy is 0.196 against 0.163 expected
# under independence. The effective sample size is 32, not 256.
#
# Every interval in this module therefore resamples gt_sample WITH REPLACEMENT,
# carrying all of a script's rows together. Row-level resampling -- which is what
# visualize_frontier.bootstrap_ci does -- understates these intervals by 1.2x on
# tb0 and 2.2x on 3.7-flash-high, worst exactly where the arms sit near ceiling.
# Do not use bootstrap_ci for anything in these panels.
BOOT_N = 4000
BOOT_SEED = 42

# ── CI cache ────────────────────────────────────────────────────────────────
#
# The bootstrap is the slowest thing in the report by two orders of magnitude,
# and it depends only on the DATA, never on layout. Every interval is therefore
# cached to JSON, keyed on (statistic name, fingerprint of the exact columns the
# statistic reads, BOOT_N, BOOT_SEED, ci). A layout-only edit reuses everything;
# changing the data, the draw count, or the seed changes the fingerprint and
# forces a clean recompute, so a stale interval cannot survive silently.
_CI_CACHE: dict = {}
_CI_CACHE_PATH: Path | None = None
_CI_ENABLED = True
_CI_HITS = _CI_MISSES = 0


def ci_cache_open(path: Path, enabled: bool = True, recompute: bool = False):
    """Load the cache. recompute=True starts empty, so every interval is redone
    and the file is rewritten from scratch."""
    global _CI_CACHE, _CI_CACHE_PATH, _CI_ENABLED, _CI_HITS, _CI_MISSES
    _CI_CACHE_PATH, _CI_ENABLED, _CI_HITS, _CI_MISSES = path, enabled, 0, 0
    _CI_CACHE = {}
    if enabled and not recompute and path.exists():
        try:
            _CI_CACHE = json.loads(path.read_text())
        except Exception as exc:
            print(f"[combined] CI cache unreadable ({str(exc)[:50]}) -- recomputing")
            _CI_CACHE = {}


def ci_cache_close():
    if _CI_ENABLED and _CI_CACHE_PATH is not None:
        _CI_CACHE_PATH.write_text(json.dumps(_CI_CACHE))
    total = _CI_HITS + _CI_MISSES
    if total:
        print(f"[combined] CI cache: {_CI_HITS}/{total} hits, {_CI_MISSES} computed"
              + (f" -> {_CI_CACHE_PATH.name}" if _CI_ENABLED and _CI_CACHE_PATH else " (cache off)"))


def _fingerprint(df, fp):
    """Hash of exactly what the statistic depends on: the cluster key plus the
    caller-declared columns. Anything the statistic reads MUST be in `fp`, or a
    data change could go unnoticed and serve a stale interval."""
    h = hashlib.sha1()
    h.update(str(len(df)).encode())
    series = [df["gt_sample"]] + list(fp or [])
    for sr in series:
        h.update(pd.util.hash_pandas_object(sr, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _clusters(df, key="gt_sample"):
    """(cluster keys, {key: row-index}) for cluster resampling."""
    ks = df[key].unique()
    return ks, {k: df.index[df[key] == k] for k in ks}


def _cluster_boot(df, stat, name=None, fp=None, n=BOOT_N, ci=95, key="gt_sample"):
    """(lo, point, hi): percentile CI resampling whole clusters.

    `stat` must RECOMPUTE from a DataFrame rather than average row values -- AUC
    is a U-statistic over valid x invalid pairs and cannot be bootstrapped as a
    mean. Draws that return NaN (a resample lacking one class) are dropped.

    Pass `name` (a stable statistic label) and `fp` (the columns the statistic
    reads) to enable caching. Omitting `name` computes uncached."""
    global _CI_HITS, _CI_MISSES
    ck = None
    if name is not None and _CI_ENABLED:
        ck = f"{name}|{_fingerprint(df, fp)}|{n}|{BOOT_SEED}|{ci}"
        hit = _CI_CACHE.get(ck)
        if hit is not None:
            _CI_HITS += 1
            return tuple(hit)
    _CI_MISSES += 1

    point = stat(df)
    ks, idx = _clusters(df, key)
    if len(ks) < 2:
        out = (point, point, point)
    else:
        # positional take() rather than label .loc[]: no index alignment, and
        # the positions are precomputed once instead of per draw
        pos = {k: df.index.get_indexer(v) for k, v in idx.items()}
        rng = np.random.default_rng(BOOT_SEED)
        vals = []
        for _ in range(n):
            v = stat(df.take(np.concatenate(
                [pos[k] for k in rng.choice(ks, len(ks), replace=True)])))
            if v == v:
                vals.append(v)
        out = ((float(np.percentile(vals, (100 - ci) / 2)), point,
                float(np.percentile(vals, ci + (100 - ci) / 2)))
               if vals else (point, point, point))
    if ck is not None:
        _CI_CACHE[ck] = list(out)
    return out


def _err(lo, point, hi):
    """plotly error_y dict from a (lo, point, hi) triple, or None if degenerate."""
    if lo != lo or hi != hi or hi - lo <= 0:
        return None
    return dict(type="data", symmetric=False,
                array=[hi - point], arrayminus=[point - lo],
                thickness=1.2, width=4, color="#333")


def _errs(triples):
    """Vectorised _err for a whole trace (one triple per x category)."""
    return dict(type="data", symmetric=False,
                array=[t[2] - t[1] for t in triples],
                arrayminus=[t[1] - t[0] for t in triples],
                thickness=1.2, width=4, color="#333")


def _paired_delta(df, col_a="s1_valid_match", col_b="s2_valid_match"):
    """S2-S1 accuracy difference in points, clustered.

    S1 and S2 are the SAME episodes, so this is a paired contrast. Marginal
    per-bar intervals overlap heavily even where this excludes zero, which is
    why the panels annotate it rather than inviting readers to eyeball overlap.
    The difference is taken PER EPISODE first, then bootstrapped -- never by
    differencing two bootstrap distributions."""
    d = df.dropna(subset=[col_a, col_b])
    return _cluster_boot(d, lambda s: 100 * (s[col_b] - s[col_a]).mean(),
                         name=f"paired_delta|{col_a}|{col_b}",
                         fp=[d[col_a], d[col_b]])


def _paired_auc_delta(df):
    """S2-S1 AUC difference, clustered. Same rows generate both, so each draw
    recomputes BOTH AUCs on the resampled frame and differences them there."""
    return _cluster_boot(df, lambda s: _auc(s, "s2") - _auc(s, "s1"),
                         name="paired_auc_delta",
                         fp=[df[c] for c in ("s1_validity_score", "s2_validity_score",
                                             "gt_valid") if c in df.columns])


def _judge_gt_sample(jd):
    """Judge rows carry `title` but no gt_sample. Derive it exactly as
    run_belief_revision.get_base_sample does: 'Burgers_Comm_InValid_1' ->
    'Burgers_1'. Cluster sizes are unequal (1-4) because judge eligibility is
    conditional on the model having answered correctly."""
    return jd["title"].map(lambda t: f"{t.split('_')[0]}_{t.split('_')[-1]}")


def _tool_counts(df):
    """(edits, diagnostics) summed over every episode's action_trace."""
    e = d = 0
    for tr in df["action_trace"]:
        for t in (tr or []):
            if t.get("tool") == "edit_source":
                e += 1
            elif t.get("tool") == "run_diagnostic":
                d += 1
    return e, d


# ── COMBINED PANEL 1/2: 8-line accuracy and PDE control ─────────────────────

def p_lines(arms, metric, title, ylab):
    """One line per (arm, stage) = 8 lines. Colour = arm, shade+dash = stage.
    legendgroup per arm so the legend reads as four groups of two rather than
    eight flat entries."""
    fig = go.Figure()
    conds = [m for m in MOD_ORDER if m in set(arms[0]["df"]["mod_type"].unique())]
    labels = [MOD_SHORT.get(c, c) for c in conds]
    for i, arm in enumerate(arms):
        light, dark = ARM_COLORS[i % 4]
        for stage, colour, dash in (("s1", light, "dash"), ("s2", dark, "solid")):
            ys, lo_e, hi_e = [], [], []
            for c in conds:
                sub = arm["df"][arm["df"]["mod_type"] == c][f"{stage}_{metric}"].dropna()
                lo, mean, hi = bootstrap_ci(sub)
                ys.append(mean * 100); lo_e.append((mean - lo) * 100); hi_e.append((hi - mean) * 100)
            fig.add_trace(go.Scatter(
                x=labels, y=ys, mode="lines+markers", name=stage.upper(),
                line=dict(color=colour, dash=dash, width=2), marker=dict(size=7),
                error_y=dict(type="data", symmetric=False, array=hi_e, arrayminus=lo_e,
                             thickness=1.2, width=3),
                legendgroup=arm["label"], legendgrouptitle_text=arm["label"],
            ))
    fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 105], title_text=ylab)
    fig.update_layout(title=title, font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1350, height=620, plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_tickangle=30)
    return fig


def p_auc_lines(arms):
    """Score AUC per valid/invalid PAIRING -- 8 lines, same colour scheme as the
    accuracy panel.

    x has 4 points, not 8: AUC is intrinsically a comparison between the two
    classes, so each point merges a valid mod_type with its own invalid
    counterpart (Comm_Valid vs Comm_InValid, and so on). Reading it next to the
    accuracy panel separates DISCRIMINATION (can the model rank clean above
    defective within a lexical condition?) from the binary call's accuracy."""
    fig = go.Figure()
    labels = [MOD_SHORT.get(p, p) for p in PAIRING_ORDER]
    for i, arm in enumerate(arms):
        light, dark = ARM_COLORS[i % 4]
        df = arm["df"]
        for stage, colour, dash in (("s1", light, "dash"), ("s2", dark, "solid")):
            col = f"{stage}_validity_score"
            ys = []
            for valid_mt in PAIRING_ORDER:
                inv_mt = next((k for k, v in INVALID_TO_VALID_MOD_TYPE.items() if v == valid_mt), None)
                pos = [int(x) for x in df[df["mod_type"] == valid_mt][col].dropna()]
                neg = [int(x) for x in df[df["mod_type"] == inv_mt][col].dropna()] if inv_mt else []
                ys.append(score_auc(pos, neg))
            fig.add_trace(go.Scatter(
                x=labels, y=ys, mode="lines+markers", name=stage.upper(),
                line=dict(color=colour, dash=dash, width=2), marker=dict(size=7),
                legendgroup=arm["label"], legendgrouptitle_text=arm["label"],
                hovertemplate=f"{arm['label']} {stage.upper()}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#aaa",
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1.06], title_text="AUC (valid vs its own invalid twin)")
    fig.update_layout(title="Score ranking AUC by mod_type — S1 vs S2, all arms",
                      font=FONT, legend=LEGEND, margin=MARGIN, width=1350, height=620,
                      plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=30)
    return fig


# ── COMBINED PANEL 3: summary accuracy + AUC ────────────────────────────────

def p_summary(arms):
    """Accuracy and AUC per arm, with clustered 95% CIs and the paired S2-S1
    delta annotated.

    The delta is annotated deliberately: S1 and S2 are the same episodes, so
    reading overlap between the grey and blue intervals as "no effect" is wrong
    and would UNDERSTATE the result (tb0's marginals overlap heavily while its
    paired delta is +9pp with an interval clear of zero)."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=["Validity accuracy (binary call)",
                                        "Validity score AUC"])
    x = [a["label"] for a in arms]
    # panel 1 is a percentage (0-100), panel 2 a raw AUC (0-1) -- separate axes
    for col, fn, fmt, rng, dtick, ytitle, dfn, dfmt in (
            (1, _acc, "{:.1f}", [0, 128], None, "Accuracy (%)",
             _paired_delta, "{:+.1f}pp [{:+.1f}, {:+.1f}]"),
            (2, _auc, "{:.3f}", [0, 1.28], 0.2, "AUC",
             _paired_auc_delta, "{:+.3f} [{:+.3f}, {:+.3f}]")):
        for stage, colour in (("s1", "#7f8c8d"), ("s2", "#2980b9")):
            tri = [_cluster_boot(
                a["df"], lambda d, st=stage: fn(d, st),
                name=f"{ytitle}|{stage}|{a['label']}",
                fp=[a["df"][c] for c in ([f"{stage}_valid_match"] if col == 1
                    else [f"{stage}_validity_score", "gt_valid"]) if c in a["df"].columns])
                for a in arms]
            ys = [t[1] for t in tri]
            fig.add_trace(go.Bar(
                x=x, y=ys, name=f"{stage.upper()} ({'static read' if stage=='s1' else 'agentic'})",
                marker_color=colour, legendgroup=stage, showlegend=(col == 1),
                error_y=_errs(tri),
                text=[fmt.format(y) for y in ys],
                # INSIDE and MIDDLE: plotly places outside text above the bar
                # but not above the whisker, and anchoring to the bar top puts
                # the label straight through the whisker's lower half, which is
                # drawn INTO the bar. The bar midpoint clears both.
                textposition="inside", insidetextanchor="middle",
                textfont=dict(color="white", size=10),
            ), row=1, col=col)
        # paired S2-S1 delta, above each arm, clear of the whiskers
        top = rng[1]
        for i, a in enumerate(arms):
            lo, pt, hi = dfn(a["df"])
            sig = "" if lo <= 0 <= hi else "<b>"
            end = "" if lo <= 0 <= hi else "</b>"
            fig.add_annotation(
                x=x[i], y=top * 0.955, showarrow=False, row=1, col=col,
                text=f"{sig}&#916; {dfmt.format(pt, lo, hi)}{end}",
                font=dict(size=9, color="#1a5276" if not sig else "#0b3d5c"))
        # explicit tickvals so the 2% of headroom does not add a 102 / 1.02 tick
        fig.update_yaxes(range=rng, tickmode="array",
                         tickvals=[0, 20, 40, 60, 80, 100] if col == 1
                                  else [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         row=1, col=col, title_text=ytitle)
    fig.update_layout(title="Summary: static reading vs agentic execution, all four arms",
                      barmode="group", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1350, height=560, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(tickangle=20)
    return fig


# ── COMBINED PANEL 4: Agent choices ─────────────────────────────────────────

def p_agent_choices(arms):
    """Absolute actions per episode, split by tool, with the composition % as an
    annotation.

    Deliberately absolute rather than percentage-only: composition tracks the
    THINKING level, not model strength. 3.7-flash-low is the most edit-heavy arm
    of all four by percentage (88%) while doing the fewest actions of any arm
    (1.20/episode), so a %-only chart would show it as the worst offender and
    invert the actual finding. Volume is what separates the models; the % table
    at the end of the panel keeps the composition view available."""
    fig = go.Figure()
    x = [a["label"] for a in arms]
    # Per-row tool counts are derived ONCE per arm. The bootstrap statistic is
    # then a column mean, instead of re-walking every episode's action_trace on
    # all 4000 draws -- which dominated the runtime of this panel.
    e_tri, d_tri = [], []
    for a in arms:
        df = a["df"]
        cnt = df["action_trace"].map(
            lambda tr: (sum(1 for t in (tr or []) if t.get("tool") == "edit_source"),
                        sum(1 for t in (tr or []) if t.get("tool") == "run_diagnostic")))
        d2 = df.assign(_edits=[c[0] for c in cnt], _diags=[c[1] for c in cnt])
        e_tri.append(_cluster_boot(d2, lambda d: d["_edits"].mean(),
                                   name=f"edits_per_ep|{a['label']}", fp=[d2["_edits"]]))
        d_tri.append(_cluster_boot(d2, lambda d: d["_diags"].mean(),
                                   name=f"probes_per_ep|{a['label']}", fp=[d2["_diags"]]))
    edits = [t[1] for t in e_tri]
    diags = [t[1] for t in d_tri]
    # Same grey/blue pair as the Summary panel (#7f8c8d / #2980b9), so a reader
    # carries one colour convention across the two cross-arm bar charts.
    # Value labels are annotations above the UPPER WHISKER, not trace text:
    # plotly puts "outside" text at the bar top, which the whisker runs through,
    # and the 0.15 bar is far too short for an inside label.
    fig.add_trace(go.Bar(x=x, y=edits, name="edit_source (rewrite)",
                         marker_color="#7f8c8d", error_y=_errs(e_tri)))
    fig.add_trace(go.Bar(x=x, y=diags, name="run_diagnostic (read-only probe)",
                         marker_color="#2980b9", error_y=_errs(d_tri)))
    # 2 offsetgroups at -OFF / +OFF from the category centre
    OFF = (0.8 / 2 / 2) * (1200 - MARGIN["l"] - MARGIN["r"]) / len(arms)
    for shift, tri in ((-OFF, e_tri), (OFF, d_tri)):
        for i, t in enumerate(tri):
            fig.add_annotation(x=x[i], y=t[2], yshift=9, xshift=shift,
                               showarrow=False, text=f"{t[1]:.2f}",
                               font=dict(size=10, color="#333"))
    # headroom must clear the upper whisker, not just the bar
    hi_max = max([t[2] for t in e_tri] + [t[2] for t in d_tri])
    for i, (e, d) in enumerate(zip(edits, diags)):
        tot = e + d
        fig.add_annotation(x=x[i], y=hi_max * 1.16, showarrow=False,
                           text=f"<b>{tot:.2f} actions/ep</b><br>{100*e/tot:.0f}% edit / {100*d/tot:.0f}% probe",
                           font=dict(size=10, color="#444"))
    fig.update_yaxes(range=[0, hi_max * 1.34], title_text="Actions per episode")
    fig.update_layout(title="Agent choices: what the investigative budget was spent on",
                      barmode="group", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1200, height=560, plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_tickangle=20)

    # ---- stopping-behaviour table ----
    rows = []
    for a in arms:
        df = a["df"]
        acts = [x for x in df["s2_action_count"].dropna()]
        forced = df[df["actions_remaining_at_submission"] == 0]
        vol = df[df["actions_remaining_at_submission"] != 0]
        MIN_SUBSET = 10

        def cell(sub, fn, fmt):
            # Both AUC and accuracy are meaningless on a handful of rows, and AUC
            # additionally needs both classes present. 3.7 has 0 and 2 forced
            # episodes, so its forced cells are genuinely undefined -- that
            # emptiness IS the result. Printing "0.0%" off n=2 would read as a
            # catastrophic score rather than as no data, so anything under
            # MIN_SUBSET rows reports n/a with the count shown alongside.
            if len(sub) < MIN_SUBSET:
                return "n/a"
            v = fn(sub)
            return "n/a" if v != v else fmt.format(v)
        rows.append([
            a["label"], f"{statistics.median(acts):.0f}", f"{statistics.mean(acts):.2f}",
            f"{100*len(vol)/len(df):.0f}%", str(len(vol)),
            cell(vol, lambda d: _auc(d, "s2"), "{:.3f}"), cell(vol, lambda d: _acc(d, "s2"), "{:.1f}"),
            f"{100*len(forced)/len(df):.0f}%", str(len(forced)),
            cell(forced, lambda d: _auc(d, "s2"), "{:.3f}"), cell(forced, lambda d: _acc(d, "s2"), "{:.1f}"),
        ])
    tbl = _pre_table(
        ["arm", "med acts", "mean acts", "voluntary %", "n", "AUC", "acc %",
         "forced %", "n", "AUC", "acc %"], rows,
        caption="Table 1 — stopping behaviour. 'Voluntary' = submitted with budget "
                "remaining; 'forced' = budget exhausted. n/a where the subset is too "
                "small to compute (3.7 has 0 and 2 forced episodes respectively -- that "
                "emptiness is the result, not missing data).")

    # ---- composition table, per the panel spec ----
    crows = []
    for a in arms:
        e, d = _tool_counts(a["df"]); n = len(a["df"]); tot = e + d
        crows.append([a["label"], f"{e/n:.2f}", f"{d/n:.2f}", f"{tot/n:.2f}",
                      f"{100*e/tot:.1f}%", f"{100*d/tot:.1f}%", f"{e/max(d,1):.1f} : 1"])
    tbl += _pre_table(
        ["arm", "edits/ep", "probes/ep", "actions/ep", "edit %", "probe %", "edit:probe"],
        crows,
        caption="Table 2 — composition. Read alongside the absolute counts above: the "
                "percentages track the thinking level rather than model strength, so "
                "3.7-flash-low looks the most edit-heavy (88%) while spending the fewest "
                "actions of any arm.")
    return fig, tbl


def p_reasoning_quality(arms, show_ci=False, paper=False):
    """Judge results for all four arms in one figure, Stage 2 only.

    Three bar positions per arm, all percentages of that arm's judged rows:
      1. category, STACKED none/some/all -- sums to 100% by construction.
      2. false code claims -- asserts something untrue about the code.
      3. runtime claims -- cites runtime detail the judge cannot verify.

    The three bars mean structurally different things while sharing one axis, so
    they are colour-coded into two families and labelled by what they measure:
    category is RECALL (did the justification name the real defect), false code
    claims is PRECISION (did it also assert something false), and runtime claims
    is DESCRIPTIVE -- it sits at 80-90% in every arm, so it does not discriminate
    and must not be read as a headline. It is included because its near-constancy
    is itself the point.

    show_ci adds clustered 95% CIs (HTML only -- the exported PDF is built with
    show_ci=False so it stays unchanged). Only the two SIMPLE PROPORTION bars
    get whiskers: per-segment intervals on the stacked none/some/all composition
    are not interpretable, since the segments are constrained to sum to 100. The
    stack instead carries an annotated interval on the combined some+all rate,
    which is the quantity of interest (did it name the real defect at all)."""
    fig = go.Figure()
    x = [a["label"] for a in arms]
    stats = []
    for a in arms:
        jd = a["judge"]
        n = max(len(jd), 1)
        stats.append({
            "n": len(jd),
            "none": 100 * (jd["category"] == "none").sum() / n if len(jd) else 0,
            "some": 100 * (jd["category"] == "some").sum() / n if len(jd) else 0,
            "all": 100 * (jd["category"] == "all").sum() / n if len(jd) else 0,
            "bad": 100 * jd["contains_incorrect_code_claims"].fillna(False).sum() / n if len(jd) else 0,
            "exe": 100 * jd["contains_unverifiable_execution_claims"].fillna(False).sum() / n if len(jd) else 0,
        })
    # Clustered intervals, keyed on the gt_sample derived from `title` (judge
    # rows carry no gt_sample column). Clusters are unequal in size because
    # eligibility is conditional on the model having answered correctly.
    tri_bad = tri_exe = tri_named = None
    if show_ci:
        tri_bad, tri_exe, tri_named = [], [], []
        for a in arms:
            jd = a["judge"]
            if not len(jd):
                z = (0.0, 0.0, 0.0)
                tri_bad.append(z); tri_exe.append(z); tri_named.append(z); continue
            jd = jd.reset_index(drop=True).assign(gt_sample=_judge_gt_sample(jd.reset_index(drop=True)))
            tri_bad.append(_cluster_boot(
                jd, lambda d: 100 * d["contains_incorrect_code_claims"].fillna(False).mean(),
                name=f"judge_bad|{a['label']}", fp=[jd["contains_incorrect_code_claims"]]))
            tri_exe.append(_cluster_boot(
                jd, lambda d: 100 * d["contains_unverifiable_execution_claims"].fillna(False).mean(),
                name=f"judge_exe|{a['label']}", fp=[jd["contains_unverifiable_execution_claims"]]))
            tri_named.append(_cluster_boot(
                jd, lambda d: 100 * d["category"].isin(["some", "all"]).mean(),
                name=f"judge_named|{a['label']}", fp=[jd["category"]]))
    # bar 1: the stacked category composition
    for cat, colour in (("none", "#c0392b"), ("some", "#f39c12"), ("all", "#27ae60")):
        fig.add_trace(go.Bar(
            x=x, y=[st[cat] for st in stats], name=f"category = {cat}",
            marker_color=colour, offsetgroup="category", legendgroup="category",
            legendgrouptitle_text="Recall: did it name the real defect(s)?",
            text=[f"{st[cat]:.0f}" if st[cat] >= 6 else "" for st in stats],
            textposition="inside", insidetextanchor="middle",
            hovertemplate="%{x}<br>category=" + cat + ": %{y:.1f}%<extra></extra>"))
    # bar 2: precision
    fig.add_trace(go.Bar(
        x=x, y=[st["bad"] for st in stats], name="false code claims",
        marker_color="#7b241c", marker_pattern_shape="x",
        offsetgroup="bad", legendgroup="precision",
        legendgrouptitle_text="Precision / description",
        error_y=_errs(tri_bad) if show_ci else None,
        text=None if show_ci else [f"{st['bad']:.1f}%" for st in stats],
        textposition="outside",
        hovertemplate="%{x}<br>false code claims: %{y:.1f}%<extra></extra>"))
    # bar 3: descriptive, explicitly de-emphasised
    fig.add_trace(go.Bar(
        x=x, y=[st["exe"] for st in stats], name="runtime claims mentioned",
        marker_color="#aeb6bf", marker_pattern_shape="/",
        offsetgroup="exec", legendgroup="precision",
        error_y=_errs(tri_exe) if show_ci else None,
        text=None if show_ci else [f"{st['exe']:.1f}%" for st in stats],
        textposition="outside",
        hovertemplate="%{x}<br>runtime claims: %{y:.1f}%<extra></extra>"))
    for i, st in enumerate(stats):
        fig.add_annotation(x=x[i], y=-9, showarrow=False, text=f"n={st['n']}",
                           font=dict(size=10, color="#666"))
    if show_ci:
        # Interval on the STACK as a whole (some+all = named the defect), since
        # the individual segments are constrained to sum to 100.
        #
        # The three offsetgroups sit at -OFF / 0 / +OFF from the category centre;
        # OFF = (1-bargap)/ngroups * plot_width/ncategories. Value+interval is
        # annotated above the UPPER WHISKER rather than left as trace text,
        # which plotly would place above the bar and under the whisker.
        OFF = (0.8 / 3) * (1300 - MARGIN["l"] - MARGIN["r"]) / len(arms)
        for i, t in enumerate(tri_named):
            fig.add_annotation(
                x=x[i], y=106, showarrow=False, xshift=-OFF,
                text=f"named {t[1]:.0f}% [{t[0]:.0f}, {t[2]:.0f}]",
                font=dict(size=8.5, color="#1e8449"))
        for shift, tri, col in ((0, tri_bad, "#7b241c"), (OFF, tri_exe, "#616a6b")):
            for i, t in enumerate(tri):
                fig.add_annotation(
                    x=x[i], y=t[2] + 3.5, showarrow=False, xshift=shift,
                    text=f"{t[1]:.1f}% [{t[0]:.0f}, {t[2]:.0f}]",
                    font=dict(size=8.5, color=col))
    fig.update_yaxes(range=[-14, 118], title_text="% of judged rows")
    fig.update_layout(
        title=None if paper else
              "Reasoning quality across arms (Stage 2): recall, precision, and runtime claims",
        barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
        width=1300, height=600, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=15)
    if not paper:
        return fig

    # ---- paper variant: Times, and the two-tier x axis used by fig1b/fig2b ----
    W, H, M = 1300, 560, dict(l=76, r=210, t=16, b=104)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.update_layout(width=W, height=H, margin=M,
                      font=dict(family="Times New Roman, Times, serif",
                                size=15, color="#222"),
                      # match fig1b/fig2b: no tinted panel behind the legend
                      legend=dict(bgcolor="rgba(255,255,255,0)", bordercolor="#999",
                                  borderwidth=0.6, font=dict(size=14)))
    fig.update_xaxes(tickangle=0, tickmode="array", tickvals=x,
                     ticktext=[REORG_ARM_HEADER.get(a["label"], ("", a["label"]))[1]
                               .replace("thinking", "think") for a in arms],
                     tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15), title_font=dict(size=17))
    PH = H - M["t"] - M["b"]
    RULE_Y, FAM_Y = -34 / PH, -38 / PH
    fams = [REORG_ARM_HEADER.get(a["label"], ("", ""))[0] for a in arms]
    seen = {}
    for i, fam in enumerate(fams):
        seen.setdefault(fam, []).append(i)
    for fam, idxs in seen.items():
        a0 = min(idxs) / len(arms) + 0.004
        b0 = (max(idxs) + 1) / len(arms) - 0.004
        fig.add_shape(type="line", x0=a0, x1=b0, y0=RULE_Y, y1=RULE_Y,
                      xref="paper", yref="paper", line=dict(color="#999", width=0.8))
        fig.add_annotation(x=(a0 + b0) / 2, y=FAM_Y, xref="paper", yref="paper",
                           text=fam, showarrow=False, xanchor="center",
                           yanchor="top", font=dict(size=16))
    return fig


# ── PAPER FIGURE 1: merged summary + agent choices (EXPORT ONLY) ────────────

def p_paper_fig1(arms):
    """Three panels side by side: validity accuracy | score AUC | actions per
    episode. Built ONLY for the exported PDF -- the HTML report keeps Summary and
    Agent choices as separate panels in their own palettes.

    Panels 1-2 keep the grey/blue S1/S2 pair. Panel 3 is recoloured to orange and
    green: merged into one figure, reusing grey/blue there would make the same
    two colours mean S1/S2 on the left and edit/probe on the right.

    Two legends, each inside its own panel (plotly >=5.15 multi-legend): the
    S1/S2 key in panel 1, the edit/probe key in panel 3. Panel 3's legend sits
    at the right, below the annotation row and above the 3.7 arms' bars
    (0.15-1.27 against a top of ~3.4) -- that band is empty."""
    EDIT_C, PROBE_C = "#d35400", "#1e8449"
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.075,
                        subplot_titles=["Validity accuracy (binary call)",
                                        "Validity score AUC",
                                        "Investigative actions per episode"])
    x = [a["label"] for a in arms]

    # panels 1-2: identical series to p_summary()
    # panel 1 is a percentage (0-100), panel 2 a raw AUC (0-1) -- separate axes
    for col, fn, fmt, rng, dtick, ytitle in (
            (1, _acc, "{:.1f}", [0, 112], None, "Accuracy (%)"),
            (2, _auc, "{:.3f}", [0, 1.12], 0.2, "AUC")):
        for stage, colour in (("s1", "#7f8c8d"), ("s2", "#2980b9")):
            ys = [fn(a["df"], stage) for a in arms]
            fig.add_trace(go.Bar(
                x=x, y=ys,
                # "static"/"agentic" alone -- the S1/S2 prefix carried no
                # information and cost the width that keeps the legend inside
                # the panel's free band.
                name="static" if stage == "s1" else "agentic",
                marker_color=colour, legendgroup=stage, showlegend=(col == 1),
                legend="legend",
                text=[fmt.format(y) for y in ys],
                textposition="outside", textfont=dict(size=10),
            ), row=1, col=col)
        # explicit tickvals so the 2% of headroom does not add a 102 / 1.02 tick
        fig.update_yaxes(range=rng, tickmode="array",
                         tickvals=[0, 20, 40, 60, 80, 100] if col == 1
                                  else [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         row=1, col=col, title_text=ytitle)

    # panel 3: absolute actions per episode, split by tool
    edits, probes = [], []
    for a in arms:
        e, d = _tool_counts(a["df"])
        n = len(a["df"])
        edits.append(e / n)
        probes.append(d / n)
    top = max(edits + probes)
    for ys, nm, colour in ((edits, "edit_source", EDIT_C),
                           (probes, "run_diagnostic", PROBE_C)):
        fig.add_trace(go.Bar(x=x, y=ys, name=nm, marker_color=colour,
                             legend="legend2", showlegend=True,
                             text=[f"{v:.2f}" for v in ys], textposition="outside",
                             textfont=dict(size=10)), row=1, col=3)
    for i, (e, d) in enumerate(zip(edits, probes)):
        tot = e + d
        fig.add_annotation(x=x[i], y=top * 1.12, showarrow=False,
                           text=f"<b>{tot:.2f}</b> total actions"
                                f"<br>{100*e/tot:.0f}% edit_source"
                                f"<br>{100*d/tot:.0f}% run_diagnostic",
                           font=dict(size=8.5, color="#444"), row=1, col=3)
    fig.update_yaxes(range=[0, top * 1.22], title_text="Actions per episode", row=1, col=3)

    fig.update_layout(
        barmode="group", font=FONT,
        margin=dict(l=64, r=26, t=54, b=86),
        width=1900, height=540, plot_bgcolor="white", paper_bgcolor="white",
        # both legends sit over their own subplot, in paper coordinates
        legend=dict(x=0.005, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#bbb", borderwidth=1,
                    font=dict(size=10.5)),
        legend2=dict(x=0.995, y=0.60, xanchor="right", yanchor="top",
                     bgcolor="rgba(255,255,255,0.88)", bordercolor="#bbb", borderwidth=1,
                     font=dict(size=10.5)),
    )
    fig.update_xaxes(tickangle=18, tickfont=dict(size=10))
    return fig


def p_paper_fig1b(arms):
    """Export-only rebuild of agent_fig1: accuracy | AUC | actions.

    Differences from p_paper_fig1, all requested: clustered CIs added, subplot
    titles dropped, paired-delta annotations dropped, panel-3 total/percentage
    annotations dropped, every value label moved to white text inside its bar,
    and the axes capped at their natural ceilings (100 / 1.0 / 3.5).

    Authored at 2x the NeurIPS text width (792 = 2 x 396pt) so the fonts below
    print at half their value -- the old 1900pt canvas scaled to 0.21x and took
    10pt text down to ~2pt. Times, to match agent_fig2b.

    Cache names are identical to p_summary's and p_agent_choices', so this
    figure reuses their cached intervals rather than recomputing.
    """
    EDIT_C, PROBE_C = "#d35400", "#1e8449"
    F_BAR, F_XTICK, F_YTICK, F_AXT, F_LEG, F_FAM = 9.5, 10.0, 12.0, 14.0, 9.0, 13.0
    # 258 -> 222 pt saves 36pt authored = 18pt printed, about 1.5 lines of
    # NeurIPS body text. t/b hold the top tick label and the two-tier x axis
    # (ticks + family rule + family label); trimming them below this clips both.
    FIG_W, FIG_H = 792, 228
    MARG = dict(l=29, r=2, t=10, b=58)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05)
    x = [a["label"] for a in arms]
    # Two-tier x labels, as in agent_fig2b: the thinking descriptor per tick and
    # the model family on a spanning rule beneath. "thinking" -> "think" keeps
    # the longest label ("think (8192)", ~50pt at 10pt Times) inside a ~56pt
    # category, so every tick fits on ONE line.
    tick_txt = [REORG_ARM_HEADER.get(a["label"], ("", a["label"]))[1]
                .replace("thinking", "think") for a in arms]
    fams = [REORG_ARM_HEADER.get(a["label"], ("", ""))[0] for a in arms]

    for col, fn, fmt, rng, dtick, ytitle, mcol in (
            (1, _acc, "{:.1f}", [0, 102], 20, "Validity accuracy (%)", "Accuracy (%)"),
            (2, _auc, "{:.3f}", [0, 1.02], 0.2, "Validity AUC", "AUC")):
        # 4 characters max per label -- see the textangle note below
        for stage, colour in (("s1", "#7f8c8d"), ("s2", "#2980b9")):
            tri = [_cluster_boot(
                a["df"], lambda d, st=stage: fn(d, st),
                name=f"{mcol}|{stage}|{a['label']}",
                fp=[a["df"][c] for c in ([f"{stage}_valid_match"] if col == 1
                    else [f"{stage}_validity_score", "gt_valid"]) if c in a["df"].columns])
                for a in arms]
            ys = [t[1] for t in tri]
            fig.add_trace(go.Bar(
                x=x, y=ys,
                # "static"/"agentic" alone -- the S1/S2 prefix carried no
                # information and cost the width that keeps the legend inside
                # the panel's free band.
                name="static" if stage == "s1" else "agentic",
                marker_color=colour, legendgroup=stage, showlegend=(col == 1),
                legend="legend", error_y=_errs(tri),
                # textangle=0 is load-bearing: 8 bars share ~215pt, so plotly
                # rotates a label that does not fit rather than overflowing it.
                # Sizing every label to 4 characters (AUC drops its leading
                # zero) keeps them horizontal at a legible size.
                text=[fmt.format(y).lstrip("0") if col == 2 else fmt.format(y)
                      for y in ys],
                textposition="inside", insidetextanchor="middle", textangle=0,
                constraintext="none",
                insidetextfont=dict(color="white", size=F_BAR),
                outsidetextfont=dict(color="#333", size=F_BAR),
            ), row=1, col=col)
        # explicit tickvals so the 2% of headroom does not add a 102 / 1.02 tick
        fig.update_yaxes(range=rng, tickmode="array",
                         tickvals=[0, 20, 40, 60, 80, 100] if col == 1
                                  else [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         row=1, col=col, title_text=ytitle)

    # panel 3: actions per episode, split by tool
    e_tri, d_tri = [], []
    for a in arms:
        df = a["df"]
        cnt = df["action_trace"].map(
            lambda tr: (sum(1 for t in (tr or []) if t.get("tool") == "edit_source"),
                        sum(1 for t in (tr or []) if t.get("tool") == "run_diagnostic")))
        d2 = df.assign(_edits=[c[0] for c in cnt], _diags=[c[1] for c in cnt])
        e_tri.append(_cluster_boot(d2, lambda d: d["_edits"].mean(),
                                   name=f"edits_per_ep|{a['label']}", fp=[d2["_edits"]]))
        d_tri.append(_cluster_boot(d2, lambda d: d["_diags"].mean(),
                                   name=f"probes_per_ep|{a['label']}", fp=[d2["_diags"]]))
    # A bar shorter than this cannot hold a white label; those few are annotated
    # ABOVE THEIR UPPER WHISKER instead. Letting plotly place them "outside"
    # puts the number on the whisker, and forcing them inside would paint white
    # text onto white paper.
    P3_TOP, MIN_INSIDE = 3.5, 0.30
    OFF = 0.228 * ((1 - 2 * 0.065) / 3) * (792 - 52 - 10) / len(arms)
    for tri, nm, colour, shift in ((e_tri, "edit_source", EDIT_C, -OFF),
                                   (d_tri, "run_diagnostic", PROBE_C, OFF)):
        fig.add_trace(go.Bar(
            x=x, y=[t[1] for t in tri], name=nm, marker_color=colour,
            legend="legend2", showlegend=True, error_y=_errs(tri),
            text=[f"{t[1]:.2f}" if t[1] >= MIN_INSIDE else "" for t in tri],
            textposition="inside", insidetextanchor="middle", textangle=0,
            constraintext="none",
            insidetextfont=dict(color="white", size=F_BAR),
        ), row=1, col=3)
        for i, t in enumerate(tri):
            if t[1] < MIN_INSIDE:
                fig.add_annotation(x=x[i], y=t[2], yshift=8, xshift=shift,
                                   showarrow=False, text=f"{t[1]:.2f}",
                                   font=dict(size=F_BAR, color="#333"),
                                   row=1, col=3)
    # 3.5 is the cap, not a gridline worth labelling -- its tick sits at the very
    # top of the panel and runs into the legend.
    fig.update_yaxes(range=[0, P3_TOP], tickmode="array",
                     tickvals=[0, 0.5, 1, 1.5, 2, 2.5, 3],
                     title_text="Actions per episode", row=1, col=3)

    fig.update_layout(
        barmode="group", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman, Times, serif", size=F_YTICK, color="#222"),
        width=FIG_W, height=FIG_H, margin=MARG,
        bargap=0.14, bargroupgap=0.06,
        # Unboxed, two tight rows, INSIDE the panels: top-left of the accuracy
        # panel and top-right of the actions panel. Dropping the box and the
        # S1/S2 prefix is what makes them narrow enough to sit in the free band
        # without covering a bar or an interval (verified by pixel diff).
        # Plotly's own legends are off: their box ignores font size, they carry
        # unavoidable internal padding, and anchoring above y=1 trips automargin.
        # Both legends are drawn by hand below, so position and box size are
        # exact rather than negotiated.
        showlegend=False,
    )
    # kaleido draws a faint outline on bars (and therefore on the legend
    # swatches) unless the marker line is explicitly zeroed
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.update_xaxes(tickangle=0, tickmode="array", tickvals=x, ticktext=tick_txt,
                     tickfont=dict(size=F_XTICK), ticklabelstandoff=1)
    # standoff pulls the tick numbers and the axis title in toward the bars
    fig.update_yaxes(tickfont=dict(size=F_YTICK), title_font=dict(size=F_AXT),
                     ticklabelstandoff=-1, title_standoff=4)

    # ---- hand-drawn legends: swatch + label + a tight box, in points ----
    PW = FIG_W - MARG["l"] - MARG["r"]
    PH = FIG_H - MARG["t"] - MARG["b"]

    def _px(pt):
        return (pt - MARG["l"]) / PW

    def _py(pt):
        return 1 - (pt - MARG["t"]) / PH

    def _legend(entries, x_pt, y_pt, align="left", fs=F_LEG):
        SW, GAP, PAD, PITCH = 7.0, 3.5, 4.0, 11.5
        tw = max(len(t) for _, t in entries) * fs * 0.42
        w = PAD * 2 + SW + GAP + tw
        h = PAD * 2 + PITCH * len(entries)
        x0 = x_pt if align == "left" else x_pt - w
        fig.add_shape(type="rect", xref="paper", yref="paper",
                      x0=_px(x0), x1=_px(x0 + w), y0=_py(y_pt + h), y1=_py(y_pt),
                      line=dict(color="#555", width=0.6), fillcolor="rgba(0,0,0,0)",
                      layer="above")
        for i, (colour, label) in enumerate(entries):
            cy = y_pt + PAD + PITCH * (i + 0.5)
            fig.add_shape(type="rect", xref="paper", yref="paper",
                          x0=_px(x0 + PAD), x1=_px(x0 + PAD + SW),
                          y0=_py(cy + SW / 2), y1=_py(cy - SW / 2),
                          line=dict(width=0), fillcolor=colour, layer="above")
            fig.add_annotation(x=_px(x0 + PAD + SW + GAP), y=_py(cy),
                               xref="paper", yref="paper", text=label,
                               showarrow=False, xanchor="left", yanchor="middle",
                               font=dict(size=fs, color="#222"))

    # panel a: top-left, inside the band above the tb0 interval (which tops at 76)
    _legend([("#7f8c8d", "static"), ("#2980b9", "agentic")],
            MARG["l"] + 5.0, MARG["t"] + 2.0)
    # panel c: top-right, above the 3.7 bars (which top near 1.4 of 3.5)
    _legend([(EDIT_C, "edit_source"), (PROBE_C, "run_diagnostic")],
            FIG_W - MARG["r"] - 2.0, MARG["t"] + 2.0, align="right")

    # ---- family tier: one rule + label per family, per panel ----
    plot_h = PH
    def pyb(px):
        return -px / plot_h
    RULE_Y, FAM_Y = pyb(22), pyb(25)
    doms = [fig.layout[f"xaxis{'' if i == 1 else i}"].domain for i in (1, 2, 3)]
    for d0, d1 in doms:
        w = (d1 - d0) / len(arms)
        seen = {}
        for i, fam in enumerate(fams):
            seen.setdefault(fam, []).append(i)
        for fam, idxs in seen.items():
            a = d0 + min(idxs) * w
            b = d0 + (max(idxs) + 1) * w
            # inset slightly so the two families' rules read as separate spans
            a, b = a + 0.004, b - 0.004
            fig.add_shape(type="line", x0=a, x1=b, y0=RULE_Y, y1=RULE_Y,
                          xref="paper", yref="paper", line=dict(color="#999", width=0.8))
            fig.add_annotation(x=(a + b) / 2, y=FAM_Y, xref="paper", yref="paper",
                               text=fam, showarrow=False, xanchor="center",
                               yanchor="top", font=dict(size=F_FAM))
    return fig


# ── REORGANIZED TRANSITIONS FIGURE (2 blocks x 4 arms, normalized) ──────────

# Stack order is the legend order: both correct outcomes first, then both
# incorrect. Distinct from TRANS_ORDER (which interleaves them) but reuses
# TRANS_COLORS, so the colour meaning is identical to the 2x2 grid panel.
REORG_TRANS_ORDER = ["right->right", "wrong->right", "right->wrong", "wrong->wrong"]
REORG_TRANS_LABEL = {
    "right->right": "stayed right (right \u2192 right)",
    "wrong->right": "fixed (wrong \u2192 right)",
    "right->wrong": "broke (right \u2192 wrong)",
    "wrong->wrong": "stayed wrong (wrong \u2192 wrong)",
}
# Within-block row labels: the valid/invalid half is carried by the rotated block
# label, so each row names only its lexical condition.
REORG_BLOCKS = [
    ("VALID CODE",   ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar"]),
    ("INVALID CODE", ["Comm_InValid", "NoComm_InValid", "CorrComm_Invalid",
                      "NoComm_CorrVar_InValid"]),
]
REORG_ROW_LABEL = ["+ comment", "no comment", "corrupt comment", "obfuscated var"]
# Arm label -> (family, thinking descriptor). Keyed on the CLI arm label.
REORG_ARM_HEADER = {
    "2.5-flash-tb0":    ("2.5-flash", "no thinking"),
    "2.5-flash-tb8192": ("2.5-flash", "thinking (8192)"),
    "3.7-flash-low":    ("3.7-flash", "low thinking"),
    "3.7-flash-high":   ("3.7-flash", "high thinking"),
}
REORG_TRANS_SHORT = {
    "right->right": "stayed right",
    "wrong->right": "fixed",
    "right->wrong": "broke",
    "wrong->wrong": "stayed wrong",
}
# Below this share of the bar, a segment gets no in-bar number. The threshold is
# a fit constraint, not a taste one: text is never rotated or shrunk, so it must
# be set so that two digits fit at the mode's font size. The compact/PDF mode has
# panels ~1/2 the width at ~3/4 the font, hence the higher floor.
REORG_MIN_LABEL_PCT = 8.0
REORG_MIN_LABEL_PCT_COMPACT = 13.0


def p_transitions_reorganized(arms, *, show_s2=True, compact=False, title=True):
    """Transitions as a 2-block x 4-arm lattice of horizontal 100% stacked bars.

    Rows are the four lexical conditions; the valid/invalid half is carried by a
    rotated block label rather than by the row name. Bars are normalized to
    share of episodes (n=32 per bar), so the green+blue length IS that cell's
    Stage-2 accuracy. In-bar percentages are suppressed below
    REORG_MIN_LABEL_PCT, where white text would overflow its segment.

    compact=True is the paper-PDF geometry: whitespace squeezed hard, especially
    vertically. show_s2=False drops the right-of-bar Stage-2 numbers, which lets
    the x range end at 100 so the bars fill their panels. All header/label
    offsets are computed in PIXELS off the resolved plot box, not as fixed paper
    fractions -- a fraction that clears the rule at height 620 collides with it
    at 430."""
    if compact:
        # Authored at 2x the printed size: NeurIPS is single-column with
        # \textwidth = 5.5in = 396pt, and plotly's width/height become PDF
        # POINTS -- so a 1360-wide figure is a 14in page that LaTeX shrinks to
        # 0.39x, taking an 8pt label down to 3pt. At 792pt wide the page halves
        # to 396pt and every font below halves with it: 14->7, 15->7.5, 17->8.5,
        # 20->10. Those are the sizes that actually print.
        # Wide-and-short, per the mockup. Printed width is pinned at NeurIPS
        # \textwidth (5.5in = 396pt), so aspect is set by the canvas: 1100x390
        # prints as 5.5 x 1.95in at scale 396/1100 = 0.36. Fonts are authored at
        # printed_pt / 0.36, i.e. ~2.8x the number you want on the page.
        W, H, M = 1100, 390, dict(l=196, r=34, t=64, b=86)
        GAP_IN, GAP_FAM, VGAP = 0.030, 0.052, 0.10
        F_ROW, F_BAR, F_SUB, F_FAM, F_BLK, F_LEG = 19.4, 18.0, 21.5, 25.0, 15.0, 12.5
    else:
        W, H, M = 1500, 620, dict(l=130, r=30, t=120, b=96)
        GAP_IN, GAP_FAM, VGAP = 0.028, 0.075, 0.10
        F_ROW, F_BAR, F_SUB, F_FAM, F_BLK, F_LEG = 9.0, 8.5, 10.0, 12.0, 10.5, 10.0

    plot_w = W - M["l"] - M["r"]
    plot_h = H - M["t"] - M["b"]
    def py(px):   # pixels above the plot box -> paper y
        return 1.0 + px / plot_h
    def pyb(px):  # pixels below the plot box -> paper y
        return -px / plot_h

    fig = make_subplots(rows=2, cols=4, horizontal_spacing=GAP_IN,
                        vertical_spacing=VGAP)

    # Manual x-domains: plotly spaces columns uniformly, but the two model
    # families need a wider gap between them than the arms within a family.
    w = (1.0 - 2 * GAP_IN - GAP_FAM) / 4
    starts, x0 = [], 0.0
    for i in range(4):
        starts.append(x0)
        x0 += w + (GAP_FAM if i == 1 else GAP_IN)
    for c in range(4):
        for r in (1, 2):
            fig.update_xaxes(domain=[starts[c], starts[c] + w], row=r, col=c + 1)

    xmax = 118 if show_s2 else 100
    lab_map = REORG_TRANS_LABEL
    min_pct = REORG_MIN_LABEL_PCT_COMPACT if compact else REORG_MIN_LABEL_PCT
    for bi, (_, conds) in enumerate(REORG_BLOCKS):
        row = bi + 1
        for ci, arm in enumerate(arms):
            df, col = arm["df"], ci + 1
            # Natural order + autorange="reversed" puts the first condition at
            # the TOP. Reversing the lists as well would cancel that out.
            ys, order = list(REORG_ROW_LABEL), list(conds)
            tot = [max(int((df["mod_type"] == m).sum()), 1) for m in order]
            for t in REORG_TRANS_ORDER:
                pct = [100.0 * int((df[df["mod_type"] == m]["transition"] == t).sum()) / n
                       for m, n in zip(order, tot)]
                fig.add_trace(go.Bar(
                    y=ys, x=pct, orientation="h",
                    name=lab_map[t], legendgroup=t,
                    showlegend=(bi == 0 and ci == 0),
                    marker_color=TRANS_COLORS[t],
                    text=[f"{v:.0f}" if v >= min_pct else "" for v in pct],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="white", size=F_BAR), textangle=0,
                    constraintext="none", cliponaxis=False,
                ), row=row, col=col)
            if show_s2:
                for m, yl, n in zip(order, ys, tot):
                    d = df[df["mod_type"] == m]["transition"]
                    acc = 100.0 * int(((d == "right->right") | (d == "wrong->right")).sum()) / n
                    fig.add_annotation(x=103, y=yl, text=f"{acc:.0f}%", showarrow=False,
                                       xanchor="left", yanchor="middle",
                                       font=dict(size=F_BAR, color="#333"),
                                       row=row, col=col)
            fig.update_yaxes(autorange="reversed", row=row, col=col,
                             showticklabels=(col == 1), tickfont=dict(size=F_ROW),
                             ticklabelstandoff=7 if compact else 5,
                             showline=False, ticks="")
            fig.update_xaxes(range=[0, xmax], tickvals=[0, 50, 100],
                             ticktext=["0", "50", "100"],
                             showticklabels=(row == 2), tickfont=dict(size=F_BAR),
                             showgrid=False, zeroline=False, row=row, col=col)

    # Two-tier column headers. Stacked bottom-up off the plot box: subtitle,
    # then the family rule, then the family name above it.
    SUB_PX, RULE_PX = (8, 40) if compact else (5, 23)
    fam_span = {}
    for ci, arm in enumerate(arms):
        fam, sub = REORG_ARM_HEADER.get(arm["label"], ("", arm["label"]))
        fig.add_annotation(x=starts[ci] + w / 2, y=py(SUB_PX), xref="paper", yref="paper",
                           text=sub, showarrow=False, font=dict(size=F_SUB),
                           xanchor="center", yanchor="bottom")
        fam_span.setdefault(fam, []).append(ci)
    for fam, cols in fam_span.items():
        a, b = starts[min(cols)], starts[max(cols)] + w
        fig.add_shape(type="line", x0=a, x1=b, y0=py(RULE_PX), y1=py(RULE_PX),
                      xref="paper", yref="paper", line=dict(color="#999", width=1))
        fig.add_annotation(x=(a + b) / 2, y=py(RULE_PX + 3), xref="paper", yref="paper",
                           text=f"<b>{fam}</b>", showarrow=False,
                           font=dict(size=F_FAM), xanchor="center", yanchor="bottom")

    # Rotated block labels, left of the row labels
    # Anchored a fixed inset from the CANVAS edge, not offset from an estimated
    # row-label width: that estimate assumed ~0.58em/char and overshot badly in
    # Times (~0.42em), pushing the labels off the page entirely.
    blk_x = -(M["l"] - 18) / plot_w
    for bi, (blab, _) in enumerate(REORG_BLOCKS):
        centre = 1.0 - VGAP / 2 - (1.0 - VGAP) / 4 if bi == 0 else (1.0 - VGAP) / 4
        fig.add_annotation(x=blk_x, y=centre, xref="paper", yref="paper",
                           text=f"<b>{blab}</b>", showarrow=False, textangle=-90,
                           xanchor="center", yanchor="middle", font=dict(size=F_BLK))

    fig.update_layout(
        title=("Transitions by mod_type, reorganized \u2014 share of episodes "
               "(n=32 per bar); % right of each bar is Stage-2 accuracy") if title else None,
        barmode="stack", plot_bgcolor="white", paper_bgcolor="white",
        font=(dict(FONT, family="Times New Roman, Times, serif") if compact else FONT),
        width=W, height=H, margin=M, bargap=0.26 if compact else 0.30,
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=pyb(64 if compact else 52), yanchor="top",
                    font=dict(size=F_LEG)),
    )
    fig.add_annotation(x=0.5, y=pyb(34 if compact else 24), xref="paper", yref="paper",
                       text="share of episodes (%)", showarrow=False,
                       font=dict(size=F_SUB), xanchor="center", yanchor="top")
    return fig


# ── PER-ARM PANELS (rendered as four stacked figures) ───────────────────────

def p_transitions_grid(arms):
    """All four arms as a 2x2 grid of stacked transition counts.

    Above each bar: the Stage-2 rate for that mod_type -- labelled `spec` on the
    four valid mod_types and `rec` on the four invalid ones. Both are the same
    computation (mean of s2_valid_match, since a match means "correct" either
    way); only the name differs, because on valid code being right means calling
    it valid and on invalid code it means calling it invalid. Showing them side
    by side is the point: recall sits near ceiling in every arm and every lexical
    condition, while specificity is where the arms and the conditions separate."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[a["label"] for a in arms],
                        horizontal_spacing=0.07, vertical_spacing=0.15)
    for i, arm in enumerate(arms):
        r, c = i // 2 + 1, i % 2 + 1
        df = arm["df"]
        conds = [m for m in MOD_ORDER if m in set(df["mod_type"].unique())]
        xs = [MOD_SHORT.get(m, m) for m in conds]
        for t in TRANS_ORDER:
            fig.add_trace(go.Bar(
                name=TRANS_SHORT[t], x=xs,
                y=[int((df[df["mod_type"] == m]["transition"] == t).sum()) for m in conds],
                marker_color=TRANS_COLORS[t], legendgroup=t, showlegend=(i == 0)),
                row=r, col=c)
        for j, m in enumerate(conds):
            sub = df[df["mod_type"] == m]
            n = len(sub)
            rate = 100 * sub["s2_valid_match"].dropna().mean()
            # The rate IS specificity on the four valid mod_types and recall on the
            # four invalid ones -- the same computation either way, so the bare
            # percentage is unambiguous once the reader knows which half is which
            # (stated in the panel description). No tag, to keep the grid readable.
            colour = "#1a7a3c" if rate >= 85 else ("#b9770e" if rate >= 60 else "#a93226")
            fig.add_annotation(x=xs[j], y=n + 0.6, text=f"<b>{rate:.0f}%</b>",
                               showarrow=False, font=dict(size=9.5, color=colour),
                               yanchor="bottom", row=r, col=c)
        fig.update_yaxes(range=[0, 38], title_text="Episodes" if c == 1 else None, row=r, col=c)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8.5), row=r, col=c)
    # make_subplots places its titles as paper-referenced annotations well above
    # each cell; nudge them down so the gap between a title and its plot matches
    # the (now tighter) gap between rows.
    arm_labels = {a["label"] for a in arms}
    for ann in fig.layout.annotations:
        if ann.text in arm_labels:
            ann.y = ann.y - 0.018
            ann.font.size = 13
    fig.update_layout(
        title="Transitions by mod_type, all arms — Stage-2 rate annotated above each bar",
        barmode="stack", font=FONT, legend=LEGEND,
        margin=dict(l=60, r=150, t=64, b=70),
        width=1400, height=820, plot_bgcolor="white", paper_bgcolor="white")
    return fig


def f_actions(arm):
    df = arm["df"]
    conds = [m for m in MOD_ORDER if m in set(df["mod_type"].unique())]
    ys, lo_e, hi_e = [], [], []
    for c in conds:
        lo, mean, hi = bootstrap_ci(df[df["mod_type"] == c]["s2_action_count"].dropna())
        ys.append(mean); lo_e.append(mean - lo); hi_e.append(hi - mean)
    fig = go.Figure(go.Bar(x=[MOD_SHORT.get(c, c) for c in conds], y=ys, marker_color="#3a7bdd",
                           error_y=dict(type="data", symmetric=False, array=hi_e, arrayminus=lo_e)))
    fig.update_layout(title=f"Investigative actions used — {arm['label']}",
                      font=FONT, margin=MARGIN, width=1000, height=400,
                      yaxis_title="Mean s2_action_count", plot_bgcolor="white",
                      paper_bgcolor="white", xaxis_tickangle=30)
    return fig


def f_recall_spec(arm):
    df = arm["df"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=["Recall (defective called invalid)",
                                        "Specificity (correct called valid)"])
    for ci, mods in enumerate([INVALID_MOD_TYPES, VALID_MOD_TYPES], start=1):
        xl = ["Aggregate"] + [MOD_SHORT.get(m, m) for m in mods]
        for stage, colour in (("s1", "#7f8c8d"), ("s2", "#2980b9")):
            col = f"{stage}_valid_match"
            ys = [_rate(df, mods, col)[0]] + [_rate(df, [m], col)[0] for m in mods]
            ns = [_rate(df, mods, col)[1]] + [_rate(df, [m], col)[1] for m in mods]
            fig.add_trace(go.Bar(x=xl, y=ys, name=stage.upper(), marker_color=colour,
                                 text=[f"n={n}" for n in ns], textposition="outside",
                                 legendgroup=stage, showlegend=(ci == 1)), row=1, col=ci)
        fig.update_yaxes(range=[0, 115], title_text="Rate (%)", row=1, col=ci)
    fig.update_layout(title=f"Recall vs specificity — {arm['label']}", barmode="group",
                      font=FONT, legend=LEGEND, margin=MARGIN, width=1350, height=470,
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(tickangle=20)
    return fig


def f_score_dist(arm):
    df = arm["df"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=["Stage 1 (static read)", "Stage 2 (agentic)"])
    xs = list(range(SCORE_MIN, SCORE_MAX + 1))
    for ci, stage in enumerate(("s1", "s2"), start=1):
        for gt, name, colour in ((True, "truly valid", "#27ae60"), (False, "truly invalid", "#c0392b")):
            vals = [int(x) for x in df[df["gt_valid"] == gt][f"{stage}_validity_score"].dropna()]
            fig.add_trace(go.Bar(x=xs, y=[vals.count(v) for v in xs], name=name,
                                 marker_color=colour, legendgroup=name,
                                 showlegend=(ci == 1)), row=1, col=ci)
        fig.update_xaxes(title_text="validity_score", dtick=1, row=1, col=ci)
        fig.update_yaxes(title_text="Rows", row=1, col=ci)
    fig.update_layout(title=f"Validity-score distribution — {arm['label']}", barmode="group",
                      font=FONT, legend=LEGEND, margin=MARGIN, width=1250, height=400,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def p_paired_grid(arms, paper=False):
    """Clean vs its own defective twin, all four arms as a 2x2 grid.

    Two bars per x position: S1 hatched, S2 solid, each stacked into clean-higher
    / tied / inverted. Annotated above every bar is the share of that position's
    pairs that are INVERTED -- the clean version scoring at or below its own
    corrupted twin, which is the unambiguous ranking failure. Ties are counted in
    the stack but not in this percentage; a high tie count is a separate warning
    that the scale is being used too coarsely."""
    OUT = ["clean higher", "tied", "inverted"]
    COL = {"clean higher": "#27ae60", "tied": "#95a5a6", "inverted": "#c0392b"}
    xs = ["Aggregate"] + [MOD_SHORT.get(p, p) for p in PAIRING_ORDER]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[a["label"] for a in arms],
                        horizontal_spacing=0.07, vertical_spacing=0.09)
    for i, arm in enumerate(arms):
        r, c = i // 2 + 1, i % 2 + 1
        df = arm["df"]
        for stage, pat, shift in (("s1", "/", -22), ("s2", "", 22)):
            pairs = {}
            for _, row in df.iterrows():
                if not row.get("gt_sample"):
                    continue
                sc = row.get(f"{stage}_validity_score")
                key = (row["gt_sample"],
                       INVALID_TO_VALID_MOD_TYPE.get(row["mod_type"], row["mod_type"]))
                # None marks PRESENT BUT UNSCORED (a blank forced answer), which
                # is different from the twin being absent from the dataset.
                pairs.setdefault(key, {})[bool(row["gt_valid"])] = (
                    int(sc) if sc == sc and sc is not None else None)
            cnt = {k: 0 for k in OUT}
            per = {p: {k: 0 for k in OUT} for p in PAIRING_ORDER}
            for (_, pairing), v in pairs.items():
                if True not in v or False not in v:
                    continue
                if v[True] is None or v[False] is None:
                    # Present but unscored -> "tied". No score exists, so the
                    # model expressed no ranking either way, and _ratio already
                    # excludes ties from its denominator -- which is exactly
                    # where an uninformative pair belongs. Dropping it instead
                    # would make one arm's pair total differ from the rest for a
                    # reason unrelated to ranking; calling it "inverted" would
                    # assert a ranking that was never made.
                    k = "tied"
                else:
                    k = ("clean higher" if v[True] > v[False]
                         else ("tied" if v[True] == v[False] else "inverted"))
                cnt[k] += 1
                if pairing in per:
                    per[pairing][k] += 1
            for k in OUT:
                fig.add_trace(go.Bar(
                    name=k, x=xs,
                    y=[cnt[k]] + [per[p][k] for p in PAIRING_ORDER],
                    marker_color=COL[k], marker_pattern_shape=pat,
                    offsetgroup=stage, legendgroup=k, showlegend=(i == 0 and stage == "s1"),
                    hovertemplate=f"{stage.upper()}<br>{k}: %{{y}}<extra></extra>"),
                    row=r, col=c)
            # % inverted above each bar, one annotation per (position, stage)
            cols = [cnt] + [per[p] for p in PAIRING_ORDER]
            for j, d in enumerate(cols):
                tot = sum(d.values())
                if not tot:
                    continue
                pct = 100 * d["inverted"] / tot
                colour = "#1a7a3c" if pct < 5 else ("#b9770e" if pct < 12 else "#a93226")
                fig.add_annotation(x=xs[j], y=tot + 1, xshift=shift,
                                   text=f"{pct:.1f}%", showarrow=False,
                                   font=dict(size=11 if paper else 8.5, color=colour),
                                   yanchor="bottom", row=r, col=c)
        fig.update_yaxes(range=[0, 150], title_text="Matched pairs" if c == 1 else None,
                         row=r, col=c)
        fig.update_xaxes(tickangle=25, tickfont=dict(size=11.5 if paper else 8.5),
                         row=r, col=c)
    # Legend-only traces so the hatch convention survives PDF export, where the
    # figure title (which is where "S1 hatched vs S2" was stated) is stripped.
    # Zero-height bars on the first subplot: they contribute nothing to the data
    # but give the legend two swatches showing the pattern.
    for nm, pat in (("S1 (hatched)", "/"), ("S2 (solid)", "")):
        fig.add_trace(go.Bar(x=[xs[0]], y=[0], name=nm, marker_color="#b3b6b7",
                             marker_pattern_shape=pat, showlegend=True,
                             hoverinfo="skip"), row=1, col=1)
    arm_labels = {a["label"] for a in arms}
    for ann in fig.layout.annotations:
        if ann.text in arm_labels:
            ann.y = ann.y - 0.018
            ann.font.size = 16 if paper else 13
    fig.update_layout(
        # no title in the PDF -- the LaTeX float supplies the caption
        title=None if paper else
              "Paired ranking: clean vs its own defective twin — S1 (hatched) vs S2, % inverted annotated",
        barmode="stack", legend=LEGEND,
        font=(dict(family="Times New Roman, Times, serif", size=13.5, color="#222")
              if paper else FONT),
        margin=dict(l=60, r=150, t=20 if paper else 64, b=70),
        width=1400, height=760, plot_bgcolor="white", paper_bgcolor="white")
    return fig


CAT_ORDER = ["none", "some", "all"]
CAT_COLORS = {"none": "#c0392b", "some": "#f39c12", "all": "#27ae60"}


def f_judge(arm, by):
    jd = arm["judge"]
    if jd.empty:
        return None
    keys = INVALID_MOD_TYPES if by == "mod_type" else [p for p in PDE_ORDER if p in set(jd["pde_class"])]
    labels = [(MOD_SHORT if by == "mod_type" else PDE_LABEL).get(k, k) for k in keys]
    fig = go.Figure()
    for cat in CAT_ORDER:
        fig.add_trace(go.Bar(name=cat, x=labels,
                             y=[int(((jd[by] == k) & (jd["category"] == cat)).sum()) for k in keys],
                             marker_color=CAT_COLORS[cat]))
    fig.update_layout(title=f"Invalidity-reasoning correctness by {by} — {arm['label']}",
                      barmode="stack", font=FONT, legend=LEGEND, margin=MARGIN,
                      width=1000, height=400, yaxis_title="Count",
                      plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=25)
    return fig


# ── precision-rate table, one row per arm ───────────────────────────────────

PRECISION_FIELDS = [("contains_incorrect_code_claims", "bad_code_claims"),
                    ("contains_unverifiable_execution_claims", "unverif_exec_claims"),
                    ("contains_incorrect_claims", "legacy_merged")]


def t_precision(arms):
    cols = [(c, o) for c, o in PRECISION_FIELDS
            if any(c in a["judge"].columns and a["judge"][c].notna().any() for a in arms)]
    if not cols:
        return ""
    rows = []
    for a in arms:
        jd = a["judge"]
        if jd.empty:
            rows.append([a["label"], "0", *["n/a"] * len(cols)]); continue
        cells = []
        for c, _ in cols:
            if c in jd.columns and jd[c].notna().any():
                k = int(jd[c].sum()); cells.append(f"{k}/{len(jd)} ({100*k/len(jd):.1f}%)")
            else:
                cells.append("n/a")
        rows.append([a["label"], str(len(jd)), *cells])
    return _pre_table(["arm", "n judged"] + [o for _, o in cols], rows,
                      caption="Table — judge precision rates. bad_code_claims is the real "
                              "precision signal (asserts something false about the code). "
                              "unverif_exec_claims fires on runtime figures the judge cannot "
                              "check; it is NOT an error and runs near-constant, so read it as "
                              "a description of how these models argue.")


def t_s1_vs_s2(arms):
    """Paired reasoning-quality comparison, static read vs after execution.

    Only rendered for arms supplied with a Stage-1 judge pass. Paired on titles
    present in BOTH judge files -- the sets differ because a row wrong at Stage 1
    has no invalidity justification to judge, so rows that execution *fixed* are
    necessarily absent. The measurement is therefore "did execution improve the
    justification on rows where the verdict was already right".

    Every number is recomputed from the two judge files here; nothing is carried
    over from an earlier analysis."""
    out = ""
    for a in arms:
        j1, j2 = a.get("judge_s1"), a["judge"]
        if j1 is None or j1.empty or j2.empty:
            continue
        m1 = {r["title"]: r for _, r in j1.iterrows()}
        m2 = {r["title"]: r for _, r in j2.iterrows()}
        common = sorted(set(m1) & set(m2))
        if not common:
            continue
        n = len(common)

        def cat(m, c):
            return sum(1 for t in common if m[t].get("category") == c)

        def flag(m, f):
            return sum(1 for t in common if m[t].get(f))

        spec = [("category = all", lambda m: cat(m, "all")),
                ("category = some", lambda m: cat(m, "some")),
                ("category = none", lambda m: cat(m, "none")),
                ("false code claims", lambda m: flag(m, "contains_incorrect_code_claims")),
                ("runtime claims", lambda m: flag(m, "contains_unverifiable_execution_claims"))]
        rows = []
        for lab, fn in spec:
            k1, k2 = fn(m1), fn(m2)
            p1, p2 = 100 * k1 / n, 100 * k2 / n
            rows.append([lab, f"{k1} ({p1:.1f}%)", f"{k2} ({p2:.1f}%)", f"{p2 - p1:+.1f}pp"])
        # per-row movement, so the reader can tell a real shift from churn
        order = {"none": 0, "some": 1, "all": 2}
        same = sum(1 for t in common
                   if order.get(m1[t].get("category")) == order.get(m2[t].get("category")))
        better = sum(1 for t in common
                     if order.get(m1[t].get("category"), 0) < order.get(m2[t].get("category"), 0))
        worse = n - same - better
        out += _pre_table(
            ["metric", "S1 (static read)", "S2 (agentic)", "change"], rows,
            caption=f"Table — reasoning quality S1 vs S2, {a['label']}, paired on n={n} rows "
                    f"judged at both stages. Per-row category movement: {same} unchanged, "
                    f"{better} improved, {worse} worse. Rows that execution CORRECTED are "
                    f"absent by construction (their S1 justification argues for validity, so "
                    f"there is nothing comparable to judge), so this measures whether execution "
                    f"improves the justification where the verdict was already right. "
                    f"'runtime claims' means the justification cites runtime detail: at S2 that "
                    f"may be an observation, at S1 it can only be a prediction.")
    return out


# ── LaTeX table export (shown under the plot, copy-pasteable) ───────────────

def _latex_block(colspec, header, rows, caption, label):
    """booktabs tabular wrapped in a table float, rendered in a <pre> so it can
    be copied straight into the manuscript."""
    esc = lambda t: (str(t).replace("_", r"\_").replace("%", r"\%")
                     .replace("&", r"\&").replace("→", r"$\rightarrow$"))
    out = ["\\begin{table}[t]", "\\centering", "\\small",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join(esc(h) for h in header) + r" \\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(esc(c) for c in r) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    body = "\n".join(out)
    return ('<div style="font-weight:600;margin:18px 0 4px;">LaTeX (copy)</div>'
            f'<pre style="font-family:ui-monospace,monospace;font-size:12px;'
            f'line-height:1.45;background:#0f1420;padding:10px;'
            f'border:1px solid #1e2130;overflow-x:auto;">{body}</pre>')


def latex_by_modtype(arms, metric, mods, caption, label, pct=True):
    header = ["Arm", "Stage"] + [MOD_SHORT.get(m, m) for m in mods]
    rows = []
    for a in arms:
        for stage in ("s1", "s2"):
            cells = []
            for m in mods:
                v = a["df"][a["df"]["mod_type"] == m][f"{stage}_{metric}"].dropna()
                cells.append(f"{100*v.mean():.1f}" if len(v) else "--")
            rows.append([a["label"], stage.upper()] + cells)
    return _latex_block("ll" + "r" * len(mods), header, rows, caption, label)


def latex_auc_by_pairing(arms, caption, label):
    header = ["Arm", "Stage"] + [MOD_SHORT.get(p, p) for p in PAIRING_ORDER]
    rows = []
    for a in arms:
        df = a["df"]
        for stage in ("s1", "s2"):
            col = f"{stage}_validity_score"
            cells = []
            for vm in PAIRING_ORDER:
                im = next((k for k, v in INVALID_TO_VALID_MOD_TYPE.items() if v == vm), None)
                pos = [int(x) for x in df[df["mod_type"] == vm][col].dropna()]
                neg = [int(x) for x in df[df["mod_type"] == im][col].dropna()] if im else []
                v = score_auc(pos, neg)
                cells.append(f"{v:.3f}" if v == v else "--")
            rows.append([a["label"], stage.upper()] + cells)
    return _latex_block("ll" + "r" * len(PAIRING_ORDER), header, rows, caption, label)


# ── HTML ─────────────────────────────────────────────────────────────────────

def _slug(t):
    """Filesystem-safe stub from a panel title or arm label."""
    out = "".join(ch if ch.isalnum() else "_" for ch in t)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()[:44]


def _write_pdf(fig, path, label):
    """Write one paper PDF with the top-level title stripped."""
    pdf = go.Figure(fig)
    pdf.update_layout(title_text=None)
    m = pdf.layout.margin
    pdf.update_layout(margin=dict(l=m.l, r=m.r, b=m.b, t=max(m.t or 50, 50)))
    try:
        pdf.write_image(str(path))
        print(f"[combined] {path.name}  <- {label}")
        return True
    except Exception as exc:
        print(f"[combined] PDF failed for {path.name}: {str(exc)[:70]}")
        return False


def export_static(sections, out_dir, arms, extra_pdfs=()):
    """Write one PNG per figure plus one .txt per table.

    The HTML is the browsing artifact; these are what goes into the paper. Named
    P<nn>_<panel-slug>[_<arm>].png so ordering matches the report's panel order
    and per-arm figures are unambiguous. PNG export needs kaleido -- if it is
    missing this reports the miss rather than failing the whole run, since the
    HTML is already written by then."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_png = n_txt = n_pdf = n_fail = 0
    for fig, name, label in extra_pdfs:
        if _write_pdf(fig, fig_dir / f"{name}.pdf", label):
            n_pdf += 1
        else:
            n_fail += 1
    for idx, (title, _desc, figs, extra) in enumerate(sections, start=1):
        stub = f"P{idx:02d}_{_slug(title)}"
        real = [f for f in figs if f is not None]
        if title in PDF_FROM_PANEL and len(real) == 1:
            if _write_pdf(real[0], fig_dir / f"{PDF_FROM_PANEL[title]}.pdf", title):
                n_pdf += 1
            else:
                n_fail += 1
        for k, f in enumerate(real):
            # a panel with 4 figures is one-per-arm, in the arms' fixed order
            suffix = f"_{_slug(arms[k]['label'])}" if len(real) == len(arms) > 1 else ""
            path = fig_dir / f"{stub}{suffix}.png"
            try:
                f.write_image(str(path), scale=2)
                n_png += 1
            except Exception as exc:
                print(f"[combined] PNG failed for {path.name}: {str(exc)[:70]}")
                n_fail += 1
        if extra:
            # Break at block boundaries BEFORE stripping tags, otherwise the
            # caption <div> runs straight into the first row of the <pre> it
            # labels and the file is unreadable.
            txt = re.sub(r"</(div|pre)>", "\n\n", extra)
            txt = re.sub(r"<pre[^>]*>", "\n", txt)
            txt = re.sub(r"<[^>]+>", "", txt)
            txt = (txt.replace("&amp;", "&").replace("&lt;", "<")
                      .replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"'))
            txt = re.sub(r"[ \t]+\n", "\n", txt)
            txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
            if txt:
                (fig_dir / f"{stub}_tables.txt").write_text(txt + "\n")
                n_txt += 1
    print(f"[combined] exported {n_pdf} PDF + {n_png} PNG + {n_txt} table files -> {fig_dir}"
          + (f"  ({n_fail} PNG failures)" if n_fail else ""))


def write_html(sections, out_dir, arms):
    out_dir.mkdir(parents=True, exist_ok=True)
    nav = content = ""
    first = True
    for idx, (title, desc, figs, extra) in enumerate(sections):
        active = "active" if idx == 0 else ""
        nav += f'<button class="nav-btn {active}" onclick="show({idx})">{title}</button>\n'
        body = ""
        for f in figs:
            if f is None:
                continue
            body += f.to_html(full_html=False, include_plotlyjs=first, default_width="100%")
            first = False
        content += (f'<div class="section {active}" id="sec{idx}">'
                    f'<div class="chart-header"><h2>{title}</h2>'
                    f'<p class="question">{desc}</p></div>{body}{extra}</div>')
    arm_list = "<br>".join(a["label"] for a in arms)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Validity benchmark — all arms</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d0f18;
     color:#e0e0e0;display:flex;height:100vh;overflow:hidden}}
#sidebar{{width:275px;min-width:275px;background:#12141e;border-right:1px solid #1e2130;
         overflow-y:auto}}
#sidebar h1{{font-size:.85rem;color:#fff;padding:14px 12px 10px;border-bottom:1px solid #1e2130;
            line-height:1.4}}
.meta{{font-size:.7rem;color:#555;padding:8px 12px 12px;border-bottom:1px solid #1e2130;
      line-height:1.7}}
.meta strong{{color:#888}}
.nav-btn{{display:block;width:100%;background:none;border:none;border-left:3px solid transparent;
         color:#777;padding:7px 12px 7px 9px;cursor:pointer;font-size:.76rem;text-align:left;
         line-height:1.4}}
.nav-btn:hover{{background:#191c2a;color:#ccc}}
.nav-btn.active{{background:#171d30;color:#7eb8ff;border-left-color:#3a7bdd}}
#main{{flex:1;overflow-y:auto;padding:22px 26px}}
.section{{display:none}} .section.active{{display:block}}
.chart-header h2{{margin:5px 0 10px;font-size:1rem;color:#ddd;font-weight:500}}
.question{{font-size:.8rem;color:#5a80b0;max-width:860px;border-left:3px solid #1e3560;
          padding-left:10px;line-height:1.6;margin-bottom:16px}}
</style></head><body>
<div id="sidebar"><h1>Validity benchmark — all arms</h1>
<div class="meta"><strong>Arms:</strong><br>{arm_list}<br><br>
<strong>Dataset:</strong> 256 scripts, 8 mod_types (jul28)<br>
<strong>Stages:</strong> S1 static read, S2 agentic execution</div>
{nav}</div><div id="main">{content}</div>
<script>function show(i){{
document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
document.getElementById('sec'+i).classList.add('active');
document.querySelectorAll('.nav-btn')[i].classList.add('active');
window.dispatchEvent(new Event('resize'));}}</script></body></html>"""
    p = out_dir / "belief_revision_all_arms.html"
    p.write_text(html)
    print(f"[combined] wrote {p}  ({p.stat().st_size/1e6:.2f} MB)")


def main():
    args = parse_args()
    arms = load_arms(args.arm)
    # Opened before any panel is built: every interval routes through the cache.
    ci_cache_open(Path(args.out_dir) / ".ci_cache.json",
                  enabled=not args.no_ci_cache, recompute=args.recompute_ci)
    S = []

    S.append(("Summary — accuracy and AUC across arms",
              "Validity accuracy (the model's binary call) and score AUC (rank-based "
              "discrimination), static read vs agentic execution, for all four arms. The "
              "whole experiment in one figure: execution's yield is inversely proportional "
              "to how much the static read already solved.",
              [p_summary(arms)], ""))

    S.append(("Validity accuracy by mod_type — all 8 lines",
              "One line per arm per stage. Colour identifies the arm, light+dashed is the "
              "static read and dark+solid is after execution. Error bars are bootstrap 95% CI "
              "on n=32 per cell, so a single row is worth ~3.1pp.",
              [p_lines(arms, "valid_match", "Validity accuracy by mod_type — S1 vs S2, all arms",
                       "Accuracy (%)")],
              latex_by_modtype(arms, "valid_match", MOD_ORDER,
                               "Validity accuracy (\\%) by lexical condition, static read (S1) "
                               "versus agentic execution (S2). $n=32$ per cell.",
                               "tab:agent-validity-modtype")))

    S.append(("Score ranking AUC by mod_type",
              "The score analogue of the panel above, and the sharper measure: per lexical "
              "condition, how reliably does the model rank a clean script above its OWN "
              "defective twin? Four x-points rather than eight, because AUC merges each valid "
              "mod_type with its invalid counterpart. Colour identifies the arm, light+dashed "
              "is the static read and dark+solid is after execution. 32 rows per side per "
              "point, so read directions rather than small magnitudes.",
              [p_auc_lines(arms)],
              latex_auc_by_pairing(arms,
                                   "Score ranking AUC per valid/invalid pairing. Each column "
                                   "merges a valid mod\\_type with its own invalid counterpart; "
                                   "$n=32$ per side.",
                                   "tab:agent-auc-pairing")))

    S.append(("PDE identification by mod_type (control)",
              "The control task. Should stay high and flat: identifying which equation is "
              "being solved is largely insensitive to execution, unlike judging whether it "
              "is solved correctly.",
              [p_lines(arms, "pde_match", "PDE identification by mod_type — S1 vs S2, all arms",
                       "Accuracy (%)")],
              latex_by_modtype(arms, "pde_match", MOD_ORDER,
                               "PDE identification accuracy (\\%) by lexical condition. Control "
                               "task: near-flat across stages, unlike validity. $n=32$ per cell.",
                               "tab:agent-pde-modtype")))

    fig, tbl = p_agent_choices(arms)
    S.append(("Agent choices",
              "What each arm did with its investigative budget, and whether it knew when to "
              "stop. Bars are ABSOLUTE actions per episode split by tool, with composition "
              "annotated -- percentage alone would mislead, since composition tracks the "
              "thinking level rather than model strength.",
              [fig], tbl))

    S.append(("Reasoning quality across arms",
              "Judge verdicts for all four arms, Stage 2 only. Per arm: a stacked bar giving "
              "the none/some/all composition (recall -- did the justification name a real "
              "defect), then false code claims (precision -- did it also assert something "
              "untrue about the code), then runtime claims. The last of those sits at 80-90% "
              "in every arm, so it does not discriminate between models; it is shown greyed "
              "because its near-constancy is the point, not its height. Scope: rows the model "
              "correctly called invalid, so a row it got wrong contributes nothing.",
              [p_reasoning_quality(arms, show_ci=True)], ""))

    S.append(("Transitions by mod_type",
              "Stacked transition counts per lexical condition, all four arms as a 2x2 grid. "
              "R->R correct at both stages, W->R corrected by execution, R->W a regression, "
              "W->W wrong throughout. The percentage above each bar is that condition's "
              "Stage-2 rate: on the FOUR LEFT (valid) conditions it is specificity, on the FOUR "
              "RIGHT (invalid) conditions it is recall. It is the same computation either way -- "
              "on valid code being right means calling it valid, on invalid code it means calling "
              "it invalid. Read the two halves against each other: recall is 91-97% in all "
              "sixteen cells, every arm and every lexical condition, while specificity spans "
              "25-94% and varies along both axes. The lexical sensitivity is confined to the "
              "weakest arm and disappears with capability -- 25-59% across conditions at "
              "thinking budget 0, tightening to 56-66% at 8192, and flat at 94% for "
              "3.7-flash-high.",
              [p_transitions_grid(arms)], ""))

    S.append(("Transitions by mod_type (reorganized)",
              "The same transition data as the panel above, reorganized: rows are the four "
              "lexical conditions, the valid/invalid half is carried by the rotated block "
              "label, and columns are the four arms grouped by model family. Bars are "
              "normalized to share of episodes (n=32 per bar, so one episode is 3.1%), which "
              "makes the green+blue length equal to that cell's Stage-2 accuracy -- printed to "
              "the right of each bar. In-bar percentages are suppressed below "
              "{}% of the bar, where white text would overflow its segment; a blank "
              "segment is therefore small, not absent. Colour meaning is unchanged from the "
              "panel above.".format(int(REORG_MIN_LABEL_PCT)),
              [p_transitions_reorganized(arms)], ""))

    per_arm = [("Investigative actions by mod_type", f_actions,
                "Mean investigative actions per episode (edit_source + run_diagnostic; submit "
                "never counts against the budget of 6)."),
               ("Recall vs specificity", f_recall_spec,
                "Recall over the four defective mod_types, specificity over the four valid "
                "ones, both stages. Recall is near-ceiling in every arm; specificity is where "
                "the benchmark's difficulty lives."),
               ("Validity-score distribution", f_score_dist,
                "Score histograms split by ground truth, both stages. Read before the AUC and "
                "paired-ranking panels: a mean over a two-humped ordinal is a weak summary."),
               ]
    for title, fn, desc in per_arm:
        S.append((title, desc + " Four figures, one per arm, in fixed order.",
                  [fn(a) for a in arms], ""))

    S.append(("Paired ranking: clean vs its own defective twin",
              "Per matched pair, did the clean version outscore its OWN corrupted twin? Pairs "
              "keyed on (gt_sample, mod_type pairing), not gt_sample alone. All four arms as a "
              "2x2 grid; within each, S1 is hatched and S2 solid. The percentage above every bar "
              "is the share of that position's pairs that are INVERTED -- clean scoring at or "
              "below its own defective twin. Ties appear in the stack but not in that "
              "percentage, deliberately: a tall grey band is a separate problem, meaning the "
              "model gave the clean and corrupted versions the same score and so told you "
              "nothing. The S1->S2 movement is the intended read, and it differs by arm -- for "
              "2.5 at budget 0 execution mainly converts ties, while at 8192 it is the only arm "
              "where execution actually reduces inversions.",
              [p_paired_grid(arms)], ""))

    for by, nice in (("mod_type", "mod_type"), ("pde_class", "PDE class")):
        figs = [f_judge(a, by) for a in arms]
        if any(f is not None for f in figs):
            S.append((f"Invalidity-reasoning correctness by {nice}",
                      "LLM-judge classification of the model's own justification, restricted to "
                      "rows it correctly called invalid. 'none' means no genuine defect was "
                      "identified despite the right answer. Four figures, one per arm.",
                      figs, ""))

    S.append(("Judge precision rates",
              "Precision counterpart to the category panels, which are recall-only: a "
              "justification can name every real defect and still assert something false. "
              "Where a Stage-1 judge pass was supplied, the paired S1-vs-S2 comparison "
              "below shows whether execution improves the justification at all.",
              [], t_precision(arms) + t_s1_vs_s2(arms)))

    # Reorder to the report's intended reading order; anything unlisted keeps its
    # build order and follows.
    rank = {t: i for i, t in enumerate(PANEL_ORDER)}
    S.sort(key=lambda sec: rank.get(sec[0], len(PANEL_ORDER) + 1))
    print("[combined] panel order:")
    for i, sec in enumerate(S, 1):
        tag = f"  -> {PDF_FROM_PANEL[sec[0]]}.pdf" if sec[0] in PDF_FROM_PANEL else ""
        print(f"    {i:2d}. {sec[0]}{tag}")

    write_html(S, Path(args.out_dir), arms)
    if not args.no_png:
        # agent_fig1 is export-only: merged Summary + Agent choices, recoloured.
        export_static(S, Path(args.out_dir), arms,
                      extra_pdfs=[
                          (p_paper_fig1(arms), "agent_fig1",
                           "merged summary + agent choices (export-only)"),
                          # Same data as agent_fig2's 2x2 grid, reorganized layout.
                          # Kept as a separate name so the existing fig2 is not
                          # overwritten until the layout is chosen.
                          (p_transitions_reorganized(arms, show_s2=False, compact=True,
                                                     title=False),
                           "agent_fig2b", "transitions reorganized (export-only)"),
                          (p_paper_fig1b(arms), "agent_fig1b",
                           "summary + agent choices, with CIs (export-only)"),
                          # Rebuilt with show_ci=False so the error bars added to
                          # the HTML panel do not leak into the PDF.
                          (p_reasoning_quality(arms, show_ci=True, paper=True),
                           "agent_fig3", "reasoning quality, CIs + Times (export-only)"),
                          (p_paired_grid(arms, paper=True),
                           "agent_suppfig1", "paired ranking, Times (export-only)"),
                      ])
    ci_cache_close()
    print(f"[combined] {len(S)} panels")


if __name__ == "__main__":
    main()
