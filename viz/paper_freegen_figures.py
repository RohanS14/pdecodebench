"""
paper_freegen_figures.py — free-generation paper figures, as light PNGs.

Renamed from paper_hedge_figures.py: it outgrew the name once the confident/hedged
split was dropped from the breakdowns and the file picked up the resampling and
PDE-identification figures.

Separate from paper_figures.py because that script loads the MC logprob data and
resolves its inputs relative to viz/; these two figures need only the free-generation
aggregate. Light background (plotly_white) and scale=2 PNG, so the output goes into a
paper without a screenshot.

Two figures, and the second is the one that carries the argument:

    paper_hedge_pooled.png     eight perturbations, every model pooled
    paper_hedge_by_model.png   the same, one small multiple per model
    paper_overclaim.png        stated confidence against resampling stability
    paper_pde_naming.png       does the model name the PDE, by class and by
                               annotation-only perturbation

The pooled figure alone is misleading on this roster. The newer checkpoints answer
the invalid half almost perfectly and the valid half barely better than chance; the
older ones do the reverse. Averaging two opposite biases produces a bar that reads as
moderate competence at both, which is the one thing these figures must not say.

Error bars
----------
EVERY figure in this file carries 95% intervals, from cluster_ci(): a percentile
bootstrap over the 32 base SYSTEMS, not over rows. Eight conditions, eight models
and k=3 draws all share each base solver, so rows within a system are not
independent and a row bootstrap would narrow every interval here by roughly the
square root of that clustering. The estimand is always a proportion of 0/1 draws,
so a percentile interval cannot leave [0, 100] and none of these is drawn
symmetric -- both of which the NeurIPS checklist asks about explicitly.

On the STACKED figures (fig_pooled, fig_by_model) the interval sits on the
cumulative boundary between segments rather than on each segment's own share; see
the comment in fig_pooled for why the share's interval belongs at a different
height than the segment is drawn at.

Usage:
    python viz/paper_freegen_figures.py
    PDE_FREEGEN_CSV=results/freegen_static_judgments.csv OUT_DIR=figures python viz/paper_freegen_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from freegen_static_judgments.parse_score import classify_valid_confidence_2x2   # noqa: E402

CSV = os.environ.get("PDE_FREEGEN_CSV", "results/freegen_static_judgments.csv")
# The paper refers to the by-model validity panel as validity_static. It was kept in
# sync by copying the file by hand, which lasted exactly as long as someone remembered
# to: regenerating paper_hedge_by_model.png left validity_static.png behind at the
# previous roster, and nothing about the stale file said so. Written from the same
# Figure object here, so the two cannot disagree.
ALIASES = {"paper_hedge_by_model": ["validity_static"]}
OUT_DIR = os.environ.get("OUT_DIR", "figures")
SCALE = int(os.environ.get("SCALE", "2"))

# Same eight conditions, same order, same truth marking as the dual report. The
# label carries the answer that is CORRECT for that condition, because the same
# lean is right on the valid rows and wrong on the invalid ones, and a chart with
# no truth marker invites reading "more green" as "better".
COND_ORDER = [
    ("Comm_Valid",              "Clean+Comment",          True),
    ("NoComm_Valid",            "Clean, No Comment",      True),
    ("CorrComm",                "Corrupt Comment",        True),
    ("NoComm_CorrVar",          "Obfuscated Vars",        True),
    ("Comm_InValid",            "Invalid+Comment",        False),
    ("NoComm_InValid",          "Invalid, No Comment",    False),
    ("CorrComm_Invalid",        "CorrComment+Invalid",    False),
    ("NoComm_CorrVar_InValid",  "Obfuscated+Invalid",     False),
]

# "no lean" is a real bucket and sits BETWEEN the two directions. An answer with no
# direction at all ("depends on the time step and grid resolution") is an abstention,
# not a hedged yes or a hedged no. Dropping it made every bar fall short of 100% by a
# different amount with nothing saying why.
# The breakdown figures show DIRECTION only. The confident/hedged split used to be
# four legend entries here, and it overstated what the verdict field can support: the
# prompt asks for a terse fill-in-the-blank answer, so 85% of answers are a bare
# "yes" or "no" and the hedged bands were slivers that invited reading a real
# confidence signal off a format artefact. Confidence is measured properly in
# fig_overclaim() below, by resampling, and that is where the split belongs.
ORDER = ["predicts invalid", "predicts valid"]
COLOR = {"predicts invalid": "#c0392b", "predicts valid": "#27ae60"}
# Verdict bucket -> the direction it states. An answer with NO direction ("depends on
# the time step and grid resolution") is dropped rather than drawn, so the bars are a
# share of answers that stated a direction and still sum to 100.
#
# Worth knowing what that excludes: 92 of 5,426 draws, and 88 of those 92 are
# Nemotron -- 11.5% of its answers, against 0-2 draws for every other model. So this
# is not an even 1.7% trim off every bar; Nemotron's bars rest on a base 11.5%
# smaller than the rest of the roster's, and its declining to answer is a real
# behaviour that these figures no longer show.
DIRECTION = {"Confident No": "predicts invalid", "Hedged No": "predicts invalid",
             "Confident Yes": "predicts valid", "Hedged Yes": "predicts valid"}
INVALID_BAND = "rgba(125,60,152,0.10)"
DIVIDER = "rgba(125,60,152,0.75)"


def load():
    df = pd.read_csv(CSV)
    if "no_verdict" in df.columns:
        keep = ~df["no_verdict"].astype(str).str.lower().isin(("true", "1"))
        print(f"[paper] dropping {int((~keep).sum())} no-verdict row(s)")
        df = df[keep]
    df["bucket"] = (df["parsed_valid"].map(classify_valid_confidence_2x2)
                    .fillna("").replace("", "no lean"))
    # Kept alongside `bucket`, not instead of it: fig_overclaim() needs the
    # confident/hedged split that `direction` throws away.
    df["direction"] = df["bucket"].map(DIRECTION)
    n_nolean = int(df["direction"].isna().sum())
    if n_nolean:
        worst = (df[df["direction"].isna()]["model"].value_counts().head(1))
        print(f"[paper] dropping {n_nolean} draw(s) with no stated direction "
              f"({100 * n_nolean / len(df):.2f}%); {worst.iloc[0]} of them are "
              f"{worst.index[0].split('/')[-1]}")
        df = df[df["direction"].notna()]
    return df


def cluster_ci(sub, col, n_boot=2000, seed=20260820):
    """Mean and 95% interval, resampling the 32 base SYSTEMS rather than the rows.

    Eight conditions and eight models share each base solver, so rows within a
    system are not independent; a row bootstrap would narrow every interval here by
    roughly the square root of that clustering.
    """
    vals = sub[col].dropna()
    if vals.empty:
        return float("nan"), float("nan"), float("nan")
    g = sub[["gt_sample", col]].dropna(subset=[col]).groupby("gt_sample")[col]
    sums = g.sum().to_numpy(dtype=float)
    counts = g.size().to_numpy(dtype=float)
    keep = counts > 0
    sums, counts = sums[keep], counts[keep]
    if len(counts) < 2:
        return float(vals.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    # The mean of the concatenated resampled clusters is sum-of-sums over
    # sum-of-counts. Written that way rather than as 2000 np.concatenate calls
    # because this is now called once per figure CELL (~170 times per run, against
    # 16 when only the two pooled figures used it) and the loop form took minutes.
    # Identical arithmetic and identical rng draws, so the intervals do not move.
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ci_err(sub, col, scale=100.0):
    """cluster_ci as (plus, minus) error-bar lengths, ready for plotly `error_*`.

    Returns 0.0 rather than nan for a cell too small to bootstrap: plotly draws a
    nan-length bar as a full-height spike, which reads as a huge interval when the
    truth is that there is no interval to report.
    """
    mn, lo, hi = cluster_ci(sub, col)
    if not (mn == mn and lo == lo and hi == hi):
        return 0.0, 0.0
    return (hi - mn) * scale, (mn - lo) * scale


ERR_STYLE = dict(color="rgba(0,0,0,0.55)", thickness=1.2, width=4)


def paired_delta_ci(a_sub, b_sub, col, n_boot=2000, seed=20260820, scale=100.0):
    """95% interval on (mean b - mean a) when a and b are the SAME 32 systems.

    The two arms of every contrast in this file are the same base solvers seen under
    different perturbations, so their errors are strongly correlated and the marginal
    intervals are much wider than the interval on the difference. Reading "the two
    error bars overlap, so nothing happened" off a pair of marginal intervals is the
    standard way to miss a real within-subject effect, and on the annotation contrast
    it would reverse the paper's conclusion: the marginal bars overlap heavily for
    every model while the paired delta clears zero for half the roster.

    Systems are resampled ONCE and both arms are recomputed on that same resample --
    resampling the arms independently would throw the pairing away and reproduce the
    marginal width.
    """
    def agg(sub):
        g = sub[["gt_sample", col]].dropna(subset=[col]).groupby("gt_sample")[col]
        return g.sum(), g.size()

    a_s, a_n = agg(a_sub)
    b_s, b_n = agg(b_sub)
    systems = a_s.index.intersection(b_s.index)
    if len(systems) < 2:
        return float("nan"), float("nan"), float("nan")
    a_s, a_n = a_s.loc[systems].to_numpy(float), a_n.loc[systems].to_numpy(float)
    b_s, b_n = b_s.loc[systems].to_numpy(float), b_n.loc[systems].to_numpy(float)
    obs = b_s.sum() / b_n.sum() - a_s.sum() / a_n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(systems), size=(n_boot, len(systems)))
    boots = (b_s[idx].sum(axis=1) / b_n[idx].sum(axis=1)
             - a_s[idx].sum(axis=1) / a_n[idx].sum(axis=1))
    return (obs * scale, float(np.percentile(boots, 2.5)) * scale,
            float(np.percentile(boots, 97.5)) * scale)


def present_conds(df):
    return [(k, lab, gt) for k, lab, gt in COND_ORDER if (df["mod_type"] == k).any()]


def labels_for(present):
    return [(lab + " ✓") if gt else ("⚠ " + lab) for _, lab, gt in present]


def fig_pooled(df):
    present = present_conds(df)
    labels = labels_for(present)
    split = sum(1 for _, _, gt in present if gt)

    # Intervals go on the CUMULATIVE BOUNDARIES, not on each segment's own share.
    # A stacked segment is DRAWN at the running total but its share is a different
    # quantity, so attaching the share's interval to that position puts the topmost
    # bar's error bar above 100% on a chart whose bars all sum to exactly 100. The
    # boundary is a real estimate -- "the share of answers at or below this bucket"
    # -- and the interval belongs to it. The final boundary is 100% by construction,
    # so it carries none.
    fig = go.Figure()
    for i, bucket in enumerate(ORDER):
        shares, hi_err, lo_err = [], [], []
        upto = ORDER[:i + 1]
        for code, _, _ in present:
            sub = df[df["mod_type"].eq(code)].copy()
            shares.append(100 * (sub["direction"] == bucket).mean())
            if i == len(ORDER) - 1:
                hi_err.append(0.0); lo_err.append(0.0)
                continue
            sub["_cum"] = sub["direction"].isin(upto).astype(float)
            mn, lo, hi = cluster_ci(sub, "_cum")
            hi_err.append((hi - mn) * 100 if hi == hi else 0.0)
            lo_err.append((mn - lo) * 100 if lo == lo else 0.0)
        fig.add_bar(name=bucket, x=labels, y=shares, marker_color=COLOR[bucket],
                    marker_line=dict(color="white", width=0.6),
                    error_y=dict(type="data", symmetric=False, array=hi_err,
                                 arrayminus=lo_err, color="rgba(0,0,0,0.55)",
                                 thickness=1.2, width=5))

    b = split - 0.5
    fig.add_vrect(x0=b, x1=len(present) - 0.5, fillcolor=INVALID_BAND, line_width=0, layer="below")
    fig.add_vline(x=b, line=dict(color=DIVIDER, width=1.5, dash="dash"))
    fig.add_annotation(x=0, y=103.5, text="VALID CODE — correct answer is Yes",
                       showarrow=False, xanchor="left", font=dict(size=11, color="#1a8040"))
    fig.add_annotation(x=len(present) - 1, y=103.5, text="INVALID CODE — correct answer is No",
                       showarrow=False, xanchor="right", font=dict(size=11, color="#7d3c98"))
    fig.update_layout(
        template="plotly_white", barmode="stack",
        yaxis=dict(title="% of answers", range=[0, 105], dtick=20),
        xaxis=dict(tickangle=-25),
        width=1100, height=520, margin=dict(l=70, r=40, t=42, b=150),
        legend=dict(orientation="h", y=-0.38, x=0.5, xanchor="center", title=""),
        font=dict(size=13))
    return fig


def fig_by_model(df):
    present = present_conds(df)
    labels = labels_for(present)
    split = sum(1 for _, _, gt in present if gt)
    models = sorted(df["model"].unique(),
                    key=lambda m: df[df["model"].eq(m)]["bucket"]
                    .isin(["Hedged Yes", "Confident Yes"]).mean())

    ncols = 4
    nrows = (len(models) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True,
                        subplot_titles=[m.split("/")[-1] for m in models],
                        vertical_spacing=0.09, horizontal_spacing=0.045)

    for i, model in enumerate(models):
        r, c = i // ncols + 1, i % ncols + 1
        g = df[df["model"].eq(model)]
        # Intervals on the CUMULATIVE BOUNDARY, exactly as in fig_pooled: a segment
        # is drawn at the running total, so its own share's interval would sit at
        # the wrong height and push the top bar past 100 on bars that sum to 100.
        # The last boundary is 100% by construction and carries none.
        for j, bucket in enumerate(ORDER):
            ys, hi_err, lo_err = [], [], []
            upto = ORDER[:j + 1]
            for code, _, _ in present:
                sub = g[g["mod_type"].eq(code)].copy()
                ys.append(100 * (sub["direction"] == bucket).mean() if len(sub) else 0.0)
                if j == len(ORDER) - 1 or not len(sub):
                    hi_err.append(0.0); lo_err.append(0.0)
                    continue
                sub["_cum"] = sub["direction"].isin(upto).astype(float)
                p, m = ci_err(sub, "_cum")
                hi_err.append(p); lo_err.append(m)
            fig.add_bar(x=labels, y=ys, name=bucket, marker_color=COLOR[bucket],
                        marker_line=dict(color="white", width=0.4),
                        error_y=dict(type="data", symmetric=False, array=hi_err,
                                     arrayminus=lo_err, **ERR_STYLE),
                        legendgroup=bucket, showlegend=(i == 0), row=r, col=c)
        fig.add_vrect(x0=split - 0.5, x1=len(present) - 0.5, fillcolor=INVALID_BAND,
                      line_width=0, layer="below", row=r, col=c)
        fig.add_vline(x=split - 0.5, line=dict(color=DIVIDER, width=1, dash="dash"),
                      row=r, col=c)

    fig.update_layout(template="plotly_white", barmode="stack",
                      width=1500, height=380 * nrows + 150,
                      margin=dict(l=75, r=40, t=42, b=215),
                      # Legend BELOW the panels, just clear of the vertical tick
                      # labels and no further. At y=-0.12 it sat in the same strip as
                      # them and clipped the middle two columns ("Corrupt Comment"
                      # rendered as "upt Comment"); at -0.30 it cleared them and left
                      # a band of dead space. The labels need ~150px, so the legend
                      # sits just past that and the bottom margin holds exactly both.
                      legend=dict(orientation="h", y=-0.235, x=0.5, xanchor="center",
                                  yanchor="top", title=""),
                      font=dict(size=12))
    fig.update_yaxes(range=[0, 100], title_text="")
    # Axis title on the LEFT COLUMN only -- shared_yaxes hides the ticks on the
    # inner panels, so repeating the title there would label an axis with no scale.
    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text="% of answers", row=r, col=1)
    # Tick labels on the BOTTOM row of each column only. Eight labels at -55 degrees
    # are taller than the gap between rows, so each one collided with the subplot
    # title beneath it. The columns share one category axis, so the surviving label
    # names its whole column.
    # Vertical, not angled. At -55 degrees a 19-character label extends well to
    # the LEFT of its tick, past the subplot's own domain, and the neighbouring
    # panel clips it -- "Obfuscated Variables" rendered as "scated Variables".
    # Vertical labels are centred under their tick and cannot overflow sideways.
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showticklabels=False)
    for col in range(ncols):
        rows = [i // ncols + 1 for i in range(len(models)) if i % ncols == col]
        if rows:
            fig.update_xaxes(showticklabels=True, row=max(rows), col=col + 1)
    for ann in fig.layout.annotations[:len(models)]:
        ann.font = dict(size=13)
    return fig


# ── Stated confidence vs behavioural confidence ───────────────────────────────
# The breakdown figures above measure what the model SAYS. They cannot measure how
# sure it is, because the prompt asks for a terse fill-in-the-blank verdict ("Be
# concise. Output only: ... valid: ____"), so a model that deliberated for twenty
# thousand tokens and one that answered reflexively both write "valid: no".
#
# The k=3 draws give the missing half for free. Three samples of one prompt at
# temperature 0.6: if they disagree, the model is not sure, whatever any single draw
# claimed. That is BEHAVIOURAL confidence, and it needs no new run.
#
# Pooled over the roster: 90.1% of items have every draw stating an unqualified
# verdict, but only 67.4% of items have all three draws agreeing. Among the items
# where every draw was confident, 28.0% flip under resampling.
STATE_ORDER = ["confident, stable", "confident, flips",
               "hedged, stable", "hedged, flips"]
STATE_COLOR = {"confident, stable": "#27ae60", "confident, flips": "#c0392b",
               "hedged, stable": "#a8d3ae", "hedged, flips": "#e8a3ae"}


def item_frame(df):
    """One row per (model, item): did every draw claim confidence, and did they agree?

    Restricted to items with all three draws present, so "the draws disagree" is
    always a statement about the same number of draws.
    """
    from freegen_static_judgments.parse_score import valid_intent
    d = df.copy()
    d["lean"] = d["parsed_valid"].map(valid_intent)
    d["conf"] = d["bucket"].str.startswith("Confident")
    g = d.groupby(["model", "gt_sample", "title", "mod_type"])
    it = g.agg(n=("lean", "size"),
               yes=("lean", lambda s: (s == True).sum()),
               no=("lean", lambda s: (s == False).sum()),
               all_conf=("conf", "all")).reset_index()
    it = it[it["n"] == 3].copy()
    # Items whose draws all stated a direction. load() already removed the no-lean
    # DRAWS, so an item that lost one is short of three and is dropped here -- which
    # is what we want: "the draws disagree" must not be able to mean "one of them
    # declined to answer".
    it["stable"] = (it["yes"] == 3) | (it["no"] == 3)
    it["state"] = [("confident, " if c else "hedged, ") + ("stable" if s else "flips")
                   for c, s in zip(it["all_conf"], it["stable"])]
    return it


def fig_overclaim(df):
    it = item_frame(df)
    models = []
    for m, g in it.groupby("model"):
        conf = g[g["all_conf"]]
        if not len(conf):
            continue
        sub = conf.assign(_flip=(~conf["stable"]).astype(float))
        mn, lo, hi = cluster_ci(sub, "_flip")
        models.append(dict(model=m.split("/")[-1], rate=mn * 100,
                           lo=(mn - lo) * 100 if lo == lo else 0.0,
                           hi=(hi - mn) * 100 if hi == hi else 0.0,
                           n_conf=len(conf), n=len(g), g=g))
    models.sort(key=lambda r: r["rate"])
    names = [r["model"] for r in models]

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.16, shared_yaxes=True,
        subplot_titles=("Confident and yet unstable<br>"
                        "<sup>of items where every draw gave an unqualified verdict,"
                        " the share that flips on resampling</sup>",
                        "What the roster actually does<br>"
                        "<sup>every item, by what it claimed and whether the three"
                        " draws agreed</sup>"))

    fig.add_bar(y=names, x=[r["rate"] for r in models], orientation="h",
                marker_color="#c0392b", showlegend=False,
                error_x=dict(type="data", symmetric=False,
                             array=[r["hi"] for r in models],
                             arrayminus=[r["lo"] for r in models],
                             color="rgba(0,0,0,0.55)", thickness=1.2, width=5),
                row=1, col=1)
    # A value COLUMN at a fixed x, not textposition="outside": plotly places an
    # outside label at the end of the BAR, which is underneath its own error bar.
    for r in models:
        fig.add_annotation(x=68, y=r["model"], xref="x1", yref="y1",
                           text=f"{r['rate']:.0f}%  <span style='opacity:0.6'>"
                                f"(n={r['n_conf']})</span>",
                           showarrow=False, xanchor="left", font=dict(size=12))

    for state in STATE_ORDER:
        fig.add_bar(y=names, x=[100 * (r["g"]["state"] == state).mean() for r in models],
                    orientation="h", name=state, marker_color=STATE_COLOR[state],
                    marker_line=dict(color="white", width=0.5),
                    legendgroup=state, row=1, col=2)

    fig.update_layout(
        template="plotly_white", barmode="stack",
        width=1500, height=520, margin=dict(l=210, r=60, t=95, b=110),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", title=""),
        font=dict(size=13))
    fig.update_xaxes(title_text="% of confident items that flip", range=[0, 92],
                     row=1, col=1)
    fig.update_xaxes(title_text="% of items", range=[0, 100], row=1, col=2)
    return fig


# ── Does the model name the PDE? ──────────────────────────────────────────────
# A different question from validity, on the same 256 items. `pde_match` is an
# alias-aware keyword match of the model's `pde:` field against ground truth, so it
# credits "inviscid Burgers" for "burgers" and does not credit a near miss.
#
# Two things worth separating, hence two panels. Some PDEs are simply harder --
# Navier-Stokes sits ~20 points below the other three for every model on the roster.
# And naming degrades when only the ANNOTATION is perturbed: a corrupted comment or
# obfuscated identifiers leave the physics byte-identical, so a model reading the
# code should be unaffected. It is not, which is the same label-reading effect the
# validity figures show, measured on a task where the right answer does not move.
PDE_CLASSES = ["burgers", "heat", "wave", "navier-stokes"]
PDE_CLASS_COLOR = {"burgers": "#2980b9", "heat": "#e67e22",
                   "wave": "#27ae60", "navier-stokes": "#8e44ad"}
CLEAN_CONDS = ["Comm_Valid", "NoComm_Valid"]
ANNOT_CONDS = ["CorrComm", "NoComm_CorrVar"]


def _pde_label(d, model, full=256):
    """Model name, marked with its item count while it is still short.

    GLM's first batch walks gt_samples in order, so its 32 items are ALL Burgers --
    three of the four class bars are missing rather than low, and an unmarked row
    invites reading its single bar as a roster-topping score.
    """
    n = d[d["model"].eq(model)].groupby(["title", "mod_type"]).ngroups
    short = model.split("/")[-1]
    return short if n >= full else f"{short} (partial, {n}/{full})"


def fig_pde_naming(df):
    # The run-on rows are EXCLUDED here, not just noted. On those, parsed_pde holds
    # the entire single-line answer, so the alias match fires on a string that also
    # contains the method and behaviour text -- it scores 0.818 against Nemotron's
    # own 0.680 on normally parsed rows. That is the mis-parse flattering the metric,
    # not the model doing better, and it would put Nemotron's bar in the wrong place.
    d = df[df["parsed_valid"].notna()].copy()
    n_drop = len(df) - len(d)
    if n_drop:
        print(f"[paper] pde figure: excluding {n_drop} run-on row(s) whose parsed_pde "
              f"holds the whole answer")

    models = sorted(d["model"].unique(),
                    key=lambda m: d[d["model"].eq(m)]["pde_match"].mean())

    # A model still generating is marked, with its item count. GLM's first batch
    # walks gt_samples in order, so its 32 items are ALL Burgers -- three of the
    # four class bars are missing rather than low, and an unmarked row invites
    # reading its single bar as a roster-topping score.
    names = [_pde_label(d, m) for m in models]

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.13, shared_yaxes=True,
        subplot_titles=("Naming accuracy by PDE class<br>"
                        "<sup>95% CI, clustered on the 32 base systems; "
                        "a class bar rests on 8 of them</sup>",
                        "Cost of perturbing the annotation alone<br>"
                        "<sup>clean → corrupted comment or obfuscated identifiers;"
                        " the physics is unchanged<br>"
                        "bar is the 95% CI on the PAIRED delta, not on the endpoint"
                        "</sup>"))

    for cls in PDE_CLASSES:
        xs, hi_err, lo_err = [], [], []
        for m in models:
            sub = d[d["model"].eq(m) & d["pde_class"].eq(cls)]
            xs.append(100 * sub["pde_match"].mean() if len(sub) else float("nan"))
            # A class bar rests on eight of the 32 systems, so these are the widest
            # intervals in the file -- which is the point. The partial rosters marked
            # in _pde_label are the same warning stated as a number.
            p, m_ = ci_err(sub, "pde_match") if len(sub) else (0.0, 0.0)
            hi_err.append(p); lo_err.append(m_)
        fig.add_bar(y=names, x=xs, orientation="h", name=cls,
                    marker_color=PDE_CLASS_COLOR[cls],
                    error_x=dict(type="data", symmetric=False, array=hi_err,
                                 arrayminus=lo_err, **ERR_STYLE),
                    marker_line=dict(color="white", width=0.4), row=1, col=1)

    # Right panel: one row per model, clean and annotation-perturbed joined.
    for i, m in enumerate(models):
        g = d[d["model"].eq(m)]
        clean, annot = g[g["mod_type"].isin(CLEAN_CONDS)], g[g["mod_type"].isin(ANNOT_CONDS)]
        a = clean["pde_match"].mean() * 100
        b = annot["pde_match"].mean() * 100
        # The connecting line is a DELTA claim -- perturbing the annotation alone
        # costs this much -- so the interval that belongs on this panel is the
        # interval on the DELTA, drawn at the perturbed endpoint. Marginal intervals
        # on both ends were the first thing tried here and they are actively
        # misleading: both arms are the same 32 systems, so the marginals are roughly
        # twice the width of the paired interval and they overlap for every model on
        # the roster, which renders a contrast the paper reports as real as a picture
        # of nothing happening. The clean endpoint carries no bar; it is the
        # reference the delta is measured from.
        _, d_lo, d_hi = paired_delta_ci(clean, annot, "pde_match")
        b_hi = (d_hi - (b - a)) if d_hi == d_hi else 0.0
        b_lo = ((b - a) - d_lo) if d_lo == d_lo else 0.0
        fig.add_scatter(x=[a, b], y=[names[i], names[i]], mode="lines",
                        line=dict(color="#95a5a6", width=2), showlegend=False,
                        hoverinfo="skip", row=1, col=2)
        fig.add_scatter(x=[a], y=[names[i]], mode="markers", name="clean",
                        marker=dict(color="#27ae60", size=11),
                        showlegend=(i == 0), legendgroup="clean",
                        hovertemplate="clean: %{x:.1f}%<extra></extra>", row=1, col=2)
        fig.add_scatter(x=[b], y=[names[i]], mode="markers",
                        name="annotation corrupted",
                        marker=dict(color="#c0392b", size=11, symbol="circle-open",
                                    line=dict(width=2.5, color="#c0392b")),
                        error_x=dict(type="data", symmetric=False, array=[b_hi],
                                     arrayminus=[b_lo], **ERR_STYLE),
                        showlegend=(i == 0), legendgroup="annot",
                        hovertemplate="annotation corrupted: %{x:.1f}%<extra></extra>",
                        row=1, col=2)

    fig.update_layout(
        template="plotly_white", barmode="group", bargap=0.25,
        width=1500, height=520, margin=dict(l=215, r=50, t=115, b=105),
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                    yanchor="top", title=""),
        font=dict(size=13))
    fig.update_xaxes(title_text="% of items where the PDE was named correctly",
                     range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="% named correctly", range=[0, 105], row=1, col=2)
    return fig


# ── Every perturbation, per model, on PDE naming ──────────────────────────────
# The cleanest test in the dataset. Unlike validity, the correct answer here does
# NOT move across the eight conditions: the same solver is the same PDE whether its
# comment lies, its identifiers are foobar_N, or its physics is broken. So every
# bar in a panel should be the same height, and any deviation is the model reading
# something other than the code.
#
# What comes out is specific rather than general. It is not perturbation that hurts,
# it is IDENTIFIER OBFUSCATION: the two conditions with foobar_N names are the low
# bars for every model that moves at all, and the invalid-physics conditions barely
# register. Qwen3.5/3.6/3.8 are flat across all eight (spread 3.1-5.1 points) while
# Qwen3-32B, R1-Distill and Nemotron swing 21-24.
# What was done to the ANNOTATION, which is the axis this figure is about. A two-way
# obfuscated/intact split was wrong: CorrComm and CorrComm_Invalid carry a corrupted
# COMMENT, so grouping them with the untouched conditions as "identifiers intact"
# implied their annotation was clean when a different part of it had been attacked.
# Three states, and the physics-invalid conditions are annotation-intact -- broken
# physics is not an annotation change.
ANNOTATION_STATE = {
    "Comm_Valid":             "annotation intact",
    "NoComm_Valid":           "annotation intact",
    "Comm_InValid":           "annotation intact",
    "NoComm_InValid":         "annotation intact",
    "CorrComm":               "comment corrupted",
    "CorrComm_Invalid":       "comment corrupted",
    "NoComm_CorrVar":         "identifiers obfuscated",
    "NoComm_CorrVar_InValid": "identifiers obfuscated",
}
ANNOTATION_COLOR = {"annotation intact": "#5b7c99",
                    "comment corrupted": "#e67e22",
                    "identifiers obfuscated": "#8e44ad"}


def fig_pde_by_perturbation(df):
    d = df[df["parsed_valid"].notna()].copy()      # run-on rows: see fig_pde_naming
    present = present_conds(d)
    labels = labels_for(present)
    split = sum(1 for _, _, gt in present if gt)

    # Ordered by SPREAD, so the flat models group together and the sensitive ones
    # read as a block rather than being scattered by overall accuracy.
    def spread(m):
        g = d[d["model"].eq(m)]
        v = [g[g["mod_type"].eq(c)]["pde_match"].mean() for c, _, _ in present]
        v = [x for x in v if x == x]
        return (max(v) - min(v)) if v else 0.0
    models = sorted(d["model"].unique(), key=spread)

    ncols = 4
    nrows = (len(models) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True,
                        subplot_titles=[_pde_label(d, m) for m in models],
                        vertical_spacing=0.09, horizontal_spacing=0.045)

    for i, m in enumerate(models):
        r, c = i // ncols + 1, i % ncols + 1
        g = d[d["model"].eq(m)]
        ys, cols, hi_err, lo_err = [], [], [], []
        for code, _, _ in present:
            sub = g[g["mod_type"].eq(code)]
            ys.append(100 * sub["pde_match"].mean() if len(sub) else 0.0)
            cols.append(ANNOTATION_COLOR[ANNOTATION_STATE[code]])
            # This figure's claim is about the SPREAD across the eight bars within a
            # panel -- "flat at 3-5 points" against "swings 21-24". Without intervals
            # a reader cannot tell either apart from 96 draws of noise, and the flat
            # models are the ones where that distinction is the whole result.
            p, m = ci_err(sub, "pde_match") if len(sub) else (0.0, 0.0)
            hi_err.append(p); lo_err.append(m)
        fig.add_bar(x=labels, y=ys, marker_color=cols, showlegend=False,
                    marker_line=dict(color="white", width=0.4),
                    error_y=dict(type="data", symmetric=False, array=hi_err,
                                 arrayminus=lo_err, **ERR_STYLE),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>", row=r, col=c)
        # No baseline rule. The two Clean bars ARE the reference and they are right
        # there at the left of every panel, so a line drawn through their mean added
        # a third thing to decode without adding a number the reader cannot already
        # see.
        fig.add_vline(x=split - 0.5, line=dict(color="#b0b7c3", width=1, dash="dash"),
                      row=r, col=c)

    # Legend faked with invisible traces: the bars carry two meanings through one
    # colour array, which plotly cannot legend on its own.
    for name in ("annotation intact", "comment corrupted", "identifiers obfuscated"):
        fig.add_bar(x=[None], y=[None], name=name,
                    marker_color=ANNOTATION_COLOR[name])

    # The bars are MARGINAL intervals, and every comparison a reader will make in
    # this figure is between two bars of the SAME panel -- i.e. the same 32 systems
    # under two perturbations. Those marginals are about twice the width of the
    # paired interval on the difference, so eyeballing their overlap understates
    # every within-panel effect. On the clean/annotation contrast the marginals
    # overlap for all eight models while the paired delta clears zero for four of
    # them. Said on the figure because that is where the mistake gets made.
    fig.add_annotation(
        text="95% CI, bootstrap over the 32 base systems. Intervals are marginal — "
             "within-panel comparisons are paired, so overlap here does not mean "
             "no difference (see paired deltas in text).",
        xref="paper", yref="paper", x=0.5, y=-0.30, xanchor="center", yanchor="top",
        showarrow=False, font=dict(size=11, color="#5a6270"))

    fig.update_layout(
        template="plotly_white", barmode="group", bargap=0.25,
        width=1500, height=380 * nrows + 190,
        margin=dict(l=75, r=40, t=42, b=255),
        legend=dict(orientation="h", y=-0.235, x=0.5, xanchor="center",
                    yanchor="top", title=""),
        font=dict(size=12))
    fig.update_yaxes(range=[0, 105], title_text="")
    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text="% PDE named correctly", row=r, col=1)
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showticklabels=False)
    for col in range(ncols):
        rows = [i // ncols + 1 for i in range(len(models)) if i % ncols == col]
        if rows:
            fig.update_xaxes(showticklabels=True, row=max(rows), col=col + 1)
    for ann in fig.layout.annotations[:len(models)]:
        ann.font = dict(size=13)
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    n_models = df["model"].nunique()
    print(f"[paper] {len(df)} rows, {n_models} models from {CSV}")
    for name, fig in (("paper_hedge_pooled", fig_pooled(df)),
                      ("paper_hedge_by_model", fig_by_model(df)),
                      ("paper_overclaim", fig_overclaim(df)),
                      ("paper_pde_naming", fig_pde_naming(df)),
                      ("paper_pde_by_perturbation", fig_pde_by_perturbation(df))):
        # PNG only. The HTML twins were never opened -- these go into a paper, and
        # an interactive copy beside every figure is just another file to keep in
        # sync with the PNG that is actually used.
        for out in [name] + ALIASES.get(name, []):
            png = os.path.join(OUT_DIR, f"{out}.png")
            fig.write_image(png, scale=SCALE)
            print(f"[paper] wrote {png}")


if __name__ == "__main__":
    main()
