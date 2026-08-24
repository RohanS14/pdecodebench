"""
paper_hedge_figures.py — the validity-confidence breakdown, as light PNGs.

Separate from paper_figures.py because that script loads the MC logprob data and
resolves its inputs relative to viz/; these two figures need only the free-generation
aggregate. Light background (plotly_white) and scale=2 PNG, so the output goes into a
paper without a screenshot.

Two figures, and the second is the one that carries the argument:

    paper_hedge_pooled.png     eight perturbations, every model pooled
    paper_hedge_by_model.png   the same, one small multiple per model
    paper_overclaim.png        stated confidence against resampling stability

The pooled figure alone is misleading on this roster. The newer checkpoints answer
the invalid half almost perfectly and the valid half barely better than chance; the
older ones do the reverse. Averaging two opposite biases produces a bar that reads as
moderate competence at both, which is the one thing these figures must not say.

Usage:
    python viz/paper_hedge_figures.py
    PDE_FREEGEN_CSV=results/freegen_xmodal.csv OUT_DIR=figures python viz/paper_hedge_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from freegen.parse_score import classify_valid_confidence_2x2   # noqa: E402

CSV = os.environ.get("PDE_FREEGEN_CSV", "results/freegen_xmodal.csv")
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
ORDER = ["says invalid", "no lean", "says valid"]
COLOR = {"says invalid": "#c0392b", "no lean": "#95a5a6", "says valid": "#27ae60"}
# Verdict bucket -> the direction it states. "no lean" has no direction and stays.
DIRECTION = {"Confident No": "says invalid", "Hedged No": "says invalid",
             "Confident Yes": "says valid", "Hedged Yes": "says valid",
             "no lean": "no lean"}
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
    df["direction"] = df["bucket"].map(DIRECTION).fillna("no lean")
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
    groups = [g[col].dropna().to_numpy() for _, g in sub.groupby("gt_sample")]
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return float(vals.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boots = np.array([np.concatenate([groups[j] for j in row]).mean() for row in idx])
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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
        title="What the models say — all perturbations, all models pooled<br><sup>95% bootstrap CI over the 32 base systems, on each cumulative boundary</sup>",
        yaxis=dict(title="% of answers", range=[0, 105], dtick=20),
        xaxis=dict(tickangle=-25),
        width=1100, height=560, margin=dict(l=70, r=40, t=70, b=150),
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
        for bucket in ORDER:
            ys = []
            for code, _, _ in present:
                sub = g[g["mod_type"].eq(code)]
                ys.append(100 * (sub["direction"] == bucket).mean() if len(sub) else 0.0)
            fig.add_bar(x=labels, y=ys, name=bucket, marker_color=COLOR[bucket],
                        marker_line=dict(color="white", width=0.4),
                        legendgroup=bucket, showlegend=(i == 0), row=r, col=c)
        fig.add_vrect(x0=split - 0.5, x1=len(present) - 0.5, fillcolor=INVALID_BAND,
                      line_width=0, layer="below", row=r, col=c)
        fig.add_vline(x=split - 0.5, line=dict(color=DIVIDER, width=1, dash="dash"),
                      row=r, col=c)

    fig.update_layout(template="plotly_white", barmode="stack",
                      title="What each model says — the same eight perturbations",
                      width=1500, height=380 * nrows + 240,
                      margin=dict(l=60, r=40, t=125, b=250),
                      # Legend ABOVE the panels. At the bottom it sat in the same
                      # strip as the vertical tick labels and clipped the middle two
                      # columns -- "Corrupt Comment" rendered as "upt Comment". The
                      # top strip is empty apart from the title.
                      legend=dict(orientation="h", y=1.045, x=0.5, xanchor="center",
                                  yanchor="bottom", title=""),
                      font=dict(size=12))
    fig.update_yaxes(range=[0, 100], title_text="")
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
    from freegen.parse_score import valid_intent
    d = df.copy()
    d["lean"] = d["parsed_valid"].map(valid_intent)
    d["conf"] = d["bucket"].str.startswith("Confident")
    g = d.groupby(["model", "gt_sample", "title", "mod_type"])
    it = g.agg(n=("lean", "size"),
               yes=("lean", lambda s: (s == True).sum()),
               no=("lean", lambda s: (s == False).sum()),
               all_conf=("conf", "all")).reset_index()
    it = it[it["n"] == 3].copy()
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
        title="Stated confidence is not confidence<br><sup>Three samples of one prompt "
              "at temperature 0.6. Pooled: 90.1% of items have every draw stating an "
              "unqualified verdict; only 67.4% have all three draws agreeing.</sup>",
        width=1500, height=560, margin=dict(l=210, r=60, t=170, b=110),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", title=""),
        font=dict(size=13))
    fig.update_xaxes(title_text="% of confident items that flip", range=[0, 92],
                     row=1, col=1)
    fig.update_xaxes(title_text="% of items", range=[0, 100], row=1, col=2)
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    n_models = df["model"].nunique()
    print(f"[paper] {len(df)} rows, {n_models} models from {CSV}")
    for name, fig in (("paper_hedge_pooled", fig_pooled(df)),
                      ("paper_hedge_by_model", fig_by_model(df)),
                      ("paper_overclaim", fig_overclaim(df))):
        png = os.path.join(OUT_DIR, f"{name}.png")
        fig.write_image(png, scale=SCALE)
        fig.write_html(os.path.join(OUT_DIR, f"{name}.html"), include_plotlyjs="cdn")
        print(f"[paper] wrote {png}")


if __name__ == "__main__":
    main()
