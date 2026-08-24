"""tradeoff.py — what the conditional ladder metric cannot show.

The ladder reports localization accuracy CONDITIONAL on the model having committed
to a view. That conditioning is deliberate and defensible: it is the one accuracy a
flag-everything model cannot inflate, because a model that cries foul on every item
still has to point somewhere. But it buys that immunity by removing the clean items
from the denominator entirely, and the clean items are where crying foul is scored.

So the conditional metric is silent on the failure mode the clean items measure, and
across the Qwen ladder those two quantities move in OPPOSITE directions. Reading the
ladder alone, a reader concludes the models are getting better at this task. Reading
it with the clean items back in, the picture is that the models became much better at
naming the right view and meaningfully worse at not raising the alarm on items where
nothing was wrong.

This module reports both, plus the one number that cannot be gamed by either
behaviour on its own:

  MACRO-5 RECALL. Five answer classes -- the four representations, plus "nothing was
  corrupted" -- each scored on its own items and then averaged unweighted. A model
  that names a view on everything scores 0 on the fifth class. A model that says
  everything agrees scores 0 on the other four. Both halves have to work.

  Unweighted is load-bearing. The design puts trajectory in 4 of its 7 corrupted
  conditions and clean items in 1 of 8, so a micro average over draws is 57%
  trajectory and 12.5% clean; a model that answers "trajectory" by reflex is paid
  for it. See the design appendix for where those proportions come from.

Everything is computed on the same common item set and bootstrapped over the same
solver systems as ladder.py, so the numbers here and there are commensurable.
"""
import numpy as np
import pandas as pd

from . import style
from .constants import MODALITIES, MODALITY_LABELS, NONE
from . import ladder as L

import matplotlib.pyplot as plt

# The fifth class. Named here rather than reusing NONE so the intent is legible at
# every call site: this is "the model correctly declined to accuse anything".
CLEAN = "none"
CLASSES = MODALITIES + (CLEAN,)
CLASS_LABELS = {**MODALITY_LABELS, CLEAN: "nothing corrupted"}
# Colour-blind-safe, matching the report's modality palette; the fifth class is
# deliberately outside it because it is not a representation.
CLASS_COLORS = {"C": "#0072B2", "T": "#D55E00", "D": "#009E73", "M": "#CC79A7",
                CLEAN: "#666666"}


def _counts_by_solver(g):
    """Per solver, per class: (n correct, n items of that class).

    Returned as two (S, 5) integer arrays plus the solver order. Bootstrapping then
    resamples ROWS of those arrays, which is resampling solver systems -- the
    independent unit -- and recomputes any statistic from the sums exactly. Doing it
    this way rather than re-filtering the frame 2,000 times is what keeps a macro
    statistic's interval affordable.
    """
    solvers = sorted(set(g["solver_id"]))
    idx = {s: i for i, s in enumerate(solvers)}
    right = np.zeros((len(solvers), len(CLASSES)), dtype=np.int64)
    total = np.zeros((len(solvers), len(CLASSES)), dtype=np.int64)
    cls = {c: k for k, c in enumerate(CLASSES)}
    for s, t, p in zip(g["solver_id"], g["true_outlier"], g["pred_outlier"]):
        k = cls.get(t if t in MODALITIES else CLEAN)
        i = idx[s]
        total[i, k] += 1
        # For the four representation classes, correct means naming that view. For
        # the clean class, correct means naming NO view -- an answer of "none", not
        # a lucky match. An unparsed answer is not a decline and is not correct.
        if t in MODALITIES:
            right[i, k] += int(p == t)
        else:
            right[i, k] += int(p == NONE)
    return right, total, solvers


def _detected(pred_agree):
    """The model's claim that something disagrees.

    Exactly metrics.prepare()'s definition -- `pred_agree == "no"` -- rather than a
    second one written here. Detection and commitment are NOT the same event: a model
    can flag an item and then name no readable view, and conditioning on commitment
    silently drops those rows while conditioning on detection scores them as the
    failures they are.
    """
    return str(pred_agree).strip().lower() == "no"


def _det_counts_by_solver(g):
    """Per solver: (named the right view, flagged) over CORRUPTED items.

    Denominator is every corrupted item the model flagged, whether or not it went on
    to name something readable. That is what "given the model detected an
    inconsistency" means, and it is a stricter denominator than the ladder's
    "given it committed to a view".
    """
    solvers = sorted(set(g["solver_id"]))
    idx = {s: i for i, s in enumerate(solvers)}
    num = np.zeros(len(solvers), dtype=np.int64)
    den = np.zeros(len(solvers), dtype=np.int64)
    for s, t, p, a in zip(g["solver_id"], g["true_outlier"], g["pred_outlier"],
                          g["pred_agree"]):
        if t in MODALITIES and _detected(a):
            i = idx[s]
            den[i] += 1
            num[i] += int(p == t)
    return num, den


def _fa_counts_by_solver(g):
    """Per solver: (flagged, all clean items) -- the false-alarm rate.

    Clean items only. On an item where all four views agree there is nothing to
    detect, so every flag here is a false alarm by construction, not a low-
    probability guess.
    """
    solvers = sorted(set(g["solver_id"]))
    idx = {s: i for i, s in enumerate(solvers)}
    num = np.zeros(len(solvers), dtype=np.int64)
    den = np.zeros(len(solvers), dtype=np.int64)
    for s, t, a in zip(g["solver_id"], g["true_outlier"], g["pred_agree"]):
        if t == NONE:
            i = idx[s]
            den[i] += 1
            num[i] += int(_detected(a))
    return num, den


def _pct(lo, hi, vals):
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return (np.nan, np.nan)
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))


def per_model(t, items, models, n_boot=L.N_BOOT, seed=L.BOOT_SEED):
    """Every trade-off metric for every model, on the common item set only."""
    rng = np.random.default_rng(seed)
    out = []
    for m in models:
        g = t[(t["model"] == m) & t["item_id"].isin(items)]
        right, total, solvers = _counts_by_solver(g)
        cnum, cden = _det_counts_by_solver(g)
        fnum, fden = _fa_counts_by_solver(g)
        S = len(solvers)
        if not S:
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            rec = np.where(total.sum(0) > 0, right.sum(0) / total.sum(0), np.nan)
        macro = float(np.nanmean(rec))
        cond = float(cnum.sum() / cden.sum()) if cden.sum() else np.nan

        # One resample of solver systems, reused for every statistic in this row, so
        # the intervals below are mutually consistent rather than drawn independently.
        bi = rng.integers(0, S, size=(n_boot, S))
        br, bt = right[bi].sum(axis=1), total[bi].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            brec = np.where(bt > 0, br / bt, np.nan)
        bmacro = np.nanmean(brec, axis=1)
        bn, bd = cnum[bi].sum(axis=1), cden[bi].sum(axis=1)
        fn_, fd_ = fnum[bi].sum(axis=1), fden[bi].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bcond = np.where(bd > 0, bn / bd, np.nan)
            bfa = np.where(fd_ > 0, fn_ / fd_, np.nan)

        row = {"model": m, "n_solvers": S,
               "loc_det": cond, "macro5": macro,
               "fa_rate": float(fnum.sum() / fden.sum()) if fden.sum() else np.nan,
               "clean_ok": float(rec[CLASSES.index(CLEAN)])}
        row["loc_det_lo"], row["loc_det_hi"] = _pct(2.5, 97.5, bcond)
        row["fa_lo"], row["fa_hi"] = _pct(2.5, 97.5, bfa)
        row["macro5_lo"], row["macro5_hi"] = _pct(2.5, 97.5, bmacro)
        k = CLASSES.index(CLEAN)
        row["clean_ok_lo"], row["clean_ok_hi"] = _pct(2.5, 97.5, brec[:, k])
        for j, c in enumerate(CLASSES):
            row[f"rec_{c}"] = float(rec[j])
            row[f"rec_{c}_lo"], row[f"rec_{c}_hi"] = _pct(2.5, 97.5, brec[:, j])
        out.append(row)
    return pd.DataFrame(out)


def _decorate(fr, cfg):
    """Attach the declared short names and split ladder from reference."""
    short = {m["model_id"]: m["short"] for m in cfg["ladder"] + cfg["reference"]}
    par = {m["model_id"]: m["params_b"] for m in cfg["ladder"] + cfg["reference"]}
    fr = fr.copy()
    fr["short"] = fr["model"].map(short)
    fr["params_b"] = fr["model"].map(par)
    lad_ids = [m["model_id"] for m in cfg["ladder"]]
    ref_ids = [m["model_id"] for m in cfg["reference"]]
    f = fr.set_index("model")
    return (f.loc[[i for i in lad_ids if i in f.index]].reset_index(),
            f.loc[[i for i in ref_ids if i in f.index]].reset_index())


def fig_opposing(lad, ref):
    """The two halves, side by side, on one shared y-axis.

    Shared y is the whole point of the figure. On separate auto-scaled axes both
    panels would show a line crossing most of its own box and the reader would have
    to compare two different rulers to see that one is rising and the other falling
    over the same range. One ruler, two directions.
    """
    style.apply(style.theme())
    metrics = [("loc_det", "loc_det_lo", "loc_det_hi"),
               ("fa_rate", "fa_lo", "fa_hi")]
    ylim = L._limits(lad, ref, metrics)
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(1.95, 4.15), sharey=True)
    L._panel(axes[0], lad, ref, *metrics[0],
             "Localization accuracy,\ngiven it detected an inconsistency", ylim=ylim)
    # Right panel is a RATE OF ERROR, so up is bad here and up is good on the left.
    # That is the figure's whole content and the reason both panels are drawn on one
    # ruler: the two lines rise together, and only one of those rises is progress.
    L._panel(axes[1], lad, ref, *metrics[1],
             "False-alarm rate: flagged a disagreement\non items where there was none",
             ylim=ylim)
    axes[1].annotate("higher is worse", xy=(0.02, 0.965), xycoords="axes fraction",
                     ha="left", va="top", fontsize=8, style="italic",
                     color=style.colors()["muted"])
    axes[0].annotate("higher is better", xy=(0.02, 0.965), xycoords="axes fraction",
                     ha="left", va="top", fontsize=8, style="italic",
                     color=style.colors()["muted"])
    axes[0].set_ylabel("proportion", fontsize=9)
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(right=0.855, wspace=0.42, top=0.80, bottom=0.225, left=0.075)
    # Title COMPUTED from the endpoints, not written. "Both lines rise" was true of
    # this data and would have quietly become false the day a release brought the
    # false-alarm rate back down -- and a caption that asserts the finding is the
    # last place anyone looks for a stale claim.
    d_loc = (lad["loc_det"].iloc[-1] - lad["loc_det"].iloc[0]) * 100
    d_fa = (lad["fa_rate"].iloc[-1] - lad["fa_rate"].iloc[0]) * 100
    if d_loc > 0 and d_fa > 0:
        head = (f"Localization improved {abs(d_loc):.0f}pp and false alarms rose "
                f"{abs(d_fa):.0f}pp. Only one of those is progress.")
    elif d_loc > 0 and d_fa <= 0:
        head = (f"Localization improved {abs(d_loc):.0f}pp and false alarms fell "
                f"{abs(d_fa):.0f}pp \u2014 both halves moved the right way.")
    else:
        head = (f"Localization moved {d_loc:+.0f}pp, false alarms {d_fa:+.0f}pp, "
                f"first release to last.")
    fig.suptitle(head, fontsize=11.5, y=0.985)
    return fig


def fig_classes(lad, ref):
    """Five per-class recalls across the ladder, with the unweighted mean in bold.

    Unconditional: each line's denominator is every item of that class, so declining
    to answer costs the same as answering wrongly. That is what makes the fifth line
    comparable to the other four rather than a different kind of quantity.
    """
    style.apply(style.theme())
    c = style.colors()
    fig, ax = plt.subplots(figsize=style.figsize(1.95, 3.6))
    x = np.arange(len(lad))

    if len(ref):
        v = ref["macro5"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            ax.axhspan(v.min(), v.max(), color=c["muted"], alpha=0.11, zorder=0)
            ax.axhline(np.median(v), color=c["muted"], linewidth=0.9, alpha=0.5,
                       zorder=1)

    # Right-margin labels are nudged apart before they are drawn. Trajectory and
    # math finish 1pp apart on the newest model and printed at their true heights
    # they overlap into an unreadable smear -- the same failure the ladder's
    # singleton labels hit, so the same fix.
    ends = np.array([lad[f"rec_{c}"].to_numpy(dtype=float)[-1] for c in CLASSES])
    label_y = L._spread(ends, min_gap=0.055)
    for cls, ly in zip(CLASSES, label_y):
        y = lad[f"rec_{cls}"].to_numpy(dtype=float)
        ax.plot(x, y, "-o", color=CLASS_COLORS[cls], linewidth=1.3, markersize=4.5,
                alpha=0.9, zorder=2)
        if abs(ly - y[-1]) > 1e-9:
            ax.plot([x[-1] + 0.04, x[-1] + 0.14], [y[-1], ly],
                    color=CLASS_COLORS[cls], linewidth=0.7, alpha=0.55,
                    clip_on=False, zorder=2)
        ax.annotate(CLASS_LABELS[cls], xy=(x[-1] + 0.16, ly), va="center",
                    ha="left", fontsize=8, color=CLASS_COLORS[cls],
                    annotation_clip=False)

    y = lad["macro5"].to_numpy(dtype=float)
    yerr = np.vstack([y - lad["macro5_lo"], lad["macro5_hi"] - y])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    ax.errorbar(x, y, yerr=yerr, fmt="-o", color=c["fg"], ecolor=c["fg"],
                elinewidth=1.3, capsize=4, markersize=7, linewidth=2.4,
                markeredgecolor=c["bg"], markeredgewidth=0.8, zorder=4)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi * 100:.0f}%", xy=(xi, yi), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=c["fg"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(lad["short"], fontsize=9)
    ax.set_xlim(-0.35, len(lad) - 1 + 0.62)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("recall on that class's own items", fontsize=9)
    ax.set_title("Each answer class scored on its own items, unweighted mean in black",
                 fontsize=10.5, pad=10)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v * 100:.0f}%")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], color=c["fg"], marker="o", linewidth=2.4, markersize=6,
                   label="macro-5: the unweighted mean of the five (95% CI)"),
        plt.Line2D([], [], color=c["muted"], linewidth=0.9, alpha=0.5,
                   label="band = macro-5 range of the 4 other-lab models, line = median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(right=0.80, top=0.88, bottom=0.155, left=0.085)
    return fig


def verdict(lad, ref):
    """Generated from the numbers. Says "fell" when it fell."""
    if len(lad) < 2:
        return ""
    a, b = lad.iloc[0], lad.iloc[-1]

    def mv(col):
        return (b[col] - a[col]) * 100

    d_cond, d_fa, d_macro = mv("loc_det"), mv("fa_rate"), mv("macro5")
    word = lambda v: "rose" if v > 0 else "fell"
    ref_sp = (np.nanmax(ref["macro5"]) - np.nanmin(ref["macro5"])) * 100 if len(ref) else np.nan
    inside = np.isfinite(ref_sp) and abs(d_macro) <= ref_sp
    best_clean = lad["fa_rate"].idxmin()
    return (
        f"Across the four releases ({a['short']} to {b['short']}), localization "
        f"accuracy given detection {word(d_cond)} {abs(d_cond):.0f}pp, and the "
        f"false-alarm rate {word(d_fa)} {abs(d_fa):.0f}pp. The two are not "
        f"the same finding and only one of them is on the ladder figure. Putting "
        f"both into one unweighted five-class score, the ladder {word(d_macro)} "
        f"{abs(d_macro):.0f}pp"
        + (f", which is inside the {ref_sp:.0f}pp spread across the four "
           f"contemporary other-lab models" if inside else
           f", against a {ref_sp:.0f}pp spread across the four contemporary "
           f"other-lab models" if np.isfinite(ref_sp) else "")
        + f". The lowest false-alarm rate of the four is {lad.loc[best_clean, 'short']}"
        + (" — the oldest." if best_clean == lad.index[0] else ".")
    )


def table(lad, ref):
    import html as _h
    f = lambda v: "&mdash;" if not np.isfinite(v) else f"{v * 100:.1f}%"
    ci = lambda a, b: ("" if not (np.isfinite(a) and np.isfinite(b)) else
                       f" <span style='color:var(--dim)'>({a * 100:.0f}&ndash;{b * 100:.0f})</span>")
    head = ("<tr><th>Model</th><th>Role</th>"
            "<th>Localization, given detection</th>"
            "<th>False-alarm rate</th>"
            + "".join(f"<th>{CLASS_LABELS[c]}</th>" for c in MODALITIES)
            + "<th>Macro-5</th></tr>")
    body = ""
    for fr, role in ((lad, "ladder"), (ref, "other lab")):
        for _, r in fr.iterrows():
            body += (
                f"<tr><td>{_h.escape(str(r['short']))}</td><td>{role}</td>"
                f"<td>{f(r['loc_det'])}{ci(r['loc_det_lo'], r['loc_det_hi'])}</td>"
                f"<td>{f(r['fa_rate'])}{ci(r['fa_lo'], r['fa_hi'])}</td>"
                + "".join(f"<td>{f(r[f'rec_{c}'])}</td>" for c in MODALITIES)
                + f"<td><b>{f(r['macro5'])}</b>{ci(r['macro5_lo'], r['macro5_hi'])}</td></tr>")
    return ("<div style='overflow-x:auto'><table><thead>" + head
            + "</thead><tbody>" + body + "</tbody></table></div>")


def build_section(t, items, cfg, n_boot=None):
    """The whole section as one HTML <section>. Pure function of the tidy frame."""
    import html as _h
    lad_ids = [m["model_id"] for m in cfg["ladder"]]
    ref_ids = [m["model_id"] for m in cfg["reference"]]
    kw = {} if n_boot is None else {"n_boot": n_boot}
    fr = per_model(t, items, lad_ids + ref_ids, **kw)
    if fr.empty:
        return ""
    lad, ref = _decorate(fr, cfg)
    if len(lad) < 2:
        return ""

    a, b = lad.iloc[0], lad.iloc[-1]
    d_fa = (b["fa_rate"] - a["fa_rate"]) * 100
    worst = lad.loc[lad["fa_rate"].idxmax()]

    return (
        '<section id="tradeoff">'
        '<h2>Does it actually get better over time?</h2>'
        '<p class="sub" style="margin-bottom:18px">The ladder figure above reports '
        'localization accuracy <i>given the model committed to naming a view</i>. '
        'That conditioning is there for a reason &mdash; it is the one accuracy a '
        'flag-everything model cannot inflate &mdash; but it pays for that by '
        'dropping the clean items out of the denominator entirely, and the clean '
        'items are exactly where flagging everything is scored. So the ladder '
        'figure is silent about half the task, and over these four releases the two '
        f'halves move <b>the same way, and only one of them is progress</b>: '
        f'localization given detection rises '
        f'{(b["loc_det"] - a["loc_det"]) * 100:.0f}pp, and the false-alarm rate '
        f'{"rises" if d_fa > 0 else "falls"} {abs(d_fa):.0f}pp. '
        '<b>Read the ladder figure with this section, not on its own.</b></p>'

        '<h3 class="pmh">1 &middot; The two halves</h3>'
        '<p class="sub">Same y-axis in both panels, same bootstrap over the same 32 '
        'solver systems, same common item set. Left is the quantity the ladder '
        'plots &mdash; here conditioned on the model having <i>detected</i> an '
        'inconsistency, a slightly stricter denominator than the ladder&rsquo;s '
        '&ldquo;given it committed to a view&rdquo;, because an item flagged but '
        'left unnamed is scored as the failure it is. Right is the quantity both '
        'of them condition away: on the items where all four views agreed, how '
        'often the model raised the alarm anyway. The shaded band and the '
        'margin ticks are the four contemporary models from other labs, unchanged '
        'in meaning from the ladder figure.</p>'
        f'<figure>{_h.escape("")}{_svg(fig_opposing(lad, ref))}</figure>'
        f'<p class="sub"><b>The newest model is not the best at this.</b> '
        f'{_h.escape(str(worst["short"]))} has the worst false-alarm rate of the '
        f'four at {worst["fa_rate"] * 100:.0f}%, and every release after '
        f'{_h.escape(str(a["short"]))} raises more false alarms than it does. A '
        'reader who saw only the left panel would not suspect the right one '
        'exists.</p>'

        '<h3 class="pmh">2 &middot; Both halves in one number</h3>'
        '<p class="sub">Five answer classes &mdash; the four representations, plus '
        '&ldquo;nothing was corrupted&rdquo; &mdash; each scored on its own items '
        'and averaged <b>unweighted</b>. Nothing is conditioned away: declining to '
        'answer costs exactly what answering wrongly costs, so a model cannot buy a '
        'high score with a bias in either direction. A model that accuses something '
        'on every item scores zero on the fifth class; a model that waves everything '
        'through scores zero on the other four.</p>'
        f'<figure>{_svg(fig_classes(lad, ref))}</figure>'
        f'<p class="sub"><b>Verdict.</b> {verdict(lad, ref)}</p>'
        '<p class="sub"><b>Why unweighted.</b> The design places trajectory in four '
        'of its seven corrupted conditions and clean items in one of eight, so a '
        'micro average over draws is 57% trajectory and 12.5% clean and pays a '
        'model for answering &ldquo;trajectory&rdquo; by reflex. Those proportions '
        'are a property of the item construction, not of the models; see the design '
        'appendix.</p>'
        '<details><summary class="sub">Details &mdash; every class, every model, '
        'with 95% intervals</summary>'
        f'{table(lad, ref)}'
        '<p class="sub" style="margin-top:10px">Intervals resample the 32 solver '
        'systems, not the draws, and one resample is shared across all the '
        'statistics in a row so they stay mutually consistent. The four '
        'representation columns are <b>unconditional</b>: their denominator is every '
        'item of that class, which is why each sits at or below the conditional '
        'figure in column three.</p>'
        '</details>'
        '</section>')


def _svg(fig):
    from .claim_report import _svg as s
    return s(fig)
