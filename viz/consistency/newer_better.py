"""Are newer models better at this? Rates only -- no d', no z-transform.

WHY A SEPARATE MODULE FROM generational.py
==========================================
generational.py answers the same question in d' and is wired into the pooled
figures. This one is deliberately in percentage points end to end, because d'
answers "how separable are the two distributions" and the question here is the
plainer one: does a newer model flag more, false-alarm more, and land on the right
view more often. Mixing the two in one section would invite reading a d' movement
as a rate movement.

WHY MACRO, NOT MICRO
====================
Conditional localization pooled over items -- "micro" -- is weighted by how many
items each true-outlier class contributes, and the design gives trajectory FOUR of
the seven corrupted conditions against one each for code, description and math. A
model that answers "trajectory" reflexively is therefore scored on the class it
gets right four times as heavily as on the three it does not. The effect is not
hypothetical and not small: on the current roster GLM-4.7-Flash reads 78.0% micro
against 63.8% macro, because it localizes trajectory at 92% and code at 44%; and
Qwen3.5-27B reads 83.1% against 72.6% for the same reason.

macro takes the unweighted mean over the four true-outlier classes, so a model has
to localize all four to score well. Micro is reported beside it in the details
table rather than dropped, because the gap between them IS the diagnostic -- a
large gap is the signature of a model riding one class.

WHY THE TREND IS REPORTED WITH LEAVE-ONE-OUT
============================================
Eight models is not enough for a rank correlation to be stable. Dropping one point
moves rho far more than its p-value suggests, so every rho here is published with
the full leave-one-out range and the model whose removal produces each extreme. A
rho quoted without that range at n=8 is a number pretending to be a finding.
"""
import html as _h

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import metrics as M
from . import style
from .claim_report import _svg
from .constants import MODALITIES, MODALITY_LABELS

ALPHA = 0.05
# The two series in the top panel. Named here so the figure, the verdict and the
# table cannot disagree about which is which.
FLAG_C, FLAG_K = "flag_corrupted", "flag_clean"
# Series colours, from the same colour-blind-safe set the modality palette uses.
# Not MODALITY_COLORS itself: these series are not modalities, and borrowing the
# code/trajectory colours here would imply a link to those views that is not there.
C_FLAGGED, C_CLEAN, C_MACRO = "#0072B2", "#D55E00", "#009E73"
# Which side each singleton's name sits on. Hand-set because the collisions are a
# property of THIS roster's dates and values, not something a rule can infer without
# a full label-placement solver.
_LABEL_ABOVE = {"R1-Distill-32B": True, "QwQ-32B": False,
                "Nemotron-3-Nano-30B": True, "GLM-4.7-Flash": False}


def _spearman(ranks, vals):
    """rho and its two-sided p. scipy if present, exact-enough fallback if not."""
    r = np.asarray(ranks, float)
    v = np.asarray(vals, float)
    ok = np.isfinite(r) & np.isfinite(v)
    r, v = r[ok], v[ok]
    if len(r) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(r, v)
        return float(rho), float(p)
    except Exception:                                             # noqa: BLE001
        rr = pd.Series(r).rank().to_numpy()
        vv = pd.Series(v).rank().to_numpy()
        rho = float(np.corrcoef(rr, vv)[0, 1])
        n = len(r)
        t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return rho, float(p)


def trend(res, order, key):
    """rho, p, and the leave-one-out range with the model behind each extreme."""
    mids = [m for m in order if m in res]
    ranks = list(range(len(mids)))
    vals = [res[m][key] for m in mids]
    rho, p = _spearman(ranks, vals)
    loo = []
    for i, drop in enumerate(mids):
        keep = [j for j in range(len(mids)) if j != i]
        r2, _ = _spearman(list(range(len(keep))), [vals[j] for j in keep])
        loo.append((r2, res[drop]["short"]))
    loo = [x for x in loo if np.isfinite(x[0])]
    lo = min(loo) if loo else (float("nan"), "")
    hi = max(loo) if loo else (float("nan"), "")
    return {"rho": rho, "p": p, "n": len(mids),
            "loo_lo": lo[0], "loo_lo_drop": lo[1],
            "loo_hi": hi[0], "loo_hi_drop": hi[1],
            "reliable": np.isfinite(p) and p <= ALPHA}


def metrics_by_model(d, cfg, items=None):
    """The three rates per model, plus micro and the per-class parts behind macro."""
    spec = {m["model_id"]: m for m in cfg["ladder"] + cfg["reference"]}
    dd = M.prepare(d)
    if items is not None:
        keep = dd["run_id"].astype(str).str.split("|").str[0].isin(set(items))
        dd = dd[keep]
    out = {}
    for mid, meta in spec.items():
        s = dd[dd["model"].astype(str).eq(mid)]
        if s.empty:
            continue
        corr, clean = s[s["is_corrupted"]], s[~s["is_corrupted"]]
        el = s[s["localization_eligible"]]
        per = {}
        for m in MODALITIES:
            g = el[el["true_outlier"].eq(m)]
            per[m] = (float(g["localization_correct"].mean()) if len(g)
                      else float("nan"))
        vals = [per[m] for m in MODALITIES if np.isfinite(per[m])]
        out[mid] = {
            "short": meta["short"], "release": meta["release"],
            "role": "ladder" if any(x["model_id"] == mid for x in cfg["ladder"])
                    else "singleton",
            FLAG_C: float(corr["detected"].mean()) if len(corr) else float("nan"),
            FLAG_K: float(clean["detected"].mean()) if len(clean) else float("nan"),
            "loc_micro": (float(el["localization_correct"].mean()) if len(el)
                          else float("nan")),
            "loc_macro": float(np.mean(vals)) if vals else float("nan"),
            "per_class": per,
            "n_items": int(s["run_id"].astype(str).str.split("|").str[0].nunique()),
            "n_solvers": int(s["solver_id"].nunique()),
            "n_flagged": len(el),
        }
    return out


def _x(res, mids):
    return [pd.Timestamp(res[m]["release"]).toordinal() for m in mids]


def _draw(ax, res, order, key, colour, label, c):
    """Ladder connected, singletons as free markers with their names beside them."""
    lad = [m for m in order if m in res and res[m]["role"] == "ladder"]
    sing = [m for m in order if m in res and res[m]["role"] == "singleton"]
    if lad:
        ax.plot(_x(res, lad), [100 * res[m][key] for m in lad], color=colour,
                linewidth=1.6, marker="o", markersize=5, zorder=4, label=label)
    for m in sing:
        ax.scatter([_x(res, [m])[0]], [100 * res[m][key]], s=34, color=colour,
                   marker="D", zorder=4,
                   label=None)
    return lad, sing


def fig_newer(res, order):
    """Two stacked panels on a shared release-date axis. No fitted line anywhere.

    A regression line through eight points, four of which are unrelated singletons,
    would assert a rate of change the design cannot support. The rank correlation is
    printed instead, with its p and its leave-one-out range.
    """
    style.apply(style.theme())
    c = style.colors()
    mids = [m for m in order if m in res]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=style.figsize(1.35, 4.6))
    if not mids:
        style.empty_axes(ax1, "no models")
        style.empty_axes(ax2, "no models")
        return fig

    lad = [m for m in mids if res[m]["role"] == "ladder"]
    # Shade the gap between the two rates -- that band IS the discriminative part.
    # Only across the ladder, where the points are connected and the band means
    # something; between unrelated singletons it would imply a path that is not there.
    if len(lad) > 1:
        ax1.fill_between(_x(res, lad),
                         [100 * res[m][FLAG_C] for m in lad],
                         [100 * res[m][FLAG_K] for m in lad],
                         color=c["bar2"], alpha=0.30, zorder=1,
                         label="gap = discriminative part")
    for m in [x for x in mids if res[x]["role"] == "singleton"]:
        ax1.plot([_x(res, [m])[0]] * 2,
                 [100 * res[m][FLAG_K], 100 * res[m][FLAG_C]],
                 color=c["muted"], linewidth=1.0, alpha=0.55, zorder=2)
    _draw(ax1, res, order, FLAG_C, C_FLAGGED, "flagged | something corrupted", c)
    _draw(ax1, res, order, FLAG_K, C_CLEAN, "flagged | nothing corrupted", c)
    _draw(ax2, res, order, "loc_macro", C_MACRO,
          "named the right view (macro over 4 classes)", c)

    t_c, t_k = trend(res, order, FLAG_C), trend(res, order, FLAG_K)
    t_m = trend(res, order, "loc_macro")
    ax1.annotate(
        f"flagged | corrupted:  rho={t_c['rho']:+.2f}, p={t_c['p']:.3f}\n"
        f"flagged | clean:       rho={t_k['rho']:+.2f}, p={t_k['p']:.3f}",
        (0.015, 0.03), xycoords="axes fraction", va="bottom",
        fontsize=style.ANNOT_PT, color=c["muted"])
    ax2.annotate(f"rho={t_m['rho']:+.2f}, p={t_m['p']:.3f}",
                 (0.015, 0.03), xycoords="axes fraction", va="bottom",
                 fontsize=style.ANNOT_PT, color=c["muted"])

    for ax in (ax1, ax2):
        ax.grid(True, axis="y", linewidth=0.4, color=c["faint"])
        ax.set_axisbelow(True)
        ax.set_ylim(0, 105)
    ax1.set_ylabel("% flagged")
    ax2.set_ylabel("% right view")
    # One legend for the whole figure, below it. Per-axes legends landed in the
    # upper left of each panel, which is where the singleton labels and the QwQ
    # marker already are -- the legend printed straight over them.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, frameon=False,
               fontsize=style.TICK_PT - 1, handlelength=1.4,
               bbox_to_anchor=(0.5, -0.055))

    # Singletons carry their names on the plot; the ladder is named on the axis,
    # because four connected points in date order read as a series and labelling
    # each one repeats what the line already says.
    for ax, key in ((ax1, FLAG_C), (ax2, "loc_macro")):
        for m in mids:
            if res[m]["role"] != "singleton":
                continue
            # Alternating above/below, with an opaque bbox. Every singleton label
            # sat 7pt above its marker before, which put four of them on top of the
            # ladder line and one underneath the legend.
            up = _LABEL_ABOVE.get(res[m]["short"], True)
            ax.annotate(res[m]["short"], (_x(res, [m])[0], 100 * res[m][key]),
                        xytext=(0, 9 if up else -9), textcoords="offset points",
                        ha="center", va="bottom" if up else "top",
                        fontsize=style.TICK_PT - 2, color=c["muted"],
                        bbox=dict(facecolor=c["bg"], edgecolor="none",
                                  boxstyle="square,pad=0.12", alpha=0.85))
    ax2.set_xticks(_x(res, mids))
    ax2.set_xticklabels([str(res[m]["release"])[:7] for m in mids],
                        rotation=35, ha="right", fontsize=style.TICK_PT - 1)
    ax2.set_xlabel("release date")
    fig.tight_layout(h_pad=1.0, rect=(0, 0.035, 1, 1))
    return fig


def verdict(res, order):
    """One line per panel. p > ALPHA is stated as no reliable trend, never as gain."""
    out = []
    for key, label in ((FLAG_C, "flagging when something IS corrupted"),
                       (FLAG_K, "flagging when nothing is corrupted"),
                       ("loc_macro", "naming the right view (macro)")):
        t = trend(res, order, key)
        first = res[order[0]][key] if order[0] in res else float("nan")
        last = res[order[-1]][key] if order[-1] in res else float("nan")
        span = (f"{100 * first:.0f}% at the oldest model to {100 * last:.0f}% at "
                f"the newest")
        if not t["reliable"]:
            body = (f"<b>No reliable trend</b> in {label}: Spearman rho="
                    f"{t['rho']:+.2f} against release order, p={t['p']:.3f} "
                    f"(n={t['n']}), which does not clear {ALPHA}. The rates run "
                    f"{span}, but that ordering is not distinguishable from chance "
                    f"here.")
            cls = "v-inconclusive"
        else:
            body = (f"{label.capitalize()} <b>rises with release order</b>: rho="
                    f"{t['rho']:+.2f}, p={t['p']:.3f} (n={t['n']}); {span}.")
            cls = "v-supported"
        body += (f" Leave-one-out rho spans {t['loo_lo']:+.2f} to "
                 f"{t['loo_hi']:+.2f} &mdash; dropping {_h.escape(t['loo_lo_drop'])} "
                 f"gives the low end, dropping {_h.escape(t['loo_hi_drop'])} the "
                 f"high.")
        out.append((cls, body))
    return out


def details_table(res, order):
    head = ("<tr><th>model</th><th>released</th><th>role</th>"
            "<th>flagged | corrupted</th><th>flagged | clean</th>"
            "<th>right view, micro</th><th>right view, macro</th>"
            "<th>macro &minus; micro</th><th>items</th><th>solvers</th></tr>")
    body = []
    for m in order:
        if m not in res:
            continue
        r = res[m]
        gap = r["loc_macro"] - r["loc_micro"]
        body.append(
            f"<tr><td>{_h.escape(r['short'])}</td><td>{r['release']}</td>"
            f"<td>{r['role']}</td>"
            f"<td>{100 * r[FLAG_C]:.1f}%</td><td>{100 * r[FLAG_K]:.1f}%</td>"
            f"<td>{100 * r['loc_micro']:.1f}%</td>"
            f"<td><b>{100 * r['loc_macro']:.1f}%</b></td>"
            f"<td>{100 * gap:+.1f} pp</td>"
            f"<td>{r['n_items']:,}</td><td>{r['n_solvers']}</td></tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


def per_class_table(res, order):
    head = ("<tr><th>model</th>"
            + "".join(f"<th>{MODALITY_LABELS[m]} was broken</th>"
                      for m in MODALITIES)
            + "<th>macro</th></tr>")
    body = []
    for m in order:
        if m not in res:
            continue
        r = res[m]
        cells = "".join(
            "<td>&mdash;</td>" if not np.isfinite(r["per_class"][k]) else
            f"<td>{100 * r['per_class'][k]:.0f}%</td>" for k in MODALITIES)
        body.append(f"<tr><td>{_h.escape(r['short'])}</td>{cells}"
                    f"<td><b>{100 * r['loc_macro']:.0f}%</b></td></tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


def trend_table(res, order):
    head = ("<tr><th>metric</th><th>rho</th><th>p</th><th>reads as</th>"
            "<th>leave-one-out rho</th></tr>")
    body = []
    for key, label in ((FLAG_C, "flagged | corrupted"),
                       (FLAG_K, "flagged | clean"),
                       ("loc_micro", "right view, micro"),
                       ("loc_macro", "right view, macro")):
        t = trend(res, order, key)
        reads = ("rises with release order" if t["reliable"]
                 else "<b>no reliable trend</b>")
        body.append(
            f"<tr><td>{label}</td><td>{t['rho']:+.3f}</td><td>{t['p']:.4f}</td>"
            f"<td>{reads}</td>"
            f"<td>{t['loo_lo']:+.3f} to {t['loo_hi']:+.3f} "
            f"<span style='color:var(--dim)'>(drop "
            f"{_h.escape(t['loo_lo_drop'])} / {_h.escape(t['loo_hi_drop'])})</span>"
            f"</td></tr>")
    return "<table class='tbl'>" + head + "".join(body) + "</table>"


def build_section(d, cfg, items=None, provisional=None):
    """The section. `provisional` is a dict of MEASURED coverage, never a literal."""
    res = metrics_by_model(d, cfg, items=items)
    order = sorted(res, key=lambda m: pd.Timestamp(res[m]["release"]))
    if not order:
        return ""
    svg = _svg(fig_newer(res, order))
    verdicts = "".join(f'<div class="verdict {cls}">{body}</div>'
                       for cls, body in verdict(res, order))

    # The caveat is COMPUTED. A hand-written coverage line goes stale the moment an
    # arm finishes, and a stale caveat in a report is worse than none: it tells the
    # reader the evidence is thinner than it is, in the report's own voice.
    n_items = min(r["n_items"] for r in res.values())
    n_solv = min(r["n_solvers"] for r in res.values())
    incomplete = (provisional or {}).get("incomplete") or []
    prov = ("" if not incomplete else
            " Arms still generating at build time: "
            + ", ".join(_h.escape(x) for x in incomplete)
            + " &mdash; those rows are <b>provisional</b>.")
    worst = max(res.values(), key=lambda r: r["loc_micro"] - r["loc_macro"])

    return (
        '<section id="newer-better">'
        '<div class="warnbanner">Exploratory. Rank correlation over '
        f'<b>{len(order)} models</b>, of which only four are an ordered family; the '
        'other four are singletons from different labs and have a date without '
        'having a position in any series. At n=8 a single point moves rho a long '
        'way, so every correlation below is published with its leave-one-out range. '
        f'Coverage: {n_items:,} items and {n_solv} solver systems per model.'
        f'{prov}</div>'
        '<h2>Are newer models better at this?</h2>'
        f'{verdicts}'
        '<p class="sub" style="margin-bottom:14px">Three rates, no d&prime; and no '
        'z-transform &mdash; percentage points end to end. <b>Flagged | '
        'corrupted</b> is how often a model says the views disagree when one of '
        'them really was broken; <b>flagged | clean</b> is how often it says so '
        'when nothing was. The two are plotted together because neither means '
        'anything alone: a model can raise the first simply by raising the second, '
        'and the shaded band between them is the part that is actually '
        'discriminative.</p>'
        '<p class="sub" style="margin-bottom:14px"><b>The lower panel uses the '
        'MACRO average</b> &mdash; the unweighted mean over the four true-outlier '
        'classes &mdash; not the item-pooled micro version used elsewhere in this '
        'report. The design gives trajectory four of the seven corrupted '
        'conditions and the other three views one each, so micro scores a model '
        'four times as heavily on the class it may simply be defaulting to. The gap '
        'is real: '
        f'<b>{_h.escape(worst["short"])}</b> reads '
        f'{100 * worst["loc_micro"]:.1f}% micro against '
        f'{100 * worst["loc_macro"]:.1f}% macro, because it localizes '
        f'{MODALITY_LABELS["T"]} at '
        f'{100 * worst["per_class"]["T"]:.0f}% and code at '
        f'{100 * worst["per_class"]["C"]:.0f}%. Micro is kept beside macro in the '
        'table below rather than dropped, because the distance between them is '
        'itself the diagnostic.</p>'
        f'<figure>{svg}</figure>'
        '<p class="sub"><b>Caption.</b> Ladder models are connected; the four '
        'singletons are unconnected diamonds, labelled, because they have a release '
        'date but no position in a series. <b>No line is fitted.</b> A regression '
        'through eight points, half of them unrelated, would assert a rate of '
        'change this design cannot measure. The rank correlation against release '
        'order is printed in each panel instead.</p>'
        '<details><summary class="sub">Details &mdash; per-model rates, '
        'per-class localization, and every rho with its leave-one-out range'
        '</summary>'
        f'<div style="overflow-x:auto">{details_table(res, order)}</div>'
        '<p class="sub" style="margin-top:18px">The four class scores behind each '
        'macro average. A model whose row is high on trajectory and low on code is '
        'the case macro exists to catch.</p>'
        f'<div style="overflow-x:auto">{per_class_table(res, order)}</div>'
        '<p class="sub" style="margin-top:18px"><b>Leave-one-out.</b> Each rho '
        'recomputed with one model dropped, over all eight drops. The range is '
        'wide by construction at this n &mdash; it is reported so the headline rho '
        'is read as one draw from that range rather than as a point estimate.</p>'
        f'<div style="overflow-x:auto">{trend_table(res, order)}</div>'
        '</details></section>')
