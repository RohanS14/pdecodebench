"""stability.py — do individual judgements change when identifiers are obfuscated?

The Q4 figures above it report RATES. Two conditions can post identical accuracy and
still disagree on every single item: a model that gets 60% right under real names and
60% right under obfuscated ones may be getting the same 60% right, or a disjoint 60%.
The aggregate cannot tell those apart, and they mean opposite things. This module
pairs the items and looks at what moved.

WHAT IS PAIRED. Each pair is one item under real names against the same item under
obfuscated ones: same solver system, same corruption condition, same presentation
order, same model. Order is included in the key even though the spec asked only for
(solver, condition, model), because the design generates TWO orders per cell and the
real/obfuscated versions of a cell share an ordering (verified in the design
appendix). Pairing without it would silently average two orders together on each
side and attribute any order effect to the naming manipulation.

THE NOISE FLOOR IS THE POINT. An instability number alone is unreadable: at
temperature 0.6 a model's answer moves between draws for no reason at all. So the
identical decomposition is computed between two draws under ONE naming condition,
which is how much the judgement moves from resampling alone, and the manipulation is
reported against it.

Because that comparison is the whole argument, it is made at MATCHED aggregation.
The spec's headline uses the modal answer over three draws per side, and a modal of
three is less noisy than a single draw -- so comparing modal-vs-modal against
draw-vs-draw would credit the manipulation with a stability it gets from averaging,
and bias the read toward "names changed nothing". Three bars are therefore drawn:
the modal comparison the spec asked for, the same comparison at single-draw
aggregation, and the floor. The generated verdict uses the two that match.
"""
import collections
import html as _h

import numpy as np
import pandas as pd

from . import style
from .constants import (MODALITIES, MODALITY_LABELS, NONE, OUTLIER_COLORS,
                        NONE_COLOR)

import matplotlib.pyplot as plt

# Categories, in the order they are drawn and summed. Mutually exclusive and
# exhaustive over the pair count -- asserted in decompose(), not hoped for.
SAME_CORRECT = "same_correct"
SAME_WRONG = "same_wrong"
FLIP_TO_CORRECT = "flip_to_correct"
FLIP_TO_WRONG = "flip_to_wrong"
WRONG_TO_WRONG = "wrong_to_wrong"
COMMIT_ASYM = "commit_asymmetry"
CATEGORIES = (SAME_CORRECT, SAME_WRONG, FLIP_TO_CORRECT, FLIP_TO_WRONG,
              WRONG_TO_WRONG, COMMIT_ASYM)
# Stable under the manipulation. Everything else is movement.
STABLE = (SAME_CORRECT, SAME_WRONG)

CAT_LABELS = {
    SAME_CORRECT: "same view, correct both times",
    SAME_WRONG: "same view, wrong both times",
    FLIP_TO_CORRECT: "wrong → correct",
    FLIP_TO_WRONG: "correct → wrong",
    WRONG_TO_WRONG: "blamed a different view, wrong both times",
    COMMIT_ASYM: "named a view under one naming, said all four agree under the other",
}
# The verdict level is a BINARY answer, so "same view" is the wrong noun for it.
# Same categories, same colours, same order -- only the words change, because a
# legend that says "view" on a figure about flag/no-flag misnames what it labels.
CAT_LABELS_VERDICT = {
    SAME_CORRECT: "same verdict, correct both times",
    SAME_WRONG: "same verdict, wrong both times",
    FLIP_TO_CORRECT: "wrong → correct",
    FLIP_TO_WRONG: "correct → wrong",
    WRONG_TO_WRONG: "(unreachable at this level)",
    COMMIT_ASYM: "(unreachable at this level)",
}
CAT_COLORS = {
    SAME_CORRECT: "#009E73",
    SAME_WRONG: "#8C8C8C",
    FLIP_TO_CORRECT: "#0072B2",
    FLIP_TO_WRONG: "#D55E00",
    WRONG_TO_WRONG: "#CC79A7",
    COMMIT_ASYM: "#E69F00",
}

REAL, OBF = "real", "obfuscated"
N_BOOT = 2000
BOOT_SEED = 20260824


# ── building the paired frame ────────────────────────────────────────────────

def tidy(d, raw):
    """One row per draw, carrying the pairing key and both answer levels.

    `d` is the adapter frame (which already resolved each row's slot answer into the
    view it actually accused, through that row's own permutation) and `raw` is the
    frame it was built from, which is where item_id and sample_idx live. They are
    positionally aligned by construction; this asserts it rather than trusting it.
    """
    if len(d) != len(raw):
        raise ValueError(f"frames not aligned: {len(d)} vs {len(raw)}")
    key = raw["item_id"].astype(str).str.split("|", expand=True)
    if key.shape[1] != 4:
        raise ValueError("item_id is not the 4-field design key")
    return pd.DataFrame({
        "model": d["model"].astype(str).values,
        "solver_id": d["solver_id"].astype(str).values,
        "system": key[0].values,
        "condition": key[1].values,
        "naming": key[2].values,
        "order": key[3].values,
        "sample_idx": pd.to_numeric(raw["sample_idx"], errors="coerce").values,
        "true_outlier": d["true_outlier"].astype(str).values,
        "pred_outlier": d["pred_outlier"].astype(str).values,
        "pred_agree": d["pred_agree"].astype(str).values,
    })


def _modal(values):
    """The majority answer, or "" when there is no answer to take a majority of.

    A tie among three draws is broken by first appearance, which is the draw order
    the eval wrote. Ties are reported by the caller rather than hidden: a pair whose
    modal rests on a 1-1-1 split is a weaker observation than one that rests on 3-0.
    """
    vals = [v for v in values if v]
    if not vals:
        return ""
    c = collections.Counter(vals)
    top = max(c.values())
    for v in vals:                      # first-appearance tie-break
        if c[v] == top:
            return v
    return ""


def _flagged(pred_agree, pred_outlier):
    """Did the model say the four views disagree? "" when it gave no verdict."""
    a = str(pred_agree).strip().lower()
    if a in ("no", "false", "disagree"):
        return "flag"
    if a in ("yes", "true", "agree"):
        return "noflag"
    # Fall back to the outlier field only when the verdict field is unreadable, so a
    # row is never dropped for a missing verdict it effectively gave.
    if pred_outlier in MODALITIES:
        return "flag"
    if pred_outlier == NONE:
        return "noflag"
    return ""


def _answers(g, level, how, draw_a=0, draw_b=1):
    """{(key): answer} for one naming condition, at one aggregation.

    how="modal": the majority over that condition's draws.
    how="draw":  a single named draw, so the floor and the manipulation can be
                 compared without one of them being quietly averaged.
    """
    out = {}
    for k, sub in g.groupby(["model", "system", "condition", "order"], sort=False):
        if level == "outlier":
            vals = list(sub["pred_outlier"])
            vals = [v if (v in MODALITIES or v == NONE) else "" for v in vals]
        else:
            vals = [_flagged(a, p) for a, p in
                    zip(sub["pred_agree"], sub["pred_outlier"])]
        if how == "modal":
            out[k] = (_modal(vals), sub["true_outlier"].iloc[0],
                      sub["solver_id"].iloc[0])
        else:
            idx = list(sub["sample_idx"])
            pick = ""
            for want in (draw_a,):
                for v, i in zip(vals, idx):
                    if i == want:
                        pick = v
                        break
            out[k] = (pick, sub["true_outlier"].iloc[0], sub["solver_id"].iloc[0])
    return out


def _correct(ans, true_outlier, level):
    if level == "outlier":
        return ans == true_outlier
    return (ans == "flag") if true_outlier != NONE else (ans == "noflag")


def _committed(ans, level):
    """Named an actual view (outlier level) / raised the alarm (verdict level)."""
    return ans in MODALITIES if level == "outlier" else ans == "flag"


def classify(a, b, true_outlier, level):
    """One pair -> one category. `a` is the reference side, `b` the compared side.

    COMMIT_ASYMMETRY is an OUTLIER-level category only. At the verdict level the
    answer IS the flag, so "committed under one naming and not the other" is not a
    separate phenomenon from "the verdict changed" -- routing it to its own bucket
    there would swallow every flip and leave the flip categories permanently empty,
    which is the opposite of informative. At the verdict level a difference always
    means one side right and one side wrong, so it resolves to a flip and both
    COMMIT_ASYMMETRY and WRONG_TO_WRONG are unreachable by construction.
    """
    ca, cb = _correct(a, true_outlier, level), _correct(b, true_outlier, level)
    if level == "outlier" and _committed(a, level) != _committed(b, level):
        return COMMIT_ASYM
    if a == b:
        return SAME_CORRECT if ca else SAME_WRONG
    if ca and not cb:
        return FLIP_TO_WRONG
    if cb and not ca:
        return FLIP_TO_CORRECT
    return WRONG_TO_WRONG


def decompose(t, level, how, side_b=OBF, draw_b=1, models=None):
    """Pair every item and count the categories.

    side_b=OBF pairs real against obfuscated -- the manipulation.
    side_b=REAL pairs draw `draw_b` against draw 0 within the REAL condition only --
    the noise floor. That branch never reads an obfuscated row: the floor has to be
    movement from resampling alone, and a floor computed across namings would have
    the manipulation baked into it.
    """
    if models is not None:
        t = t[t["model"].isin(models)]
    real = t[t["naming"].eq(REAL)]
    if side_b == OBF:
        A = _answers(real, level, how)
        B = _answers(t[t["naming"].eq(OBF)], level, how)
    else:
        A = _answers(real, level, "draw", draw_a=0)
        B = _answers(real, level, "draw", draw_a=draw_b)

    rows, dropped = [], 0
    for k, (a, true_o, solver) in A.items():
        if k not in B:
            continue
        b, _t2, _s2 = B[k]
        if not a or not b:
            dropped += 1
            continue
        rows.append({"model": k[0], "solver_id": solver, "condition": k[2],
                     "true_outlier": true_o,
                     "category": classify(a, b, true_o, level)})
    fr = pd.DataFrame(rows)
    counts = collections.Counter(fr["category"]) if len(fr) else collections.Counter()
    # Mutually exclusive and exhaustive. Asserted, because a category added later
    # without a branch here would silently vanish from a bar that still reads 100%.
    assert sum(counts.values()) == len(fr), (sum(counts.values()), len(fr))
    assert set(counts) <= set(CATEGORIES), set(counts) - set(CATEGORIES)
    return {"pairs": fr, "counts": dict(counts), "n": len(fr), "dropped": dropped}


def instability(counts, n):
    """1 - same_correct - same_wrong: the share of pairs whose answer moved."""
    if not n:
        return np.nan
    return 1.0 - sum(counts.get(c, 0) for c in STABLE) / n


def _boot_diff(pairs_a, pairs_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """CI for (instability_a - instability_b), resampling SOLVER SYSTEMS.

    Solvers, not pairs: there are 32 independent physical systems behind these
    pairs, and resampling pairs would report an interval far too narrow for the
    number of systems actually observed. Both sides are resampled with the SAME
    solver draw, because they are measured on the same systems and the quantity of
    interest is their difference.
    """
    if not len(pairs_a) or not len(pairs_b):
        return (np.nan, np.nan)
    solvers = sorted(set(pairs_a["solver_id"]) | set(pairs_b["solver_id"]))
    idx = {s: i for i, s in enumerate(solvers)}

    def vec(fr):
        mov = np.zeros(len(solvers))
        tot = np.zeros(len(solvers))
        for s, c in zip(fr["solver_id"], fr["category"]):
            i = idx[s]
            tot[i] += 1
            mov[i] += int(c not in STABLE)
        return mov, tot

    ma, ta = vec(pairs_a)
    mb, tb = vec(pairs_b)
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, len(solvers), size=(n_boot, len(solvers)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ia = np.where(ta[bi].sum(1) > 0, ma[bi].sum(1) / ta[bi].sum(1), np.nan)
        ib = np.where(tb[bi].sum(1) > 0, mb[bi].sum(1) / tb[bi].sum(1), np.nan)
    d = ia - ib
    d = d[np.isfinite(d)]
    if not len(d):
        return (np.nan, np.nan)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


# ── the three bars ───────────────────────────────────────────────────────────

def bars(t, level, models=None):
    """The three comparisons the figure draws, in drawing order."""
    return [
        ("obfuscated vs real\n(modal of 3 draws)",
         decompose(t, level, "modal", side_b=OBF, models=models)),
        ("obfuscated vs real\n(single draw — matched to the floor)",
         decompose(t, level, "draw", side_b=OBF, models=models)),
        ("real vs real, draw 1 vs draw 2\n(sampling noise floor)",
         decompose(t, level, "draw", side_b=REAL, draw_b=1, models=models)),
    ]


def fig_stability(rows, title, level="outlier"):
    """Stacked horizontal bars, one per comparison, shared categories and order."""
    labels = CAT_LABELS if level == "outlier" else CAT_LABELS_VERDICT
    style.apply(style.theme())
    c = style.colors()
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.46 * len(rows) + 1.55))
    ypos = [len(rows) - i for i in range(len(rows))]

    seen, share_max = set(), {}
    for yi, (label, res) in zip(ypos, rows):
        n = res["n"]
        if not n:
            continue
        left = 0.0
        for cat in CATEGORIES:
            w = res["counts"].get(cat, 0) / n
            if w <= 0:
                continue
            seen.add(cat)
            share_max[cat] = max(share_max.get(cat, 0.0), w)
            ax.barh(yi, w, left=left, height=0.55, color=CAT_COLORS[cat],
                    edgecolor=c["panel"], linewidth=1.4)
            if w >= 0.055:
                ax.text(left + w / 2, yi, f"{100 * w:.0f}%", ha="center",
                        va="center", fontsize=style.ANNOT_PT, color=c["bg"],
                        zorder=5)
            left += w
        # The stable/unstable boundary, marked once per bar. It is the number the
        # verdict quotes, and without the rule a reader has to add segments by eye.
        stable = sum(res["counts"].get(k, 0) for k in STABLE) / n
        ax.plot([stable, stable], [yi - 0.30, yi + 0.30], color=c["fg"],
                linewidth=1.6, zorder=6)
        ax.annotate(f"{(1 - stable) * 100:.0f}% moved", xy=(1.005, yi),
                    va="center", ha="left", fontsize=8, color=c["fg"],
                    annotation_clip=False)

    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v * 100:.0f}%")
    ax.set_xlabel("share of paired items", fontsize=9)
    ax.set_title(title, fontsize=10.5, pad=10)
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          color=CAT_COLORS[k], label=labels[k])
               for k in CATEGORIES if k in seen]
    handles.append(plt.Line2D([], [], color=c["fg"], linewidth=1.6,
                              label="vertical rule = boundary between stable and moved"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
               ncol=1, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.30, right=0.90, top=0.84, bottom=0.20)
    return fig


# ── verdict and tables ───────────────────────────────────────────────────────

def verdict(rows):
    """Generated. Leads with the comparison against the floor, at matched
    aggregation, and picks its clause from the bootstrap interval."""
    if len(rows) < 3:
        return ""
    _l1, modal = rows[0]
    _l2, single = rows[1]
    _l3, floor = rows[2]
    x_modal = instability(modal["counts"], modal["n"])
    x = instability(single["counts"], single["n"])
    y = instability(floor["counts"], floor["n"])
    lo, hi = _boot_diff(single["pairs"], floor["pairs"])
    if not np.isfinite(lo):
        clause = "the interval on the difference could not be computed"
    elif lo <= 0 <= hi:
        clause = ("names move judgements no more than sampling noise does "
                  f"(difference {100 * (x - y):+.1f}pp, 95% CI "
                  f"{100 * lo:+.1f} to {100 * hi:+.1f}pp, which spans zero)")
    elif lo > 0:
        clause = ("judgements are partly driven by identifiers rather than by the "
                  f"represented physics (difference {100 * (x - y):+.1f}pp, 95% CI "
                  f"{100 * lo:+.1f} to {100 * hi:+.1f}pp)")
    else:
        clause = ("obfuscation moves judgements LESS than resampling does "
                  f"(difference {100 * (x - y):+.1f}pp, 95% CI {100 * lo:+.1f} to "
                  f"{100 * hi:+.1f}pp), which is not a result about naming")
    return (f"Obfuscating identifiers changes the named outlier on "
            f"{100 * x:.0f}% of items, against {100 * y:.0f}% from resampling "
            f"alone. {clause[0].upper()}{clause[1:]}. "
            f"At the modal-of-three aggregation the manipulation moves "
            f"{100 * x_modal:.0f}% of items, which is the more stable estimate but "
            f"is not comparable to the floor.")


def counts_table(rows, level="outlier"):
    labels = CAT_LABELS if level == "outlier" else CAT_LABELS_VERDICT
    head = ("<tr><th>Comparison</th>"
            + "".join(f"<th>{labels[c]}</th>" for c in CATEGORIES)
            + "<th>pairs</th><th>dropped</th><th>moved</th></tr>")
    body = ""
    for label, res in rows:
        n = res["n"]
        body += (f"<tr><td>{_h.escape(label.replace(chr(10), ' '))}</td>"
                 + "".join(
                     f"<td>{res['counts'].get(c, 0):,}"
                     + (f" <span style='color:var(--dim)'>({100 * res['counts'].get(c, 0) / n:.0f}%)</span>"
                        if n else "") + "</td>" for c in CATEGORIES)
                 + f"<td>{n:,}</td><td>{res['dropped']:,}</td>"
                 + f"<td><b>{100 * instability(res['counts'], n):.0f}%</b></td></tr>")
    return ("<div style='overflow-x:auto'><table><thead>" + head
            + "</thead><tbody>" + body + "</tbody></table></div>")


def by_model_table(t, level, models):
    head = ("<tr><th>Model</th><th>obfuscation moved<br>(single draw)</th>"
            "<th>noise floor</th><th>difference</th>"
            f"<th>{CAT_LABELS[WRONG_TO_WRONG]}</th><th>pairs</th></tr>")
    body = ""
    for mid, short in models:
        a = decompose(t, level, "draw", side_b=OBF, models=[mid])
        b = decompose(t, level, "draw", side_b=REAL, draw_b=1, models=[mid])
        if not a["n"] or not b["n"]:
            continue
        x, y = instability(a["counts"], a["n"]), instability(b["counts"], b["n"])
        w2w = a["counts"].get(WRONG_TO_WRONG, 0)
        body += (f"<tr><td>{_h.escape(short)}</td>"
                 f"<td>{100 * x:.0f}%</td><td>{100 * y:.0f}%</td>"
                 f"<td><b>{100 * (x - y):+.0f}pp</b></td>"
                 f"<td>{w2w:,} <span style='color:var(--dim)'>"
                 f"({100 * w2w / a['n']:.0f}%)</span></td>"
                 f"<td>{a['n']:,}</td></tr>")
    return ("<div style='overflow-x:auto'><table><thead>" + head
            + "</thead><tbody>" + body + "</tbody></table></div>")


# ── directional blame: where the mass moved, not how much churned ────────────

BLAME_LEVELS = MODALITIES + (NONE,)
BLAME_LABELS = {**MODALITY_LABELS, NONE: "named none"}
VERDICT_LEVELS = ("flag", "noflag")
VERDICT_LABELS = {"flag": "flagged a disagreement",
                  "noflag": "said all four agree"}
VERDICT_COLORS = {"flag": "#D55E00", "noflag": "#0072B2"}


def paired_answers(t, level="outlier", how="modal", draw_a=0, models=None):
    """Every item that has a readable answer under BOTH namings.

    Returns (list of (key, solver, true_outlier, real_answer, obf_answer), dropped).
    One list, built once, so the shares, the churn rate and the transition table are
    all computed over the SAME pairs and cannot disagree about the denominator.
    """
    if models is not None:
        t = t[t["model"].isin(models)]
    A = _answers(t[t["naming"].eq(REAL)], level, how, draw_a=draw_a)
    B = _answers(t[t["naming"].eq(OBF)], level, how, draw_a=draw_a)
    out, dropped = [], 0
    for k, (a, true_o, solver) in A.items():
        if k not in B:
            continue
        b = B[k][0]
        if not a or not b:
            dropped += 1
            continue
        out.append((k, solver, true_o, a, b))
    return out, dropped


def shares(pairs, levels):
    """Blame share per level under each naming. Asserts the shares close to 1."""
    n = len(pairs)
    real = {v: 0 for v in levels}
    obf = {v: 0 for v in levels}
    for _k, _s, _t, a, b in pairs:
        if a in real:
            real[a] += 1
        if b in obf:
            obf[b] += 1
    if n:
        # Every answer must land in a named level. A level added to the vocabulary
        # without being added here would silently shrink both distributions while
        # the figure still read as a share of 100%.
        assert sum(real.values()) == n, (sum(real.values()), n)
        assert sum(obf.values()) == n, (sum(obf.values()), n)
    f = lambda d: {v: (d[v] / n if n else np.nan) for v in levels}
    return {"n": n, "real_n": real, "obf_n": obf,
            "real": f(real), "obf": f(obf)}


def share_cis(pairs, levels, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap CI on (obf share - real share) per level, over SOLVER SYSTEMS.

    Solvers rather than pairs: 32 independent physical systems sit behind these
    items, and resampling pairs would report an interval far too narrow for the
    number of systems actually observed.
    """
    solvers = sorted({s for _k, s, _t, _a, _b in pairs})
    if not solvers:
        return {v: (np.nan, np.nan) for v in levels}
    idx = {s: i for i, s in enumerate(solvers)}
    L_ = len(levels)
    ra = np.zeros((len(solvers), L_))
    ob = np.zeros((len(solvers), L_))
    tot = np.zeros(len(solvers))
    col = {v: j for j, v in enumerate(levels)}
    for _k, s, _t, a, b in pairs:
        i = idx[s]
        tot[i] += 1
        if a in col:
            ra[i, col[a]] += 1
        if b in col:
            ob[i, col[b]] += 1
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, len(solvers), size=(n_boot, len(solvers)))
    T = tot[bi].sum(axis=1)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for v, j in col.items():
            # ob/ra are (solvers, levels); take the column FIRST, then resample.
            # ob[bi, :, j] indexes three axes into a two-axis array.
            d = np.where(T > 0,
                         (ob[:, j][bi].sum(1) - ra[:, j][bi].sum(1)) / T, np.nan)
            d = d[np.isfinite(d)]
            out[v] = ((float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
                      if len(d) else (np.nan, np.nan))
    return out


def churn(pairs):
    """Share of paired items whose answer differs between namings."""
    if not pairs:
        return np.nan
    return sum(1 for _k, _s, _t, a, b in pairs if a != b) / len(pairs)


def truth_shares(pairs, levels):
    """Share of the paired items where each level was the RIGHT answer.

    Without this the blame figure is unreadable: "named none on 29% of items" means
    nothing until you know that only 12.5% of items were clean. The design fixes
    these proportions -- trajectory carries four of the eight conditions and every
    other answer carries one -- so this is a property of the item construction, not
    of any model, and it is the reference every blame share has to be read against.
    """
    n = len(pairs)
    c = {v: 0 for v in levels}
    for _k, _s, true_o, _a, _b in pairs:
        key = true_o if true_o in c else (NONE if NONE in c else None)
        if key is not None:
            c[key] += 1
    return {v: (c[v] / n if n else np.nan) for v in levels}


def direction_rows(pairs, levels, n_boot=N_BOOT):
    """One row per level: shares, signed delta, CI, significance. Sorted by |delta|."""
    sh = shares(pairs, levels)
    ci = share_cis(pairs, levels, n_boot=n_boot)
    truth = truth_shares(pairs, levels)
    rows = []
    for v in levels:
        lo, hi = ci[v]
        d = sh["obf"][v] - sh["real"][v]
        rows.append({
            "level": v, "real": sh["real"][v], "obf": sh["obf"][v],
            "real_n": sh["real_n"][v], "obf_n": sh["obf_n"][v],
            "delta": d, "lo": lo, "hi": hi, "truth": truth[v],
            "sig": bool(np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi)),
        })
    rows.sort(key=lambda r: -abs(r["delta"]) if np.isfinite(r["delta"]) else 0.0)
    return rows, sh["n"]


def fig_blame_direction(rows, title, colors, labels, xlabel="blame share"):
    """Dot-and-arrow, one row per level. Direction of travel is the whole figure."""
    style.apply(style.theme())
    c = style.colors()
    fig, ax = plt.subplots(figsize=style.figsize(1.0, 0.50 * len(rows) + 1.35))
    ypos = [len(rows) - i for i in range(len(rows))]

    # Padding has to hold the two value labels, which are drawn OUTSIDE their dots.
    # At 0.14 of the data range the rightmost label ran into the difference column
    # and the leftmost ran into the row names. The labels are the constraint here,
    # not the data, so the padding is sized for them.
    vals = [v for r in rows for v in (r["real"], r["obf"], r.get("truth"))
            if v is not None and np.isfinite(v)]
    lo_x, hi_x = (min(vals), max(vals)) if vals else (0.0, 1.0)
    pad = max((hi_x - lo_x) * 0.30, 0.06)
    lo_x, hi_x = max(0.0, lo_x - pad), min(1.02, hi_x + pad)

    for yi, r in zip(ypos, rows):
        col = colors.get(r["level"], NONE_COLOR)
        a, b = r["real"], r["obf"]
        # Ground truth FIRST, behind the dots: it is the reference the two shares are
        # read against, not a third measurement.
        tv = r.get("truth")
        if tv is not None and np.isfinite(tv):
            ax.plot([tv, tv], [yi - 0.26, yi + 0.26], color=c["fg"], linewidth=1.5,
                    alpha=0.75, zorder=2)

        if np.isfinite(a) and np.isfinite(b) and abs(b - a) > 1e-9:
            ax.annotate("", xy=(b, yi), xytext=(a, yi),
                        arrowprops=dict(arrowstyle="-|>", color=col, linewidth=1.6,
                                        shrinkA=6, shrinkB=7, alpha=0.85))
        ax.plot([a], [yi], "o", color=col, markersize=8, zorder=3)
        # facecolor "none", not the page background: on a row whose shift is under a
        # pixel the two dots coincide, and a background-filled ring would hide the
        # filled dot entirely, reading as "only one condition was measured". A true
        # open ring lets the filled dot show through, so a zero-length arrow looks
        # like what it is -- two coincident values.
        ax.plot([b], [yi], "o", markerfacecolor="none", markeredgecolor=col,
                markeredgewidth=2.0, markersize=9, zorder=4)

        far = np.isfinite(a) and np.isfinite(b) and abs(b - a) >= 0.02
        if far:
            # Labels on the OUTSIDE of each dot, so the arrow between them is never
            # overwritten by its own endpoints' text.
            ax.annotate(f"{a * 100:.1f}%", xy=(a, yi),
                        xytext=(-13 if a <= b else 13, 0), textcoords="offset points",
                        ha="right" if a <= b else "left", va="center", fontsize=8,
                        color=c["muted"])
            ax.annotate(f"{b * 100:.1f}%", xy=(b, yi),
                        xytext=(13 if a <= b else -13, 0), textcoords="offset points",
                        ha="left" if a <= b else "right", va="center", fontsize=8,
                        color=col, fontweight="bold")
        else:
            # Too close to label separately without the two texts colliding with
            # each other and with the row name. One combined label, to the right of
            # the pair, which is also the only side guaranteed to be clear.
            # ABOVE the pair, not to either side. Left runs into the row name and
            # right runs into the ground-truth rule, which on these rows sits within
            # a few points of the dots -- that is precisely why they are close.
            # Above is clear on every row, and the truth label sits below.
            ax.annotate(f"{a * 100:.1f}% \u2192 {b * 100:.1f}%",
                        xy=((a + b) / 2, yi), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=col, fontweight="bold")
        # Right-hand difference column. Greyed and marked n.s. when the interval
        # covers zero, so an unresolved shift can never be read as a finding.
        txt = (f"{r['delta'] * 100:+.1f}pp "
               f"({r['lo'] * 100:+.1f} to {r['hi'] * 100:+.1f})"
               if np.isfinite(r["lo"]) else f"{r['delta'] * 100:+.1f}pp")
        if not r["sig"]:
            txt += "  n.s."
        ax.annotate(txt, xy=(1.04, yi), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=8,
                    color=c["fg"] if r["sig"] else c["muted"],
                    fontweight="bold" if r["sig"] else "normal",
                    annotation_clip=False)

    ax.set_yticks(ypos)
    # The ground-truth share rides in the ROW NAME rather than floating next to its
    # rule. As a floating label it collided with the neighbouring row's value label
    # on every close pair -- and those are exactly the rows where the rule sits
    # nearest the dots. Here it cannot collide with anything and it reads as what it
    # is: a property of the row, not a third measurement.
    ax.set_yticklabels(
        [f"{labels.get(r['level'], r['level'])}\n"
         + (f"({r['truth'] * 100:.0f}% of items)"
            if r.get("truth") is not None and np.isfinite(r["truth"]) else "")
         for r in rows], fontsize=9)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v * 100:.0f}%")
    ax.set_xlabel(f"{xlabel} (%)", fontsize=9)
    ax.set_title(title, fontsize=10.5, pad=10)
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   color=c["muted"], label="real identifiers"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=c["bg"], markeredgecolor=c["muted"],
                   markeredgewidth=2.0, color="none", label="obfuscated identifiers"),
        plt.Line2D([], [], color=c["fg"], linewidth=1.5, alpha=0.75,
                   label="| = where the correct answer rate sits"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.44, -0.05))
    fig.subplots_adjust(left=0.23, right=0.60, top=0.86, bottom=0.26)
    return fig


def _named(rows, level_labels):
    """The rows whose interval excludes zero, largest movement first."""
    return [r for r in rows if r["sig"]]


def churn_line(rows, churn_rate, labels):
    """The one-sentence churn read-out. Leads with a resolved row if one exists."""
    sig = _named(rows, labels)
    if sig:
        r = sig[0]
        way = "off" if r["delta"] < 0 else "onto"
        return (f"Blame moves {way} {labels[r['level']]} by "
                f"{abs(r['delta']) * 100:.1f}pp "
                f"({r['lo'] * 100:+.1f} to {r['hi'] * 100:+.1f}), while "
                f"{churn_rate * 100:.0f}% of individual items changed which view "
                f"was blamed. The distribution barely moves; the judgements "
                f"underneath it do.")
    mx = max((abs(r["delta"]) for r in rows if np.isfinite(r["delta"])), default=0.0)
    return (f"Blame shifts by at most {mx * 100:.1f}pp in any direction, but "
            f"{churn_rate * 100:.0f}% of individual items changed which view was "
            f"blamed. The distribution is stable while the judgements underneath "
            f"it are not.")


def direction_verdict(rows, churn_rate, labels):
    """Directional verdict. Never presents an unresolved shift as a finding."""
    sig = _named(rows, labels)
    top = [r for r in rows if np.isfinite(r["delta"])][:2]
    if not sig:
        clause = ("in no direction that this sample size resolves &mdash; the "
                  "largest shifts are "
                  + " and ".join(
                      f"{'off' if r['delta'] < 0 else 'onto'} {labels[r['level']]} "
                      f"({r['delta'] * 100:+.1f}pp, CI {r['lo'] * 100:+.1f} to "
                      f"{r['hi'] * 100:+.1f})" for r in top)
                  + ", and both intervals span zero")
        tail = ("No single view&rsquo;s shift is resolved at this sample size, yet "
                f"{churn_rate * 100:.0f}% of individual items changed which view was "
                "blamed &mdash; the population distribution is stable while "
                "individual judgements are not.")
    else:
        clause = " and ".join(
            f"{'off' if r['delta'] < 0 else 'onto'} {labels[r['level']]} "
            f"({r['delta'] * 100:+.1f}pp)" for r in sig[:2])
        tail = (f"{churn_rate * 100:.0f}% of individual items changed which view was "
                "blamed, so the shift in the distribution understates how much "
                "individual judgements moved.")
    return f"Obfuscating identifiers moves blame {clause}. {tail}"


def share_table(rows, labels, n):
    head = ("<tr><th>View blamed</th><th>really was the answer</th>"
            "<th>real names</th><th>obfuscated</th>"
            "<th>difference (95% CI)</th></tr>")
    body = ""
    for r in rows:
        sig = r["sig"]
        d = (f"{r['delta'] * 100:+.1f}pp "
             f"<span style='color:var(--dim)'>({r['lo'] * 100:+.1f} to "
             f"{r['hi'] * 100:+.1f})</span>" if np.isfinite(r["lo"])
             else f"{r['delta'] * 100:+.1f}pp")
        if not sig:
            d += " <span style='color:var(--dim)'>n.s.</span>"
        tv = r.get("truth")
        body += (f"<tr><td>{_h.escape(labels.get(r['level'], r['level']))}</td>"
                 + (f"<td>{tv * 100:.1f}%</td>" if tv is not None
                    and np.isfinite(tv) else "<td>&mdash;</td>")
                 + f"<td>{r['real'] * 100:.1f}% "
                 f"<span style='color:var(--dim)'>({r['real_n']:,}/{n:,})</span></td>"
                 f"<td>{r['obf'] * 100:.1f}% "
                 f"<span style='color:var(--dim)'>({r['obf_n']:,}/{n:,})</span></td>"
                 f"<td>{'<b>' if sig else ''}{d}{'</b>' if sig else ''}</td></tr>")
    return ("<div style='overflow-x:auto'><table><thead>" + head
            + "</thead><tbody>" + body + "</tbody></table></div>")


def per_model_share_table(t, models, n_boot=400):
    head = ("<tr><th>Model</th>"
            + "".join(f"<th>{BLAME_LABELS[v]}</th>" for v in BLAME_LEVELS)
            + "<th>churn</th><th>pairs</th></tr>")
    body = ""
    for mid, short in models:
        pairs, _drop = paired_answers(t, "outlier", "modal", models=[mid])
        if not pairs:
            continue
        rows, n = direction_rows(pairs, BLAME_LEVELS, n_boot=n_boot)
        by = {r["level"]: r for r in rows}
        body += (f"<tr><td>{_h.escape(short)}</td>"
                 + "".join(
                     f"<td>{by[v]['delta'] * 100:+.1f}pp"
                     + ("" if by[v]["sig"] else
                        " <span style='color:var(--dim)'>n.s.</span>") + "</td>"
                     for v in BLAME_LEVELS)
                 + f"<td>{churn(pairs) * 100:.0f}%</td><td>{n:,}</td></tr>")
    return ("<div style='overflow-x:auto'><table><thead>" + head
            + "</thead><tbody>" + body + "</tbody></table></div>"
            "<p class='sub' style='margin-top:8px'>Each cell is the signed shift in "
            "that view&rsquo;s blame share, obfuscated minus real. Per-model cells "
            "are an eighth of the pooled n, so intervals are wide and n.s. is the "
            "expected reading almost everywhere; the column to watch is churn.</p>")


def _svg(fig):
    from .claim_report import _svg as s
    return s(fig)


def build_block(d, raw, models):
    """The whole subsection as HTML, for splicing into the Q4 answer.

    The section's question is DIRECTIONAL -- does blame move toward a particular
    view -- so the main figures answer that and nothing else. The transition
    decomposition that used to be plotted answers a magnitude question; it is still
    computed, and now lives in the details block as a table, where a reader who
    wants "how much churned" can find it without the main flow implying that churn
    was the question.
    """
    t = tidy(d, raw)
    pairs, dropped = paired_answers(t, "outlier", "modal")
    if not pairs:
        return ""
    rows, n = direction_rows(pairs, BLAME_LEVELS)
    ch = churn(pairs)

    vpairs, vdropped = paired_answers(t, "verdict", "modal")
    vrows, vn = direction_rows(vpairs, VERDICT_LEVELS)
    vch = churn(vpairs)
    # The verdict panel earns its place only if it is directional. With two
    # complementary levels the two deltas are equal and opposite by construction,
    # so it is worth drawing when something actually moved and worth omitting when
    # nothing did -- a two-row figure of two n.s. zeros is a distraction.
    v_worth = bool(vrows) and (any(r["sig"] for r in vrows)
                               or max(abs(r["delta"]) for r in vrows) >= 0.01)

    single, single_drop = paired_answers(t, "outlier", "draw", draw_a=0)
    ch_single = churn(single)

    # The transition decomposition, still computed, now tabulated rather than drawn.
    trans = [("obfuscated vs real (modal of 3 draws)",
              decompose(t, "outlier", "modal", side_b=OBF)),
             ("obfuscated vs real (single draw)",
              decompose(t, "outlier", "draw", side_b=OBF)),
             ("real vs real, draw 1 vs draw 2 (sampling noise floor)",
              decompose(t, "outlier", "draw", side_b=REAL, draw_b=1))]

    verdict_panel = ""
    if v_worth:
        verdict_panel = (
            '<h4 class="subfig">And whether it flagged anything at all</h4>'
            f'<figure>{_svg(fig_blame_direction(vrows, "Verdict share under each naming", VERDICT_COLORS, VERDICT_LABELS, xlabel="share of paired items"))}</figure>'
            '<p class="figcap">The same encoding on the agree/disagree verdict. The '
            'two rows are complementary, so their shifts are equal and opposite by '
            f'construction. {vch * 100:.0f}% of individual items changed verdict '
            'between namings.</p>')

    return (
        '<h4 class="subfig">When identifiers are obfuscated, does blame move toward '
        'a particular view?</h4>'
        '<p class="sub">The figures above report accuracy under each naming. This '
        'one reports <b>where the blame went</b>. Each row is one answer the model '
        'could give; the filled dot is its share under real identifiers, the hollow '
        'dot its share under obfuscated ones, and the arrow is the direction of '
        'travel. Shares are over all paired items and sum to 100% within each '
        'naming condition.</p>'
        f'<figure>{_svg(fig_blame_direction(rows, "Blame share under each naming", OUTLIER_COLORS, BLAME_LABELS))}</figure>'
        f'<p class="sub">{churn_line(rows, ch, BLAME_LABELS)}</p>'
        '<p class="figcap"><b>Caption.</b> Each item is paired against itself: same '
        'solver system, same corruption, same presentation order, same model, real '
        'names against obfuscated ones. The answer is the modal one across that '
        f'condition&rsquo;s three draws. {n:,} pairs; {dropped:,} dropped for having '
        'no readable answer on one side. Intervals resample the 32 solver systems, '
        'not the items. <b>&ldquo;named none&rdquo; is a row here</b> because moving '
        'from naming a view to declining to name one is itself a directional change, '
        'and excluding it would let that movement leave the figure without being '
        'counted anywhere.</p>'
        f'<p class="sub"><b>Verdict.</b> {direction_verdict(rows, ch, BLAME_LABELS)}</p>'
        + verdict_panel +
        '<details><summary class="sub">Details &mdash; shares and counts, churn at '
        'both aggregations, the transition breakdown, and the per-model table'
        '</summary>'
        '<p class="sub" style="margin-top:10px"><b>Blame share per view, with '
        'counts.</b></p>'
        f'{share_table(rows, BLAME_LABELS, n)}'
        '<p class="sub" style="margin-top:14px"><b>Verdict share.</b></p>'
        f'{share_table(vrows, VERDICT_LABELS, vn)}'
        '<p class="sub" style="margin-top:14px"><b>Churn, at both aggregations.</b> '
        f'Modal of three draws: <b>{ch * 100:.1f}%</b> of {n:,} pairs changed which '
        f'view was blamed. Single draw: <b>{ch_single * 100:.1f}%</b> of '
        f'{len(single):,} pairs. The two are <b>not interchangeable</b> &mdash; a '
        'modal answer over three draws is less noisy than one draw, so the modal '
        'figure is the smaller of the two by construction and not because the '
        'manipulation is weaker there. The headline number above uses the '
        '<b>modal</b> aggregation, matching the shares in the figure.</p>'
        '<p class="sub" style="margin-top:14px"><b>Transition breakdown.</b> Where '
        'the churn went, by category. The third row is the sampling-noise floor: '
        'the same decomposition between two draws under one naming, which is how '
        'much the judgement moves from resampling alone. Read the first two rows '
        'against it, and note that the floor is at single-draw aggregation, so only '
        'the second row is like-for-like with it.</p>'
        f'{counts_table(trans)}'
        f'<p class="sub" style="margin-top:14px"><b>{trans[1][1]["counts"].get(WRONG_TO_WRONG, 0):,} pairs '
        'blamed a different view and were wrong both times.</b> Nothing about the '
        'model&rsquo;s accuracy changed on those items &mdash; renaming the '
        'variables changed only <i>which</i> representation it held responsible, on '
        'an item where the physics, the equations, the prose and the numbers were '
        'all untouched.</p>'
        '<p class="sub" style="margin-top:14px"><b>By model.</b> The eight '
        'checkpoints differ enough in how readily they flag anything that pooling '
        'can hide two models shifting in opposite directions.</p>'
        f'{per_model_share_table(t, models)}'
        '</details>')
