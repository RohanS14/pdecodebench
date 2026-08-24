"""Computed claims: every verdict line in the report is generated from here.

The rule this module exists to enforce is that no sentence in the report asserts a
direction the interval does not support. A verdict is a dataclass with a `direction`
that is set to "inconclusive" whenever the CI crosses the null, and the renderer has
no way to phrase it otherwise -- there is no hand-written prose path.

Differences of proportions use Newcombe's hybrid score interval, built from the two
Wilson intervals. Deterministic (no bootstrap), correct near 0 and 1, and it cannot
produce a bound outside [-1, 1] the way a Wald interval on a rare cell does.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .constants import (CONDITIONS, MODALITIES, MODALITY_LABELS, NAMING_LEVELS,
                        NONE, OUTLIER_LEVELS)

MIN_N = 20          # cells thinner than this are shown but never claimed on


@dataclass
class Verdict:
    question: str
    phrase: str                     # for the summary table
    sentence: str                   # the verdict line
    direction: str                  # "supported" | "inconclusive" | "unmeasured"
    n: int = 0
    detail: str = ""
    rows: list = field(default_factory=list)     # (label, value) for the details table


def newcombe_diff(k1, n1, k2, n2):
    """CI for p1 - p2 from two Wilson intervals. Returns (diff, lo, hi)."""
    if not n1 or not n2:
        return (float("nan"),) * 3
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = M.wilson_ci(k1, n1)
    l2, u2 = M.wilson_ci(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def crosses_null(lo, hi, null=0.0):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return True
    return lo <= null <= hi


def pct(x):
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}%"


def pp(x):
    return "n/a" if not np.isfinite(x) else f"{100 * x:+.1f}pp"


# ── baselines, computed from the actual condition frequencies ────────────────
def baselines(df):
    """What a model with no ability would score, given THIS design's mix.

    Computed rather than assumed: the clean/corrupted split is 1:7 here, so
    "always disagree" scores far better than the 50% a balanced design would give
    it, and a figure without that line invites reading 0.62 as competence.
    """
    d = M.prepare(df)
    n = len(d)
    if not n:
        return {}
    p_clean = float((~d["is_corrupted"]).mean())
    p_corrupt = 1.0 - p_clean

    # Most common corrupted modality, for the always-disagree strategy.
    corrupted = d[d["is_corrupted"]]
    if len(corrupted):
        top = corrupted["true_outlier"].value_counts().idxmax()
        p_top_given_corrupt = float((corrupted["true_outlier"] == top).mean())
    else:
        top, p_top_given_corrupt = "", 0.0

    k = len(OUTLIER_LEVELS)                      # 5 possible answers
    return {
        "always-agree": dict(
            detection=p_clean, localization=float("nan"),
            label=f"always agree ({pct(p_clean)})"),
        "always-disagree": dict(
            detection=p_corrupt, localization=p_top_given_corrupt,
            label=(f"always disagree, blame {MODALITY_LABELS.get(top, top)} "
                   f"({pct(p_corrupt)})")),
        "uniform-random": dict(
            # Picks one of the five answers uniformly: right on a clean item only by
            # answering "none", right on a corrupted one only by naming its view.
            detection=(1 / k) * p_clean + (1 - 1 / k) * p_corrupt,
            localization=1 / len(MODALITIES),
            label=f"uniform over {k} answers"),
    }


# ── Q1: detection ────────────────────────────────────────────────────────────
def q1(df):
    d = M.prepare(df)
    corrupted, clean = d[d["is_corrupted"]], d[~d["is_corrupted"]]
    hit_k, hit_n = int(corrupted["detected"].sum()), len(corrupted)
    fa_k, fa_n = int(clean["detected"].sum()), len(clean)
    diff, lo, hi = newcombe_diff(hit_k, hit_n, fa_k, fa_n)
    inconclusive = crosses_null(lo, hi)
    hit, fa = hit_k / hit_n if hit_n else float("nan"), fa_k / fa_n if fa_n else float("nan")

    if inconclusive:
        sentence = (f"Inconclusive. The model flags a corrupted item {pct(hit)} of the "
                    f"time and a clean one {pct(fa)} of the time; the difference is "
                    f"{pp(diff)} ({pp(lo)} to {pp(hi)}), which includes zero.")
        phrase = "inconclusive"
    else:
        sentence = (f"Yes, but weakly. The model flags a corrupted item {pct(hit)} of "
                    f"the time against {pct(fa)} for a clean one — a difference of "
                    f"{pp(diff)} ({pp(lo)} to {pp(hi)} CI), on n={hit_n:,} corrupted "
                    f"and n={fa_n:,} clean.")
        phrase = f"discriminates, {pp(diff)}"
    return Verdict("Q1", phrase, sentence,
                   "inconclusive" if inconclusive else "supported", n=len(d),
                   rows=[("flags a corrupted item", f"{pct(hit)}  (n={hit_n:,})"),
                         ("flags a clean item", f"{pct(fa)}  (n={fa_n:,})"),
                         ("difference", f"{pp(diff)}  [{pp(lo)}, {pp(hi)}]")])


# ── Q2: localization given detection ─────────────────────────────────────────
# ── Q3: which view is blamed when the answer is wrong ────────────────────────
def blame_contrasts(df):
    """Pairwise differences in false-blame rate between modalities."""
    d = M.prepare(df)
    stats = {}
    for m in MODALITIES:
        den = d[d["true_outlier"].ne(m)]
        stats[m] = (int(den["pred_outlier"].eq(m).sum()), len(den))
    out = []
    for i, a in enumerate(MODALITIES):
        for b in MODALITIES[i + 1:]:
            ka, na = stats[a]
            kb, nb = stats[b]
            diff, lo, hi = newcombe_diff(ka, na, kb, nb)
            out.append(dict(a=a, b=b, label=f"{MODALITY_LABELS[a]} − {MODALITY_LABELS[b]}",
                            diff=diff, lo=lo, hi=hi, na=na, nb=nb,
                            thin=min(na, nb) < MIN_N))
    return out, stats


def q3(df):
    contrasts, stats = blame_contrasts(df)
    usable = [c for c in contrasts if not c["thin"] and np.isfinite(c["diff"])]
    if not usable:
        return Verdict("Q3", "no data", "No modality pair has enough observations.",
                       "unmeasured")
    top = max(usable, key=lambda c: abs(c["diff"]))
    sig = [c for c in usable if not crosses_null(c["lo"], c["hi"])]
    rates = {m: (k / n if n else float("nan")) for m, (k, n) in stats.items()}
    most = max(rates, key=lambda m: rates[m])
    if not sig:
        sentence = (f"Inconclusive. The most blamed view is "
                    f"{MODALITY_LABELS[most]} at {pct(rates[most])}, but no pairwise "
                    f"difference excludes zero; the largest is {top['label']} at "
                    f"{pp(top['diff'])} ({pp(top['lo'])} to {pp(top['hi'])}).")
        phrase = "no clear preference"
        direction = "inconclusive"
    else:
        big = max(sig, key=lambda c: abs(c["diff"]))
        sentence = (f"The model blames {MODALITY_LABELS[most]} most, at "
                    f"{pct(rates[most])} of the items where it is NOT the corrupted "
                    f"view. The largest supported gap is {big['label']} at "
                    f"{pp(big['diff'])} ({pp(big['lo'])} to {pp(big['hi'])} CI); "
                    f"{len(sig)} of {len(usable)} pairwise contrasts exclude zero.")
        phrase = f"prefers {MODALITY_LABELS[most]}"
        direction = "supported"
    rows = [(f"blamed when not the corrupted view — {MODALITY_LABELS[m]}",
             f"{pct(rates[m])}  (n={stats[m][1]:,})") for m in MODALITIES]
    rows += [(f"difference: {c['label']}",
              f"{pp(c['diff'])} [{pp(c['lo'])}, {pp(c['hi'])}]"
              + ("  (thin cell)" if c["thin"] else ""))
             for c in contrasts]
    return Verdict("Q3", phrase, sentence, direction,
                   n=sum(n for _, n in stats.values()), rows=rows)


# ── Q4: is that preference lexical? ──────────────────────────────────────────
def q4(df, stats=None):
    """Generated from the PRIMARY test and the INTERACTION only.

    Hard rule: no number reaches this sentence unless its interval excludes zero.
    The previous version quoted "trajectory -4.3pp" from a row the same figure drew
    as not significant.
    """
    from .figures import fig7_prior_weakening, interaction_statement
    import matplotlib.pyplot as _plt
    r = stats
    if r is None:
        fig, r, _ = fig7_prior_weakening(df)
        _plt.close(fig)
    p = r.primary
    if p is None or not np.isfinite(p.diff):
        return Verdict("Q4", "no data", "No paired observations.", "unmeasured")

    lead = (f"Given that the model committed to a verdict, obfuscating identifiers "
            f"changes its chance of naming the right outlier by "
            f"{pp(p.diff)} ({pp(p.lo)} to {pp(p.hi)}, n={p.n_solvers} solvers)")
    if not p.significant:
        lead += " — an interval that includes zero, so no change is established"
    lead += "."

    _, resolved = interaction_statement(r)
    if resolved:
        it = r.interaction
        lead += (f" The effect is not symmetric: obfuscation moves correctness "
                 f"{pp(it.real)} where the code is innocent and {pp(it.obf)} where "
                 f"the code is the culprit, a difference of {pp(it.diff)} "
                 f"({pp(it.lo)} to {pp(it.hi)}).")
        phrase = f"asymmetric, {pp(it.diff)}"
    else:
        lead += " The per-corruption pattern is not resolved at this sample size."
        phrase = ("no change established" if not p.significant
                  else f"primary {pp(p.diff)}")
    direction = "supported" if (p.significant or resolved) else "inconclusive"
    return Verdict("Q4", phrase, lead, direction, n=p.n_solvers, rows=_q4_rows(r))


def _q4_rows(r):
    from .constants import MODALITY_LABELS as ML

    def row(c, tag=""):
        return (f"{c.name}{tag}",
                f"{pct(c.real)} -> {pct(c.obf)}   {pp(c.diff)} "
                f"[{pp(c.lo)}, {pp(c.hi)}]   n={c.n_solvers}"
                + (f"   MDE(80%)={pp(c.mde)}" if np.isfinite(c.mde) else "")
                + ("" if c.significant else ("  not tested" if c.exploratory
                                             else "  n.s.")))

    out = [("delta convention", "obfuscated - real, paired within solver"),
           ("pairing key", "solver_id, condition, reasoning, model"),
           ("bootstrap resamples (of solvers)", f"{r.n_boot:,}"),
           ("unpaired cells dropped", f"{r.n_pairs_dropped}")]
    out.append(("PRIMARY (pre-specified)", ""))
    out.append(row(r.primary))
    out.append(("mechanism", ""))
    out.append(row(r.detection))
    out.append(row(r.specificity))
    out.append(("identity check: detection x conditional = overall accuracy",
                f"max abs error {r.identity_max_err:.2e}"))
    out.append(("EXPLORATORY per corruption — not tested, no claim made", ""))
    for c in r.per_outlier:
        out.append(row(c, tag=f"  ({ML.get(c.name, c.name)} corrupted)"))
    out.append(("pre-specified asymmetry contrast", ""))
    out.append(row(r.interaction))
    out.append(("blame share (appendix only)",
                "where blame went, NOT whether it was correct — a shift between two "
                "wrong views moves this and leaves correctness unchanged, so it is "
                "not used to answer this question"))
    return out


# ── Q5: right for the right reason ───────────────────────────────────────────
def q5(df):
    if "judge_correct" not in df.columns or not df["judge_correct"].notna().any():
        return Verdict(
            "Q5", "not measured",
            "Not measured. Answering this requires knowing whether the model's stated "
            "justification names the real defect, which means an LLM-judge pass over "
            "the justification column. No such pass has been run, so there is no "
            "judge_correct field and no number to report here.",
            "unmeasured", n=0,
            detail="Produced by: an LLM-judge pass over the justification column.")
    d = M.prepare(df)
    elig = d[d["localization_eligible"]]
    right = elig[elig["localization_correct"]]
    k, n = int(right["judge_correct"].sum()), len(right)
    rate = k / n if n else float("nan")
    lo, hi = M.wilson_ci(k, n)
    return Verdict("Q5", f"{pct(rate)} justified",
                   f"When the model names the right view, its stated reason matches the "
                   f"real defect {pct(rate)} of the time ({pct(lo)}-{pct(hi)} CI, "
                   f"n={n:,}).", "supported", n=n)




# ── Q1 (new): is some corruption simply easier to see? ───────────────────────
def q_sensitivity(df):
    """Leads with the false-alarm floor, not the ordering.

    The floor is the number that makes every other number readable: a 60% flag rate
    means nothing until you know the model flags 38% of items where nothing is wrong.
    The ordering claim is downgraded whenever the corruptions are not severity-matched,
    because then the ranking may be reporting how each corruption was generated.
    """
    from .sensitivity import detection_sensitivity, severity_tiers
    r = detection_sensitivity(df)
    table, matched, common = severity_tiers(df)
    usable = [x for x in r.rows if not x["empty"] and not x["thin"]]
    if len(usable) < 2:
        return Verdict("Q1", "no data",
                       "Too few corrupted items to compare.", "unmeasured")
    usable.sort(key=lambda x: -x["hit_rate"])
    best, worst = usable[0], usable[-1]
    lead = (f"The model reports a disagreement on {pct(r.fa_rate)} of items where all "
            f"four representations agree. Corruption raises that to between "
            f"{pct(worst['hit_rate'])} and {pct(best['hit_rate'])}")
    gloss = (" d\u2032 is detection sensitivity: how far a corruption's flag rate sits "
             "above that floor.")

    if not matched:
        # Banner active: no cross-representation ordering may be asserted.
        lifts = ", ".join(
            f"{x['label']} {pp(x['hit_rate'] - r.fa_rate)}" for x in usable)
        sentence = (lead + ". Lift over that floor, per corruption: " + lifts +
                    ". The corruptions are NOT severity-matched across "
                    "representations, so these rows are not ranked against each "
                    "other: the ordering could reflect how each corruption was "
                    "generated rather than what the model trusts." + gloss)
        phrase = f"floor {pct(r.fa_rate)}; ordering not claimed"
        direction = "inconclusive"
    else:
        resolved = best["lo"] > worst["hi"]
        if resolved:
            sentence = (lead + f", with {best['label']} easiest to detect and "
                        f"{worst['label']} hardest; those intervals are disjoint."
                        + gloss)
            phrase = f"floor {pct(r.fa_rate)}; {best['label']} easiest"
            direction = "supported"
        else:
            sentence = (lead + f", with {best['label']} highest and "
                        f"{worst['label']} lowest — but those intervals overlap, so "
                        f"the ordering is not resolved." + gloss)
            phrase = f"floor {pct(r.fa_rate)}; ordering not resolved"
            direction = "inconclusive"

    rows = [(f"{x['label']}: flagged / items",
             f"{x['n_hit']}/{x['n_signal']} = {pct(x['hit_rate'])}   "
             f"d\u2032={x['dprime']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
             + ("   (thin)" if x["thin"] else "")) for x in r.rows if not x["empty"]]
    rows.append(("baseline — nothing corrupted",
                 f"{r.n_fa}/{r.n_noise} = {pct(r.fa_rate)}"))
    rows.append(("correction",
                 "log-linear (Hautus): 0.5 added to each count, 1 to each total, "
                 "applied to COUNTS not rates"))
    if table is not None and len(table):
        rows.append(("severity tiers (corruption-generation method)",
                     "matched across representations" if matched
                     else "NOT matched — see counts below"))
        for m in table.index:
            per = {t: int(table.loc[m, t]) for t in table.columns
                   if int(table.loc[m, t]) > 0}
            rows.append((f"  {MODALITY_LABELS[m]}",
                         ", ".join(f"{t}: {n}" for t, n in per.items())))
        rows.append(("severity tier common to all four",
                     ", ".join(common) if common else "none — no matched figure"))
    return Verdict("Q1", phrase, sentence, direction,
                   n=sum(x["n_signal"] for x in r.rows), rows=rows)


# ── Q2 (new): does blame carry information about the true outlier? ───────────
def q_blame_information(df):
    from .sensitivity import blame_information
    b = blame_information(df)
    if b.n == 0:
        return Verdict("Q2", "no data",
                       "No item was both corrupted and flagged.", "unmeasured")
    gloss = ("The comparison is against the best FIXED answer: always naming one "
             "view, whichever is most often the corrupted one. A stereotyped responder "
             "reduces exactly to that, so clearing it is the bar.")
    claim = ("blame tracks the actual outlier" if b.informative
             else "blame is indistinguishable from a fixed response bias")
    sentence = (f"The model names the corrupted view on {pct(b.localization)} of the "
                f"{b.n:,} flagged items, against {pct(b.best_constant)} for the best "
                f"fixed answer — a margin of "
                f"{pp(b.localization - b.best_constant)} "
                f"({pp(b.margin_lo)} to {pp(b.margin_hi)}, bootstrapped over solvers) "
                f"— {claim}. {gloss}")
    rows = [("names the corrupted view", pct(b.localization)),
            ("best fixed answer", pct(b.best_constant)),
            ("margin", f"{pp(b.localization - b.best_constant)} "
                       f"[{pp(b.margin_lo)}, {pp(b.margin_hi)}]"),
            ("Cramer's V", f"{b.cramers_v:.3f}"),
            ("flagged items analysed", f"{b.n:,}")]
    rows += [(f"  from condition {k}", f"{v:,}")
             for k, v in sorted(b.composition.items(), key=lambda kv: -kv[1])]
    # The pooled blame distribution used to be drawn as a reference row. It is still
    # the thing each row should be compared against, so it stays here as numbers.
    if b.marginal is not None:
        rows.append(("pooled blame distribution (all flagged items)",
                     ", ".join(f"{MODALITY_LABELS.get(k, k)} {100 * v:.1f}%"
                               for k, v in b.marginal.items() if v > 0)))
    return Verdict("Q2", claim if b.informative else "uninformative", sentence,
                   "supported" if b.informative else "inconclusive", n=b.n, rows=rows)


QUESTIONS = [
    # One detection section, not two. "Can it tell at all" and "is some corruption
    # easier to spot" are answered by the same figure -- per-corruption flag rate
    # against the clean-item floor -- so they were duplicating each other.
    ("q1", "Can the model tell when the representations disagree at all?",
     q_sensitivity),
    ("q2", "When it says something disagrees, does it know which thing?",
     q_blame_information),
    # The old "does it pick the right outlier?" section is gone: it asked the same
    # thing as Q2 and answered it with a weaker statistic (per-condition accuracy
    # with no baseline). Q2's details block carries the accuracy number.
    ("q3", "Which representation does it trust when it gets the outlier wrong?", q3),
    ("q4", "Is that trust driven by physics or by lexical cues?", q4),
    ("q5", "When it's right, is it right for the right reason?", q5),
]


def all_verdicts(df, shared=None):
    """`shared` carries precomputed stats objects so a section's verdict and its
    figure are generated from the same numbers rather than two separate passes."""
    shared = shared or {}
    out = []
    for qid, title, fn in QUESTIONS:
        if qid == "q4" and "obfuscation" in shared:
            out.append((qid, title, fn(df, stats=shared["obfuscation"])))
        else:
            out.append((qid, title, fn(df)))
    return out
