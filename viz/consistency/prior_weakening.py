"""Does stripping identifiers weaken a lexical prior, and does that help correctness?

Replaces a blame-SHARE analysis, which could not answer the question it was under:
share says where blame went, not whether it went to the right place. A model that
moves blame from one wrong view to another wrong view has a large share change and
zero change in correctness.

PRIMARY OUTCOME is conditional localization -- given the model committed to a
verdict, was that verdict right. That isolates the prior-weakening question from the
separate, already-established fact that obfuscation makes the model decline more
often; overall accuracy multiplies the two together and cannot distinguish them.

One pre-specified primary test, one pre-specified interaction. The per-corruption
breakdown is exploratory and labelled as such: with 32 solvers it is underpowered,
and four more n.s. markers would be multiplicity theatre rather than evidence.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import MODALITIES, NAMING_LEVELS, NONE

PAIR_KEY = ["solver_id", "condition", "reasoning", "model"]
N_BOOT = 10_000
SEED = 20260820
Z_ALPHA, Z_POWER = 1.959963984540054, 0.8416212335729143   # two-sided 95%, 80% power


@dataclass
class Contrast:
    name: str = ""
    real: float = float("nan")
    obf: float = float("nan")
    diff: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n_solvers: int = 0
    n_items: int = 0            # size of THIS row's denominator, pooled
    denom: str = ""             # what that denominator is, in words
    sd: float = float("nan")
    mde: float = float("nan")
    exploratory: bool = False
    conditional: object = None      # per-representation companion, when relevant
    @property
    def significant(self):
        return (not self.exploratory and np.isfinite(self.lo)
                and not (self.lo <= 0.0 <= self.hi))


@dataclass
class PriorWeakening:
    overall: Contrast = None        # the direct answer: right view / all broken items
    primary: Contrast = None        # the same thing, conditioned on committing
    detection: Contrast = None
    specificity: Contrast = None
    per_outlier: list = field(default_factory=list)
    interaction: Contrast = None
    n_pairs_dropped: int = 0
    n_boot: int = N_BOOT
    identity_max_err: float = float("nan")


def _matched(d):
    """Cells present under BOTH naming conditions. Returns (frame, dropped_count)."""
    seen = d.groupby(PAIR_KEY)["naming"].nunique()
    complete = seen[seen == len(NAMING_LEVELS)].index
    dropped = int((seen != len(NAMING_LEVELS)).sum())
    if not len(complete):
        return d.iloc[0:0], dropped
    return d.set_index(PAIR_KEY).loc[complete].reset_index(), dropped


def _solver_outcome(sub, kind):
    """Outcome per (solver, naming). NaN where the denominator is empty."""
    corrupted = sub["true_outlier"].ne(NONE)
    committed = sub["pred_agree"].eq("no")
    hit = sub["pred_outlier"].eq(sub["true_outlier"])
    if kind == "detection":
        num, den = committed & corrupted, corrupted
    elif kind == "cond_localization":
        num, den = hit & committed & corrupted, committed & corrupted
    elif kind == "overall_accuracy":
        num, den = hit & committed & corrupted, corrupted
    elif kind == "specificity":
        num, den = sub["pred_agree"].eq("yes") & ~corrupted, ~corrupted
    else:
        raise ValueError(kind)
    t = sub.assign(_n=num.astype(float), _d=den.astype(float))
    g = t.groupby(["solver_id", "naming"])[["_n", "_d"]].sum()
    return (g["_n"] / g["_d"].where(g["_d"] > 0, np.nan)).unstack("naming")


def _contrast(tab, name, n_boot, seed, exploratory=False,
              n_items=0, denom=""):
    """Paired obfuscated-minus-real, bootstrapped over solvers."""
    c = Contrast(name=name, exploratory=exploratory, n_items=n_items,
                 denom=denom)
    if tab is None or tab.empty or not set(NAMING_LEVELS) <= set(tab.columns):
        return c
    t = tab.dropna()
    if t.empty:
        return c
    diff = (t[NAMING_LEVELS[1]] - t[NAMING_LEVELS[0]]).to_numpy()
    c.n_solvers = len(diff)
    c.real = float(t[NAMING_LEVELS[0]].mean())
    c.obf = float(t[NAMING_LEVELS[1]].mean())
    c.diff = float(diff.mean())
    c.sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    if np.isfinite(c.sd) and len(diff):
        # Smallest paired effect detectable at 80% power with this many solvers.
        c.mde = float((Z_ALPHA + Z_POWER) * c.sd / np.sqrt(len(diff)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    c.lo, c.hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return c


def analyse(d, n_boot=N_BOOT, seed=SEED):
    matched, dropped = _matched(d.copy())
    res = PriorWeakening(n_pairs_dropped=dropped, n_boot=n_boot)
    if matched.empty:
        return res

    # The question as asked: of every item where something WAS wrong, how often did
    # the model name the right view. Conditioning on commitment answers a narrower
    # question and was burying this one.
    n_corrupt = int(matched["true_outlier"].ne(NONE).sum())
    n_commit = int((matched["true_outlier"].ne(NONE)
                    & matched["pred_agree"].eq("no")).sum())
    n_clean = int(matched["true_outlier"].eq(NONE).sum())
    res.overall = _contrast(_solver_outcome(matched, "overall_accuracy"),
                            "named the right view, of all broken items",
                            n_boot, seed, n_items=n_corrupt,
                            denom="all items where something was broken")
    res.primary = _contrast(_solver_outcome(matched, "cond_localization"),
                            "correct outlier, given it committed", n_boot, seed,
                            n_items=n_commit,
                            denom="only the items it flagged")
    res.detection = _contrast(_solver_outcome(matched, "detection"),
                              "committed to a verdict at all", n_boot, seed,
                              n_items=n_corrupt,
                              denom="all broken items — a flag rate, NOT correctness")
    res.specificity = _contrast(_solver_outcome(matched, "specificity"),
                                "correctly said 'all agree'", n_boot, seed,
                                n_items=n_clean,
                                denom="clean items only")

    # Identity check: overall accuracy must equal detection x conditional.
    det = _solver_outcome(matched, "detection")
    con = _solver_outcome(matched, "cond_localization")
    ovr = _solver_outcome(matched, "overall_accuracy")
    prod = (det * con).dropna()
    common = prod.index.intersection(ovr.dropna().index)
    if len(common):
        res.identity_max_err = float(
            np.nanmax(np.abs(prod.loc[common].to_numpy()
                             - ovr.dropna().loc[common].to_numpy())))

    # Per representation, on the SAME outcome as the headline: of the items where
    # THIS view was the broken one, how often the model named it. Corruption subtypes
    # stay pooled -- splitting trajectory four ways multiplies comparisons this
    # design cannot power, and the question is about representations, not rungs.
    for m in MODALITIES:
        sub = matched[matched["true_outlier"].eq(m)]
        if sub.empty:
            continue
        c_overall = _contrast(_solver_outcome(sub, "overall_accuracy"), m,
                              n_boot, seed, exploratory=True, n_items=len(sub),
                              denom=f"items where {m} was the broken view")
        c_cond = _contrast(_solver_outcome(sub, "cond_localization"), m,
                           n_boot, seed, exploratory=True)
        c_overall.conditional = c_cond
        res.per_outlier.append(c_overall)

    # Pre-specified interaction: does obfuscation help where code is innocent and
    # hurt where code is guilty? That asymmetry is what "the prior was lexical" means.
    a_tab = _solver_outcome(matched[matched["true_outlier"].isin(["T", "D", "M"])],
                            "cond_localization")
    b_tab = _solver_outcome(matched[matched["true_outlier"].eq("C")],
                            "cond_localization")
    res.interaction = _interaction(a_tab, b_tab, n_boot, seed)
    return res


def _interaction(a_tab, b_tab, n_boot, seed):
    """(delta where code is innocent) - (delta where code is guilty), paired."""
    c = Contrast(name="asymmetry: innocent-code minus guilty-code")
    if a_tab is None or b_tab is None or a_tab.empty or b_tab.empty:
        return c
    a, b = a_tab.dropna(), b_tab.dropna()
    common = a.index.intersection(b.index)
    if not len(common):
        return c
    da = (a.loc[common, NAMING_LEVELS[1]] - a.loc[common, NAMING_LEVELS[0]]).to_numpy()
    db = (b.loc[common, NAMING_LEVELS[1]] - b.loc[common, NAMING_LEVELS[0]]).to_numpy()
    inter = da - db
    c.n_solvers = len(inter)
    c.real, c.obf = float(np.mean(da)), float(np.mean(db))
    c.diff = float(inter.mean())
    c.sd = float(inter.std(ddof=1)) if len(inter) > 1 else float("nan")
    if np.isfinite(c.sd):
        c.mde = float((Z_ALPHA + Z_POWER) * c.sd / np.sqrt(len(inter)))
    rng = np.random.default_rng(seed + 7)
    idx = rng.integers(0, len(inter), size=(n_boot, len(inter)))
    boots = inter[idx].mean(axis=1)
    c.lo, c.hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return c
