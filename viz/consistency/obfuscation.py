"""Paired within-solver analysis of the naming manipulation.

The unit of analysis is a solver, not a row. Every solver appears under both naming
conditions, so a marginal comparison of group means would throw away that pairing and
absorb between-solver variance into the error term -- with 32 solvers of very
different difficulty, that is most of the variance in the design.

Observations are matched on (solver_id, condition, reasoning, model) before any
differencing. A cell present under one naming condition and absent under the other is
DROPPED, and the count of drops is returned rather than swallowed: a silent drop here
would quietly unbalance the pairing while still producing a plausible number.

Two disjoint slices, never pooled:

  innocent  true_outlier != "C"  — does obfuscation make blameless code look guilty?
  guilty    true_outlier == "C"  — does obfuscation hide a real defect in the code?

They are expected to move in opposite directions, so a single pooled rate would net
them against each other and report the difference of two real effects as no effect.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import DELTA_IS, MODALITIES, NAMING_LEVELS, NONE

BLAME_ROWS = list(MODALITIES) + [NONE]      # C, T, D, M, none — in the spec's order
PAIR_KEY = ["solver_id", "condition", "reasoning", "model"]
N_BOOT = 10_000
ALPHA = 0.05
MIN_N = 20
SEED = 20260820


@dataclass
class PairedResult:
    rows: list = field(default_factory=list)   # per-category dicts
    delta_is: str = DELTA_IS
    renormalised: bool = False
    n_solvers: int = 0
    dropped_cells: int = 0
    dropped_unparsed: int = 0
    n_boot: int = N_BOOT
    alpha: float = ALPHA
    alpha_corrected: float = ALPHA
    slice_name: str = ""


def match_pairs(d, slice_name):
    """Keep only (solver, condition, reasoning, model) cells present in BOTH namings.

    Returns (matched_frame, dropped_cell_count, dropped_unparsed_row_count).
    """
    sub = d[d["true_outlier"].ne("C")] if slice_name == "innocent" \
        else d[d["true_outlier"].eq("C")]
    # A row with no parseable verdict is neither a blame nor an explicit "none";
    # folding it into `none` would invent an answer the model never gave, so it is
    # removed and counted. Shares below therefore sum to 1 over the five real answers.
    unparsed = int((~sub["pred_outlier"].isin(BLAME_ROWS)).sum())
    sub = sub[sub["pred_outlier"].isin(BLAME_ROWS)]
    if sub.empty:
        return sub, 0, unparsed

    seen = sub.groupby(PAIR_KEY)["naming"].nunique()
    complete = seen[seen == len(NAMING_LEVELS)].index
    dropped = int((seen != len(NAMING_LEVELS)).sum())
    matched = sub.set_index(PAIR_KEY).loc[complete].reset_index() \
        if len(complete) else sub.iloc[0:0]
    return matched, dropped, unparsed


def _solver_shares(matched, renormalise=False):
    """(naming -> DataFrame indexed by solver, columns BLAME_ROWS, rows summing to 1)."""
    out = {}
    for naming in NAMING_LEVELS:
        g = matched[matched["naming"].eq(naming)]
        tab = (g.groupby(["solver_id", "pred_outlier"]).size().unstack(fill_value=0)
               .reindex(columns=BLAME_ROWS, fill_value=0))
        if renormalise:
            # 'none' is not a modality -- it is a refusal to name one. Dropping it and
            # renormalising makes the four modality rows a composition of BLAME, which
            # is what "which view does it blame" actually means.
            tab = tab[list(MODALITIES)]
        tot = tab.sum(axis=1)
        out[naming] = tab.div(tot.where(tot > 0, np.nan), axis=0)
    return out


def paired_blame_shift(d, slice_name="innocent", n_boot=N_BOOT, seed=SEED,
                       renormalise=False):
    """Paired blame-share delta per category, bootstrapped over solvers.

    Sign is DELTA_IS (obfuscated - real), imported from constants so the figure and
    the verdict line cannot disagree about which direction is positive.
    """
    matched, dropped, unparsed = match_pairs(d, slice_name)
    res = PairedResult(dropped_cells=dropped, dropped_unparsed=unparsed,
                       n_boot=n_boot, slice_name=slice_name,
                       renormalised=renormalise)
    if matched.empty:
        return res

    cats = list(MODALITIES) if renormalise else BLAME_ROWS
    shares = _solver_shares(matched, renormalise=renormalise)
    solvers = sorted(set(shares[NAMING_LEVELS[0]].index)
                     & set(shares[NAMING_LEVELS[1]].index))
    real = shares[NAMING_LEVELS[0]].reindex(solvers)
    obf = shares[NAMING_LEVELS[1]].reindex(solvers)
    keep = real.notna().all(axis=1) & obf.notna().all(axis=1)
    real, obf = real[keep], obf[keep]
    solvers = list(real.index)
    res.n_solvers = len(solvers)
    if not solvers:
        return res

    diff = (obf - real).to_numpy()                       # solvers x categories
    rng = np.random.default_rng(seed)
    # Resample SOLVERS, not rows: rows within a solver are not independent, and a
    # row-level bootstrap would shrink the interval by pretending they are.
    idx = rng.integers(0, len(solvers), size=(n_boot, len(solvers)))
    boots = diff[idx].mean(axis=1)                       # n_boot x categories

    res.alpha_corrected = ALPHA / len(cats)              # Bonferroni across the rows
    lo_p, hi_p = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    clo_p, chi_p = 100 * res.alpha_corrected / 2, 100 * (1 - res.alpha_corrected / 2)

    for j, cat in enumerate(cats):
        n_real = int(matched[matched["naming"].eq(NAMING_LEVELS[0])]
                     ["pred_outlier"].eq(cat).sum())
        n_obf = int(matched[matched["naming"].eq(NAMING_LEVELS[1])]
                    ["pred_outlier"].eq(cat).sum())
        b = boots[:, j]
        lo, hi = np.percentile(b, [lo_p, hi_p])
        clo, chi = np.percentile(b, [clo_p, chi_p])
        thin = min(n_real, n_obf) < MIN_N
        res.rows.append(dict(
            category=cat,
            real=float(real[cat].mean()), obf=float(obf[cat].mean()),
            diff=float(diff[:, j].mean()),
            lo=float(lo), hi=float(hi),               # 95% whisker
            clo=float(clo), chi=float(chi),           # Bonferroni-corrected
            n_real=n_real, n_obf=n_obf, thin=thin,
            # Significance is judged on the CORRECTED interval; the drawn whisker is
            # the uncorrected 95% one, and the caption says both.
            significant=bool(not thin and not (clo <= 0.0 <= chi)),
        ))
    return res


def paired_guilty_recall(d, n_boot=N_BOOT, seed=SEED):
    """Paired shift in how often a genuinely broken code view is named, A-C only."""
    matched, dropped, unparsed = match_pairs(d, "guilty")
    res = PairedResult(dropped_cells=dropped, dropped_unparsed=unparsed,
                       n_boot=n_boot, slice_name="guilty")
    if matched.empty:
        return res
    hit = matched.assign(hit=matched["pred_outlier"].eq("C").astype(float))
    tab = hit.pivot_table(index="solver_id", columns="naming", values="hit",
                          aggfunc="mean")
    tab = tab.dropna()
    if tab.empty:
        return res
    res.n_solvers = len(tab)
    diff = (tab[NAMING_LEVELS[1]] - tab[NAMING_LEVELS[0]]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    res.alpha_corrected = ALPHA          # single quantity, nothing to correct across
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_real = int(matched["naming"].eq(NAMING_LEVELS[0]).sum())
    n_obf = int(matched["naming"].eq(NAMING_LEVELS[1]).sum())
    res.rows.append(dict(
        category="names the broken code", real=float(tab[NAMING_LEVELS[0]].mean()),
        obf=float(tab[NAMING_LEVELS[1]].mean()), diff=float(diff.mean()),
        lo=float(lo), hi=float(hi), clo=float(lo), chi=float(hi),
        n_real=n_real, n_obf=n_obf, thin=min(n_real, n_obf) < MIN_N,
        significant=bool(min(n_real, n_obf) >= MIN_N and not (lo <= 0.0 <= hi))))
    return res


def paired_refusal_by_condition(d, n_boot=N_BOOT, seed=SEED):
    """Paired obfuscated-minus-real change in "no view blamed", per condition.

    The blame TARGETS cannot be disaggregated -- the model names a view, not a
    trajectory rung -- but the SLICE can. This asks whether obfuscation quiets the
    model uniformly, or only when particular views are corrupted; a uniform shift is
    a response-policy effect, while one concentrated on a single corruption would be
    something about that corruption's content.
    """
    from .constants import CONDITIONS, CONDITION_OUTLIER
    out = []
    d = d.copy()
    d["_refused"] = (~d["pred_outlier"].isin(MODALITIES)).astype(float)
    for cond in CONDITIONS:
        sub = d[d["condition"].eq(cond)]
        if sub.empty:
            continue
        tab = sub.pivot_table(index="solver_id", columns="naming", values="_refused",
                              aggfunc="mean")
        if not set(NAMING_LEVELS) <= set(tab.columns):
            continue
        tab = tab.dropna()
        if tab.empty:
            continue
        diff = (tab[NAMING_LEVELS[1]] - tab[NAMING_LEVELS[0]]).to_numpy()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boots = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        # Bonferroni across the conditions shown in this block.
        k = len([c for c in CONDITIONS if CONDITION_OUTLIER[c] != NONE]) + 1
        clo, chi = np.percentile(boots, [100 * (ALPHA / k) / 2,
                                         100 * (1 - (ALPHA / k) / 2)])
        n_real = int(sub["naming"].eq(NAMING_LEVELS[0]).sum())
        n_obf = int(sub["naming"].eq(NAMING_LEVELS[1]).sum())
        out.append(dict(
            condition=cond, category=cond, modality=CONDITION_OUTLIER[cond],
            real=float(tab[NAMING_LEVELS[0]].mean()),
            obf=float(tab[NAMING_LEVELS[1]].mean()),
            diff=float(diff.mean()), lo=float(lo), hi=float(hi),
            clo=float(clo), chi=float(chi), n_real=n_real, n_obf=n_obf,
            n_solvers=len(tab), thin=min(n_real, n_obf) < MIN_N,
            significant=bool(min(n_real, n_obf) >= MIN_N and not (clo <= 0.0 <= chi))))
    return out
