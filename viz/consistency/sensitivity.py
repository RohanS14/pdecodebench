"""Detection sensitivity per corrupted modality (Q1) and blame informativeness (Q2).

Q1 asks whether some corruptions are simply easier to see. The quantity is d', but
d' is a derived number and this module always returns the raw hit and false-alarm
counts beside it, because a reader has to be able to check the derivation. The
false-alarm rate is SHARED across modalities -- it comes from the clean condition,
which is the same set of items regardless of which corruption we are scoring -- so
the only thing that varies row to row is the hit rate.

Q2 asks whether blame carries information about the true outlier at all. The test is
whether the model beats the BEST FIXED ANSWER on the same items -- always naming one
view, whichever is most often correct. A stereotyped responder reduces exactly to
that policy, so clearing it is the bar. The margin is bootstrapped over solvers, and
the reader can check it against the bars: if every row looks like the reference row,
the margin is zero.
"""
import sys, os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from . import metrics as M
from .constants import (TRAJ_LEVEL_LABELS, CONDITIONS, CONDITION_OUTLIER, MODALITIES,
                        MODALITY_LABELS, NONE, OUTLIER_LEVELS)

# Corrupted conditions, trajectory disaggregated into its four rungs. The
# trajectory is not one manipulation: a shape-matched noise field and the
# invalid solver's own output differ by more than whole modalities do, so
# pooling them reports the mean of a wide range as a single difficulty.
SIGNAL_CONDITIONS = [c for c in CONDITIONS if CONDITION_OUTLIER[c] != NONE]


def row_label(cond):
    """Plain label for a corrupted condition row."""
    m = CONDITION_OUTLIER[cond]
    if cond.startswith("A-T-"):
        return f"trajectory \u2014 {cond.rsplit(chr(45), 1)[1]}"
    return MODALITY_LABELS[m]


# Short, readable names for the four trajectory rungs. `rand`/`shuf`/`swap`/`exec`
# are the generator's internal names and say nothing about what was done to the data
# -- which matters most here, because those four rows are a DIFFICULTY LADDER and the
# codes give no hint of the ordering. Kept to two words so they stay scannable down a
# y-axis, and worded identically to the blame matrix so one vocabulary carries between
# the two figures.
TRAJ_SHORT = {
    "rand": "random values",
    "shuf": "shuffled",
    "swap": "wrong system",
    # The invalid solver's OWN output, not a synthetic corruption: this rung is the
    # subtlest of the four because the trajectory is physically self-consistent with
    # the (wrong) code that produced it. "solver output" alone lost that.
    "exec": "invalid solver's output",
}


def row_caption_corrupted(cond):
    """Clear caption that still SAYS "corrupted".

    The blame figures can drop the word because their axis label carries it once
    ("which view was corrupted"). The sensitivity dot plot has no such label -- its
    axis is a rate -- so each row has to say what happened to it. The trajectory
    rungs put the word on the modality rather than on the rung, because "random
    values corrupted" reads as though the random values were the victim.
    """
    if cond.startswith("A-T-"):
        return ("trajectory corrupted \u2014 "
                + TRAJ_SHORT[cond.rsplit(chr(45), 1)[1]])
    return MODALITY_LABELS[CONDITION_OUTLIER[cond]] + " corrupted"


def row_caption(cond, verbose=False):
    """Y-axis caption for a corrupted condition row.

    Defaults to the terse form: consistency_claims.html is a frozen artifact and its
    build must keep reproducing it byte for byte, so the expanded report opts in.
    The clear form drops the repeated "was corrupted" -- the axis label already says
    it once -- which is what buys the room to name the trajectory rungs.
    """
    if not verbose:
        return f"{row_label(cond)} was corrupted"
    if cond.startswith("A-T-"):
        return "trajectory \u2014 " + TRAJ_SHORT[cond.rsplit(chr(45), 1)[1]]
    return MODALITY_LABELS[CONDITION_OUTLIER[cond]]

try:
    from cross_modal_consistency.eval.parse_consistency import dprime as _dprime
except Exception:                                                # pragma: no cover
    from scipy.stats import norm as _norm

    def _dprime(n_hit, n_signal, n_fa, n_noise):
        h = (n_hit + 0.5) / (n_signal + 1)
        f = (n_fa + 0.5) / (n_noise + 1)
        return float(_norm.ppf(h) - _norm.ppf(f))

N_BOOT = 10_000
N_PERM = 10_000
MIN_N = 20
SEED = 20260820


# ── Q1 ───────────────────────────────────────────────────────────────────────
@dataclass
class Sensitivity:
    rows: list = field(default_factory=list)
    fa_rate: float = float("nan")
    fa_lo: float = float("nan")
    fa_hi: float = float("nan")
    n_fa: int = 0
    n_noise: int = 0
    n_boot: int = N_BOOT
    resolved: bool = False          # is the top-vs-bottom ordering separable?


def detection_sensitivity(df, n_boot=N_BOOT, seed=SEED):
    """d' per corrupted modality against the SHARED clean-condition false alarms."""
    d = M.prepare(df)
    clean = d[~d["is_corrupted"]]
    n_fa, n_noise = int(clean["detected"].sum()), len(clean)
    res = Sensitivity(fa_rate=(n_fa / n_noise if n_noise else float("nan")),
                      n_fa=n_fa, n_noise=n_noise, n_boot=n_boot)
    if not n_noise:
        return res
    _r = np.random.default_rng(seed + 1)
    _b = _r.binomial(n_noise, n_fa / n_noise, size=min(n_boot, 4000)) / n_noise
    res.fa_lo, res.fa_hi = (float(v) for v in np.percentile(_b, [2.5, 97.5]))

    solvers = sorted(d["solver_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(solvers), size=(n_boot, len(solvers)))
    by_solver = {s: g for s, g in d.groupby(d["solver_id"].astype(str))}

    # Precompute per-solver counts so the bootstrap is a lookup, not a regroup.
    tallies = {}
    for s, g in by_solver.items():
        gc = g[~g["is_corrupted"]]
        t = {"fa": int(gc["detected"].sum()), "noise": len(gc)}
        for cond in SIGNAL_CONDITIONS:
            gm = g[g["condition"].eq(cond)]
            t[cond] = (int(gm["detected"].sum()), len(gm))
        tallies[s] = t
    order = [tallies[s] for s in solvers]
    fa_v = np.array([t["fa"] for t in order], dtype=float)
    noise_v = np.array([t["noise"] for t in order], dtype=float)

    for cond in SIGNAL_CONDITIONS:
        m = CONDITION_OUTLIER[cond]
        sig = d[d["condition"].eq(cond)]
        n_hit, n_signal = int(sig["detected"].sum()), len(sig)
        if not n_signal:
            res.rows.append(dict(condition=cond, modality=m, label=row_label(cond),
                                 lo_rate=float("nan"), hi_rate=float("nan"),
                                 n_hit=0, n_signal=0, hit_rate=float("nan"),
                                 dprime=float("nan"), lo=float("nan"), hi=float("nan"),
                                 thin=True, empty=True))
            continue
        hit_v = np.array([t[cond][0] for t in order], dtype=float)
        sig_v = np.array([t[cond][1] for t in order], dtype=float)
        bh, bs = hit_v[draws].sum(axis=1), sig_v[draws].sum(axis=1)
        bf, bn = fa_v[draws].sum(axis=1), noise_v[draws].sum(axis=1)
        boots = np.array([_dprime(h, s, f, n)
                          for h, s, f, n in zip(bh, bs, bf, bn)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rate_lo, rate_hi = np.percentile(bh / np.where(bs > 0, bs, np.nan),
                                         [2.5, 97.5])
        res.rows.append(dict(
            lo_rate=float(rate_lo), hi_rate=float(rate_hi),
            condition=cond, modality=m, label=row_label(cond),
            n_hit=n_hit, n_signal=n_signal,
            hit_rate=n_hit / n_signal,
            dprime=float(_dprime(n_hit, n_signal, n_fa, n_noise)),
            lo=float(lo), hi=float(hi),
            thin=n_signal < MIN_N, empty=False))

    usable = [r for r in res.rows if not r["empty"] and not r["thin"]]
    if len(usable) >= 2:
        usable.sort(key=lambda r: -r["dprime"])
        top, bot = usable[0], usable[-1]
        # Overlapping intervals mean the ranking is not established by this data.
        res.resolved = top["lo"] > bot["hi"]
    res.rows.sort(key=lambda r: (-1e9 if r["empty"] else -r["dprime"]))
    return res


# ── Q2 ───────────────────────────────────────────────────────────────────────
@dataclass
class BlameInfo:
    table: object = None            # DataFrame, true x pred counts
    marginal: object = None         # Series over OUTLIER_LEVELS, the pooled reference
    null: object = None             # bootstrap draws of the margin over a fixed answer
    margin_lo: float = float("nan")
    margin_hi: float = float("nan")
    cramers_v: float = float("nan")
    localization: float = float("nan")
    best_constant: float = float("nan")
    composition: dict = field(default_factory=dict)
    n: int = 0
    n_perm: int = N_PERM
    informative: bool = False


def blame_information(df, n_perm=N_PERM, seed=SEED):
    """Does blame track the true outlier, or is it a fixed response policy?"""
    d = M.prepare(df)
    det = d[d["is_corrupted"] & d["detected"]
            & d["pred_outlier"].isin(OUTLIER_LEVELS)]
    res = BlameInfo(n=len(det), n_perm=n_perm)
    if det.empty:
        return res

    # Rows are CONDITIONS (trajectory disaggregated); columns stay modalities,
    # because the model blames "trajectory", not "trajectory-shuffled".
    table = (det.groupby(["condition", "pred_outlier"]).size().unstack(fill_value=0)
             .reindex(index=SIGNAL_CONDITIONS, columns=list(OUTLIER_LEVELS),
                      fill_value=0))
    res.table = table
    # Row 5 of the figure: the pooled column marginal, i.e. what blame looks like
    # when you ignore what was actually corrupted.
    res.marginal = table.sum(axis=0) / table.to_numpy().sum()
    true_codes = det["true_outlier"].to_numpy()
    # Bootstrap over SOLVERS the margin between what the model achieves and what the
    # best fixed answer would achieve on the same items. A fixed policy is the thing
    # a stereotyped responder reduces to, so clearing it is the bar for "it knows
    # which thing".
    solvers = det["solver_id"].astype(str).to_numpy()
    uniq = sorted(set(solvers))
    by = {sv: det[det["solver_id"].astype(str).eq(sv)] for sv in uniq}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_perm, len(uniq)))
    margins = np.empty(n_perm)
    hit_v = np.array([int(by[sv]["localization_correct"].sum()) for sv in uniq],
                     dtype=float)
    n_v = np.array([len(by[sv]) for sv in uniq], dtype=float)
    per_true = {m: np.array([int(by[sv]["true_outlier"].eq(m).sum()) for sv in uniq],
                            dtype=float) for m in MODALITIES}   # blame is per modality
    for b in range(n_perm):
        idx = draws[b]
        n_tot = n_v[idx].sum()
        if n_tot <= 0:
            margins[b] = 0.0
            continue
        acc = hit_v[idx].sum() / n_tot
        const = max(per_true[m][idx].sum() for m in MODALITIES) / n_tot
        margins[b] = acc - const
    res.null = margins
    lo, hi = np.percentile(margins, [2.5, 97.5])
    res.margin_lo, res.margin_hi = float(lo), float(hi)
    res.informative = bool(lo > 0.0)

    obs = table.to_numpy().astype(float)
    n = obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0).sum())
    k = min(obs.shape) - 1
    res.cramers_v = float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")

    res.localization = (sum(int(table.loc[c, CONDITION_OUTLIER[c]])
                            for c in SIGNAL_CONDITIONS
                            if CONDITION_OUTLIER[c] in table.columns) / n
                        ) if n else float("nan")
    # Best a FIXED policy could do: always name one view, and be right whenever that
    # view happens to be the corrupted one. This is the number localization accuracy
    # has to beat before it means anything.
    res.best_constant = (max(
        sum(int(table.loc[c, :].sum()) for c in SIGNAL_CONDITIONS
            if CONDITION_OUTLIER[c] == m) for m in MODALITIES) / n
                         ) if n else float("nan")
    res.composition = det["condition"].value_counts().to_dict()
    return res


# ── severity matching ────────────────────────────────────────────────────────
# Cross-MODALITY ordering is only meaningful if the corruptions are comparable in
# severity. They are not here: the trajectory carries four generation methods while
# code, description and math carry one each. That is why the rows are disaggregated
# and why the banner exists -- an ordering over modalities would be reporting how
# each corruption was built, not what the model trusts.

def severity_tiers(df):
    """(table, matched, common_tiers). Tier = corruption-generation method."""
    d = df.copy()
    tier = d.get("traj_level", pd.Series("", index=d.index)).fillna("").astype(str)
    d["_tier"] = np.where(tier.eq(""), "single", tier)
    corrupted = d[d["true_outlier"].isin(MODALITIES)]
    if corrupted.empty:
        return pd.DataFrame(), True, []
    table = (corrupted.groupby(["true_outlier", "_tier"]).size()
             .unstack(fill_value=0).reindex(index=list(MODALITIES), fill_value=0))
    sets = {m: {t for t in table.columns if table.loc[m, t] > 0}
            for m in table.index if table.loc[m].sum() > 0}
    common = sorted(set.intersection(*sets.values())) if sets else []
    matched = len({frozenset(v) for v in sets.values()}) == 1
    return table, matched, common
