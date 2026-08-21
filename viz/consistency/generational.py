"""Detection sensitivity as a function of model release date.

The question: has cross-modal physics-consistency detection improved as open-weight
reasoning models have got newer? The roster is held in a narrow 27.8-36B band on
purpose, so recency is not confounded with scale, and every arm plotted here runs
reasoning-enabled, so it is not confounded with a thinking toggle either.

Two design commitments, both of which exist because the naive version is wrong:

**The y-axis is d', not hit rate.** In the published data QwQ-32B posts a hit rate of
0.928 against a false-alarm rate of 0.710, while Qwen3-32B (thinking off) posts 0.408
against 0.125. Plotting hit rate would show a large generational improvement that is
mostly response bias -- a model answering "they disagree" more often, not seeing more.
d' subtracts that off. Raw hit and false-alarm counts ride along on every row so a
reader can check the derivation rather than trust it.

**The interval is bootstrapped over SOLVERS, not rows.** There are 1024 items per arm
but only 32 physical systems behind them, and items from one solver are not
independent. Resampling rows would produce intervals about sqrt(32) too narrow and
manufacture a trend out of noise. This mirrors `sensitivity.py`, which already
clusters the same way; the two modules deliberately share the pattern.

A model missing from `data/model_registry.csv` has no release date and is dropped
rather than plotted at a guessed one -- see `adapter.from_xmodal`.
"""
import numpy as np
import pandas as pd

from . import metrics as M
from .sensitivity import MIN_N, N_BOOT, SEED, _dprime


def by_release(df, n_boot=N_BOOT, seed=SEED, arm=None):
    """One row per (model, reasoning arm), ordered by release date.

    `arm` filters to a single reasoning level ("on") when given. Rows carry the
    clustered d' interval, the raw counts behind it, and a `degenerate` flag for an
    arm that answered all-one-way, where d' is defined but meaningless.
    """
    d = M.prepare(df)
    if arm is not None and "reasoning" in d:
        d = d[d["reasoning"].astype(str).eq(arm)]
    if d.empty or "release_date" not in d or not d["release_date"].astype(str).str.len().any():
        return pd.DataFrame(columns=[
            "model", "reasoning", "release_date", "params_total_b", "family",
            "n_hit", "n_signal", "n_fa", "n_noise", "hit_rate", "fa_rate",
            "dprime", "lo", "hi", "degenerate", "thin"])

    rows = []
    for (model, reasoning), g in d.groupby([d["model"].astype(str),
                                            d["reasoning"].astype(str)]):
        release = str(g["release_date"].iloc[0])
        if not release:
            continue                       # unregistered -> no x position, drop it
        sig, clean = g[g["is_corrupted"]], g[~g["is_corrupted"]]
        n_hit, n_signal = int(sig["detected"].sum()), len(sig)
        n_fa, n_noise = int(clean["detected"].sum()), len(clean)
        if not n_signal or not n_noise:
            continue

        # Clustered bootstrap: resample the 32 solvers with replacement, recompute
        # d' from the resampled tallies. Same construction as sensitivity.py.
        solvers = sorted(g["solver_id"].astype(str).unique())
        tally = []
        for s, gs in g.groupby(g["solver_id"].astype(str)):
            gsig, gcl = gs[gs["is_corrupted"]], gs[~gs["is_corrupted"]]
            tally.append((int(gsig["detected"].sum()), len(gsig),
                          int(gcl["detected"].sum()), len(gcl)))
        t = np.array(tally, dtype=float)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(solvers), size=(n_boot, len(solvers)))
        b = t[draws].sum(axis=1)           # (n_boot, 4)
        boots = np.array([_dprime(h, s, f, n) for h, s, f, n in b])
        lo, hi = np.percentile(boots, [2.5, 97.5])

        hit_rate, fa_rate = n_hit / n_signal, n_fa / n_noise
        rows.append(dict(
            model=model, reasoning=reasoning, release_date=release,
            params_total_b=float(g["params_total_b"].iloc[0]),
            family=str(g["family"].iloc[0]),
            n_hit=n_hit, n_signal=n_signal, n_fa=n_fa, n_noise=n_noise,
            hit_rate=hit_rate, fa_rate=fa_rate,
            dprime=float(_dprime(n_hit, n_signal, n_fa, n_noise)),
            lo=float(lo), hi=float(hi),
            # An arm that answered the SAME WAY to everything carries no
            # discrimination information; d' is finite only because of the Hautus
            # correction. Flag rather than silently plot.
            #
            # Note the pairing: it is (all-yes) or (all-no), NOT "both rates are
            # extreme". A perfect detector has hit=1.0 and fa=0.0 -- both extreme,
            # but maximally informative -- and an earlier version flagged it
            # degenerate, which dropped the best model in the roster from trend().
            # This matches parse_consistency.dprime's definition; the two must not
            # diverge.
            degenerate=bool((hit_rate == 1.0 and fa_rate == 1.0)
                            or (hit_rate == 0.0 and fa_rate == 0.0)),
            thin=n_signal < MIN_N,
        ))

    out = pd.DataFrame(rows)
    return out.sort_values(["release_date", "model"]).reset_index(drop=True) \
        if not out.empty else out


def trend(rows):
    """Least-squares slope of d' on release date, in d' units per year.

    Reported with the number of points behind it, because a slope through four
    models is a description of four models and should never be read as a law. Returns
    None when there are fewer than three usable points or no date spread.
    """
    r = rows[~rows["degenerate"] & ~rows["thin"]] if len(rows) else rows
    if len(r) < 3:
        return None
    x = pd.to_datetime(r["release_date"]).map(lambda t: t.toordinal()).to_numpy(float)
    if x.max() == x.min():
        return None
    y = r["dprime"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(
        slope_per_year=float(slope * 365.25),
        r2=float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        n_models=int(len(r)),
        span_days=int(x.max() - x.min()),
    )
