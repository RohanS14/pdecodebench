"""Synthetic results, so every figure can be built and reviewed before real data lands.

The generator exists to exercise the plotting stack, not to predict results. It
injects three effects by default because those are the three the figures are
supposed to be able to *show*; if a figure cannot make an injected effect visible,
the figure is wrong, and that is the only claim this module supports.

Defaults injected:
  * low specificity on A0     -- the model flags clean items at `a0_false_alarm`
  * over-trust of description -- A-D items, when detected, get misattributed to
                                 C or T rather than D (`d_misblame_share`)
  * a C-to-T shift under obfuscation -- blame mass moves off code onto trajectory
                                 when identifiers are stripped (`obf_c_to_t_shift`)

Nothing downstream may branch on data being synthetic: `generate()` returns exactly
`SCHEMA_COLUMNS`, in that order, and that is the whole contract.
"""
import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (CONDITIONS, CONDITION_OUTLIER, CONDITION_TRAJ_LEVEL,
                        MODALITIES, MODALITY_LABELS, NAMING_LEVELS, NONE,
                        NUMERICAL_METHODS, PDE_CLASSES, REASONING_LEVELS,
                        SCHEMA_COLUMNS, TRAJ_LEVELS)


@dataclass
class Effects:
    """Effect sizes. Every field is a probability unless named otherwise."""
    n_solvers: int = 32
    models: tuple = ("model-a", "model-b", "model-c")

    # Detection. Recall is per-modality: a corrupted trajectory is grosser than a
    # corrupted equation, so they should not detect at one rate.
    a0_false_alarm: float = 0.42          # low specificity on clean items
    recall: dict = field(default_factory=lambda: {
        "C": 0.55, "T": 0.72, "D": 0.38, "M": 0.50})
    # The corruption ladder. A shape-matched noise field is obvious; the invalid
    # solver's own output is the subtlest thing in the design, because it is a real
    # trajectory that a real program really produced. Detection should fall
    # monotonically down this list, and a figure that cannot show that is not
    # earning the four extra conditions.
    traj_recall: dict = field(default_factory=lambda: {
        "rand": 0.88, "shuf": 0.74, "swap": 0.55, "exec": 0.31})
    reasoning_recall_bonus: float = 0.10  # additive, applied when reasoning == "on"

    # Localization, given the item was flagged.
    localization: dict = field(default_factory=lambda: {
        "C": 0.62, "T": 0.70, "D": 0.30, "M": 0.55})
    # Over-trust of the description: when a corrupted D is caught but mislocalized,
    # the blame lands on code or trajectory rather than spreading evenly.
    d_misblame_share: float = 0.80
    d_misblame_split: tuple = ("C", "T")

    # Obfuscation moves blame off code and onto trajectory.
    obf_c_to_t_shift: float = 0.60
    obf_recall_penalty: float = 0.05

    # The justification gap: the judge confirms the stated defect less often than
    # the slot was picked correctly, which is the point of fig4.
    judge_given_correct: float = 0.72
    judge_given_wrong: float = 0.06

    pde_accuracy: float = 0.88
    seed: int = 20260820


def _pick(rng, options, weights):
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return rng.choice(options)
    return rng.choice(options, p=w / total)


def _blame_weights(true_outlier, naming, eff):
    """Where wrong blame lands, given the true outlier and the naming condition."""
    others = [m for m in MODALITIES if m != true_outlier]
    w = {m: 1.0 for m in others}
    if true_outlier == "D":
        # Over-trust of D: its misattributions concentrate on C and T.
        targets = [m for m in eff.d_misblame_split if m in w]
        if targets:
            for m in w:
                w[m] = 0.0
            share = eff.d_misblame_share / len(targets)
            for m in targets:
                w[m] = share
            spare = 1.0 - eff.d_misblame_share
            rest = [m for m in others if m not in targets]
            for m in rest:
                w[m] = spare / len(rest) if rest else 0.0
    if naming == "obfuscated" and "C" in w and "T" in w:
        # Identifiers gone, the model leans off code and onto the numbers.
        moved = w["C"] * eff.obf_c_to_t_shift
        w["C"] -= moved
        w["T"] += moved
    return [w[m] for m in others], others


def generate(eff=None):
    """Full crossing: solver x condition x naming x reasoning x model."""
    eff = eff or Effects()
    rng = np.random.default_rng(eff.seed)

    solvers = [f"S{i:02d}" for i in range(eff.n_solvers)]
    solver_pde = {s: PDE_CLASSES[i % len(PDE_CLASSES)] for i, s in enumerate(solvers)}
    solver_method = {s: NUMERICAL_METHODS[i % len(NUMERICAL_METHODS)]
                     for i, s in enumerate(solvers)}

    rows = []
    rid = 0
    for solver in solvers:
        for condition in CONDITIONS:
            true_outlier = CONDITION_OUTLIER[condition]
            for naming in NAMING_LEVELS:
                for reasoning in REASONING_LEVELS:
                    for model in eff.models:
                        rid += 1
                        order = list(MODALITIES)
                        rng.shuffle(order)

                        traj_level = CONDITION_TRAJ_LEVEL.get(condition, "")
                        if true_outlier == NONE:
                            flagged = rng.random() < eff.a0_false_alarm
                        else:
                            # A trajectory rung overrides the modality-level recall:
                            # "how detectable is the trajectory" is not one number.
                            p = (eff.traj_recall[traj_level] if traj_level
                                 else eff.recall[true_outlier])
                            if reasoning == "on":
                                p += eff.reasoning_recall_bonus
                            if naming == "obfuscated":
                                p -= eff.obf_recall_penalty
                            flagged = rng.random() < min(max(p, 0.0), 1.0)

                        if not flagged:
                            pred_agree, pred_outlier = "yes", NONE
                            correct_slot = (true_outlier == NONE)
                        else:
                            pred_agree = "no"
                            if true_outlier == NONE:
                                # False alarm: blame is arbitrary, but obfuscation
                                # still tilts it, which is how a clean-item default
                                # shows up as a false-blame rate in fig2.
                                w, opts = _blame_weights("__none__", naming, eff)
                                pred_outlier = _pick(rng, opts, w)
                                correct_slot = False
                            elif rng.random() < eff.localization[true_outlier]:
                                pred_outlier = true_outlier
                                correct_slot = True
                            else:
                                w, opts = _blame_weights(true_outlier, naming, eff)
                                pred_outlier = _pick(rng, opts, w)
                                correct_slot = False

                        judge = rng.random() < (eff.judge_given_correct if correct_slot
                                                and true_outlier != NONE
                                                else eff.judge_given_wrong)
                        pde_true = solver_pde[solver]
                        pred_pde = (pde_true if rng.random() < eff.pde_accuracy
                                    else str(rng.choice([p for p in PDE_CLASSES
                                                         if p != pde_true])))
                        blamed = MODALITY_LABELS.get(pred_outlier, "nothing")
                        rows.append({
                            "run_id": f"r{rid:06d}",
                            "solver_id": solver,
                            "pde_class": pde_true,
                            "numerical_method": solver_method[solver],
                            "condition": condition,
                            "true_outlier": true_outlier,
                            "traj_level": traj_level,
                            "naming": naming,
                            "reasoning": reasoning,
                            "model": model,
                            "order": ",".join(order),
                            "pred_agree": pred_agree,
                            "pred_outlier": pred_outlier,
                            "pred_pde_class": pred_pde,
                            "pred_method": solver_method[solver],
                            "justification": (
                                f"The {blamed} view is inconsistent with the others; "
                                f"the remaining three agree on a {pde_true} system."
                                if pred_agree == "no" else
                                "All four representations describe the same system."),
                            "judge_correct": bool(judge),
                        })
    return pd.DataFrame(rows, columns=list(SCHEMA_COLUMNS))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/consistency_runs.csv")
    ap.add_argument("--seed", type=int, default=Effects.seed)
    ap.add_argument("--n_solvers", type=int, default=Effects.n_solvers)
    a = ap.parse_args()
    df = generate(Effects(seed=a.seed, n_solvers=a.n_solvers))
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[synth] wrote {a.out}  ({len(df):,} rows, {df.model.nunique()} models)")


if __name__ == "__main__":
    main()
