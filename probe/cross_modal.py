"""
Experiment 2, Part II — cross-modal alignment (plan §15).

Three representations of the same physical system, two of which contain no code:
the symbolic EQUATION, the executed TRAJECTORY, and the solver CODE. If the model
represents the physics rather than the program, a representation built from one
modality should locate the same physical system in another.

Two rules, both from the plan and both enforced here rather than left to the
reader:

  1. Retrieval is scored WITHIN pde_class as well as globally. There are only 4
     classes, so global retrieval (chance 1/32) can be solved by 4-way category
     matching. Within-class retrieval (chance 1/8) forces instance-level matching.
     The within-class number is the real one.

  2. Lexical overlap between the two modalities is measured and partialled out.
     A retrieval result that disappears once lexical overlap is controlled is a
     lexical result, not a physics result.

The headline pairing is equation → NoComm_CorrVar code: no comments, obfuscated
identifiers, so lexical overlap between "∂u/∂t = ν ∂²u/∂x²" and the code is close
to nil.

Usage (from repo root):
    python probe/cross_modal.py \
        --code_hidden probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz \
        --equation_hidden probe/hidden_states/Qwen_..._equation.npz \
        --trajectory_hidden probe/hidden_states/Qwen_..._trajectory.npz \
        --output_dir probe/results/ --pool mean_pool
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_similarity import jaccard  # noqa: E402

HEADLINE_CODE_CONDITION = "NoComm_CorrVar"


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def word_multiset(s: str) -> Counter:
    """Lowercased alphanumeric-run multiset — the lexical-overlap nuisance."""
    out, cur = Counter(), []
    for ch in str(s).lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out["".join(cur)] += 1
            cur = []
    if cur:
        out["".join(cur)] += 1
    return out


def retrieval(M: np.ndarray, groups: np.ndarray = None) -> dict:
    """
    Row i of M scores query i against all candidates. Correct answer is j == i.

    With `groups`, candidates are restricted to the query's own group — this is
    the within-pde_class score, whose chance level is 1/group_size rather than
    1/N. Reported alongside the global score, never instead of it.
    """
    N = M.shape[0]
    ranks = np.empty(N)
    for i in range(N):
        cand = np.arange(N) if groups is None else np.where(groups == groups[i])[0]
        scores = M[i, cand]
        order = cand[np.argsort(-scores)]
        ranks[i] = int(np.where(order == i)[0][0]) + 1
    sizes = (np.full(N, N) if groups is None
             else np.array([int((groups == g).sum()) for g in groups]))
    return {
        "top1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "chance_top1": float(np.mean(1.0 / sizes)),
    }


def align(a_keys, b_keys):
    """Indices aligning two modalities on shared gt_sample."""
    common = sorted(set(a_keys) & set(b_keys))
    ia = [list(a_keys).index(k) for k in common]
    ib = [list(b_keys).index(k) for k in common]
    return np.array(ia), np.array(ib), common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_hidden", required=True)
    ap.add_argument("--equation_hidden")
    ap.add_argument("--trajectory_hidden")
    ap.add_argument("--output_dir", default="probe/results/")
    ap.add_argument("--pool", default="mean_pool", choices=["mean_pool", "last_tok"])
    ap.add_argument("--layers", default=None)
    ap.add_argument("--equation_notation", default="unicode")
    args = ap.parse_args()

    if not (args.equation_hidden or args.trajectory_hidden):
        raise SystemExit("need at least one of --equation_hidden / --trajectory_hidden")

    cd = np.load(args.code_hidden, allow_pickle=True)
    code = {k: cd[k] for k in cd.files}
    Hc = code[args.pool]
    L = Hc.shape[1]
    model_name = str(code["model_name"]) if "model_name" in code else "unknown"
    code_mt = code["mod_types"].astype(str)
    code_gt = code["gt_samples"].astype(str)
    code_pde = code["pde_classes"].astype(str)
    code_src = code["codes"].astype(str)

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(L)))
    rows = []

    def emit(layer, stat, value, **kw):
        r = {"model": model_name, "pool": args.pool, "layer": layer,
             "rel_depth": layer / (L - 1) if L > 1 else 0.0,
             "stat": stat, "value": float(value)}
        r.update(kw)
        rows.append(r)

    modalities = {}
    if args.equation_hidden:
        d = np.load(args.equation_hidden, allow_pickle=True)
        m = {k: d[k] for k in d.files}
        keep = np.where(m["variants"].astype(str) == args.equation_notation)[0]
        if len(keep) == 0:
            raise SystemExit(f"notation {args.equation_notation} not in equation NPZ")
        modalities["equation"] = (m, keep)
        print(f"equation modality: {len(keep)} items "
              f"(notation={args.equation_notation})", flush=True)
    if args.trajectory_hidden:
        d = np.load(args.trajectory_hidden, allow_pickle=True)
        m = {k: d[k] for k in d.files}
        # one trajectory per solver: use the clean, commented VALID condition
        v = m["variants"].astype(str)
        keep = np.where(v == "Comm_Valid")[0]
        if len(keep) == 0:
            keep = np.arange(len(v))
        modalities["trajectory"] = (m, keep)
        print(f"trajectory modality: {len(keep)} items", flush=True)

    # code conditions to test against: the headline plus the plain baseline
    code_conditions = [c for c in (HEADLINE_CODE_CONDITION, "Comm_Valid")
                       if (code_mt == c).any()]
    print(f"code conditions: {code_conditions}", flush=True)

    for mod_name, (m, keep) in modalities.items():
        m_gt = m["gt_samples"].astype(str)[keep]
        m_content = m["contents"].astype(str)[keep]
        Hm = m[args.pool][keep]

        for cond in code_conditions:
            csel = np.where(code_mt == cond)[0]
            ia, ib, common = align(m_gt, code_gt[csel])
            if len(common) < 4:
                print(f"  {mod_name}->{cond}: only {len(common)} shared solvers, "
                      f"skipping", flush=True)
                continue
            groups = code_pde[csel][ib]

            # lexical overlap nuisance, per query/candidate pair
            mw = [word_multiset(s) for s in m_content[ia]]
            cw = [word_multiset(s) for s in code_src[csel][ib]]
            n = len(common)
            lex = np.array([[jaccard(mw[i], cw[j]) for j in range(n)]
                            for i in range(n)])

            lex_r = retrieval(lex)
            lex_rg = retrieval(lex, groups)
            emit(-1, "lexical_baseline_top1", lex_r["top1"],
                 modality=mod_name, code_condition=cond, scope="global")
            emit(-1, "lexical_baseline_top1", lex_rg["top1"],
                 modality=mod_name, code_condition=cond, scope="within_class")

            for layer in layers:
                A = unit(Hm[:, layer][ia].astype(np.float64))
                B = unit(Hc[csel, layer][ib].astype(np.float64))
                M = A @ B.T

                for scope, g in (("global", None), ("within_class", groups)):
                    r = retrieval(M, g)
                    for k, v in r.items():
                        emit(layer, f"retrieval_{k}", v, modality=mod_name,
                             code_condition=cond, scope=scope)

                # does the hit survive removing lexical overlap?
                resid = M - np.polyval(
                    np.polyfit(lex.ravel(), M.ravel(), 1), lex)
                for scope, g in (("global", None), ("within_class", groups)):
                    r = retrieval(resid, g)
                    emit(layer, "retrieval_top1_lexresid", r["top1"],
                         modality=mod_name, code_condition=cond, scope=scope)

                if layer % 5 == 0 or layer == layers[-1]:
                    rw = retrieval(M, groups)
                    print(f"  {mod_name}->{cond} layer {layer:3d}: "
                          f"within-class top1={rw['top1']:.3f} "
                          f"(chance {rw['chance_top1']:.3f})", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    slug = model_name.replace("/", "_")
    out = os.path.join(args.output_dir, f"cross_modal_{slug}_{args.pool}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)", flush=True)

    hl = df[(df.stat == "retrieval_top1") & (df.scope == "within_class") &
            (df.code_condition == HEADLINE_CODE_CONDITION)]
    if len(hl):
        for mod in hl.modality.unique():
            s = hl[hl.modality == mod]
            best = s.loc[s.value.idxmax()]
            ch = df[(df.stat == "retrieval_chance_top1") & (df.modality == mod) &
                    (df.scope == "within_class")]["value"]
            lb = df[(df.stat == "lexical_baseline_top1") & (df.modality == mod) &
                    (df.scope == "within_class") &
                    (df.code_condition == HEADLINE_CODE_CONDITION)]["value"]
            res = df[(df.stat == "retrieval_top1_lexresid") &
                     (df.modality == mod) & (df.scope == "within_class") &
                     (df.code_condition == HEADLINE_CODE_CONDITION) &
                     (df.layer == best.layer)]["value"]
            print(f"\nHEADLINE  {mod} -> {HEADLINE_CODE_CONDITION} (within-class):")
            print(f"  peak top-1 {best.value:.3f} at layer {int(best.layer)} "
                  f"({best.rel_depth:.2f} rel depth), chance "
                  f"{ch.iloc[0] if len(ch) else float('nan'):.3f}")
            print(f"  lexical-only baseline {lb.iloc[0] if len(lb) else float('nan'):.3f}")
            print(f"  after removing lexical overlap "
                  f"{res.iloc[0] if len(res) else float('nan'):.3f}")


if __name__ == "__main__":
    sys.exit(main())
