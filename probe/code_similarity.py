"""
Code-similarity nuisance regressors for the physics-vs-code dissociation.

These exist so that "representations cluster by pde_class" can be turned into a
claim about physics. Without them the clustering is uninterpretable: solvers of
the same PDE are also written similarly, so lexical and structural overlap is a
complete alternative explanation for any clustering we observe.

Three measures, deliberately capturing different things:

    token_jaccard  surface lexical overlap  (identifiers and literals included)
    ast_ngram      structural overlap       (identifiers STRIPPED — pure shape)
    len_diff       code size difference

`ast_ngram` strips identifier names on purpose: it is the regressor that stays
high between two solvers written the same way but named differently, which is
exactly the confound `NoComm_CorrVar` does *not* remove.
"""
import ast
import io
import tokenize
from collections import Counter

import numpy as np


def token_multiset(code: str) -> Counter:
    """Surface tokens: names, numbers, operators. Comments and strings dropped."""
    out = Counter()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in (tokenize.NAME, tokenize.NUMBER, tokenize.OP):
                out[tok.string] += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Fall back to whitespace split rather than failing the whole run
        out.update(code.split())
    return out


def ast_ngram_multiset(code: str, n: int = 3) -> Counter:
    """
    Multiset of node-TYPE n-grams along root-to-leaf paths. Identifier names,
    attribute names and literal values never enter — only the shape of the tree.

    This is a fast stand-in for tree edit distance, which is O(n^4) exact and far
    too slow for the pairwise matrix we need. Named honestly: it is an n-gram
    profile, not an edit distance.
    """
    out = Counter()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return out

    def walk(node, path):
        path = (path + [type(node).__name__])[-n:]
        if len(path) == n:
            out[tuple(path)] += 1
        for child in ast.iter_child_nodes(node):
            walk(child, path)

    walk(tree, [])
    return out


def jaccard(a: Counter, b: Counter) -> float:
    """Weighted (min/max) Jaccard over two multisets. 1.0 = identical."""
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


def build_similarity_matrices(codes: list) -> dict:
    """
    Return dict of name -> (S, S) matrix. Similarities are converted to
    DISTANCES so every regressor points the same way as the representational
    distance being modelled (larger = more different).
    """
    S = len(codes)
    tok = [token_multiset(c) for c in codes]
    ast_ = [ast_ngram_multiset(c) for c in codes]
    lens = np.array([len(c) for c in codes], dtype=float)

    token_d = np.zeros((S, S))
    ast_d = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            token_d[i, j] = token_d[j, i] = 1.0 - jaccard(tok[i], tok[j])
            ast_d[i, j] = ast_d[j, i] = 1.0 - jaccard(ast_[i], ast_[j])

    len_d = np.abs(lens[:, None] - lens[None, :])
    len_d = len_d / len_d.max() if len_d.max() > 0 else len_d

    n_empty = sum(1 for a in ast_ if not a)
    if n_empty:
        print(f"  WARNING: {n_empty}/{S} solvers produced an empty AST profile "
              f"(parse failure) — their ast_ngram distances are degenerate.",
              flush=True)

    return {"token_jaccard": token_d, "ast_ngram": ast_d, "len_diff": len_d}
