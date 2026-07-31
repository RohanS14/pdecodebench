"""
Unit tests for probe/code_similarity.py and probe/physics_vs_code.py.
Runs locally, no GPU, no model.

The dataset gives us a free ground truth for the structural regressor:
`NoComm_Valid` and `NoComm_CorrVar` are the SAME program with every author-chosen
identifier renamed. So for each solver:

    ast_ngram distance   must be ~0   (identifiers are stripped)
    token_jaccard distance must be >0 (identifiers are not)

If that separation ever breaks, the variance partitioning loses its ability to
tell "written the same way" from "named the same way", and every physics claim
built on it becomes unsafe.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probe"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from code_similarity import (  # noqa: E402
    ast_ngram_multiset, build_similarity_matrices, jaccard, token_multiset,
)
from physics_vs_code import (  # noqa: E402
    cosine_distance_matrix, ols_r2, partial_mantel, upper,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "merged_mod_jul28.csv")


@pytest.fixture(scope="module")
def pairs():
    """(NoComm_Valid, NoComm_CorrVar) code pairs — pure identifier renames."""
    pd = pytest.importorskip("pandas")
    if not os.path.exists(DATA):
        pytest.skip("jul28 dataset not present")
    df = pd.read_csv(DATA)
    a = df[df.mod_type == "NoComm_Valid"].set_index("gt_sample")["code"]
    b = df[df.mod_type == "NoComm_CorrVar"].set_index("gt_sample")["code"]
    common = sorted(set(a.index) & set(b.index))
    return [(a[k], b[k]) for k in common]


# --- the identifier-stripping property --------------------------------------

def test_ast_profile_is_blind_to_identifier_renaming(pairs):
    for orig, renamed in pairs:
        d = 1.0 - jaccard(ast_ngram_multiset(orig), ast_ngram_multiset(renamed))
        assert d < 1e-9, f"ast_ngram moved under a pure rename (d={d})"


def test_token_profile_is_sensitive_to_identifier_renaming(pairs):
    moved = sum(1.0 - jaccard(token_multiset(o), token_multiset(r)) > 0.01
                for o, r in pairs)
    # obfuscation renames author identifiers, so nearly every solver should move
    assert moved >= 0.8 * len(pairs), f"only {moved}/{len(pairs)} moved lexically"


def test_the_two_regressors_are_not_redundant(pairs):
    """If ast_ngram and token_jaccard were the same thing, partialling would be
    double-counting rather than separating structure from surface."""
    codes = [p[0] for p in pairs]
    sim = build_similarity_matrices(codes)
    r = float(np.corrcoef(upper(sim["ast_ngram"]), upper(sim["token_jaccard"]))[0, 1])
    assert r < 0.98, f"regressors are near-collinear (r={r:.3f})"


# --- similarity primitives --------------------------------------------------

def test_jaccard_bounds_and_identity():
    from collections import Counter
    a, b = Counter("aabbc"), Counter("abbcc")
    assert jaccard(a, a) == 1.0
    assert 0.0 <= jaccard(a, b) <= 1.0
    assert jaccard(Counter(), Counter()) == 1.0


def test_similarity_matrices_are_symmetric_zero_diagonal():
    codes = ["x = 1\ny = x + 1\n", "a = 1\nb = a + 2\n", "for i in range(3):\n    pass\n"]
    for name, M in build_similarity_matrices(codes).items():
        assert np.allclose(M, M.T), f"{name} not symmetric"
        assert np.allclose(np.diag(M), 0), f"{name} has nonzero diagonal"


def test_unparseable_code_does_not_crash():
    m = build_similarity_matrices(["def f(:\n  bad", "x = 1\n"])
    assert np.isfinite(m["ast_ngram"]).all()
    assert np.isfinite(m["token_jaccard"]).all()


# --- variance partitioning --------------------------------------------------

def _class_labels(n=32, k=4):
    return np.array([i % k for i in range(n)])


def test_ols_r2_is_one_for_exact_fit():
    x = np.arange(20.0)
    assert ols_r2(3 * x + 5, x[:, None]) == pytest.approx(1.0, abs=1e-9)


def test_ols_r2_is_zero_for_constant_target():
    assert ols_r2(np.ones(10), np.arange(10.0)[:, None]) == 0.0


def test_partial_mantel_detects_planted_structure():
    """Representation driven purely by class -> large incremental R2, small p."""
    rng = np.random.default_rng(0)
    lab = _class_labels()
    dirs = rng.standard_normal((4, 64))
    X = np.array([5 * dirs[c] + rng.standard_normal(64) for c in lab])
    D_rep = cosine_distance_matrix(X)
    D_class = 1.0 - (lab[:, None] == lab[None, :]).astype(float)
    res = partial_mantel(D_rep, D_class, [], 500, np.random.default_rng(1))
    assert res["incremental_r2"] > 0.5
    assert res["p"] < 0.01


def test_partial_mantel_null_when_no_structure():
    rng = np.random.default_rng(2)
    lab = _class_labels()
    D_rep = cosine_distance_matrix(rng.standard_normal((32, 64)))
    D_class = 1.0 - (lab[:, None] == lab[None, :]).astype(float)
    res = partial_mantel(D_rep, D_class, [], 500, np.random.default_rng(3))
    assert res["p"] > 0.05


def test_partial_mantel_absorbs_a_confounded_nuisance():
    """
    THE test. The representation is driven by a nuisance variable that is itself
    correlated with class. Marginally, class looks predictive. After partialling,
    its incremental R2 must collapse.
    """
    rng = np.random.default_rng(4)
    lab = _class_labels()
    nuis_dirs = rng.standard_normal((4, 64))
    # nuisance tracks class exactly, and the representation is built from nuisance
    X = np.array([5 * nuis_dirs[c] + rng.standard_normal(64) for c in lab])
    D_rep = cosine_distance_matrix(X)
    D_class = 1.0 - (lab[:, None] == lab[None, :]).astype(float)
    D_nuis = D_class.copy()

    marginal = partial_mantel(D_rep, D_class, [], 300, np.random.default_rng(5))
    partialled = partial_mantel(D_rep, D_class, [D_nuis], 300, np.random.default_rng(5))
    assert marginal["incremental_r2"] > 0.5, "setup failed: no marginal signal"
    assert partialled["incremental_r2"] < 0.01, "nuisance was not absorbed"


def test_permutation_preserves_matrix_structure_not_pair_order():
    """
    The Mantel null must permute SOLVERS, not the 496 upper-triangle entries.
    Permuting entries destroys the dependency between pairs sharing a solver and
    yields a null that is far too narrow. Guard: a permuted target matrix must
    still be a valid symmetric distance matrix.
    """
    rng = np.random.default_rng(6)
    lab = _class_labels()
    D_class = 1.0 - (lab[:, None] == lab[None, :]).astype(float)
    p = rng.permutation(32)
    Dp = D_class[np.ix_(p, p)]
    assert np.allclose(Dp, Dp.T)
    assert np.allclose(np.diag(Dp), 0)
    assert sorted(upper(Dp).tolist()) == sorted(upper(D_class).tolist())
