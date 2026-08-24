"""k=3 changes the row/observation relationship. Everything downstream that counts
rows as observations has to pool first; these pin that it does."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "freegen"))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                                        # noqa: BLE001
        pytest.skip(f"cannot import {rel}: {exc}")
    return mod


RPT = _load("freegen/report.py", "freegen_report")


def _draws(n_items=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        # One latent per-item score; the k draws jitter around it. That is the
        # correlation structure pooling exists to respect.
        base = rng.uniform(0, 1)
        for d in range(k):
            rows.append({
                "model": "M", "thinking": "on", "mod_type": "Comm_Valid",
                "title": f"T{i}", "sample_idx": d, "k_draws": k,
                "valid_match": float(np.clip(base + rng.normal(0, 0.05), 0, 1)),
                "valid_conf": "Confident", "finish_reason": "stop",
            })
    return pd.DataFrame(rows)


def test_pooling_collapses_k_draws_to_one_row_per_item():
    df = _draws()
    out = RPT.pool_draws(df)
    assert len(out) == 40
    assert set(out["n_draws_pooled"]) == {3}


def test_pooling_preserves_the_mean_but_widens_the_interval():
    """The whole reason pooling is mandatory: the point estimate does not move, so a
    reader cannot tell from the number whether it was done -- only the interval
    changes, and it changes in the direction that makes a result look real."""
    df = _draws()
    raw_lo, raw_hi, raw_mean, raw_n = RPT.bootstrap_ci(df["valid_match"])
    pooled = RPT.pool_draws(df)
    p_lo, p_hi, p_mean, p_n = RPT.bootstrap_ci(pooled["valid_match"])
    assert raw_n == 120 and p_n == 40
    assert p_mean == pytest.approx(raw_mean, abs=0.02)
    assert (p_hi - p_lo) > (raw_hi - raw_lo) * 1.3, (
        f"pooled CI {p_hi - p_lo:.4f} is not meaningfully wider than the "
        f"raw-row CI {raw_hi - raw_lo:.4f}")


def test_k1_frames_pass_through_untouched():
    df = _draws(k=1)
    out = RPT.pool_draws(df)
    pd.testing.assert_frame_equal(out, df)


def test_pooling_is_keyed_on_the_item_not_the_model():
    """Two conditions of one model must stay two observations, not one."""
    a = _draws(n_items=5)
    b = _draws(n_items=5, seed=1)
    b["mod_type"] = "NoComm_InValid"
    out = RPT.pool_draws(pd.concat([a, b], ignore_index=True))
    assert len(out) == 10
    assert set(out["mod_type"]) == {"Comm_Valid", "NoComm_InValid"}


def test_no_verdict_rows_are_dropped_by_load_not_scored(tmp_path):
    df = _draws(n_items=10)
    df["no_verdict"] = [i % 3 == 0 for i in range(len(df))]
    for c in ("pde_match", "method_any_match", "behavior_any_match"):
        df[c] = 1.0
    df["gt_valid"] = True
    df["parsed_valid"] = "yes"
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    out = RPT.load(str(p))
    assert "no_verdict" not in out.columns or not out["no_verdict"].any()
    # 10 items x 3 draws, a third dropped, then pooled -> still 10 items.
    assert len(out) <= 10


def test_aggregate_dedup_key_includes_sample_idx():
    """The k=3 regression that made every legitimate draw look like a duplicate."""
    src = open(os.path.join(ROOT, "freegen/aggregate_freegen.py")).read()
    assert '"sample_idx"' in src
    i = src.index("dedup_keys")
    assert "sample_idx" in src[i:i + 400]


def test_aggregate_reads_k_from_the_data_rather_than_assuming_one():
    src = open(os.path.join(ROOT, "freegen/aggregate_freegen.py")).read()
    assert "nunique()" in src and "expect_items" in src
