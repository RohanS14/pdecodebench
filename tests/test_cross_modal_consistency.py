"""
Unit tests for the cross-modal consistency experiment (plan Part III).

Runs locally, no GPU, no model. The point is to catch a silently wrong answer
before any GPU time is spent, so the emphasis is on the places where this design
could look fine and be broken:

  * T_shuf must preserve the valid trajectory's ENTIRE value multiset. That is the
    whole reason the rung exists -- it is the control that no position-blind
    statistic can pass -- and a shuffle that quietly altered the values would turn
    the sharpest control in the design into a second T_rand.
  * All four rungs must render to the same size. Otherwise the outlier is findable
    from the shape of the table, with no physics involved.
  * Detection must be scored as d', not accuracy. Seven of eight conditions are
    corrupted, so "always say no" scores 0.875 while being worthless.
  * Localization must be undefined, not zero, where the model failed to detect.
    Scoring it zero conflates missing the corruption with finding it and pointing
    at the wrong view -- the two things the corruption ladder exists to separate.
  * A thinking model that hits its token limit emits an unclosed <think> and never
    answers. That must parse as a failure, not as whatever text the trace ended on.
"""
import ast
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from crossmodal.datagen.corrupt_trajectory import (  # noqa: E402
    build_ladder, decimate_frames, make_random, make_shuffled, make_time_shuffled,
)
from crossmodal.datagen.render_trajectory_table import (  # noqa: E402
    choose_grid, render, reconstruction_error, resample,
)
from crossmodal.eval.parse_consistency import (  # noqa: E402
    detection_dprime, detection_summary, dprime, parse_consistency,
    score_consistency, strip_think,
)


def smooth_field(T=10, X=64, Y=1, C=1, seed=0):
    """A synthetic trajectory with real spatial and temporal structure -- a
    travelling, decaying Gaussian pulse, so downsampling has something to lose."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, X)
    y = np.linspace(0, 1, Y) if Y > 1 else np.array([0.5])
    out = np.zeros((T, X, Y, C))
    for t in range(T):
        centre = 0.2 + 0.6 * t / max(1, T - 1)
        prof = np.exp(-((x - centre) ** 2) / (2 * 0.05 ** 2))
        for c in range(C):
            out[t, :, :, c] = np.outer(prof, np.exp(-((y - 0.5) ** 2) / 0.1)) * (1 + c)
    return out + 1e-6 * rng.standard_normal(out.shape)


WRONG_TEXT = repr(smooth_field(T=10, X=33, seed=9).tolist())


# --------------------------------------------------------------- corruption ladder

def test_shuffle_preserves_the_entire_value_multiset():
    valid = smooth_field()
    shuf = make_shuffled(valid, "Sys_1")
    assert shuf.shape == valid.shape
    assert np.array_equal(np.sort(shuf.ravel()), np.sort(valid.ravel()))
    # Every position-blind statistic is therefore identical, which is the property
    # that forces a model to read arrangement rather than summary statistics.
    for stat in (np.mean, np.std, np.min, np.max, np.median):
        assert stat(shuf) == pytest.approx(stat(valid))
    assert not np.array_equal(shuf, valid)


def test_random_matches_shape_and_moments_but_not_structure():
    valid = smooth_field()
    rand = make_random(valid, "Sys_1")
    assert rand.shape == valid.shape
    assert rand.mean() == pytest.approx(valid.mean(), abs=0.05 * (valid.std() + 1e-9))
    assert rand.std() == pytest.approx(valid.std(), rel=0.2)
    # Structure, not just moments, is what should differ: neighbouring cells in the
    # valid field are correlated and in the random one they are not.
    def roughness(a):
        return float(np.abs(np.diff(a, axis=1)).mean())
    assert roughness(rand) > 5 * roughness(valid)


def test_time_shuffle_keeps_frames_intact_but_reorders_them():
    valid = smooth_field()
    ts = make_time_shuffled(valid, "Sys_1")
    assert ts.shape == valid.shape
    # Each frame survives untouched; only the sequence is non-causal.
    for frame in ts:
        assert any(np.array_equal(frame, v) for v in valid)
    assert not np.array_equal(ts, valid)


def test_ladder_draws_are_deterministic_and_system_specific():
    valid = smooth_field()
    a = build_ladder(valid, WRONG_TEXT, "Sys_1")
    b = build_ladder(valid, WRONG_TEXT, "Sys_1")
    c = build_ladder(valid, WRONG_TEXT, "Sys_2")
    for level in ("T_rand", "T_shuf"):
        assert np.array_equal(a[level], b[level]), f"{level} not reproducible"
        assert not np.array_equal(a[level], c[level]), f"{level} shared across systems"


def test_decimate_matches_the_dataset_frame_convention():
    """The stored trajectories are linspace(0, N-1, 10) over the whole run --
    verified against Heat_1, whose frames are indices 0,111,...,1000 of 1001. T_exec
    arrives from re-execution and must land on that same convention."""
    hist = np.arange(1001).reshape(1001, 1, 1, 1).astype(float)
    got = decimate_frames(hist, 10)
    assert got.shape[0] == 10
    assert [int(v) for v in got[:, 0, 0, 0]] == [0, 111, 222, 333, 444, 556, 667, 778, 889, 1000]


def test_decimate_is_a_noop_when_history_is_already_short():
    hist = smooth_field(T=6)
    assert np.array_equal(decimate_frames(hist, 10), hist)


# ------------------------------------------------------------------- rendering

def test_every_rung_renders_to_identical_size():
    valid = smooth_field()
    ladder = build_ladder(valid, WRONG_TEXT, "Sys_1", include_time_shuffle=True)
    grid = choose_grid(valid.shape)
    lengths = {len(render(a, grid)) for a in [valid] + list(ladder.values())}
    assert len(lengths) == 1, f"rendered sizes differ: {sorted(lengths)}"


def test_render_size_is_independent_of_sign():
    """Negative numbers must not make the table longer, or the character count
    would track content and give the outlier away."""
    grid = (4, 8, 1, 1)
    pos = np.abs(smooth_field(T=4, X=8))
    assert len(render(pos, grid)) == len(render(-pos, grid))


def test_swap_of_a_different_shape_still_renders_to_the_receiver_grid():
    """30 of 32 delivered swaps have a different shape from their valid twin and 14
    flip 1-D <-> 2-D. Resampling onto the receiver's grid is what stops that from
    being a free answer."""
    valid = smooth_field(T=10, X=64)
    donor = smooth_field(T=10, X=41, Y=41, C=2, seed=3)
    grid = choose_grid(valid.shape)
    assert resample(donor, grid).shape == resample(valid, grid).shape
    assert len(render(donor, grid)) == len(render(valid, grid))


def test_reconstruction_error_falls_as_resolution_rises():
    """The metric that calibrates the grid must be monotone in resolution.
    Counting local extrema, tried first, is not: its denominator collapses on
    near-monotone fields and aliasing can inflate the count as easily as reduce it."""
    a = smooth_field(T=10, X=128)
    errs = [reconstruction_error(a, (10, n, 1, 1)) for n in (8, 16, 32, 64)]
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < errs[0]


def test_render_never_drops_a_channel():
    a = smooth_field(T=10, X=41, Y=41, C=2)
    grid = choose_grid(a.shape)
    assert grid[3] == 2
    assert "component=1" in render(a, grid)


# --------------------------------------------------------------------- parsing

def test_unclosed_think_block_is_a_parse_failure_not_a_guess():
    out = parse_consistency("<think>I will start by examining view 1, which appears")
    assert out["parse_route"] == "failed"
    assert out["agree"] is None and out["outlier"] is None


def test_closed_think_block_is_stripped_before_parsing():
    text = ('<think>view 3 looks wrong to me</think>'
            '{"agree":"no","outlier":"view_2","system_pde_class":"heat",'
            '"system_num_method":"implicit","justification":"x"}')
    out = parse_consistency(text)
    assert out["parse_route"] == "json"
    assert out["outlier"] == "view_2"          # not view_3 from the discarded trace
    assert "view 3" not in (out["justification"] or "")


def test_strip_think_handles_the_three_shapes():
    assert strip_think("<think>a</think>answer") == "answer"
    assert strip_think("trace</think>answer") == "answer"
    assert strip_think("<think>never closed") == ""


def test_protocol_violations_are_flagged_both_ways():
    agreed_but_named = parse_consistency(
        '{"agree":"yes","outlier":"view_2","system_pde_class":"heat",'
        '"system_num_method":"implicit","justification":"x"}')
    assert agreed_but_named["protocol_violation"] == "agreed_but_named_outlier"

    disagreed_without = parse_consistency(
        '{"agree":"no","outlier":"none","system_pde_class":"heat",'
        '"system_num_method":"implicit","justification":"x"}')
    assert disagreed_without["protocol_violation"] == "disagreed_without_outlier"


def test_invalid_enum_values_are_rejected_rather_than_passed_through():
    out = parse_consistency(
        '{"agree":"maybe","outlier":"view_9","system_pde_class":"heat",'
        '"system_num_method":"implicit","justification":"x"}')
    assert out["agree"] is None
    assert out["outlier"] is None


# --------------------------------------------------------------------- scoring

CORRUPTED = {"corrupted_view": "math", "outlier_slot": "2",
             "gt_pde_class": "heat", "gt_num_method": "implicit"}
CLEAN = {"corrupted_view": "none", "outlier_slot": "",
         "gt_pde_class": "heat", "gt_num_method": "implicit"}


def _resp(agree, outlier="none"):
    return parse_consistency(
        f'{{"agree":"{agree}","outlier":"{outlier}","system_pde_class":"heat equation",'
        f'"system_num_method":"crank-nicolson","justification":"x"}}')


def test_localization_is_undefined_when_detection_failed():
    """Not zero. A model that missed the corruption never had the chance to
    localize, and folding that into the localization rate would make the corruption
    ladder unreadable."""
    s = score_consistency(_resp("yes"), CORRUPTED)
    assert s["detection_correct"] == 0
    assert s["localization_correct"] is None


def test_localization_scores_only_the_named_slot():
    assert score_consistency(_resp("no", "view_2"), CORRUPTED)["localization_correct"] == 1
    assert score_consistency(_resp("no", "view_3"), CORRUPTED)["localization_correct"] == 0


def test_detection_is_scored_on_both_kinds_of_item():
    assert score_consistency(_resp("no", "view_2"), CORRUPTED)["detection_correct"] == 1
    assert score_consistency(_resp("yes"), CLEAN)["detection_correct"] == 1
    assert score_consistency(_resp("no", "view_1"), CLEAN)["detection_correct"] == 0


def test_system_identification_uses_the_existing_alias_tables():
    s = score_consistency(_resp("no", "view_2"), CORRUPTED)
    assert s["pde_class_match"] == 1        # "heat equation" -> heat
    assert s["num_method_match"] == 1       # "crank-nicolson" -> implicit


def test_always_saying_no_looks_accurate_and_is_flagged_degenerate():
    """The reason detection is never reported as accuracy, and never as a bare d'.

    With 7 of 8 conditions corrupted, a model that never agrees is 87.5% accurate
    and has learned nothing. Its hit rate and false-alarm rate are both exactly 1,
    so the underlying d' is 0 -- but the log-linear correction cannot express that
    at unequal n (896 signal vs 128 noise here), and returns a spuriously positive
    number. So the summary carries both rates and a degenerate flag, and that flag
    is what a reader must check before quoting d'.
    """
    scored = ([score_consistency(_resp("no", "view_2"), CORRUPTED)] * 7 +
              [score_consistency(_resp("no", "view_1"), CLEAN)] * 1)
    accuracy = sum(s["detection_correct"] for s in scored) / len(scored)
    assert accuracy == pytest.approx(0.875)

    summary = detection_summary(scored)
    assert summary["hit_rate"] == 1.0
    assert summary["false_alarm_rate"] == 1.0
    assert summary["degenerate"] is True


def test_always_saying_yes_is_also_flagged_degenerate():
    scored = ([score_consistency(_resp("yes"), CORRUPTED)] * 7 +
              [score_consistency(_resp("yes"), CLEAN)] * 1)
    summary = detection_summary(scored)
    assert summary["hit_rate"] == 0.0
    assert summary["false_alarm_rate"] == 0.0
    assert summary["degenerate"] is True


def test_a_discriminating_responder_is_not_flagged():
    scored = ([score_consistency(_resp("no", "view_2"), CORRUPTED)] * 7 +
              [score_consistency(_resp("yes"), CLEAN)] * 1)
    summary = detection_summary(scored)
    assert summary["degenerate"] is False
    assert summary["dprime"] > 1.0


def test_dprime_is_zero_at_chance_regardless_of_class_imbalance():
    """Chance performance must read as zero even at the 7:1 imbalance this design
    has, or every model would look better than it is."""
    assert dprime(448, 896, 64, 128) == pytest.approx(0.0, abs=1e-9)
    assert dprime(50, 100, 50, 100) == pytest.approx(0.0, abs=1e-9)


def test_dprime_ordering_and_finiteness():
    assert dprime(90, 100, 10, 100) > dprime(70, 100, 30, 100) > 0
    assert dprime(20, 100, 80, 100) < 0
    # Perfect performance must stay finite, or the headline number becomes an
    # artifact of sample size rather than a measurement.
    assert np.isfinite(dprime(100, 100, 0, 100))


def test_summary_is_none_without_both_item_kinds():
    only_corrupted = [score_consistency(_resp("no", "view_2"), CORRUPTED)] * 5
    assert detection_summary(only_corrupted) is None
    assert detection_dprime(only_corrupted) is None


# ------------------------------------------------------------------ instrumentation

from crossmodal.datagen.instrument_history import (  # noqa: E402
    INSTRUMENT_SPEC, NOT_INSTRUMENTABLE, assert_write_only, find_time_loop,
    instrument, stack_history,
)

LOOP_SRC = """
import numpy as np
u = np.zeros(10)
for n in range(5):
    u[1:-1] = u[1:-1] + 1.0
"""

DECOY_SRC = """
import numpy as np
u = np.zeros(10)
p = np.zeros(10)
def solve():
    for n in range(5):
        for q in range(3):
            p[:] = p + 1.0
        u[1:-1] = u[1:-1] + 1.0
"""


def test_instrumentation_is_write_only():
    out = instrument(LOOP_SRC, "Heat_5")
    assert assert_write_only(out)
    assert "_HISTORY.append([u.copy()])" in out


def test_a_recorder_that_is_read_is_rejected():
    """The write-only proof is the whole safety argument -- if the recorder can be
    read, the instrumentation has entered the computation and the trajectory is no
    longer the one the untouched solver produces."""
    with pytest.raises(AssertionError):
        assert_write_only("_HISTORY = []\n_HISTORY.append(1)\nx = _HISTORY[0]\n")
    with pytest.raises(AssertionError):
        assert_write_only("_HISTORY = []\nfor v in _HISTORY:\n    pass\n")


def test_outer_time_loop_is_chosen_over_an_inner_decoy():
    """Both NavierStokes solvers wrap a pressure-Poisson loop inside the time loop.
    It iterates and mutates an array, so a naive 'find a loop that mutates something'
    rule would record 50 sub-iterations per timestep instead of the trajectory."""
    tree = ast.parse(DECOY_SRC)
    loop = find_time_loop(tree, ["u"], function="solve")
    assert isinstance(loop.target, ast.Name) and loop.target.id == "n"


def test_spec_targets_only_real_identifier_variants():
    """Obfuscated variants are AST-identical to their real-name twins and produce
    the same numbers, so executing them would burn CPU reproducing a known result --
    and their fields are renamed, so the spec would not match anyway."""
    for spec in INSTRUMENT_SPEC.values():
        assert all(not f.startswith("foobar_") for f in spec["fields"])


def test_heat_8_is_refused_rather_than_silently_mishandled():
    """It has no time loop -- spectral, closed form at t_final. Producing frames
    would mean evaluating its solution operator at ten times, which is a different
    program, so T_exec simply does not exist for it."""
    assert "Heat_8" in NOT_INSTRUMENTABLE
    assert "Heat_8" not in INSTRUMENT_SPEC
    with pytest.raises(ValueError, match="not instrumentable"):
        instrument("import numpy as np\nu = np.zeros(3)\n", "Heat_8")


def test_stack_history_produces_the_dataset_layout():
    scalar = [[np.zeros(7)] for _ in range(4)]
    assert stack_history(scalar).shape == (4, 7, 1, 1)
    vector = [[np.zeros((5, 6)), np.ones((5, 6))] for _ in range(4)]
    assert stack_history(vector).shape == (4, 5, 6, 2)


def test_stack_history_refuses_an_empty_run():
    with pytest.raises(ValueError, match="never ran"):
        stack_history([])


# ------------------------------------------------------------------ k sampling

def test_checkpoint_counts_samples_so_a_partial_item_is_rerun(tmp_path):
    """With K draws per item, resume must not treat 1-of-3 as done.

    A set-based checkpoint would, leaving a ragged K across the dataset that nothing
    would flag -- some items with 3 draws, some with 1, and a pooled rate silently
    weighting them unequally.
    """
    from crossmodal.eval.run_cross_modal_consistency import load_checkpoint
    import json as _json
    p = tmp_path / "out.jsonl"
    rows = [{"item_id": "A", "model": "m", "thinking": "on"},
            {"item_id": "A", "model": "m", "thinking": "on"},
            {"item_id": "B", "model": "m", "thinking": "on"}]
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    done = load_checkpoint(str(p))
    assert done[("A", "m", "on")] == 2
    assert done[("B", "m", "on")] == 1
    # under k=3 both are unfinished; under k=1 both are finished
    assert [i for i in ("A", "B") if done.get((i, "m", "on"), 0) < 3] == ["A", "B"]
    assert [i for i in ("A", "B") if done.get((i, "m", "on"), 0) < 1] == []


def test_sampling_params_carry_k_and_the_models_own_settings():
    """The protocol is 'each model at the settings its authors specify'. Greedy is
    explicitly contraindicated for these reasoning models -- at temperature 0 they
    emit endless repetition -- so a silently dropped temperature would reintroduce
    exactly the failure this change exists to remove."""
    pytest.importorskip("vllm")   # library-backed; runs on the cluster, skips locally
    from crossmodal.eval.run_cross_modal_consistency import build_sampling_params
    sp = build_sampling_params("prompt_only", 32768,
                               gen={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                               n=3, seed=42)
    assert sp.n == 3
    assert sp.temperature == 0.6 and sp.top_p == 0.95 and sp.top_k == 20
    assert sp.seed == 42
    assert sp.max_tokens == 32768


def test_top_k_zero_is_dropped_not_passed_through():
    """HF writes top_k=0 for 'disabled'; vLLM rejects 0 and wants it absent."""
    pytest.importorskip("transformers")
    from crossmodal.eval.run_cross_modal_consistency import recommended_sampling
    import unittest.mock as _m
    class G:
        temperature, top_p, top_k = 1.0, 1.0, 0
    with _m.patch("transformers.GenerationConfig.from_pretrained", return_value=G()):
        got = recommended_sampling("fake/model")
    assert "top_k" not in got and got["temperature"] == 1.0
