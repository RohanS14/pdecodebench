"""The free-generation re-run exists to be compared per model against the
cross-representation consistency arms. That comparison is only meaningful if both
sides run the same checkpoints under the same decoding, so both are pinned here
rather than left to drift the next time either file is edited."""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "freegen_static_judgments"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                                        # noqa: BLE001
        pytest.skip(f"cannot import {path}: {exc}")
    return mod


RE = _load("freegen_static_judgments/run_eval.py", "freegen_run_eval")
XM = _load("cross_modal_consistency/eval/run_cross_modal_consistency.py", "xmodal_runner")

def _consistency_roster():
    """Derived from the consistency runner, not typed out here.

    The hardcoded literal below is kept only as a tripwire: if the runner's roster
    changes, the assertion in test_roster_is_derived_not_asserted fails and someone
    has to look. A test that pins a copy of the thing it is guarding cannot detect
    the drift it exists to detect.
    """
    return set(getattr(XM, "TOGGLE_THINKING_MODELS", set())
               | getattr(XM, "TOGGLEABLE", set())
               | getattr(XM, "ALWAYS_THINKING_MODELS", set())
               | getattr(XM, "ALWAYS_THINKING", set()))


# The eight checkpoints the consistency experiment ran.
CONSISTENCY_ROSTER = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-32B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "zai-org/GLM-4.7-Flash",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.8-27B",
]


def test_decoding_regime_matches_the_consistency_runner():
    assert RE.UNIFORM_SAMPLING == XM.UNIFORM_SAMPLING, (
        "free generation and the consistency arms must decode identically; "
        f"{RE.UNIFORM_SAMPLING} vs {XM.UNIFORM_SAMPLING}")


def test_temperature_is_not_greedy():
    """Greedy is a measured failure mode for this roster -- Nemotron looped on 30 of
    its first 64 consistency items at temperature 0."""
    assert RE.UNIFORM_SAMPLING["temperature"] > 0


def test_every_consistency_checkpoint_is_configured_for_the_thinking_arm():
    known = RE.TOGGLE_THINKING_MODELS | RE.ALWAYS_THINKING_MODELS
    missing = [m for m in CONSISTENCY_ROSTER if m not in known]
    assert not missing, f"cannot run a thinking arm for: {missing}"
    for m in CONSISTENCY_ROSTER:
        assert RE.resolve_thinking(m, "on") == "on", m


def test_token_budgets_mirror_the_consistency_runner():
    for m in CONSISTENCY_ROSTER:
        theirs = XM.gen_budget(m, "on")
        ours = RE.arm_max_tokens(m, "on")
        assert ours == theirs, f"{m}: freegen_static_judgments {ours} vs consistency {theirs}"


def test_k_draws_matches():
    assert RE.K_DRAWS == 3


@pytest.mark.parametrize("text,finish,expected", [
    ("pde: heat", "length", True),               # truncated: no answer reached
    ("<think>unclosed reasoning", "stop", True),  # never closed the block
    ("<think>x</think>pde: heat", "stop", False),
    ("reasoning</think>pde: heat", "stop", False),  # no opening tag, still fine
    ("pde: heat", "stop", False),
])
def test_no_verdict_detection(text, finish, expected):
    from parse_score import is_no_verdict
    assert is_no_verdict(text, finish) is expected


def test_no_verdict_is_actually_wired_into_the_runner():
    """It was defined and never called once already. The output row must carry it."""
    src = open(os.path.join(ROOT, "freegen_static_judgments/run_eval.py")).read()
    assert "is_no_verdict(text, finish_reason)" in src
    assert '"no_verdict":' in src
    assert '"sample_idx":' in src


def test_roster_is_derived_not_asserted():
    """Every model this test file pins must be one the consistency runner knows."""
    known = _consistency_roster()
    if not known:
        pytest.skip("consistency runner exposes no roster sets to compare against")
    missing = [m for m in CONSISTENCY_ROSTER if m not in known]
    assert not missing, (
        "these are pinned here but the consistency runner does not list them, so the "
        f"two rosters have drifted: {missing}")


def test_sampling_seed_matches_the_consistency_default():
    """They happen to be equal; nothing held them there until now."""
    theirs = getattr(XM, "SAMPLING_SEED", None) or getattr(XM, "DEFAULT_SEED", None)
    if theirs is None:
        import re
        src = open(os.path.join(ROOT, "cross_modal_consistency/eval/run_cross_modal_consistency.py")).read()
        # The default is wrapped in an env lookup:
        #   default=int(os.environ.get("SAMPLING_SEED", "20260821"))
        # so the digits are inside the quoted fallback, not bare after `default=`.
        m = re.search(r'--seed"[^)]*?SAMPLING_SEED"\s*,\s*"(\d+)"', src, re.S)
        if not m:
            pytest.skip("cannot locate the consistency runner's seed default")
        theirs = int(m.group(1))
    assert RE.SAMPLING_SEED == int(theirs)


def test_k_actually_reaches_sampling_params():
    """C1's lesson: a constant that exists and is never wired is not a fix. K_DRAWS
    being 3 says nothing unless n=K_DRAWS reaches the engine."""
    src = open(os.path.join(ROOT, "freegen_static_judgments/run_eval.py")).read()
    assert "n=K_DRAWS" in src
    assert "**UNIFORM_SAMPLING" in src


def test_no_verdict_rows_are_excluded_downstream_not_merely_flagged():
    """The column existing is not the fix; something has to read it."""
    rpt = open(os.path.join(ROOT, "freegen_static_judgments/report.py")).read()
    assert "no_verdict" in rpt and "~nv" in rpt
    agg = open(os.path.join(ROOT, "freegen_static_judgments/aggregate_freegen.py")).read()
    assert "no_verdict" in agg


# ── Experiment 1 in the dual report must pool k draws ─────────────────────────
# pde_dual_report.py was written when Experiment 1 was k=1, so every row was one
# item and n was just len(df). The roster now runs k=3 to match the consistency
# arms. Handing that frame to the old code triples every n: the point estimates
# are unchanged and every interval narrows by about sqrt(3), which is 42% in the
# direction that makes a result look real.

import os
import pandas as pd

_VIZ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dual_report_source():
    with open(os.path.join(_VIZ_ROOT, "viz", "pde_dual_report.py")) as f:
        return f.read()


def test_dual_report_imports_the_one_pooler():
    src = _dual_report_source()
    assert "from freegen_static_judgments.report import pool_draws" in src, (
        "pde_dual_report.py must import pool_draws rather than reimplement it -- a "
        "second copy lets Experiment 1 and the free-gen report drift apart while "
        "still looking comparable in one document")
    assert "pool_draws(df)" in src, "pool_draws is imported but never applied"


def test_pooling_collapses_k_draws_and_preserves_the_mean():
    from freegen_static_judgments.report import pool_draws
    rows = []
    for item in range(4):
        for draw, score in enumerate((1.0, 0.0, 1.0)):
            rows.append({"model": "m", "thinking": "on", "mod_type": "Comm_Valid",
                         "title": f"t{item}", "sample_idx": draw, "k_draws": 3,
                         "valid_match": score})
    df = pd.DataFrame(rows)
    out = pool_draws(df)
    assert len(out) == 4, f"12 draws of 4 items must pool to 4 items, got {len(out)}"
    assert abs(out["valid_match"].mean() - df["valid_match"].mean()) < 1e-9, (
        "pooling must not move the point estimate, only the n behind it")


def test_pooling_is_a_noop_on_k1_so_older_csvs_still_work():
    from freegen_static_judgments.report import pool_draws
    df = pd.DataFrame([{"model": "m", "thinking": "on", "mod_type": "Comm_Valid",
                        "title": f"t{i}", "valid_match": 1.0} for i in range(5)])
    assert len(pool_draws(df)) == 5


def test_every_dual_report_panel_returns_rendered_markup():
    """A panel's third element must be an HTML STRING, not a plotly Figure.

    The renderer drops the tuple's third element into the document as-is. Returning
    the Figure object instead of fig_html(fig) raises nothing and writes no warning
    -- the panel's title and caption appear in the page and the chart is simply
    absent. Caught only by looking at the report, which is exactly what happened.
    """
    import re
    src = _dual_report_source()
    # Panel builders end with `return [( "title", "caption", <third> )]`; the third
    # element is what matters. Any bare `fig)]` is a Figure escaping unrendered.
    offenders = [m.start() for m in re.finditer(r"\n\s+fig\)\]", src)]
    assert not offenders, (
        f"{len(offenders)} panel(s) return a raw Figure instead of fig_html(fig); "
        f"they render as a title and caption with no chart")
