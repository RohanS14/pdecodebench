"""Tests for the reasoning-arm resolution in freegen/run_eval.py.

These exist because the previous plumbing failed silently: a set named
THINKING_MODELS was used to set `enable_thinking=False`, and MODEL_CONFIGS carried
a `thinking` key nothing read. Qwen3-32B was configured as a thinking model, run as
a non-thinking one, and the output had no column that would have revealed it. Every
test below is aimed at that class of failure -- an arm that is not what it says.
"""
import os
import sys

import pytest

sys.path[:0] = [
    os.path.join(os.path.dirname(__file__), ".."),
    os.path.join(os.path.dirname(__file__), "..", "eval"),
    os.path.join(os.path.dirname(__file__), "..", "freegen"),
]

from freegen.run_eval import (  # noqa: E402
    ALWAYS_THINKING_MODELS, MODEL_CONFIGS, TOGGLE_THINKING_MODELS,
    arm_max_tokens, get_model_config, resolve_thinking)


def test_a_model_cannot_be_both_toggleable_and_always_thinking():
    assert not (TOGGLE_THINKING_MODELS & ALWAYS_THINKING_MODELS)


@pytest.mark.parametrize("model", sorted(ALWAYS_THINKING_MODELS))
def test_always_thinking_models_refuse_an_off_arm(model):
    """Silently returning 'on' for --thinking off would mislabel the arm."""
    with pytest.raises(SystemExit):
        resolve_thinking(model, "off")
    assert resolve_thinking(model, "auto") == "on"
    assert resolve_thinking(model, "on") == "on"


@pytest.mark.parametrize("model", sorted(TOGGLE_THINKING_MODELS))
def test_toggle_models_honour_both_arms(model):
    assert resolve_thinking(model, "on") == "on"
    assert resolve_thinking(model, "off") == "off"


def test_non_reasoning_model_refuses_an_on_arm():
    with pytest.raises(SystemExit):
        resolve_thinking("Qwen/Qwen3-Coder-30B-A3B-Instruct", "on")
    assert resolve_thinking("Qwen/Qwen3-Coder-30B-A3B-Instruct", "auto") == "off"


def test_unknown_model_defaults_off_and_refuses_on():
    assert resolve_thinking("meta-llama/Llama-3.3-70B-Instruct", "auto") == "off"
    with pytest.raises(SystemExit):
        resolve_thinking("meta-llama/Llama-3.3-70B-Instruct", "on")


def test_qwen3_32b_auto_reproduces_the_jul28_arm():
    """jul28 ran Qwen3-32B with enable_thinking=False. `auto` must still mean that,
    or re-running the roster would silently change an already-published arm."""
    assert resolve_thinking("Qwen/Qwen3-32B", "auto") == "off"


def test_every_configured_model_has_a_generation_budget():
    for model, cfg in MODEL_CONFIGS.items():
        assert cfg.get("max_tokens", 0) > 0, model


def test_newly_added_models_are_all_reachable_arms():
    """Every added model must resolve to some runnable arm under 'auto'."""
    added = [
        "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "microsoft/Phi-4-reasoning-plus",
        "openai/gpt-oss-120b", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507",
    ]
    for m in added:
        assert m in MODEL_CONFIGS, m
        assert resolve_thinking(m, "auto") in ("on", "off")


def test_multi_gpu_models_declare_tp():
    """A 70B/120B/235B arm launched at tp=1 OOMs an hour into the queue."""
    for m in ("openai/gpt-oss-120b", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
              "Qwen/Qwen3-235B-A22B-Thinking-2507"):
        assert get_model_config(m).get("tp", 1) > 1, m


# ── generation budget, per ARM ───────────────────────────────────────────────
# The earlier version of this guard iterated ALWAYS_THINKING_MODELS only, so a
# toggle model running a thinking arm on its non-thinking budget passed. The rule
# is about whether the arm emits reasoning, not about which set the model is in.

def _reasoning_arms():
    for m in MODEL_CONFIGS:
        if m == "__default__":
            continue
        if m in ALWAYS_THINKING_MODELS:
            yield m, "on"
        elif m in TOGGLE_THINKING_MODELS:
            yield m, "on"


@pytest.mark.parametrize("model,arm", list(_reasoning_arms()))
def test_every_reasoning_arm_clears_the_32k_floor(model, arm):
    grandfathered = {"Qwen/QwQ-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
    budget = arm_max_tokens(model, arm)
    if model in grandfathered:
        assert budget == 16384      # published at this; changing it needs approval
    else:
        assert budget >= 30720, f"{model} [{arm}] would truncate reasoning"


@pytest.mark.parametrize("model", sorted(TOGGLE_THINKING_MODELS))
def test_toggle_models_budget_the_two_arms_separately(model):
    assert arm_max_tokens(model, "on") > arm_max_tokens(model, "off")


def test_generation_budget_fits_inside_the_context_cap():
    """max_model_len must exceed the generation budget, or the engine aborts at init."""
    for model, arm in _reasoning_arms():
        cfg = get_model_config(model)
        budget = arm_max_tokens(model, arm)
        cap = cfg.get("max_model_len", budget + 4096)
        assert cap > budget, f"{model} [{arm}] leaves no room for the prompt"


def test_phi4_budget_respects_its_32k_context():
    """Phi-4-reasoning-plus tops out at 32768 total; the budget cannot consume it all."""
    cfg = get_model_config("microsoft/Phi-4-reasoning-plus")
    assert cfg["max_model_len"] == 32768
    assert arm_max_tokens("microsoft/Phi-4-reasoning-plus", "on") < 32768


# ── the v2 validity prompt (ported from the cluster, 2026-08-20) ─────────────
def test_prompt_is_the_disambiguated_v2_wording():
    """The cluster generated all 2816 published rows under v2; local had drifted to
    the compound v1, so uploading local would have reverted the prompt mid-dataset."""
    from freegen.run_eval import PROMPT_TEMPLATE, PROMPT_VERSION
    assert PROMPT_VERSION == "v2-valid-disambiguated"
    assert "does this simulation produce a physically correct solution" in PROMPT_TEMPLATE
    assert "Running without error is not sufficient." in PROMPT_TEMPLATE
    assert "does this code run and produce" not in PROMPT_TEMPLATE


def test_symmetric_confidence_classifier_hedges_both_directions():
    """The published classifier has an uncertain-yes bucket and no uncertain-no, so
    a hedged negative is filed as a confident one. The 2x2 rule must not do that."""
    from freegen.parse_score import (classify_valid_confidence,
                                     classify_valid_confidence_2x2)
    hedged_no = "no, though it might be fine for small dt"
    assert classify_valid_confidence(hedged_no) == "Confident No"   # the old asymmetry
    assert classify_valid_confidence_2x2(hedged_no) == "Hedged No"  # fixed here
    assert classify_valid_confidence_2x2("yes, but the BCs are wrong") == "Hedged Yes"
    assert classify_valid_confidence_2x2("yes") == "Confident Yes"
    assert classify_valid_confidence_2x2("no") == "Confident No"
    # A pure abstention has no direction and must not be forced onto a side.
    assert classify_valid_confidence_2x2("unclear") == ""
    assert classify_valid_confidence_2x2("") == ""


# ── method_axis: ported back from the cluster, 2026-08-20 ────────────────────
# The refactor from eval/ to freegen/ dropped this function. rescore_jsonl.py
# imports it under a bare `except ImportError: method_axis = None`, so its absence
# was silent: the column simply stopped being written, and viz/pde_dual_report.py
# reads it in four places. Pin it so a later move cannot lose it the same way.

def test_method_axis_exists_and_is_importable_bare():
    """rescore_jsonl.py does `from parse_score import method_axis` unqualified."""
    from parse_score import method_axis          # noqa: F401  (bare import on purpose)
    from freegen.parse_score import method_axis as m
    assert callable(m)


def test_method_axis_treats_off_axis_answers_as_abstention_not_error():
    """The ground truth labels time integration and spectral bases only. A response
    naming a spatial discretization answered a question that was never asked."""
    from freegen.parse_score import method_axis
    assert method_axis("crank-nicolson") == "on"
    assert method_axis("spectral") == "on"
    assert method_axis("finite difference method") == "off"
    assert method_axis("upwind scheme") == "off"
    assert method_axis("") == "none"
    assert method_axis(None) == "none"


def test_score_row_emits_method_axis():
    """Defining the function but not calling it would be a cosmetic port."""
    from freegen.parse_score import score_row
    out = score_row({"pde": "heat", "method": "finite difference", "behavior": "diffusion",
                     "valid": "yes"},
                    {"pde_class": "Heat", "num_method": "crank-nicolson",
                     "phys_process": "diffusion", "phys_valid": True})
    assert out["method_axis"] == "off"
