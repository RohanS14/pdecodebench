"""
model_pricing.py — per-model token pricing, cost math, and the derived
per-episode cost cap. Single source of truth: previously these constants were
duplicated across run_belief_revision.py, run_belief_revision_agentic.py,
run_stratified_sweep.py (which imported one of them) and
judge_invalidity_reasoning.py, all hardcoded to one model's rates.

Two corrections are baked in here, both from the current pricing doc
(https://ai.google.dev/gemini-api/docs/pricing):

1. The old gemini-2.5-flash constants were WRONG -- $0.15 in / $1.25 out
   against a listed $0.30 / $2.50. Every cost figure computed before this
   module existed understates input and output by ~2x.
2. There is no separate thinking rate. The doc lists "Output price (including
   thinking tokens)", so thinking is billed at the output rate, not the
   $3.50/1M the old code used. The cost formula is therefore uniform across
   every model -- only the numbers differ.

Raw token counts are persisted per row throughout this pipeline precisely so
costs can be recomputed when constants turn out to be wrong, as they did here.
No network calls, no Gemini imports -- exercised entirely by
tests/test_model_pricing.py with no API key required.
"""

# USD per token. Keys are exact model IDs; lookup falls back to longest
# matching prefix so dated/preview variants of a known family resolve.
#
# `verified` marks whether the numbers came from the pricing doc directly.
# The Pro-tier judge rates were carried over from the previous hardcoded
# constants and have NOT been re-checked against the doc -- they are flagged so
# a stale judge price can't masquerade as a verified one.
MODEL_PRICING = {
    # Flash tier -- from the pricing doc.
    "gemini-2.5-flash":        {"input": 0.30 / 1e6, "output": 2.50 / 1e6, "verified": True},
    "gemini-3.5-flash":        {"input": 1.50 / 1e6, "output": 9.00 / 1e6, "verified": True},
    # 3.6/3.7 share introductory pricing through 2026-12-31, rising to
    # $1.50/$7.50 on 2027-01-01. Revisit before any run in 2027.
    "gemini-3.6-flash":        {"input": 0.75 / 1e6, "output": 3.75 / 1e6, "verified": True},
    "gemini-3.7-flash":        {"input": 0.75 / 1e6, "output": 3.75 / 1e6, "verified": True},

    # Pro tier -- judge models. Verified against the pricing doc: $2.00 in /
    # $12.00 out per 1M, thinking included in output.
    #
    # These are the <=200k-prompt rates. The doc also lists a >200k tier
    # ($4.00 in / $18.00 out for 3.1 Pro), deliberately NOT modelled here:
    # measured with count_tokens on real judge prompts, the largest is ~29k
    # tokens (Wave_NoComm_CorrVar_InValid_1), median ~2.1k -- an order of
    # magnitude below the threshold. Add the tier if prompt sizes ever change.
    "gemini-pro-latest":       {"input": 2.00 / 1e6, "output": 12.00 / 1e6, "verified": True},
    "gemini-3.1-pro-preview":  {"input": 2.00 / 1e6, "output": 12.00 / 1e6, "verified": True},
}

# Thinking-parameter family. Confirmed live, and the docs agree:
#   - gemini-2.5-flash REJECTS thinking_level ("Thinking level is not supported
#     for this model") and honors thinking_budget=0 (0 thinking tokens).
#   - gemini-3.7-flash ACCEPTS thinking_budget and SILENTLY IGNORES IT (still
#     spent 73 thinking tokens with thinking_budget=0), and requires
#     thinking_level. `minimal` is rejected on 3.7; its floor is `low`.
# Crossing them is the worst failure mode available here: it looks like it works.
FAMILY_THINKING_BUDGET = "thinking_budget"   # Gemini 2.5
FAMILY_THINKING_LEVEL = "thinking_level"     # Gemini 3.x

# Per-episode token headroom, priced at each model's own rates to produce the
# cost cap. The guard exists to stop a runaway episode, which is a *token*
# phenomenon, so the allowance is held constant across models and only the price
# varies -- a fixed dollar cap would silently give a pricier model less
# investigative runway than a cheaper one for the same work.
#
# Sized to land near the historical $0.50 cap on gemini-2.5-flash (~$0.48), so
# behavior on the already-validated model is effectively unchanged.
EPISODE_INPUT_TOKEN_ALLOWANCE = 500_000
EPISODE_OUTPUT_TOKEN_ALLOWANCE = 130_000


class UnknownModelPricingError(KeyError):
    """Raised rather than guessing. A silently wrong price feeds every cost
    guard and budget cap in the harness, so an unknown model must fail loudly
    instead of defaulting to some other model's rates."""


def pricing_for(model: str) -> dict:
    """Exact match, else the longest matching prefix (so e.g. a dated preview
    variant of a known family resolves). Raises rather than defaulting."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    prefixes = [k for k in MODEL_PRICING if model.startswith(k)]
    if prefixes:
        return MODEL_PRICING[max(prefixes, key=len)]
    raise UnknownModelPricingError(
        f"No pricing for model {model!r}. Add it to MODEL_PRICING -- do not let it "
        f"fall back to another model's rates, since this feeds every cost guard."
    )


def thinking_family(model: str) -> str:
    """Which thinking parameter this model takes. Gemini 3.x uses
    thinking_level; 2.5 uses the numeric thinking_budget."""
    if model.startswith("gemini-3"):
        return FAMILY_THINKING_LEVEL
    return FAMILY_THINKING_BUDGET


def is_gemini_3(model: str) -> bool:
    """Gemini 3.x needs: thinking_level instead of thinking_budget, no
    temperature/top_p/top_k, role='user' function responses with the call id
    echoed, and the model's full response content replayed verbatim (any Part
    may carry a thought_signature)."""
    return thinking_family(model) == FAMILY_THINKING_LEVEL


def token_cost(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    """Uniform formula for every model: thinking is billed at the output rate.

    `candidates_token_count` and `thoughts_token_count` are separate fields in
    usage_metadata, so thinking is ADDED here rather than assumed already
    included in the output count.
    """
    p = pricing_for(model)
    return input_tokens * p["input"] + (output_tokens + thinking_tokens) * p["output"]


def episode_cost_cap(model: str) -> float:
    """Per-episode cost cap, derived from this model's real rates rather than a
    flat dollar figure. Tripping it collapses the tool list exactly as budget
    exhaustion does -- it does NOT raise -- so a cap set too low silently ends
    investigation early and quietly changes the experiment."""
    p = pricing_for(model)
    return (EPISODE_INPUT_TOKEN_ALLOWANCE * p["input"]
            + EPISODE_OUTPUT_TOKEN_ALLOWANCE * p["output"])


def estimate_prompt_cost(model: str, prompt_text: str, output_token_guess: int = 120) -> float:
    """Rough pre-flight estimate for the session-level spend guards, from
    prompt text alone. Replaces run_belief_revision.py's estimate_cost_usd(),
    which hardcoded one model's input rate.

    3 chars/token, not the docs' 4: that figure is for English prose, and
    measured with count_tokens on this pipeline's actual prompts, PDE source
    runs 1.7-2.8 chars/token. /4 therefore underestimated input tokens ~2x,
    making the session spend guard looser than intended. /3 is a deliberate
    middle estimate, still optimistic against the dense end -- this is a
    heuristic guard, not billing, and real cost is always recomputed from
    persisted token counts."""
    p = pricing_for(model)
    return (len(prompt_text) / 3) * p["input"] + output_token_guess * p["output"]


# ── Thinking-condition identity ──────────────────────────────────────────────
# Which thinking_level values each model actually accepts, from the docs' own
# support table. Encoded rather than assumed because the levels are NOT uniform:
# gemini-3.7-flash rejects "minimal" (confirmed live: "Thinking level MINIMAL is
# not supported for this model"), while 3.5/3.6 accept it. Validating locally
# turns a wasted API call into an immediate error.
THINKING_LEVELS_BY_MODEL = {
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
}
DEFAULT_THINKING_LEVELS = ("low", "medium", "high")


def thinking_levels_for(model: str) -> tuple:
    """Accepted thinking_level values for a Gemini 3 model."""
    for name in sorted(THINKING_LEVELS_BY_MODEL, key=len, reverse=True):
        if model.startswith(name):
            return THINKING_LEVELS_BY_MODEL[name]
    return DEFAULT_THINKING_LEVELS


def parse_thinking_setting(model: str, raw: str | None):
    """Turn a CLI string into this model's thinking setting.

    Gemini 2.5 takes a numeric thinking_budget; Gemini 3.x takes a level name.
    None means "that family's default". Validates against the model's real
    accepted levels so an unsupported level fails here rather than after a
    paid call.
    """
    if raw is None or raw == "":
        return None
    if is_gemini_3(model):
        level = str(raw).strip().lower()
        allowed = thinking_levels_for(model)
        if level not in allowed:
            raise ValueError(
                f"{model} does not accept thinking_level {level!r}; allowed: {allowed}."
            )
        return level
    try:
        return int(str(raw).strip())
    except ValueError:
        raise ValueError(
            f"{model} takes a numeric thinking_budget, not {raw!r} "
            f"(it rejects thinking_level outright)."
        ) from None


def condition_slug(model: str, thinking_setting) -> str:
    """Filesystem/analysis-safe name for one thinking condition.

    Replaces the old nothink/think naming, which was derived from
    `thinking_budget == 0` and therefore cannot express a Gemini 3 condition at
    all. The setting is named explicitly so a directory or a JSONL row is
    self-describing rather than requiring the reader to know which sweep it came
    from.
    """
    if isinstance(thinking_setting, str):
        return f"tl_{thinking_setting.lower()}"
    # -1 means dynamic (the model picks a budget per request, and it is 2.5's
    # default). Named rather than rendered as "tb-1", which reads as a typo.
    if int(thinking_setting) < 0:
        return "tb_dynamic"
    return f"tb{int(thinking_setting)}"
