"""
Unit tests for model_pricing.py — runs locally, no network, no API key.

Guards the two corrections this module exists to make: the old gemini-2.5-flash
constants were wrong ($0.15/$1.25 against a listed $0.30/$2.50), and there is no
separate thinking rate (thinking bills at the output rate). Also guards the
condition-naming and thinking-family helpers, since crossing the families is a
silent misconfiguration on Gemini 3 rather than an error.
"""
import glob
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from frontier.model_pricing import (
    EPISODE_INPUT_TOKEN_ALLOWANCE,
    EPISODE_OUTPUT_TOKEN_ALLOWANCE,
    MODEL_PRICING,
    UnknownModelPricingError,
    condition_slug,
    episode_cost_cap,
    estimate_prompt_cost,
    is_gemini_3,
    parse_thinking_setting,
    pricing_for,
    thinking_levels_for,
    token_cost,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


print("\n── rates match the pricing doc ──")
check("2.5-flash input is $0.30/1M (NOT the old wrong $0.15)",
      abs(pricing_for("gemini-2.5-flash")["input"] - 0.30 / 1e6) < 1e-12)
check("2.5-flash output is $2.50/1M (NOT the old wrong $1.25)",
      abs(pricing_for("gemini-2.5-flash")["output"] - 2.50 / 1e6) < 1e-12)
check("3.7-flash is $0.75 in / $3.75 out (introductory)",
      abs(pricing_for("gemini-3.7-flash")["input"] - 0.75 / 1e6) < 1e-12
      and abs(pricing_for("gemini-3.7-flash")["output"] - 3.75 / 1e6) < 1e-12)
check("3.5-flash is MORE expensive than 3.7-flash (so it is not a cheap extra datapoint)",
      pricing_for("gemini-3.5-flash")["input"] > pricing_for("gemini-3.7-flash")["input"]
      and pricing_for("gemini-3.5-flash")["output"] > pricing_for("gemini-3.7-flash")["output"])
check("every flash-tier entry is marked verified against the doc",
      all(MODEL_PRICING[m]["verified"] for m in MODEL_PRICING if "flash" in m))
# Pro rates confirmed against the pricing doc: $2.00 in / $12.00 out per 1M,
# thinking included in output. The doc's >200k tier ($4.00/$18.00) is
# deliberately not modelled -- measured judge prompts top out ~29k tokens.
check("Pro/judge rates are marked verified (confirmed against the pricing doc)",
      all(MODEL_PRICING[m]["verified"] for m in MODEL_PRICING if "pro" in m))
check("Pro rates are the <=200k tier figures",
      abs(pricing_for("gemini-3.1-pro-preview")["input"] - 2.00 / 1e6) < 1e-12
      and abs(pricing_for("gemini-3.1-pro-preview")["output"] - 12.00 / 1e6) < 1e-12)
check("gemini-pro-latest and gemini-3.1-pro-preview price identically "
      "(the alias resolves to it -- confirmed via resp.model_version)",
      pricing_for("gemini-pro-latest") == pricing_for("gemini-3.1-pro-preview"))
# /3 not /4: code runs 1.7-2.8 chars/token, so /4 underestimated ~2x.
_code = "u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - u_nm1[1:N_x,1:N_y]\n" * 40
check("estimate_prompt_cost uses 3 chars/token, not 4",
      abs(estimate_prompt_cost("gemini-3.1-pro-preview", _code, output_token_guess=0)
          - (len(_code) / 3) * (2.00 / 1e6)) < 1e-12)

print("\n── cost formula: thinking bills at the OUTPUT rate, uniformly ──")
p = pricing_for("gemini-2.5-flash")
check("thinking tokens are priced identically to output tokens",
      abs(token_cost("gemini-2.5-flash", 0, 1000, 0)
          - token_cost("gemini-2.5-flash", 0, 0, 1000)) < 1e-15)
check("thinking is ADDED, not assumed already inside the output count",
      abs(token_cost("gemini-2.5-flash", 0, 1000, 1000)
          - 2000 * p["output"]) < 1e-15)
check("input and output are summed correctly",
      abs(token_cost("gemini-2.5-flash", 1000, 500, 0)
          - (1000 * p["input"] + 500 * p["output"])) < 1e-15)
check("zero tokens costs nothing", token_cost("gemini-2.5-flash", 0, 0, 0) == 0.0)

print("\n── unknown models fail loudly rather than borrowing another model's rates ──")
try:
    pricing_for("gemini-9.9-imaginary")
    check("unknown model raises", False, "no exception")
except UnknownModelPricingError:
    check("unknown model raises UnknownModelPricingError", True)
check("a dated/preview variant resolves via longest-prefix match",
      pricing_for("gemini-3.7-flash-preview-01-2027")["input"]
      == pricing_for("gemini-3.7-flash")["input"])

print("\n── episode cost cap scales with the model, not a flat dollar figure ──")
cap25, cap37, cap35 = (episode_cost_cap(m) for m in
                       ("gemini-2.5-flash", "gemini-3.7-flash", "gemini-3.5-flash"))
check("cap is derived from the token allowance at that model's rates",
      abs(cap25 - (EPISODE_INPUT_TOKEN_ALLOWANCE * p["input"]
                   + EPISODE_OUTPUT_TOKEN_ALLOWANCE * p["output"])) < 1e-12)
check("a pricier model gets a proportionally higher dollar cap (same token runway)",
      cap37 > cap25 and cap35 > cap37, f"{cap25:.4f} {cap37:.4f} {cap35:.4f}")
check("2.5-flash cap stays close to the historical $0.50 (behavior unchanged there)",
      0.40 <= cap25 <= 0.60, f"{cap25:.4f}")

print("\n── thinking family and condition identity ──")
check("2.5 is not Gemini 3", not is_gemini_3("gemini-2.5-flash"))
check("3.7 is Gemini 3", is_gemini_3("gemini-3.7-flash"))
check("3.7 does NOT accept 'minimal' (confirmed live)",
      "minimal" not in thinking_levels_for("gemini-3.7-flash"))
check("3.5 DOES accept 'minimal' (so it is the only model here spanning minimal->high)",
      "minimal" in thinking_levels_for("gemini-3.5-flash"))
check("parse: a level name on 3.7", parse_thinking_setting("gemini-3.7-flash", "high") == "high")
check("parse: a numeric budget on 2.5", parse_thinking_setting("gemini-2.5-flash", "1536") == 1536)
check("parse: None means the family default", parse_thinking_setting("gemini-3.7-flash", None) is None)
for model, raw, label in [("gemini-3.7-flash", "minimal", "unsupported level on 3.7"),
                          ("gemini-3.7-flash", "1536", "numeric on a level model"),
                          ("gemini-2.5-flash", "high", "level name on 2.5")]:
    try:
        parse_thinking_setting(model, raw)
        check(f"parse rejects {label}", False, "no exception")
    except ValueError:
        check(f"parse rejects {label}", True)

check("condition slug names a numeric budget explicitly",
      condition_slug("gemini-2.5-flash", 0) == "tb0"
      and condition_slug("gemini-2.5-flash", 1536) == "tb1536")
check("condition slug names a level explicitly (the old nothink/think naming "
      "could not express a Gemini 3 condition at all)",
      condition_slug("gemini-3.7-flash", "high") == "tl_high")
check("slugs are filesystem-safe",
      all(c.isalnum() or c == "_" for c in condition_slug("gemini-3.7-flash", "high")))

print("\n── recompute a real stored row from its persisted token counts ──")
# The whole reason raw token counts are persisted per row: when the pricing
# constants turn out to be wrong, cost is recoverable without re-running.
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..',
                'results/frontier/stratified_256/*/*belief_revision_agentic.jsonl'))):
    with open(f) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if rows:
        break

if not rows:
    print("  (skipped: no stored results on disk to recompute)")
else:
    r = next((x for x in rows if x.get("s2_input_tokens")), None)
    check("a stored row carries raw token counts, not just derived cost", r is not None)
    if r:
        recomputed = token_cost("gemini-2.5-flash", r["s2_input_tokens"],
                                r["s2_output_tokens"], r["s2_think_tokens"])
        check("recomputed cost is a real positive number", recomputed > 0, str(recomputed))
        # Old (wrong) constants: 0.15 in / 1.25 out / 3.50 thinking.
        old = (r["s2_input_tokens"] * 0.15 / 1e6 + r["s2_output_tokens"] * 1.25 / 1e6
               + r["s2_think_tokens"] * 3.50 / 1e6)
        check("corrected cost differs from what the old constants produced "
              "(so previously reported figures really were wrong)",
              abs(recomputed - old) > 1e-9, f"new={recomputed:.6f} old={old:.6f}")

print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
