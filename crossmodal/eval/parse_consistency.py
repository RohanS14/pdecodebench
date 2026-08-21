"""
parse_consistency.py — parsing and scoring for the cross-modal consistency
experiment (plan Part III). No GPU required.

Open-weight models have no function-calling contract, so the output arrives as
text. Where the compute node's vLLM supports guided decoding it is already valid
JSON; where it does not, it is whatever the model wrote. Both paths land here, and
the parse route is RECORDED per row rather than silently normalized -- if the
fallback is carrying a meaningful share of the results, that belongs in the report,
not buried.

Scoring follows the plan:

  * Detection is d', not accuracy. Only one of the eight conditions is all-agree,
    so a model that answers "no" to everything scores 7/8 while being useless. d'
    separates the hit rate on corrupted items from the false-alarm rate on A0.
  * Localization is scored only where the model said "no", against chance 1/4.
  * System identification reuses freegen/parse_score.py's alias tables rather than a
    second, divergent copy.
"""
import json
import math
import os
import re
import sys

# The alias table is shared with the free-generation half on purpose: identifying a
# PDE class must mean the same thing in both experiments, or the two halves cannot be
# compared. parse_score.py lives under freegen/ since the 2026-08-20 reorganisation.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from freegen.parse_score import _any_alias_match               # noqa: E402

FIELDS = ("agree", "outlier", "system_pde_class", "system_num_method", "justification")
VALID_OUTLIERS = {"view_1", "view_2", "view_3", "view_4", "none"}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK = re.compile(r"^.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def strip_think(text):
    """Remove reasoning traces before parsing the answer.

    Handles the unclosed case too: a thinking model that hits its token limit emits
    an opening <think> and never closes it, and treating that whole run as the
    answer would produce nonsense rather than an honest parse failure.

    Deliberately single-vocabulary. A two-vocabulary version was added on 2026-08-20
    for google/gemma-4, which wraps reasoning in <|channel>thought ... <channel|>
    rather than <think>. gemma-4 was then dropped from the roster (vLLM 0.19.1 cannot
    build its config -- per-layer head_dim), so that code became unreachable, and an
    unreachable branch that has never been checked against real sampled output is a
    liability: a literal <channel|> quoted inside a justification would have
    truncated the answer. Reverted, which also keeps this parser byte-identical to
    the one that produced the published arms -- so frozen and new rows are scored the
    same way. Restore it from git history if a channel-format model rejoins.
    """
    text = _THINK.sub("", text)
    if "<think>" in text:
        return ""
    if "</think>" in text:
        text = _UNCLOSED_THINK.sub("", text)
    return text.strip()


def _coerce(obj):
    out = {f: None for f in FIELDS}
    if not isinstance(obj, dict):
        return out
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    for f in FIELDS:
        v = lowered.get(f)
        if v is not None:
            out[f] = str(v).strip()
    return out


def parse_consistency(text):
    """Parse one model response.

    Returns the five fields plus `parse_route` (json | fenced_json | embedded_json |
    regex | failed) and `protocol_violation`, which flags a response that is
    well-formed but breaks the stated contract -- an outlier named while claiming
    agreement, or agreement denied with no outlier given.
    """
    body = strip_think(text or "")
    parsed, route = {f: None for f in FIELDS}, "failed"

    for candidate, name in ((body, "json"),
                            (_FENCE.search(body), "fenced_json"),
                            (_OBJECT.search(body), "embedded_json")):
        raw = candidate if isinstance(candidate, str) else (
            candidate.group(1) if name == "fenced_json" and candidate else
            candidate.group(0) if candidate else None)
        if not raw:
            continue
        try:
            got = _coerce(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            continue
        if got["agree"] is not None:
            parsed, route = got, name
            break

    if route == "failed" and body:
        # Regex cascade, mirroring run_mc_eval.extract_answer_letter's role for
        # reasoning models whose structured output does not survive.
        got = {f: None for f in FIELDS}
        for f in FIELDS:
            m = re.search(rf'(?im)"?{f}"?\s*[:=]\s*"?([^"\n,}}]+)"?', body)
            if m:
                got[f] = m.group(1).strip()
        if got["agree"] is None:
            m = re.search(r"(?i)\b(yes|no)\b", body)
            if m:
                got["agree"] = m.group(1)
        if got["outlier"] is None:
            m = re.search(r"(?i)\bview[_ ]?([1-4])\b", body)
            if m:
                got["outlier"] = f"view_{m.group(1)}"
        if got["agree"] is not None:
            parsed, route = got, "regex"

    agree = (parsed["agree"] or "").strip().lower()
    agree = agree if agree in ("yes", "no") else None
    outlier = (parsed["outlier"] or "").strip().lower().replace(" ", "_")
    outlier = outlier if outlier in VALID_OUTLIERS else None

    violation = ""
    if agree == "yes" and outlier not in (None, "none"):
        violation = "agreed_but_named_outlier"
    elif agree == "no" and outlier in (None, "none"):
        violation = "disagreed_without_outlier"

    return {
        "agree": agree,
        "outlier": outlier,
        "system_pde_class": parsed["system_pde_class"],
        "system_num_method": parsed["system_num_method"],
        "justification": parsed["justification"],
        "parse_route": route,
        "protocol_violation": violation,
    }


def score_consistency(parsed, item):
    """Score one response against its item's ground truth.

    detection_correct is defined for every item. localization_correct is None where
    the model said "yes" or gave no usable outlier -- scoring those as wrong would
    conflate failing to detect with detecting and mislocating, which are the two
    things the corruption ladder exists to separate.
    """
    corrupted = item["corrupted_view"] != "none"
    agree = parsed["agree"]

    detection = None if agree is None else int((agree == "no") == corrupted)

    localization = None
    if corrupted and agree == "no" and parsed["outlier"] not in (None, "none"):
        localization = int(parsed["outlier"] == f"view_{item['outlier_slot']}")

    failed = parsed.get("parse_route") == "failed"
    pde = parsed["system_pde_class"] or ""
    method = parsed["system_num_method"] or ""
    gt_methods = [t.strip() for t in str(item["gt_num_method"]).split("/") if t.strip()]

    return {
        "detection_correct": detection,
        "localization_correct": localization,
        # L2 (red team 2026-08-19): a total parse failure used to score 0 here,
        # indistinguishable from a confident wrong answer, so identification
        # accuracy was deflated by exactly the parse-failure rate and invisibly.
        # Detection and localization already return None for this; these now match.
        # An empty field on a row that DID parse is still a 0 -- that is a real miss.
        "pde_class_match": (None if failed else
                            int(_any_alias_match(pde, item["gt_pde_class"], "pde")) if pde else 0),
        "num_method_match": (None if failed else
                             int(any(_any_alias_match(method, t, "method")
                                     for t in gt_methods)) if method else 0),
        "is_corrupted": int(corrupted),
    }


def dprime(n_hit, n_signal, n_fa, n_noise):
    """d' from COUNTS, using the log-linear (Hautus) correction.

    Takes counts rather than rates because the correction has to be applied to the
    counts to be well behaved. An earlier version clamped the two rates
    independently using each condition's own n, which manufactures signal: a model
    that answers "no" to everything has a hit rate and a false-alarm rate that are
    both exactly 1, so d' must be 0, but clamping 1.0 against n=896 and against
    n=128 gives two different values and a spurious d' of 0.6. The design is
    deliberately 7:1 corrupted-to-clean, so that asymmetry is not hypothetical.

    The log-linear correction adds 0.5 to each count and 1 to each total. It still
    cannot make d' meaningful for a degenerate responder -- nothing can, because the
    rates carry no information -- which is why detection_summary() reports the two
    rates alongside and flags the degenerate case rather than leaving a reader to
    trust a single number.
    """
    hit_rate = (n_hit + 0.5) / (n_signal + 1)
    fa_rate = (n_fa + 0.5) / (n_noise + 1)
    return _z(hit_rate) - _z(fa_rate)


def _z(p):
    """Inverse normal CDF."""
    return math.sqrt(2) * _erfinv(2 * p - 1)


def _erfinv(y):
    """Inverse error function (Newton refinement on a rational seed)."""
    if y <= -1 or y >= 1:
        raise ValueError("erfinv domain is (-1, 1)")
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    x = math.copysign(math.sqrt(max(0.0, math.sqrt(t * t - ln / a) - t)), y)
    for _ in range(3):
        err = math.erf(x) - y
        x -= err / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def detection_summary(scored):
    """Hit rate, false-alarm rate, d', and whether the responder is degenerate.

    Reported together on purpose. d' alone hides the case that matters most here:
    with 7 of 8 conditions corrupted, a model that never agrees is 87.5% accurate
    and has learned nothing, and no correction to d' can express that honestly.
    `degenerate` is True when the model never used one of the two responses, and a
    degenerate d' should not be quoted as a result.
    """
    usable = [s for s in scored if s["detection_correct"] is not None]
    signal = [s for s in usable if s["is_corrupted"]]
    noise = [s for s in usable if not s["is_corrupted"]]
    if not signal or not noise:
        return None

    n_hit = sum(s["detection_correct"] for s in signal)
    n_fa = len(noise) - sum(s["detection_correct"] for s in noise)
    hit_rate = n_hit / len(signal)
    fa_rate = n_fa / len(noise)
    # "Never said yes" == every corrupted item a hit AND every clean item a false
    # alarm. "Never said no" is the mirror.
    degenerate = (n_hit == len(signal) and n_fa == len(noise)) or (n_hit == 0 and n_fa == 0)

    return {
        "n_signal": len(signal),
        "n_noise": len(noise),
        "hit_rate": round(hit_rate, 4),
        "false_alarm_rate": round(fa_rate, 4),
        "dprime": round(dprime(n_hit, len(signal), n_fa, len(noise)), 4),
        "degenerate": degenerate,
    }


def detection_dprime(scored):
    """d' alone. Prefer detection_summary() -- see its docstring."""
    summary = detection_summary(scored)
    return None if summary is None else summary["dprime"]
