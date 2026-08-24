"""Parsing and scoring logic for pde-llm-eval. No GPU required."""
import re
import numpy as np
from typing import Optional

_GT_TYPOS = {
    "behavior": {"difffusion": "diffusion"},
}

ALIASES = {
    "pde": {
        "wave":          ["wave", "wave equation"],
        "heat":          ["heat", "heat equation", "diffusion equation"],
        "burgers":       ["burgers", "burgers equation"],
        "navier-stokes": ["navier-stokes", "navier stokes", "ns equation",
                          "incompressible flow", "navier stokes equations"],
    },
    "method": {
        "explicit": ["explicit", "explicit euler", "forward euler",
                     "rk4", "runge-kutta", "runge kutta"],
        "implicit": ["implicit", "implicit euler", "crank-nicolson",
                     "crank nicolson", "backward euler"],
        "spectral": ["spectral", "fft", "fourier"],
    },
    "behavior": {
        "oscillation": ["oscillation", "oscillatory", "wave propagation", "vibration"],
        "diffusion":   ["diffusion", "diffusive", "heat conduction", "spreading"],
        "advection":   ["advection", "advective", "transport", "convection"],
        "restoration": ["restoration", "restoring", "damping", "decay"],
    },
}

VALID_MAPPING = {
    "yes": True, "no": False,
    "true": True, "false": False,
    "1": True, "0": False,
    "valid": True, "invalid": False,
}

# Placeholder tokens emitted when a model echoes the prompt template
_PLACEHOLDER = re.compile(r"^[_\s]*$")


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK = re.compile(r"^.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning traces before parsing the answer.

    Byte-identical to crossmodal/eval/parse_consistency.strip_think, deliberately:
    the two experiments run the same eight checkpoints over the same 32 solver
    systems, and a paired comparison between them is only meaningful if a reasoning
    trace is stripped the same way on both sides.

    The single-regex version this replaces required a MATCHED pair of tags, which is
    wrong for this roster in two distinct ways, both measured on the consistency run:

      * Some models never emit an opening <think>. Their chat template opens the
        reasoning block in the PROMPT, so only the closing tag comes back. The
        matched-pair regex finds nothing and the whole trace stays in the text the
        field parser reads. parse_response's "last non-placeholder match wins" rule
        happens to survive the common ordering -- a draft answer inside the trace is
        overwritten by the real one after it -- so this is a latent hazard here
        rather than a measured loss: it bites when the trace mentions a field AFTER
        the answer. Stripping at the closing tag removes the hazard outright.
      * A response truncated at the token cap never closes the tag at all, and this
        one is not latent. MEASURED on the case below: a run cut off mid-reasoning,
        having answered nothing, parses under the old regex to the draft value
        sitting in its own deliberation -- "pde: heat equation" reported as the
        model's answer when the model never gave one. Returning "" makes the row
        parse to all-None, and is_no_verdict() then marks it so it is dropped and
        counted rather than scored.

    On the consistency arms the same class of failure produced a verdict the model
    never gave for 907 of Nemotron's 3,072 rows (29.5%) and 256 of GLM's 2,304
    (11.1%), and those invented verdicts skewed toward one answer -- so it biased
    the result, not just the noise. Both models are in this roster.
    """
    text = _THINK.sub("", text)
    if "<think>" in text:
        return ""
    if "</think>" in text:
        text = _UNCLOSED_THINK.sub("", text)
    return text.strip()


def is_no_verdict(text: str, finish_reason: str = "") -> bool:
    """True when this run never reached an answer and must not be scored.

    Two causes, one consequence. The run hit its token budget (finish_reason
    "length"), or it opened a reasoning block it never closed. Either way the model
    produced no answer, and both of the alternatives to dropping the row are wrong:
    scoring it as an incorrect answer counts a run that said nothing as if it had
    said something false, and scoring the scavenged text as the answer invents one.

    Callers should record this as its own outcome and report the count, exactly as
    the consistency report does -- a dropped row that nobody counts is a silent
    change to the denominator.
    """
    if str(finish_reason).strip().lower() == "length":
        return True
    body = _THINK.sub("", text or "")
    return "<think>" in body


def parse_response(text: str) -> dict:
    """
    Parse structured output from a model response.
    Returns dict with keys: pde, method, behavior, valid (strings or None).

    Handles:
    - Standard format:    field: value
    - Bullet format:      - field: value  (phi-4, Mistral, Qwen2.5)
    - Template echo:      Llama echoes "pde: ____" before real answer — skip placeholders,
                          use the LAST non-placeholder match for each field.
    - Thinking blocks:    <think>...</think> stripped first.
    - Markdown emphasis:  "- **pde:** value", "**pde**: value", "*pde*: value"
                          (Qwen3-Coder-30B-A3B answers this way on every row; before
                          2026-08-19 all four fields came back None for it, which read
                          as a refusal rather than the complete, correct answer it is).
    - Numbered lists:     "1. pde: value"
    - RUN-ON single line: "pde: Burgers method: upwind behavior: advection valid: no"
                          Nemotron-3-Nano writes all four fields on ONE line with no
                          separators. The value pattern used to run to end-of-line and
                          the label had to sit at the START of one, so `pde` swallowed
                          the whole answer and the other three came back None. That is
                          88 of its 3 draws x 256 items (11.5%), and those rows scored
                          0.000 on method recall, behaviour AND validity -- not because
                          the model was wrong but because nothing was read. Its
                          published validity accuracy carried that. Values now stop at
                          the next field label, and labels are recognised mid-line.
    """
    text = strip_think(text or "")

    result = {}
    for field in ("pde", "method", "behavior", "valid"):
        # Find ALL matches (model may echo template before giving real answer)
        # Bullet or numbered marker, then the field name optionally wrapped in
        # markdown emphasis. The colon may sit inside the emphasis ("**pde:**") or
        # outside it ("**pde**:"), so allow marks on both sides and after.
        # A label may open a line OR follow whitespace mid-line (the run-on case).
        # The value stops at the next field label on the same line, so one run-on
        # line yields four values instead of one that contains the other three.
        _LABELS = "pde|method|behavior|valid"
        _MARK = r"(?:[-*+\u2022]|\d+[.)])?\s*[*_`]{0,3}\s*"
        # SAME-LINE whitespace only in the terminator. \s matches newlines, so a
        # terminator built from \s reached across the line break to find the NEXT
        # line's label and cut the value short: "pde: Burgers (u_t + (u^2/2)_x = 0)"
        # ended at the "=" because " 0)\nmethod:" satisfied the lookahead. That
        # silently truncated 60 Qwen3.8 rows -- every one whose answer ended in an
        # equation. A value never spans lines, so the terminator must not either.
        _H = r"[^\S\n]"
        _MARK_H = rf"(?:[-*+\u2022]|\d+[.)])?{_H}*[*_`]{{0,3}}{_H}*"
        matches = re.findall(
            rf"(?im)(?:^|(?<=\s)){_MARK}{field}\s*[*_`]{{0,3}}"
            rf"\s*:\s*[*_`]{{0,3}}\s*"
            rf"(.+?)"
            rf"(?={_H}+{_MARK_H}(?:{_LABELS}){_H}*[*_`]{{0,3}}{_H}*:|\n|$)",
            text,
        )
        val = None
        for raw_val in reversed(matches):  # last non-placeholder wins
            # Trailing emphasis from "- **pde:** Burgers' equation**" style answers
            cleaned = raw_val.strip().strip("*`").strip()
            cleaned = cleaned.strip("_").strip("-").strip()
            if cleaned and not _PLACEHOLDER.match(cleaned):
                val = cleaned
                break
        result[field] = val

    return result


def _any_alias_match(needle: str, gt_token: str, field: str) -> bool:
    """Return True if gt_token or any alias appears in needle (case-insensitive)."""
    needle_lower = needle.lower()
    candidates = [gt_token.lower()] + ALIASES.get(field, {}).get(gt_token.lower(), [])
    return any(c in needle_lower for c in candidates)


def score_pde(parsed_pde: Optional[str], gt_pde: str,
              embed_model=None) -> dict:
    """Keyword match (binary) + embedding similarity (continuous)."""
    text = (parsed_pde or "").lower()
    match = _any_alias_match(text, gt_pde, "pde") if text else False

    embed_sim = 0.0
    if embed_model and parsed_pde:
        try:
            e_gt   = embed_model.encode(gt_pde,    convert_to_numpy=True)
            e_pred = embed_model.encode(parsed_pde, convert_to_numpy=True)
            denom  = np.linalg.norm(e_gt) * np.linalg.norm(e_pred) + 1e-8
            embed_sim = float(np.dot(e_gt, e_pred) / denom)
            embed_sim = max(0.0, min(1.0, embed_sim))
        except Exception:
            embed_sim = 0.0

    return {"pde_match": int(match), "pde_embed_sim": round(embed_sim, 4)}


def score_multival(parsed_text: Optional[str], gt_string: str,
                   field: str) -> dict:
    """
    Score a /-separated multi-value GT field (method or behavior).
    any_match: 1 if ANY GT token found in response.
    recall:    fraction of GT tokens found (partial credit: 0, 0.5, 1.0).
    """
    typos = _GT_TYPOS.get(field, {})
    gt_tokens = [typos.get(t.strip().lower(), t.strip().lower()) for t in gt_string.split("/") if t.strip()]
    text = (parsed_text or "").strip()

    if not gt_tokens:
        return {f"{field}_any_match": 0, f"{field}_recall": 0.0}

    matched   = [t for t in gt_tokens if text and _any_alias_match(text, t, field)]
    any_match = int(len(matched) > 0)
    recall    = round(len(matched) / len(gt_tokens), 4)

    return {f"{field}_any_match": any_match, f"{field}_recall": recall}


def score_valid(parsed_valid: Optional[str], gt_valid: bool) -> dict:
    """
    Predict True/False from free-form valid field.
    Priority order:
      1. Exact dict lookup (yes/no/true/false/valid/invalid)
      2. Starts with yes → True, no → False
      3. Negative keywords (not valid, not fully valid, not physically valid) → False
      4. Positive keywords (physically valid, valid simulation, valid approach) → True
      5. Hedge phrases (potentially, conditionally, unknown) → abstain (score 0)
    """
    raw = (parsed_valid or "").lower().strip()

    # 1. Exact match
    pred = VALID_MAPPING.get(raw)

    if pred is None:
        # 2. Starts with yes/no
        if raw.startswith("yes"):
            pred = True
        elif raw.startswith("no"):
            pred = False
        # 3. Negative keywords (check before positive to catch "not fully valid")
        elif re.search(r"\bnot\b.{0,20}\bvalid\b", raw) or \
             re.search(r"\bnot fully valid\b", raw) or \
             re.search(r"\bnot physically valid\b", raw) or \
             raw.startswith("- the simulation is not"):
            pred = False
        # 4. Positive keywords
        elif re.search(r"\bphysically valid\b", raw) or \
             re.search(r"\bvalid simulation\b", raw) or \
             re.search(r"\bvalid numerical approach\b", raw) or \
             re.search(r"\bvalid approach\b", raw) or \
             re.search(r"\bgenerally valid\b", raw):
            pred = True
        # 5. Hedge → abstain (pred stays None → score 0, not penalized as wrong)

    match = int(pred == gt_valid) if pred is not None else 0
    return {"valid_match": match}


def score_row(parsed: dict, row: dict, embed_model=None) -> dict:
    """Score all four fields for a single dataset row."""
    out = {}
    out.update(score_pde(parsed.get("pde"), str(row["pde_class"]), embed_model))
    out.update(score_multival(parsed.get("method"),   str(row["num_method"]),  "method"))
    # method_recall above is left as-is for continuity; method_axis says whether it
    # is interpretable. Analysis takes recall over axis == "on" and reports the
    # off-axis rate as its own number.
    out["method_axis"] = method_axis(parsed.get("method"))
    out.update(score_multival(parsed.get("behavior"), str(row["phys_process"]), "behavior"))
    out.update(score_valid(parsed.get("valid"), bool(row["phys_valid"])))
    return out


# ── Hedge / validity-confidence classifier ────────────────────────────────────
# Canonical home for the rule that was previously duplicated in four places:
# viz/visualize_v3.py, viz/visualize_v4_enhanced.py, viz/paper_figures.py, and
# eval/frontier/parse_frontier.py. The body is the frontier version verbatim —
# it is strictly more specific than the three viz copies (it has an explicit
# hedge lexicon), and keeping it byte-identical makes this consolidation a no-op
# for the already-published frontier results.
#
# NOTE: bucket shares are therefore NOT comparable to writeup.pdf Figure 1,
# which was produced by the looser viz rule where "possibly valid" fell into a
# catch-all rather than into Hedged.

VALID_CONF_CLASSES = ("Confident Yes", "Uncertain Yes", "Hedged", "Confident No")


def classify_valid_confidence(raw: str) -> str:
    """
    Bucket a free-form `valid:` answer by stated confidence.

    Returns one of VALID_CONF_CLASSES. Empty / unparseable input is 'Hedged'
    (the model committed to nothing), matching the frontier behaviour.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "Hedged"
    s = raw.lower().strip()

    if s in ("yes", "true", "valid"):
        return "Confident Yes"
    if s in ("no", "false", "invalid"):
        return "Confident No"
    if s.startswith("yes"):
        return "Uncertain Yes"
    if s.startswith("no"):
        return "Confident No"
    if (re.search(r"\bnot\b.{0,20}\bvalid\b", s)
            or "not fully valid" in s
            or "not physically valid" in s):
        return "Confident No"
    if re.search(r"\bphysically valid\b|\bvalid simulation\b|\bvalid approach\b|\bgenerally valid\b", s):
        return "Uncertain Yes"
    if re.search(r"\b(unclear|cannot determine|uncertain|depends|potentially|possibly|might be|may be)\b", s):
        return "Hedged"
    if re.search(r"\byes\b|\bvalid\b|\bcorrect\b", s):
        return "Uncertain Yes"
    if re.search(r"\bno\b|\binvalid\b|\bincorrect\b", s):
        return "Confident No"
    return "Hedged"


VALID_CONF_2X2_CLASSES = ("Confident No", "Hedged No", "Hedged Yes", "Confident Yes")

# Epistemic hedges only. Contrast conjunctions ("but", "however", "although") are
# handled separately below, because "yes, but the BCs are wrong" is a qualified
# verdict while "the flux is wrong, however the grid is fine" is not a hedge at all.
_HEDGE_RE = re.compile(r"""\b(?:
    likely|unlikely|probably|probable|may|might|maybe|possibly|possible|
    potentially|potential|appears?|seems?|suggests?|
    unclear|uncertain|unsure|not\s+sure|cannot\s+determine|can['']?t\s+determine|
    hard\s+to\s+say|depends?|questionable|doubtful|dubious|suspect|borderline|
    partially|partly|mostly|somewhat|largely|roughly|approximately|
    arguably|presumably|apparently|in\s+principle
)\b""", re.X)

# A positive term sitting inside a negation scope is a NEGATIVE verdict. This is the
# case a bare `\bcorrect\b` scan gets wrong: in "doesn't represent a correct upwind
# discretization", the word "correct" is inside the thing being denied.
_NEGATOR = r"""\b(?:
    not|never|without|lacks?|fails?\s+to|unable\s+to|cannot|can['']?t|
    does\s+n[o'']?t|doesn['']?t|is\s+not|isn['']?t|are\s+not|aren['']?t|
    will\s+not|won['']?t|would\s+not|wouldn['']?t|no\s+longer
)\b"""

_NEGATED_POS_RE = re.compile(_NEGATOR + r"""
    # The negation must reach the positive term within ONE clause. Without this the
    # window jumps a conjunction -- "does not diverge and is correct" reads as
    # "not ... correct" -- and negates a virtue the sentence actually affirms.
    (?:(?!\b(?:and|but|or|yet|while|whereas|though|although)\b)[^.;!?]){0,60}?
    \b(?:valid|correct(?:ly)?|accurate(?:ly)?|stable|sound|reliable)\b""", re.X)

_NEGATED_NEG_RE = re.compile(_NEGATOR + r"""
    (?:(?!\b(?:and|but|or|yet|while|whereas|though|although)\b)[^.;!?]){0,60}?
    \b(?:invalid|incorrect|inaccurate|unstable|unphysical|wrong|flawed|
         diverges?|fails?|violates?)\b""", re.X)

_NEG_TERM_RE = re.compile(r"""\b(?:
    no|invalid|incorrect|inaccurate|unstable|unphysical|non-?physical|
    wrong|flawed|broken|erroneous|buggy|unreliable|inconsistent|
    fails?|failure|violates?|diverges?
)\b""", re.X)

_POS_TERM_RE = re.compile(r"""\b(?:
    yes|valid(?:ly)?|correct(?:ly)?|accurate(?:ly)?|sound|acceptable|plausible
)\b""", re.X)

# Adverbs that can stand in for the whole verdict when the response opens with one.
_LEAD_POLARITY = {
    "likely": True, "probably": True, "presumably": True, "apparently": True,
    "partially": True, "partly": True, "mostly": True, "largely": True,
    "generally": True, "possibly": True, "potentially": True, "maybe": True,
    "unlikely": False, "doubtful": False, "questionable": False,
    "uncertain": None, "unclear": None, "unknown": None, "indeterminate": None,
}
_LEAD_ADVERB_RE = re.compile(r"^(" + "|".join(sorted(_LEAD_POLARITY, key=len, reverse=True))
                             + r")\b")

_CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|except|caveat|"
                          r"provided|assuming|unless)\b")


_WRAPPER_RE = re.compile(r"\\(?:boxed|text|mathrm|textbf|mbox)\s*\{|[{}`*$]|\\\\")


def _strip_wrappers(s):
    """Remove LaTeX and markdown packaging so the verdict sits at position 0.

    Reasoning models wrap the final answer often enough that a leading-token test
    silently stops working: "\\boxed{\\text{Yes, but ...}}" opens with a backslash,
    so every ^(yes|no) rule misses it and the answer falls through to a keyword scan.
    """
    return _WRAPPER_RE.sub("", s).strip()


def valid_direction(raw):
    """True / False / None for the verdict, by EARLIEST decisive marker.

    Deterministic, no model in the loop. The ordering matters more than the lexicon:
    both existing rules scan the whole string for a bare keyword and take whichever
    check happens to run first, so `classify_valid_confidence` (positives first)
    and `valid_intent` (negatives first) can read the same sentence in opposite
    directions. Scanning left to right instead means the verdict the model actually
    opened with wins, which is what a reader would do.

    Negated positives are resolved before bare positives, so "does not produce a
    correct solution" is scored on "not ... correct" rather than on "correct".
    """
    if not isinstance(raw, str):
        return None
    s = _strip_wrappers(raw.lower().strip())
    if not s:
        return None

    pred = VALID_MAPPING.get(s)
    if pred is not None:
        return pred
    # An opening yes/no is the verdict; whatever follows is justification.
    if re.match(r"^(yes|valid|true)\b", s):
        return True
    if re.match(r"^(no|invalid|false)\b", s):
        return False

    lead = _LEAD_ADVERB_RE.match(s)
    if lead:
        rest = s[lead.end():].lstrip(" ,;:-—")
        polarity = _LEAD_POLARITY[lead.group(1)]
        if polarity is None:
            return None                       # "unknown (because ...)" -> abstention
        inner = valid_direction(rest) if rest else None
        if inner is not None:
            return inner                      # "likely invalid ..."   -> False
        return polarity                       # "likely, given ..."    -> True

    negated = [(m.start(), False) for m in _NEGATED_POS_RE.finditer(s)]
    negated += [(m.start(), True) for m in _NEGATED_NEG_RE.finditer(s)]
    # Bare positives that fall inside a negation span are already counted above.
    spans = ([(m.start(), m.end()) for m in _NEGATED_POS_RE.finditer(s)]
             + [(m.start(), m.end()) for m in _NEGATED_NEG_RE.finditer(s)])
    def _inside(i):
        return any(a <= i < b for a, b in spans)

    marks = negated
    marks += [(m.start(), False) for m in _NEG_TERM_RE.finditer(s) if not _inside(m.start())]
    marks += [(m.start(), True) for m in _POS_TERM_RE.finditer(s) if not _inside(m.start())]
    if not marks:
        return None
    return min(marks, key=lambda t: t[0])[1]


def classify_valid_confidence_2x2(raw):
    """Symmetric bucket: {Confident, Hedged} x {Yes, No}. "" when there is no lean.

    SEPARATE FROM classify_valid_confidence ON PURPOSE. That rule has an
    "Uncertain Yes" bucket and no uncertain-no, so a hedged negative has nowhere
    intact to go: it loses either its hedge or its direction. The published
    `valid_conf` column is left byte-identical, because recomputing it would
    restate results already on the Hub; this is derived at report time.

    A response with no direction at all returns "" rather than being filed as a
    hedged yes or a hedged no -- an abstention is not a lean.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = raw.lower().strip()
    lean = valid_direction(raw)
    if lean is None:
        return ""
    hedged = bool(_HEDGE_RE.search(s))
    # "yes, but ..." is a verdict the model then qualified; "no. the BCs are wrong,
    # but the grid is fine" is not -- the contrast has to attach to the verdict.
    if not hedged and re.match(r"^(yes|no|valid|invalid)\b\s*[,;:-]", s):
        head = s[:120]
        hedged = bool(_CONTRAST_RE.search(head))
    return f"{'Hedged' if hedged else 'Confident'} {'Yes' if lean else 'No'}"


def valid_intent(parsed_valid: Optional[str]) -> Optional[bool]:
    """
    The yes/no the model actually leaned toward, independent of whether
    score_valid credits it.

    score_valid abstains on a hedge (scores 0), so accuracy *within* the Hedged
    bucket would be 0 by construction if computed from valid_match. This reads
    the directional lean instead: score_valid's own ladder first, then a bare
    keyword fallback for hedged phrasings like "potentially valid, but ...".

    Returns True / False, or None when the response leans neither way (those
    rows must be reported as a separate count, never silently dropped).
    """
    # Not `parsed_valid or ""`: a pandas NaN is a float and is truthy, so that
    # form crashes on any CSV round-trip where the field was missing.
    if not isinstance(parsed_valid, str):
        return None
    raw = parsed_valid.lower().strip()
    if not raw:
        return None

    # Reuse the scoring ladder so a committed answer is read identically.
    pred = VALID_MAPPING.get(raw)
    if pred is not None:
        return pred
    if raw.startswith("yes"):
        return True
    if raw.startswith("no"):
        return False
    if (re.search(r"\bnot\b.{0,20}\bvalid\b", raw)
            or "not fully valid" in raw
            or "not physically valid" in raw):
        return False
    if re.search(r"\bphysically valid\b|\bvalid simulation\b"
                 r"|\bvalid numerical approach\b|\bvalid approach\b|\bgenerally valid\b", raw):
        return True

    # Hedged phrasings: fall back to the bare directional keyword. Negatives
    # first, so "not entirely correct" is not read as a yes.
    if re.search(r"\bnot\b|\binvalid\b|\bincorrect\b|\bno\b", raw):
        return False
    if re.search(r"\bvalid\b|\byes\b|\bcorrect\b", raw):
        return True
    return None


# ── Method answer axis ────────────────────────────────────────────────────────
# The ground-truth method label is not one axis. explicit/implicit describe time
# integration; spectral describes the spatial basis, and appears iff the solver uses
# FFT (checked: 6/6 across the 32 base solvers, and no FFT-free solver is labelled
# spectral). Finite-difference-in-space is never labelled at all.
#
# So a response of "finite difference method" or "upwind scheme" names an axis the
# ground truth deliberately leaves blank. Scoring it 0 reads as a wrong answer and
# pushes method_recall toward a floor where per-condition degradation cannot be
# detected. Treat it as an abstention instead - the same treatment score_valid gives
# a hedge - and report the rate separately.

_METHOD_OFF_AXIS = [
    "finite difference", "finite-difference", "finite volume", "finite element",
    "upwind", "central difference", "godunov", "muscl", "weno", "tvd",
    "minmod", "limiter", "jacobi", "gauss-seidel", "gauss seidel",
    "conjugate gradient", "thomas algorithm", "tridiagonal", "method of lines",
]


def method_axis(parsed_method: Optional[str]) -> str:
    """
    Which axis did the response actually answer on?

      'on'   - names a time-integration scheme or a spectral basis: the labelled axis,
               so method_recall is meaningful whether it is right or wrong.
      'off'  - names only a spatial discretization or linear solver, which the ground
               truth never labels. An abstention, not an error.
      'none' - empty, or nothing recognisable either way.
    """
    text = (parsed_method or "").lower().strip() if isinstance(parsed_method, str) else ""
    if not text:
        return "none"
    on_axis = set(ALIASES["method"])
    for aliases in ALIASES["method"].values():
        on_axis.update(aliases)
    if any(term in text for term in on_axis):
        return "on"
    if any(term in text for term in _METHOD_OFF_AXIS):
        return "off"
    return "none"
