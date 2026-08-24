"""Parsing and scoring logic for pde-llm-eval. No GPU required."""
import re
import numpy as np
from typing import Optional

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
    """
    # Strip thinking tokens
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    result = {}
    for field in ("pde", "method", "behavior", "valid"):
        # Find ALL matches (model may echo template before giving real answer)
        matches = re.findall(
            rf"(?im)^[-*]?\s*{field}\s*:\s*(.+?)(?:\n|$)", text
        )
        val = None
        for raw_val in reversed(matches):  # last non-placeholder wins
            cleaned = raw_val.strip().strip("_").strip("-").strip()
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
    gt_tokens = [t.strip().lower() for t in gt_string.split("/") if t.strip()]
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
    out.update(score_multival(parsed.get("behavior"), str(row["phys_process"]), "behavior"))
    out.update(score_valid(parsed.get("valid"), bool(row["phys_valid"])))
    return out
