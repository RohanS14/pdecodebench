"""
Question generation and logprob extraction for pde-mc-logprob eval.

Each code snippet generates 9 question instances:
  - 1 × pde_class  (4-way MC, options shuffled)
  - 4 × phys_process (T/F per candidate: advection, diffusion, oscillation, restoration)
  - 3 × num_method   (T/F per candidate: explicit, implicit, spectral)
  - 1 × phys_valid   (T/F, A/B shuffled)
"""
import hashlib
import json
import math
import random
from typing import Optional

# ── Vocabulary ──────────────────────────────────────────────────────────────
PDE_OPTIONS = ["burgers", "heat", "navier-stokes", "wave"]
PROCESS_CANDIDATES = ["advection", "diffusion", "oscillation", "restoration"]
METHOD_CANDIDATES = ["explicit", "implicit", "spectral"]
QUESTIONS_PER_ROW = 1 + len(PROCESS_CANDIDATES) + len(METHOD_CANDIDATES) + 1  # 9

# Normalize dataset typos before any GT comparison
_PROCESS_TYPOS = {"difffusion": "diffusion"}

# Letter variants that tokenizers may emit for A/B/C/D
_LETTER_VARIANTS = {
    "A": ["A", " A", "▁A", "\u0041"],
    "B": ["B", " B", "▁B", "\u0042"],
    "C": ["C", " C", "▁C", "\u0043"],
    "D": ["D", " D", "▁D", "\u0044"],
}

# ── Prompt templates ─────────────────────────────────────────────────────────
_PDE_PROMPT = """\
You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

What type of PDE is this simulation solving?

A) {opt_A}
B) {opt_B}
C) {opt_C}
D) {opt_D}

Output only the letter of the correct answer.\
"""

_TF_PROCESS_PROMPT = """\
You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

Is {candidate} one of the dominant physical processes in this simulation?

A) {opt_A}
B) {opt_B}

Output only the letter of the correct answer.\
"""

_TF_METHOD_PROMPT = """\
You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

Is {candidate} one of the numerical methods used in this simulation?

A) {opt_A}
B) {opt_B}

Output only the letter of the correct answer.\
"""

_TF_VALID_PROMPT = """\
You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

Does this code run and produce a correct physical solution for the PDE?

A) {opt_A}
B) {opt_B}

Output only the letter of the correct answer.\
"""


# ── GT normalization ─────────────────────────────────────────────────────────
def normalize_gt_list(raw: str, field: str) -> list[str]:
    """Split /-separated GT string into list of canonical values."""
    items = [v.strip() for v in str(raw).split("/") if v.strip()]
    if field == "phys_process":
        items = [_PROCESS_TYPOS.get(v, v) for v in items]
    return items


# ── Question builders ─────────────────────────────────────────────────────────
def make_pde_question(code: str, gt_pde: str, seed: int) -> dict:
    """4-way MC for pde_class. Returns one question dict."""
    rng = random.Random(seed)
    options = PDE_OPTIONS.copy()
    rng.shuffle(options)
    letters = {chr(ord("A") + i): opt for i, opt in enumerate(options)}

    # Find which letter holds the GT class
    gt_norm = gt_pde.strip().lower()
    correct_letter = next(
        (l for l, opt in letters.items() if opt.lower() == gt_norm), None
    )

    prompt = _PDE_PROMPT.format(
        code=code,
        opt_A=letters["A"],
        opt_B=letters["B"],
        opt_C=letters["C"],
        opt_D=letters["D"],
    )
    return {
        "question_type": "pde_class",
        "candidate": None,
        "letters": json.dumps(letters),
        "correct_letter": correct_letter,
        "prompt": prompt,
        "applicable_letters": ["A", "B", "C", "D"],
    }


def make_tf_question(
    code: str,
    question_type: str,
    candidate: str,
    gt_values: list[str],
    seed: int,
) -> dict:
    """T/F question for a specific candidate. A/B assignment is randomized."""
    rng = random.Random(seed)
    yes_is_A = rng.random() < 0.5
    if yes_is_A:
        letters = {"A": "Yes", "B": "No"}
    else:
        letters = {"A": "No", "B": "Yes"}

    is_positive = candidate.lower() in [v.lower() for v in gt_values]
    correct_letter = "A" if (
        (is_positive and yes_is_A) or (not is_positive and not yes_is_A)
    ) else "B"

    if question_type == "phys_process":
        prompt = _TF_PROCESS_PROMPT.format(
            code=code,
            candidate=candidate,
            opt_A=letters["A"],
            opt_B=letters["B"],
        )
    elif question_type == "num_method":
        prompt = _TF_METHOD_PROMPT.format(
            code=code,
            candidate=candidate,
            opt_A=letters["A"],
            opt_B=letters["B"],
        )
    else:  # phys_valid
        prompt = _TF_VALID_PROMPT.format(
            code=code,
            opt_A=letters["A"],
            opt_B=letters["B"],
        )

    return {
        "question_type": question_type,
        "candidate": candidate,
        "letters": json.dumps(letters),
        "correct_letter": correct_letter,
        "prompt": prompt,
        "applicable_letters": ["A", "B"],
    }


def make_all_questions(row: dict, base_seed: int) -> list[dict]:
    """
    Generate all 9 question dicts for one dataset row.
    base_seed is derived from (title, model) so shuffles are reproducible.
    """
    code = str(row["code"])
    gt_pde = str(row["pde_class"]).strip().lower()
    gt_process = normalize_gt_list(str(row["phys_process"]), "phys_process")
    gt_method = normalize_gt_list(str(row["num_method"]), "num_method")
    gt_valid = bool(row["phys_valid"])

    questions = []
    seed_counter = base_seed

    # pde_class (4-way MC)
    questions.append(make_pde_question(code, gt_pde, seed_counter))
    seed_counter += 1

    # phys_process (T/F × 4)
    for candidate in PROCESS_CANDIDATES:
        questions.append(
            make_tf_question(code, "phys_process", candidate, gt_process, seed_counter)
        )
        seed_counter += 1

    # num_method (T/F × 3)
    for candidate in METHOD_CANDIDATES:
        questions.append(
            make_tf_question(code, "num_method", candidate, gt_method, seed_counter)
        )
        seed_counter += 1

    # phys_valid (T/F × 1)
    gt_valid_list = ["valid"] if gt_valid else []
    questions.append(
        make_tf_question(code, "phys_valid", "valid", gt_valid_list, seed_counter)
    )

    return questions


def row_seed(title: str, model: str) -> int:
    """Deterministic seed from (title, model) — uses SHA-256 (stable across Python versions)."""
    h = hashlib.sha256(f"{title}|||{model}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ── Logprob extraction ───────────────────────────────────────────────────────
def extract_letter_logprobs(
    vllm_logprobs: list[dict],
) -> dict[str, Optional[float]]:
    """
    Extract logprobs for A, B, C, D from vLLM output.

    vllm_logprobs is output.outputs[0].logprobs — a list (one dict per token
    position). We look at position 0 only (max_tokens=1).

    Each dict maps token_id -> Logprob(logprob=float, decoded_token=str).
    We search for all known variants of each letter (space-prefixed, etc.).

    Returns: {"A": float|None, "B": float|None, "C": float|None, "D": float|None}
    """
    result: dict[str, Optional[float]] = {"A": None, "B": None, "C": None, "D": None}

    if not vllm_logprobs:
        return result

    pos0 = vllm_logprobs[0]  # dict: token_id -> Logprob object

    for letter, variants in _LETTER_VARIANTS.items():
        for _token_id, lp_obj in pos0.items():
            decoded = getattr(lp_obj, "decoded_token", None)
            if decoded in variants:
                result[letter] = lp_obj.logprob
                break  # first match wins

    return result


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(
    logprobs_by_letter: dict[str, Optional[float]],
    correct_letter: str,
    applicable_letters: list[str],
) -> dict:
    """
    Compute derived metrics from letter logprobs.

    applicable_letters: ["A","B","C","D"] for MC, ["A","B"] for T/F.
    """
    # Gather logprobs for applicable letters only
    lps = {l: logprobs_by_letter[l] for l in applicable_letters}
    available = {l: v for l, v in lps.items() if v is not None}

    predicted_letter: Optional[str] = None
    correct: Optional[bool] = None
    logprob_correct: Optional[float] = logprobs_by_letter.get(correct_letter)
    margin: Optional[float] = None
    entropy: Optional[float] = None

    if available:
        predicted_letter = max(available, key=lambda l: available[l])
        correct = predicted_letter == correct_letter if correct_letter else None

        if logprob_correct is not None and len(available) > 1:
            other_lps = [v for l, v in available.items() if l != correct_letter]
            if other_lps:
                margin = logprob_correct - max(other_lps)

        # Entropy over softmax of available logprobs
        vals = list(available.values())
        max_v = max(vals)
        exps = [math.exp(v - max_v) for v in vals]
        total = sum(exps)
        probs = [e / total for e in exps]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)

    return {
        "predicted_letter": predicted_letter,
        "correct": correct,
        "logprob_correct": logprob_correct,
        "margin": margin,
        "entropy": entropy,
    }


def build_result_row(
    row: dict,
    model: str,
    question: dict,
    logprobs_by_letter: dict[str, Optional[float]],
    finish_reason: str,
) -> dict:
    """Assemble one output JSONL row from all components."""
    metrics = compute_metrics(
        logprobs_by_letter,
        question["correct_letter"],
        question["applicable_letters"],
    )
    return {
        "title":           row["title"],
        "pde_class":       row["pde_class"],
        "mod_type":        row["mod_type"],
        "model":           model,
        "question_type":   question["question_type"],
        "candidate":       question["candidate"],
        "letters":         question["letters"],
        "correct_letter":  question["correct_letter"],
        # All four logprobs always stored; null for inapplicable letters
        "logprob_A":       logprobs_by_letter["A"],
        "logprob_B":       logprobs_by_letter["B"],
        "logprob_C":       logprobs_by_letter["C"],
        "logprob_D":       logprobs_by_letter["D"],
        "predicted_letter": metrics["predicted_letter"],
        "correct":         metrics["correct"],
        "logprob_correct": metrics["logprob_correct"],
        "margin":          metrics["margin"],
        "entropy":         metrics["entropy"],
        "finish_reason":   finish_reason,
    }
