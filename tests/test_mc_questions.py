"""Unit tests for mc_questions.py"""
import json, math, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mc_logprob'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'freegen_static_judgments'))
from mc_questions import (
    normalize_gt_list,
    make_pde_question,
    make_tf_question,
    make_all_questions,
    row_seed,
    extract_letter_logprobs,
    compute_metrics,
    build_result_row,
    PDE_OPTIONS,
    PROCESS_CANDIDATES,
    METHOD_CANDIDATES,
    QUESTIONS_PER_ROW,
)


# ── normalize_gt_list ────────────────────────────────────────────────────────
def test_normalize_single():
    assert normalize_gt_list("explicit", "num_method") == ["explicit"]

def test_normalize_slash_sep():
    result = normalize_gt_list("explicit/spectral", "num_method")
    assert result == ["explicit", "spectral"]

def test_normalize_typo_difffusion():
    result = normalize_gt_list("difffusion", "phys_process")
    assert result == ["diffusion"]

def test_normalize_typo_in_multi():
    result = normalize_gt_list("advection/difffusion", "phys_process")
    assert "diffusion" in result
    assert "difffusion" not in result

def test_normalize_no_typo_other_fields():
    # Typo normalization should not apply to other fields
    result = normalize_gt_list("difffusion", "num_method")
    assert result == ["difffusion"]


# ── make_pde_question ────────────────────────────────────────────────────────
def test_pde_question_correct_letter_in_options():
    q = make_pde_question("x = 1", "heat", seed=42)
    letters = json.loads(q["letters"])
    assert q["correct_letter"] in letters
    assert letters[q["correct_letter"]] == "heat"

def test_pde_question_all_options_present():
    q = make_pde_question("x = 1", "wave", seed=42)
    letters = json.loads(q["letters"])
    assert set(letters.values()) == set(PDE_OPTIONS)

def test_pde_question_four_letters():
    q = make_pde_question("x = 1", "burgers", seed=7)
    letters = json.loads(q["letters"])
    assert set(letters.keys()) == {"A", "B", "C", "D"}

def test_pde_question_shuffle_varies():
    # Different seeds should produce different orderings
    q1 = json.loads(make_pde_question("x = 1", "heat", seed=1)["letters"])
    q2 = json.loads(make_pde_question("x = 1", "heat", seed=2)["letters"])
    # At least some seeds differ (not always identical ordering)
    orderings = set()
    for s in range(20):
        q = make_pde_question("x = 1", "heat", seed=s)
        orderings.add(json.loads(q["letters"])["A"])
    assert len(orderings) > 1

def test_pde_applicable_letters():
    q = make_pde_question("x = 1", "navier-stokes", seed=0)
    assert q["applicable_letters"] == ["A", "B", "C", "D"]


# ── make_tf_question ─────────────────────────────────────────────────────────
def test_tf_correct_letter_when_positive():
    # diffusion IS in gt_values → correct answer is "Yes"
    q = make_tf_question("x=1", "phys_process", "diffusion", ["diffusion", "advection"], seed=0)
    letters = json.loads(q["letters"])
    assert letters[q["correct_letter"]] == "Yes"

def test_tf_correct_letter_when_negative():
    # oscillation NOT in gt_values → correct answer is "No"
    q = make_tf_question("x=1", "phys_process", "oscillation", ["diffusion"], seed=0)
    letters = json.loads(q["letters"])
    assert letters[q["correct_letter"]] == "No"

def test_tf_ab_shuffle_produces_both_orderings():
    yes_a_count = 0
    for seed in range(50):
        q = make_tf_question("x=1", "phys_process", "diffusion", ["diffusion"], seed=seed)
        letters = json.loads(q["letters"])
        if letters["A"] == "Yes":
            yes_a_count += 1
    # Should see both orderings across 50 seeds
    assert 10 < yes_a_count < 40

def test_tf_applicable_letters():
    q = make_tf_question("x=1", "num_method", "explicit", ["explicit"], seed=0)
    assert q["applicable_letters"] == ["A", "B"]

def test_tf_candidate_stored():
    q = make_tf_question("x=1", "num_method", "spectral", ["explicit"], seed=0)
    assert q["candidate"] == "spectral"

def test_tf_question_type_stored():
    q = make_tf_question("x=1", "phys_valid", "valid", ["valid"], seed=0)
    assert q["question_type"] == "phys_valid"


# ── make_all_questions ───────────────────────────────────────────────────────
SAMPLE_ROW = {
    "title": "test_title",
    "code": "u = 0",
    "pde_class": "heat",
    "mod_type": "NoComm_Valid",
    "phys_process": "diffusion",
    "num_method": "explicit",
    "phys_valid": True,
}

def test_make_all_questions_count():
    qs = make_all_questions(SAMPLE_ROW, base_seed=0)
    assert len(qs) == QUESTIONS_PER_ROW

def test_make_all_questions_types():
    qs = make_all_questions(SAMPLE_ROW, base_seed=0)
    types = [q["question_type"] for q in qs]
    assert types.count("pde_class") == 1
    assert types.count("phys_process") == len(PROCESS_CANDIDATES)
    assert types.count("num_method") == len(METHOD_CANDIDATES)
    assert types.count("phys_valid") == 1

def test_make_all_questions_process_candidates():
    qs = make_all_questions(SAMPLE_ROW, base_seed=0)
    process_qs = [q for q in qs if q["question_type"] == "phys_process"]
    candidates = [q["candidate"] for q in process_qs]
    assert set(candidates) == set(PROCESS_CANDIDATES)

def test_make_all_questions_phys_valid_positive():
    qs = make_all_questions(SAMPLE_ROW, base_seed=0)
    valid_q = next(q for q in qs if q["question_type"] == "phys_valid")
    letters = json.loads(valid_q["letters"])
    assert letters[valid_q["correct_letter"]] == "Yes"

def test_make_all_questions_phys_valid_negative():
    row = {**SAMPLE_ROW, "phys_valid": False}
    qs = make_all_questions(row, base_seed=0)
    valid_q = next(q for q in qs if q["question_type"] == "phys_valid")
    letters = json.loads(valid_q["letters"])
    assert letters[valid_q["correct_letter"]] == "No"

def test_make_all_questions_difffusion_normalized():
    row = {**SAMPLE_ROW, "phys_process": "difffusion"}
    qs = make_all_questions(row, base_seed=0)
    diffusion_q = next(
        q for q in qs
        if q["question_type"] == "phys_process" and q["candidate"] == "diffusion"
    )
    letters = json.loads(diffusion_q["letters"])
    assert letters[diffusion_q["correct_letter"]] == "Yes"

def test_row_seed_deterministic():
    assert row_seed("title_a", "modelX") == row_seed("title_a", "modelX")

def test_row_seed_differs_by_title():
    assert row_seed("title_a", "modelX") != row_seed("title_b", "modelX")


# ── extract_letter_logprobs ──────────────────────────────────────────────────
class _FakeLp:
    def __init__(self, logprob, decoded_token):
        self.logprob = logprob
        self.decoded_token = decoded_token

def _make_logprobs(mapping: dict) -> list[dict]:
    """Build a fake vLLM logprobs structure for position 0."""
    pos0 = {i: _FakeLp(lp, tok) for i, (tok, lp) in enumerate(mapping.items())}
    return [pos0]

def test_extract_plain_letters():
    lps = _make_logprobs({"A": -0.1, "B": -1.5, "C": -2.0, "D": -3.0})
    result = extract_letter_logprobs(lps)
    assert abs(result["A"] - (-0.1)) < 1e-9
    assert abs(result["B"] - (-1.5)) < 1e-9
    assert abs(result["C"] - (-2.0)) < 1e-9
    assert abs(result["D"] - (-3.0)) < 1e-9

def test_extract_space_prefix():
    lps = _make_logprobs({" A": -0.5, " B": -1.0, " C": -2.0, " D": -3.5})
    result = extract_letter_logprobs(lps)
    assert result["A"] is not None
    assert result["B"] is not None

def test_extract_missing_letter_is_none():
    lps = _make_logprobs({"A": -0.1, "B": -1.5})
    result = extract_letter_logprobs(lps)
    assert result["C"] is None
    assert result["D"] is None

def test_extract_empty_logprobs():
    result = extract_letter_logprobs([])
    assert all(v is None for v in result.values())


# ── compute_metrics ──────────────────────────────────────────────────────────
def test_metrics_predicted_letter():
    lps = {"A": -0.1, "B": -2.0, "C": None, "D": None}
    m = compute_metrics(lps, "A", ["A", "B"])
    assert m["predicted_letter"] == "A"
    assert m["correct"] is True

def test_metrics_wrong_prediction():
    lps = {"A": -2.0, "B": -0.1, "C": None, "D": None}
    m = compute_metrics(lps, "A", ["A", "B"])
    assert m["predicted_letter"] == "B"
    assert m["correct"] is False

def test_metrics_margin():
    lps = {"A": -1.0, "B": -3.0, "C": None, "D": None}
    m = compute_metrics(lps, "A", ["A", "B"])
    assert abs(m["margin"] - 2.0) < 1e-9

def test_metrics_entropy_positive():
    lps = {"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0}
    m = compute_metrics(lps, "A", ["A", "B", "C", "D"])
    assert m["entropy"] > 0.0

def test_metrics_entropy_near_zero_when_certain():
    lps = {"A": 0.0, "B": -100.0, "C": None, "D": None}
    m = compute_metrics(lps, "A", ["A", "B"])
    assert m["entropy"] < 0.01

def test_metrics_none_when_no_logprobs():
    lps = {"A": None, "B": None, "C": None, "D": None}
    m = compute_metrics(lps, "A", ["A", "B"])
    assert m["predicted_letter"] is None
    assert m["correct"] is None
    assert m["entropy"] is None


# ── build_result_row ─────────────────────────────────────────────────────────
def test_build_result_row_all_columns_present():
    required = {
        "title", "pde_class", "mod_type", "model",
        "question_type", "candidate", "letters", "correct_letter",
        "logprob_A", "logprob_B", "logprob_C", "logprob_D",
        "predicted_letter", "correct", "logprob_correct", "margin", "entropy",
        "finish_reason",
    }
    lps = {"A": -0.5, "B": -1.5, "C": None, "D": None}
    q = make_tf_question("x=1", "phys_process", "diffusion", ["diffusion"], seed=0)
    result = build_result_row(SAMPLE_ROW, "test/model", q, lps, "stop")
    assert required.issubset(set(result.keys()))

def test_build_result_row_tf_c_d_stored():
    # C and D should always be stored, even for T/F (may be None if not in top-10)
    lps = {"A": -0.5, "B": -1.5, "C": None, "D": None}
    q = make_tf_question("x=1", "phys_valid", "valid", ["valid"], seed=0)
    result = build_result_row(SAMPLE_ROW, "test/model", q, lps, "stop")
    assert "logprob_C" in result
    assert "logprob_D" in result

def test_build_result_row_c_d_nonzero_if_extracted():
    # If the model put mass on C/D even during T/F, it should be stored
    lps = {"A": -0.5, "B": -1.5, "C": -4.0, "D": -5.0}
    q = make_tf_question("x=1", "phys_valid", "valid", ["valid"], seed=0)
    result = build_result_row(SAMPLE_ROW, "test/model", q, lps, "stop")
    assert result["logprob_C"] == -4.0
    assert result["logprob_D"] == -5.0
