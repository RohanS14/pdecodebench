"""Tests for frontier/judge_audit: scoring on a synthetic key, plus (when the local
run outputs are present) a check that the expert packet shows exactly the judge's
inputs."""
import csv
import math
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "frontier"))

from frontier.judge_audit.score_audit import (  # noqa: E402
    cohen_kappa, normalize, read_grading, read_key, score, wilson,
)

KEY_FIELDS = ["item_id", "model_family", "setting", "setting_label", "title", "mod_type", "pde_class",
              "judge_category", "judge_incorrect_code_claims", "judge_unverifiable_execution_claims",
              "judge_explanation", "judge_model"]
GRADE_FIELDS = ["item_id", "pde", "category", "incorrect_code_claims", "unverifiable_execution_claims",
                "missed_discrepancies", "explanation", "confidence", "minutes_spent"]

# judge labels for 4 synthetic items: (family, category, incorrect, unverifiable)
JUDGE = {
    "A01": ("gemini-2.5-flash", "none", "True", "True"),
    "A02": ("gemini-2.5-flash", "all", "False", "False"),
    "A03": ("gemini-3.7-flash", "all", "False", "True"),
    "A04": ("gemini-3.7-flash", "some", "True", "False"),
}


def write_key(path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEY_FIELDS)
        w.writeheader()
        for item, (fam, cat, inc, unv) in JUDGE.items():
            w.writerow({"item_id": item, "model_family": fam, "setting": "s", "setting_label": "s",
                        "title": f"T_{item}", "mod_type": "Comm_InValid", "pde_class": "heat",
                        "judge_category": cat, "judge_incorrect_code_claims": inc,
                        "judge_unverifiable_execution_claims": unv, "judge_explanation": f"judge on {item}",
                        "judge_model": "j"})
    return path


def grades_matching_judge():
    yn = {"True": "yes", "False": "no"}
    return [{"item_id": i, "pde": "Heat", "category": c, "incorrect_code_claims": yn[inc],
             "unverifiable_execution_claims": yn[unv], "missed_discrepancies": "x" if c == "some" else "",
             "explanation": "e", "confidence": "high", "minutes_spent": "5"}
            for i, (_, c, inc, unv) in JUDGE.items()]


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GRADE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def write_xlsx(path, rows):
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = "instructions"
    ws = wb.create_sheet("grading")
    ws.append(GRADE_FIELDS)
    for r in rows:
        ws.append([r[c] for c in GRADE_FIELDS])
    wb.save(path)
    return path


def run(tmp_path, rows):
    key = read_key(write_key(tmp_path / "key.csv"))
    return score(normalize(read_grading(write_csv(tmp_path / "g.csv", rows)), key), key)


def test_perfect_agreement_pooled_and_per_family(tmp_path):
    res = run(tmp_path, grades_matching_judge())
    for name, g in res["groups"].items():
        for m, a in g["agreement"].items():
            assert a["k"] == a["n"], (name, m)
    assert res["groups"]["pooled"]["n"] == 4
    assert res["groups"]["gemini-2.5-flash"]["n"] == res["groups"]["gemini-3.7-flash"]["n"] == 2


def test_known_disagreements(tmp_path):
    rows = grades_matching_judge()
    rows[0]["category"] = "some"                 # judge none -> expert some: category + named_defect differ
    rows[0]["missed_discrepancies"] = "missed the sign flip"
    rows[2]["incorrect_code_claims"] = "yes"     # judge False -> expert yes
    res = run(tmp_path, rows)
    pooled = res["groups"]["pooled"]["agreement"]
    assert (pooled["category"]["k"], pooled["named_defect"]["k"]) == (3, 3)
    assert pooled["incorrect_code_claims"]["k"] == 3
    assert pooled["unverifiable_execution_claims"]["k"] == 4
    fam25 = res["groups"]["gemini-2.5-flash"]
    assert fam25["agreement"]["named_defect"]["k"] == 1
    assert fam25["rates"]["named_defect"] == {"expert": 2, "judge": 1, "n": 2}
    assert res["groups"]["gemini-3.7-flash"]["rates"]["incorrect_code_claims"] == {"expert": 2, "judge": 1, "n": 2}


def test_csv_and_xlsx_score_identically(tmp_path):
    rows = grades_matching_judge()
    rows[1]["category"] = "none"
    key = read_key(write_key(tmp_path / "key.csv"))
    a = score(normalize(read_grading(write_csv(tmp_path / "g.csv", rows)), key), key)
    b = score(normalize(read_grading(write_xlsx(tmp_path / "g.xlsx", rows)), key), key)
    assert a == b


def test_incomplete_sheet_is_rejected(tmp_path):
    key = read_key(write_key(tmp_path / "key.csv"))
    rows = grades_matching_judge()
    rows[1]["incorrect_code_claims"] = ""
    with pytest.raises(ValueError, match="A02: incorrect_code_claims"):
        normalize(read_grading(write_csv(tmp_path / "a.csv", rows)), key)
    with pytest.raises(ValueError, match="ungraded items: A04"):
        normalize(read_grading(write_csv(tmp_path / "b.csv", grades_matching_judge()[:3])), key)
    rows = grades_matching_judge()
    rows[3]["missed_discrepancies"] = ""
    with pytest.raises(ValueError, match="A04: category 'some' requires missed_discrepancies"):
        normalize(read_grading(write_csv(tmp_path / "c.csv", rows)), key)


def test_wilson_and_kappa():
    lo, hi = wilson(18, 20)
    assert math.isclose(lo, 0.699, abs_tol=1e-3) and math.isclose(hi, 0.972, abs_tol=1e-3)
    assert wilson(20, 20)[1] == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None          # undefined, not 1.0
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0


KEY_PATH = os.path.join(ROOT, "results", "frontier", "judge_audit", "private", "judge_audit_key.csv")


@pytest.mark.skipif(not os.path.exists(KEY_PATH), reason="needs the local audit key and run outputs")
def test_packet_shows_exactly_the_judges_inputs():
    from frontier.judge_audit.build_packet import build_items
    from frontier.judge_invalidity_reasoning import build_judge_prompt
    items = build_items()
    assert len(items) == 20
    for it in items:
        prompt = build_judge_prompt(
            it["valid_code"], it["corrupted_code"], it["invalidity_note"], it["model_justification"],
            gt_valid_code=it["reference_valid_code"] or None,
            gt_invalid_code=it["reference_corrupted_code"] or None,
            caveat_note=it["caveat_note"], stage="s2")
        for field in ("caveat_note", "valid_code", "corrupted_code", "reference_valid_code",
                      "reference_corrupted_code", "invalidity_note", "model_justification"):
            assert it[field] in prompt, (it["item_id"], field)
        # the reference block is shown iff the judge got one
        assert ("<ground_truth_valid_code>" in prompt) == bool(it["reference_valid_code"]), it["item_id"]
