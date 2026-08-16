"""
Unit tests for judge_invalidity_reasoning.py — runs locally, no network.
Validates the mod_type mapping dict and the valid-counterpart lookup logic
against the REAL dataset (not a fake), since the whole point of this mapping
is to be correct against the dataset's actual (irregular) mod_type naming.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

import pandas as pd

from frontier.judge_invalidity_reasoning import (
    INVALID_TO_VALID_MOD_TYPE,
    _NEEDS_GROUND_TRUTH_REFERENCE,
    build_caveat_note,
    build_judge_prompt,
    find_ground_truth_reference,
    find_valid_counterpart_code,
    map_to_valid_mod_type,
)
from dataset_io import DEFAULT_MOD_DATASET

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATASET_PATH = os.path.join(REPO_ROOT, DEFAULT_MOD_DATASET)


# ── mod_type mapping dict ─────────────────────────────────────────────────────

print("\n── INVALID_TO_VALID_MOD_TYPE mapping ──")

check("has exactly 4 entries", len(INVALID_TO_VALID_MOD_TYPE) == 4, str(INVALID_TO_VALID_MOD_TYPE))
check("Comm_InValid -> Comm_Valid",                 map_to_valid_mod_type("Comm_InValid") == "Comm_Valid")
check("NoComm_InValid -> NoComm_Valid",             map_to_valid_mod_type("NoComm_InValid") == "NoComm_Valid")
check("NoComm_CorrVar_InValid -> NoComm_CorrVar",   map_to_valid_mod_type("NoComm_CorrVar_InValid") == "NoComm_CorrVar")
check("CorrComm_Invalid -> CorrComm (irregular capitalization, no 'Valid' suffix)",
      map_to_valid_mod_type("CorrComm_Invalid") == "CorrComm")
check("a valid mod_type itself returns None (not an invalid-condition input)",
      map_to_valid_mod_type("Comm_Valid") is None)
check("an unrecognized mod_type returns None", map_to_valid_mod_type("NotARealModType") is None)


# ── find_valid_counterpart_code against the REAL dataset ─────────────────────

print("\n── find_valid_counterpart_code (against the real dataset) ──")

df = pd.read_csv(DATASET_PATH)

check("every mod_type in the real dataset is either a key or a value in the mapping",
      set(df["mod_type"].unique()) == set(INVALID_TO_VALID_MOD_TYPE.keys()) | set(INVALID_TO_VALID_MOD_TYPE.values()),
      str(sorted(df["mod_type"].unique())))

# Heat_Comm_InValid_1's valid counterpart is Heat_Comm_Valid_1 -- a pair
# established and diffed manually earlier in this project's development.
invalid_row = df[df["title"] == "Heat_Comm_InValid_1"].iloc[0]
base_code = find_valid_counterpart_code(df, invalid_row["gt_sample"], invalid_row["mod_type"])
valid_row = df[df["title"] == "Heat_Comm_Valid_1"].iloc[0]
check("Heat_Comm_InValid_1's counterpart code matches Heat_Comm_Valid_1's code exactly",
      base_code == valid_row["code"], (base_code, valid_row["code"]))
check("the counterpart code is NOT the same as the invalid row's own code",
      base_code != invalid_row["code"])

# Every invalid-condition row in the real dataset must resolve to a real
# counterpart -- no silent gaps.
invalid_rows = df[df["mod_type"].isin(INVALID_TO_VALID_MOD_TYPE.keys())]
missing = []
for _, r in invalid_rows.iterrows():
    if find_valid_counterpart_code(df, r["gt_sample"], r["mod_type"]) is None:
        missing.append(r["title"])
check(f"every invalid-condition row in the full dataset ({len(invalid_rows)} rows) resolves to a real counterpart",
      len(missing) == 0, str(missing[:10]))

# A CorrComm_Invalid row specifically, since it's the one irregular mapping.
corrcomm_invalid = df[df["mod_type"] == "CorrComm_Invalid"].iloc[0]
corrcomm_base = find_valid_counterpart_code(df, corrcomm_invalid["gt_sample"], corrcomm_invalid["mod_type"])
check("CorrComm_Invalid resolves to a real CorrComm sibling", corrcomm_base is not None)


# ── find_ground_truth_reference against the REAL dataset ─────────────────────

print("\n── find_ground_truth_reference (against the real dataset) ──")

check("_NEEDS_GROUND_TRUTH_REFERENCE has exactly 3 entries (not Comm_InValid)",
      _NEEDS_GROUND_TRUTH_REFERENCE == {"NoComm_InValid", "NoComm_CorrVar_InValid", "CorrComm_Invalid"},
      str(_NEEDS_GROUND_TRUTH_REFERENCE))

comm_invalid_row = df[df["mod_type"] == "Comm_InValid"].iloc[0]
gt_valid, gt_invalid = find_ground_truth_reference(df, comm_invalid_row["gt_sample"], "Comm_InValid")
check("Comm_InValid itself returns (None, None) -- no reference needed", (gt_valid, gt_invalid) == (None, None))

check("an unrecognized mod_type returns (None, None)",
      find_ground_truth_reference(df, comm_invalid_row["gt_sample"], "NotARealModType") == (None, None))

for mod_type in sorted(_NEEDS_GROUND_TRUTH_REFERENCE):
    rows = df[df["mod_type"] == mod_type]
    missing = []
    for _, r in rows.iterrows():
        gv, gi = find_ground_truth_reference(df, r["gt_sample"], mod_type)
        if gv is None or gi is None:
            missing.append(r["title"])
    check(f"every {mod_type} row ({len(rows)} rows) resolves to a real Comm_Valid+Comm_InValid reference pair",
          len(missing) == 0, str(missing[:10]))

# Spot-check one concrete pair: the reference for a CorrVar row must have
# real variable names (no foobar_N), unlike the row's own obfuscated code.
corrvar_invalid = df[df["mod_type"] == "NoComm_CorrVar_InValid"].iloc[0]
gt_valid, gt_invalid = find_ground_truth_reference(df, corrvar_invalid["gt_sample"], "NoComm_CorrVar_InValid")
check("CorrVar's ground-truth reference has no foobar_N obfuscation",
      "foobar" not in gt_valid and "foobar" not in gt_invalid, (gt_valid[:100], gt_invalid[:100]))
check("the row's own code IS obfuscated (confirms the reference is actually different)",
      "foobar" in str(corrvar_invalid["code"]))


# ── build_caveat_note ─────────────────────────────────────────────────────────

print("\n── build_caveat_note ──")

check("Comm_InValid gets no caveat (empty string)", build_caveat_note("Comm_InValid", None) == "")
check("unrecognized mod_type gets no caveat", build_caveat_note("NotARealModType", None) == "")
check("NoComm_InValid caveat mentions no comments", "no comments" in build_caveat_note("NoComm_InValid", None))
check("NoComm_CorrVar_InValid caveat mentions obfuscated variable names",
      "obfuscated" in build_caveat_note("NoComm_CorrVar_InValid", None) and "foobar_N" in build_caveat_note("NoComm_CorrVar_InValid", None))
corrcomm_note = build_caveat_note("CorrComm_Invalid", "NavierStokes")
check("CorrComm_Invalid caveat substitutes the corruption_source_pde", "NavierStokes" in corrcomm_note, corrcomm_note)
check("CorrComm_Invalid caveat falls back to generic text when corruption_source_pde is None",
      "a different PDE class" in build_caveat_note("CorrComm_Invalid", None))


# ── build_judge_prompt structural checks ─────────────────────────────────────

print("\n── build_judge_prompt (structural) ──")

no_ref_prompt = build_judge_prompt("BASE_CODE", "CORRUPTED_CODE", "NOTE_TEXT", "MODEL_EXP")
check("no-reference prompt omits ground_truth blocks entirely",
      "ground_truth_valid_code" not in no_ref_prompt and "ground_truth_invalid_code" not in no_ref_prompt)
check("no-reference prompt still includes the base/corrupted/note/exp content",
      all(s in no_ref_prompt for s in ["BASE_CODE", "CORRUPTED_CODE", "NOTE_TEXT", "MODEL_EXP"]))

ref_prompt = build_judge_prompt(
    "BASE_CODE", "CORRUPTED_CODE", "NOTE_TEXT", "MODEL_EXP",
    gt_valid_code="GT_VALID_CODE", gt_invalid_code="GT_INVALID_CODE",
    caveat_note="CAVEAT_TEXT",
)
check("reference prompt includes all 4 code blocks + caveat + note + exp",
      all(s in ref_prompt for s in ["BASE_CODE", "GT_VALID_CODE", "CORRUPTED_CODE", "GT_INVALID_CODE",
                                     "CAVEAT_TEXT", "NOTE_TEXT", "MODEL_EXP"]))

# Block order: caveat -> base -> gt_valid -> corrupted -> gt_invalid -> note -> exp
order = [ref_prompt.index(s) for s in
         ["CAVEAT_TEXT", "BASE_CODE", "GT_VALID_CODE", "CORRUPTED_CODE", "GT_INVALID_CODE", "NOTE_TEXT", "MODEL_EXP"]]
check("blocks appear in the agreed order (caveat, base, gt_valid, corrupted, gt_invalid, note, exp)",
      order == sorted(order), str(order))


print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
