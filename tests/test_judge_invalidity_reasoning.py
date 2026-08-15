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


print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
