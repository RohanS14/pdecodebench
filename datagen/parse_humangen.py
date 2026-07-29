"""
parse_humangen.py — Parse data/Physics_Code_HumanGen.xlsx (Lohit/Shreya's
human-authored PDE solvers, fixed+expanded jul28 version) into the same core
schema used by parse_newcode.py, so both sources can be merged cleanly.

Source file structure (Sheet1 is authoritative; Sheet2 is a stale duplicate of
the Burgers/NavierStokes rows minus the Num Lines column — verified
byte-identical code for all 16 overlapping titles, so it's ignored entirely):

  - Wave/Heat (24 rows): Comm_Valid, NoComm_Valid, NoComm_InValid given
    directly, titled "{Class}_{mod_type}_{i}". Comm_InValid is MISSING —
    derived here via datagen/build_comm_invalid.py's existing position-based
    comment injection (Comm_Valid's comments -> NoComm_InValid).
  - Burgers/NavierStokes (16 rows): only "{Class}_Valid{i}" / "{Class}_Invalid{i}"
    given (no underscore before the index, no mod_type segment) — this content
    is byte-identical to the old v4 dataset's Comm_Valid/Comm_InValid for these
    classes (comments already whole-line style, inline NavierStokes_3/_4 heavy
    deps — mpi4py/mpi4py_fft, jax/jax_cfd — unchanged). NoComm_Valid/
    NoComm_InValid are MISSING — derived here via simple '#'-line stripping
    (no inline-comment issue like the synthetic data, confirmed by inspection).

Known data-quality bugs fixed here (not just documented — the fixes are
mechanical/unambiguous, not judgment calls):
  - Literal '\\n' artifact: every Wave/Heat line carries a literal backslash-n
    immediately before the real newline (e.g. "import numpy as np\\n\\nfrom...").
    Fixed via `code.replace('\\\\n\\n', '\\n')` — verified this exact transform
    reproduces the row's own stated Num Lines value where present (38 for
    Heat_Comm_Valid_3), so it is not a guess.
  - `Num characters` wrong on 37/40 rows, `Num Lines` empty on all 16
    Burgers/NavierStokes rows — both recomputed fresh, never trusted from source.
  - `Phys Valid` mixes Yes/yes/No/no casing — normalized to bool.
  - `Numerical method` has trailing whitespace on half its distinct values —
    stripped.
  - `PDE Classification` uses full names ("Wave Equation") — mapped to the
    slug vocab (wave/heat/burgers/navier-stokes) used everywhere else.
  - `Invalid Change Type` sometimes prefixed "Why Invalid: " and sometimes
    not — prefix stripped for a consistent `invalidity_note` across all rows.
  - `NavierStokes_3`'s source script ends with a hardcoded energy-conservation
    self-check (`assert round(float(k) - 0.124953117517, 7) == 0`) that fires on
    every invalid variant by construction (the perturbed physics never matches
    that reference constant) — this crashes execution instead of letting the
    invalid trajectory run to completion, so no _mod variant of NavierStokes_3
    can ever be inspected for NaN/spike anomalies. The `assert` line is stripped
    (print + `FFT.destroy()` kept) from both the valid and invalid raw code
    before any mod_type derivation, so the fix propagates to every downstream
    variant (NoComm_*, CorrComm, NoComm_CorrVar, ...).

Public entry points:
  load_raw_rows(path)          -> list of dicts, one per source row, normalized
  build_core_mod_rows(path)    -> list of dicts, the 4 core mod_types
                                   (Comm_Valid/NoComm_Valid/Comm_InValid/
                                   NoComm_InValid) x 16 gt_samples = 64 rows
  build_base_rows(core_rows)   -> list of dicts, 32 rows (Valid/InValid only,
                                   titled "{Class}_Valid_{i}"/"{Class}_InValid_{i}")
"""

import re

import pandas as pd

HUMANGEN_PATH = "data/Physics_Code_HumanGen.xlsx"

PDE_CLASS_MAP = {
    "wave equation": ("Wave", "wave"),
    "heat equation": ("Heat", "heat"),
    "burgers equation": ("Burgers", "burgers"),
    "navier stokes equation": ("NavierStokes", "navier-stokes"),
}

# Wave/Heat titles: "{Class}_{ModType}_{i}"
WAVEHEAT_TITLE_RE = re.compile(r"^(Wave|Heat)_(Comm_Valid|NoComm_Valid|NoComm_InValid)_(\d+)$")
# Burgers/NavierStokes titles: "{Class}_{Valid|Invalid}{i}" (no underscore before digit)
BURGERSNS_TITLE_RE = re.compile(r"^(Burgers|NavierStokes)_(Valid|Invalid)(\d+)$")


_LITERAL_NEWLINE_RE = re.compile(r"\\n[ \t]*\n")


def _fix_literal_newlines(code: str) -> str:
    code = _LITERAL_NEWLINE_RE.sub("\n", code)
    # the last line has no trailing real newline to pair with, so a literal
    # '\n' artifact there survives the pass above -- strip it explicitly.
    code = code.rstrip()
    if code.endswith("\\n"):
        code = code[: -len("\\n")]
    return code


def _strip_comments(code: str) -> str:
    return "\n".join(l for l in code.split("\n") if not l.strip().startswith("#"))


def _strip_ns3_assertion(code: str) -> str:
    """NavierStokes_3 only: remove the hardcoded energy-conservation `assert`
    (fires on every invalid variant by construction, crashing execution before
    any NaN/spike anomaly can be inspected). Keeps the surrounding print and
    FFT.destroy() untouched."""
    return "\n".join(
        l for l in code.split("\n") if not l.strip().startswith("assert ")
    )


def _clean_invalidity_note(raw) -> str | None:
    if pd.isna(raw):
        return None
    note = str(raw).strip()
    if note.lower().startswith("why invalid:"):
        note = note[len("why invalid:"):].strip()
    return note


def load_raw_rows(path: str = HUMANGEN_PATH) -> list[dict]:
    s1 = pd.read_excel(path, sheet_name="Sheet1")
    rows = []
    for _, r in s1.iterrows():
        title_raw = str(r["Title"]).strip()

        m1 = WAVEHEAT_TITLE_RE.match(title_raw)
        m2 = BURGERSNS_TITLE_RE.match(title_raw)
        if m1:
            prefix, mod_type, idx = m1.group(1), m1.group(2), int(m1.group(3))
        elif m2:
            prefix, variant, idx = m2.group(1), m2.group(2), int(m2.group(3))
            mod_type = "Comm_Valid" if variant == "Valid" else "Comm_InValid"
        else:
            raise ValueError(f"Unrecognized title format: {title_raw!r}")

        code = _fix_literal_newlines(str(r["Code"]))
        if prefix == "NavierStokes" and idx == 3:
            code = _strip_ns3_assertion(code)
        pde_class_raw = str(r["PDE Classification "]).strip().lower()
        if pde_class_raw not in PDE_CLASS_MAP:
            raise KeyError(f"Unrecognized PDE Classification: {pde_class_raw!r}")
        _, pde_class = PDE_CLASS_MAP[pde_class_raw]

        phys_valid_raw = str(r["Phys Valid"]).strip().lower()
        phys_valid = phys_valid_raw == "yes"

        rows.append(
            {
                "title": f"{prefix}_{mod_type}_{idx}",
                "code": code,
                "num_lines": len(code.split("\n")),
                "num_char": len(code),
                "pde_class": pde_class,
                "phys_process": str(r["Dominant Phys Process"]).strip(),
                "phys_valid": phys_valid,
                "num_method": str(r["Numerical method"]).strip(),
                "corruption_source_id": None,
                "corruption_source_pde": None,
                "injected_comments": None,
                "delta_comments": None,
                "num_comments": sum(1 for l in code.split("\n") if l.strip().startswith("#")),
                "gt_sample": f"{prefix}_{idx}",
                "mod_type": mod_type,
                "invalidity_note": _clean_invalidity_note(r["Invalid Change Type"]),
            }
        )
    return rows


def build_core_mod_rows(path: str = HUMANGEN_PATH) -> list[dict]:
    """Returns 64 rows: 16 gt_samples x 4 core mod_types (Comm_Valid,
    NoComm_Valid, Comm_InValid, NoComm_InValid), deriving whichever of these
    aren't given directly in the source file.

    NoComm_Valid is ALWAYS derived from Comm_Valid by stripping comments --
    never trusted from source, even for Wave/Heat where the source file gives
    a separately-authored NoComm_Valid row directly. Found by diffing: Heat_1's
    given NoComm_Valid had genuinely drifted from Comm_Valid beyond just
    comments (t_steps = 1000 vs 1001, a real numeric difference, presumably a
    copy-paste slip when the row was typed out independently). Deriving
    uniformly for all 4 classes eliminates this whole class of drift risk --
    verified this changes nothing for Heat_2/3/4 and Wave_1-4, where the given
    NoComm_Valid already exactly matched strip-comments(Comm_Valid).

    NoComm_InValid, by contrast, is NOT mechanically recoverable from
    Comm_Valid/Comm_InValid (it encodes the actual invalid-physics change) --
    for Wave/Heat it's trusted from source (and needed there as the input to
    the Comm_InValid derivation below); for Burgers/NavierStokes it's derived
    from Comm_InValid by stripping comments (comments are whole-line style
    there, so this is safe and was already the previous approach)."""
    from build_comm_invalid import build_comm_invalid

    raw_rows = load_raw_rows(path)
    df = pd.DataFrame(raw_rows)

    # Derive Comm_InValid for Wave/Heat via position-based comment injection
    # (uses the given NoComm_InValid + Comm_Valid, both trusted from source)
    comm_invalid_derived = build_comm_invalid(df)
    if len(comm_invalid_derived) > 0:
        for col in df.columns:
            if col not in comm_invalid_derived.columns:
                comm_invalid_derived[col] = pd.NA
        df = pd.concat([df, comm_invalid_derived[df.columns]], ignore_index=True)

    # Drop the given NoComm_Valid rows entirely -- re-derived below for all classes
    df = df[df["mod_type"] != "NoComm_Valid"].reset_index(drop=True)

    nocomm_rows = []
    # NoComm_Valid: derive from Comm_Valid for ALL classes (never trust source)
    for _, row in df[df["mod_type"] == "Comm_Valid"].iterrows():
        code = _strip_comments(row["code"])
        new_row = row.to_dict()
        new_row.update(
            {
                "code": code,
                "num_lines": len(code.split("\n")),
                "num_char": len(code),
                "num_comments": 0,
                "mod_type": "NoComm_Valid",
                "title": row["title"].replace("Comm_Valid", "NoComm_Valid"),
            }
        )
        nocomm_rows.append(new_row)

    # NoComm_InValid: derive from Comm_InValid for Burgers/NavierStokes only
    # (Wave/Heat's NoComm_InValid is already present, trusted from source)
    src = df[(df["mod_type"] == "Comm_InValid") & (df["pde_class"].isin(["burgers", "navier-stokes"]))]
    for _, row in src.iterrows():
        code = _strip_comments(row["code"])
        new_row = row.to_dict()
        new_row.update(
            {
                "code": code,
                "num_lines": len(code.split("\n")),
                "num_char": len(code),
                "num_comments": 0,
                "mod_type": "NoComm_InValid",
                "title": row["title"].replace("Comm_InValid", "NoComm_InValid"),
            }
        )
        nocomm_rows.append(new_row)

    df = pd.concat([df, pd.DataFrame(nocomm_rows)], ignore_index=True)
    return df.to_dict("records")


def build_base_rows(core_rows: list[dict]) -> list[dict]:
    """Returns 32 rows: 16 gt_samples x {Valid, InValid}, titled
    '{Class}_Valid_{i}' / '{Class}_InValid_{i}'. Uses the commented
    (Comm_Valid/Comm_InValid) content as the canonical pre-mod-expansion code."""
    rows = []
    for row in core_rows:
        if row["mod_type"] not in ("Comm_Valid", "Comm_InValid"):
            continue
        variant = "Valid" if row["mod_type"] == "Comm_Valid" else "InValid"
        prefix, idx = row["gt_sample"].split("_")
        new_row = dict(row)
        new_row["title"] = f"{prefix}_{variant}_{idx}"
        rows.append(new_row)
    return rows


if __name__ == "__main__":
    import os as _os
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    repo_root = _os.path.dirname(script_dir)
    _os.chdir(repo_root)

    core_rows = build_core_mod_rows()
    print(f"Core mod rows: {len(core_rows)} (expect 64 = 16 gt_samples x 4)")
    import collections
    print(collections.Counter(r["mod_type"] for r in core_rows))

    base_rows = build_base_rows(core_rows)
    print(f"Base rows: {len(base_rows)} (expect 32 = 16 gt_samples x 2)")
