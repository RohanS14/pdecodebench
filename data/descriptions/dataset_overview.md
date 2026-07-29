# PDE Benchmark Dataset — Overview

**Current version:** `pdedata_clean_v5.xlsx`  
**Rows:** 256 (32 ground-truth problems × 8 modification types)

---

## Structure

32 PDE solver code samples (`gt_sample`), each from one of 4 PDE classes:

| `pde_class` | `phys_process` | `num_method` |
|---|---|---|
| wave | oscillation | explicit / implicit / spectral |
| heat | diffusion | |
| burgers | advection + diffusion | |
| navier-stokes | advection + diffusion + restoration | |

Each `gt_sample` appears in 8 modification types (`mod_type`):

| `mod_type` | Comments | Variables | `phys_valid` |
|---|---|---|---|
| `Comm_Valid` | GT comments | GT names | True |
| `NoComm_Valid` | None | GT names | True |
| `CorrComm` | Donor comments | GT names | True |
| `NoComm_CorrVar` | None | Obfuscated (`foobar_N`) | True |
| `Comm_InValid` | GT comments | GT names | False |
| `NoComm_InValid` | None | GT names | False |
| `CorrComm_Invalid` | Donor comments | GT names | False |
| `NoComm_CorrVar_InValid` | None | Obfuscated (`foobar_N`) | False |

**Balance:** 64 rows per `pde_class`, 32 rows per `mod_type`, 128 valid / 128 invalid.

---

## Modification Details

**CorrComm / CorrComm_Invalid:** Donor comments are injected at the receiver's comment positions. Donor is selected from `Comm_Valid` rows with a different `pde_class` AND different `num_method`. The same donor is reused for the valid and invalid variant of each `gt_sample` so the only axis of variation is `phys_valid`.

**NoComm_CorrVar / NoComm_CorrVar_InValid:** Variable names obfuscated via AST renaming (`foobar_1`, `foobar_2`, …). The mapping is derived from `NoComm_Valid` and applied directly to `NoComm_InValid`, extended for any new variables introduced by the invalid code. This ensures shared variables have identical obfuscated names across validity conditions.

---

## Key Columns

| Column | Description |
|---|---|
| `gt_sample` | Base problem ID, e.g. `Wave_1` |
| `mod_type` | Modification type (see table above) |
| `pde_class` | PDE type |
| `phys_process` | Physical process(es), `/`-separated |
| `phys_valid` | Bool — physically valid implementation |
| `num_method` | Numerical method(s), `/`-separated |
| `num_comments` | Comment line count |
| `corruption_source_id` | Donor title (CorrComm rows only) |
| `corruption_source_pde` | Donor PDE class (CorrComm rows only) |
| `injected_comments` | Per-comment injection metadata (CorrComm rows only) |
| `invalidity_note` | Short failure-mode description (v5+ new samples' invalid rows only, NaN elsewhere) |

---

## Version History

### v5 — current (`pdedata_clean_v5.xlsx`)

Built from `pdedata_clean_v4.xlsx` (kept as-is) plus 16 new base problems curated by
Shreya (`data/newcode_jul28.txt`, parsed by `datagen/parse_newcode.py`), combined and
augmented by `datagen/build_v5.py`. **256 rows** (32 `gt_sample` x 8 `mod_type`,
double the v4 base-problem count), same 4-per-class balance (`Heat_5`-`Heat_8`,
`Wave_5`-`Wave_8`, `Burgers_5`-`Burgers_8`, `NavierStokes_5`-`NavierStokes_8`).

Intermediate files (also new, neither overwrites anything from v1-v4):
- `pdedata_newcode_v5_base.xlsx` — new material only, 16 samples x 4 core mod_types
  (`Comm_Valid`/`NoComm_Valid`/`Comm_InValid`/`NoComm_InValid`) = 64 rows.
- `pdedata_clean_v5_base.xlsx` — combined base (old + new), 32 samples x 4 core
  mod_types = 128 rows, pre-`CorrComm`/`NoComm_CorrVar` augmentation.

**New column:** `invalidity_note` — a short free-text description of the invalid
variant's failure mode (e.g. "blow ups, overflow", "breaks symmetry"), authored by
Shreya per new `gt_sample` and propagated across all 4 of its invalid `mod_type`
rows. Purely additive: `NaN` for every v1-v4 row and for all valid-`mod_type` rows.

**Source-material handling (new samples only):** Shreya's snippets are full
end-to-end generation artifacts — module docstrings naming the PDE/method, and
trailing matplotlib/animation code whose titles and saved filenames spell out the
PDE class and numerical method in plain text. Both are stripped entirely before
the code enters the dataset (the docstring text is mined as *input* to the
`phys_process`/`num_method`/`invalidity_note` tags, then discarded), since the
`NoComm_*` condition otherwise wouldn't hide PDE identity regardless of variable
obfuscation. `phys_process`/`num_method` tags for the 16 new samples were proposed
with reasoning and reviewed via `data/descriptions/newcode_v5_tag_review.csv`
before being locked in; a few borderline calls are flagged "NEEDS REVIEW" in that
file's `reasoning` column (Wave_8 Newmark-beta damping; NavierStokes_5/6
restoration-vs-diffusion-only).

**Fix during construction:** the new snippets follow their generation prompt's
"inline comments" instruction, so almost all of their comments are *trailing*
(`code  # comment`) rather than the whole-line style (`# comment` on its own line)
that the existing `corrupt_comment.py`/`_strip_comments` machinery detects. Left
alone, this would have made `NoComm_*` not actually comment-free and `CorrComm` a
near no-op for the new samples (only one generic whole-line comment,
"Domain and physical parameters," existed to swap). `parse_newcode.py` normalizes
every new snippet's trailing comments onto their own line (same indentation) via
`tokenize`, before anything reaches the existing pipeline scripts, which remain
unmodified.

### v4 (`pdedata_clean_v4.xlsx`)

Built from `pdedata_clean_v4_base.xlsx` by `datagen/build_v4.py`.

**Fixes:**
- **Burgers Initialization Bug:** `Burgers_2`, `Burgers_3`, and `Burgers_4` were natively missing their initialization code (`u_init`, `dx`, `t`) in their `Comm_Valid`, `Comm_InValid`, and `NoComm_InValid` variants, causing immediate `NameError` execution crashes. The missing initialization blocks were extracted from `NoComm_Valid` and injected into the broken variants in `v4_base`, ensuring they execute correctly while preserving physical validity/invalidity properties.
- **Literal Newline Parsing:** Removed escaping artifacts (`\\n`) that were improperly saved as literal characters in the code.
- **Note on `Burgers_1`:** This sample continues to remain unexecutable across all variants as it entirely lacks initialization code in the original raw dataset as well.

### v3 (`pdedata_clean_v3.xlsx`)

Built from v2 by `datagen/build_v3.py`.

**New mod_types added:**
- `CorrComm_Invalid` — corrupted comments on physically invalid code
- `NoComm_CorrVar_InValid` — obfuscated variables on physically invalid code

These were added to remove the confound between surface corruption and validity (previously, `CorrComm` and `NoComm_CorrVar` only existed as valid variants).

**Fixes:**
- **Randomized donor assignment (CorrComm):** Previously, the donor was selected deterministically as the alphabetically first qualifying `gt_sample`, meaning all receivers of the same `pde_class` got the same donor. This created a donor-fingerprint shortcut for probes. Now donors are assigned randomly (seed=42) with a cap of 4 uses per donor `pde_class` to prevent any one class dominating.
- **Shared variable mapping (NoComm_CorrVar_InValid):** The `foobar_N` mapping is inherited from the valid counterpart and only extended for new variables, ensuring probes cannot use variable-name identity as a validity signal.
- **Burgers_2 NoComm_InValid stray comments:** The source code had 4 comment lines that were never stripped. Fixed by applying comment stripping to all `NoComm_*` rows during build.
- **Trailing whitespace in `num_method` / `phys_process`:** Two `num_method` values had trailing spaces which caused silent mismatches in donor constraint logic. Stripped during build.
- **`difffusion` typo:** 8 rows had `phys_process = "difffusion"`. Fixed to `"diffusion"`.

### v2 (`pdedata_clean_v2.xlsx`)

Added `Comm_InValid` (GT comments injected into `NoComm_InValid`) to v1. 96 rows, 6 mod_types.

### v1 (`pdedata_clean.xlsx`)

Original dataset with 5 mod_types. Source code provided by Lohit and Shreya.
