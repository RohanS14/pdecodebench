# PDE Benchmark Dataset — Overview

**Current version:** jul28 release (6 CSVs under `data/`, see below)
**Rows:** merged_mod_jul28.csv has 256 (32 ground-truth problems × 8 modification types)

Older dataset versions (`pdedata_clean.xlsx` through `pdedata_clean_v5.xlsx`, plus
`physics_code.xlsx`) have been moved to `data/archive/` — kept for reference, not
actively read by any current build script (except where noted below).

---

## Structure

32 PDE solver code samples (`gt_sample`) across two independently-tracked sources,
each contributing 16 samples (4 per PDE class):

- **human** — hand-written by Lohit/Shreya, fixed+expanded jul28
  (`data/Physics_Code_HumanGen.xlsx`, parsed by `datagen/parse_humangen.py`)
- **synthetic** — LLM-generated, human-verified by Shreya, jul28
  (`data/newcode_jul28.txt`, parsed by `datagen/parse_newcode.py`)

| `pde_class` | `phys_process` | `num_method` |
|---|---|---|
| wave | oscillation | explicit / implicit / spectral |
| heat | diffusion | |
| burgers | advection + diffusion | |
| navier-stokes | advection + diffusion + restoration | |

### Base files (pre-mod-type-expansion, row-per-variant)

One row per `(gt_sample, valid/invalid)` pair, titled `{Class}_Valid_{i}` /
`{Class}_InValid_{i}`. This is the canonical commented (`Comm_Valid`/`Comm_InValid`
equivalent) code, before the 8-way `mod_type` expansion.

| File | Rows | Contents |
|---|---|---|
| `synthetic_base_jul28.csv` | 32 | 16 synthetic gt_samples x Valid/InValid |
| `human_base_jul28.csv` | 32 | 16 human gt_samples x Valid/InValid |
| `merged_base_jul28.csv` | 64 | concat of the above, + `source` column (`human`/`synthetic`) |

### Mod files (full 8-way modification-type expansion)

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

| File | Rows | Contents |
|---|---|---|
| `synthetic_mod_jul28.csv` | 128 | 16 x 8, CorrComm/CorrVar donors drawn from synthetic's own 16 only |
| `human_mod_jul28.csv` | 128 | 16 x 8, donors drawn from human's own 16 only |
| `merged_mod_jul28.csv` | 256 | concat of the above, + `source` column |

**Important:** donor selection for `CorrComm`/`NoComm_CorrVar` never crosses the
human/synthetic boundary — `merged_mod_jul28.csv` is a straight concatenation of
`human_mod_jul28.csv` and `synthetic_mod_jul28.csv`, not a fresh donor-pool build
over all 32 samples. Each `_mod` file is independently reproducible and
interpretable on its own.

**Balance (merged_mod):** 64 rows per `pde_class`, 32 rows per `mod_type`,
128 valid / 128 invalid.

---

## Modification Details

**CorrComm / CorrComm_Invalid:** Donor comments are injected at the receiver's comment positions. Donor is selected from `Comm_Valid` rows with a different `pde_class` AND different `num_method`, **within the same source group (human or synthetic) only**. The same donor is reused for the valid and invalid variant of each `gt_sample` so the only axis of variation is `phys_valid`.

**NoComm_CorrVar / NoComm_CorrVar_InValid:** Variable names obfuscated via AST renaming (`foobar_1`, `foobar_2`, …). The mapping is derived from `NoComm_Valid` and applied directly to `NoComm_InValid`, extended for any new variables introduced by the invalid code. This ensures shared variables have identical obfuscated names across validity conditions.

**Comment density (human vs synthetic):** measured on `Comm_Valid` rows —
human mean 6.12 comments/sample (n=16, range 5-10), synthetic mean 5.81 (n=16,
range 4-8). Ratio 0.95x — comparable, no flag.

---

## Key Columns

| Column | Description |
|---|---|
| `gt_sample` | Base problem ID, e.g. `Wave_1` |
| `mod_type` | Modification type (mod files only, see table above) |
| `pde_class` | PDE type |
| `phys_process` | Physical process(es), `/`-separated |
| `phys_valid` | Bool — physically valid implementation |
| `num_method` | Numerical method(s), `/`-separated |
| `num_comments` | Comment line count |
| `corruption_source_id` | Donor title (CorrComm rows only, mod files only) |
| `corruption_source_pde` | Donor PDE class (CorrComm rows only, mod files only) |
| `injected_comments` | Per-comment injection metadata (CorrComm rows only, mod files only) |
| `invalidity_note` | Short failure-mode description for invalid rows, NaN for valid rows |
| `source` | `human` or `synthetic` (merged files only) |

---

## Version History

### jul28 — current (6 CSVs, see Structure above)

Full two-source rebuild. Old single-file xlsx versions (v1-v5) moved to
`data/archive/`.

**Human source fixes (`data/Physics_Code_HumanGen.xlsx` -> `datagen/parse_humangen.py`):**
- **Literal `\n` artifact:** every Wave/Heat row carried a literal backslash-n
  immediately before the real newline (e.g. `"import numpy as np\\n\nfrom..."`).
  Needed 3 passes to fully clean: the base pattern, a regex variant for cases
  where stray whitespace separated the artifact from the real newline
  (`Wave_NoComm_Valid_1`/`_2`), and explicit handling of the trailing artifact on
  each row's last line (no real newline follows it there to pair against). Verified
  by reproducing `Heat_Comm_Valid_3`'s own stated `Num Lines` (38) exactly.
- **`Num characters`** was wrong on 37/40 rows; **`Num Lines`** was empty on all 16
  Burgers/NavierStokes rows. Both recomputed fresh, never trusted from source.
- **`Phys Valid`** mixed `Yes`/`yes`/`No`/`no` casing -> normalized to bool.
- **`Numerical method`** had trailing whitespace on 3 of 6 distinct values -> stripped.
- **`PDE Classification`** used full names (`"Wave Equation"`) -> mapped to the slug
  vocab (`wave`/`heat`/`burgers`/`navier-stokes`).
- **Titles:** Burgers/NavierStokes were `Class_Valid1`/`Class_Invalid1` (no
  underscore before the index, no mod_type segment) -> normalized to
  `Class_Comm_Valid_1`/`Class_Comm_InValid_1` (content confirmed byte-identical to
  the old v4 dataset for these classes, including `NavierStokes_3`'s real
  mpi4py/mpi4py_fft distributed-FFT code and `NavierStokes_4`'s jax_cfd code).
- **`Invalid Change Type`** sometimes prefixed "Why Invalid: ", sometimes not ->
  prefix stripped for a consistent `invalidity_note`.
- **`NavierStokes_3`'s hardcoded assertion:** the source script ended with
  `assert round(float(k) - 0.124953117517, 7) == 0`, an energy-conservation
  self-check that fires on every invalid variant by construction (perturbed
  physics never matches that reference constant), crashing execution before any
  NaN/spike anomaly could be inspected. The `assert` line is stripped (print and
  `FFT.destroy()` kept) from both valid/invalid raw code before any mod_type
  derivation, so the fix propagates to all 8 `NavierStokes_3` variants.
- **Missing mod_types derived:** Wave/Heat had `Comm_Valid`/`NoComm_Valid`/
  `NoComm_InValid` given directly but not `Comm_InValid` — derived via
  `datagen/build_comm_invalid.py`'s existing position-based comment injection
  (unmodified). Burgers/NavierStokes had `Comm_Valid`/`Comm_InValid` given directly
  but not `NoComm_Valid`/`NoComm_InValid` — derived via simple `#`-line stripping
  (comments confirmed whole-line style, unlike the synthetic source).
- **`NoComm_Valid` is never trusted from source, even where given directly:**
  found by diffing every `Comm_Valid`/`NoComm_Valid` pair (after stripping
  comments/blank-lines/whitespace) — `Heat_1`'s separately-authored `NoComm_Valid`
  row had drifted from `Comm_Valid` by a real numeric value (`t_steps = 1000` vs
  `1001`), not just comments. `NoComm_Valid` is now *always* derived from
  `Comm_Valid` by stripping comments, for all 4 PDE classes uniformly — verified
  this changes nothing for Heat_2/3/4 and Wave_1-4 (already matched exactly) and
  fixes Heat_1. Also re-verified zero import-statement differences between any
  mod_type pair across all 32 gt_samples (a separately reported concern that
  turned out not to reproduce — the real issue was this numeric drift).
- Sheet2 of the source xlsx is a stale duplicate of the Burgers/NavierStokes rows
  (byte-identical code, missing `Num Lines`) — ignored.

**Synthetic source:** same `parse_newcode.py` as before (docstring/plot stripping,
inline-comment-to-whole-line normalization via `tokenize`) — see the entry below.
Now also emits the 32-row base format via a new `build_base_rows()` function
mirroring `parse_humangen.py`'s.

**Execution check** (`eval/verify_simulations.py`, extended to accept CSV input and
an output-dir CLI arg) against all 256 `merged_mod_jul28.csv` rows:
**256/256 execute, 0 errors, 0 false positives on valid code** (after stripping
`NavierStokes_3`'s hardcoded assertion -- see fix above; all 4 of its invalid
variants now run to completion and are correctly flagged as anomalous via NaN
propagation from the division-by-zero bug). 36/128 invalid rows (28%) don't trip
the NaN/magnitude-spike heuristic — cross-checked every one against its own
`invalidity_note`: 100% match a genuinely subtle failure mode (jagged/scattered
solution, spikes without blow-up, symmetry breaking, uneven spreading, linear
drift, boundary violation, losing oscillatory character), not pipeline defects.
Specifically verified `Heat_3` (human, bounded/no-NaN, matches its subtle
"negative temperatures" note) and `Wave_4` (human, genuinely produces NaN
throughout when its `if __name__ == "__main__":` block is executed, matches its
"Causes NaNs" note) both behave exactly as designed.

### v5 (`data/archive/pdedata_clean_v5.xlsx`)

Built from `pdedata_clean_v4.xlsx` (kept as-is) plus 16 new base problems curated by
Shreya (`data/newcode_jul28.txt`, parsed by `datagen/parse_newcode.py`), combined and
augmented by `datagen/build_v5.py`. **256 rows** (32 `gt_sample` x 8 `mod_type`,
double the v4 base-problem count), same 4-per-class balance (`Heat_5`-`Heat_8`,
`Wave_5`-`Wave_8`, `Burgers_5`-`Burgers_8`, `NavierStokes_5`-`NavierStokes_8`).

Intermediate files (also archived):
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
(still live — read directly by the jul28 build too) before being locked in; a few
borderline calls are flagged "NEEDS REVIEW" in that file's `reasoning` column
(Wave_8 Newmark-beta damping; NavierStokes_5/6 restoration-vs-diffusion-only).

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

### v4 (`data/archive/pdedata_clean_v4.xlsx`)

Built from `pdedata_clean_v4_base.xlsx` by `datagen/build_v4.py`.

**Fixes:**
- **Burgers Initialization Bug:** `Burgers_2`, `Burgers_3`, and `Burgers_4` were natively missing their initialization code (`u_init`, `dx`, `t`) in their `Comm_Valid`, `Comm_InValid`, and `NoComm_InValid` variants, causing immediate `NameError` execution crashes. The missing initialization blocks were extracted from `NoComm_Valid` and injected into the broken variants in `v4_base`, ensuring they execute correctly while preserving physical validity/invalidity properties.
- **Literal Newline Parsing:** Removed escaping artifacts (`\\n`) that were improperly saved as literal characters in the code.
- **Note on `Burgers_1`:** This sample continues to remain unexecutable across all variants as it entirely lacks initialization code in the original raw dataset as well.

### v3 (`data/archive/pdedata_clean_v3.xlsx`)

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

### v2 (`data/archive/pdedata_clean_v2.xlsx`)

Added `Comm_InValid` (GT comments injected into `NoComm_InValid`) to v1. 96 rows, 6 mod_types.

### v1 (`data/archive/pdedata.xlsx`)

Original dataset with 5 mod_types. Source code provided by Lohit and Shreya.
