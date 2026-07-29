# TODO — v5 dataset expansion (Shreya's 16 new base problems)

Plan reference: `/Users/rohansubramanian/.claude/plans/tingly-jingling-eich.md`

- [x] 1. Write `datagen/parse_newcode.py` — parses `data/newcode_jul28.txt`, strips
      docstrings/plotting code, proposes `phys_process`/`num_method`/`invalidity_note` tags
- [x] 2. Run it to generate `data/descriptions/newcode_v5_tag_review.csv` (human-review gate)
      — verified: 16/16 snippets parsed, 64 base rows, all AST-valid, zero plotting residue
- [x] 3. Self-review the proposed tag CSV for sanity before locking in (flagged 3
      "NEEDS REVIEW" entries in the reasoning column for Shreya's later sign-off:
      Wave_8 Newmark-beta damping, NavierStokes_5/6 restoration-vs-diffusion-only)
- [x] 4. Write `datagen/build_v5.py` orchestrator (mirrors `build_v4.py`)
- [x] 5. Run `build_v5.py` end-to-end:
  - [x] `data/pdedata_newcode_v5_base.xlsx` (64 rows, new-only base)
  - [x] `data/pdedata_clean_v5_base.xlsx` (128 rows, combined base)
  - [x] `data/pdedata_clean_v5.xlsx` (256 rows, final combined dataset)
- [x] 6. Run `datagen/audit_dataset.py data/pdedata_clean_v5.xlsx` → confirm all checks pass
      (0 issues: no typos, phys_valid schema OK, all 32 gt_samples have all 8 mod_types)
- [x] 7. Row-count / balance verification (32/class, 128 valid/128 invalid, 32/mod_type) —
      all confirmed exactly balanced
- [x] 8. Spot-check 2-3 new `gt_sample`s across all 8 `mod_type` rows — caught a real bug:
      new snippets use inline trailing comments (`code  # comment`) but the existing
      pipeline only detects whole-line comments, so `NoComm_*` wasn't actually
      comment-free and `CorrComm` was a near no-op. Fixed via a `tokenize`-based
      normalization pass in `parse_newcode.py` (hoists trailing comments onto their
      own line, matching the old dataset's convention) — re-ran `build_v5.py`,
      verified `NoComm_Valid`/`NoComm_InValid` now have 0 comments and `CorrComm`
      is genuinely donor-corrupted. No changes needed to the untouched pipeline
      scripts. Re-confirmed audit passes and old files still byte-identical.
- [x] 9. Confirm v1-v4 xlsx files + `physics_code.xlsx` are byte-for-byte untouched
      (md5sum diff before/after build_v5.py run — no changes)
- [x] 10. Append `### v5` entry to `data/descriptions/dataset_overview.md`, and update the
       top-of-file "current version"/row-count/balance/Key Columns sections to match v5
       (these describe the live schema, not history, so they needed updating too)
- [x] 11. Note follow-up risks for later (not blocking): `eval/verify_simulations.py`
       spot-check on subtle invalid variants; scan for leaked physics-revealing kwarg names
       — both still open, listed below for whoever picks this up next

## Follow-up (not done in this pass — flagged, not blocking)

- Run `eval/verify_simulations.py` (or similar) against all 32 new-derived code
  variants to confirm valid rows execute cleanly and invalid rows actually trip
  NaN/Inf/spike detection. A few of Shreya's invalid variants are subtle
  ("breaks symmetry", "heat spreads unevenly by direction", "values climb up
  linearly") and may not trigger simple magnitude/NaN heuristics — same
  clear-vs-ambiguous distinction the paper draws in Appendix A.7.2.
- Scan the `NoComm_CorrVar`/`NoComm_CorrVar_InValid` AST-renamed output of all 32
  new samples for any physics-revealing *keyword argument names* surviving
  unobfuscated (the reason `augment_foobar_vars.py` has one-off
  `_patch_ns4_kwargs`/`_patch_wave4_kwargs` for two old samples). Quick manual
  read confirmed no external PDE-specific library calls with revealing kwargs in
  the new material, but worth a closer pass.
- Get Shreya's sign-off on `data/descriptions/newcode_v5_tag_review.csv`,
  especially the 3 "NEEDS REVIEW" `phys_process`/`num_method` entries (Wave_8
  Newmark-beta damping tag; NavierStokes_5/6 restoration-vs-diffusion-only).
