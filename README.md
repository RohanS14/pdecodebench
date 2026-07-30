`data/`: Datasets and descriptions of data. See <a href='https://github.com/RohanS14/pdecodebench/blob/main/data/descriptions/data_spec.txt'>data_spec.txt</a>
- **Current: the jul28 release — six CSVs plus an ablation file.** `merged_mod_jul28.csv` is the canonical eval
  dataset: 256 rows, 32 `gt_sample`s × 8 `mod_type`s. Balance: 64 rows per `pde_class`,
  32 per `mod_type`, 128 valid / 128 invalid.

  | File | Rows | Contents |
  |---|---|---|
  | `merged_mod_jul28.csv` | 256 | full 8-way `mod_type` expansion, both sources, + `source` column |
  | `human_mod_jul28.csv` | 128 | human source only |
  | `synthetic_mod_jul28.csv` | 128 | synthetic source only |
  | `merged_base_jul28.csv` | 64 | pre-expansion base, one row per (`gt_sample`, valid/invalid) |
  | `human_base_jul28.csv` | 32 | human source only |
  | `synthetic_base_jul28.csv` | 32 | synthetic source only |
  | `leak_ablation_jul28.csv` | 48 | **not** part of the eval set — 6 repaired samples x 4 invalid conditions x 2 `leak_variant`s (`widened` = original leaky code, `aligned` = repaired), for measuring the leak rather than only removing it |

  The 32 base problems come from two independently-tracked sources of 16 each. `CorrComm`
  and `NoComm_CorrVar` donors never cross the human/synthetic boundary, so `merged_*` is a
  straight concatenation of the two single-source files, not a fresh build over all 32.
- `Physics_Code_HumanGen.xlsx` — source workbook for the human half.
- `archive/` — all pre-jul28 versions (`pdedata.xlsx` through `pdedata_clean_v5.xlsx`, plus
  `physics_code.xlsx`). Kept for reference; not read by any current build script.
- `descriptions/dataset_overview.md` — schema, column reference, version history, and
  known limitations (including variable-obfuscation coverage).

`datagen/`: Scripts for dataset corruption and augmentation.
- `build_jul28.py` — builds all seven jul28 CSVs end to end. Deterministic: two consecutive
  runs produce byte-identical output.
- `parse_humangen.py` — parses and repairs the human half from `Physics_Code_HumanGen.xlsx`
- `parse_newcode.py` — parses the synthetic half from `data/newcode_jul28.txt`
- `corrupt_comment.py` — `CorrComm` / `CorrComm_Invalid` donor-comment injection
- `augment_foobar_vars.py` — `NoComm_CorrVar` / `_InValid` AST variable obfuscation. Renames
  every author-chosen identifier and nothing else: `visit_Call` moves a same-module callee's
  `keyword.arg` in lockstep with its renamed parameters, while library kwargs
  (`np.zeros(shape=…)`) are always left alone since they are the library's API.
- `audit_dataset.py` — fast balance/integrity check; runs inside the build, also a CLI
- `full_audit.py` — 49 structural and semantic checks (condition semantics, donor
  constraints, label leakage, cross-condition program identity). **49/49 as of 2026-07-30**
- `full_audit_exec.py` — executes all 256 rows and verifies that a `gt_sample`'s four
  surface conditions produce identical numbers
- `derive_invalidity_change.py` — derives *how* each invalid variant was made invalid from
  the valid↔invalid diff, and separates the physical error from incidental drift

Run the audits as batch jobs, never on a login node — `sbatch/run_dataset_audit.sbatch`
(CPU-only, `cpu_short`) runs both. `NavierStokes_3` additionally needs FFTW3 and a real MPI,
neither of which this cluster provides by default; `sbatch/setup_fftw_mpi.sbatch` builds them
once. See the Environment section of `data/descriptions/dataset_overview.md`. Nothing in the
benchmark itself requires executing the code — execution only validates the dataset.
- `build_v3.py` / `build_v4.py` / `build_v5.py`, `patch_*.py`, `remove_assertions.py` —
  historical builders for the archived versions. Their hardcoded input paths still point at
  `data/*.xlsx` rather than `data/archive/*.xlsx`, so they need a path fix before they will run.

`eval/`: Evaluation pipeline for model outputs.
- `dataset_io.py` — shared dataset loading. `DEFAULT_MOD_DATASET` is the single place the
  canonical dataset path is defined; `load_dataset()` reads CSV or xlsx so jul28 CSVs and
  archived xlsx files can both be passed to `--dataset`.
- `run_eval.py` — Experiment 1: free-generation accuracy across 10 LLMs
- `run_mc_eval.py` — Experiment 2: MCQ confidence via logprob extraction (with text-extraction fallback for reasoning models)
- `prepare_var_probes.py` / `run_var_logprob.py` — Experiment 3: variable log-probability evolution (Appendix A.5)
- `frontier/` — Experiment 4: belief revision with execution summaries (Gemini-2.5-Flash)

`sbatch/`: SLURM job scripts. Cluster layout follows
`/scratch/ehb7466/projects/PROJECTS.md`: code in `$HOME/pde-llm-eval` (synced from this
desktop repo), results on scratch under `projects/pde-llm-eval/{outputs,logs}`, venv at
`/scratch/ehb7466/envs/pdecodebench`.
- `setup_fftw_mpi.sbatch` — one-time: builds FFTW3 + MPI-linked `mpi4py`/`mpi4py-fft`
- `run_dataset_audit.sbatch` — full dataset audit on `cpu_short` (CPU-only)
- `run_v3_all_models.sbatch`, `run_mc_v3_all_models.sbatch`, `run_var_logprob*.sbatch` —
  **stale, will not run as-is.** Each still points at `/scratch/ehb7466/pde-llm-eval`,
  `${WORKSPACE}/venv`, and the archived `pdedata_clean_v3.xlsx`; all three paths moved. They
  need rewiring to the layout above before any eval job is submitted.

`probe/`: Probing experiments on model hidden states (Experiment 5).

`results/`: Eval outputs and model responses for experiments 1 and 2.

`dataset_construction.tex`: Paper-ready appendix on how the jul28 dataset was built —
design, source normalization, condition derivation, balance, determinism, auditing, and
known limitations. Every count in it is measured against `data/merged_mod_jul28.csv`.

`RELATED_WORK.md`: Drafted related-work section for `writeup.pdf` with full BibTeX (~45 entries), plus a positioning table (what each competing paper already establishes vs. our delta) and the four threats-to-validity with mitigations. Entries marked `VERIFY ... before submission` have unconfirmed author lists.

`viz/`: Visualization scripts.
- `paper_figures.py` — generates static figures for the paper
- `visualize_v3.py` / `visualize_v4_enhanced.py` — interactive dashboards for experiments 1 and 2
- `visualize_var_logprob.py` — interactive dashboard for experiment 3
