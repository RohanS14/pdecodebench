# pde-verify — archived from `torch:~/pde-verify` 2026-08-24

A one-off QA pass on the PDE dataset, run 2026-07-20 and never touched again
(`DONE.marker: ALLDONE Mon Jul 20 23:06:52 UTC 2026`). Archived here in full before the
cluster copy was retired, because parts of it existed nowhere else.

## What it did

`run_all.sh` extracted the 128 PDE code snippets from `data/pdedata_clean_v4.xlsx` and
**executed every one** to see which actually run:

```
Total Scripts: 128 · Execution Success: 112 · Execution Errors: 16 · Inconclusive: 0
```

It then rendered 8 Valid-vs-Invalid comparison figures (Burgers 2 & 4, Heat_NoComm 3,
Wave_NoComm 3) into `figs/`. Full stdout in `run.log`, per-script results in
`verify_report.txt`.

## Reconciliation at archive time

| file | vs local `pde-llm-eval` | kept because |
|---|---|---|
| `extract_code.py` | local is a **superset** (adds CSV support, `merged_mod_jul28.csv` default) | provenance only |
| `verify_simulations.py` | local is a **superset** (adds `sys.argv[1]` for extracted_dir) | provenance only |
| `render_sims.py` | **no local counterpart** | ONLY copy |
| `verify_report.txt`, `run.log`, `figs/` | run outputs, not reproduced elsewhere | ONLY copy |
| `data/pdedata_clean_v4.xlsx` | differs from `data/archive/pdedata_clean_v4.xlsx` (md5 `e883e9ae…` vs `4de50aad…`) | ONLY copy of this variant |
| `data/extracted_codes/` | regenerable from the xlsx via `extract_code.py` | cheap to keep |

## Superseded by

`datagen/full_audit_exec.py` in this repo does the same execute-every-row check and adds a
stronger integrity test: Comm_Valid / NoComm_Valid / CorrComm / NoComm_CorrVar are the same
program with different comments and identifiers, so all four must produce identical numbers.
Its outputs live on the cluster at `/scratch/ehb7466/projects/pde-llm-eval/verification/`.
