# pdecodebench `free_gen_exp/` — recovered 2026-08-24

Origin: `torch:~/pdecodebench/free_gen_exp/` — the **only untracked** directory in that
cluster clone of `github.com/RohanS14/pdecodebench` (HEAD `b3682f4`, Apr 16 2026).
It existed nowhere else: not on the GitHub remote, not on this Desktop, not in
`archive/cluster-code-apr2026/`.

Everything *else* in that clone was already reconciled here:

| cluster file | local counterpart | state |
|---|---|---|
| `data/pdedata.xlsx` | `data/archive/pdedata.xlsx` | byte-identical |
| `data/physics_code.xlsx` | `data/archive/physics_code.xlsx` | byte-identical |
| `data/descriptions/corruption_desc.txt` | `data/descriptions/corruption_desc.txt` | byte-identical |
| `data/descriptions/data_spec.txt` | `data/descriptions/data_spec.txt` | local has evolved |
| `datagen/fix_columns.ipynb` | `datagen/fix_columns.ipynb` | byte-identical |
| `datagen/comment_corr_example/*` (3) | same path | byte-identical |
| `datagen/augment_foobar_vars.py` | same path | local has evolved |
| `datagen/corrupt_comment.py` | same path | local has evolved |

`evaluate_llms_pde.py` (26,715 B, md5 `0291aabe399ecff54eea2228a989eabe`) is the Apr-2026
free-generation eval driver, superseded by `freegen/run_eval.py`. Kept for provenance only.
`smoke_results/` holds its two-file smoke output.

Once this is committed, `torch:~/pdecodebench` carries nothing unique and can be retired.
