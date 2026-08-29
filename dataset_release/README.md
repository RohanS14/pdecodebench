# dataset_release/

The two datasets this project contributes, as CSVs. Both are byte-faithful exports of
what is published on HuggingFace; the HF repos are the source of truth.

| file | rows x cols | HuggingFace |
|---|---|---|
| `dataset_cross_representation.csv` | 32 x 15 | [pde-llm-eval-cross-representation-dataset](https://huggingface.co/datasets/bermaneh/pde-llm-eval-cross-representation-dataset) |
| `dataset_code_perturbation.csv` | 256 x 17 | [pde-llm-eval-code-perturbation-dataset](https://huggingface.co/datasets/bermaneh/pde-llm-eval-code-perturbation-dataset) |

## dataset_cross_representation.csv

One row per physical system, 32 in total. Four representations of each system — source
code, natural-language description, governing equation, numerical trajectory — each in
its valid and its invalid form, across 13 view columns:

- `code_{valid,invalid}_{realnames,obfuscated}` — the valid solver and the physically
  invalid one, each with descriptive identifiers and with meaningless placeholders
- `description_{valid,invalid}` — the invalid form carries a small but physically
  meaningful error about the same system, not a different system's description
- `equation_{valid,invalid}` — the invalid form alters one term, typically a sign flip
- `trajectory_{valid,rand,shuf,swap,exec}` — the four corruptions are randomly generated
  values, values permuted within the original trajectory, another system's trajectory
  substituted, and the output of executing the invalid solver

Plus `gt_sample`, the system identifier, and `source`, which records whether the base
implementation was human-authored or synthetic.

## dataset_code_perturbation.csv

256 PDE solver implementations, 17 columns. 32 physical systems, each as a physically
valid and a physically invalid solver (128 / 128), each expanded by four perturbations
that change the lexical surface but NOT executable behaviour: original comments,
comments removed, comments swapped in from a different implementation, and descriptive
identifiers replaced by meaningless placeholders. 64 rows per PDE class, 32 per
`mod_type`.

## Reading these files

Both were written with `QUOTE_ALL`. Every view and code column contains newlines, and
the equation columns contain LaTeX backslashes, so **read them with a real CSV parser**
(`pandas.read_csv`) and never by splitting on newlines.

```python
import pandas as pd
xrep = pd.read_csv("dataset_cross_representation.csv")
code = pd.read_csv("dataset_code_perturbation.csv")
```

Round-trip verified: re-reading each file reproduces the published table with zero
differing cells.
