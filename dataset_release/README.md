# Solver2System datasets

These are the two datasets released with *Solver to System: LLM Reasoning Across
Representations and Execution in Scientific Simulation*, presented at the NeurIPS 2026
Workshop on Sim2Science: ML with Imperfect Scientific Models. Both are built from one corpus of 32
Python PDE solvers, 8 each of Heat, Wave, Burgers and Navier–Stokes.

| file | rows x cols | HuggingFace |
|---|---|---|
| `dataset_cross_representation.csv` | 32 x 15 | [bermaneh/pde-llm-eval-cross-representation-dataset](https://huggingface.co/datasets/bermaneh/pde-llm-eval-cross-representation-dataset) |
| `dataset_code_perturbation.csv` | 256 x 17 | [bermaneh/pde-llm-eval-code-perturbation-dataset](https://huggingface.co/datasets/bermaneh/pde-llm-eval-code-perturbation-dataset) |

The CSVs here are byte-faithful exports of the HuggingFace tables.

```python
import pandas as pd
xrep = pd.read_csv("dataset_cross_representation.csv")   # 32 x 15
code = pd.read_csv("dataset_code_perturbation.csv")      # 256 x 17

# or straight from HuggingFace
from datasets import load_dataset
xrep = load_dataset("bermaneh/pde-llm-eval-cross-representation-dataset", split="train")
code = load_dataset("bermaneh/pde-llm-eval-code-perturbation-dataset", split="train")
```

Both files were written with `QUOTE_ALL`. Every code and view column contains newlines,
and the equation columns contain LaTeX backslashes, so **read them with a real CSV
parser** such as `pandas.read_csv`. Never split them on newlines.

Every row carries `gt_sample`, the base physical system it derives from. There are only 32
independent systems, so cluster on `gt_sample` for any confidence interval.

---

## dataset_cross_representation.csv — 32 rows x 15 columns

Each of the 32 physical systems has one row with FOUR representations: source code,
natural-language description, governing equation, and numerical trajectory. Each
representation comes in a correct form and a corrupted form. The code also comes with real
and obfuscated identifiers, and the trajectory has four kinds of corruption.

| Column | Description |
|---|---|
| `gt_sample` | The physical system. One row per system, 32 in total. |
| `source` | Which half of the base data the system came from: `human` or `synthetic`. |
| `code_valid_realnames` | CODE: the physically valid solver. Comments removed, descriptive identifiers retained. |
| `code_invalid_realnames` | CODE: the physically INVALID solver. This is the valid implementation with small but consequential defects (sign flips, spurious terms, disabled numerical operations) across at most three lines. The code still executes, but the dynamics are corrupted, producing unbounded growth, spurious oscillations or NaNs. |
| `code_valid_obfuscated` | CODE: the valid solver with descriptive identifiers replaced by meaningless placeholders. Pairs with the realnames column to separate reasoning about the physics from reading the variable names. |
| `code_invalid_obfuscated` | CODE: the invalid solver, identifiers obfuscated. |
| `description_valid` | DESCRIPTION: a natural-language account of this system. |
| `description_invalid` | DESCRIPTION: THIS SAME system's description carrying a small but physically meaningful error, e.g. describing a viscous solution for an inviscid equation. It is NOT another system's description: the system described is the same, and only the physics claim is wrong. |
| `equation_valid` | EQUATION: the governing equation in LaTeX, with initial and boundary conditions. |
| `equation_invalid` | EQUATION: THIS SAME specification with one term altered, typically a single sign flip or a changed initial or boundary condition. On most systems it differs by a character or two. |
| `trajectory_valid` | TRAJECTORY: the numerical solution field, subsampled to 10 snapshots and rendered as a text table. |
| `trajectory_rand` | TRAJECTORY corrupted by randomly generated numbers at the same shape, matched to the valid trajectory's mean and variance. Retains no spatial or temporal structure. |
| `trajectory_shuf` | TRAJECTORY corrupted by permuting values WITHIN the original, across space and time together. The multiset of values is identical to the valid trajectory, so mean, variance, min, max and the full histogram are preserved exactly. No statistic that ignores position can detect it. |
| `trajectory_swap` | TRAJECTORY corrupted by substituting ANOTHER system's trajectory. Physically coherent in itself, but describing the wrong system. |
| `trajectory_exec` | TRAJECTORY produced by executing this system's invalid solver, at the same 10 snapshots. Often contains NaN or infinite values where the defect makes the solver blow up. |

---

## dataset_code_perturbation.csv — 256 rows x 17 columns

There are 256 PDE solver implementations, built from 64 base programs: 32 physically
valid, and 32 with an injected bug that still runs to completion. Each base program
appears under 4 lexical perturbation conditions:

- original comments
- comments removed
- comments swapped in from a different implementation
- descriptive identifiers replaced by meaningless placeholders

The perturbations change the lexical surface only. Executable behaviour is unchanged.

| Column | Description |
|---|---|
| `title` | Unique implementation id, e.g. `Burgers_NoComm_InValid_1`: `<PDE>_<perturbation>_<validity>_<index>`. |
| `code` | Full Python source of the solver, self-contained and runnable. This is the model input. |
| `num_lines` / `num_char` | Line and character count of `code`. |
| `pde_class` | One of heat, wave, burgers, navier-stokes. 64 rows each. |
| `phys_process` | Ground-truth physical behaviour the solver exhibits (diffusion, shock, oscillation, ...). |
| `phys_valid` | Ground truth: `True` if the code implements physically correct dynamics, `False` if a bug was injected. 128 / 128. |
| `num_method` | Ground-truth numerical scheme (explicit, implicit, spectral, ...). |
| `corruption_source_id` | For CorrComm rows, the implementation the comments were taken from. |
| `corruption_source_pde` | PDE class of that comment donor. |
| `injected_comments` | Comments swapped in from another implementation, for the CorrComm conditions. |
| `delta_comments` | Change in comment count against the unperturbed base. |
| `num_comments` | Number of comments in this variant. |
| `gt_sample` | Base physical system id. There are 32 base systems and every row derives from one, so cluster on this for any confidence interval. |
| `mod_type` | Which of the 8 perturbation conditions this row is. `Comm_*` keep original comments, `NoComm_*` strip them, `CorrComm*` swap in comments from a DIFFERENT implementation, and `NoComm_CorrVar*` additionally replace identifiers with placeholders (`foobar_N`). 32 rows each. |
| `invalidity_note` | For invalid rows, what the injected bug is. Null for valid rows. |
| `source` | `human` (collected from public GitHub repos) or `synthetic` (generated). |

---

## License and attribution

MIT (`LICENSE`). Rows with `source == "human"` keep their original licenses: see `NOTICES`,
`PROVENANCE.csv` and `LICENSES/`.

## Citation

```bibtex
@inproceedings{berman2026solver2system,
  title     = {Solver to System: {LLM} Reasoning Across Representations and Execution in Scientific Simulation},
  author    = {Berman, Eliza and Subramanian, Rohan and Jha, Shreya and Jagarapu, Lohit and Lopez, Romain and Ranganath, Rajesh},
  booktitle = {NeurIPS 2026 Workshop on Sim2Science: ML with Imperfect Scientific Models},
  year      = {2026}
}
```
