# PDE Interpretability Probe Experiment

## Goal

Determine what PDE properties are encoded in the internal representations of
**Qwen2.5-Coder-7B-Instruct** when it reads PDE solver code. Specifically: do
hidden states encode the type of PDE, the physical processes involved, the
numerical method, and whether the physics is valid — and does this encoding
degrade when the code is corrupted?

---

## Dataset

**`data/pdedata_clean_v2.xlsx`** — 96 rows total.

16 ground-truth PDE problems (`gt_sample`), each appearing in 6 variants (`mod_type`):

| mod_type | Description |
|---|---|
| `Comm_Valid` | Clean code with correct comments |
| `NoComm_Valid` | Clean code, no comments |
| `CorrComm` | Correct code, deliberately wrong/misleading comments |
| `NoComm_CorrVar` | Clean code, no comments, obfuscated variable names |
| `Comm_InValid` | Commented code with physically invalid implementation |
| `NoComm_InValid` | No comments, physically invalid implementation |

**Labels probed (9 total):**
- `pde_class`: 4-class — wave, heat, burgers, navier-stokes (chance = 0.25)
- `process_{diffusion, advection, oscillation, restoration}`: binary (chance = 0.50)
- `method_{explicit, implicit, spectral}`: binary (chance = 0.50)
- `phys_valid`: binary — physically valid vs. invalid (chance = 0.50)

---

## Step 1 — Hidden State Extraction (`extract_hidden.py`)

Runs all 96 examples through Qwen2.5-Coder-7B-Instruct using a chat template.
For each example and each of the 29 layers (0 = embedding, 1–28 = transformer layers):
- **mean_pool**: average hidden state over code token positions only
  (identified via `return_offsets_mapping=True` to exclude system/user prompt tokens)
- **last_tok**: hidden state at the final input token

Output: `probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz`
Shape: `(96, 29, 3584)` for each pooling strategy, plus metadata arrays.

---

## Step 2 — RSA Analysis (`rsa_analysis.py`)

Representational Similarity Analysis: computes pairwise cosine distance matrices
across all 96 examples, for each layer and both pooling strategies.

**Outputs:**
- Layer heatmaps sorted by (pde_class, gt_sample, mod_type) — block structure
  indicates PDE-class clustering in representation space
- Block RDM score vs. layer: ratio of within-class to between-class distances
- Mod-type comparison: heatmaps for Comm_Valid / CorrComm / NoComm_CorrVar
  at the best-clustering layer

**Question:** At which layers do representations cluster by PDE class? Does
corruption (wrong comments, invalid physics) change the geometry?

---

## Step 3 — Pooled Probe (`linear_probe_pooled.py`)

Logistic regression probes trained on all 96 rows with **Leave-One-Group-Out CV**
grouped by `gt_sample` (to prevent leakage — same base code appears in all 6 mod_types).

Per fold: train on 80 rows (15 gt_samples × 6 mod_types), test on 6 rows
(1 gt_sample × 6 mod_types). 16 folds total.

Also includes a **TF-IDF bag-of-words baseline** with the same LOGO-CV, as a
surface-form control: if probe accuracy ≈ BoW accuracy, the model isn't adding
value beyond keyword matching.

**Outputs:** `probe/results/probe_pooled_{pool}.csv` — accuracy per layer per label,
with bootstrap 95% CI (10,000 resamples over 16 fold accuracies). Per-mod_type
accuracy breakdown also recorded.

**Question:** Which layers best encode each PDE property? Does test accuracy on
corrupted mod_types drop compared to clean?

---

## Step 4 — Clean-Transfer Probe (`linear_probe_clean_transfer.py`)

More targeted experiment: train **only on `Comm_Valid`** (one per gt_sample = 15
training examples per fold), test on **all 6 mod_types** of the held-out gt_sample.

Per fold: train on 15 clean examples, predict all 6 variants of the held-out problem.

**Outputs:** `probe/results/probe_transfer_{pool}.csv` — per-layer accuracy broken
down by mod_type, plus overall. Bootstrap 95% CI from Comm_Valid fold accuracies.

**Question:** Does a probe trained purely on clean representations generalize to
corrupted inputs? Failure to generalize → corruption meaningfully changes the
internal representation. Success → the model encodes PDE properties robustly
regardless of surface corruption.

---

## Step 5 — Interactive Report (`viz_interactive.py`)

Single self-contained HTML file (`probe/results/report.html`) with all figures
rendered in Plotly. Four sections navigable via sticky top navbar:

1. **Overview** — experiment metadata, dataset distribution, label balance
2. **RSA Analysis** — interactive layer heatmaps with dropdown, block score vs. layer
3. **Pooled Probe** — accuracy vs. layer with CI bands, mod-type breakdown bars
4. **Transfer Probe** — label × mod_type heatmap, per-label transfer curves

CI sources:
- Overall accuracy curves: bootstrap 95% CI already in CSV
- Per-mod_type bars: Wilson binomial CI (n=16, one binary outcome per LOGO fold)
  computed in viz script using `statsmodels.stats.proportion.proportion_confint`

---

## Key Design Choices

**Why LOGO-CV by gt_sample?** Each base problem appears in all 6 mod_types. If
we split randomly, the same problem leaks into both train and test — the probe
just memorizes per-problem identity rather than PDE properties.

**Why mean-pool over code tokens only?** System/user prompt tokens introduce
positional bias. Masking to code tokens gives a cleaner representation of
how the model encodes the code content itself.

**Why 16 folds?** Dataset has 16 gt_samples. CIs are wide as a result — this is
expected and should be reported honestly. The point is directional signal,
not tight confidence intervals.
