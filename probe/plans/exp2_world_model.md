# Experiment 2 — World Model: Are Physical Defects Represented Consistently Across Descriptions?

**Status:** DESIGN. No compute until `/raca:experiment-preflight` reruns on jul28
(`flow_state.redteam_status == "pending"`).

**Dataset:** `data/merged_mod_jul28.csv` — 256 rows, 32 `gt_sample` × 8 `mod_type`,
128 human / 128 synthetic, 64 per `pde_class`.

---

## 0. What this experiment can and cannot claim

The governing question is **"do LLMs understand physics?"** — crucially *not* "do LLMs
understand code." Everything below is organised around that distinction, because the
two are easy to confuse and the naive version of this experiment confuses them.

**The Δh design in §1–§8 does not, on its own, answer the physics question.** Its four
surface conditions differ only in comments and identifier names: comment-stripped
source is byte-identical across `S_plain`/`S_bare`/`S_mislead`, and `S_obf` is
AST-isomorphic to them. A model with a purely *syntactic* representation — one that
parses the AST and encodes "there is a sign flip on line 14" — scores ~1.0 on
`match_acc` and produces a large `gap`. What that design establishes is invariance to
comments and identifier names, i.e. that the model represents program *structure*
rather than surface *text*. That is a code result. It is necessary but not sufficient.

Answering the physics question needs the orthogonal axis: cases where **code form
varies while the physics is held constant**, and where **the physics varies with the
code form held similar**. §13–§16 add that. §13 gets what it can from the existing
dataset (less than hoped — see the measured confounds), and §14–§15 add the two
modalities that contain physics with no code in them at all: the **symbolic
equation** and the **executed trajectory**.

## 1. Research question (Δh / description-invariance)

For a fixed solver, does altering the *implemented physics* move the model's internal
state in the same direction regardless of how the program is *described*?

High cross-condition consistency ⇒ the model represents the physical intervention
independently of comments and identifier names. Low consistency ⇒ its reading of the
implemented system is entangled with surface language.

Read §0 for what this does and does not license.

## 2. Why this dataset supports the claim

The 8 `mod_type`s factor cleanly into **4 surface conditions × {valid, invalid}**:

| Condition | valid arm | invalid arm | Surface manipulation |
|---|---|---|---|
| `S_plain` | `Comm_Valid` | `Comm_InValid` | correct comments |
| `S_bare` | `NoComm_Valid` | `NoComm_InValid` | comments stripped |
| `S_mislead` | `CorrComm` | `CorrComm_Invalid` | misleading donor comments |
| `S_obf` | `NoComm_CorrVar` | `NoComm_CorrVar_InValid` | no comments + AST identifier obfuscation |

Verified directly against the CSV (not assumed):

- Comment-stripped source is **identical** across `S_plain` / `S_bare` / `S_mislead`,
  for both the valid arm (32/32) and the invalid arm (32/32).
- `S_obf` differs from the others only by identifier renaming (AST-isomorphic by
  construction — `augment_foobar_vars.py`).
- The valid→invalid character delta is **the same in all three non-obfuscated
  conditions** (mean −9.2, min −242, max +46), and is exactly zero for 13/32 solvers.
- `full_audit_exec.py` already certifies that a `gt_sample`'s surface conditions
  produce identical numbers.

So the physical edit is one fixed edit rendered four ways. That is the whole design.

## 3. Primary quantity

For solver `s`, condition `c`, layer `ℓ`, pooling `p`:

```
Δh(s, c, ℓ, p) = h(invalid | s, c, ℓ, p) − h(valid | s, c, ℓ, p)
```

32 solvers × 4 conditions = **128 defect vectors** per (model, layer, pooling).

**Primary statistic — within-solver cross-condition consistency:**

```
W(s, c, c', ℓ) = cos( Δh(s,c,ℓ), Δh(s,c',ℓ) )     6 condition-pairs
```

Reported as mean over the 32 solvers, per layer, per condition-pair.

Differencing removes the layer's mean activation offset, so this is not inflated by
the usual hidden-state anisotropy. No extra centering needed — but the anisotropy
check is still in the control battery below.

## 4. Control battery (the experiment is worthless without this)

A high `W` on its own is uninterpretable. Three things could produce it:

1. a solver-specific physical-defect representation (the hypothesis),
2. a single generic "this code is broken" direction shared by all solvers,
3. geometric artifact.

Distinguishing them:

| Control | Definition | What it rules out |
|---|---|---|
| **Cross-solver, same pair** | `cos(Δh(s,c), Δh(s',c'))`, `s ≠ s'` | (2) a global invalidity direction |
| **Cross-solver, same `pde_class`** | as above, restricted to matching class | class-level vs. instance-level encoding |
| **Surface-only Δ** | `h(NoComm_Valid) − h(Comm_Valid)`, etc. | gives the scale reference: is `‖Δ_physics‖` even comparable to `‖Δ_surface‖`? |
| **Random direction** | `cos(Δh, g)`, `g ~ N(0, I)` | (3) dimensionality artifact; must be ≈ 0 |
| **Norm audit** | distribution of `‖Δh‖ / ‖h‖` per layer | float16 precision floor (see §8) |

**Headline number — the consistency gap:**

```
G(ℓ) = mean_{s, c≠c'} W(s,c,c',ℓ)  −  mean_{s≠s', c,c'} cos(Δh(s,c,ℓ), Δh(s',c',ℓ))
```

Significance by **permutation over solver identity** (n = 32), not over the 128
vectors — the solver is the independent unit. Report the full layer curve plus a
permutation-corrected peak; pre-register the summary statistic before looking.

## 5. Secondary analyses

- **Defect identification (`match_acc`).** Given solver `s`'s defect under condition
  `c`, retrieve that same solver's defect from all `S` candidates under condition `c'`.
  Chance = 1/32 = 3.1%, so this is far sharper than any 0.5-chance statistic — and it
  is the statistic most directly aligned with the RQ, because it can only succeed if
  the defect representation is *both* solver-specific *and* stable across
  descriptions. **This is a co-headline number alongside `G(ℓ)`.**
- **Generic-direction transfer (`generic_transfer_acc`).** Fit the *mean* defect
  direction on all other solvers under `c`, test its sign on the held-out solver
  under `c'`. Note this is the decodable counterpart to the **cross-solver** control,
  not to `within_cos` — averaging over solvers deliberately destroys anything
  solver-specific. It measures hypothesis (2) from §4, and reading it as evidence
  for (1) is exactly the mistake this experiment exists to avoid.
- **Subspace alignment.** Principal angles between `span{Δh(·,c)}` and `span{Δh(·,c')}`
  — tests whether the defect *subspace* is shared even when individual vectors are not.
- **Stratification:** by `source` (human vs. synthetic, 16 each), by `pde_class`
  (8 each), and by the 13 solvers with **zero** valid→invalid length delta (the
  cleanest subset — no length confound at all).

## 6. Predictions worth committing to in advance

- `W(S_plain, S_bare)` highest — the code is byte-identical, only comments removed.
- `W(S_plain, S_mislead)` is the real test of comment entanglement.
- `W(S_plain, S_obf)` lowest, and the most informative: identifier tokens change
  everywhere, so surviving consistency there is strong evidence for (1).
- Consistency should peak in **middle layers** if it tracks semantics.

**CORRECTED 2026-07-31 by the canary (job 15051200).** This section previously predicted
a near-zero layer-0 gap and called any layer-0 signal a red flag to be explained away.
That prediction is wrong on this dataset, for a structural reason:

> For a given solver the **same code edit** is applied under all four surface conditions.
> So Δh at the embedding layer is nearly the same vector across conditions simply because
> the same tokens were edited — no semantics required. The cross-solver control does not
> remove this, because it only compares *different* solvers, whose edits differ anyway.

The canary measured a layer-0 gap of **+0.41** (2 solvers, so the magnitude is not
trustworthy, but the mechanism is structural and will not vanish with more solvers).

**Layer 0 is therefore a token-identity FLOOR, not a null.** A mid-layer gap is evidence
only to the extent it exceeds that floor. Report `G(ℓ) − G(0)` alongside raw `G(ℓ)`, and
treat a mid-layer peak that fails to clear layer 0 as a negative result. The same applies
to `match_acc`.

This is the single most important thing the canary produced, and it means `gap` and
`match_acc` are both partly confounded by edit-token overlap in a way §4's control battery
does not address.

## 7. Known dataset threats to fold in

- **Sampling-cadence leak** — Burgers_6/7/8 + NavierStokes_5/7 (20 rows) are flagged
  open in `flow_state.json`. Their defect may be detectable from a cadence change
  rather than physics. Pre-register these 5 solvers as a stratum;
  `leak_ablation_jul28.csv` (`widened` vs `aligned`) lets us *measure* the leak's
  contribution to `Δh` instead of only excluding it.
- **Heat_2 grid-resolution change** — second open item; same treatment.
- **Incidental drift.** `derive_invalidity_change.py` separates the physical error
  from incidental drift in the valid↔invalid diff. Primary analysis on the
  clean-edit subset; full set reported secondarily.

## 8. Implementation

Reuses `probe/extract_hidden.py` largely as-is — it already emits `mean_pool`,
`last_tok`, `code_token_spans`, `gt_samples`, `mod_types` and handles all 8
`mod_type`s. Two required changes:

1. `EXPECTED_MOD_TYPE_DIST` is hardcoded to 16 per condition (v3). jul28 is **32**.
2. **Precision.** It currently saves float16. `Δh` is a difference between two forward
   passes whose inputs differ by a handful of tokens, so `‖Δh‖ / ‖h‖` may be ~1e-2 —
   at which point float16 leaves 1–2 significant digits and the cosine is noise.
   The fix is four lines: `:253` already casts bf16→float32, and `:256-257` downcast
   to float16 only for storage, into arrays allocated at `:175-176`. Change all four
   to float32 (~1.3 GB across the full roster — trivial), and audit the
   `‖Δh‖ / ‖h‖` ratio per layer before trusting any cosine.

New: `probe/world_model_delta.py` (Δh construction + control battery + permutation
test → tidy CSV), `probe/viz_world_model.py` (report).

**Compute is small.** One forward pass per row, 256 rows, no generation. Minutes of
GPU per model; the analysis is CPU/numpy. The bottleneck is model download and GPU
memory for anything ≥32B — not throughput. Consequence: the canary is nearly the full
run, so run the canary as a genuine 2-solver × 8-condition end-to-end slice and
validate the artifact before scaling.

Blocking infra: the four eval sbatch scripts still point at three stale paths
(`/scratch/ehb7466/pde-llm-eval`, `${WORKSPACE}/venv`, archived `pdedata_clean_v3.xlsx`).
The probe job needs a script wired to the current layout — code in `$HOME/pde-llm-eval`,
results under `/scratch/ehb7466/projects/pde-llm-eval`, venv `/scratch/ehb7466/envs/pdecodebench`.

## 9. Outputs and visualization

- `probe/results/world_model_delta_{model}_{pool}.csv` — long form:
  `model, pool, layer, stat_type, condition_pair, gt_sample, source, pde_class, value`
- Figures: (a) layer × condition-pair heatmap of mean `W`; (b) consistency gap `G(ℓ)`
  vs. layer with permutation null band; (c) `‖Δ_physics‖ / ‖Δ_surface‖` vs. layer;
  (d) per-solver scatter to expose whether the mean is carried by a few solvers.
- Self-contained Plotly HTML, same pattern as `viz_interactive.py`.

## 10. Gates before compute

1. `/raca:experiment-preflight` on jul28 (`redteam_status` is `pending` — hard gate).
2. Resolve or explicitly stratify the two open dataset items (§7).
3. Rewire sbatch to the current cluster layout.
4. Canary → validate artifact → then scale.

## 11. Model roster

Four open-weight models: two scales within one family, plus two cross-family controls.

| Model | Layers (L, incl. embed) | D | float32 storage | GPU |
|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 29 | 3584 | 213 MB | 1× 40 GB |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 65 | 5120 | 682 MB | 1× 80 GB or 2× 40 GB |
| `meta-llama/Llama-3.1-8B-Instruct` | 33 | 4096 | 277 MB | 1× 40 GB |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 28 | 2048 | 117 MB | 1× 40 GB (MoE, ~2.4B active) |

Total ≈ 1.3 GB of hidden states. Storage is a non-issue; 32B weights (~65 GB bf16)
are the only real constraint.

What each buys: 7B is the comparability anchor to Experiment 5. 32B tests whether
consistency grows with scale. Llama-3.1-8B is the general-purpose (non-code-specialized)
control — if the effect needs code pretraining, it should be weakest here.
DeepSeek-Coder-V2-Lite is a different architecture *and* a code model, separating
"code training" from "Qwen quirk."

**Analysis consequences of a mixed roster:**

- **Layer counts differ (29 / 65 / 33 / 28).** All cross-model layer curves must be
  plotted on **relative depth** `ℓ / (L−1)`, never raw layer index. Absolute-index
  overlays across these four models would be meaningless.
- **Four tokenizers.** `find_code_token_span` relies on `return_offsets_mapping`, and
  the existing `validate_code_span` check must pass on every model — verify it fires
  on all 256 rows per model, not just Qwen. A silent span failure would make
  `mean_pool` average over the wrong tokens.
- **`apply_chat_template` already handles the per-model prompt format** (line 148/186),
  and `PROMPT_TEMPLATE` stays verbatim from `eval/run_eval.py`. Do not vary it.
- **MoE:** DeepSeek-Coder-V2-Lite routes experts per token, but the residual stream
  read by `output_hidden_states` is dense — no special handling needed. Worth noting
  that `Δh` there mixes representation change with routing change.

**Sequencing.** Extract and fully validate the analysis on 7B first, including the
norm-ratio audit (§8) and the layer-0 sanity check (§6). Only then extract the other
three. Debugging the control battery across four models at once is the expensive
version of the same work.

---

## 12. Files

| Path | What |
|---|---|
| `probe/extract_hidden.py` | **modified** — jul28 CSV via `eval/dataset_io`, condition balance derived from N, hidden states stored **float32**, `sources`/`model_name`/`dataset_path` added to the NPZ |
| `probe/world_model_delta.py` | Part I — Δh construction, control battery, permutation test |
| `probe/code_similarity.py` | Part II — token / AST-n-gram / length nuisance regressors |
| `probe/physics_vs_code.py` | Part II §14 — variance partitioning, method invariance, process transfer |
| `probe/extract_modality_hidden.py` | Part II — embeds the equation and trajectory modalities |
| `probe/cross_modal.py` | Part II §15 — equation ↔ code ↔ trajectory retrieval |
| `probe/geometry_battery.py` | Part II §16 — dimensionality, curvature, validity transfer, anisotropy |
| `probe/viz_world_model.py` | 7-section self-contained Plotly report (4 Part I + 3 Part II) |
| `datagen/build_equations.py` | builds `data/equations_jul28.csv` from solver source |
| `datagen/extract_trajectories.py` | executes 256 rows → trajectories + valid/invalid divergence |
| `tests/test_world_model_delta.py` | 17 power-and-null tests |
| `tests/test_physics_vs_code.py` | 12 tests incl. the real-data identifier-blindness property |
| `sbatch/run_build_modalities.sbatch` | CPU — equations + trajectories |
| `sbatch/run_extract_hidden_worldmodel.sbatch` | GPU — code hidden states, resumable |
| `sbatch/run_extract_modalities.sbatch` | GPU — equation/trajectory hidden states |
| `sbatch/run_world_model_delta.sbatch` | CPU — Part I + all Part II analyses + report |

**Not yet covered by formal tests:** `cross_modal.py`, `geometry_battery.py`,
`extract_trajectories.py` are smoke-tested only.

### Equation grounding (2026-07-30)

Review flags fell 7 → 1 once derivations were read off the solver source instead of regex
features. Four detector bugs, each of which had produced a wrong equation:

| Bug | Example | Effect |
|---|---|---|
| time-history array read as 2D domain | `u = np.zeros((nt, nx))` | Burgers_1–4 mislabelled 2D |
| Rusanov coefficient read as viscosity | `alpha = np.maximum(np.abs(u[:-1]), …)` | inviscid Burgers_4 written as viscous |
| tuple-assigned zero viscosity missed | `L, nu = 2.0, 0.0` | inviscid Burgers_5 written as viscous |
| `\bpsi\b` cannot match `psi_hat` | NavierStokes_4 | a named vorticity solver flagged ambiguous |

Remaining flag: **NavierStokes_3** — spectral, no explicit pressure solve, no streamfunction
named in the source. Needs a domain reader.

### Validation already performed

`tests/test_world_model_delta.py` plants a known ground truth in synthetic hidden
states and checks the statistics respond correctly. The decisive case is `generic`
— a single global "this code is broken" direction shared by every solver:

| regime | planted | `within_cos` | `gap` | `match_acc` | `generic_transfer_acc` |
|---|---|---|---|---|---|
| `signal` | per-solver direction, shared across conditions | +0.504 | **+0.505** | **1.000** | 0.469 |
| `null` | noise only | +0.001 | +0.001 | 0.013 | 0.500 |
| `generic` | one global direction, all solvers | **+0.492** | +0.000 | 0.021 | **1.000** |

(chance: `match_acc` = 0.031, `generic_transfer_acc` = 0.500)

The `generic` row is the whole argument for the control battery: `within_cos` of
0.492 looks like a strong positive result, and both `gap` and `match_acc` correctly
report nothing solver-specific. Reporting `within_cos` alone would have produced a
confident, publishable, wrong claim.

### Run order

```bash
# 1. canary — 2 solvers × 8 conditions, one model, end to end
GT_SAMPLES="Wave_1,Heat_1" MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" \
  sbatch sbatch/run_extract_hidden_worldmodel.sbatch

# 2. validate the canary NPZ, then the full 7B extraction
MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" \
  sbatch sbatch/run_extract_hidden_worldmodel.sbatch

# 3. analysis on 7B only — check the norm audit and layer-0 flag before scaling
N_PERM=2000 sbatch sbatch/run_world_model_delta.sbatch

# 4. remaining three models (existing NPZs are skipped automatically)
sbatch sbatch/run_extract_hidden_worldmodel.sbatch
sbatch sbatch/run_world_model_delta.sbatch
```

---

# Part II — Physics vs. code

Everything above varies the *description* of a fixed program. Part II varies the
program and the modality, which is what the physics question actually requires.

## 13. What the existing dataset can support (measured, not assumed)

I checked the label structure of `merged_mod_jul28.csv` before designing around it.
The result constrains Part II more than expected and is recorded here so no one
re-derives an over-optimistic version of it.

**`pde_class` × `num_method` is only partially crossed:**

| | explicit | expl/impl | implicit | spectral | expl/spec | e/i/s |
|---|---|---|---|---|---|---|
| burgers | **8** | 0 | 0 | 0 | 0 | 0 |
| heat | 2 | 3 | 2 | 1 | 0 | 0 |
| wave | 4 | 1 | 1 | 2 | 0 | 0 |
| navier-stokes | 3 | 1 | 1 | 0 | 2 | 1 |

Burgers is 100% explicit — perfectly confounded, and must be dropped from any
method-invariance contrast. Heat, wave and NS have real method diversity.

**`phys_process` is badly confounded with `pde_class`:**

| process | heat | wave | burgers | navier-stokes |
|---|---|---|---|---|
| diffusion | **8/8** | 0/8 | 3/8 | **8/8** |
| advection | 0/8 | 0/8 | **8/8** | 6/8 |
| oscillation | 0/8 | **8/8** | 0/8 | 0/8 |
| restoration | 0/8 | 5/8 | 0/8 | 1/8 |

This kills the attractive "compositional physics" test. A diffusion direction fit on
these labels is nearly identical to a *heat-or-NS vs. wave* direction — a class
distinction wearing a process label. Only two genuine within-class contrasts exist,
and both are weak:

- burgers diffusion, 3+ / 5− — and the 3 positives are **exactly** the three
  cadence-leak solvers (Burgers_6/7/8), all synthetic. Triple-confounded.
- NS advection, 6+ / 2− — two negatives.

**Two `navier-stokes` items are diffusion equations.** `NavierStokes_6`'s Crank-Nicolson
tridiagonal is byte-identical to `Heat_6`'s — same scheme, same equation `du/dt = D d2u/dx2`,
differing only in boundary condition and in whether the coefficient is called `nu` or
`alpha`. `NavierStokes_5` is the same plus a constant source (its own
`u_steady = G/(2*nu)*y*(H-y)`, the Poiseuille profile). They are Navier-Stokes by physical
provenance, not by governing equation.

This lands directly on §14. A `pde_class` regressor is asked to separate two items that
**are the same equation**, so whatever separates them must be surface form — `nu` vs
`alpha`, `dy` vs `dx`, `U0`. That is the code-not-physics signal this experiment exists to
detect, sitting inside the ground truth rather than in the model. Consequences:

- §14.1 must report the variance partitioning **with and without** `NavierStokes_5`/`_6`.
  If `pde_class` incremental R² falls materially when they are dropped, the surviving
  signal was partly these two items' naming conventions.
- §16.3 (validity-direction transfer) already holds them out with the `navier-stokes`
  fold, so they cannot silently carry a cross-class result.
- It partly explains the `diffusion` collinearity above: `heat` and `navier-stokes` are
  both 8/8 diffusion in part because several `navier-stokes` items *are* diffusion.

Pending a benchmark-design decision (see `dataset_overview.md` Known Limitations). Until
then treat them as a pre-registered stratum, not an exclusion.

**Conclusion: the dataset alone cannot cleanly separate physics from code at the
process level.** Process-direction transfer is demoted to exploratory, reported with
these confounds attached and never as a headline. This is the argument for §14–§15.

## 14. Physics-vs-code dissociation (`probe/physics_vs_code.py`)

### 14.1 Variance partitioning with code-similarity nuisance regressors

The highest-value zero-new-data analysis, and the correction of a real gap: the
Δh design never controls for code similarity, so any clustering it finds is
uninterpretable.

Over the 32 base solvers, per layer, regress the representational distance matrix on:

| regressor | what it is |
|---|---|
| `same_pde_class` | physics |
| `same_num_method` | algorithm |
| `token_jaccard` | surface lexical overlap |
| `ast_edit_dist` | normalised tree edit distance over ASTs with identifiers stripped |
| `len_diff` | code length difference |

The physics claim is only meaningful as: **`same_pde_class` explains variance after
partialling out `token_jaccard`, `ast_edit_dist` and `len_diff`.** Report semi-partial
correlations and the incremental R² of each regressor per layer. Run on the
`NoComm_CorrVar` condition as well as `Comm_Valid` — with comments gone and
identifiers obfuscated, surviving pde_class structure is much harder to attribute to
lexical form.

### 14.2 Method-invariance of the PDE representation

Do same-class/different-method solvers cluster above same-method/different-class?
Heat, wave and NS only (burgers is confounded). Small n — reported as suggestive,
with the exact cell counts printed alongside every number.

### 14.3 Process-direction transfer (exploratory only)

Fit on one class, test on another, for `diffusion` and `advection`. Given §13 this
cannot support a strong claim; it is retained to quantify *how much* of the apparent
process structure is class structure, which is itself worth measuring.

## 15. Cross-modal alignment: equation ↔ code ↔ trajectory

Three representations of the same physical system, two of which contain no code:

1. **Equation** — the PDE in symbolic form (`data/equations_jul28.csv`), one per
   `gt_sample`, authored from the actual solver rather than from the class name, with
   notation variants (unicode / LaTeX / plain ASCII) to separate physics from notation.
2. **Trajectory** — the executed solution (`datagen/extract_trajectories.py`), as
   time-resolved fields plus a text rendering of the dynamics.
3. **Code** — the existing eight conditions.

**Primary test: retrieval across modalities.** Given the equation, retrieve the
matching solver's code; given the trajectory, retrieve the equation; and so on.

Two design rules, both non-negotiable:

- **Score within `pde_class`, not only globally.** There are only 4 classes, so global
  retrieval (chance 1/32) can be solved by 4-way category matching. Within-class
  retrieval (chance 1/8) forces instance-level physics matching. Report both; the
  within-class number is the real one.
- **The headline pairing is equation → `NoComm_CorrVar` code.** No comments,
  obfuscated identifiers. If `∂u/∂t = ν ∂²u/∂x²` retrieves the right obfuscated,
  comment-free heat solver above within-class chance, a lexical-overlap explanation
  is very hard to sustain.

**Lexical-overlap control.** Compute token overlap between each equation and each
code body and include it as a nuisance regressor, exactly as in §14.1. A retrieval
result that vanishes once lexical overlap is partialled out is a lexical result.

### 15.1 Does ‖Δh‖ track the actual physical error?

The deepest test available, and the one that most directly earns the phrase "world
model". The invalid variants produce genuinely different numbers. Correlate, per
solver:

- `‖Δh(s, c)‖` — the representational shift, from §3
- the **numerical divergence** between the valid and invalid trajectories
  (relative L2 over the solution field; blow-up / NaN handled as a separate category)
- the **error type** from `derive_invalidity_change.py` and `invalidity_note`
  (sign flip, coefficient change, stability-condition violation, boundary error)

If the representational shift scales with the magnitude and character of the real
physical error, that is evidence no retrieval score provides. If ‖Δh‖ is instead
predicted by the size of the *code edit* and not by the physical divergence, that is
a clean negative result and worth reporting as one.

## 16. Geometry battery (`probe/geometry_battery.py`)

- **Intrinsic dimensionality** (participation ratio, two-NN estimator) of the
  representation manifold, computed separately over physics-varying and
  code-varying directions. Is the physics manifold lower-dimensional than the code
  manifold, and where in depth do they separate?
- **Linear separability vs. curvature** of `pde_class`: linear probe accuracy against
  a kernel/kNN probe. A large gap means the physics structure exists but is not
  linearly encoded — which changes how every other result here should be read.
- **Is the validity direction shared across classes?** Fit `phys_valid` on three
  classes, test on the held-out one. Cross-class transfer of validity is a strong
  claim about a general representation of physical correctness. LOGO by `gt_sample`,
  and stratified so the leak solvers cannot carry it.
- **Anisotropy baseline** for every number above: hidden states are strongly
  anisotropic, and the Δh design escapes this by differencing. §14 and §16 do **not**
  difference, so they must report mean-centred and raw variants side by side.

## 17. Revised gates

The dataset gains two new files (`equations_jul28.csv`, trajectory artifacts), which
is a material change under the workspace rules. Sequence:

1. `/raca:experiment-preflight` covering Part I **and** Part II. `redteam_status` is
   `pending`; nothing runs before it clears.
2. Trajectory extraction is a CPU batch job on `cpu_short` and must run before any
   cross-modal analysis.
3. Canary: 2 solvers × 8 conditions × 3 modalities, one model, end to end.
4. Part I (§1–§8) can report independently of Part II — it is a complete
   description-invariance result on its own, provided it is labelled as one.
