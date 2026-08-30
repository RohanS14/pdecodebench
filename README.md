# pde-llm-eval

Do language models understand the physics in numerical PDE code, or do they pattern-match
its surface?

The benchmark holds 32 ground-truth PDE simulations, each expanded 8 ways — comments
stripped, comments made deliberately wrong, variables obfuscated, the numerics quietly
broken — for 256 evaluation items balanced 128 valid / 128 invalid, 64 per `pde_class` and
32 per `mod_type`. A model is asked four things about a snippet: which PDE it solves, which
numerical method it uses, how the solution behaves, and whether the code is correct. Because
several conditions are byte-identical once comments are stripped (and one is
AST-isomorphic), a model that only reads syntax cannot separate them — which is what makes
the valid/invalid judgement informative rather than a lookup.

## Experiments

The repo is organised **one folder per experiment**. Each experiment's code folder, and its
results directory, share a name — so the same model is the same path everywhere and a
cross-experiment join is a directory name rather than a lookup table.

| experiment | question | code | results |
|---|---|---|---|
| **Static judgments** | can a model name PDE, method, behaviour and validity from code alone? | `freegen_static_judgments/` | `results/freegen_static_judgments/` |
| **MC / logprob** | does confidence degrade under perturbation even when the top-1 answer survives? | `mc_logprob/` | `results/mc_logprob/` |
| **Variable logprob** | does the model prefer the meaningful variable name over the obfuscated one, and where? | `var_logprob/` | — |
| **World model** | does the same defect move the representation the same way under different descriptions? Does similarity track the equation or the source text? | `probe/` | — |
| **Cross-modal consistency** | can a model detect *and localize* disagreement among four representations of one system? | `cross_modal_consistency/` | `results/cross_modal_consistency/` |
| **Belief revision** | does a frontier model revise a wrong judgement when shown runtime evidence? | `frontier/` | `results/frontier/` |

`frontier/` is the odd one out: it talks to the Gemini API from a laptop, needs
`GOOGLE_API_KEY`, and has never run on the cluster. Everything else is vLLM on SLURM.

## Layout

| path | what |
|---|---|
| `data/` | the jul28 release + `descriptions/`; large derived files are gitignored and published to HF |
| `datagen/` | dataset construction — corruption ladder, obfuscation, equations, audits |
| `shared/` | cross-experiment utilities: `dataset_io.py`, `upload_helper.py`, `extract_code.py`, `verify_simulations.py` |
| `viz/` | report and figure builders — see [Figures and reports](#figures-and-reports) |
| `sbatch/` | cluster job scripts |
| `tools/` | token measurement and one-off utilities |
| `tests/` | pytest suites, all on synthetic ground truth so they run before any GPU time |
| `results/` | run outputs, **one directory per model per experiment** |
| plus the six experiment folders above | |

There is no `eval/` — the name carried no meaning and its contents belonged to four
different experiments.

### Results layout

Every experiment writes `<experiment>/<model-slug>/<arm>.jsonl`, one file per model:

```
results/cross_modal_consistency/qwen__qwen3-8-27b/Qwen__Qwen3.8-27B__think_on__consistency.jsonl
results/freegen_static_judgments/qwen__qwen3-8-27b/Qwen__Qwen3.8-27B__think-on.jsonl
```

Model slugs match across experiments (`qwen__qwq-32b`, `zai-org__glm-4-7-flash`, …). The
same tree exists on the cluster under `outputs/`, so a script written against one works
against the other.

**Readers must glob recursively.** One directory per model means `<dir>/*.jsonl` matches
nothing, and returns an empty list rather than raising — which reads downstream as "the run
produced no rows". Three separate readers had that bug:

```python
paths = sorted(p for p in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)
               if not p.endswith((".prerescore", ".pretruncfix", ".prenormalize")))
```

Raw per-model JSONL is gitignored — it runs to hundreds of MB per arm and two files exceed
GitHub's 100 MB per-file limit. It lives on HuggingFace; the CSVs regenerate via
`viz/refresh_report.sh`.

## Figures and reports

Two families, and the prefix tells you which: `report_*` builds an interactive HTML
dashboard, `figures_*` writes static PNG/PDF for the paper. The suffix names the
experiment, so a filename answers "what is this for" without opening it.

| script | experiment | writes |
|---|---|---|
| `report_freegen_and_cross_modal.py` | static judgments + cross-modal | the joint report — the two experiments share 32 base solvers, which is the point of putting them in one document |
| `report_cross_modal_claims_roster.py` | cross-modal | claim-by-claim report on the current (generational) roster |
| `build_cross_modal_claims_frozen.sh` | cross-modal | the same report **pinned to the frozen 3-model repo**; it backs published claims and must never fold in the new models |
| `figures_cross_modal.py` | cross-modal | the blame-matrix/trust/obfuscation figures, frozen and roster rosters to separate directories |
| `figures_freegen.py` | static judgments | resampling + PDE-identification paper figures |
| `figures_mc_logprob.py` | MC / logprob | `paper_fig1-3.png` — Δconfidence, Δentropy, reasoning vs non-reasoning |
| `report_var_logprob.py` | variable logprob | 7 charts of token probability vs `code_fraction`, read from HF |
| `report_belief_revision.py` | belief revision | single-turn, 5 figures |
| `report_belief_revision_agentic.py` | belief revision | multi-turn stratified-64 run |
| `figures_belief_revision.py` | belief revision | the horizontal transition figure for the paper |
| `report_freegen_mc_full.py` | static judgments + MC, **old 10-model roster** | 19-chart dashboard with CIs and baselines |
| `report_freegen_mc_compact.py` | same | NeurIPS-compact variant; `report_freegen_mc_full.py` now covers everything it does except the Spearman correlation matrix |
| `report_freegen_mc_alias_confidence.py` | same | answer-alias distribution + confidence under corruption |

The last three read `results/pde_llm_eval.csv` and `results/pde_mc_logprob.csv` — the
**old 10-model roster**, not the 8 checkpoints the live experiments use. They are kept
because that roster covers seven models nothing else does (Gemma, both Llamas, Phi-4,
Mistral-Nemo, both Qwen2.5-Coders). Nothing in the current pipeline calls them.

`viz/consistency/` is a library, not an entrypoint: the 18 modules behind the cross-modal
reports. HTML and PNG output is gitignored — regenerate with `viz/refresh_report.sh`.

## Running it

```bash
# 1. build the item set (no compute)
python datagen/build_multimodal_items.py --out data/multimodal_items_v1.csv
python cross_modal_consistency/datagen/audit_multimodal_items.py   # expect 12/12
python cross_modal_consistency/datagen/validate_merged.py          # standing leak audit

# 2. T_exec, on the cluster (needs mpi4py_fft / jax_cfd for NavierStokes_3/4)
sbatch sbatch/run_exec_trajectories.sbatch

# 3. canary: 4 systems, all 8 conditions, both reasoning arms
SYSTEMS=Heat_1,Wave_1,Burgers_1,NavierStokes_1 LIMIT=0 SKIP_EXEC=true \
  sbatch sbatch/run_cross_modal_consistency.sbatch

# 4. full run, one model per submission
MODELS=Qwen/Qwen3-32B sbatch sbatch/run_cross_modal_consistency.sbatch
```

Roughly 3–5 h per model on one H200 for 2048 generations. Only `Qwen3-32B` and `QwQ-32B`
support both reasoning arms; everything else is a between-model comparison.

Code reaches the cluster with `bash sbatch/sync_code_to_cluster.sh` (dry run; `APPLY=1` to
upload) — a plain mirror, same folder names on both sides. Never `git push` to the cluster.

## Tests

```bash
python -m pytest -q          # 532 passing; the agentic suites need `google-genai`
```

Everything runs on synthetic ground truth, before any GPU time. Several tests exist because
the bug they pin was invisible locally and only appeared on the cluster —
`test_import_shadowing.py` and `test_results_layout_discovery.py` both read source or build
a fake tree rather than trusting the local filesystem.

## Reproducing a number

Experiment design, red-team briefs, activity logs and HuggingFace artifact registries live
outside this repo, under `notes/experiments/pde-{llm-eval,freegen-xmodal,mc-logprob,var-logprob-evolution}/`.
Published datasets are under `bermaneh/pde-llm-eval-*` on HuggingFace.
