# April 2026 cluster code — recovered, not curated

These 76 files lived **only** at `/scratch/ehb7466/projects/pde-llm-eval/code/` on
torch. They were never on the desktop. On 2026-08-24 they were moved to
`_trash` as "the disposable scratch code copy" — which was wrong, because
`PROJECTS.md` calls that copy disposable *on the assumption that the desktop is the
source of truth*, and for these files it was not. They are pulled back here so the
same mistake cannot cost them. This is the second time unbacked cluster-only code
has turned up in this project; the first was `consolidate_arms.py`.

**Nothing here is live.** The current pipeline is `crossmodal/`, `freegen/`, and
`sbatch/` at the repo root. Do not import from this directory.

## What it is

The April 19–23 multiple-choice, PCA, and layer-probe runs — the work that produced
`outputs/results_mc*`, `outputs/results_pca`, and `outputs/results_layer_probes` on
scratch.

| | |
|---|---|
| `run_mc_*.sbatch`, `run_mc_qwq_text.py` | the MC-question arm |
| `run_pca.sbatch`, `run_pca_repr.py`, `analyze_pca_repr.py` | representation PCA |
| `run_layer_probes.{sbatch,py}` | layer probes |
| `upload_gemma.py`, `upload_final_mc.py`, `upload_mc_helper.py` | pre-`hf_utility` uploaders |
| `packages/` | a vendored snapshot of `key_handler` and `hf_utility` |
| `parse_score.py` | an April ancestor of `freegen/parse_score.py` |

`run_all_models_v2 / _v2_fixed / _v3 / _v4 / _final` (and the `run_mc_all_models_*`
set) are five successive edits of one script kept as separate files. They are
preserved as found rather than deduplicated, because which one actually produced a
given result is decided by the job logs, not by the filename.

`__pycache__/`, `*.egg-info/`, and `packages/hf_utility/build/` were dropped on the
way down — 113 files on the cluster, 76 of them source.
