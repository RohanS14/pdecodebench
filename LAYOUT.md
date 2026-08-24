# Layout — local repo, cluster `$HOME`, cluster scratch

The canonical model is `/scratch/ehb7466/projects/PROJECTS.md` on torch:

> **Code source of truth = your LOCAL DESKTOP** (`private_projects/<slug>`).
> Flow: **desktop → cluster `$HOME/<slug>`**. The cluster scratch copy of code is
> disposable; only its **results** matter. Results live on scratch:
> `projects/<slug>/{outputs,data,logs}`.

This file records where that model is *actually* honoured for `pde-llm-eval`, and
where it is not — because the gaps have cost real jobs.

## The three places

| | path | what lives there | authority |
|---|---|---|---|
| **local** | `private_projects/pde-llm-eval` | all code | **source of truth** |
| **cluster code** | `/home/ehb7466/pde-llm-eval` | code, pushed with `raca upload` | mirror of local |
| **cluster results** | `/scratch/ehb7466/projects/pde-llm-eval/{outputs,logs}` | every artifact | the only thing that matters on scratch |

Never `git push` to reach the cluster. `raca upload <local> <path-under-$HOME>`.

## Shared, cross-project paths

| | path | note |
|---|---|---|
| model weights | `/scratch/ehb7466/hf_cache` | **one cache for every project.** Set `HF_HOME` to it and nothing else |
| JIT caches | `/scratch/ehb7466/jit_cache/{triton,flashinfer,tvm-ffi}` | symlinked from `$HOME`; `$HOME` is capped by FILE COUNT (30k), and JIT caches are what fills it |
| venv | `/scratch/ehb7466/envs/pdecodebench-vllm` | **not** `pdecodebench` — that one's `bin/python3` resolves to Python 3.9 on compute nodes |
| retired files | `/scratch/ehb7466/_trash/<date>/` | move here rather than deleting |

Setting `HF_HOME` **and** `HF_HUB_CACHE` to the same path produces a split layout — a
nested `hub/` alongside flat `models--*` directories — because downloads honour
`HF_HUB_CACHE` while some readers derive `HF_HOME/hub`. Set the root only.

## Reconciliation, 2026-08-24 — the cluster was AHEAD

All five `crossmodal/eval/*.py` differed between local and cluster by 27–87 lines.
The cluster's copies were the real ones: they had been hand-edited after upload and
never pulled back. Local has now been updated **from** the cluster, with the import
paths rewritten back to package form (see the transform below).

What that recovered — none of it was on the desktop:

- **`consolidate_arms.py` existed only on the cluster.** Unbacked code, one `rm` from
  gone.
- **Qwen3.8's generation budget is 65536, not 131072.** Lowered on 2026-08-22 after
  measuring 1,152 draws (p50 17,961 / p90 51,500 / p99 77,267). A 65,536 cap truncates
  2.9% of draws against 131,072's 0.7% — and those 0.7 points cost a doubled straggler
  tail, because `llm.generate()` writes nothing until every sequence in the batch
  returns. One trace running to 131k holds ~190 finished ones unwritten for an extra
  hour and collapses GPU utilization at the batch tail, which invites the cluster's
  sub-60% killer. Two consistency shards sat 3h at 189/192 with zero rows on disk from
  exactly this. The truncated 2.9% are recovered by continuation instead.
- Assorted comment and threshold edits across the other four files.

The Qwen3.8 budget was caught by `tests/test_freegen_parity.py`, which imports both
runners and compares them — not by reading the diff. The free-generation job for that
model was already running at 131072 when the test failed, and was cancelled and
resubmitted at 65536.

**Local and cluster are now content-identical modulo the import rewrite**, and
`sbatch/sync_code_to_cluster.sh` performs that rewrite mechanically so the cluster
copy stays derivable instead of becoming a second original again.

### The one exception the transform has to special-case

`parse_score` flattens **into** `eval/` on the cluster but lives under `freegen/`
locally. The generic `eval.* → crossmodal.eval.*` rule sends it to a module that
exists on neither side, so it is rewritten by name in both directions. A blanket regex
got this wrong once and broke every local import of `crossmodal.eval.parse_consistency`.

## The transform, and what it has to do beyond imports

`sbatch/sync_code_to_cluster.sh` is the only sanctioned way to push code. It has to
change three things, not one, and two of them were discovered the hard way on
2026-08-24:

| | local | cluster |
|---|---|---|
| imports | `from crossmodal.eval.x import` | `from eval.x import` |
| **sys.path depth** | `crossmodal/eval/f.py` — **three** dirnames to the repo root | `eval/f.py` — **two** |
| prose paths | `python crossmodal/eval/f.py` | `python eval/f.py` |

The depth one is the dangerous one. Uploading local verbatim puts `/home/ehb7466` on
`sys.path` instead of the repo root, and every `from eval.…` import in the consistency
runner fails. The transform is written in Python rather than sed because that
expression spans two lines. It also has to handle `from crossmodal.eval import X` —
the old sed only matched `from crossmodal.eval.X import`, so `backfill_no_verdict.py`
went out with an import of `crossmodal.eval.parse_consistency`, which exists on
neither side.

The prose rule is deliberately narrowed to paths followed by a real `.py` filename. A
bare `crossmodal/eval/` in a comment is a comment *about* the two layouts; rewriting
it makes the sentence contrast `eval/` with itself.

**The diff check was also broken and reported a comfortable lie.** It ran `raca ssh`
inside a `while read` loop fed by a process substitution, so the first ssh consumed
the whole file list and it printed "0 file(s) differ" no matter what had changed. It
now collects every remote digest in one ssh and reads the list from a variable.

`viz/` and `tests/` are **not** mirrored. Nothing on the cluster runs them, and
`$HOME` is capped by file count.

## Where the cluster does NOT mirror local

`$HOME/pde-llm-eval/eval/` is **flatter than local**. It holds local's `eval/` files
*plus* flattened copies of things that live elsewhere locally:

| cluster path | local path | why it is there |
|---|---|---|
| `eval/run_cross_modal_consistency.py` | `crossmodal/eval/…` | flattened at upload; this is the **live production path** for the consistency jobs |
| `eval/parse_consistency.py`, `eval/consistency_prompts.py`, `eval/aggregate_cross_modal.py`, `eval/backfill_no_verdict.py`, `eval/consolidate_arms.py` | `crossmodal/eval/…` | same |
| `eval/upload_helper.py` | `shared/upload_helper.py` | pre-`shared/` leftover; `shared/` now also exists on the cluster |
| `eval/parse_score.py` | `freegen/parse_score.py` | **pre-split copy** |

### The shadowing trap this creates

`freegen/run_eval.py` puts both `freegen/` and `eval/` on `sys.path`, because
`dataset_io` lives only in `eval/` and `parse_score` only in `freegen/` — *locally*.
On the cluster **both** directories hold a `parse_score.py`, so whichever comes first
wins.

It used to be `sys.path.insert(0, …/eval)` **after** inserting `freegen/`, which put
`eval/` first. Locally that is harmless — there is no `eval/parse_score.py` — so the
bug is invisible on the desktop and fatal on torch:

```
ImportError: cannot import name 'is_no_verdict' from 'parse_score'
             (/home/ehb7466/pde-llm-eval/eval/parse_score.py)
```

Four canary jobs, twenty seconds each, 2026-08-24.

Two things now hold it shut, and both are needed:

1. `eval/` is **appended**, never inserted at 0, in `freegen/run_eval.py` and
   `freegen/report.py`. Pinned by `tests/test_import_shadowing.py`, which reads the
   source rather than the resolved import — the local filesystem cannot reproduce the
   condition.
2. `eval/parse_score.py` on the cluster was **converged** onto `freegen/parse_score.py`
   (verified a superset: every name in the old file exists in the new one), so both
   paths now resolve to the same code and `eval/frontier/parse_frontier.py` gains
   `strip_think()` and `is_no_verdict()` rather than losing anything. The pre-split
   copy is kept at `tools/_superseded/parse_score.py.eval-presplit-2026-08-19`.

**Rule:** before adding a `sys.path` entry in this repo, check whether the cluster's
flat `eval/` already holds a module of that name. Local absence proves nothing.

## SLURM

| | value |
|---|---|
| GPU partition / account | `h200_courant` / `torch_pr_427_courant` |
| CPU partition / account | `cpu_short` / `torch_pr_427_general` |

**CPU and memory are capped per GPU.** `--mem=256G --cpus-per-task=32` against
`--gres=gpu:h200:1` is refused outright with *"partition is not valid for this job /
GPU job setup is not valid"* — not a warning, a rejection at submit. One h200 takes
`--mem=128G --cpus-per-task=8`. That is what `run_cross_modal_consistency.sbatch`
runs these same models at. Scale both whenever you change the GPU count.

**`--export` splits on commas.** `sbatch --export=A=1,GT_SAMPLES=a,b,c` silently
truncates at the first comma inside the value and the job exits 0 having run a
different subset. Export in the parent shell and submit with `--export=ALL`; that is
what `sbatch/launch_freegen_xmodal.sh` does.

## Jobs die in three ways here, and only one of them is your bug

From the 2026-08-24 free-generation roster — eight one-model jobs, five finished:

| symptom | what it is | what to do |
|---|---|---|
| `CANCELLED by 0`, no comment | root's GPU-utilization sweep | raise the batch size |
| `TIMEOUT` at the wall | under-provisioned wall or a stalled batch size | both, usually |
| exit != 0 | actually your bug | read the log |

`CANCELLED by 0` looks alarming and is routine. Both jobs that took it were partway
through a 96-sequence batch that had drained to its last few stragglers — the exact
window a utilization sampler sees as idle. The cure is more concurrent work, not a
shorter generation: the consistency runner drives these same models at `BATCH_SIZE=64`
(192 concurrent sequences at k=3) and completes full arms on one H200.

Batch size is safe to change: `SamplingParams` carries a **per-request** seed, so one
prompt yields the same k draws regardless of who shares its batch. It moves throughput
and flush granularity, nothing observable.

**Everything here is resumable and you should rely on that.** `run_eval.py` skips any
`(title, mod_type, model, sample_idx)` already on disk, so a killed job is re-submitted
as-is and continues. Do not restart from zero.

### Uploads must not be tied to a model finishing

`upload_partial` used to run only at the *end* of a model, in both branches of the
roster loop. bash keeps going after a crash, so those branches cover a crash — but a
kill takes the whole script down and neither runs. Three jobs died mid-model and 1,176
finished draws sat on scratch with nothing on HF.

There is now a background heartbeat uploader (`UPLOAD_EVERY`, default 1200s) plus a
`trap … EXIT TERM INT` flush. It is lock-guarded against the end-of-model upload,
because `push_dataset_to_hub` **replaces** the split and two overlapping uploads of one
repo race. It runs in a background *process* — HF's threads kill vLLM's EngineCore if
they are created in the same one. Pinned by `tests/test_partial_upload_wiring.py`,
including that it survives `set -e`: `[ … ] && continue` as the last command of a loop
body exits the background subshell and silently disarms the heartbeat.

### "Uploaded" is not "visible"

`hf_utility` writes the manifest's `experiment_id` from `metadata["experiment_id"]`,
and `import_experiments.py` keeps only manifest rows that have one. Nothing set it, so
all eight arms uploaded cleanly, verified cleanly, and did not appear on the Artifacts
tab. `upload_helper.py` now takes `--experiment_id` separately from `--experiment`:
the first is the notes-folder name the dashboard joins on (`pde-freegen-xmodal`), the
second is the HF naming slug that prefixes the dataset names (`pde-llm-eval`). They are
different strings and collapsing them breaks the naming check.

## `$HOME` hygiene

Operational scripts go in `pde-llm-eval/tools/`, never loose in `$HOME`. See
`tools/README.md` on the cluster for what each one is and whether it is still live.
Retired one-shots go to `/scratch/ehb7466/_trash/<date>-<what>/`; eleven of them plus a
backup tarball were moved out of the `$HOME` top level on 2026-08-24, which now holds
no loose files at all (10,583 of 30,000).

`$HOME` is 50GB but only **30,000 files**, and the file quota is the one that binds.
Anything regenerable belongs on scratch with a symlink back.
