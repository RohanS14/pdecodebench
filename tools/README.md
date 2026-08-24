# tools/

One-off operational scripts for the cross-representation consistency run. They lived
loose in $HOME until 2026-08-24; nothing referenced them by an absolute home path, so
moving them here broke nothing.

These are **operational**, not experimental. The experiment itself is `eval/` and
`sbatch/`; the canonical source of both is the laptop repo at
`private_projects/pde-llm-eval`, pushed here with `raca upload`.

| script | what it does | still useful? |
|---|---|---|
| `probe_configs.py` | CPU-side vLLM `ModelConfig` probe. Registry membership is not loadability; this is the free gate that catches a model that cannot build its config before an allocation is spent on it. | **yes** — this is the Gate 0 the v6 review asks for |
| `probe.sbatch` | cpu_short wrapper for `probe_configs.py` | **yes** |
| `measure_tokens.py` | Per-model prompt length, with that model's own tokenizer and chat template. A chars-to-tokens ratio is not portable across tokenizers. | **yes** — how the token budgets were sized |
| `tokens.sbatch` | cpu_short wrapper for `measure_tokens.py` | yes |
| `stage_upload.sh` | (lives at `../stage_upload.sh`) snapshot an arm, drop a torn final line, upload from the snapshot | yes |
| `launch_gen.sh` | one job per model, each with its own OUTPUT_DIR and HF repo so concurrent jobs cannot clobber each other's uploads | yes — the pattern the v6 run needs |
| ~~`swap_nemotron.sh`~~ | gave Nemotron a full wall instead of a shared roster's leftover | **removed 2026-08-24** |
| ~~`relaunch.sh`, `relaunch2.sh`~~ | resubmission after the QOS gpu48 2-GPU ceiling was diagnosed | **removed 2026-08-24** |
| ~~`smoke.sbatch`~~ | single-model smoke test on h200_courant | **removed 2026-08-24** |
| ~~`patch_cluster.py`~~ | one-shot in-place roster expansion of the cluster's flat copy | **removed 2026-08-24** — effect verified in `eval/run_cross_modal_consistency.py`: all five added models present |
| ~~`patch_shepherd.py`~~ | one-shot fix of the octal bug in `sbatch/autofollow_xmodal.sh` | **removed 2026-08-24** — effect verified: four `10#` guards present in that script |
| ~~`fix_shard.py`~~ | one-shot repair of a Qwen3.8 shard | **removed 2026-08-24** — effect verified: `qwen3-8-27b-final` holds 3,072 rows |
| `verify_hf_merge.py` | proves one HF cache is fully contained in another before anything is deleted. Matching directory names are not evidence — three "duplicate" models held blobs the destination did not. | **yes** |

`_superseded/` is gone. It held two files: `stage_upload.sh.homecopy`, verified
byte-identical to `../stage_upload.sh` (both md5 `cec0cb7d…`), and
`parse_score.py.eval-presplit-2026-08-19`, superseded by `freegen/parse_score.py`
after the cluster's copy was converged onto it. Keeping a directory of files whose
only job is "so nothing is lost" is what git is for; both are in history at 50a31c4.

The four historical scripts above were removed on the same reasoning. They
resubmitted specific, long-finished jobs — `relaunch.sh` opens by cancelling
`16135020 16135021 16135023 16137257` — so re-running one now would act on
whatever holds those IDs today. What was worth keeping from them is the finding,
and it is written down: the 2-GPU ceiling came from QOS `gpu48`, auto-assigned at
walltimes of 48h or less, not from any partition.

Three one-shot patch scripts were removed on 2026-08-24 at the researcher's
request. Each mutated a file in place and each mutation was confirmed still
present before deletion, so the patched files ARE the record; re-running any of
them against an already-patched file would have been the actual risk.
