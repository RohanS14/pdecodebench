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
| `swap_nemotron.sh` | gave Nemotron a full wall instead of a shared roster's leftover | historical |
| `relaunch.sh`, `relaunch2.sh` | resubmission after the QOS gpu48 2-GPU ceiling was diagnosed | historical |
| `smoke.sbatch` | single-model smoke test on h200_courant | historical |
| ~~`patch_cluster.py`~~ | one-shot in-place roster expansion of the cluster's flat copy | **removed 2026-08-24** — effect verified in `eval/run_cross_modal_consistency.py`: all five added models present |
| ~~`patch_shepherd.py`~~ | one-shot fix of the octal bug in `sbatch/autofollow_xmodal.sh` | **removed 2026-08-24** — effect verified: four `10#` guards present in that script |
| ~~`fix_shard.py`~~ | one-shot repair of a Qwen3.8 shard | **removed 2026-08-24** — effect verified: `qwen3-8-27b-final` holds 3,072 rows |

`_superseded/` holds files kept only so nothing is lost:
`stage_upload.sh.homecopy` was byte-identical (md5 cec0cb7d…) to `../stage_upload.sh`.


Three one-shot patch scripts were removed on 2026-08-24 at the researcher's
request. Each mutated a file in place and each mutation was confirmed still
present before deletion, so the patched files ARE the record; re-running any of
them against an already-patched file would have been the actual risk.
