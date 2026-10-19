"""
upload_traces.py — publish the full agentic belief-revision traces behind §4.2
(static vs agentic validity judgments) to a public HuggingFace dataset.

Four arms, 256 episodes each, one per row of the full Code Validity dataset
(32 solvers x 8 conditions):

  arm                    model              thinking         source run
  gemini25flash_tb0      gemini-2.5-flash   budget 0         results/frontier/score_256_25_tb0
  gemini25flash_tb8192   gemini-2.5-flash   budget 8192      results/frontier/score_256_25_tb8192
  gemini37flash_low      gemini-3.7-flash   level low        results/frontier/score_256_37_low
  gemini37flash_high     gemini-3.7-flash   level high       results/frontier/score_256_37_pathfix

Three configs, each with an `arm` column:
  episodes  Stage 2 agentic episodes (full action_trace, every tool call and result)
  stage1    Stage 1 static judgments (prompt, code, full response)
  judge     LLM-judge ratings of the invalid-row justifications

The raw JSONL files are uploaded too, under raw/<arm>/, so the repo's analysis
scripts can be pointed at them unchanged.

The one transformation applied: this machine's absolute repo-root prefix is
stripped from every string (episode_dir, and Python tracebacks the sandbox returned
to the model), leaving repo-relative paths. Nothing is truncated. Nested
action_trace / s2_submit_args are stored as JSON strings in the parquet configs so
the schema is stable across arms; the raw/ copies keep them nested.

Needs the RACA workspace's hf_utility + key_handler (run with the workspace's
.tools-venv/bin/python).

Usage:
  python frontier/upload_traces.py --dry-run
  python frontier/upload_traces.py
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "frontier"

DATASET_NAME = "agentic-belief-revision-traces-v1"
EXPERIMENT = "agentic-belief-revision"

ARMS = {
    "gemini25flash_tb0":    ("score_256_25_tb0",     "gemini25flash", "tb0"),
    "gemini25flash_tb8192": ("score_256_25_tb8192",  "gemini25flash", "tb8192"),
    "gemini37flash_low":    ("score_256_37_low",     "gemini37flash", "tl_low"),
    "gemini37flash_high":   ("score_256_37_pathfix", "gemini37flash", "tl_high"),
}
EXPECTED = {"episodes": 1024, "stage1": 1024, "judge": 488}
# The 3.7-high run was judged first with the v1 judge (one merged
# contains_incorrect_claims flag), then re-judged with the v2 split flags that every
# other arm and the paper use; its judge_results.jsonl is the stale v1 output.
JUDGE_FILE = {"gemini37flash_high": "judge_results_v2_split.jsonl"}
# Paper §4.2: share of judged justifications with a false claim about the code.
PAPER_FALSE_CLAIM_RATE = {"gemini25flash_tb0": 0.388, "gemini25flash_tb8192": 0.208,
                          "gemini37flash_low": 0.049, "gemini37flash_high": 0.032}
JSON_COLUMNS = {"episodes": ["action_trace", "s2_submit_args", "s2_tools_used"]}

PREFIX = str(REPO_ROOT) + "/"
LEAK_MARKERS = [str(REPO_ROOT), str(Path.home()), "/Users/", "/private/tmp/"]


def arm_files(arm: str) -> dict[str, Path]:
    run, slug, cond = ARMS[arm]
    root = RESULTS / run / slug
    return {
        "episodes": root / cond / f"{slug}__belief_revision_agentic.jsonl",
        "stage1": root / cond / "stage1_cache" / f"{slug}__stage1_cache.jsonl",
        "judge": root / "judge" / JUDGE_FILE.get(arm, "judge_results.jsonl"),
    }


def scrub(obj, counter: list[int]):
    """Recursively strip the absolute repo-root prefix from every string."""
    if isinstance(obj, str):
        if PREFIX in obj:
            counter[0] += obj.count(PREFIX)
            return obj.replace(PREFIX, "")
        return obj
    if isinstance(obj, list):
        return [scrub(x, counter) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v, counter) for k, v in obj.items()}
    return obj


def load_all() -> tuple[dict[str, list[dict]], dict[str, dict[str, list[dict]]], dict[str, int]]:
    """Returns (rows per config, raw scrubbed rows per arm per config, scrub counts)."""
    rows = {c: [] for c in EXPECTED}
    raw = {arm: {} for arm in ARMS}
    scrubbed = {c: 0 for c in EXPECTED}
    for arm in ARMS:
        for config, path in arm_files(arm).items():
            counter = [0]
            recs = [scrub(json.loads(line), counter) for line in path.open()]
            scrubbed[config] += counter[0]
            raw[arm][config] = recs
            for r in recs:
                flat = {"arm": arm, **r}
                for col in JSON_COLUMNS.get(config, []):
                    flat[col] = json.dumps(flat[col], ensure_ascii=False)
                rows[config].append(flat)
    return rows, raw, scrubbed


def unify_types(rows: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Arrow needs one type per column. A few columns differ by arm -- e.g.
    thinking_setting is an int budget (0 / 8192) for 2.5 Flash and a level string
    ("low" / "high") for 3.7 Flash -- so store any mixed-type column as str."""
    coerced = {}
    for config, recs in rows.items():
        cols = {k for r in recs for k in r}
        for col in sorted(cols):
            types = {type(r.get(col)) for r in recs if r.get(col) is not None}
            if len(types) > 1 and types != {int, float}:
                for r in recs:
                    if r.get(col) is not None:
                        r[col] = str(r[col])
                coerced.setdefault(config, []).append(col)
    return coerced


def check(rows: dict[str, list[dict]], raw) -> None:
    for config, n in EXPECTED.items():
        assert len(rows[config]) == n, f"{config}: {len(rows[config])} rows, expected {n}"
    for config, recs in rows.items():
        blob = json.dumps(recs, ensure_ascii=False)
        for m in LEAK_MARKERS:
            assert m not in blob, f"{config}: '{m}' still present after scrub"
    blob = json.dumps(raw, ensure_ascii=False)
    for m in LEAK_MARKERS:
        assert m not in blob, f"raw: '{m}' still present after scrub"
    # every config has one schema across arms (from_list would silently drop extras)
    for config, recs in rows.items():
        keysets = {frozenset(r) for r in recs}
        assert len(keysets) == 1, f"{config}: {len(keysets)} different column sets across rows"
    for arm, rate in PAPER_FALSE_CLAIM_RATE.items():
        js = [r for r in rows["judge"] if r["arm"] == arm]
        got = sum(bool(r["contains_incorrect_code_claims"]) for r in js) / len(js)
        assert round(got, 3) == rate, f"{arm}: false-claim rate {got:.3f} != paper {rate}"
    # one episode per (arm, title) -- duplicates would mean a resumed run double-wrote
    keys = [(r["arm"], r["title"]) for r in rows["episodes"]]
    assert len(set(keys)) == len(keys), "duplicate (arm, title) in episodes"


COLUMN_DOCS = {
    "episodes": {
        "arm": "Which of the four model x thinking arms produced this row (see table above)",
        "title": "Row id from the Code Validity dataset, e.g. Heat_Comm_Valid_7; joins to stage1/judge",
        "mod_type": "Lexical/validity condition of the snippet (Comm_Valid, NoComm_InValid, CorrComm, NoComm_CorrVar, ...)",
        "gt_valid": "Ground truth: True if the solver implements valid physics",
        "s1_response": "Full Stage 1 (static, no tools) model response text, untruncated",
        "s1_validity_score": "Stage 1 validity confidence score parsed from the response",
        "s2_submit_args": "JSON string: the arguments of the model's final submit call (pde, method, behavior, valid, validity_score and their explanations)",
        "s2_validity_score": "Stage 2 (agentic) validity confidence score from the final submission",
        "transition": "Stage 1 -> Stage 2 validity verdict transition (e.g. correct->correct, wrong->correct)",
        "action_trace": "JSON string: list of every turn -- tool (edit_source / run_diagnostic / submit), args, full tool result returned to the model, reasoning_text, thought_summary, token counts, finish_reason",
        "s2_action_count": "Number of edit_source + run_diagnostic actions taken (budget 6)",
        "s2_tools_used": "JSON string: list of tool names used in the episode",
        "episode_dir": "Repo-relative path of the episode's sandbox working directory (opaque slug)",
        "total_cost_usd": "API cost of the episode in USD (Stage 2; see frontier/model_pricing.py)",
    },
    "stage1": {
        "arm": "Which of the four model x thinking arms produced this row",
        "title": "Row id from the Code Validity dataset; joins to episodes/judge",
        "code": "The solver code shown to the model, exactly as sent",
        "prompt_s1": "Full Stage 1 prompt",
        "s1_response": "Full Stage 1 model response text, untruncated",
        "s1_parsed_valid": "Validity verdict parsed from the response",
        "s1_valid_match": "1 if the parsed verdict matches ground truth",
        "s1_cost_usd": "API cost of the Stage 1 call in USD",
    },
    "judge": {
        "arm": "Which agent arm's Stage 2 justification was judged",
        "title": "Row id; joins to episodes",
        "category": "How many of the true causal defects the justification identified: all / some / none",
        "contains_incorrect_code_claims": "True if the justification asserts something false about the code (the paper's false-claim rate)",
        "contains_unverifiable_execution_claims": "True if it cites runtime figures the judge cannot check (the judge sees no execution output); not an error",
        "explanation": "Judge's full rationale",
        "used_ground_truth_reference": "Whether the judge was given the valid reference solver to diff against",
        "judge_model": "Model used as judge",
    },
}


def readme(org: str) -> str:
    configs_yaml = "\n".join(
        f"- config_name: {c}\n  data_files:\n  - split: train\n    path: {c}/train-*" for c in EXPECTED)
    col_tables = []
    for config, docs in COLUMN_DOCS.items():
        lines = "\n".join(f"| `{k}` | {v} |" for k, v in docs.items())
        col_tables.append(f"### `{config}` ({EXPECTED[config]} rows)\n\nKey columns (all others are "
                          f"documented in the source code):\n\n| column | description |\n|---|---|\n{lines}")
    return f"""---
license: mit
tags:
- agents
- code
- scientific-reasoning
- pde
configs:
{configs_yaml}
---

# Agentic belief-revision traces (Solver2System, §4.2)

Full model traces for the agentic validity experiment in *Solver to System: LLM Reasoning
Across Representations and Execution in Scientific Simulation* (Sim2Science @ NeurIPS 2026).
Code: https://github.com/RohanS14/pdecodebench (`frontier/`).

Each episode: the model first judges a PDE solver statically (Stage 1), then may take up to
**6 actions** — `edit_source` (patch the solver) and `run_diagnostic` (execute a script in a
sandbox) — before submitting a revised judgment (Stage 2). Inputs are all 256 rows
(32 solvers × 8 conditions) of the [Code Validity dataset](https://huggingface.co/datasets/bermaneh/pde-llm-eval-code-perturbation-dataset).

| arm | model | thinking | episodes | judged |
|---|---|---|---|---|
| `gemini25flash_tb0` | gemini-2.5-flash | budget 0 | 256 | 121 |
| `gemini25flash_tb8192` | gemini-2.5-flash | budget 8192 | 256 | 120 |
| `gemini37flash_low` | gemini-3.7-flash | level low | 256 | 123 |
| `gemini37flash_high` | gemini-3.7-flash | level high | 256 | 124 |

Generation: `max_output_tokens=50000` per call; tool output returned to the model capped at
4000 chars (`--truncate_chars 4000`, an experimental setting — recorded in full here as the model
saw it); 120 s subprocess timeout. Judge: gemini-3.1-pro-preview, run on the invalid rows the
agent correctly called invalid (`gt_valid=False` and `s2_valid_match=1`), rating its Stage 2 justification.

```python
from datasets import load_dataset
eps = load_dataset("{org}/{DATASET_NAME}", "episodes", split="train")
s1 = load_dataset("{org}/{DATASET_NAME}", "stage1", split="train")
judge = load_dataset("{org}/{DATASET_NAME}", "judge", split="train")
import json; trace = json.loads(eps[0]["action_trace"])
```

`raw/<arm>/{{episodes,stage1,judge}}.jsonl` are the same records in the repo's native JSONL
format (nested fields not stringified) for use with the repo's analysis scripts.

**Processing:** the only change from the run outputs is that the absolute path of the machine's
repo root was stripped from all strings (episode working dirs and Python tracebacks), leaving
repo-relative paths. No responses are truncated. In the parquet configs, `thinking_setting` is a
string (`"0"`, `"8192"`, `"low"`, `"high"`) because its type differs between model families;
the raw/ JSONL keeps the original types.

## Columns

{chr(10).join(col_tables)}

## Citation

```bibtex
@inproceedings{{berman2026solver2system,
  title     = {{Solver to System: {{LLM}} Reasoning Across Representations and Execution in Scientific Simulation}},
  author    = {{Berman, Eliza and Subramanian, Rohan and Jha, Shreya and Jagarapu, Lohit and Lopez, Romain and Ranganath, Rajesh}},
  booktitle = {{NeurIPS 2026 Workshop on Sim2Science: ML with Imperfect Scientific Models}},
  year      = {{2026}}
}}
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Load, scrub and check; print a summary; upload nothing")
    p.add_argument("--org", default=None, help="HF org (default: resolved by hf_utility)")
    p.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    p.add_argument("--only", choices=list(EXPECTED), default=None,
                   help="Re-upload just this config (+ its raw/ files and the card)")
    args = p.parse_args()

    rows, raw, scrubbed = load_all()
    coerced = unify_types(rows)
    check(rows, raw)
    print(f"mixed-type columns stored as str: {coerced or 'none'}")
    for config in EXPECTED:
        per_arm = {a: sum(r["arm"] == a for r in rows[config]) for a in ARMS}
        print(f"{config:9s} {len(rows[config]):5d} rows  {per_arm}  prefixes stripped: {scrubbed[config]}")
    print("leak check: OK (no absolute paths remain)")

    if args.dry_run:
        s = rows["episodes"][0]
        trace = json.loads(s["action_trace"])
        print(f"\nsample episode: arm={s['arm']} title={s['title']} mod_type={s['mod_type']} "
              f"gt_valid={s['gt_valid']} transition={s['transition']} actions={s['s2_action_count']}")
        print(f"  episode_dir: {s['episode_dir']}")
        print(f"  s1_response: {len(s['s1_response'])} chars; trace turns: {len(trace)}; "
              f"tools: {[t['tool'] for t in trace]}")
        print(f"  final submit: {s['s2_submit_args'][:300]}")
        return

    sys.path.insert(0, str(REPO_ROOT.parent.parent / "packages" / "key_handler" / "key_handler"))
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
    from datasets import Dataset
    from huggingface_hub import HfApi
    from hf_utility import push_dataset_to_hub
    from hf_utility.config import get_org
    from hf_utility.manifest import update_manifest_entry
    from hf_utility.upload import upload_file_to_repo

    org = args.org or get_org()
    repo_id = f"{org}/{DATASET_NAME}"
    # push_dataset_to_hub passes private=False, which only applies when it creates the
    # repo -- so create it first with the visibility we want, then assert it held.
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    metadata = {
        "script_name": "frontier/upload_traces.py",
        "model": "gemini-2.5-flash, gemini-3.7-flash (judge: gemini-3.1-pro-preview)",
        "description": "Full agentic belief-revision traces (Stage 1, Stage 2 episodes, judge) for the 4 arms in Solver2System §4.2",
        "hyperparameters": {"action_budget": 6, "max_output_tokens": 50000, "truncate_chars": 4000,
                            "subprocess_timeout_s": 120, "thinking": "2.5: budget 0/8192; 3.7: level low/high"},
        "input_datasets": ["bermaneh/pde-llm-eval-code-perturbation-dataset"],
        "experiment_name": EXPERIMENT,
        "artifact_status": "final",
        "canary": False,
    }
    configs = [args.only] if args.only else list(EXPECTED)
    for config in configs:
        push_dataset_to_hub(dataset=Dataset.from_list(rows[config]), dataset_name=DATASET_NAME,
                            metadata=metadata, tags=[EXPERIMENT, config], org=org, config_name=config,
                            skip_readme=True, skip_manifest=True)

    with tempfile.TemporaryDirectory() as tmp:
        for arm, per_config in raw.items():
            for config, recs in per_config.items():
                if args.only and config != args.only:
                    continue
                f = Path(tmp) / f"{arm}__{config}.jsonl"
                f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
                upload_file_to_repo(DATASET_NAME, str(f), f"raw/{arm}/{config}.jsonl",
                                    org=org, max_size_bytes=100 * 1024 * 1024)

    api.upload_file(path_or_fileobj=readme(org).encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    private = api.repo_info(repo_id, repo_type="dataset").private
    assert private == (not args.public), f"{repo_id} visibility is private={private}"
    update_manifest_entry(dataset_name=DATASET_NAME, metadata=metadata, tags=[EXPERIMENT])
    print(f"\nDone ({'private' if private else 'public'}): https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
