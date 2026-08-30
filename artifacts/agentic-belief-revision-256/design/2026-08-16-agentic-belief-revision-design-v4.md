# Agentic Belief-Revision Experiment Design v4 (supersedes v3, reviewed and approved by Rohan)

## Context

The belief-revision experiment tests whether a frontier model's lexical priors about
a PDE solver snippet (its judgment of `pde` / `method` / `behavior` / `valid`) can be
overcome by execution evidence, and — where the snippet is genuinely invalid — whether
the model can localize the actual defect rather than just flag its presence.

Stage 1 is a single read-only turn: the model sees the snippet and answers all four
fields with no tool access. Stage 2 is agentic: the model is given three tools
(`edit_source`, `run_diagnostic`, `submit_final_answer`) and a fixed investigative
action budget, and gathers whatever evidence it chooses before answering again. The
harness never pre-selects "the important variable" or hands the model a precomputed
trajectory summary — every fact the model sees in Stage 2, beyond the bare source
text, is something its own code explicitly computed and printed.

Stage 2 now runs as **two independently-scored conditions** per row — thinking budget
0 ("nothink") and 1536 ("think") — sharing one cached, deterministic Stage-1 answer.
A separate LLM-judge module additionally scores, for rows the model correctly called
invalid, whether its justification identifies the real injected defect or just the
right label. Together these turn a single S1-vs-S2 accuracy comparison into a richer
three-condition, mechanism-aware picture of how much and what kind of evidence
overcomes a syntactic prior.

## Goals

- Give the model real agency over evidence-gathering: it chooses what to inspect, what
  to instrument, and when to stop, rather than being handed a fixed evidence bundle.
- Preserve the dataset's actual obfuscation/lexical-prior manipulation — never let the
  harness reveal which variable matters, only raw structural facts (or nothing, until
  the model creates something itself).
- Keep the harness minimal: reuse existing Gemini API plumbing, scoring code, and
  subprocess-execution patterns rather than building new infrastructure where an
  existing pattern already works.
- Run reliably at scale: episodes must survive real disk/cost/encoding failure modes
  without crashing the whole sweep, and a sweep must be safely resumable so a larger
  run can reuse a smaller one's results rather than repeating them.

## Architecture

### Tool-calling mechanism

Uses the existing `google-genai` API (`client.models.generate_content`), **manual
(non-automatic) function calling**:

- Tools declared as `types.FunctionDeclaration(name=..., description=...,
  parameters_json_schema={...})`, wrapped in `types.Tool(function_declarations=[...])`,
  passed via `GenerateContentConfig(tools=[...])`.
- `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)` — the
  harness, not the SDK, executes every tool call inside the sandbox with its own
  timeouts and logging.
- With AFC disabled: `response.function_calls` gives the parsed tool name + args; the
  harness executes it, wraps the result as
  `types.Part.from_function_response(name=..., response={...})` inside
  `types.Content(role='tool', parts=[...])`, and appends it to the growing `contents`
  list before the next `generate_content()` call.
- **Model**: `gemini-2.5-flash` throughout (Stage 1 and both Stage-2 conditions),
  matching the static-experiment data for a clean comparison. Caveat: this model is on
  a deprecation path (scheduled shutdown October 16, 2026) — worth re-checking
  availability before any run past that date.

### Function-calling mode per turn

Gemini's `FunctionCallingConfig.mode` is actively managed every turn, not left at a
single fixed setting:

- **Default: `VALIDATED`**, on every turn including the terminal (budget-exhausted)
  phase. This does not force a function call — the model can respond with plain text
  (or nothing) first, giving it room to reason before committing to a structured call,
  rather than being coerced into an immediate tool call.
- **`ANY`** (which does force a call) is used only as a bounded, one-shot escalation:
  - a text-only response under `VALIDATED` (the model explicitly chose not to act) —
    escalates to `ANY` immediately on the next turn, no retry.
  - two consecutive fully-empty responses (no call, no text) under `VALIDATED` — one
    retry is given first (with feedback appended), escalating to `ANY` only on the
    second consecutive empty.
  - the escalation is consumed after one turn — the turn after an escalation always
    reverts to `VALIDATED`, never sticky.
- Both empty and text-only non-action turns under `VALIDATED` still consume one action
  from the investigative budget — that, not the mode switch itself, is what
  discourages stalling.
- `ANY` is not guaranteed to produce a call. If it still doesn't, the harness gives up
  and forces an empty synthetic submission (`_forced_reason: "no_function_call"`)
  rather than looping forever.
- Separately, a model calling a tool that wasn't declared to it that turn (the API
  does not enforce the declared-tool-list itself) is rejected and bounded: after
  `MAX_PROTOCOL_VIOLATIONS` (2) such rejections, the harness forces an empty synthetic
  submission rather than allowing an indefinite loop.

### The three tools

```
edit_source(diff: str)
  - Purpose: collect new evidence. Available every turn; counts against budget.
  - Model outputs a unified diff against the latest version of the source. The
    harness applies it to a NEW versioned file (solver_v1.py, v2.py, ...), never
    overwriting in place, and reruns the FULL simulation via the sandbox (subprocess,
    eval/.venv, timeout). An empty/no-op diff is valid — it reruns the unchanged
    code, not an error.
  - Returns, in the same turn: which file the diff applied to and was saved as, plus
    the actual stdout/stderr from that run (including print()s and assertion
    failures) — this is deliberate, matching real "add a print, rerun, see the
    output" debugging.
  - The harness does NOT automatically capture, snapshot, or persist anything from
    the execution's namespace. Anything the model wants available to a later
    run_diagnostic call without re-running the simulation must be explicitly written
    to disk by its own diff (e.g. np.savez(...)) at a plain relative path (cwd is
    already the episode's scratch directory).

run_diagnostic(script: str)
  - Purpose: run further analysis on evidence already collected. Available every
    turn (though a poor first move, since only the source snippet exists at turn 0);
    counts against budget. Also the model's only avenue to do things like list a
    directory's contents or inspect a saved .npz's keys — no separate free "look
    around" step exists.
  - Model outputs a standalone Python script, run as-is with no modification, from
    the same sandbox directory as the most recent edit_source execution. It can read
    the current source file(s) and anything the model's own code has previously
    written to disk — nothing else. Never reruns or touches the simulation itself.
  - Returns, in the same turn: which filename the script was saved as, plus its
    actual stdout/stderr — same truncation and no-automatic-summarization rules as
    edit_source.

submit_final_answer(pde, method, behavior, valid, pde_exp, method_exp, behavior_exp, valid_exp)
  - Purpose: end the episode. Available every turn regardless of remaining budget —
    the model can stop voluntarily, or be forced into it once edit_source/
    run_diagnostic have dropped from the tool list.
  - Does NOT count against the budget — the budget measures investigation, not the
    act of ending an episode.
  - Arguments mirror the 4-field answer format used everywhere else in this
    pipeline, plus 4 explanation fields (pde_exp/method_exp/behavior_exp/valid_exp),
    a 1-2 sentence rationale for the answer and for whether/why it changed from the
    Stage-1 answer.
```

### Ending an episode

Every episode ends one of three ways:

1. **Voluntary submission** — the model calls `submit_final_answer` while budget
   remains. The harness intercepts the *first* such call: it does not end the episode,
   logs it with `provisional_submit: True`, and replies (as the tool's function
   response) with the ORIGINAL, unedited code snippet plus the model's own Stage-1
   answer (`build_submit_confirmation_reminder`), requiring a second
   `submit_final_answer` call to actually finalize. This exists because, without it, a
   model can judge the validity of its own *edited* copy of the code and report that
   judgment as if it applied to the original snippet.
2. **Forced completion** — the budget is exhausted, `submit_final_answer` is the only
   tool left. Since the harness knows in advance this turn is submit-only, it shows the
   same original-code-plus-Stage-1-answer content *proactively*, before the model's one
   deterministic attempt, rather than intercepting reactively as in (1).
3. **Disk-safety abort** — a guard below force-ends the episode; there is no
   `submit_args` to score.

Both (1) and (2) share a `actions_remaining_at_submission` field (0 for forced
completion, >0 for a voluntary stop), so post-hoc analysis can separate "the model
decided it was done early" from "it ran out of runway."

### Thinking-budget conditions

Stage 2 runs as two independent sweeps per row: `thinking_budget=0` ("nothink") and
`thinking_budget=1536` ("think"). Stage 1 is unaffected by and shared across both —
`call_gemini()` hardcodes `thinking_budget=0` for Stage 1 regardless of which Stage-2
condition it's paired with, and is fully deterministic (temperature=0), so it's
computed once per row and reused, never re-run per condition.

### Sandbox / file layout

- One isolated scratch directory per episode:
  `results/frontier/_agentic_work/<title>/<run_id>/`. `solver_v0.py` is the original
  given code, copied in at episode setup; `solver_v1.py`, `v2.py`, ... follow, one per
  successful `edit_source` application, never overwritten in place.
- Every subprocess (both tools) runs with `cwd` set to this directory, so anything the
  model's own code writes lands there and is visible to later `run_diagnostic` calls
  with no extra plumbing.
- **Retention: kept, not cleaned up** — needed for post-hoc debugging and analysis.
- Sandboxing is a *soft* boundary (timeouts + `cwd` isolation), not a hard security
  boundary.
- **Disk-safety limits**, enforced on every execution:
  - A hard **per-file write cap** (`MAX_FILE_SIZE_BYTES`, 1GB) via `RLIMIT_FSIZE` —
    kernel-level, the only mechanism that can actually prevent an oversized write
    rather than clean up after it. In practice CPython ignores `SIGXFSZ`, so the write
    is truncated exactly at the cap and the process exits normally (`OSError: [Errno
    27] File too large` if uncaught) rather than being signal-killed — detection can't
    key off exit code/signal, so the oversized-candidate scan below runs
    unconditionally after every execution.
  - After every execution, the harness scans top-level files at or above 90% of the
    cap. Exactly one match → moved to a model-invisible quarantine directory (sibling
    of the episode dir), episode continues normally. Zero or multiple matches →
    ambiguous, `abort_reason="ambiguous_oversized_write"`, nothing moved, episode
    force-ended for manual inspection.
  - A **cumulative episode-directory cap** (`MAX_EPISODE_DIR_BYTES`, 1.5GB), checked
    regardless of outcome — `RLIMIT_FSIZE` is strictly per-file, so this is the only
    guard against many separate under-cap files compounding past a reasonable total.
    Detective only (whatever was written is already on disk); exceeding it sets
    `abort_reason="cumulative_dir_size_exceeded"`.

### Turn snapshots for logging

- After each turn, the harness copies the full current episode dir into an isolated,
  model-invisible per-turn snapshot:
  `results/frontier/_agentic_work/<title>/<run_id>_snapshots/turn<i>/`.
- Files at or above a 30MB dedup threshold that are byte-identical (same size and
  mtime) to their nearest previous *real* copy are not re-copied — a
  `<name>.unchanged.txt` placeholder is written instead, pointing at the real copy.
  Placeholder chains always resolve to the nearest real copy, not necessarily the
  immediately preceding turn. Turn 0 never dedups. Small scripts never cross the
  threshold, so are always copied normally — no extension-based special-casing.

### Budget, truncation, and cost controls

- Default **investigative budget**: 6 actions per episode, counting only
  `edit_source`/`run_diagnostic` calls; `submit_final_answer` never counts. Once
  exhausted, both drop from the tool list — only `submit_final_answer` remains.
- Every tool result is hard-truncated (character cap, default 4000) before being added
  to context, with an explicit "truncated, N more characters" note when it fires.
- Every model call (Stage 1, every Stage-2 turn, and the judge) caps
  `max_output_tokens` at 50,000 — necessary because thinking tokens share the same
  output budget as visible/function-call tokens, so an unbounded cap risks silently
  truncated function-call arguments, not just verbosity.
- A per-episode `EpisodeCostGuard` (default cap $0.50) tracks cumulative Stage-2 API
  cost; tripping it collapses the tool list identically to budget exhaustion (same
  `tools_available()` check covers both conditions). This is independent of the
  session-level cost caps the orchestrator and judge module enforce separately (see
  below).

### Orchestration & checkpoint/resume

`run_stratified_sweep.py` runs three phases in order against a dataset CSV and an
output root:

1. **Shared Stage-1 cache pass** — one call per row, independent of thinking budget,
   checkpointed to `stage1_cache/<slug>__stage1_cache.jsonl`.
2. **Stage 2, nothink** — the full agentic loop for every row, checkpointed to
   `nothink/<slug>__belief_revision_agentic.jsonl`.
3. **Stage 2, think** — same, to `think/<slug>__belief_revision_agentic.jsonl`.

Each phase has its own file-based checkpoint, keyed by row title, and its own
session-level cost cap (`--max_cost_usd_stage1`, `--max_cost_usd_per_condition`,
independent per condition). A resumed or re-pointed run skips every title already
present in the relevant checkpoint file and only processes the delta — this is what
lets a larger sweep (e.g. a 256-row dataset that is a superset of an already-completed
64-row stratified sample) reuse prior results without re-paying for them, simply by
pointing `--dataset`/`--output_root` at the larger file/directory.

### Scoring

- Stage 1 is scored via the existing `parse_response()` + `score_row()` pipeline,
  unchanged.
- Stage 2's `submit_final_answer` arguments arrive already structured — `parse_response()`
  is bypassed entirely, and the tool call's argument dict is fed directly into the same
  unmodified `score_row()`.
- Per-row output includes: field matches (`valid`/`pde`/`method_any`/`behavior_any`),
  hedge classification (`s1_hedge_class`/`s2_hedge_class` via `classify_hedge()`),
  `transition` (S1→S2 answer-class transition via `get_transition()`),
  `thinking_budget`, `aborted`/`abort_reason` (normal `s2_*` fields are omitted
  entirely when aborted, so downstream analysis can't mistake it for a real answer),
  full action bookkeeping (`s2_action_count`, `tools_used`, `used_edit_source`,
  `actions_remaining_at_submission`, `cost_guard_tripped`, `protocol_violations`, the
  complete turn-by-turn `action_trace`), and raw token counts alongside derived cost
  (so cost can always be recomputed later even if pricing constants change).
- **Primary comparison structure**: recall (fraction of truly-invalid rows correctly
  called invalid) and specificity (fraction of truly-valid rows correctly called
  valid), computed per mod_type stratum — each mod_type is homogeneous in `gt_valid`
  by construction, so the match column computed within one mod_type directly is the
  recall/specificity indicator — across all three conditions (S1, S2 nothink, S2
  think). This is richer than one pooled accuracy number and is what the
  visualization and judge analysis are built around.
- First-pass bug-localization metric: since a `gt_sample`'s valid/invalid pair differs
  by a small, near-mechanically-diffable change, check whether any `edit_source` diff
  the model applied touches the true differing line(s) — no LLM judge needed for this.
- **"Does this sample's code track full history vs. only the current timestep" is a
  fixed, pre-known property of the dataset** (the `SAVES_HISTORY` field in the
  independently-completed trajectory-saving audit), not something to infer from any
  episode's transcript — results should be stratified against that fixed label, not
  pooled. Known, accepted limitation: whichever variable happens to carry multi-step
  history is itself a soft hint about physical importance, independent of name
  obfuscation, and can't be hidden without discarding real data.

### Judge module (invalidity-reasoning quality)

A separate component, `judge_invalidity_reasoning.py`, scores *why* the model thinks
truly-invalid code is invalid, not just whether it said so:

- **Scope**: rows where `gt_valid == False AND s2_valid_match == 1` — the model must
  have also correctly classified the code as invalid; a row where the model said
  "valid" has no invalidity justification to judge.
- **Ground truth for the judge**: the row's valid-counterpart code (looked up via a
  hardcoded invalid→valid mod_type mapping and matching `gt_sample`), shown side by
  side with the corrupted code — not a pre-computed diff, and not solely the dataset's
  `invalidity_note`, which is confirmed generic/templated (identical wording reused
  across different rows and PDE classes).
- **Ground-truth-reference augmentation**: for 3 of the 4 invalid mod_types
  (`NoComm_InValid`, `NoComm_CorrVar_InValid`, `CorrComm_Invalid` — not `Comm_InValid`,
  which would be redundant), the judge is additionally shown the row's
  `Comm_Valid`/`Comm_InValid` sibling pair (real variable names + genuine, accurate
  comments), plus a mod_type-specific caveat note explaining what's obfuscated,
  missing, or misleading in the code it's about to see — needed because the judge
  otherwise has no way to translate obfuscated `foobar_N` names, absent comments, or
  deliberately-mismatched comments back to physical meaning.
- **Output schema**: `category` — a 3-way `none`/`some`/`all` classification (not
  binary, since correctly identifying one of several independently-injected defects
  while missing another is real, sufficient, correct reasoning, not a wrong answer)
  and `contains_incorrect_claims` — a separate bool (since `category` alone is a
  recall-only metric; a crisp correct explanation and a "shotgun" explanation with one
  real claim buried in wrong ones would otherwise score identically), plus a 1-3
  sentence explanation.
- **Model and cost**: a separate, stronger judge model (Pro-tier, currently resolved
  via the `gemini-pro-latest` alias) than the `gemini-2.5-flash` being evaluated, to
  reduce self-preference bias (accepted caveat: same-family models can still share
  blind spots) — with its own pricing constants, its own session-wide cost cap
  (`--max_cost_usd`), and per-row cost persisted.
- **Checkpoint**: keyed by `(title, thinking_budget)` pair, so a single
  `judge_results.jsonl` accumulates verdicts across multiple `--stage2_jsonl` inputs
  (nothink and think) without reprocessing either.
- Single-pass judge (no majority-vote redundancy) for now — accepted as noisier than a
  voting design.

### Visualization

`viz/report_belief_revision_agentic.py` reads both condition JSONLs plus the judge output and
produces one combined, self-contained HTML report (reusing `viz/report_belief_revision.py`'s
shared `save()`/`write_combined_html()` infra and constants rather than duplicating
them): accuracy-by-mod_type (3 lines — S1 shared, S2 nothink, S2 think), transition
stacked bars per condition, turns-used-by-mod_type per condition, judge
category/pde_class breakdowns plus the `contains_incorrect_claims` rate, and the
recall/specificity panel (bar charts plus plain-text tables, aggregate and per
mod_type) — the primary lens onto the experiment's central question.

## Dataset & pilot scale

- Stratification: 8 mod_types (`Comm_Valid`/`InValid`, `NoComm_Valid`/`InValid`,
  `CorrComm`/`_Invalid`, `NoComm_CorrVar`/`_InValid`) × 4 PDE classes (Burgers, Heat,
  NavierStokes, Wave).
- A 64-row stratified pilot (8 rows per mod_type) has been completed across both
  thinking conditions plus a judge pass. The full 256-row dataset (32 rows per
  mod_type) is the current scale-up, run incrementally by resuming the 64-row pilot's
  checkpoints against the full dataset rather than rerunning from scratch.
- Observed real per-row costs at pilot scale (`gemini-2.5-flash`): nothink ≈
  $0.020/row, think ≈ $0.037/row, judge (eligible rows only, Pro-tier) ≈ $0.035/row —
  useful for planning further scale-ups.

## Changelog (since v3 was approved)

- **Disk-safety guards** added after two real ENOSPC crashes on the same row
  (`Wave_NoComm_CorrVar_InValid_2`, a model dt-reduction bug producing a 4.83GB
  `.npz`) — per-file write cap, cumulative directory cap, quarantine handling,
  snapshot-copy dedup.
- **Cost-spiral hardening**: unbounded `max_output_tokens` risked silently truncated
  function-call arguments (thinking tokens share the visible-output budget) — added a
  50k cap to Stage 1, every Stage-2 turn, and the judge; the judge also gained its own
  session-wide cost cap.
- **Submission-confirmation mechanism** added after observing the model could judge
  its own edited code and report that judgment as the original snippet's answer.
- **`ANY`/`VALIDATED` function-calling mode management** introduced to give the model
  room to reason before a forced call, with a bounded one-shot escalation path for
  text-only or empty non-responses.
- **Thinking budget as a swept condition** (0 vs. 1536) added, extending the original
  single Stage-2 pass into two independently-scored conditions sharing one cached
  Stage-1 pass.
- **LLM-judge module** added as an entirely new component, assessing invalidity-
  reasoning *quality*, not just valid/invalid accuracy.
- **Ground-truth-reference fix** added to the judge after finding it scored
  suspiciously many `NoComm_CorrVar_InValid` rows as `category=none` — traced to the
  judge having no way to see past obfuscated, absent, or misleading code text.
- **Subprocess `UnicodeDecodeError` fix** (`errors="replace"`) added after a stray
  non-UTF8 byte in a child process's stdout crashed an entire 256-row sweep mid-run.
- **Orchestration script** (`run_stratified_sweep.py`) added to formalize the
  shared-Stage-1-cache-plus-per-condition-checkpoint pattern, enabling safe
  incremental scale-ups.

## Known issues, limitations, and outstanding items

- `gemini-2.5-flash`'s deprecation date (October 16, 2026) — re-check availability
  before any run past that date.
- The submission-confirmation mechanism only catches the direct case (the model
  submitting a judgment of its own edited code as if it were the original). It does
  not catch a subtler version observed in the pilot (`Burgers_CorrComm_Valid_1`, think
  condition), where the model used its own edited "corrected" version as an implicit
  comparison point to argue the original is deficient by contrast, without ever
  literally confusing which code was which.
- The `SAVES_HISTORY`/variable-naming soft-hint limitation (see Scoring) remains
  unresolved — can't be engineered away without discarding real data.
- The judge module has not yet been re-run at 256-row scale, pending explicit
  go-ahead given real API cost.
- Single-pass judge (no majority-vote redundancy) — noisier than a voting design,
  only mitigated by manual spot-checking a handful of verdicts.
- The 256-row core sweep (Stage-2 nothink + think) is in progress as of this writing.
