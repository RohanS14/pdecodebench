# Agentic Belief-Revision Experiment Design v5 (supersedes v4, reviewed and approved by Rohan)

> **Scope note.** v5 is a NEW experiment, not a continuation of the v4 run. It adds a
> graded 0-10 validity score to the answer schema and sweeps multiple models
> (`gemini-2.5-flash` -> Gemini 3.x). Because the score changes the prompt, it changes
> the binary answer too — so the completed v4 256-row binary-only data is preserved
> untouched as the **control arm** and is never pooled with v5 results. Artifacts live
> under `artifacts/agentic-belief-revision-256-score/`, separate from
> `artifacts/agentic-belief-revision-256/`.

## Context

The belief-revision experiment tests whether a frontier model's lexical priors about
a PDE solver snippet (its judgment of `pde` / `method` / `behavior` / `valid`) can be
overcome by execution evidence, and — where the snippet is genuinely invalid — whether
the model can localize the actual defect rather than just flag its presence.

Stage 1 is a single read-only turn: the model sees the snippet and answers all four
fields (plus the v5 validity score) with no tool access. Stage 2 is agentic: the model is given three tools
(`edit_source`, `run_diagnostic`, `submit_final_answer`) and a fixed investigative
action budget, and gathers whatever evidence it chooses before answering again. The
harness never pre-selects "the important variable" or hands the model a precomputed
trajectory summary — every fact the model sees in Stage 2, beyond the bare source
text, is something its own code explicitly computed and printed.

Stage 2 runs as **two independently-scored thinking conditions** per row — a
minimal-thinking condition ("nothink") and an extended-thinking condition ("think") —
sharing one cached, deterministic Stage-1 answer. A separate LLM-judge module
additionally scores, for rows the model correctly called invalid, whether its
justification identifies the real injected defect or just the right label. Together
these turn a single S1-vs-S2 accuracy comparison into a richer three-condition,
mechanism-aware picture of how much and what kind of evidence overcomes a syntactic
prior.

v5 adds two things on top of that structure:

- a **graded 0-10 validity score** alongside the existing binary `valid` field, at both
  stages, so the analysis is no longer hostage to the model's own binary threshold (see
  "Validity score"); and
- a **cross-model sweep** (`gemini-2.5-flash` -> Gemini 3.x), run one model at a time,
  to test whether the v4 findings are properties of the task or of one model (see
  "Model comparison").

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
- Separate **discrimination** from **calibration**: measure whether the model can rank
  physically-faithful code above defective code at all, independently of where it
  chooses to put its binary valid/invalid cutoff.
- Add the score without leaking task direction: the scale must be defined in terms of
  degrees of physical fidelity, never in terms of defect types or diagnostics to run.

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
- **Model**: one model per sweep, held fixed across Stage 1 and both Stage-2
  conditions within that sweep (see "Model comparison" for the set and the
  per-family branching this requires).

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
edit_source(full_source: str)
  - Purpose: collect new evidence. Available every turn; counts against budget.
  - Model outputs the COMPLETE new file contents — not a diff. The harness writes
    exactly what it was given to a NEW versioned file (solver_v1.py, v2.py, ...),
    never overwriting in place, and reruns the FULL simulation via the sandbox
    (subprocess, eval/.venv, timeout). An empty/whitespace-only full_source is a
    valid no-op — it reruns the current latest version unchanged, not an error (a
    new version file is still created, preserving the never-overwrite convention).
  - Returns, in the same turn: which file the new version was saved as, plus the
    actual stdout/stderr from that run (including print()s and assertion
    failures) — this is deliberate, matching real "add a print, rerun, see the
    output" debugging.
  - The harness does NOT automatically capture, snapshot, or persist anything from
    the execution's namespace. Anything the model wants available to a later
    run_diagnostic call without re-running the simulation must be explicitly written
    to disk by its own source (e.g. np.savez(...)) at a plain relative path (cwd is
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

submit_final_answer(pde, method, behavior, valid, validity_score, pde_exp,
                    method_exp, behavior_exp, valid_exp)
  - Purpose: end the episode. Available every turn regardless of remaining budget —
    the model can stop voluntarily, or be forced into it once edit_source/
    run_diagnostic have dropped from the tool list.
  - Does NOT count against the budget — the budget measures investigation, not the
    act of ending an episode.
  - Arguments mirror the 4-field answer format used everywhere else in this
    pipeline, plus the v5 integer `validity_score` (see "Validity score") and 4
    explanation fields (pde_exp/method_exp/behavior_exp/valid_exp), a 1-2 sentence
    rationale for the answer and for whether/why it changed from the Stage-1
    answer. `valid_exp` is reused to justify BOTH the binary call and the score —
    no separate score-explanation field, to avoid schema bloat.
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

### Validity score

A graded `validity_score` is collected **alongside**, never instead of, the binary
`valid` field, at **both** Stage 1 and Stage 2 (mirroring how `valid` is already
tracked at both stages, so investigation's effect on the score is visible even when it
doesn't flip the binary call).

- **Scale**: integer 0-10, oriented as *physical fidelity*, not severity of
  invalidity. **0 = the trajectory retains none of the intended system's physics;
  10 = it reproduces the intended system's physics exactly.** The two orientations
  are not interchangeable and the sign must not be flipped anywhere downstream.
- **Why fidelity, not severity**: "how severe is the invalidity" presupposes there is
  invalidity to grade, and a genuinely valid row has no natural answer to it. Fidelity
  is a neutral question every row can answer, which matters because the valid
  conditions are exactly where the model over-flags.
- **Why any score at all**: `valid` in this dataset means "passed human review, no
  defect deliberately injected" — a provenance fact, not a threshold on a latent
  quality continuum. Valid rows can still carry real, minor pre-existing quirks (see
  the `NavierStokes_CorrComm_Valid_1` / `NavierStokes_NoComm_CorrVar_1` trace
  deep-dives), and a binary call gives the model nowhere to put "found a real but
  minor wrinkle, still basically fine." The score is what lets the analysis distinguish
  that from "confidently, seriously broken."
- **Anchoring: endpoints only.** The two endpoints above are stated; no interior tiers,
  no example defect types, no suggested diagnostics. This is a direct consequence of
  the v4 H1 finding that *lexical availability drives false alarms*: a rubric naming
  defect categories ("a small boundary-condition error", "check conservation") would
  inject that same vocabulary uniformly across all eight conditions, partially
  collapsing the very contrast the dataset manipulates. The scale is framed as one
  continuous comparison — how closely the produced trajectory matches the intended
  system's — rather than a checklist to sum over.
- **No forced consistency** between `valid` and `validity_score`. A `valid=No` paired
  with a high fidelity score is not rejected or repaired: it is one of the experiment's
  most informative observations (the model explicitly flagging code it also considers
  basically fine), and is reported as a first-class rate.
- **Known risk, measured not assumed**: with endpoints only, the score may come back
  bimodal or clustered on round numbers, which would compress the rank-based metrics
  via ties. Distribution shape is therefore checked on the smoke test and first partial
  batch as a gate BEFORE any score-based claim is trusted, with adding interior
  fidelity anchors held in reserve as the defined fallback.
- **Elicitation is an intervention, not instrumentation**: adding the field plausibly
  changes the binary answer (the model now has somewhere to put minor concerns). That
  is why v4's binary-only run is retained as a control arm — the arm contrast is itself
  a result, not a confound to be explained away.

### Thinking-budget conditions

Stage 2 runs as two independent sweeps per row: a **minimal-thinking** condition
("nothink") and an **extended-thinking** condition ("think"). Stage 1 is unaffected by
and shared across both — Stage 1 always runs at the minimal-thinking setting
regardless of which Stage-2 condition it's paired with, and is fully deterministic
(temperature=0), so it's computed once per row and reused, never re-run per condition.

The concrete parameter differs by model family (see "Model comparison"): Gemini 2.5
uses a numeric `thinking_budget` (0 vs. 1536, as in v4); Gemini 3.x uses the
`thinking_level` enum instead. The nothink/think two-condition design is preserved
across families, but the mapping onto `thinking_level` values must be confirmed against
current docs, not inferred — and the condition-directory naming can no longer be keyed
off `thinking_budget == 0`.

### Model comparison

v5 sweeps more than one model in order to test whether v4's findings (the
recall/specificity asymmetry, investigation hurting obfuscated-valid rows, thinking not
improving reasoning quality) are properties of the task or of `gemini-2.5-flash`
specifically.

- **Models**: `gemini-2.5-flash` (re-run under the v5 schema, so it is comparable to
  the new models rather than only to its own v4 control arm) plus Gemini 3.x flash-tier.
  Exact model ID strings and quota are confirmed live (e.g. `client.models.list()`) or
  from current docs before any paid run — never guessed from training-data knowledge.
- **One model per sweep**, run sequentially. No single combined run.
- **Held fixed across models**: the dataset, the prompts, the answer schema, and the
  **investigative action budget (6)** — the budget is the experimental variable whose
  comparability matters most, so it does not scale with model economics.
- **Scaled per model**: the **episode cost cap** (v4's flat $0.50) is derived from each
  model's real per-token pricing rather than left as a fixed dollar figure, so a pricier
  model doesn't have a legitimately-necessary turn choked off. This requires replacing
  the hardcoded `gemini-2.5-flash` pricing constants with a per-model pricing lookup —
  those constants currently feed every cost guard and budget cap in the harness, so
  running another model against them would silently misreport and mis-enforce cost.
- **Family branching required**: the thinking-control parameter shape differs (numeric
  `thinking_budget` on 2.5 vs. the `thinking_level` enum on 3.x; `thinking_budget` is
  retained on 3.x only for backwards compatibility and is documented as able to cause
  unexpected performance), so both Stage 1's call and Stage 2's per-turn config
  construction branch on model family.
- **Thought signatures**: Gemini 3 hard-errors on a missing `thought_signature` during
  function calling, so every branch of the Stage-2 loop must replay the model's actual
  response content before appending harness feedback — including the empty-response and
  text-only escalation branches, which in v4 appended only the feedback message and
  silently dropped that turn's signature. Confirmed live that a text-only Gemini 3
  response with no function call still carries a real signature.
- **Comparability discipline**: a "7" from one model is not the same quantity as a "7"
  from another. Cross-model score comparisons use rank-based measures only (AUC); raw
  score means are never compared across models.

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
- A per-episode `EpisodeCostGuard` tracks cumulative Stage-2 API cost; tripping it
  collapses the tool list identically to budget exhaustion (same `tools_available()`
  check covers both conditions). Its cap is **derived per model** from that model's real
  pricing (v4's flat $0.50 was tuned to `gemini-2.5-flash` economics), so it stays
  equivalently permissive rather than equivalently priced. This is independent of the
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

**v5 checkpoint-compatibility warning**: v4 checkpoints are NOT resumable into a v5
sweep. The score changes `PROMPT_S1_AGENTIC`, which invalidates the entire shared
Stage-1 cache, and the arm/model separation means every v5 sweep starts from an empty
output root. Budget for a full duplicate sweep per model (Stage 1 + both Stage-2
conditions), not an increment on v4. Output roots are keyed by model so the arms and
models can never collide or be accidentally pooled.

### Scoring

- Stage 1 is scored via the existing `parse_response()` + `score_row()` pipeline,
  unchanged.
- Stage 2's `submit_final_answer` arguments arrive already structured — `parse_response()`
  is bypassed entirely, and the tool call's argument dict is fed directly into the same
  unmodified `score_row()`.
- The `validity_score` argument is carried through unmodified and persisted as
  `s1_validity_score` / `s2_validity_score`. It is NOT fed to `score_row()` and has no
  match column — there is no ground-truth score to match against; ground truth is the
  binary `gt_valid` provenance fact, and the score's job is to be *ranked* against it,
  not matched. A missing or out-of-range score is persisted as null rather than
  coerced, so downstream analysis can count non-compliance instead of silently
  absorbing it.
- Per-row output includes: field matches (`valid`/`pde`/`method_any`/`behavior_any`),
  hedge classification (`s1_hedge_class`/`s2_hedge_class` via `classify_hedge()`),
  `transition` (S1→S2 answer-class transition via `get_transition()`),
  `thinking_budget`, `aborted`/`abort_reason` (normal `s2_*` fields are omitted
  entirely when aborted, so downstream analysis can't mistake it for a real answer),
  full action bookkeeping (`s2_action_count`, `tools_used`, `used_edit_source`,
  `actions_remaining_at_submission`, `cost_guard_tripped`, `protocol_violations`, the
  complete turn-by-turn `action_trace`), and raw token counts alongside derived cost
  (so cost can always be recomputed later even if pricing constants change).
- **Binary comparison structure** (inherited from v4): recall (fraction of
  truly-invalid rows correctly called invalid) and specificity (fraction of truly-valid
  rows correctly called valid), computed per mod_type stratum — each mod_type is
  homogeneous in `gt_valid` by construction, so the match column computed within one
  mod_type directly is the recall/specificity indicator — across all three conditions
  (S1, S2 nothink, S2 think). Richer than one pooled accuracy number.
- **Score comparison structure** (new in v5, and the primary lens on the score):
  rank-based and threshold-free throughout. Raw score means are not the headline
  statistic — the score and the binary call come from the same model on the same
  evidence and correlate near-1, so means by condition largely re-derive
  recall/specificity in different units. Instead:
  - **AUC** of `validity_score` against `gt_valid` (ties credited 0.5), equal to a
    normalized Mann-Whitney U: the probability that a randomly chosen valid row
    outscores a randomly chosen invalid row. Computed per condition (S1, S2 nothink,
    S2 think) and per mod_type pairing.
  - **Paired-by-`gt_sample` ranking** as the primary statistic: valid and invalid rows
    come in matched pairs differing by one injected defect, so ask, per pair, whether
    the clean member outscored its own defective counterpart. This removes
    between-sample difficulty variance and is far better powered at n=256 than the
    unpaired version.
  - **Discrimination vs. calibration**: accuracy at the score's best threshold, minus
    the accuracy of the model's own binary call, quantifies in accuracy points how much
    is lost to threshold miscalibration alone.
  - **Score/binary inconsistency rate** (`valid=No` with a high score, and vice versa)
    reported as a first-class number.
  - Full score distributions and ROC curves are reported alongside every scalar AUC —
    a scalar hides the bimodality the distribution gate is checking for.
  - Cross-condition AUC comparisons use paired tests on the same rows (DeLong or a
    bootstrap resampling `gt_sample` pairs), not independent-sample tests.
- First-pass bug-localization metric: since a `gt_sample`'s valid/invalid pair differs
  by a small, near-mechanically-diffable change, check whether any `edit_source`
  rewrite the model applied touches the true differing line(s) (compared against the
  previous version, since the model submits full source rather than a diff) — no LLM
  judge needed for this.
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
- **Score-groundedness cross-check** (new in v5): a high AUC proves the score *ranks*
  correctly, not that it is *grounded* in anything the model actually found. The
  judge's `category`/`contains_incorrect_claims` verdicts on the same `valid_exp` text
  are the external check — false alarms the judge scores `none` (no real defect
  identified) should carry high fidelity scores if the score is honest rather than
  confabulated post-hoc to agree with the binary call.
- **Model and cost**: a separate, stronger judge model (Pro-tier, currently resolved
  via the `gemini-pro-latest` alias) than the flash-tier model being evaluated, to
  reduce self-preference bias (accepted caveat: same-family models can still share
  blind spots) — with its own pricing constants, its own session-wide cost cap
  (`--max_cost_usd`), and per-row cost persisted.
- **Checkpoint**: keyed by `(title, thinking condition)` pair, so a single
  `judge_results.jsonl` accumulates verdicts across multiple `--stage2_jsonl` inputs
  (nothink and think) without reprocessing either.
- Single-pass judge (no majority-vote redundancy) for now — accepted as noisier than a
  voting design.

### Visualization

`viz/visualize_agentic.py` reads both condition JSONLs plus the judge output and
produces one combined, self-contained HTML report (reusing `viz/visualize_frontier.py`'s
shared `save()`/`write_combined_html()` infra and constants rather than duplicating
them): accuracy-by-mod_type (3 lines — S1 shared, S2 nothink, S2 think), transition
stacked bars per condition, turns-used-by-mod_type per condition, judge
category/pde_class breakdowns plus the `contains_incorrect_claims` rate, and the
recall/specificity panel (bar charts plus plain-text tables, aggregate and per
mod_type) — the primary lens onto the experiment's central binary question.

v5 adds a score panel: score distribution histograms per condition (the visual form of
the distribution gate), ROC curves with AUC per condition, per-mod_type AUC bars
alongside the existing per-mod_type accuracy bars (so a threshold effect and a
discrimination effect are visually distinguishable), the paired-by-`gt_sample` ranking
breakdown (clean-higher / tied / inverted), and the score/binary inconsistency rate.
Cross-model plots show AUC only, never raw score means.

## Dataset & pilot scale

- Stratification: 8 mod_types (`Comm_Valid`/`InValid`, `NoComm_Valid`/`InValid`,
  `CorrComm`/`_Invalid`, `NoComm_CorrVar`/`_InValid`) × 4 PDE classes (Burgers, Heat,
  NavierStokes, Wave).
- The same 256-row dataset (32 rows per mod_type) as v4, unchanged — reusing it is what
  makes the v4 binary-only control arm a valid comparison, and what makes the
  paired-by-`gt_sample` analysis possible.
- **Arms**: (a) v4 binary-only, `gemini-2.5-flash`, already complete — control, never
  re-run; (b) v5 score-schema, `gemini-2.5-flash` — the arm contrast against (a)
  isolates the elicitation effect with model held fixed; (c) v5 score-schema, Gemini
  3.x — the model contrast against (b) with schema held fixed. Each arm is a full
  independent sweep from an empty output root.
- Observed real per-row costs at v4 scale (`gemini-2.5-flash`, binary-only): nothink ≈
  $0.020/row, think ≈ $0.037/row, judge (eligible rows only, Pro-tier) ≈ $0.035/row —
  the v5 baseline for planning, with the caveat that they do not transfer to a Gemini
  3.x sweep, whose real cost is unknown until measured.
- **Smoke-test-first sequencing** (from `agent_docs/TODO_part2.md`, not optional): the
  score schema is smoke-tested end-to-end on a single cheap row against the
  already-understood `gemini-2.5-flash` BEFORE the multi-model branching lands, and the
  multi-model branching is smoke-tested against the actual new model before any real
  sweep — the only way to confirm the `thinking_level` mapping rather than assume it.

## Changelog (since v4 was approved)

- **Graded 0-10 `validity_score`** added to `submit_final_answer` and to the Stage-1
  answer format, oriented as physical fidelity (0 = no physics preserved, 10 = exact),
  endpoints-only anchoring, `valid_exp` reused as its justification. Binary `valid` is
  retained unchanged alongside it.
- **v4 data reclassified as a control arm** rather than a baseline to extend: the score
  changes the prompt and therefore the binary answer, so the arms are compared as arms
  and never pooled. v4 checkpoints are explicitly not resumable into v5.
- **Rank-based score analysis** specified up front (AUC, paired-by-`gt_sample` ranking,
  discrimination-vs-calibration decomposition, inconsistency rate) with a
  distribution-shape gate that must pass before any score claim is trusted. Raw score
  means are deliberately demoted.
- **Multi-model support**: per-model pricing lookup replacing the hardcoded
  `gemini-2.5-flash` constants, model-family branching for the thinking-control
  parameter (`thinking_budget` vs. `thinking_level`), per-model-derived episode cost cap
  with the action budget held fixed at 6, and live confirmation of model IDs/quota
  before any paid run.
- **Thought-signature replay fix** on the empty-response and text-only escalation
  branches of the Stage-2 loop — v4 appended only the harness feedback message there,
  dropping that turn's `thought_signature`, which Gemini 3 hard-errors on during
  function calling.
- **`edit_source` documented as full-source replacement**, correcting v4's stale "unified
  diff" prose — the implemented tool has always taken complete file contents
  (`full_source`), never a diff.

## Changelog (v4, since v3 was approved)

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
  before any run past that date. This now threatens the arm-(b) comparison specifically:
  if the model becomes unavailable, the elicitation effect can no longer be measured
  with model held fixed.
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
- The v4 256-row core sweep (Stage-2 nothink + think) and its judge pass are complete;
  results are written up in `artifacts/agentic-belief-revision-256/docs/`.
- **The score's scale usage is unvalidated.** Endpoints-only anchoring may produce a
  bimodal or round-number-clustered distribution, which compresses AUC via ties. The
  distribution gate detects this but does not fix it; adding interior fidelity anchors
  is the reserved fallback, and it carries its own (smaller) vocabulary-leak risk.
- **The score cannot be validated against a ground-truth score** — none exists. Its
  only external checks are ordinal (AUC against the binary provenance label) and
  indirect (the judge's verdict on the accompanying explanation text).
- **Per-mod_type AUC at 32 vs. 32 is underpowered** for anything but large gaps, the
  same caveat that applied to the v4 pilot's n=8 cells. Directions are defensible at
  that resolution; magnitudes are not.
- **No v5 sweep has been run.** The `thinking_level` mapping for the nothink/think
  conditions is unconfirmed, per-model pricing is not yet implemented, and nothing here
  has been smoke-tested. No paid run proceeds until the implementation plan is reviewed
  and each change is smoke-tested in isolation.
