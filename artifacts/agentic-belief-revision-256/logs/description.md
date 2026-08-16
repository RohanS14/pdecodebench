# Sample logs — agentic belief-revision, 256-row scale-up

Each log is a full turn-by-turn transcript of one episode: the model's Stage-1
(read-only) answer, then every Stage-2 tool call (edit_source / run_diagnostic /
submit_final_answer) with its actual arguments and real stdout/stderr, ending
in the final submitted answer and how it scored against ground truth. See
`../design/2026-08-16-agentic-belief-revision-design-v4.md` for the full
mechanics, and `../docs/hypotheses_results.md` for the aggregate findings these
examples illustrate.

## Highlighted examples

**`success_Burgers_NoComm_CorrVar_1__{nothink,think}.txt`** — the code is a
valid, inviscid Burgers' equation solver with obfuscated variable names. The
Stage-1 read misidentifies both the governing equation (called it "the
advection equation") and, in the no-extended-thinking run, the validity call.
The extended-thinking run catches its own inconsistency mid-episode — it
re-examines the flux formula it had extracted, recognizes the quadratic term
as the nonlinearity that distinguishes Burgers' equation from linear
advection, revises the equation-class answer, and correctly reinterprets the
observed norm decay as expected shock dissipation rather than a bug.

**`failure_NavierStokes_CorrComm_Valid_1__{nothink,think}.txt`** — the code is
valid; only its comments were swapped in from a different PDE class's code
(the code itself is untouched). Both runs perform similar standard checks
(numerical stability, boundary behavior). The extended-thinking run digs
further, surfaces a real boundary-condition detail (a corner-cell overwrite
in how edges are applied), and treats it as disqualifying — flipping a
correct "valid" into an incorrect "invalid" based on a real but pre-existing
property of the original (already-valid) code, not an injected defect.

## `false_alarms_valid_called_invalid/`

For domain-expert (PDE) review: 8 snippets, randomly sampled 2 per PDE class
(Burgers, Heat, Wave, Navier-Stokes) from the full set of rows where the
model incorrectly called genuinely valid code "invalid" (70 unique such
snippets exist across both conditions in the full 256-row run; this is a
random subsample, not the complete set — ask if you want the rest). Each
snippet gets its own subfolder with both the no-extended-thinking (`nothink.txt`)
and extended-thinking (`think.txt`) transcripts for the same code, so a
reviewer can compare what each condition actually found/argued.

Sampled snippets:
- Burgers: `Burgers_NoComm_Valid_3`, `Burgers_Comm_Valid_4`
- Heat: `Heat_Comm_Valid_1`, `Heat_NoComm_Valid_1`
- Wave: `Wave_CorrComm_Valid_6`, `Wave_CorrComm_Valid_5`
- Navier-Stokes: `NavierStokes_CorrComm_Valid_1`, `NavierStokes_Comm_Valid_7`

(`NavierStokes_CorrComm_Valid_1` is the same snippet as the standalone
"failure" example above — it was re-selected by the random draw from the
same false-alarm population, not a duplicate placed deliberately.)

**What to look for**: in each case the model's final answer was "invalid" for
code a human reviewer already confirmed correct. The question for review is
*why* — does the model's stated `valid_exp` objection point at something
that's actually wrong (in which case the dataset's own valid/invalid labeling
may be worth a second look), or is it a spurious/misapplied concern (confirming
the model's own false-alarm tendency, not a real code issue)? See
`hypotheses_results.md`'s H1a/H1c sections for why this asymmetry (missing bugs
is largely fixed by investigation, false alarms are not) is the central finding
this sample is meant to help sanity-check.
