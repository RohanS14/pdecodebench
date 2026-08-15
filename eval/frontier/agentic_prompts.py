"""
agentic_prompts.py — tool schemas and Stage-1/Stage-2 prompt text for the
agentic belief-revision episode loop.

Kept separate from run_belief_revision_agentic.py's orchestration logic so the
tool contracts and prompt copy can be reviewed or tuned independently of the
loop mechanics. Schemas are plain dicts shaped as
types.FunctionDeclaration(**decl) keyword arguments -- google-genai isn't
imported here, only in the test that validates the shape against the real API,
and in the orchestration script that actually builds types.Tool objects.

Also carries PROMPT_S1_AGENTIC, an agentic-only Stage-1 prompt that adds
explanation fields on top of the shared, untouched run_belief_revision.py's
PROMPT_S1 -- kept here (not in that shared file) so the static experiment's
Stage-1 prompt and parsing stay byte-identical to what's already been run.
"""
import re


def _flow(text: str) -> str:
    """Collapse line breaks used purely to keep the source readable into
    single spaces, so the model receives natural prose instead of arbitrary
    mid-sentence breaks. Blank lines (real paragraph breaks) are preserved.
    Only applied to continuous-prose strings -- NOT to PROMPT_S1_AGENTIC or
    build_validated_reminder_final, whose line breaks are structural (field
    lists, <tag> blocks) and must stay as real newlines.
    """
    paragraphs = text.strip("\n").split("\n\n")
    return "\n\n".join(" ".join(p.split()) for p in paragraphs)


EDIT_SOURCE_DECL = {
    "name": "edit_source",
    "description": _flow("""\
Collect new evidence by replacing the current source snippet with a new version.
Provide the complete new file contents in full_source, not a diff. The harness writes
exactly what you provide as a new versioned file (solver_v1.py, v2.py, ...) and reruns
the full simulation. The only useful ways to edit source are 1) writing to stdout or
stderr, which are added to the next turn's context or 2) saving numeric data (npz, csv, txt)
which can be read by a future run_diagnostic call. Images do not persist, so do not plot.
Nothing from the run's internal namespace is captured automatically: if you want data
available to a later run_diagnostic call without rerunning the simulation, your full_source
must explicitly save it to disk (e.g. np.savez('history.npz', u=u, t=t)) using a plain
relative path. Leaving full_source empty (or whitespace-only) is a valid no-op: the harness
reruns the current latest version completely unchanged - use this if you want to rerun
without changing anything yet, rather than retyping the file verbatim. Avoid long reasoning
written out inside comments in the code. Counts against the investigative action budget,
including a rewrite that fails to run.\
"""),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "full_source": {
                "type": "string",
                "description": (
                    "The complete new file contents. Leave empty (or "
                    "whitespace-only) to rerun the current latest version "
                    "unchanged. Provide the exact raw Python source, as it "
                    "would appear in a .py file -- do not wrap it in quotes, "
                    "triple quotes, or markdown code fences."
                ),
            },
        },
        "required": ["full_source"],
    },
}

RUN_DIAGNOSTIC_DECL = {
    "name": "run_diagnostic",
    "description": _flow("""\
Run further analysis on evidence you have already collected. Provide a standalone Python script;
the harness saves it to the episode's sandbox directory and runs it there with no arguments.
It can only read the current source file(s) directly (static analysis, no execution needed)
and any file a prior edit_source call explicitly saved to disk. You cannot view image files,
only read numeric data or see printed text. It never reruns or touches the simulation itself.
There is no automatic 'look around' step: listing files in the directory or listing what
arrays a saved .npz contains must be written explicitly, at the cost of this action. Do not
call before the first edit_source call as there is nothing saved yet to analyze beyond the
source text itself. Avoid long reasoning written out inside comments in the code. Counts against
the investigative action budget.\
"""),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "A standalone Python script to run in the sandbox "
                    "directory. Provide the exact raw Python source, as it "
                    "would appear in a .py file -- do not wrap it in quotes, "
                    "triple quotes, or markdown code fences."
                ),
            },
        },
        "required": ["script"],
    },
}

SUBMIT_FINAL_ANSWER_DECL = {
    "name": "submit_final_answer",
    "description": _flow("""\
End the episode by providing your final answer.
You must answer based on the ORIGINAL code snippet (not any version you may have edited)
and any evidence you have collected.
If using execution evidence to arrive at your answer, ensure specific, documented findings
are referenced. An error caused by a mistake in your own edit (e.g. a syntax error you
introduced while rewriting) is not evidence about the original snippet's validity.
An issue that also occurs in a faithful or logically-equivalent version of the original code
is real evidence about the original and can be cited. For example, if you found evidence
that the original code was physically invalid and then generated a fixed version that was
physically valid, the final answer should report invalid.
Always available, regardless of remaining budget, and never counts
against the investigative action budget.\
"""),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "pde": {"type": "string", "description": "The type of PDE being solved."},
            "method": {"type": "string", "description": "Numerical method(s) used -- list all that apply."},
            "behavior": {"type": "string", "description": "Dominant physical process(es) -- list all that apply."},
            "valid": {"type": "string", "description": "Does this code run and produce a correct physical solution for the PDE?"},
            "pde_exp": {"type": "string", "description": "1-2 sentence explanation for the pde answer, including whether/why it changed from your first-stage answer."},
            "method_exp": {"type": "string", "description": "1-2 sentence explanation for the method answer, including whether/why it changed from your first-stage answer."},
            "behavior_exp": {"type": "string", "description": "1-2 sentence explanation for the behavior answer, including whether/why it changed from your first-stage answer."},
            "valid_exp": {"type": "string", "description": "1-2 sentence explanation for the valid answer, including whether/why it changed from your first-stage answer."},
        },
        "required": ["pde", "method", "behavior", "valid", "pde_exp", "method_exp", "behavior_exp", "valid_exp"],
    },
}

ALL_TOOL_DECLS = {
    "edit_source": EDIT_SOURCE_DECL,
    "run_diagnostic": RUN_DIAGNOSTIC_DECL,
    "submit_final_answer": SUBMIT_FINAL_ANSWER_DECL,
}


# ── Agentic-only Stage 1 (adds explanations; shared PROMPT_S1/parse_response
# in run_belief_revision.py/parse_score.py are untouched) ────────────────────

PROMPT_S1_AGENTIC = """\
You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

Answer the following about this simulation. Be concise.

Output only:
pde: ____
method: ____
behavior: ____
valid: ____
pde_exp: ____
method_exp: ____
behavior_exp: ____
valid_exp: ____

- pde: the type of PDE being solved
- method: numerical method(s) used — list all that apply
- behavior: dominant physical process(es) — list all that apply
- valid: does this code run and produce a correct physical solution for the PDE?
- pde_exp/method_exp/behavior_exp/valid_exp: 1-2 sentence explanation for each answer above\
"""


def parse_s1_explanations(text: str) -> dict:
    """Extract the 4 explanation fields from an agentic Stage-1 response.
    Mirrors parse_score.parse_response()'s regex approach for the 4 core
    fields, but kept separate so the static experiment's Stage-1 parsing
    (parse_score.py) stays completely untouched."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    result = {}
    for field in ("pde_exp", "method_exp", "behavior_exp", "valid_exp"):
        matches = re.findall(rf"(?im)^[-*]?\s*{field}\s*:\s*(.+?)(?:\n|$)", text)
        result[field] = matches[-1].strip() if matches else None
    return result


def build_stage2_prompt(budget: int) -> str:
    """Stage-2 agentic prompt handed to the model right after its Stage-1
    answer. Substitutes the tool list and budget into the fixed sentence the
    design doc specifies."""
    tool_names = "edit_source, run_diagnostic, submit_final_answer"
    return _flow(f"""\
In the previous stage, you answered questions about the code snippet's PDE type, numerical
method, behavior, and physical validity just by reading. The PDE code snippet provided in
the previous stage runs without execution errors but could be logically or physically
invalid. You can therefore use execution to check and, if needed, correct your answers from
the previous stage. The goal of this stage is to find the correct answers about the original
code snippet's PDE type, numerical method, behavior, and physical validity. You can now use
the following tools to collect evidence before deciding whether to preserve or change answers
from the previous stage: {tool_names}, with an investigative budget of {budget} edit_source/run_diagnostic
actions total (submit_final_answer does not count against this budget). Once the budget is used up,
only submit_final_answer remains available. Use the submit_final_answer tool to provide a structured
final answer, do not give a text-only answer. Answers with both text and an action are preferred.
Think about what actions to take before taking an action such as writing code. Ensure any conclusions
based on execution evidence use specific findings. Text written to stdout and stderr and saved numeric data
(npz, txt, csv) persist between turns, but images cannot be accessed. Avoid long reasoning written out
inside comments in the code. Answers that do not include tool calls still count against the 
{budget}-turn budget, and a text-only answer will trigger a subsequent turn where you are forced to take an action.\
""")


# ── Per-turn reminders (VALIDATED-mode turns) and retry feedback ─────────────

def build_validated_reminder_investigative(actions_used: int, budget: int) -> str:
    """Per-turn reminder for investigative-phase VALIDATED turns. States the
    live budget count so the model knows how much runway is left without
    having to infer it from turn count alone."""
    return (
        f"You have used {actions_used} of {budget} actions so far. "
        "Explain what you want to check and why, then take an action using "
        "one of your available tools."
    )


def _original_code_and_stage1_recap(original_code: str, s1_text: str) -> str:
    """Shared original-code + Stage-1-answer block, reused by
    build_validated_reminder_final (forced/terminal path, unchanged) and
    build_submit_confirmation_reminder (voluntary path)."""
    return (
        "<original_code>\n"
        f"{original_code}\n"
        "</original_code>\n\n"
        "For reference, here is your own answer from Stage 1, before you had "
        "access to any execution evidence:\n\n"
        "<stage1_answer>\n"
        f"{s1_text}\n"
        "</stage1_answer>\n\n"
    )


def build_validated_reminder_final(original_code: str, s1_text: str) -> str:
    """Per-turn reminder for the terminal (budget-exhausted) VALIDATED turn.
    Re-shows the original snippet's exact text and the model's own Stage-1
    answer: by this point both may be buried under many turns of tool calls
    in the conversation history, and the final answer must be about THIS
    code (not whatever the model has since rewritten it to), reasoned from
    all evidence collected -- not just restated from Stage 1."""
    return (
        "Your investigative budget is exhausted. You must submit_final_answer "
        "now based on the ORIGINAL code snippet below (not any version you "
        "may have edited) and any evidence you have collected.\n\n"
        + _original_code_and_stage1_recap(original_code, s1_text) +
        "Provide your final answer based on all the evidence you have "
        "collected. This may confirm or revise your Stage 1 answer, "
        "whichever the evidence actually supports."
    )


def build_submit_confirmation_reminder(original_code: str, s1_text: str) -> str:
    """Shown once, as the function_response to a VOLUNTARY (budget not yet
    exhausted) submit_final_answer call -- intercepts the first such call
    and requires a second to actually finalize. The forced/terminal path
    does NOT use this: it already gets build_validated_reminder_final
    pre-call, before its one deterministic attempt, which is strictly
    better than reactive correction when the timing is known in advance.
    Reuses the same recap content as build_validated_reminder_final, minus
    the 'investigative budget is exhausted' framing, which doesn't apply
    to an early submission."""
    return (
        "Before finalizing: you must answer based on the ORIGINAL code "
        "snippet below (not any version you may have edited) and any "
        "evidence you have collected.\n\n"
        + _original_code_and_stage1_recap(original_code, s1_text) +
        "Call submit_final_answer again with your final answer. This may "
        "confirm or revise your Stage 1 answer, whichever the evidence "
        "actually supports."
    )


TEXT_ONLY_FEEDBACK = (
    "Your previous response didn't include a tool call -- you must take an "
    "action now."
)

EMPTY_RESPONSE_FEEDBACK = (
    "No text or action was received and one turn from the budget was used. "
    "Try again."
)
