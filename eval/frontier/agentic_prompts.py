"""
agentic_prompts.py — tool schemas and Stage-2 prompt text for the agentic
belief-revision episode loop.

Kept separate from run_belief_revision_agentic.py's orchestration logic so the
tool contracts and prompt copy can be reviewed or tuned independently of the
loop mechanics. Schemas are plain dicts shaped as
types.FunctionDeclaration(**decl) keyword arguments -- google-genai isn't
imported here, only in the test that validates the shape against the real API,
and in the orchestration script that actually builds types.Tool objects.
"""

EDIT_SOURCE_DECL = {
    "name": "edit_source",
    "description": (
        "Collect new evidence by editing the current source snippet. Provide a "
        "unified diff against the latest version; the harness applies it to a "
        "new versioned file (solver_v1.py, v2.py, ...) and reruns the full "
        "simulation. An empty diff is a valid no-op rerun. File paths inside the "
        "diff's --- / +++ headers are ignored by the harness (it always patches "
        "the current latest version directly) -- they don't need to match any "
        "exact filename. Nothing from the run's internal namespace is captured "
        "automatically: if you want data available to a later run_diagnostic "
        "call without rerunning the simulation, your diff must explicitly save "
        "it to disk (e.g. np.savez('history.npz', u=u, t=t)) using a plain "
        "relative path. Counts against the investigative action budget, "
        "including a diff that fails to apply."
    ),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "diff": {
                "type": "string",
                "description": "A unified diff to apply to the latest source version.",
            },
        },
        "required": ["diff"],
    },
}

RUN_DIAGNOSTIC_DECL = {
    "name": "run_diagnostic",
    "description": (
        "Run further analysis on evidence you have already collected. Provide "
        "a standalone Python script; the harness saves it to the episode's "
        "sandbox directory and runs it there with no arguments. It can read "
        "the current source file(s) directly (static analysis, no execution "
        "needed) and any file a prior edit_source call explicitly saved to "
        "disk -- nothing else. It never reruns or touches the simulation "
        "itself. There is no automatic 'look around' step: even listing "
        "files in the directory or listing what arrays a saved .npz contains "
        "must be written explicitly, at the cost of this action, same as "
        "everything else. Calling this before any edit_source call is "
        "technically allowed but pointless -- there is nothing saved yet to "
        "analyze beyond the source text itself. Counts against the "
        "investigative action budget."
    ),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "A standalone Python script to run in the sandbox directory.",
            },
        },
        "required": ["script"],
    },
}

SUBMIT_FINAL_ANSWER_DECL = {
    "name": "submit_final_answer",
    "description": (
        "End the episode by providing your final answer. Always available, "
        "regardless of remaining budget, and never counts against the "
        "investigative action budget -- this ends the episode, it isn't an "
        "investigative act."
    ),
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


def build_stage2_prompt(budget: int) -> str:
    """Stage-2 agentic prompt handed to the model right after its Stage-1
    answer. Substitutes the tool list and budget into the fixed sentence the
    design doc specifies."""
    tool_names = "edit_source, run_diagnostic, submit_final_answer"
    return (
        "The code runs without execution errors but could be logically or "
        "physically invalid. You can now use the following tools to collect "
        f"evidence before deciding whether to preserve or change answers from "
        f"the previous stage: {tool_names}, with an investigative budget of "
        f"{budget} edit_source/run_diagnostic actions total (submit_final_answer "
        "does not count against this budget). Once the budget is used up, only "
        "submit_final_answer remains available."
    )
