"""
smoke_test_live_api.py — minimal, cheap, REAL-API check of the exact manual
function-calling mechanics the agentic Stage 2 design depends on:
  1. FunctionDeclaration(parameters_json_schema=...) is accepted by this SDK version.
  2. AutomaticFunctionCallingConfig(disable=True) actually disables SDK-side execution
     -- response.function_calls is populated, nothing is auto-run.
  3. A types.Content(role="tool", parts=[Part.from_function_response(...)]) turn is
     accepted as valid history and the model continues the conversation from it.
  4. The tool list can change turn-to-turn within the same growing `contents` list
     (mirrors dropping edit_source/run_diagnostic once budget is exhausted).

Requires GOOGLE_API_KEY (pulled from KeyHandler if available). Costs a few cents at
most (2 tiny calls, no thinking). Not part of the automated test suite -- run
manually, once, before building the rest of the agentic harness on top of these
assumptions, and again later if the google-genai SDK version changes.

Usage:
  .venv/bin/python frontier/smoke_test_live_api.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent  # mlproj -> private_projects -> raca-torch
sys.path.insert(0, str(WORKSPACE_ROOT / "packages" / "key_handler" / "key_handler"))
try:
    from key_handler import KeyHandler
    KeyHandler.set_env_key()
except ImportError:
    pass

from google import genai
from google.genai import types

PING_DECL = {
    "name": "ping",
    "description": "Call this with a short greeting string to say hello.",
    "parameters_json_schema": {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
    },
}

DONE_DECL = {
    "name": "done",
    "description": "Call this to end the conversation.",
    "parameters_json_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}


def main() -> int:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not set.")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    # ── Call 1: two tools available, expect a `ping` call, not auto-executed ──
    contents = [
        types.Content(role="user", parts=[types.Part(text=
            "Call the ping tool with greeting='hello world'. Do not respond with plain text."
        )]),
    ]
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(**PING_DECL),
            types.FunctionDeclaration(**DONE_DECL),
        ])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    calls = resp.function_calls or []
    assert calls, f"Expected a function call, got none. resp.text={resp.text!r}"
    call = calls[0]
    print(f"[1/3] OK: model called {call.name}({dict(call.args)}) -- AFC did not auto-execute")

    # ── Call 2: append the call + a tool-role response, narrow the tool list to
    #    just `done` (mirrors budget exhaustion), expect a coherent `done` call ──
    contents.append(types.Content(role="model", parts=[types.Part(function_call=call)]))
    contents.append(types.Content(
        role="tool",
        parts=[types.Part.from_function_response(name=call.name, response={"result": "pong"})],
    ))
    contents.append(types.Content(role="user", parts=[types.Part(text=
        "Now call the done tool with a one-word summary."
    )]))
    config2 = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**DONE_DECL)])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    resp2 = client.models.generate_content(model=model, contents=contents, config=config2)
    calls2 = resp2.function_calls or []
    assert calls2, f"Expected a function call, got none. resp2.text={resp2.text!r}"
    assert calls2[0].name == "done", f"Expected 'done', got {calls2[0].name!r}"
    print(f"[2/3] OK: role='tool' history accepted, model called done({dict(calls2[0].args)})")
    print("[3/3] OK: turn-to-turn tool-list narrowing works within one growing contents list")

    print("\nAll manual function-calling mechanics confirmed live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
