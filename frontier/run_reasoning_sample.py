"""
run_reasoning_sample.py — Ask Gemini to explain its valid/invalid judgment on 12 targeted scripts.

Targets:
  - 8 valid scripts where Gemini said No across all mod_types (unexplained failures)
  - 4 invalid scripts the model got right (control)

Stage 1 only (no trajectory, no Stage 2). Appends a 'reason:' line to the prompt.

Output: results/frontier/reasoning_sample.jsonl  (also printed to stdout)

Usage:
    .venv/bin/python frontier/run_reasoning_sample.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frontier.parse_frontier import parse_response  # noqa: E402
from shared.dataset_io import DEFAULT_MOD_DATASET, load_dataset  # noqa: E402

# ── Target scripts ────────────────────────────────────────────────────────────

TARGETS = [
    # Valid code — model said No across all mod_types (unexplained failures)
    "Burgers_Comm_Valid_3",
    "Burgers_Comm_Valid_4",
    "Heat_Comm_Valid_1",
    "Heat_Comm_Valid_2",
    "NavierStokes_Comm_Valid_2",
    "Wave_Comm_Valid_1",
    "Wave_Comm_Valid_2",
    "Wave_Comm_Valid_3",
    # Invalid code — model got right (control: what does a correct No look like?)
    "Burgers_Comm_InValid_1",
    "Burgers_Comm_InValid_2",
    "Burgers_Comm_InValid_3",
    "Burgers_Comm_InValid_4",
]

PROMPT = """\
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
reason: ____

- pde: the type of PDE being solved
- method: numerical method(s) used — list all that apply
- behavior: dominant physical process(es) — list all that apply
- valid: does this code run and produce a correct physical solution for the PDE?
- reason: 1-2 sentences explaining your valid/invalid judgment\
"""

# Gemini 2.5 Flash pricing
_PRICE_IN  = 0.15 / 1_000_000
_PRICE_OUT = 1.25 / 1_000_000


# ── Rate limiter (copied from run_belief_revision.py) ─────────────────────────

class RateLimiter:
    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.time()


# ── Gemini call (copied from run_belief_revision.py) ─────────────────────────

def call_gemini(client, model: str, contents: list, max_retries: int = 4):
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            err = str(exc)
            retryable = any(c in err for c in ("429", "503", "500")) \
                        or "quota" in err.lower() or "rate" in err.lower()
            if retryable and attempt < max_retries:
                wait = (2 ** attempt) * 5
                print(f"  [retry {attempt+1}] {err[:80]} — wait {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise


def parse_reason(text: str) -> str:
    """Extract the reason: field from the response."""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("reason:"):
            return line[len("reason:"):].strip()
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not set.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model  = "gemini-2.5-flash"

    df = load_dataset(REPO_ROOT / DEFAULT_MOD_DATASET)
    df = df[df["title"].isin(TARGETS)].set_index("title")

    missing = [t for t in TARGETS if t not in df.index]
    if missing:
        print(f"WARNING: {len(missing)} targets not found in dataset: {missing}")

    out_path = REPO_ROOT / "results" / "frontier" / "reasoning_sample.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    limiter     = RateLimiter(rpm=30)
    total_cost  = 0.0
    results     = []

    for i, title in enumerate(TARGETS):
        if title not in df.index:
            continue

        row      = df.loc[title]
        code     = str(row["code"])
        gt_valid = bool(row["phys_valid"])
        prompt   = PROMPT.format(code=code)

        print(f"\n[{i+1}/{len(TARGETS)}] {title}  (gt_valid={gt_valid})", flush=True)

        limiter.wait()
        try:
            contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
            resp     = call_gemini(client, model, contents)
            text     = resp.text or ""
        except Exception as exc:
            print(f"  ERROR: {exc}")
            text = ""
            resp = None

        parsed = parse_response(text)
        reason = parse_reason(text)

        usage    = resp.usage_metadata if resp else None
        inp      = getattr(usage, "prompt_token_count",     0) or 0
        out      = getattr(usage, "candidates_token_count", 0) or 0
        cost     = inp * _PRICE_IN + out * _PRICE_OUT
        total_cost += cost

        result = {
            "title":        title,
            "gt_valid":     gt_valid,
            "parsed_valid": parsed.get("valid"),
            "reason":       reason,
            "full_response": text,
            "cost_usd":     round(cost, 6),
        }
        results.append(result)

        print(f"  valid: {parsed.get('valid')}  (gt={gt_valid})")
        print(f"  reason: {reason}")

    # Write output
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*70}")
    print(f"Done. {len(results)} results → {out_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
