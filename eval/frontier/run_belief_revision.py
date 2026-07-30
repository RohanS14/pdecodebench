"""
run_belief_revision.py — Two-stage frontier model eval (Gemini API).

Stage 1: Model reads PDE code and answers (pde / method / behavior / valid).
Stage 2: Same conversation thread. Model is shown the precomputed trajectory
         summary and asked to re-evaluate using both the code and runtime evidence.

One JSONL row per script stores both stage responses and the transition.

Usage:
  # Install deps (in .venv, not .pde_venv):
  pip install google-genai pandas openpyxl

  # Set API key:
  export GOOGLE_API_KEY="..."

  # Smoke test (5 scripts):
  python eval/frontier/run_belief_revision.py --limit 5

  # Full run:
  python eval/frontier/run_belief_revision.py

Output: results/frontier/<model_slug>__belief_revision.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

from frontier.parse_frontier import (  # noqa: E402
    classify_hedge,
    compute_traj_signal,
    format_trajectory_block,
    parse_response,
    score_row,
)
from dataset_io import DEFAULT_MOD_DATASET, load_dataset  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

# Burgers_1 is fixed — set is empty. Add e.g. {"Burgers_1"} to re-exclude.
EXCLUDED_SAMPLES: set = set()

# Gemini 2.5 Flash pricing (USD/token, May 2025)
_PRICE_IN    = 0.15  / 1_000_000   # input
_PRICE_OUT   = 0.60  / 1_000_000   # non-thinking output
_PRICE_THINK = 3.50  / 1_000_000   # thinking tokens

# ── Prompts ───────────────────────────────────────────────────────────────────

# Stage 1: identical to existing run_eval.py prompt — model does not know
# a second stage is coming.
PROMPT_S1 = """\
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

- pde: the type of PDE being solved
- method: numerical method(s) used — list all that apply
- behavior: dominant physical process(es) — list all that apply
- valid: does this code run and produce a correct physical solution for the PDE?\
"""

# Stage 2: show trajectory summary, ask model to re-evaluate.
# Explicit but neutral framing — tells the model to use the new evidence
# without telling it what to conclude.
# The preamble explains the format; inline annotations clarify the 4 non-obvious
# fields without interpreting what they mean for validity.
PROMPT_S2 = """\
Here is the runtime output from executing this simulation.
Fields below summarize execution of the script. Statistics are computed on the
variable identified as the primary solution array; axis 0 of that array is
treated as the time dimension.

<runtime_summary>
{traj_block}
</runtime_summary>

Re-evaluate your analysis using both the code and this runtime evidence.

Output only:
pde: ____
method: ____
behavior: ____
valid: ____\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def sha256_hex(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def get_base_sample(title: str) -> str:
    """'Burgers_Comm_InValid_1' → 'Burgers_1'"""
    parts = title.split("_")
    return f"{parts[0]}_{parts[-1]}"


def load_trajectories(path: Path) -> dict[str, dict]:
    traj: dict[str, dict] = {}
    if not path.exists():
        return traj
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                traj[r["title"]] = r
    return traj


def load_checkpoint(path: Path) -> set[str]:
    """Return set of titles already processed."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["title"])
    print(f"[frontier] Checkpoint: {len(done)} rows already done.", flush=True)
    return done


def append_result(path: Path, result: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def get_transition(s1_match: int | None, s2_match: int | None) -> str:
    s1 = bool(s1_match) if s1_match is not None else False
    s2 = bool(s2_match) if s2_match is not None else False
    return f"{'right' if s1 else 'wrong'}->{'right' if s2 else 'wrong'}"


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter. Each API call (both stages) consumes one token."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.time()


# ── Gemini API ────────────────────────────────────────────────────────────────

def call_gemini(client, model: str, contents: list, max_retries: int = 4):
    """Single Gemini API call with exponential backoff on 429/503."""
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            return resp
        except Exception as exc:
            err = str(exc)
            retryable = any(code in err for code in ("429", "503", "500")) \
                        or "quota" in err.lower() or "rate" in err.lower()
            if retryable and attempt < max_retries:
                wait = (2 ** attempt) * 5
                print(f"    [retry {attempt+1}] {err[:80]} — wait {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise


def token_cost(usage) -> tuple[int, int, int, float]:
    """Returns (input_tokens, output_tokens, think_tokens, cost_usd)."""
    if usage is None:
        return 0, 0, 0, 0.0
    inp   = getattr(usage, "prompt_token_count",     0) or 0
    out   = getattr(usage, "candidates_token_count", 0) or 0
    think = getattr(usage, "thoughts_token_count",   0) or 0
    cost  = inp * _PRICE_IN + out * _PRICE_OUT + think * _PRICE_THINK
    return inp, out, think, cost


def estimate_cost_usd(prompt_text: str) -> float:
    """Pre-flight cost estimate (character-count proxy, ~4 chars/token)."""
    return (len(prompt_text) / 4) * _PRICE_IN + 120 * _PRICE_OUT


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frontier belief-revision eval")
    p.add_argument("--model",        default="gemini-2.5-flash")
    p.add_argument("--dataset",      default=str(REPO_ROOT / DEFAULT_MOD_DATASET))
    p.add_argument("--traj_file",    default=str(REPO_ROOT / "data" / "trajectories.jsonl"))
    p.add_argument("--output_dir",   default=str(REPO_ROOT / "results" / "frontier"))
    p.add_argument("--max_cost_usd", type=float, default=5.0)
    p.add_argument("--rpm",          type=float, default=8.0,
                   help="API calls per minute (both stages count separately)")
    p.add_argument("--limit",        type=int,   default=0,
                   help="Max scripts to process; 0 = no limit")
    p.add_argument("--max_retries",  type=int,   default=4)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        sys.exit("[frontier] ERROR: GOOGLE_API_KEY not set.")

    from google import genai
    client = genai.Client(api_key=api_key)

    # ── Output path
    os.makedirs(args.output_dir, exist_ok=True)
    slug     = args.model.replace("/", "__").replace("-", "").replace(".", "")
    out_path = Path(args.output_dir) / f"{slug}__belief_revision.jsonl"

    # ── Load dataset
    df = load_dataset(args.dataset)
    df["_base"] = df["title"].apply(get_base_sample)
    df = df[~df["_base"].isin(EXCLUDED_SAMPLES)].drop(columns=["_base"])
    print(f"[frontier] {len(df)} scripts after exclusions", flush=True)

    # ── Load trajectories
    traj_map = load_trajectories(Path(args.traj_file))
    if not traj_map:
        sys.exit(f"[frontier] ERROR: no trajectories found in {args.traj_file}. "
                 "Run precompute_trajectories.py first.")
    print(f"[frontier] {len(traj_map)} trajectory entries loaded", flush=True)

    # Warn about scripts with missing or failed trajectories
    missing = [r["title"] for _, r in df.iterrows()
               if r["title"] not in traj_map
               or not traj_map[r["title"]].get("ran_to_completion")]
    if missing:
        print(f"[frontier] WARNING: {len(missing)} scripts have no valid trajectory "
              f"(will skip Stage 2 block): {missing[:5]}{'...' if len(missing)>5 else ''}",
              flush=True)

    # ── Resume
    done = load_checkpoint(out_path)
    todo = df[~df["title"].isin(done)].reset_index(drop=True)
    print(f"[frontier] To process: {len(todo)}", flush=True)

    if todo.empty:
        print("[frontier] All done.", flush=True)
        return

    limiter      = RateLimiter(args.rpm)
    session_cost = 0.0
    new_rows     = 0

    for _, row in todo.iterrows():
        if args.limit and new_rows >= args.limit:
            print(f"[frontier] --limit {args.limit} reached.", flush=True)
            break

        title    = row["title"]
        mod_type = row["mod_type"]
        code     = str(row["code"])
        gt_valid = bool(row["phys_valid"])
        traj     = traj_map.get(title, {})

        prompt_s1 = PROMPT_S1.format(code=code)

        # Pre-flight cost check (two calls estimated)
        est = estimate_cost_usd(prompt_s1) * 2
        if session_cost + est > args.max_cost_usd:
            print(
                f"[frontier] Cost limit ${args.max_cost_usd:.2f} reached "
                f"(current=${session_cost:.4f}). Stopping.", flush=True
            )
            break

        print(f"  [{new_rows+1}] {title}", end=" ", flush=True)
        retry_count = 0

        # ── Stage 1 ──────────────────────────────────────────────────────────
        limiter.wait()
        t0 = time.time()
        try:
            from google.genai import types
            contents_s1 = [
                types.Content(role="user",
                              parts=[types.Part(text=prompt_s1)])
            ]
            resp1       = call_gemini(client, args.model, contents_s1,
                                      args.max_retries)
            s1_text     = resp1.text or ""
        except Exception as exc:
            print(f"S1-ERROR({exc})", flush=True)
            retry_count = args.max_retries
            s1_text = ""
            resp1   = None
        s1_latency = round(time.time() - t0, 3)
        s1_inp, s1_out, s1_think, s1_cost = token_cost(
            resp1.usage_metadata if resp1 else None
        )
        session_cost += s1_cost

        # ── Stage 2 ──────────────────────────────────────────────────────────
        traj_block = format_trajectory_block(traj) if traj else "ran_to_completion: unknown"
        prompt_s2  = PROMPT_S2.format(traj_block=traj_block)

        limiter.wait()
        t0 = time.time()
        try:
            # Continue the same conversation thread — model sees its own S1 answer
            contents_s2 = [
                types.Content(role="user",
                              parts=[types.Part(text=prompt_s1)]),
                types.Content(role="model",
                              parts=[types.Part(text=s1_text)]),
                types.Content(role="user",
                              parts=[types.Part(text=prompt_s2)]),
            ]
            resp2   = call_gemini(client, args.model, contents_s2,
                                  args.max_retries)
            s2_text = resp2.text or ""
        except Exception as exc:
            print(f"S2-ERROR({exc})", flush=True)
            retry_count = max(retry_count, args.max_retries)
            s2_text = ""
            resp2   = None
        s2_latency = round(time.time() - t0, 3)
        s2_inp, s2_out, s2_think, s2_cost = token_cost(
            resp2.usage_metadata if resp2 else None
        )
        session_cost += s2_cost

        # ── Parse & score (both stages) ───────────────────────────────────────
        row_dict = row.to_dict()

        p1      = parse_response(s1_text)
        scores1 = score_row(p1, row_dict, embed_model=None)
        hedge1  = classify_hedge(p1.get("valid") or "")

        p2      = parse_response(s2_text)
        scores2 = score_row(p2, row_dict, embed_model=None)
        hedge2  = classify_hedge(p2.get("valid") or "")

        vm1 = scores1.get("valid_match")
        vm2 = scores2.get("valid_match")

        result = {
            # ── Identity
            "model":       args.model,
            "title":       title,
            "mod_type":    mod_type,
            "gt_valid":    gt_valid,
            "gt_pde":      str(row["pde_class"]),
            "gt_method":   str(row["num_method"]),
            "gt_behavior": str(row["phys_process"]),
            "code_hash":   sha256_hex(code),

            # ── Stage 1 (code only)
            "s1_response":         s1_text,
            "s1_parsed_pde":       p1.get("pde"),
            "s1_parsed_method":    p1.get("method"),
            "s1_parsed_behavior":  p1.get("behavior"),
            "s1_parsed_valid":     p1.get("valid"),
            "s1_valid_match":      vm1,
            "s1_pde_match":        scores1.get("pde_match"),
            "s1_method_any_match": scores1.get("method_any_match"),
            "s1_behavior_any_match": scores1.get("behavior_any_match"),
            "s1_hedge_class":      hedge1,

            # ── Stage 2 (code + trajectory)
            "s2_response":         s2_text,
            "s2_parsed_pde":       p2.get("pde"),
            "s2_parsed_method":    p2.get("method"),
            "s2_parsed_behavior":  p2.get("behavior"),
            "s2_parsed_valid":     p2.get("valid"),
            "s2_valid_match":      vm2,
            "s2_pde_match":        scores2.get("pde_match"),
            "s2_method_any_match": scores2.get("method_any_match"),
            "s2_behavior_any_match": scores2.get("behavior_any_match"),
            "s2_hedge_class":      hedge2,

            # ── Transition analysis
            "transition":         get_transition(vm1, vm2),
            "delta_valid_match":  (vm2 - vm1) if (vm1 is not None and vm2 is not None) else None,
            "traj_signal":        compute_traj_signal(traj, gt_valid),

            # ── Trajectory metadata (for analysis; not shown to model)
            "traj_ran":            traj.get("ran_to_completion"),
            "traj_array_name":     traj.get("main_array_name"),
            "traj_shape":          traj.get("shape"),
            "traj_has_nan":        traj.get("has_nan"),
            "traj_has_inf":        traj.get("has_inf"),
            "traj_spike_ratio":    traj.get("spike_ratio"),
            "traj_large_spike":    traj.get("large_spike_detected"),  # internal only

            # ── Bookkeeping
            "s1_input_tokens":   s1_inp,
            "s1_output_tokens":  s1_out,
            "s1_think_tokens":   s1_think,
            "s2_input_tokens":   s2_inp,
            "s2_output_tokens":  s2_out,
            "s2_think_tokens":   s2_think,
            "total_cost_usd":    round(s1_cost + s2_cost, 8),
            "s1_latency_sec":    s1_latency,
            "s2_latency_sec":    s2_latency,
            "retry_count":       retry_count,
        }

        append_result(out_path, result)
        new_rows += 1

        print(
            f"s1={hedge1}({'✓' if vm1 else '✗'}) "
            f"→ s2={hedge2}({'✓' if vm2 else '✗'}) "
            f"[{result['transition']}] "
            f"${s1_cost+s2_cost:.5f}",
            flush=True,
        )

    print(
        f"\n[frontier] Done. {new_rows} new rows → {out_path}",
        flush=True,
    )
    print(f"[frontier] Session cost: ${session_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
