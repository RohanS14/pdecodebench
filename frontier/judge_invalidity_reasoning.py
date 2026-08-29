"""
judge_invalidity_reasoning.py — LLM-as-judge evaluation of whether a model's
Stage-2 `valid_exp` justification for a *correctly-identified* invalid row
actually corresponds to a real discrepancy between that row's code and its
valid counterpart, vs. the dataset's (generic, symptom-level) invalidity_note.

Design decided after a dedicated red-team discussion (see the plan doc):
  - Scope: only rows where gt_valid == False AND s2_valid_match == 1 -- the
    model must have ALSO correctly classified the code as invalid. A row
    where the model said "valid" has no invalidity justification to judge.
  - invalidity_note is confirmatory context, NOT a strict mechanism rubric --
    confirmed it's generic/templated (identical wording reused across
    different rows and PDE classes). The judge's real ground truth comes
    from comparing the corrupted code against its valid counterpart
    directly (both given side by side, no pre-computed diff).
  - category is a 3-way classification (none/some/all), not binary, since a
    model that correctly identifies ONE of several independently-injected
    defects (e.g. an exponent change, while missing a separate sign flip)
    gave real, sufficient, correct reasoning -- a binary true/false would
    unfairly score that as "wrong" for not being exhaustive.
  - contains_incorrect_claims is a separate bool, since category alone is a
    recall-only metric with no precision sensitivity (a crisp correct
    explanation and a "shotgun" explanation with one real claim buried in
    wrong ones would otherwise score identically).
  - Uses a separate, stronger judge model (gemini-2.5-pro) than the
    gemini-2.5-flash being evaluated, to reduce self-preference bias --
    accepted caveat: same-family models can still share blind spots.
  - Single-pass judge (no majority-vote redundancy) for this first run --
    accepted as noisier than a voting design; mitigated only by manual
    spot-checking a handful of verdicts (see the plan's Verification step).
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frontier.run_belief_revision_agentic import call_gemini_agentic  # noqa: E402
from shared.dataset_io import load_dataset  # noqa: E402

# Judge model pricing -- Gemini 3.1 Pro (the model actually used for the real
# test run), confirmed via web search this session: $2.00/M input,
# $12.00/M output. Deliberately separate from run_belief_revision.py's
# _PRICE_IN/_PRICE_OUT, which are gemini-2.5-flash rates -- the judge uses a
# different, more expensive model on purpose (see module docstring).
_PRICE_IN_JUDGE = 2.00 / 1_000_000
_PRICE_OUT_JUDGE = 12.00 / 1_000_000


def _judge_token_cost(usage) -> float:
    if usage is None:
        return 0.0
    inp = getattr(usage, "prompt_token_count", 0) or 0
    out = getattr(usage, "candidates_token_count", 0) or 0
    think = getattr(usage, "thoughts_token_count", 0) or 0
    return inp * _PRICE_IN_JUDGE + out * _PRICE_OUT_JUDGE + think * _PRICE_OUT_JUDGE

# ── Valid/invalid mod_type mapping ───────────────────────────────────────────
# The dataset's naming is irregular -- note the inconsistent "Invalid" vs
# "InValid" capitalization -- confirmed by inspection that no single
# suffix-strip rule covers all 4 cases correctly, so this is hardcoded.
INVALID_TO_VALID_MOD_TYPE = {
    "Comm_InValid": "Comm_Valid",
    "NoComm_InValid": "NoComm_Valid",
    "NoComm_CorrVar_InValid": "NoComm_CorrVar",
    "CorrComm_Invalid": "CorrComm",
}


def map_to_valid_mod_type(mod_type: str) -> str | None:
    """Returns the valid-counterpart mod_type for an invalid one, or None if
    mod_type isn't one of the 4 recognized invalid types."""
    return INVALID_TO_VALID_MOD_TYPE.get(mod_type)


def find_valid_counterpart_code(df, gt_sample: str, mod_type: str) -> str | None:
    """Looks up the sibling valid-counterpart row's code by matching
    gt_sample + the mapped valid mod_type. Returns None if no mapping exists
    for mod_type, or no matching row is found (should not happen for a
    well-formed stratified/full sample, but callers must handle it)."""
    valid_mod_type = map_to_valid_mod_type(mod_type)
    if valid_mod_type is None:
        return None
    match = df[(df["gt_sample"] == gt_sample) & (df["mod_type"] == valid_mod_type)]
    if match.empty:
        return None
    return str(match.iloc[0]["code"])


# ── Ground-truth reference (real variable names + genuine comments) ─────────
# Per data/descriptions/data_spec.txt, mod_type Comm_Valid/Comm_InValid are
# the only ones guaranteed to have BOTH real variable names (confirmed:
# obfuscated `foobar_N` names appear exclusively in NoComm_CorrVar[_InValid])
# AND genuine, accurate comments (as opposed to NoComm's absent comments or
# CorrComm's comments deliberately spliced in from a different PDE class --
# corruption_source_pde/injected_comments record exactly which one and
# where). Genuine comments carry real explanatory value beyond the bare
# code -- they annotate intent -- so this single reference pair is given for
# every invalid mod_type except Comm_InValid itself (where it would just
# duplicate what's already shown).
_NEEDS_GROUND_TRUTH_REFERENCE = {"NoComm_InValid", "NoComm_CorrVar_InValid", "CorrComm_Invalid"}


def find_ground_truth_reference(df, gt_sample: str, mod_type: str) -> tuple[str | None, str | None]:
    """Returns (gt_valid_code, gt_invalid_code) from the Comm_Valid/Comm_InValid
    siblings (same gt_sample) -- real names + genuine comments. Returns
    (None, None) if mod_type is Comm_InValid itself (nothing to add), is
    unrecognized, or a sibling row is missing."""
    if mod_type not in _NEEDS_GROUND_TRUTH_REFERENCE:
        return None, None
    valid_match = df[(df["gt_sample"] == gt_sample) & (df["mod_type"] == "Comm_Valid")]
    invalid_match = df[(df["gt_sample"] == gt_sample) & (df["mod_type"] == "Comm_InValid")]
    gt_valid = str(valid_match.iloc[0]["code"]) if not valid_match.empty else None
    gt_invalid = str(invalid_match.iloc[0]["code"]) if not invalid_match.empty else None
    return gt_valid, gt_invalid


def build_caveat_note(mod_type: str, corruption_source_pde: str | None) -> str:
    """Mod_type-specific note explaining what's obfuscated/missing/misleading
    in the code the judge is about to see, and that a genuine reference
    follows immediately. Empty string for Comm_InValid (no reference is
    added, so no caveat is needed) or an unrecognized mod_type."""
    if mod_type == "NoComm_InValid":
        return (
            "Note: the code below has no comments. A reference version with "
            "the same code, showing genuine, accurate comments, is provided "
            "immediately after each code block so you can understand the "
            "code's intended behavior."
        )
    if mod_type == "NoComm_CorrVar_InValid":
        return (
            "Note: the code below has its variable names obfuscated (replaced "
            "with generic placeholders like foobar_N) and has no comments. A "
            "reference version with the original variable names and genuine, "
            "accurate comments is provided immediately after each code block "
            "so you can accurately map between them."
        )
    if mod_type == "CorrComm_Invalid":
        pde = corruption_source_pde or "a different PDE class"
        return (
            f"Note: the comments in the code below were deliberately replaced "
            f"with comments taken from {pde} code and do NOT describe this "
            f"code's actual logic -- do not rely on them. A reference version "
            f"with genuine, accurate comments is provided immediately after "
            f"each code block."
        )
    return ""


# ── Judge prompt + structured output ─────────────────────────────────────────

JUDGE_DECL = {
    "name": "submit_judgment",
    "description": (
        "Submit your judgment of whether the model's justification for "
        "invalidity corresponds to a real discrepancy between the base and "
        "corrupted code."
    ),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["none", "some", "all"],
                "description": (
                    "'none' if the justification identifies no genuine, "
                    "verifiable reason the corrupted code is invalid. 'some' "
                    "if it identifies at least one real, verifiable "
                    "discrepancy between the base and corrupted code that "
                    "would independently cause invalidity, even if it does "
                    "not account for every discrepancy present. 'all' if it "
                    "accounts for every distinct discrepancy you can find "
                    "between the base and corrupted code."
                ),
            },
            "contains_incorrect_claims": {
                "type": "boolean",
                "description": (
                    "True if the justification ALSO includes claims that are "
                    "extraneous, speculative, or factually wrong (in addition "
                    "to any correct ones). False if every claim made is "
                    "accurate, even if incomplete."
                ),
            },
            "explanation": {
                "type": "string",
                "description": (
                    "1-3 sentence justification for the category and "
                    "contains_incorrect_claims choices. If category is "
                    "'some', explicitly note any additional real "
                    "discrepancies between the base and corrupted code that "
                    "the justification did NOT mention."
                ),
            },
        },
        "required": ["category", "contains_incorrect_claims", "explanation"],
    },
}


def build_judge_prompt(
    base_code: str, corrupted_code: str, invalidity_note: str, valid_exp: str,
    gt_valid_code: str | None = None, gt_invalid_code: str | None = None,
    caveat_note: str = "",
) -> str:
    """Block order (each ground-truth block placed immediately after its
    paired original, for easy cross-referencing; caveat note before any code
    so the judge knows how to interpret what follows; judgment materials
    last, right before the final instruction):
      intro -> caveat_note? -> base_valid_code -> ground_truth_valid_code? ->
      corrupted_code_that_was_evaluated -> ground_truth_invalid_code? ->
      invalidity_note -> model_justification -> final instruction.
    gt_valid_code/gt_invalid_code/caveat_note default to the no-reference
    case (omitted entirely), so a Comm_InValid row's prompt is unchanged
    from before this reference was added."""
    parts = [
        "You are evaluating whether a model's stated justification for why a "
        "piece of PDE solver code is physically/numerically invalid actually "
        "corresponds to a real discrepancy in the code, compared to a known-"
        "valid reference version of the same simulation.\n\n"
    ]
    if caveat_note:
        parts.append(f"{caveat_note}\n\n")
    parts.append(f"<base_valid_code>\n{base_code}\n</base_valid_code>\n\n")
    if gt_valid_code is not None:
        parts.append(f"<ground_truth_valid_code>\n{gt_valid_code}\n</ground_truth_valid_code>\n\n")
    parts.append(
        f"<corrupted_code_that_was_evaluated>\n{corrupted_code}\n"
        "</corrupted_code_that_was_evaluated>\n\n"
    )
    if gt_invalid_code is not None:
        parts.append(f"<ground_truth_invalid_code>\n{gt_invalid_code}\n</ground_truth_invalid_code>\n\n")
    parts.append(
        "For context, here is a human annotator's note on the general "
        "symptom this corruption is expected to cause (this is a general "
        "description, not necessarily a specific mechanism -- do not treat "
        "it as the definitive or only correct answer):\n\n"
        f"<invalidity_note>\n{invalidity_note}\n</invalidity_note>\n\n"
        "Here is the model's own justification for why it judged the "
        "corrupted code invalid:\n\n"
        f"<model_justification>\n{valid_exp}\n</model_justification>\n\n"
        "Compare the base and corrupted code (and reference versions, if "
        "provided) directly to determine what actually changed. Then call "
        "submit_judgment with your assessment."
    )
    return "".join(parts)


def call_judge(client, model: str, base_code: str, corrupted_code: str,
                invalidity_note: str, valid_exp: str, max_retries: int = 4,
                gt_valid_code: str | None = None, gt_invalid_code: str | None = None,
                caveat_note: str = "") -> dict:
    from google.genai import types

    prompt = build_judge_prompt(base_code, corrupted_code, invalidity_note, valid_exp,
                                 gt_valid_code=gt_valid_code, gt_invalid_code=gt_invalid_code,
                                 caveat_note=caveat_note)
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=50_000,
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**JUDGE_DECL)])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["submit_judgment"]
            )
        ),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    resp = call_gemini_agentic(client, model, contents, config, max_retries)
    cost_usd = _judge_token_cost(getattr(resp, "usage_metadata", None))
    calls = getattr(resp, "function_calls", None) or []
    if not calls:
        return {"category": None, "contains_incorrect_claims": None,
                "explanation": "(judge produced no function call)", "_raw_text": getattr(resp, "text", None),
                "cost_usd": cost_usd}
    args = dict(calls[0].args or {})
    return {
        "category": args.get("category"),
        "contains_incorrect_claims": args.get("contains_incorrect_claims"),
        "explanation": args.get("explanation"),
        "cost_usd": cost_usd,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-judge invalidity-reasoning eval")
    p.add_argument("--judge_model", default="gemini-pro-latest",
                   help="Default is an alias that always resolves to the current Pro-tier model "
                        "-- gemini-2.5-pro itself 404s ('no longer available to new users') despite "
                        "still appearing in the models.list() response, confirmed live.")
    p.add_argument("--dataset", default=str(REPO_ROOT / "results" / "frontier" / "stratified_64" / "dataset.csv"))
    p.add_argument("--stage2_jsonl", action="append", required=True,
                   help="Path to a Stage-2 output JSONL (pass once per thinking condition).")
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "frontier" / "stratified_64" / "judge" / "judge_results.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_cost_usd", type=float, default=1.0,
                   help="Session-wide cost cap across all judge calls (all "
                        "stage2_jsonl inputs combined). NOTE: a real 31-row "
                        "nothink pass under the PRE-ground-truth-reference "
                        "prompt already cost $0.9042 -- adding the "
                        "ground-truth reference blocks (see "
                        "find_ground_truth_reference) increases per-row cost "
                        "further for ~29/31 of those rows. Re-check/raise "
                        "this default before a real re-run rather than "
                        "trusting the old $0.3-0.4 estimate.")
    p.add_argument("--max_retries", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        workspace_root = REPO_ROOT.parent.parent
        sys.path.insert(0, str(workspace_root / "packages" / "key_handler" / "key_handler"))
        from key_handler import KeyHandler
        KeyHandler.set_env_key()
    except ImportError:
        pass

    import os
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        sys.exit("[judge] ERROR: GOOGLE_API_KEY not set (checked KeyHandler and os.environ).")

    from google import genai
    client = genai.Client(api_key=api_key)

    df = load_dataset(args.dataset)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["title"], r["thinking_budget"]))
        print(f"[judge] Checkpoint: {len(done)} (title, thinking_budget) pairs already judged.", flush=True)

    n_judged = 0
    session_cost = 0.0
    cost_limit_hit = False
    for jsonl_path in args.stage2_jsonl:
        if cost_limit_hit:
            break
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        eligible = [r for r in rows if r.get("gt_valid") is False and r.get("s2_valid_match") == 1]
        print(f"[judge] {jsonl_path}: {len(rows)} rows, {len(eligible)} eligible "
              f"(gt_valid=False AND s2_valid_match=1)", flush=True)

        for row_result in eligible:
            title = row_result["title"]
            thinking_budget = row_result["thinking_budget"]
            if (title, thinking_budget) in done:
                continue
            if args.limit and n_judged >= args.limit:
                print(f"[judge] --limit {args.limit} reached.", flush=True)
                break
            if session_cost >= args.max_cost_usd:
                print(f"[judge] --max_cost_usd {args.max_cost_usd:.2f} reached. Stopping.", flush=True)
                cost_limit_hit = True
                break

            dataset_row = df[df["title"] == title]
            if dataset_row.empty:
                print(f"[judge] WARNING: {title} not found in --dataset, skipping.", flush=True)
                continue
            dataset_row = dataset_row.iloc[0]

            base_code = find_valid_counterpart_code(df, dataset_row["gt_sample"], dataset_row["mod_type"])
            if base_code is None:
                print(f"[judge] WARNING: no valid counterpart found for {title} "
                      f"(mod_type={dataset_row['mod_type']}), skipping.", flush=True)
                continue

            gt_valid_code, gt_invalid_code = find_ground_truth_reference(
                df, dataset_row["gt_sample"], dataset_row["mod_type"]
            )
            caveat_note = build_caveat_note(dataset_row["mod_type"], dataset_row.get("corruption_source_pde"))

            corrupted_code = str(dataset_row["code"])
            invalidity_note = str(dataset_row.get("invalidity_note") or "")
            valid_exp = str(row_result.get("s2_submit_args", {}).get("valid_exp") or "")

            print(f"  [{n_judged+1}] {title} (thinking_budget={thinking_budget})", end=" ", flush=True)
            verdict = call_judge(client, args.judge_model, base_code, corrupted_code,
                                  invalidity_note, valid_exp, max_retries=args.max_retries,
                                  gt_valid_code=gt_valid_code, gt_invalid_code=gt_invalid_code,
                                  caveat_note=caveat_note)

            out_row = {
                "title": title,
                "thinking_budget": thinking_budget,
                "mod_type": dataset_row["mod_type"],
                "pde_class": dataset_row["pde_class"],
                "category": verdict["category"],
                "contains_incorrect_claims": verdict["contains_incorrect_claims"],
                "explanation": verdict["explanation"],
                "used_ground_truth_reference": gt_valid_code is not None or gt_invalid_code is not None,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(out_row) + "\n")
            n_judged += 1
            session_cost += verdict["cost_usd"]
            print(f"category={verdict['category']} contains_incorrect_claims={verdict['contains_incorrect_claims']} "
                  f"${verdict['cost_usd']:.5f}", flush=True)

    print(f"\n[judge] Done. {n_judged} new judgments -> {out_path}", flush=True)
    print(f"[judge] Session cost: ${session_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
