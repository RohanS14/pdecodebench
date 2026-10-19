"""
sign_flip_detail_csv.py — builds 3 CSVs for detailed row-by-row reading:
  1. sign_flip_detail_pure.csv   -- the 17 pure SIGN FLIP gt_samples
  2. sign_flip_detail_mixed.csv  -- the 3 MIXED gt_samples
  3. sign_flip_detail_all_correctly_invalid.csv -- every judge-eligible row
     (all 32 gt_samples, any defect type), for comparison against (1)/(2)

Columns: title, condition, invalidity_note, defect_description,
agent_valid_exp, judge_category, judge_explanation.

defect_description is my manual, diff-based one-line classification from
agent_docs/sign_flip_classification.txt (all 32 gt_samples have one, not
just the 17 pure sign-flip ones) -- distinct from invalidity_note, the
dataset's own generic/templated field.

Usage: python3 frontier/sign_flip_detail_csv.py
Output: agent_docs/sign_flip_detail_{pure,mixed,all_correctly_invalid}.csv
"""
import csv
import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent

# gt_sample -> (verdict, one-line description), transcribed from
# agent_docs/sign_flip_classification.txt (manual, diff-based inspection).
CLASSIFICATION = {
    "Burgers_1": ("SIGN FLIP", "du_dt = (f[0:n]-f[1:n+1])/dx -> (f[1:n+1]-f[0:n])/dx: operand swap negates the flux-difference term."),
    "Burgers_2": ("OTHER", "f[ul<=ur] = np.minimum(...) -> np.maximum(...): Godunov flux branch selection changed, not a sign."),
    "Burgers_3": ("OTHER", "if/else branches swap which neighbor index (f_bar[i-1] vs f_bar[i]) is used: wrong-neighbor reference, not a sign."),
    "Burgers_4": ("OTHER", "Deletes the upwind/dissipation correction term entirely (Lax-Friedrichs -> unstable central scheme)."),
    "Burgers_5": ("SIGN FLIP", "F_left = np.roll(F,1) -> np.roll(F,-1): negates the roll-shift argument (sign flip on a shift index, not a coefficient)."),
    "Burgers_6": ("SIGN FLIP", "u = u - ... + diffusion -> ... - diffusion: turns diffusion into anti-diffusion."),
    "Burgers_7": ("SIGN FLIP", "u = u + dt*(-advect + diffuse) -> (-advect - diffuse): sign flip on the diffuse term."),
    "Burgers_8": ("SIGN FLIP", "u,v updates: (-(...) + nu*lap) -> (-(...) - nu*lap), both components: turns viscous diffusion into anti-diffusion."),
    "Heat_1": ("MIXED", "Negates A (real sign flip on the diffusion operator) AND separately changes dx**2->dx**3 and t_steps 1001->1000 (unrelated)."),
    "Heat_2": ("SIGN FLIP", "return (matmul(A,u)+b) -> (matmul(A,-u)-b): negates u and b together, equivalent to negating the whole RHS."),
    "Heat_3": ("SIGN FLIP", "toeplitz([-2.0,1.0]+...) -> ([-2.0,-1.0]+...): breaks the symmetric [1,-2,1] Laplacian stencil."),
    "Heat_4": ("SIGN FLIP", "dudt = k*matmul(D,u[t])/dx**2 -> k*matmul(D,-u[t])/dx**2: negates u inside a linear operator."),
    "Heat_5": ("SIGN FLIP", "u_new = u + r*(...) -> u - r*(...): turns diffusion into anti-diffusion (backward heat equation)."),
    "Heat_6": ("OTHER", "rhs left-neighbor term u[:-2] replaced with center u[1:-1]: indexing error, not a sign."),
    "Heat_7": ("SIGN FLIP", "rhs += (r/2)*(...) -> rhs -= (r/2)*(...): sign flip on the implicit coupling correction term."),
    "Heat_8": ("SIGN FLIP", "decay = exp(-alpha*K2*t) -> exp(alpha*K2*t): removes the negative sign in the exponent, decay becomes blowup."),
    "NavierStokes_1": ("SIGN FLIP", "Sign flip on the connector before the pressure-gradient term, in both u and v momentum equations."),
    "NavierStokes_2": ("SIGN FLIP", "Same pressure-gradient-term sign flip repeated across 6 periodic-boundary branches."),
    "NavierStokes_3": ("OTHER", "K/np.where(K2==0,1,K2) -> K/K2: removes the div-by-zero safety guard, not a sign change."),
    "NavierStokes_4": ("OTHER", "Dealiasing filter body replaced with return jnp.ones(...): deleted-functionality defect."),
    "NavierStokes_5": ("SIGN FLIP", "u_new = u + dt*(G + nu*(...)) -> dt*(G - nu*(...)): turns viscous diffusion into anti-diffusion."),
    "NavierStokes_6": ("MIXED", "Off-diagonal terms cleanly negated (real sign flips) but center coefficient also changes (1-r)->(1+r): bundled coefficient change."),
    "NavierStokes_7": ("SIGN FLIP", "Same pressure-gradient-term sign flip pattern as NavierStokes_1/2, in both u and v momentum equations."),
    "NavierStokes_8": ("OTHER", "psi_hat = omega_hat/K2safe -> omega_hat*K2: divide->multiply AND drops the safety-guarded variable, not a sign change."),
    "Wave_1": ("MIXED", "Most corrupted lines are a clean 2*u_n -> -2*u_n sign flip, but 3 boundary-edge lines instead change the coefficient magnitude (2->10/5/3)."),
    "Wave_2": ("OTHER", "Coefficients replaced with unrelated positive magnitudes (e.g. -u_jm1 -> 10*u_jm1): not a simple negation."),
    "Wave_3": ("OTHER", "Adds a constant forcing term (+= np.ones_like(...)) that wasn't in the original: added spurious term, not a sign flip."),
    "Wave_4": ("OTHER", "Removes the np.abs()/zero-mode safety clamp on k_abs: guard removal, not a single sign flip."),
    "Wave_5": ("OTHER", "u_prev replaced with u_curr in the update formula: wrong-variable reference, not a sign flip."),
    "Wave_6": ("OTHER", "cos/sin replaced with cosh/sinh: oscillatory solution becomes exponential growth, a function-type swap, not a sign."),
    "Wave_7": ("SIGN FLIP", "u_next = (2*u_curr - u_prev + r2*lap)*sponge -> (... - r2*lap)*sponge: sign flip on the Laplacian/propagation term."),
    "Wave_8": ("SIGN FLIP", "K = -(c**2)*Lap -> K = (c**2)*Lap: removes the negative sign on the stiffness operator entirely."),
}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)



# ── Path arguments ───────────────────────────────────────────────────────────
# Defaults point at the completed gemini-2.5-flash 256-row run so existing usage
# is unchanged, but every path is overridable: a new model's sweep lands under a
# different, model-keyed output root, and hand-editing this file per model was
# the previous alternative.
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage2", action="append", default=[],
                   help="Stage-2 JSONL. Repeatable, once per condition. Each file's "
                        "condition is read from its own rows, so a single-condition "
                        "Gemini 3 arm and the original two-condition 2.5 run both work.")
    p.add_argument("--first", default=None,
                   help="DEPRECATED alias for the first --stage2 path.")
    p.add_argument("--second", default=None,
                   help="DEPRECATED alias for the second --stage2 path.")
    p.add_argument("--judge", default=str(REPO_ROOT / "results/frontier/stratified_256/judge/judge_results.jsonl"),
                   help="Judge results JSONL. Must come from the SAME model and "
                        "condition as the Stage-2 files above -- a judge pass from "
                        "another arm silently answers a different question.")
    p.add_argument("--out_prefix", default="sign_flip_detail",
                   help="Output basename prefix under agent_docs/. Set this per arm "
                        "(e.g. sign_flip_detail_37high) so one run's sheets never "
                        "overwrite another's.")
    return p.parse_args()


def _condition_of(row) -> str:
    """Condition identity for a Stage-2 or judge row.

    `condition` is the field going forward -- the only one that can name a
    Gemini 3 condition (e.g. "tl_high"). Older 2.5 rows predate it and carry
    only a numeric thinking_budget, so that is the fallback. Same precedence as
    judge_invalidity_reasoning._condition_of() and the viz's
    with_condition_column(), deliberately: keying this script on the raw numeric
    thinking_budget (the previous {0: nothink, 1536: think} dict) raised KeyError
    on every Gemini 3 row, since thinking_budget is None there.
    """
    c = row.get("condition")
    if c is not None and c == c and str(c).strip():
        return str(c)
    tb = row.get("thinking_budget")
    if tb is not None and tb == tb:
        return f"tb{int(tb)}"
    return "unknown"


def main() -> None:
    args = _parse_args()
    dataset = pd.read_csv(REPO_ROOT / "data" / "merged_mod_jul28.csv")
    title_to_gt_sample = dict(zip(dataset["title"], dataset["gt_sample"]))
    title_to_invalidity_note = dict(zip(dataset["title"], dataset["invalidity_note"]))

    paths = list(args.stage2) or [p for p in (args.first, args.second) if p]
    if not paths:
        raise SystemExit("Pass at least one --stage2 <path>.")
    judge = load_jsonl(args.judge)

    # Each Stage-2 file is keyed by the condition its OWN rows declare, rather
    # than by position or by a hardcoded thinking_budget number.
    sweep_by_condition = {}
    for path in paths:
        df = load_jsonl(Path(path))
        if df.empty:
            continue
        conds = {_condition_of(r) for _, r in df.iterrows()}
        if len(conds) != 1:
            raise SystemExit(f"{path} mixes conditions {sorted(conds)}; expected exactly one.")
        sweep_by_condition[conds.pop()] = df
    print(f"Stage-2 conditions loaded: {sorted(sweep_by_condition)}")

    rows_out = []
    missing_cond = set()
    for _, jrow in judge.iterrows():
        title = jrow["title"]
        cond = _condition_of(jrow)
        gt_sample = title_to_gt_sample.get(title)
        verdict, description = CLASSIFICATION.get(gt_sample, ("UNKNOWN", ""))

        sweep_df = sweep_by_condition.get(cond)
        if sweep_df is None:
            missing_cond.add(cond)
            continue
        sweep_row = sweep_df[sweep_df["title"] == title]
        agent_valid_exp = ""
        agent_score = ""
        if not sweep_row.empty:
            submit_args = sweep_row.iloc[0].get("s2_submit_args") or {}
            if isinstance(submit_args, str):
                submit_args = json.loads(submit_args)
            agent_valid_exp = submit_args.get("valid_exp", "")
            score = sweep_row.iloc[0].get("s2_validity_score")
            agent_score = "" if score is None or score != score else int(score)

        rows_out.append({
            "title": title,
            "condition": cond,
            "defect_verdict": verdict,
            "invalidity_note": title_to_invalidity_note.get(title, ""),
            "defect_description": description,
            "agent_validity_score": agent_score,
            "agent_valid_exp": agent_valid_exp,
            "judge_category": jrow.get("category", ""),
            # The precision signals. judge_category is recall-only (did the
            # justification name the real defect), so a row can be "all" and
            # still assert something false -- only visible here. Two columns:
            # wrong-about-the-code is the real error, unverifiable runtime
            # figures are not. Older judge runs have only the merged
            # `contains_incorrect_claims`; it is carried through as-is so old
            # and new sheets stay readable side by side.
            "contains_incorrect_code_claims": jrow.get("contains_incorrect_code_claims", ""),
            "contains_unverifiable_execution_claims": jrow.get("contains_unverifiable_execution_claims", ""),
            "contains_incorrect_claims_legacy": jrow.get("contains_incorrect_claims", ""),
            "judge_explanation": jrow.get("explanation", ""),
        })

    if missing_cond:
        print(f"WARNING: judge rows for condition(s) {sorted(missing_cond)} had no "
              f"matching --stage2 file and were SKIPPED.")

    all_df = pd.DataFrame(rows_out)
    fieldnames = ["title", "condition", "invalidity_note", "defect_description",
                  "agent_validity_score", "agent_valid_exp", "judge_category",
                  "contains_incorrect_code_claims",
                  "contains_unverifiable_execution_claims",
                  "contains_incorrect_claims_legacy", "judge_explanation"]

    def write_csv(df: pd.DataFrame, name: str) -> None:
        out_path = REPO_ROOT / "agent_docs" / name
        df[fieldnames].to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Wrote {out_path} ({len(df)} rows)")

    pre = args.out_prefix
    write_csv(all_df[all_df["defect_verdict"] == "SIGN FLIP"], f"{pre}_pure.csv")
    write_csv(all_df[all_df["defect_verdict"] == "MIXED"], f"{pre}_mixed.csv")
    write_csv(all_df, f"{pre}_all_correctly_invalid.csv")


if __name__ == "__main__":
    main()
