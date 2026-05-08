"""
parse_frontier.py — Parsing and scoring utilities for the frontier belief-revision eval.

Re-uses parse_response() and score_row() from ../parse_score.py unchanged so
that scoring is identical to prior experiments.

Adds:
  classify_hedge(raw)         — hedge class label, matches E9 in visualize_v4_enhanced.py
  format_trajectory_block()   — formats precomputed trajectory for Stage 2 prompt
  compute_traj_signal()       — classifies trajectory signal strength for analysis
"""

import re
import sys
from pathlib import Path

# ── Re-export from parent parse_score.py (unchanged) ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from parse_score import parse_response, score_row  # noqa: F401 — intentional re-export


# ── Hedge classifier ──────────────────────────────────────────────────────────
# Ported directly from _classify_conf() in visualize_v4_enhanced.py (lines 484-495)
# so hedge labels are identical across old and new experiments.

def classify_hedge(raw: str) -> str:
    """
    Returns: 'Confident Yes' | 'Uncertain Yes' | 'Hedged' | 'Confident No'
    Matches _classify_conf in visualize_v4_enhanced.py exactly.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "Hedged"
    s = raw.lower().strip()

    if s in ("yes", "true", "valid"):
        return "Confident Yes"
    if s in ("no", "false", "invalid"):
        return "Confident No"
    if s.startswith("yes"):
        return "Uncertain Yes"
    if s.startswith("no"):
        return "Confident No"
    if (re.search(r"\bnot\b.{0,20}\bvalid\b", s)
            or "not fully valid" in s
            or "not physically valid" in s):
        return "Confident No"
    if re.search(r"\bphysically valid\b|\bvalid simulation\b|\bvalid approach\b|\bgenerally valid\b", s):
        return "Uncertain Yes"
    if re.search(r"\b(unclear|cannot determine|uncertain|depends|potentially|possibly|might be|may be)\b", s):
        return "Hedged"
    if re.search(r"\byes\b|\bvalid\b|\bcorrect\b", s):
        return "Uncertain Yes"
    if re.search(r"\bno\b|\binvalid\b|\bincorrect\b", s):
        return "Confident No"
    return "Hedged"


# ── Trajectory signal classifier ──────────────────────────────────────────────

def compute_traj_signal(traj: dict, gt_valid: bool) -> str:
    """
    Classify how much information the trajectory gives about validity.

    clear_invalid   — trajectory has obvious numerical anomaly (NaN/Inf/spike)
                      and the script is actually invalid
    ambiguous_invalid — script is invalid but trajectory looks clean
                        (subtle bugs: sign flips, bounded instability)
    clear_valid     — script is valid, trajectory looks clean (expected)
    """
    has_nan   = traj.get("has_nan") or False
    has_inf   = traj.get("has_inf") or False
    spike     = traj.get("spike_ratio") or 0.0
    final_abs = traj.get("max_abs_final") or 0.0
    obvious   = has_nan or has_inf or spike > 1e3 or final_abs > 1e6

    if gt_valid:
        return "clear_valid"
    return "clear_invalid" if obvious else "ambiguous_invalid"


# ── Trajectory block formatter ────────────────────────────────────────────────

def format_trajectory_block(traj: dict) -> str:
    """
    Format trajectory dict as a plain key: value block for the Stage 2 prompt.

    Design decisions:
    - tracked_variable uses the name exactly as it appears in the executed code.
      For CorrVar scripts this is the obfuscated name (e.g. foobar_23), not the
      physical name — no leakage of variable semantics beyond what the code shows.
    - Inline comments clarify the 4 non-obvious fields without interpreting
      what the values mean for validity.
    - large_spike_detected is omitted — it pre-summarises the verdict.
    """
    lines: list[str] = []

    lines.append(f"ran_to_completion: {str(traj.get('ran_to_completion', False)).lower()}")

    # Use name exactly as it appears in the code (may be obfuscated).
    arr_name = traj.get("main_array_name")
    shape    = traj.get("shape")
    if arr_name and shape:
        lines.append(
            f"tracked_variable: {arr_name}  (shape: {shape})"
            f"  # name as used in the code above"
        )
    elif arr_name:
        lines.append(f"tracked_variable: {arr_name}  # name as used in the code above")

    for field in ("has_nan", "has_inf"):
        v = traj.get(field)
        if v is not None:
            lines.append(f"{field}: {str(v).lower()}")

    v = traj.get("finite_fraction")
    if v is not None:
        lines.append(f"finite_fraction: {v}")

    v = traj.get("first_nonfinite_time_index")
    if v is not None:
        lines.append(
            f"first_nonfinite_time_index: {v}"
            f"  # time-axis index where NaN or Inf first appears"
        )

    for field in ("max_abs_initial", "max_abs_final"):
        v = traj.get(field)
        if v is not None:
            lines.append(f"{field}: {v}")

    sampled = traj.get("max_abs_over_time_sampled")
    if sampled:
        lines.append(f"max_abs_over_time_sampled: {sampled}")

    v = traj.get("max_abs_before_nan")
    if v is not None:
        lines.append(
            f"max_abs_before_nan: {v}"
            f"  # max absolute value at the last all-finite time step"
        )

    v = traj.get("spike_ratio")
    if v is not None:
        lines.append(
            f"spike_ratio: {v}"
            f"  # max_abs_final / max_abs_initial"
        )

    return "\n".join(lines)

