"""
Unit test for episode_log.py -- renders a synthetic result row and confirms
the log text contains the expected sections/fields. Zero network calls.
"""
import shutil
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from frontier.episode_log import render_episode_log, write_episode_log

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


ROW = {
    "model": "gemini-2.5-flash",
    "title": "_test_episode_log_row",
    "gt_valid": True, "gt_pde": "heat", "gt_method": "explicit", "gt_behavior": "diffusion",
    "episode_dir": "/tmp/fake_episode_dir",
    "total_cost_usd": 0.01234,
    "protocol_violations": 0,
    "cost_guard_tripped": False,
    "s1_response": "pde: heat\nvalid: yes",
    "s1_parsed_pde": "heat", "s1_parsed_method": "explicit",
    "s1_parsed_behavior": "diffusion", "s1_parsed_valid": "yes",
    "s1_valid_match": 1, "s1_pde_match": 1,
    "s1_pde_exp": "exp1", "s1_method_exp": "exp2", "s1_behavior_exp": "exp3", "s1_valid_exp": "exp4",
    "action_trace": [
        {
            "turn": 1, "tool": "edit_source", "args": {"full_source": "import numpy\n"},
            "result": "stdout='ok'", "thought_summary": "thinking about it",
            "reasoning_text": "let me check something",
        },
        {
            "turn": 2, "tool": "submit_final_answer",
            "args": {"pde": "heat", "valid": "yes"}, "result": None,
            "thought_summary": "final thoughts", "reasoning_text": "pde: heat\nvalid: yes",
        },
    ],
    "s2_submit_args": {"pde": "heat", "valid": "yes"},
    "s2_valid_match": 1, "s2_pde_match": 1,
    "s2_action_count": 1, "actions_remaining_at_submission": 5,
    "s2_tools_used": ["edit_source"], "s2_used_edit_source": True,
}


print("\n── render_episode_log ──")

text = render_episode_log(ROW)

check("includes the title", "_test_episode_log_row" in text)
check("includes ground truth line", "valid=True  pde=heat" in text)
check("includes Stage 1 section header", "STAGE 1 (read-only, no tools)" in text)
check("includes Stage 1 response text", "pde: heat\nvalid: yes" in text)
check("includes Stage 2 section header", "STAGE 2 (agentic loop)" in text)
check("includes turn 1's thought_summary", "thinking about it" in text)
check("includes turn 1's reasoning_text", "let me check something" in text)
check("includes turn 2's tool name in the turn header", "Turn 2 (tool=submit_final_answer)" in text)
check("includes the final submitted answer block", '"pde": "heat"' in text and "FINAL SUBMITTED ANSWER" in text)
check("includes cost", "$0.01234" in text)


print("\n── write_episode_log ──")

tmp_dir = Path("/tmp/_test_episode_log_write_dir")
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)

path = write_episode_log(ROW, tmp_dir, "unittest_run")
check("returns the expected path", path == tmp_dir / "logs" / f"{ROW['title']}__unittest_run.txt", str(path))
check("file actually exists", path.exists())
check("file content matches render_episode_log's output", path.read_text() == render_episode_log(ROW))

shutil.rmtree(tmp_dir)


print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
