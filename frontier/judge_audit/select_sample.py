"""
select_sample.py — draw the human-audit sample for the §4.2 judge.

5 judged rows per thinking setting (4 settings, 20 total), uniformly at random
without replacement, independent of the judge's labels. The population is every
row the judge rated: gt_valid=False and the model's Stage 2 answer said invalid
(s2_valid_match=1) -- 121 / 120 / 123 / 124 rows.

The seed is fixed before drawing and the draw is accepted as-is: this script
refuses to run if a manifest already exists, so a sample cannot be quietly
redrawn. Rows are sorted by title first so the draw does not depend on file order.

Writes (results/frontier/judge_audit/private/, never shared):
  judge_audit_key.csv    item_id -> setting, title, judge labels and explanation
  sample_manifest.json   seed, git SHA, population sizes, SHA-256 of every input

Usage:
  python frontier/judge_audit/select_sample.py --seed 20261021
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "frontier"))

from upload_traces import ARMS, arm_files, load_all  # noqa: E402

OUT_ROOT = REPO_ROOT / "results" / "frontier" / "judge_audit"
PRIVATE = OUT_ROOT / "private"
PER_SETTING = 5

FAMILY = {
    "gemini25flash_tb0": "gemini-2.5-flash", "gemini25flash_tb8192": "gemini-2.5-flash",
    "gemini37flash_low": "gemini-3.7-flash", "gemini37flash_high": "gemini-3.7-flash",
}
SETTING_LABEL = {
    "gemini25flash_tb0": "thinking budget 0", "gemini25flash_tb8192": "thinking budget 8192",
    "gemini37flash_low": "thinking level low", "gemini37flash_high": "thinking level high",
}
KEY_COLUMNS = [
    "item_id", "model_family", "setting", "setting_label", "title", "mod_type", "pde_class",
    "judge_category", "judge_incorrect_code_claims", "judge_unverifiable_execution_claims",
    "judge_explanation", "judge_model",
]


def dataset_csv(setting: str) -> Path:
    """The dataset file the run itself used (identical rows to data/merged_mod_jul28.csv)."""
    return REPO_ROOT / "results" / "frontier" / ARMS[setting][0] / "dataset256_balanced.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def draw(seed: int) -> tuple[list[dict], dict[str, int]]:
    """Returns (key rows in shuffled presentation order, population size per setting)."""
    _, raw, _ = load_all()
    rng = random.Random(seed)
    picked, population = [], {}
    for setting in ARMS:  # fixed order: 2.5 tb0, 2.5 tb8192, 3.7 low, 3.7 high
        judged = sorted(raw[setting]["judge"], key=lambda r: r["title"])
        population[setting] = len(judged)
        for j in rng.sample(judged, PER_SETTING):
            picked.append({
                "model_family": FAMILY[setting],
                "setting": setting,
                "setting_label": SETTING_LABEL[setting],
                "title": j["title"],
                "mod_type": j["mod_type"],
                "pde_class": j["pde_class"],
                "judge_category": j["category"],
                "judge_incorrect_code_claims": bool(j["contains_incorrect_code_claims"]),
                "judge_unverifiable_execution_claims": bool(j["contains_unverifiable_execution_claims"]),
                "judge_explanation": j["explanation"],
                "judge_model": j["judge_model"],
            })
    rng.shuffle(picked)
    for i, row in enumerate(picked, 1):
        row["item_id"] = f"A{i:02d}"
    return picked, population


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True, help="Fixed before drawing; recorded in the manifest")
    args = p.parse_args()

    manifest_path = PRIVATE / "sample_manifest.json"
    if manifest_path.exists():
        sys.exit(f"{manifest_path} exists -- the sample has already been drawn. Not redrawing.")

    rows, population = draw(args.seed)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / "judge_audit_key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEY_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    inputs = {}
    for setting in ARMS:
        files = arm_files(setting)
        for name in ("episodes", "judge"):
            inputs[f"{setting}/{name}"] = {"path": str(files[name].relative_to(REPO_ROOT)), "sha256": sha256(files[name])}
        inputs[f"{setting}/dataset"] = {"path": str(dataset_csv(setting).relative_to(REPO_ROOT)),
                                        "sha256": sha256(dataset_csv(setting))}
    manifest = {
        "seed": args.seed,
        "per_setting": PER_SETTING,
        "population": population,
        "drawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "inputs": inputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    per = {s: sum(r["setting"] == s for r in rows) for s in ARMS}
    print(f"seed {args.seed}: drew {len(rows)} items {per} from populations {population}")
    print(f"wrote {PRIVATE / 'judge_audit_key.csv'} and {manifest_path}")


if __name__ == "__main__":
    main()
