"""Migrate pre-v6 free-generation results to one file per reasoning arm.

Pre-v6, results were one file per model (`{slug}.jsonl`) and no row recorded which
reasoning arm produced it. v6 writes `{slug}__think-{arm}.jsonl` with a `thinking`
column. This backfills the column into legacy files and renames them.

WHY THIS IS DANGEROUS AND WHY IT DEFAULTS TO A DRY RUN
------------------------------------------------------
`run_eval.append_result` opens the results file BY PATH on every single write. If a
job is appending while this script renames the file, the next append recreates the
old path and the run's output silently splits across two files, with the checkpoint
seeing only one of them. So this script refuses to touch any file that looks live,
and it does nothing at all without --apply.

Note that migrating is OPTIONAL: run_eval.py resumes from the legacy filename when
the arm matches, so a wall-timeout resubmit is already safe without running this.
Migrate for tidiness, once the jobs are done -- never to unblock a running one.

Usage:
    python freegen/migrate_arm_filenames.py --results_dir /path/to/outputs
    python freegen/migrate_arm_filenames.py --results_dir /path --apply
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import ALWAYS_THINKING_MODELS, legacy_arm  # noqa: E402

# A file touched more recently than this is assumed to be under active append.
LIVE_WINDOW_S = 30 * 60


def squeue_running():
    """Job names/ids currently queued or running, or None if squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                              "-o", "%i %T %j"], capture_output=True, text=True,
                             timeout=20)
        if out.returncode != 0:
            return None
        return [l for l in out.stdout.splitlines() if l.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def model_of(path):
    """Read the model id out of the file's first row."""
    with open(path) as f:
        for line in f:
            if line.strip():
                return json.loads(line).get("model")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually migrate. Without this, only report the plan.")
    ap.add_argument("--force", action="store_true",
                    help="Migrate even files that look actively written. Do not use "
                         "this while any free-gen job is in the queue.")
    a = ap.parse_args()

    jobs = squeue_running()
    if jobs:
        print(f"[migrate] WARNING: {len(jobs)} job(s) in your queue:")
        for j in jobs:
            print(f"[migrate]   {j}")
        print("[migrate] A running free-gen job appends by path; migrating its file "
              "splits the output. Wait for it to finish.")
    elif jobs is None:
        print("[migrate] note: squeue unavailable here, cannot check for live jobs.")

    legacy = [p for p in sorted(glob.glob(os.path.join(a.results_dir, "*.jsonl")))
              if "__think-" not in os.path.basename(p)]
    if not legacy:
        print(f"[migrate] nothing to do: no pre-v6 files in {a.results_dir}")
        return

    now = time.time()
    plan, skipped = [], []
    for path in legacy:
        model = model_of(path)
        if not model:
            skipped.append((path, "empty file / no model field"))
            continue
        arm = legacy_arm(model)
        dest = path[:-len(".jsonl")] + f"__think-{arm}.jsonl"
        age = now - os.path.getmtime(path)
        if age < LIVE_WINDOW_S and not a.force:
            skipped.append((path, f"modified {age/60:.1f} min ago — looks live"))
            continue
        if os.path.exists(dest):
            skipped.append((path, f"destination already exists: {os.path.basename(dest)}"))
            continue
        plan.append((path, dest, model, arm))

    print(f"\n[migrate] {len(plan)} file(s) to migrate, {len(skipped)} skipped\n")
    for path, dest, model, arm in plan:
        print(f"  {os.path.basename(path)}")
        print(f"    -> {os.path.basename(dest)}   (thinking={arm}, {model})")
    for path, why in skipped:
        print(f"  SKIP {os.path.basename(path)}: {why}")

    if not a.apply:
        print("\n[migrate] dry run — nothing written. Re-run with --apply.")
        return
    if not plan:
        print("\n[migrate] nothing to apply.")
        return

    for path, dest, model, arm in plan:
        # Write the backfilled copy first, fsync it, and only then drop the original,
        # so an interruption leaves the original intact rather than a half file.
        tmp = dest + ".partial"
        n = 0
        with open(path) as src, open(tmp, "w") as out:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("thinking", arm)
                out.write(json.dumps(row) + "\n")
                n += 1
            out.flush()
            os.fsync(out.fileno())
        shutil.move(tmp, dest)
        backup = path + ".pre-v6"
        shutil.move(path, backup)
        print(f"[migrate] {os.path.basename(dest)}: {n} rows, thinking={arm} "
              f"(original kept as {os.path.basename(backup)})")

    print(f"\n[migrate] done. Originals kept with a .pre-v6 suffix; delete them only "
          f"after the migrated files have been verified and uploaded.")


if __name__ == "__main__":
    main()
