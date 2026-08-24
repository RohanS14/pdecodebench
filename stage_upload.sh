#!/bin/bash
# Stage-and-upload arms that are still being written. A live JSONL can end in a torn
# line, and uploading that straight would either abort the helper or silently ship a
# truncated final row -- so each arm is snapshotted, its last line validated, and the
# upload runs from the snapshot instead of from under the writer.
set -uo pipefail
S=/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen
STAGE=/scratch/ehb7466/projects/pde-llm-eval/outputs/.upload_stage
PY=/scratch/ehb7466/envs/pdecodebench/bin/python
WORK=/home/ehb7466/pde-llm-eval
PKGS=/scratch/ehb7466/shared/raca-packages

for arm in "$@"; do
    src="$S/$arm"; dst="$STAGE/$arm"
    [ -d "$src" ] || { echo "== $arm  MISSING, skip"; continue; }
    rm -rf "$dst"; mkdir -p "$dst"; cp "$src"/*.jsonl "$dst"/ 2>/dev/null
    kept=$("$PY" - "$dst" <<'PYEOF'
import json, os, sys
d = sys.argv[1]; total = 0
for f in sorted(os.listdir(d)):
    if not f.endswith(".jsonl"):
        continue
    p = os.path.join(d, f)
    good = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except Exception:
            continue          # torn tail from the live writer
        good.append(line)
    with open(p, "w") as w:
        w.write("\n".join(good) + "\n")
    total += len(good)
print(total)
PYEOF
)
    echo "== $arm  $kept valid rows -> bermaneh/pde-llm-eval-xmodal-gen-$arm"
    "$PY" "$WORK/eval/upload_helper.py" \
        --results_dir "$dst" \
        --hf_dataset "bermaneh/pde-llm-eval-xmodal-gen-$arm" \
        --workspace "$WORK" --packages_dir "$PKGS" \
        --experiment pde-llm-eval --artifact_status partial \
        --job_id "torch:login" --cluster torch \
        --dataset_file multimodal_items_v1.csv 2>&1 | tail -3
done
