#!/bin/bash
# Rebuild the claim-driven consistency report from the published results.
set -euo pipefail
PY="${PY:-/Users/bermaneh/raca-tools-venv/bin/python}"
cd "$(dirname "$0")/.."
"$PY" - <<'EOF'
import pandas as pd
from viz.consistency.adapter import load_real
from viz.consistency import claim_report as CR
# PINNED to the frozen repo. load_real() now defaults to BOTH the published repo and
# the generational one; calling it bare here would fold the new models into every
# pooled panel of consistency_claims.html and then copy that over the notes-folder
# copy two lines below -- silently rewriting the published report. The generational
# roster has its own figure and must never enter this one.
d = load_real(repo="bermaneh/pde-llm-eval-xmodal-consistency")
items = pd.read_csv("data/multimodal_items_v1.csv").drop_duplicates("gt_sample")
defects = dict(zip(items["gt_sample"].astype(str), items["invalidity_note"].astype(str)))
CR.build(d, out="viz/consistency_claims.html", defects=defects)
EOF
cp viz/consistency_claims.html \
   /Users/bermaneh/Desktop/raca/notes/experiments/pde-llm-eval/consistency_claims.html
echo "[claims] copied to the experiment folder"
