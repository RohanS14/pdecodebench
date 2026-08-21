#!/bin/bash
# Rebuild the dual-experiment report from the published artifacts.
# Safe to re-run at any time: it reads whatever rows exist right now, and panels
# whose data has not landed render as placeholders naming the job that produces them.
set -euo pipefail
PY="${PY:-/Users/bermaneh/Desktop/raca/.tools-venv/bin/python}"
REPO_DIR="${REPO_DIR:-/Users/bermaneh/Desktop/raca/private_projects/pde-llm-eval}"
NOTES_DIR="${NOTES_DIR:-/Users/bermaneh/Desktop/raca/notes/experiments/pde-llm-eval}"
cd "$REPO_DIR"

# The d' panel needs the clustered bootstrap; skip it silently if no rows yet.
if compgen -G "results/xmodal/*.jsonl" > /dev/null; then
    "$PY" crossmodal/eval/aggregate_cross_modal.py --results_dir results/xmodal \
        --out results/xmodal_summary.json --csv_out results/xmodal_summary.csv || \
        echo "[refresh] aggregate failed; report will show the d' panel as pending"
fi

"$PY" viz/pde_dual_report.py \
    --freegen_hf bermaneh/pde-llm-eval-results-jul28 \
    --xmodal_hf  bermaneh/pde-llm-eval-xmodal-consistency \
    --xmodal_summary results/xmodal_summary.json \
    --out viz/pde_dual_report.html

cp viz/pde_dual_report.html "$NOTES_DIR/pde_dual_report.html"
echo "[refresh] report rebuilt and copied to the experiment folder"
