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
    "$PY" cross_modal_consistency/eval/aggregate_cross_modal.py --results_dir results/xmodal \
        --out results/xmodal_summary.json --csv_out results/xmodal_summary.csv || \
        echo "[refresh] aggregate failed; report will show the d' panel as pending"
fi

# Experiment 1 now runs the SAME eight checkpoints as Experiment 2 Part III, at the
# same decoding (0.6 / 0.95 / 20, k=3, seed 20260821). That is the whole point of the
# document: two experiments over the same 32 base solvers are only comparable if they
# are also over the same models. The old --freegen_hf pde-llm-eval-results-jul28 was a
# different, larger, k=1 roster, so the two halves were describing different systems.
#
# Read from the local aggregate rather than one HF repo because the roster is
# published as one repo PER MODEL -- the launcher does that so concurrent jobs cannot
# clobber each other's uploads, since push_dataset_to_hub replaces the split. Rebuild
# the aggregate first if the arms have moved:
#   python freegen_static_judgments/aggregate_freegen.py --results_dir results/freegen_static_judgments \
#          --out results/freegen_static_judgments.csv --expect_items 256
FREEGEN_CSV="${FREEGEN_CSV:-results/freegen_static_judgments.csv}"

"$PY" viz/pde_dual_report.py \
    --freegen    "$FREEGEN_CSV" \
    --xmodal_hf  bermaneh/pde-llm-eval-cross-modal-consistency \
    --xmodal_summary results/xmodal_summary.json \
    --out viz/pde_dual_report.html

cp viz/pde_dual_report.html "$NOTES_DIR/pde_dual_report.html"
echo "[refresh] report rebuilt and copied to the experiment folder"
