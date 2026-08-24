#!/bin/bash
# Hand Nemotron a FULL wall instead of xg_redraw's leftover.
#
# xg_redraw works a 6-model roster and Nemotron is last. By the time it gets there it
# has ~4h of wall left against ~7-8h of work (734 draws, 689 of them decode loops that
# each run to the cap), so it would be killed mid-roster. Same one GPU either way --
# this only moves the work to a job that can actually finish it.
#
# The wait is not optional: starting a second Nemotron job while xg_redraw is still
# alive would put two writers on nemotron-3-nano-30b-backfill2's JSONL, interleaving
# rows and redrawing the same draws twice. So we wait for Qwen3.6's merge line, which
# is the last thing xg_redraw does before opening Nemotron, and only then swap.
set -uo pipefail
LOG=/scratch/ehb7466/projects/pde-llm-eval/logs/backfill_16211015.out
S=/scratch/ehb7466/projects/pde-llm-eval/outputs/xmodal_gen
DEADLINE=$(( $(date +%s) + 5400 ))

while true; do
    if ! squeue -h -j 16211015 -o '%T' 2>/dev/null | grep -q RUNNING; then
        echo "[swap] xg_redraw is no longer running; proceeding to submit Nemotron"
        break
    fi
    if grep -aq 'merged arm ->.*qwen3-6-27b-backfill2' "$LOG" 2>/dev/null; then
        echo "[swap] Qwen3.6 merged at $(date -u '+%H:%M UTC') -- cancelling xg_redraw before it opens Nemotron"
        scancel 16211015
        sleep 20
        break
    fi
    # Safety: if Nemotron's arm has ROWS, xg_redraw already opened it. Testing for the
    # directory alone is not enough -- an empty leftover dir from an aborted attempt
    # was already sitting there at 22:46 and would have false-aborted this swap. and a swap would
    # now be the very race this script exists to avoid. Abort rather than double-write.
    if [ -s "$(ls "$S"/nemotron-3-nano-30b-backfill2/*.jsonl 2>/dev/null | head -1)" ] 2>/dev/null; then
        echo "[swap] ABORT -- xg_redraw already started Nemotron; leaving it alone"
        exit 0
    fi
    [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[swap] ABORT -- 90 min elapsed, Qwen3.6 never merged"; exit 1; }
    sleep 120
done

cd /home/ehb7466/pde-llm-eval || exit 1
sbatch --job-name=xg_redraw \
    --partition=h200_courant --account=torch_pr_427_courant \
    --export=ALL,BF_ROSTER="nemotron-3-nano-30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|-backfill",SRC_SUFFIX="-backfill",OUT_SUFFIX="-backfill2",REDRAW_LOOPS="1" \
    sbatch/backfill_no_verdict.sbatch
echo "[swap] submitted at $(date -u '+%Y-%m-%d %H:%M UTC')"
