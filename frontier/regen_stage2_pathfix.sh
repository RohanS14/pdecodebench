#!/usr/bin/env bash
# Commands for regenerating Stage 2 (gemini-3.7-flash @ thinking_level=high, 256 rows)
# under the path-leak fix. Saved for reference -- run the steps by hand.
#
# The original run (results/frontier/score_256_37) named each episode's working dir
# after the row title, which contains the label ("_InValid_1"). That dir is the
# subprocess cwd, so Python printed it in warnings and tracebacks and the harness
# returned it to the model. Fixed via opaque episode_slug() dirs + stdout/stderr
# scrubbing in agentic_sandbox.py.
#
# Stage 1 is reused: it is read-only, runs no subprocess, and never saw a path, so
# its cache is untainted. Saves ~$2.50 and ~1h. Stage 2 costs ~$15, ~7h.
#
# Every generation parameter below matches the original run so the regeneration
# isolates the path fix. Don't change one without re-fingerprinting the prompts.

cd "$(dirname "$0")/.."   # repo root

NEW=results/frontier/score_256_37_pathfix
OLD=results/frontier/score_256_37
# The 256-row balanced sample the original run used (identical across all four
# score_256_* arms). Override to point at your copy.
DATASET_CSV=${DATASET_CSV:-results/frontier/score_256_25_tb0/dataset256_balanced.csv}

# --- setup: fresh root, reuse Stage-1 cache, copy the dataset into the run root
mkdir -p $NEW/gemini37flash/tl_high/stage1_cache
cp $OLD/gemini37flash/tl_high/stage1_cache/gemini37flash__stage1_cache.jsonl \
   $NEW/gemini37flash/tl_high/stage1_cache/
cp "$DATASET_CSV" $NEW/dataset256_balanced.csv

# expect 256, and no Stage-2 file yet
wc -l < $NEW/gemini37flash/tl_high/stage1_cache/gemini37flash__stage1_cache.jsonl
ls $NEW/gemini37flash/tl_high/

# --- run Stage 2 (background; resumable by re-issuing -- checkpoint is keyed by title)
.venv/bin/python frontier/run_stratified_sweep.py \
  --model gemini-3.7-flash \
  --thinking_settings high \
  --dataset $NEW/dataset256_balanced.csv \
  --output_root $NEW \
  --run_id_prefix score256_37_pathfix \
  --budget 6 \
  --truncate_chars 4000 \
  --subprocess_timeout 120 \
  --rpm 8.0 \
  --max_cost_usd_stage1 1.0 \
  --max_cost_usd_per_condition 25.0 \
  2>&1 | tee $NEW/run.log

# Cost args are ceilings, not targets: the original Stage 2 totalled $15.39.
# max_cost_usd_stage1 should never engage (all 256 rows cached) -- it's set low so a
# missing cache fails cheaply instead of silently re-running Stage 1.
