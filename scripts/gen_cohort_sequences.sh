#!/bin/bash
# ── Build QuerySeqSchema tasks for a supplied cohort ───────────
#
# Turns a parquet of (subject_id, prediction_time) rows into labeled query sequences that
# EQ_predict_sequences can score.  Edit the knobs below, then:
#
#   bash scripts/gen_cohort_sequences.sh
#   EQ_predict_sequences model_run_dir="$MODEL_DIR" tasks_dir="$OUT_DIR" \
#     output_parquet=predictions.parquet split="$SPLIT"
#
# ───────────────────────────────────────────────────────────────

set -euo pipefail

# Both overridable from the environment: the defaults name one specific archive, which is a
# footgun for anyone else running this template unedited.
ARCHIVE=${ARCHIVE:-/experiments/EQ_conditional_experiments}
COHORT_DIR=${COHORT_DIR:-"$ARCHIVE/data/tensorized_cohort"}   # holds data/<split>/ and metadata/

CONTEXTS_PARQUET=cohort.parquet                # <-- your index df
OUT_DIR=cohort_tasks                           # -> $OUT_DIR/$SPLIT/<stem>__0000.parquet
SPLIT=held_out                                 # must contain your subjects; train is rejected by predict

N_REPLICATES=1                                 # sequences sampled per supplied row
MIN_QUERIES=5                                  # K (min==max => fixed length); 5 matches big_v2 training
MAX_QUERIES=5
DURATION_MIN=1
DURATION_MAX=365
SEED=1

cd "$(git rev-parse --show-toplevel)"

uv run EQ_generate_query_sequences \
    contexts_path="$CONTEXTS_PARQUET" \
    n_replicates="$N_REPLICATES" \
    data_dir="$COHORT_DIR" \
    query_codes="$COHORT_DIR" \
    out_dir="$OUT_DIR" \
    split="$SPLIT" \
    min_queries="$MIN_QUERIES" \
    max_queries="$MAX_QUERIES" \
    duration_min="$DURATION_MIN" \
    duration_max="$DURATION_MAX" \
    seed="$SEED"
