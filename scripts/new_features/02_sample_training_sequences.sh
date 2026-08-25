#!/usr/bin/env bash
#
# Step 2 — sample + label the TRAINING query sequences, with ALL new features on.
#
# Features exercised here (two of the three; rope-time is a model/dataset-side flag only):
#   * event-bounded durations   -> eventbound_fraction=0.5
#   * DAG-aware queries         -> ontology_dir=$NF_ONTOLOGY_DIR
#     (puts every ancestor NODE into the query universe, as a query AND as an event boundary,
#      and explodes the event stream through the closure so an ancestor query is labelled by
#      ordinary occurrence of any descendant)
#
# Usage:  bash scripts/new_features/02_sample_training_sequences.sh <split> <num_sequences> [seed]

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

SPLIT="${1:?usage: 02_sample_training_sequences.sh <split> <num_sequences> [seed]}"
NUM_SEQ="${2:?usage: 02_sample_training_sequences.sh <split> <num_sequences> [seed]}"
SEED="${3:-1}"

LOG="${NF_LOG_DIR}/02_sample_train_${SPLIT}.log"
echo "split=${SPLIT} num_sequences=${NUM_SEQ} seed=${SEED}"
echo "out_dir : $NF_TRAIN_TASKS_DIR"
echo "log     : $LOG"

START=$SECONDS

"$EQ_PY" -m every_query.generate_tasks.sample_query_sequences \
    data_dir="$TOKENIZED_EVENTS_DIR" \
    out_dir="$NF_TRAIN_TASKS_DIR" \
    query_codes="$TENSORIZED_COHORT_DIR" \
    ontology_dir="$NF_ONTOLOGY_DIR" \
    split="$SPLIT" \
    seed="$SEED" \
    num_sequences="$NUM_SEQ" \
    min_queries=1 \
    max_queries=5 \
    duration_min=1 \
    duration_max=731 \
    duration_distribution=log-uniform \
    min_prediction_times_per_subject=50 \
    eos_first_fraction=0.0 \
    duration_mode=random \
    eventbound_fraction=0.5 \
    overwrite=true \
    > "$LOG" 2>&1

echo "elapsed: $((SECONDS - START))s"
echo
echo "--- feature evidence (Feature 2 + Feature 3) ---"
grep -E 'Event bounds:|query universe is|contributes no usable|Ontology' "$LOG" | head -20 || true
echo
echo "--- errors ---"
grep -E 'Error|Traceback|CRITICAL' "$LOG" | head -20 || true
