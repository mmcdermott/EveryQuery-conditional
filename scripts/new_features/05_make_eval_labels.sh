#!/usr/bin/env bash
#
# Step 5 — label the designed evaluation sequences on the HELD_OUT split.
#
# Runs once per sequence length.  Both runs use the SAME seed / split / n_contexts, and
# `sample_grid_contexts` seeds on `derive_seed(seed, "contexts")`, so the two grids are scored at
# identical contexts -- which is what makes the length-1 vs length-3 comparison meaningful.
#
# ontology_dir is REQUIRED here: without it an ancestor query is either rejected outright or
# (for a dual-role name) labelled exact-match-only -- a well-formed parquet full of wrong labels.
#
# Usage:  bash scripts/new_features/05_make_eval_labels.sh <len1|len3> [n_contexts]

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

WHICH="${1:?usage: 05_make_eval_labels.sh <len1|len3> [n_contexts]}"
N_CONTEXTS="${2:-4000}"
SPEC_DIR="${NF_ROOT}/eval_specs"
LOG="${NF_LOG_DIR}/05_eval_labels_${WHICH}.log"

echo "which      : $WHICH"
echo "n_contexts : $N_CONTEXTS"
echo "specs      : ${SPEC_DIR}/designed_${WHICH}.yaml"
echo "out_dir    : ${NF_EVAL_TASKS_DIR}/${WHICH}"
echo "log        : $LOG"

START=$SECONDS
STATUS=0

"$EQ_PY" -m every_query.generate_tasks.sample_evaluation_query_sequences \
    data_dir="$TOKENIZED_EVENTS_DIR" \
    out_dir="${NF_EVAL_TASKS_DIR}/${WHICH}" \
    query_codes="$TENSORIZED_COHORT_DIR" \
    ontology_dir="$NF_ONTOLOGY_DIR" \
    split=held_out \
    seed=1 \
    n_contexts="$N_CONTEXTS" \
    min_prediction_times_per_subject=50 \
    sequences_path="${SPEC_DIR}/designed_${WHICH}.yaml" \
    per_spec_dirs=false \
    duration_min=1 \
    duration_max=731 \
    duration_distribution=log-uniform \
    overwrite=true \
    > "$LOG" 2>&1 || STATUS=$?

echo "elapsed: $((SECONDS - START))s   exit=$STATUS"
echo
echo "--- summary ---"
grep -E 'Ontology|contexts|Wrote|complete|Event bounds:' "$LOG" | grep -v 'never fires' | tail -15 || true
echo
echo "--- errors ---"
grep -E 'Error|Traceback|CRITICAL' "$LOG" | head -20 || true
echo
echo "--- outputs ---"
find "${NF_EVAL_TASKS_DIR}/${WHICH}" -name '*.parquet' 2>/dev/null | head
exit $STATUS
