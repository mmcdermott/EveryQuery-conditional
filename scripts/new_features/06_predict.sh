#!/usr/bin/env bash
#
# Step 6 — score the held-out evaluation grid with the trained checkpoint.
#
# EQ_predict_sequences is the ONLY correct scoring path for an all-features checkpoint: it calls
# `setup_model` + `instantiate(train_cfg.datamodule)`, which replays `strip_delta_tokens`,
# `use_rope_time` and BOTH `ontology_dir` keys verbatim from the run's resolved_config.yaml.
# There is no CLI key for any of those flags -- do not try to pass them.  (The older
# scripts/eval_v3.py hand-builds the dataset with no dataset_kwargs, so it cannot score this
# checkpoint at all: ancestor queries KeyError and rope-time raises on the missing time_pos_ids.)
#
# Inference is teacher-forced: position j is conditioned on the TRUE answers at positions < j.
#
# Usage:  bash scripts/new_features/06_predict.sh <run_dir> <len1|len3> [batch_size]

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

RUN_DIR="${1:?usage: 06_predict.sh <run_dir> <len1|len3> [batch_size]}"
WHICH="${2:?usage: 06_predict.sh <run_dir> <len1|len3> [batch_size]}"
BATCH="${3:-256}"

LOG="${NF_LOG_DIR}/06_predict_${WHICH}.log"
OUT="${NF_PRED_DIR}/preds_${WHICH}.parquet"

echo "run_dir : $RUN_DIR"
echo "tasks   : ${NF_EVAL_TASKS_DIR}/${WHICH}/held_out"
echo "out     : $OUT"
echo "log     : $LOG"

START=$SECONDS
STATUS=0

"$EQ_PY" -m every_query.predict.predict_sequences \
    model_run_dir="$RUN_DIR" \
    tasks_dir="${NF_EVAL_TASKS_DIR}/${WHICH}/held_out" \
    output_parquet="$OUT" \
    split=held_out \
    batch_size="$BATCH" \
    overwrite=true \
    > "$LOG" 2>&1 || STATUS=$?

echo "elapsed: $((SECONDS - START))s   exit=$STATUS"
echo
echo "--- errors ---"
grep -E 'Error|Traceback|CRITICAL' "$LOG" | head -20 || true
echo
ls -la "$OUT" 2>/dev/null || echo "NO OUTPUT"
exit $STATUS
