#!/usr/bin/env bash
#
# Step 8 — per-task and per-category macro AUROC.  See 07_score.py for what it computes and why
# the shipped EQ_evaluate_sequences is not enough.

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$EQ_PY" scripts/new_features/07_score.py \
    --pred-dir "$NF_PRED_DIR" \
    --spec-dir "${NF_ROOT}/eval_specs" \
    --out-dir "$NF_METRICS_DIR"
