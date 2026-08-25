#!/usr/bin/env bash
#
# End-to-end: ontology -> training labels -> tiny all-features training -> held-out evaluation.
#
# Steps 01-03 are idempotent-ish but slow; skip them with SKIP_LABELS=1 once they have run.
#
# Usage:
#   bash scripts/new_features/run_all.sh
#   SKIP_LABELS=1 bash scripts/new_features/run_all.sh          # reuse existing labels/specs
#   TRAIN_MAX_TIME=00:00:16:00 N_CONTEXTS=4000 bash scripts/new_features/run_all.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_MAX_TIME="${TRAIN_MAX_TIME:-00:00:16:00}"
N_CONTEXTS="${N_CONTEXTS:-4000}"
N_TRAIN_SEQ="${N_TRAIN_SEQ:-300000}"
N_TUNE_SEQ="${N_TUNE_SEQ:-20000}"
RUN_NAME="${RUN_NAME:-cq-tiny-allfeat}"

if [[ -z "${SKIP_LABELS:-}" ]]; then
    bash "$HERE/01_build_ontology.sh"
    bash "$HERE/02_sample_training_sequences.sh" tuning "$N_TUNE_SEQ"
    bash "$HERE/02_sample_training_sequences.sh" train "$N_TRAIN_SEQ"
    bash "$HERE/03_make_eval_specs.sh"
fi

bash "$HERE/04_train.sh" "$TRAIN_MAX_TIME" "$RUN_NAME"

# The run dir is the timestamped ${output_dir}/<date>/<time> Hydra made, not output_dir itself.
RUN_DIR="$(bash "$HERE/find_run_dir.sh" "$RUN_NAME")"
echo "RUN_DIR=$RUN_DIR"

bash "$HERE/09_verify_run.sh" "$RUN_DIR"

bash "$HERE/05_make_eval_labels.sh" len1 "$N_CONTEXTS"
bash "$HERE/05_make_eval_labels.sh" len3 "$N_CONTEXTS"

bash "$HERE/06_predict.sh" "$RUN_DIR" len1
bash "$HERE/06_predict.sh" "$RUN_DIR" len3

bash "$HERE/08_score.sh"
