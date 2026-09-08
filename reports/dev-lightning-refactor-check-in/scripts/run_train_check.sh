#!/usr/bin/env bash
# The training-side check: small model, old code twice (noise floor) and dev code once, same seed;
# every retrained checkpoint is then scored on the canonical tuning grid through the dev CLI at its
# default precision and analyzed like the rest.  Runs sequentially; ~10 min per training on the GB10.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
S="${EXP}/scripts"; L="${EXP}/logs"
for spec in "old small_old_a" "old small_old_b" "new small_new_a"; do
    # shellcheck disable=SC2086
    set -- ${spec}
    "${S}/50_train_small.sh" "$1" "$2" > "${L}/50_train_$2.log" 2>&1
    # shellcheck disable=SC2012
    RUN_DIR=$(ls -d "${EXP}/runs/$2"/*/*/ | sort | tail -1)
    "${S}/10_predict.sh" new_bf16 canonical tuning "$2" "${RUN_DIR}" > "${L}/10_predict_new_bf16_canonical_tuning_$2.log" 2>&1
    "${BIN_NEW}/python" "${S}/20_analyze.py" --variant new_bf16 --grid canonical --split tuning --size "$2" \
        > "${L}/20_analyze_new_bf16_canonical_tuning_$2.log" 2>&1
    echo "=== $(date -Is) done $2"
done
"${BIN_NEW}/python" "${S}/60_train_check.py" > "${L}/60_train_check.log" 2>&1
echo "=== $(date -Is) train check done"
