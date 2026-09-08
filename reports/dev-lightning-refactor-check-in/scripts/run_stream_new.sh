#!/usr/bin/env bash
# Stream A: the dev code on every evaluation set.  The two tuning grids first (they have stored
# 2026-09-05 baselines), each size under the new default precision and under fp32; then the
# never-scored held_out grids under the new default.  Each stage is logged under logs/.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
S="${EXP}/scripts"; L="${EXP}/logs"
run() {
    "${S}/10_predict.sh" "$@" > "${L}/10_predict_$1_$2_$3_$4.log" 2>&1
    "${BIN_NEW}/python" "${S}/20_analyze.py" --variant "$1" --grid "$2" --split "$3" --size "$4" \
        > "${L}/20_analyze_$1_$2_$3_$4.log" 2>&1
    echo "=== $(date -Is) done $1 $2 $3 $4"
}
for grid in canonical horizon; do
    for size in small base large; do
        run new_bf16 "${grid}" tuning "${size}"
        run new_fp32 "${grid}" tuning "${size}"
    done
done
for grid in canonical horizon; do
    for size in small base large; do
        run new_bf16 "${grid}" held_out "${size}"
    done
done
echo "=== $(date -Is) stream_new done"
