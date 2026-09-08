#!/usr/bin/env bash
# Stream B: the control.  Re-score the stored baseline (variant "stored") with this experiment's
# analysis code, then re-run the OLD code (de0f5ee) on the two tuning grids in a venv identical to
# the dev one (same uv.lock: torch 2.11), so environment drift and code change are separable.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
S="${EXP}/scripts"; L="${EXP}/logs"
for grid in canonical horizon; do
    for size in small base large; do
        "${BIN_NEW}/python" "${S}/20_analyze.py" --variant stored --grid "${grid}" --split tuning --size "${size}" \
            > "${L}/20_analyze_stored_${grid}_tuning_${size}.log" 2>&1
    done
done
echo "=== $(date -Is) stored baseline re-scored"
for grid in canonical horizon; do
    for size in small base large; do
        "${S}/10_predict.sh" old_rerun "${grid}" tuning "${size}" > "${L}/10_predict_old_rerun_${grid}_tuning_${size}.log" 2>&1
        "${BIN_NEW}/python" "${S}/20_analyze.py" --variant old_rerun --grid "${grid}" --split tuning --size "${size}" \
            > "${L}/20_analyze_old_rerun_${grid}_tuning_${size}.log" 2>&1
        echo "=== $(date -Is) done old_rerun ${grid} tuning ${size}"
    done
done
echo "=== $(date -Is) stream_old done"
