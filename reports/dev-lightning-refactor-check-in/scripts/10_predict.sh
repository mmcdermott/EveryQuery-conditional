#!/usr/bin/env bash
# Score one stored checkpoint on one evaluation grid with one code variant.
# Usage: 10_predict.sh <variant> <grid> <split> <size>
#   variant  new_bf16   dev code (0b90790), EQ_predict_multitask defaults: Trainer.predict, precision=bf16-mixed
#            new_fp32   dev code, precision=32-true (the numerics of the pre-#30 manual fp32 loop)
#            old_rerun  de0f5ee code (the stored baseline's code) re-run today in an identical venv
#   grid     canonical | horizon      split  tuning | held_out      size  small | base | large
# Output: preds/<variant>/<grid>_<split>/<size>.parquet, one final-query probability per grid row.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
VARIANT=${1:?variant}; GRID=${2:?grid}; SPLIT=${3:?split}; SIZE=${4:?size}
case "$GRID" in
    canonical) TASKS="${OLD_EXP}/evalgrid/eval" ;;
    horizon)   TASKS="${OLD_EXP}/evalgrid_horizon/eval" ;;
    *) echo "unknown grid ${GRID}" >&2; exit 1 ;;
esac
# Optional 5th arg: an explicit run dir (the retrained small models of the training check); SIZE is then
# just the output name.
# shellcheck disable=SC2012
RUN_DIR=${5:-$(ls -d "${OLD_EXP}/runs/${SIZE}"/*/*/ | sort | tail -1)}
OUT="${EXP}/preds/${VARIANT}/${GRID}_${SPLIT}/${SIZE}.parquet"
mkdir -p "$(dirname "${OUT}")"
COMMON=(model_run_dir="${RUN_DIR}" tasks_dir="${TASKS}" output_parquet="${OUT}" split="${SPLIT}" batch_size=192 overwrite=true)
echo "=== $(date -Is) predict variant=${VARIANT} grid=${GRID} split=${SPLIT} size=${SIZE} run=${RUN_DIR}"
case "$VARIANT" in
    new_bf16)  "${BIN_NEW}/EQ_predict_multitask" "${COMMON[@]}" enable_progress_bar=false ;;
    new_fp32)  "${BIN_NEW}/EQ_predict_multitask" "${COMMON[@]}" precision=32-true enable_progress_bar=false ;;
    old_rerun) "${BIN_OLD}/EQ_predict_multitask" "${COMMON[@]}" ;;
    *) echo "unknown variant ${VARIANT}" >&2; exit 1 ;;
esac
echo "=== $(date -Is) predict ${VARIANT}/${GRID}_${SPLIT}/${SIZE} done"
