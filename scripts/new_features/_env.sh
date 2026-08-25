# shellcheck shell=bash
#
# Shared header for the new-feature (rope-time / event-bounded / DAG-aware) train+eval run.
# Source it, do not execute it.
#
# Two hazards inherited from scripts/experiments/_common.sh:
#   1. the venv lives in the MAIN checkout and is shared by every worktree;
#   2. the editable install's .pth names the MAIN checkout's src, so a worktree must put its own
#      src at the FRONT of PYTHONPATH or it silently runs the wrong branch's code.

set -euo pipefail

NF_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${NF_SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Path roots (TOKENIZED_EVENTS_DIR, TENSORIZED_COHORT_DIR, ...) come from env.sh.
# shellcheck source=/dev/null
source ./env.sh

# --- Run-specific outputs: everything this experiment writes lives under one root ----------
export NF_ROOT="${NF_ROOT:-${EQ_EXP_ROOT}/new_features_test}"
export NF_ONTOLOGY_DIR="${NF_ROOT}/ontology"
export NF_TRAIN_TASKS_DIR="${NF_ROOT}/train_sequences"
export NF_EVAL_TASKS_DIR="${NF_ROOT}/eval_sequences"
export NF_TRAIN_OUT_DIR="${NF_ROOT}/train_runs"
export NF_PRED_DIR="${NF_ROOT}/predictions"
export NF_METRICS_DIR="${NF_ROOT}/metrics"
export NF_LOG_DIR="${NF_ROOT}/logs"

mkdir -p "$NF_ROOT" "$NF_LOG_DIR" "$NF_PRED_DIR" "$NF_METRICS_DIR"

# --- W&B --------------------------------------------------------------------------------
export WANDB_PROJECT="EQ-conditional-new-features-test"
export WANDB_MODE="${WANDB_MODE:-online}"

# --- Interpreter / import path -----------------------------------------------------------
EQ_VENV="${EQ_VENV:-/home/gkondas/EveryQuery-conditional/.venv}"
export EQ_PY="${EQ_VENV}/bin/python"
if [[ ! -x "$EQ_PY" ]]; then
    echo "No interpreter at $EQ_PY. Set EQ_VENV to the shared venv." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Fail loudly rather than silently running the wrong branch.
ACTUAL_SRC="$("$EQ_PY" -c 'import every_query, pathlib; print(pathlib.Path(every_query.__file__).parent.parent)')"
if [[ "$ACTUAL_SRC" != "${REPO_ROOT}/src" ]]; then
    echo "every_query resolves to ${ACTUAL_SRC}, not ${REPO_ROOT}/src — refusing to run." >&2
    exit 1
fi

echo "checkout : $REPO_ROOT  ($(git rev-parse --short HEAD), $(git branch --show-current))"
echo "python   : $EQ_PY"
echo "nf_root  : $NF_ROOT"
