#!/usr/bin/env bash
# Source env.sh and report which path vars resolve, WITHOUT echoing the paths.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source ./env.sh
PYTHONPATH="${REPO_ROOT}/src" /home/gkondas/EveryQuery-conditional/.venv/bin/python \
    scripts/new_features/probe_env.py
