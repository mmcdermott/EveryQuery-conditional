#!/usr/bin/env bash
# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" > /dev/null
"$EQ_PY" scripts/new_features/11_compare_lengths.py "$NF_METRICS_DIR" "$@"
