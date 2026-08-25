#!/usr/bin/env bash
# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$EQ_PY" scripts/new_features/verify_labels.py "$@"
