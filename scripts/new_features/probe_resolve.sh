#!/usr/bin/env bash
# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" > /dev/null
PYTHONPATH="scripts/new_features:${PYTHONPATH}" "$EQ_PY" scripts/new_features/probe_resolve.py
