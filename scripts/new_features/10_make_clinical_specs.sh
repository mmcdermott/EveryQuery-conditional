#!/usr/bin/env bash
# Build the clinically meaningful ICU evaluation panel (length 1 and length 3).
# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" > /dev/null
PYTHONPATH="scripts/new_features:${PYTHONPATH}" "$EQ_PY" \
    scripts/new_features/10_make_clinical_specs.py --out-dir "${NF_ROOT}/eval_specs" "$@"
