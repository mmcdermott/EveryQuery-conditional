#!/usr/bin/env bash
#
# Step 3 — write the designed evaluation specs (20 tasks x 3 query types, at length 1 and 3).
# See 03_make_eval_specs.py for the sampling rules.  Prints aggregates only, never code strings.

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

export NF_SPEC_DIR="${NF_ROOT}/eval_specs"
"$EQ_PY" scripts/new_features/03_make_eval_specs.py --out-dir "$NF_SPEC_DIR" "$@"
