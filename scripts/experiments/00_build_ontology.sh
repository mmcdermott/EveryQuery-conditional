#!/usr/bin/env bash
#
# Build the ontology artifacts for the tensorized cohort.
#
#   ontology_vocab.parquet         (node_name, token_id, is_observed_code)  — the extended vocabulary, V_ext
#   embedding_mix.parquet          (target_token_id, component_token_id, unnormalized_weight)  — the embedding mix
#   event_to_query_nodes.parquet   (event_code, query_node)  — what the labeller explodes events through
#
# Reads only $TENSORIZED_COHORT_DIR/metadata/codes.parquet; the cohort is never written.
#
# Expected on MIMIC-IV MEDS 0.2.0 (13,908 codes):
#   21,064 nodes = 13,908 leaves + 7,156 ancestors, of which 399 are dual-role subtree nodes
#   V_ext 21,065, 80,257 mix entries, 64,615 closure rows
#
# `decay` sets the embedding mix only (`decay ** distance` before row normalisation); it does
# NOT affect labels.  `subtree_suffix` DOES affect labels and V_ext — see the config comments.
#
# Usage:  bash scripts/experiments/00_build_ontology.sh [decay] [subtree_suffix]

# shellcheck source=scripts/experiments/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DECAY="${1:-0.5}"
SUBTREE_SUFFIX="${2:-ANY}"
OUT="${ONTOLOGY_DIR}"
mkdir -p "$OUT"

LOG="${EQ_LOG_DIR}/build_ontology_decay${DECAY}_${SUBTREE_SUFFIX}.log"

echo "cohort  : $TENSORIZED_COHORT_DIR"
echo "out_dir : $OUT"
echo "decay   : $DECAY   subtree_suffix: $SUBTREE_SUFFIX"
echo "log     : $LOG"

"$EQ_PY" -m every_query.data.build_ontology \
    tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    out_dir="$OUT" \
    decay="$DECAY" \
    subtree_suffix="$SUBTREE_SUFFIX" \
    > "$LOG" 2>&1

echo
echo "--- summary ---"
grep -E 'Wrote ontology|Dropping' "$LOG" || true
ls -la "$OUT"
