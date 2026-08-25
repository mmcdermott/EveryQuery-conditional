#!/usr/bin/env bash
#
# Step 1 — build the ontology DAG artifacts for the tensorized cohort.
#
# Writes into $NF_ONTOLOGY_DIR:
#   ontology_vocab.parquet         (node_name, token_id, is_observed_code)   — the extended vocab V_ext
#   embedding_mix.parquet          (target_token_id, component_token_id, unnormalized_weight)
#   event_to_query_nodes.parquet   (event_code, query_node)                  — the labelling closure
#
# Reads only $TENSORIZED_COHORT_DIR/metadata/codes.parquet; the cohort is never written.
# This cohort's codes.parquet HAS a real `parent_codes` column, so the DAG comes from declared
# parents plus //-prefix structure (not prefix structure alone).
#
# Usage:  bash scripts/new_features/01_build_ontology.sh [decay] [subtree_suffix]

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

DECAY="${1:-0.5}"
SUBTREE_SUFFIX="${2:-ANY}"
LOG="${NF_LOG_DIR}/01_build_ontology.log"

echo "out_dir : $NF_ONTOLOGY_DIR"
echo "decay   : $DECAY   subtree_suffix: $SUBTREE_SUFFIX"
echo "log     : $LOG"

mkdir -p "$NF_ONTOLOGY_DIR"

"$EQ_PY" -m every_query.data.build_ontology \
    tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    out_dir="$NF_ONTOLOGY_DIR" \
    decay="$DECAY" \
    subtree_suffix="$SUBTREE_SUFFIX" \
    > "$LOG" 2>&1

echo
echo "--- summary ---"
grep -E 'Wrote ontology|Dropping|parent_codes' "$LOG" || true
ls -la "$NF_ONTOLOGY_DIR"
