#!/usr/bin/env python
"""Superseded: the sequence sampler now fans out across shards by itself.

This script existed because the fork's sampler was **shard-local** — one ``run_worker`` per
``(input_shard, task_shard)``, each drawing its own contexts from that shard's own events — so
covering a split meant driving a pool of workers from outside.

The 5-stage rewrite (Phase 2 of ``docs/history/2026-08-18-conditional-v2-integration-plan.md``) moved that fan-out inside
``EQ_generate_query_sequences``: Stage 0 scans the split once, Stages 1'-3' run in the driver, and
Stage 4' fans one labeling worker out per shard via a ``ProcessPoolExecutor`` sized by
``max_workers``.  One invocation now covers the whole split.

Equivalent command::

    EQ_generate_query_sequences \\
        data_dir=$TOKENIZED_EVENTS_DIR \\
        out_dir=$TRAINING_TASKS_DIR \\
        query_codes=$TENSORIZED_COHORT_DIR \\
        split=train \\
        num_training_sequence_examples=... \\
        min_queries=1 max_queries=5 \\
        duration_min=1 duration_max=731 \\
        min_prediction_times_per_subject=50 \\
        max_workers=8

Note the two semantic changes, both of which shift the training distribution (see §3 of the plan):

- ``--n-contexts`` was a **per-shard** count; ``num_training_sequence_examples`` is a **global** budget across the
  split, with contexts drawn weighted by each subject's prediction-time count.
- ``--min-context-per-subject`` counted *events*; ``min_prediction_times_per_subject`` counts
  distinct prediction times.  Carrying the old number across is a silent behaviour change.

This file is kept as a signpost rather than deleted so anyone rerunning an old command gets a
pointer instead of an obscure ``TypeError`` from the removed ``run_worker`` keyword arguments.
"""

import sys

MESSAGE = __doc__


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
