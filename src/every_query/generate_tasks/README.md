# `generate_tasks/`

Task-label generation stage of the EveryQuery pipeline. Home of the `EQ_generate_tasks`
console script.

## What lives here

- **`sample_tasks.py`** — sampling-first task-label generator. Draws N
    `(code, duration_days)` tasks from a configurable task distribution, N × M
    `(subject_id, prediction_time)` contexts iid with replacement from a tensorized
    MEDS shard, zips them, and writes a long-format labels parquet per worker via a
    single-pass `join_asof`. Registered as `EQ_generate_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_tasks`.
- **`configs/sample_tasks_config.yaml`** — shipped Hydra config. Path fallbacks
    resolve via the repo's `.env`-based env-var convention (`$INTERMEDIATE`,
    `$PROCESSED`, `$TASK_DIR`); everything else is a Hydra override.

## Pipeline position

```
preprocessing/     →  generate_tasks/   →  train/   →  (predict/evaluate)
EQ_process_data       EQ_generate_tasks     EQ_train
```

`generate_tasks/` consumes:

1. Event shards at `$INTERMEDIATE/data/{split}/*.parquet` (from
    [`preprocessing/`](../preprocessing/)).
2. The query-code universe at `$PROCESSED/metadata/codes.parquet`.

and writes labeled task parquets under `$TASK_DIR/{split}/*.parquet`. That
directory becomes `train/`'s `datamodule.config.task_labels_dir` input — the
parquets are already in the long `(subject_id, prediction_time, boolean_value, occurs, query, duration_days)` shape the dataloader expects, so there is no
intermediate collation step.

## Sweeping across shards

```
python -m every_query.generate_tasks.sample_tasks -m \
    input_shard=0,1,2,... task_shard=range(0,K)
```

The seed derivation (`utils.seeds.derive_seed`) separates task-axis and
context-axis randomness: fixing `task_shard` across `input_shard` values evaluates
the *same* tasks on *different* patients; fixing `input_shard` across `task_shard`
values evaluates *different* tasks on *different* patients. Each worker writes
idempotently via `.done` sentinels; re-running is a no-op.

## Related

- The sampler supersedes a previous exhaustive `(shard × duration × code)`
    enumerator that lived at `src/every_query/tasks.py`; that module and its collation
    step were removed in #76. `generate_tasks/` is the only going-forward path.
- Issue #80 (Phase 2.1 of #54) will define a shared cross-stage schema for
    `(prediction_time, subject_id, task_query)` rows that this module's output will
    conform to.
