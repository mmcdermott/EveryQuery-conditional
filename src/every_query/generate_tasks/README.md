# `generate_tasks/`

Task-label generation stage of the EveryQuery pipeline. Four console scripts live
here, in two families of (scattered-for-training, dense-for-evaluation) pairs —
one per model.

Single-query model, producing `TaskQuerySchema` rows (one scalar
`boolean_value` per `(subject, time, code, duration)`):

- **`EQ_generate_training_tasks`** — scattered shape: `N` independent
    `(query, duration_days)` tasks × `M` contexts, for pretraining.
- **`EQ_generate_evaluation_tasks`** — dense-grid shape: sampled prediction
    times × `(codes × durations)`, for feeding `EQ_predict` → `EQ_evaluate`.

Conditional query-sequence model, producing `QuerySeqSchema` rows (aligned
`queries` / `durations` / `answers` list columns per context):

- **`EQ_generate_query_sequences`** — scattered shape: every context draws its
    own independent sequence of `Uniform{min..max}` queries, for pretraining.
- **`EQ_generate_evaluation_query_sequences`** — dense-grid shape: the *same*
    `N` query sequences labeled at every context of a supplied cohort, for
    feeding `EQ_predict_sequences` → `EQ_evaluate_sequences`.

## What lives here

- **`sample_tasks.py`** — sampling-first pretraining-task generator. Draws `N`
    `(code, duration_days)` tasks from a uniform × log-uniform distribution, `N × M`
    `(subject_id, prediction_time)` contexts iid with replacement, zips them, and
    writes a long-format labels parquet per worker via a single-pass `join_asof`.
    Registered as `EQ_generate_training_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_tasks`.
- **`sample_evaluation_tasks.py`** — dense-grid evaluation-task generator.
    Samples `K` prediction times per subject, cross-joins with the full
    `(codes × durations)` grid the caller specifies, labels via the same
    `evaluate_index_df` primitive from `sample_tasks.py`. Registered as
    `EQ_generate_evaluation_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_evaluation_tasks`.
- **`sample_query_sequences.py`** — scattered conditional-sequence generator.
    Per context, draws `Uniform{min_queries..max_queries}` iid `(code, duration)`
    queries in random order (the end-of-timeline code `TIMELINE//END` is an
    ordinary code, not a privileged censor slot) and labels each with a binary
    observed-occurrence answer. Contexts come either from one shard or, with
    `contexts_path=...`, from a supplied `(subject_id, prediction_time)` parquet
    repeated `n_replicates` times. Registered as `EQ_generate_query_sequences`.
- **`sample_evaluation_query_sequences.py`** — dense-grid conditional-sequence
    generator. Takes a supplied cohort (`contexts_path`, required) and labels the
    same `N` sequences at every context in it, so the only thing varying across a
    given sequence's rows is the patient. The `N` sequences are either designed
    (`sequences_path=tasks.yaml`, a mapping of `name -> [[code, duration], ...]`)
    or drawn once from the training query distribution (`n_sequences=64`). Reuses
    `sample_query_sequences`'s cohort reader, cross-shard event gather, and
    `label_binary_occurrence` labeler; only the grid build is its own. Registered
    as `EQ_generate_evaluation_query_sequences`.
- **`configs/*_config.yaml`** — one shipped Hydra config per endpoint. Path
    fallbacks resolve via the repo's `.env`-based env-var convention
    (`$INTERMEDIATE`, `$PROCESSED`, `$TASK_DIR`); everything else is a Hydra
    override.

## Pipeline position

```
preprocessing/     →  generate_tasks/                   →  train/      →  predict/   →  evaluate/
EQ_process_data       EQ_generate_training_tasks           EQ_train       EQ_predict     EQ_evaluate
                      EQ_generate_evaluation_tasks ────────────────────►  (inference input)

                      EQ_generate_query_sequences          EQ_train       EQ_predict_sequences
                      EQ_generate_evaluation_query_sequences ───────────►  (inference input)
                                                                          EQ_evaluate_sequences
```

All four endpoints consume:

1. Event shards at `$INTERMEDIATE/data/{split}/*.parquet` (from
    [`preprocessing/`](../preprocessing/)).
2. The query-code universe at `$PROCESSED/metadata/codes.parquet` — or a CLI override:
    `query_codes=...` for training and both sequence endpoints, `codes=...` for
    `sample_evaluation_tasks`.

Training-task outputs land at `$TASK_DIR/{split}/*.parquet`; evaluation-task
outputs land at `$TASK_DIR/eval/{split}/*.parquet`. The separate `eval/`
subdirectory keeps the two row distributions from colliding in one directory.
The sequence endpoints take an explicit `out_dir` per run instead (their outputs
are cohort- and task-specific, not one canonical corpus), and both write layouts
that are directly usable as `EQ_predict_sequences tasks_dir=...` — MEDS-TorchData
rglobs that directory, so point it at exactly the parquets you want scored.

## Sweeping across shards

```
# Pretraining tasks (random tasks × random contexts):
python -m every_query.generate_tasks.sample_tasks -m \
    input_shard=0,1,2,... task_shard=range(0,K)

# Restrict sampled training queries to a YAML code list:
python -m every_query.generate_tasks.sample_tasks -m \
    input_shard=0,1,2,... task_shard=range(0,K) \
    query_codes=/path/to/train_query_codes.yaml

# `train_query_codes.yaml` may be either a flat list or `{codes: [...]}`.

# Evaluation tasks (dense grid over the held-out cohort):
python -m every_query.generate_tasks.sample_evaluation_tasks -m \
    input_shard=0,1,2,... split=held_out
```

The conditional-sequence endpoints are cohort-scoped rather than shard-scoped, so
they don't sweep the same way:

```
# Scattered training sequences (sweeps like the pretraining generator):
python -m every_query.generate_tasks.sample_query_sequences -m \
    input_shard=0,1,2,... task_shard=range(0,K)

# Dense evaluation grid: N designed sequences x every context of one cohort.
# per_spec_dirs=true writes {out_dir}/{name}/{split}/tasks.parquet, so each task
# is independently scoreable by `EQ_predict_sequences tasks_dir=...`.
EQ_generate_evaluation_query_sequences \
    contexts_path=cohort.parquet sequences_path=tasks.yaml \
    split=held_out per_spec_dirs=true

# Or N sequences drawn once from the training distribution, one combined parquet
# at {out_dir}/{split}/{cohort_stem}__sampled64.parquet.  The sampling defaults mirror
# `sample_query_sequences_config.yaml`; if the checkpoint was trained with overrides,
# pass the same ones here (e.g. min_queries=5 max_queries=5 duration_max=365):
EQ_generate_evaluation_query_sequences \
    contexts_path=cohort.parquet n_sequences=64 \
    split=held_out
```

A single dense run is one worker by design: the whole point is that all `N`
sequences are shared across all contexts, so there is no task axis to shard on.
Shard the *cohort* if it is too large to label in one pass.

The pretraining generator's seed derivation (`utils.seeds.derive_seed`)
separates task-axis and context-axis randomness so fixing `task_shard` across
`input_shard` values evaluates the *same* tasks on *different* patients; the
evaluation generator only has a prediction-time axis (codes and durations are
caller-specified), so its seed derives on `(seed, split, input_shard)`. Each
worker writes idempotently; re-running is a no-op.

`sample_evaluation_query_sequences` seeds its sequence draw on
`(seed, "eval_seq_specs", split)` alone — deliberately independent of the cohort,
so the same `(seed, split)` yields the same `N` sequences for any cohort you point
it at, and two cohorts' metrics are comparable query-for-query.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54).
- Phase 2.1 — cross-stage `TaskQuerySchema`: [#80](https://github.com/payalchandak/EveryQuery/issues/80) (closed, merged in #96).
- Phase 2.2 — `EQ_predict` (consumes `sample_evaluation_tasks` output):
    [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in #99).
