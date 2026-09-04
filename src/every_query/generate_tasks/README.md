# `generate_tasks/`

Task-label generation stage of the EveryQuery pipeline. Console scripts live here in two
families of (scattered-for-training, dense-for-evaluation) pairs — one per model.

**Single-query model**, producing `TaskQuerySchema` rows (one scalar `boolean_value` per
`(subject, time, code, duration)`):

- **`EQ_generate_training_tasks`** — scattered shape: `num_queries` independent
    `(code, duration_days)` queries × `num_contexts_per_query` patient contexts,
    for pretraining. Runs a single-process 5-stage pipeline (see below).
- **`EQ_generate_evaluation_tasks`** — dense-grid shape: sampled prediction
    times × `(codes × durations)`, for feeding `EQ_predict` → `EQ_evaluate`.

**Conditional query-sequence model** (this fork), producing `QuerySeqSchema` rows (aligned
`queries` / `durations` / `answers` list columns per context):

- **`EQ_generate_query_sequences`** — scattered shape: every context draws its own
    independent sequence of `Uniform{min_queries..max_queries}` queries, for pretraining.
- **`EQ_generate_evaluation_query_sequences`** — dense-grid shape: the *same* `N` query
    sequences labeled at `K` sampled prediction times per subject (or at a supplied cohort),
    for feeding `EQ_predict_sequences` → `EQ_evaluate_sequences`.

## What lives here

- **`sample_tasks.py`** — the 5-stage training-task sampler. Registered as
    `EQ_generate_training_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_tasks`. One console invocation runs
    the **whole** pipeline in one process on one node — Stages 0–3 sequentially in the
    driver, then Stage 4 fans out one labeling worker per shard via a
    `ProcessPoolExecutor`:

    | Stage | Where             | What                                                                                                                          |
    | ----- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | 0     | driver            | Scan shards once, dedup to distinct `(subject_id, time)`, dense-rank → prediction-time map; filter eligible subjects. Cached. |
    | 1     | driver            | Sample `num_queries` `(code, duration_days)` queries (uniform code × uniform/log-uniform duration).                           |
    | 2     | driver            | Sample `N = num_queries × num_contexts_per_query` `(subject_idx, prediction_time_index)` contexts, iid with replacement.      |
    | 3     | driver            | Resolve each context's `prediction_time` against the Stage 0 map, zip with queries, write the partitioned index.              |
    | 4     | per-shard workers | Label each index shard independently via `join_asof`; write the final per-shard parquet atomically.                           |

    See [`redesign-spec.md`](redesign-spec.md) for the full stage contract, invariants,
    and determinism model — this README does not restate them.

- **`sample_evaluation_tasks.py`** — dense-grid evaluation-task generator.
    Samples `K` prediction times per subject, cross-joins with the full
    `(codes × durations)` grid the caller specifies, labels via the same
    `evaluate_index_df` primitive from `sample_tasks.py`. Registered as
    `EQ_generate_evaluation_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_evaluation_tasks`.

- **`sample_task_tracking_pairs.py`** — compact per-task AUROC-tracking sampler.
    Consumes `EQ_generate_evaluation_tasks`'s labeled output (typically `split=tuning`)
    and, per `(query, duration_days)` task, samples exactly one positive + one negative
    row into a single small parquet (2 rows/task). Registered as
    `EQ_sample_task_tracking_pairs` and runnable as
    `python -m every_query.generate_tasks.sample_task_tracking_pairs`. Feeds
    the optional in-training `TaskAurocTrackingCallback` — see
    `trainer.callbacks.task_auroc_tracking` in `train/configs/config.yaml`.

- **`sample_query_sequences.py`** — scattered conditional-sequence generator. Per context,
    draws `Uniform{min_queries..max_queries}` iid `(code, duration)` queries in random order
    (the end-of-timeline code `TIMELINE//END` is an ordinary code, not a privileged censor
    slot) and labels each with a binary observed-occurrence answer — there is no censoring
    null, because censoring is expressed by the `TIMELINE//END` query itself. Registered as
    `EQ_generate_query_sequences`.

- **`sample_evaluation_query_sequences.py`** — dense-grid conditional-sequence generator:
    `sample_evaluation_tasks` with `SequenceSpec`s in place of `(code, duration)` grid cells.
    Same cohort knobs and seed axes (`prediction_times_per_subject`, `min_context_per_subject`,
    `subject_subsample_fraction`, one worker per shard), so for one `(seed, split, K, ...)` the
    flat grid and the sequence grid score the identical `(subject, time)` set; `contexts_path`
    overrides the sampled cohort with a supplied one. The `N` sequences are designed
    (`sequences_path=tasks.yaml`, a mapping of `name -> [[code, duration], ...]`, with
    `[code, -1, bound_event]` for an event-bounded position and a mapping entry
    `{query, duration_days, start_event | start_duration_days, bound_event}` for a window that
    opens later than the prediction time) or drawn once from the training query distribution
    (`num_evaluation_sequences=64`, 50% event-bounded by default, every window opening at the
    prediction time unless the `eventstart_fraction` / `prediction_time_start_fraction` start knobs
    say otherwise) and shared by every shard. Reuses `sample_evaluation_tasks`'s cohort sampler and
    `sample_query_sequences`'s `label_query_sequences` labeler (which routes a grid with explicit
    starts through `interval_table.py`, the multitask sampler's window resolver); only the grid
    build is its own. Registered as `EQ_generate_evaluation_query_sequences`.

- **`configs/sample_query_sequences_config.yaml`** / **`configs/sample_evaluation_query_sequences_config.yaml`**
    — the conditional endpoints' configs. Same required-path-arg contract as the single-query
    ones below: `data_dir` / `out_dir` / `query_codes` are all `???`, no env fallback.

- **`configs/sample_training_tasks_config.yaml`** / **`configs/sample_evaluation_tasks_config.yaml`**
    / **`configs/sample_task_tracking_pairs_config.yaml`**
    — shipped Hydra configs. Path roots (`data_dir`, `out_dir`, and `query_codes` — pass the
    metadata root dir to load the full universe) are **required Hydra args** — no `.env`/env-var
    fallback (see [#235](https://github.com/payalchandak/EveryQuery/issues/235)). Pass them as
    shell-expanded vars (`data_dir=$TOKENIZED_EVENTS_DIR out_dir=$TRAINING_TASKS_DIR query_codes=$TENSORIZED_COHORT_DIR`);
    everything else is a Hydra override.

## Pipeline position

```
preprocessing/     →  generate_tasks/                   →  train/      →  predict/   →  evaluate/
EQ_process_data       EQ_generate_training_tasks           EQ_train       EQ_predict     EQ_evaluate
                      EQ_generate_evaluation_tasks ────────────────────►  (inference input)
                      EQ_generate_evaluation_tasks(split=tuning)
                        → EQ_sample_task_tracking_pairs ──►  (in-training tracking input)

                      EQ_generate_query_sequences          EQ_train       EQ_predict_sequences
                      EQ_generate_evaluation_query_sequences ───────────►  (inference input)
                                                                          EQ_evaluate_sequences
```

All endpoints consume:

1. Event shards at `$TOKENIZED_EVENTS_DIR/data/{split}/*.parquet` (from
    [`preprocessing/`](../preprocessing/)).
2. The query-code universe — pass `query_codes=$TENSORIZED_COHORT_DIR` (a metadata root dir resolves to
    `$TENSORIZED_COHORT_DIR/metadata/codes.parquet`), or override with an explicit list / file path. Same
    `query_codes` knob for both training and evaluation.

**Training-task outputs** land at `$TRAINING_TASKS_DIR/{split}/{shard}.parquet` — the final
dataset is the union of the per-shard files, and that directory holds *nothing else* at rest.
All intermediates (the Stage 0 prediction-time map, Stage 3 index) live in the **sibling**
`*_artifacts` root (`{parent}/{name}_artifacts` of `$TRAINING_TASKS_DIR`, no env var of its
own), so the two trees never nest and cleanup is a single `rm -rf` of the artifacts dir.

**Evaluation-task outputs** land at `$EVAL_TASKS_DIR/eval/{split}/*.parquet` (plus the deduped
cohort under `eval_unique/`). The separate `eval/` subdirectory keeps the two row distributions
from colliding in one directory. `EQ_generate_evaluation_query_sequences` writes the same layout
under *its* `out_dir` — give it a distinct root, since the two `eval/` trees hold incompatible
schemas and `EQ_predict_sequences` rglobs whatever it is pointed at.

**Sequence outputs** follow the same two-root rule: `out_dir` holds final parquets only, with all
intermediates in the sibling `{name}_artifacts` root. Both sequence endpoints write layouts
directly usable as `EQ_predict_sequences tasks_dir=...` — MEDS-TorchData rglobs that directory, so
point it at exactly the parquets you want scored. Note that adopting this layout **invalidates run
directories produced by the fork**, and the vendored `scripts/` eval helpers still carry hardcoded
paths from the old layout.

## Running it

### Pretraining tasks — whole pipeline, one command

```bash
# Stages 0–3 run inline in the driver; Stage 4 fans out across shards in-process.
EQ_generate_training_tasks \
	split=train \
	num_queries=1024 \
	num_contexts_per_query=1

# Restrict sampled training queries to a YAML code list (flat list or `{codes: [...]}`):
EQ_generate_training_tasks \
	split=train \
	query_codes=/path/to/train_query_codes.yaml
```

There is **no** `task_shard`/`input_shard` axis and no Hydra `-m` sweep (true of the
evaluation generator too, since #279) — the single driver process samples globally and
fans Stage-4 labeling out itself. Key knobs (full list in
`configs/sample_training_tasks_config.yaml`):

- `num_queries`, `num_contexts_per_query` — sampling budget; output is
    `num_queries × num_contexts_per_query` rows.
- `min_prediction_times_per_subject` (default 50) — minimum prior **prediction times**
    (distinct `(subject_id, time)` rows, not events) before a prediction time is eligible.
- `min_duration`, `max_duration`, `duration_distribution` (`uniform | log-uniform`) — the
    Stage 1 duration draw; `duration_days` is a float (no day-rounding).
- `query_codes` — **required**: `query_codes=$TENSORIZED_COHORT_DIR` (a metadata root dir) loads the full
    vocabulary from `$TENSORIZED_COHORT_DIR/metadata/codes.parquet`; or pass an inline Hydra list
    (`query_codes=[HR,TEMP]`) or a YAML/parquet path.
- `max_workers` — optional cap on the Stage 4 pool; `null` uses cores-on-node, an int caps that
    downward only. Set it when a run OOMs.
- `seed`, `overwrite`, and optional path overrides `data_dir` / `out_dir`.

### Evaluation tasks — dense grid, one parquet per discovered shard

```bash
EQ_generate_evaluation_tasks split=held_out
```

All shards under `{data_dir}/data/{split}/*.parquet` are discovered and processed in this
one invocation (#279).

### Task-tracking pairs — one compact parquet for in-training AUROC monitoring

```bash
# First, generate dense tuning-split labels the same way as above (note split=tuning):
EQ_generate_evaluation_tasks split=tuning \
	query_codes=$TENSORIZED_COHORT_DIR durations=[30,90,180,365,731]

# Then sample one pos/neg pair per task from that output:
EQ_sample_task_tracking_pairs \
	eval_labels_dir=$EVAL_TASKS_DIR/eval out_dir=$TASK_TRACKING_DIR split=tuning
```

This is a one-time offline step; the resulting `$TASK_TRACKING_DIR/tuning/0.parquet` is
small (2 rows per task) and fixed for the duration of a training run — point
`trainer.callbacks.task_auroc_tracking.config.task_labels_dir` at `$TASK_TRACKING_DIR` to
have `EQ_train` log a macro-averaged, per-task-sampled AUROC (`tuning/occurs_auroc_macro_sampled`)
every validation pass, without paying the cost of scoring the full tuning split.

### Conditional query sequences

```bash
# Dense evaluation grid: N sequences drawn once from the training distribution x K=2 sampled
# prediction times per subject of every shard of the split -- the same cohort knobs (and cohort)
# as EQ_generate_evaluation_tasks.  One parquet per shard at {out_dir}/eval/{split}/{shard}.parquet,
# directly usable as `EQ_predict_sequences tasks_dir=$EVAL_SEQ_TASKS_DIR/eval`.  The sampling
# defaults mirror `sample_query_sequences_config.yaml`; if the checkpoint was trained with
# overrides, pass the same ones here (e.g. min_queries=5 max_queries=5 duration_max=365) — an
# out-of-distribution horizon shows up as an unexplained metric shift, not as an error:
EQ_generate_evaluation_query_sequences \
	data_dir=$TOKENIZED_EVENTS_DIR out_dir=$EVAL_SEQ_TASKS_DIR query_codes=$TENSORIZED_COHORT_DIR \
	split=held_out prediction_times_per_subject=2 num_evaluation_sequences=64

# Or designed sequences on a supplied cohort (`contexts_path` bypasses the cohort knobs; every
# subject must be in the split). Event-bounded positions use `[query, -1, bound_event]`; long-format
# parquet specs use the optional nullable `bound_event` column. Spec identity in the output is the
# (queries, durations, bound_events[, start_durations, start_events]) columns.
EQ_generate_evaluation_query_sequences \
	data_dir=$TOKENIZED_EVENTS_DIR out_dir=$EVAL_SEQ_TASKS_DIR query_codes=$TENSORIZED_COHORT_DIR \
	split=held_out contexts_path=cohort.parquet sequences_path=tasks.yaml
```

`EQ_generate_query_sequences` takes the same three path args and only ever samples its contexts;
to label a supplied cohort use `EQ_generate_evaluation_query_sequences contexts_path=...`.

#### Window starts (issue #27)

A query's window may open later than the prediction time — after a delay, or at the next
occurrence of a start event — with the end then measured from the **resolved start**:

```
start  = prediction_time + start_duration_days   OR  first start_event strictly after prediction_time
end    = start + duration_days                   OR  first bound_event strictly after the resolved start
answer = start < some occurrence of query < end  (both endpoints open)
```

A start event that never occurs after the prediction time leaves the window empty (answer `False`,
even if the end is also unresolved); an end event that never occurs after a resolved start lets the
window run to the end of the record.  These are the multitask sampler's window semantics, and
`label_query_sequences` labels a grid carrying starts through the same `interval_table.py`
resolver — so `EQ_predict_multitask` can score the grid.  Designed specs spell starts out with the
mapping entry form (missing start keys mean a prediction-time start):

```yaml
post_admission:                        # opens at the next admission, closes 30d after it
  - query: LAB//X
    start_event: HOSPITAL_ADMISSION
    duration_days: 30
delayed:                               # opens 7d after the prediction time, closes 30d later
  - query: ICD//I10
    start_duration_days: 7
    duration_days: 30
between_events:                        # opens at the admission, closes at the next discharge
  - query: PROCEDURE//X
    start_event: HOSPITAL_ADMISSION
    duration_days: -1
    bound_event: HOSPITAL_DISCHARGE
```

or, in a long-format parquet, the optional `start_duration_days` / `start_event` columns next to
`seq_id, position, query, duration_days[, bound_event]`.  Sampled specs draw starts from the
`eventstart_fraction` / `prediction_time_start_fraction` / `start_duration_min|max|distribution` /
`start_event_codes` knobs (the multitask sampler's cumulative form split) on three seed axes of their
own, so a start knob perturbs none of the query / duration / end draws; the defaults open every window
at the prediction time and reproduce the pre-#27 grid — same specs, no start columns in the output.
Output rows carry `start_durations` (`0.0` = prediction time, `> 0` = delay in days, `-1.0` = event
start) and `start_events` (the start code, or null) only when some spec has an active start.

**The ordinary sequence models accept only prediction-time starts**: `ConditionalQueryPytorchDataset`
loads a grid with absent or all-default start columns, and refuses one with any positive-delay or
event start unless built with `allow_active_starts=True` — which only the multitask prediction
adapter does.  `EQ_generate_query_sequences` never samples a start.

### Multitask boundary labels — every code at every window

`EQ_generate_multitask_sequences` (issues #20, #22, #24) samples, per patient context, a fixed sequence
of `num_bounds` (default 5) windows — each with an explicit **start** and **end** specification — and
labels **every base-vocabulary code** at every window:

```
start[i, k] = prediction_time[i] + start_duration        (0 => the prediction time itself)
              OR first occurrence of start_event strictly after prediction_time[i]  (+inf if none)
end[i, k]   = start[i, k] + duration
              OR first occurrence of bound_event strictly after start[i, k]         (+inf if none)
target[i, k, v] = start[i, k] < some occurrence of v < end[i, k]
```

The start is resolved first and the end relative to it (`7 days after prediction time for 30 days`
is `(day 7, day 37)`; `the 30 days following the next admission` opens at the admission). Both ends
are open — the events at the start instant and at the end instant are excluded — exactly as
`label_with_event_bounds` defines the scalar `(prediction_time, boundary)` window (the tests are
differential against it, feeding the resolved start as its prediction time). An event end that never
recurs is `+inf` (rest of the record); an event start that never occurs leaves the window **empty**
(every target false — it never falls back to the prediction time), and `INF + duration` saturates to
`INF`. Equal start and end codes select consecutive occurrences.

Stages 0 and 2 are the shared sampler stages. Stage 1M draws the window sequences from seven
independent RNG streams — the end axes `bound_forms`, `bound_durations`, `bound_codes`, the #22
`condition_codes` axis, and the #24 start axes `start_forms`, `start_durations`, `start_codes` — so
changing any start setting perturbs neither the contexts nor the end / conditioning draws. Per slot
one uniform picks the start form: `eventstart_fraction` of event-defined starts (codes iid from
`start_event_codes`, null => every non-PAD base code), `prediction_time_start_fraction` at the
prediction time, the rest a positive `start_duration_min..max` draw (`start_duration_distribution`);
the two fractions must be `>= 0` and sum to `<= 1`, and omitting every start key reproduces the pre-#24
prediction-time starts. Stage 3M partitions by event shard and sorts each partition by
`(subject_id, prediction_time, _ctx_id)`. Stage 4M builds an interval table per shard
(`interval_table.py`), resolves the `(N, K)` starts then ends, flattens the `N x K` windows into
logical rows sorted by `(subject_id, resolved_start)` — a context's `K` windows may open at different
times — and labels them in one shared stream by scanning each subject's interval rows once per bounded
chunk (`label_chunk_rows x K` flattened rows, the same `C x K x V` scratch), scattering each packed
chunk back to `(context_row, k)` — never one lookup per `(context, window, code)`, never `K` separate
pipelines.

```bash
EQ_generate_multitask_sequences \
	data_dir=$TOKENIZED_EVENTS_DIR out_dir=$MULTITASK_TASKS_DIR query_codes=$TENSORIZED_COHORT_DIR \
	split=train num_training_examples=100000 max_workers=8 label_chunk_rows=2000
```

`query_codes` must be the cohort's metadata root (or a `codes.parquet` path): bits are aligned to the
unchanged `code/vocab_index`, `V = max(index) + 1`, and an explicit code list is rejected. Output per
shard:

```
{out_dir}/{split}/{shard}.parquet             MultitaskBoundarySchema: subject_id, prediction_time,
                                              start_durations[K] (float32; 0 = prediction time,
                                              -1.0 = event start), start_events[K] (str | null),
                                              durations[K] (float32, days after the resolved start;
                                              -1.0 = event end), bound_events[K] (str | null),
                                              condition_codes[K-1], condition_answers[K-1]
{out_dir}/{split}/{shard}.labels.npy          uint8 (rows, K, ceil(V / 8)), bitorder="little",
                                              row-aligned with the parquet
{out_dir}/{split}/_multitask_manifest.json    written by the driver before any worker starts:
                                              format_version 3, K, V, packed width, bit order, the
                                              window semantics (window_semantics: open_open,
                                              start_reference: prediction_time,
                                              duration_end_reference: resolved_start,
                                              missing_event_start: empty_window,
                                              missing_event_end: infinity), vocabulary fingerprint,
                                              ontology_mode: "none"
```

Unpack with `np.unpackbits(packed, axis=-1, count=V, bitorder="little")`; bit 0 (PAD) is always false
and must be masked from the loss. Each worker's dense scratch is `label_chunk_rows x K x V` booleans
(~110 MB at 2000 x 5 x 11k); packed rows are written incrementally through a temporary memmap that is
flushed and atomically renamed, then the parquet, then the `_labeled/{shard}.json` provenance sidecar.
An output is reused only when both files exist with agreeing row counts, the packed shape is right,
and the index, config and vocabulary fingerprints all match — a changed duration distribution,
event-bound fraction, `num_bounds`, boundary pool, any start fraction / start-duration setting /
start-event pool, the window semantics or `codes.parquet` relabels (outputs built under different
start semantics are never reused). Per shard, the worker logs event/interval/context counts, build /
boundary-resolution / labeling seconds, contexts per second, peak RSS, output bytes, the number of
event-defined starts and the fraction that do not resolve, the fraction of event ends resolving to
`+inf` (over windows whose start resolved), the fraction of empty windows, and mean positives per
window.

`MultitaskBoundaryPytorchDataset` (`every_query.data.multitask_dataset`) consumes the layout as MTD's
`task_labels_dir`: it reads only the metadata parquets at init, opens the sidecars read-only via
`mmap_mode="r"`, tags each row with its source shard and physical row so a global shuffle keeps the
alignment, unpacks once per batch in `collate`, and refuses to load when the cohort's `codes.parquet`
(or an explicit `expected_vocab_size` / `expected_vocab_fingerprint`) disagrees with the manifest.
Its batch carries `q_start_durations` / `q_start_codes` `(B, K)` (float32 / int64; `0.0` / `0` for a
prediction-time start, `-1.0` / vocab index for an event start) next to `q_durations` /
`q_bound_codes`; start codes go through the same base vocabulary as the targets (PAD / unknown codes
are rejected), exactly one start and one end representation must be active per slot, and metadata
parquets written before #24 (format 2, no start columns) load as prediction-time starts — a split may
mix format 2 and 3 shards. No start is ever sampled in the dataset.

**MVP scope:** observable leaf codes only. A non-null `ontology_dir` raises before Stage 0; events are
never closure-expanded. Ancestor targets and boundaries plug in later through the seams
`build_target_vocabulary`, `prepare_events_for_labeling` and `resolve_event_boundaries`.

## Determinism & restartability

All training draws derive from `seed` via `utils.seeds.derive_seed`, splitting the **query
axis** (`derive_seed(seed, "queries")`) and **context axis** (`derive_seed(seed, "contexts")`)
so they reproduce independently for a fixed seed. Stage 4 writes each shard via a temp file +
`os.replace`, so a present `{shard}.parquet` is always complete; reruns skip finished shards
(idempotent), and `overwrite=true` forces relabeling. The evaluation generator has only a
prediction-time axis (codes and durations are caller-specified), so its seed derives on
`(seed, split, input_shard)` and each worker writes idempotently. The task-tracking
sampler derives its per-(task, class) sample key on `(seed, "task_tracking_pairs", split)`
and writes its single output file atomically the same way; it skips work if the output
already exists unless `overwrite=true`.

`sample_evaluation_query_sequences` seeds its sequence draw on `(seed, "eval_seq_specs", split)`
alone — deliberately independent of the cohort, so the same `(seed, split)` yields the same `N`
sequences for any cohort you point it at, and two cohorts' metrics are comparable query-for-query.
Its cohort draw uses `sample_evaluation_tasks`'s axes, `(seed, "prediction_times", split, shard)`
and `(seed, "subject_subsample", split, shard)`, so it also lands on the flat generator's cohort.
`sample_query_sequences` still uses the fork's per-shard seed axes; folding it onto upstream's
`derive_seed(seed, "queries")` / `derive_seed(seed, "contexts")` convention is part of the Phase 2
rewrite.

## Related

- Architecture reference for the training sampler: [`redesign-spec.md`](redesign-spec.md).
- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54).
- Phase 2.1 — cross-stage `TaskQuerySchema`: [#80](https://github.com/payalchandak/EveryQuery/issues/80) (closed, merged in #96).
- Phase 2.2 — `EQ_predict` (consumes `sample_evaluation_tasks` output):
    [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in #99).
