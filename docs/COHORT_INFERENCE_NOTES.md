# Scoring a supplied cohort with the pretrained conditional model — what already exists

**Status:** reference notes, not a plan. Written 2026-07-24 to answer "what's already in this repo
that I can reuse?" before starting implementation. All facts below verified against the code and
against the archived model/data at `/experiments/EQ_conditional_experiments`.

## The goal, restated

Given a parquet of `W` rows of `(subject_id, prediction_time)` and the pretrained conditional
checkpoint, produce predicted probabilities.

1. Read the supplied index df (height `W`).
2. Sample `N` query sequences per row, each of length `K`; `q_1..q_K` drawn iid within a sequence.
3. `np.repeat` the index df `N`× and pair each row with a sampled sequence → height `N*W`.
4. Label every `q_j` for every row.
5. Run inference. The model is fed `q_1..q_K` plus the **ground-truth answers to `q_1..q_{K-1}`**;
   the number we want is the predicted probability for `q_K`.

`N` and `K` are config knobs.

---

## Access

The archive is mode `0750`, owner `mmd`, group `mimic-iv`. **You are already in that group**
(`getent group mimic-iv` → `mmd,gkondas,hayk,zzw2102,fpollet`) — your login session just predates
the grant, and group membership is fixed at login. Either start a new login session, or prefix:

```bash
sg mimic-iv -c '<command>'
```

It's MIMIC-IV-derived credentialed data under a PhysioNet DUA; the archive README asks that nothing
(especially `data/`) leave the group or the machine.

---

## Headline: this is ~85% assembled already

The conditional pipeline was built for exactly this shape of data, and your collaborator archived a
self-contained model + cohort + handoff README. Inference already does precisely what you described.

```
  [supplied index df]                        <-- YOU BRING THIS
          |
          v
  build_sequence_index_df   ...............  EXISTS, reusable as-is
          |                                  (feed it an N-replicated index df, K=min=max)
          v
  label_binary_occurrence   ...............  EXISTS, reusable as-is
          |
          v
  QuerySeqSchema parquet    ...............  EXISTS (align + atomic write helpers)
          |
          v
  EQ_predict_sequences      ...............  EXISTS, runs unmodified
          |
          v
  flat (subject_id, prediction_time, position, query, duration_days, answer, answer_prob)
          |
          v
  filter position == K-1    ...............  one line
```

---

## The archive: `/experiments/EQ_conditional_experiments`

```
├── README.md                  ← collaborator's handoff guide (read it, but see WARNING below)
├── model/big_v2/              ← THE model. Self-contained: resolved_config.yaml paths were
│   ├── best_model.ckpt          rewritten to point in-folder, no reference back to /home/mmd
│   ├── checkpoints/             (last.ckpt + step-275k/285k)
│   └── resolved_config.yaml
├── data/
│   ├── tensorized_cohort/     ← 18 GB MIMIC-IV MEDS v0.3.0, 11,958 codes. Vocab lives at
│   │                            metadata/codes.parquet; event shards at data/{split}/*.parquet
│   └── query_sequences_big/   ← the 28.79M training sequences (train/tuning/held_out)
├── report/                    ← final PDFs + the result JSONs they're built from
├── code/EveryQuery-conditional/  ← source snapshot + a real .env
└── examples/build_task_from_inputs.py   ← BROKEN, see below
```

Confirmed from `model/big_v2/resolved_config.yaml`:

| knob | value | why it matters to you |
|---|---|---|
| `tensorized_cohort_dir` | `…/data/tensorized_cohort` | where subject records + vocab come from |
| `max_queries` | **8** | **hard cap on `K`** — `block_pos_embed` clamps at `max_queries-1` |
| `max_seq_len` | 256 | patient context length |
| `batch_size` | 96 | override at predict time if you like |
| `vocab_size` | 11959 | sample codes from this vocab only (Gap 4) |
| `devices` | 1 | good — predict rejects multi-device trainers |

Run inference with:

```bash
EQ_predict_sequences \
  model_run_dir=/experiments/EQ_conditional_experiments/model/big_v2 \
  tasks_dir=<your QuerySeqSchema dir> \
  output_parquet=<out>.parquet \
  split=held_out
```

### ⚠ Two warnings about the archive

**1. `examples/build_task_from_inputs.py` does not run.** The README presents it as the bridge from
"(subject, time, queries)" to a model input — exactly your step 1–4 — but it imports two symbols
deleted in commit `01c34e1`:

```python
from every_query.data.seq_dataset import CENSOR_QUERY_CODE          # gone; now EOS_CODE
from every_query.generate_tasks.sample_query_sequences import (
    label_sequence_index_df,                                        # gone; now label_binary_occurrence
)
```

It `ImportError`s immediately, same as `scripts/make_clinical_task_sequences.py` and
`scripts/make_position_probe.py`. Its *structure* is still a good template though — and its
shard-handling is the answer to Gap 3:

```python
for fp in sorted((cohort / "data" / split).glob("*.parquet")):
    ev = _read_event_shard(fp).filter(pl.col("subject_id").is_in(wanted))
    if ev.height:
        event_frames.append(ev)
events = pl.concat(event_frames, how="vertical")
```

**2. The README's "always start with a censor query" advice is v1-era and is wrong for this model.**
It says: *"Start every sequence with `(TIMELINE//END, 1.0)`. This keeps position 0 in-distribution
(the model always saw a censor query there)."* That is not true of `big_v2`. Measured over 295,800
training sequences (3 of 292 train shards):

```
seq length:                    5 for 100% of rows
position 0 == TIMELINE//END:   28 / 295,800 = 0.0095%     (≈ 1/11,959, i.e. pure uniform)
EOS rate by position:          pos0 9.5e-5, pos1 8.1e-5, pos2 6.8e-5, pos3 8.1e-5, pos4 9.8e-5
pos-0 durations:               log-uniform, min 1.0 / median 19.0 / max 365.0
pos-0 duration == 1.0:         6.8%
```

Flat EOS rate across all five positions, at exactly the uniform-sampling rate. **The model never saw
a privileged censor query at position 0.** `big_v2` was trained with `eos_first_fraction=0.0` and
fully iid sequences.

So your fully-iid design is correct and matches the training distribution exactly. Following the
README's advice would push position 0 *out* of distribution. The leak concern behind that advice
(don't teacher-force a same-horizon censor answer next to a terminal target —
`CONDITIONAL_QUERIES.md:55-61`) is real, but it's an argument against a *privileged* censor query,
which iid sampling already avoids.

**Match training exactly with:** `K = 5`, `duration_min = 1`, `duration_max = 365`,
`duration_mode = "random"`, `eos_first_fraction = 0.0`. Note the repo's config default is
`duration_max: 731` — that is *not* what `big_v2` saw.

---

## The load-bearing pieces

### Sampling

**`build_sequence_index_df`** — `src/every_query/generate_tasks/sample_query_sequences.py:77`

```python
def build_sequence_index_df(
    contexts: pl.DataFrame,          # (subject_id, prediction_time); extra cols dropped
    query_codes: list[str],
    min_queries: int, max_queries: int,
    duration_low: int, duration_high: int,
    seed: int,
    eos_first_fraction: float = 0.0,   # leave 0.0
    duration_mode: str = "random",     # leave "random"
) -> pl.DataFrame:
    # -> (_ctx_id: UInt32, _position: Int64, subject_id, prediction_time,
    #     query: Utf8, duration_days: Float32), sorted by (_ctx_id, _position)
```

This is step 2 verbatim. Codes are drawn `rng.integers(0, len(query_codes), size=total)` —
**independently per query slot**, the iid property you wanted (`:156`). Durations are independent
log-uniform draws (`:163`).

Two things to know:

- It emits **exactly one sequence per row of `contexts`**, with `_ctx_id` = the row index. So the
  N-replication is on you *before* the call. Prefer
  `index_df.join(pl.int_range(N, eager=True).to_frame("_rep"), how="cross")` over
  `pl.concat([index_df] * N)` — it keeps a `_rep` column you'll want (Gap 2).
- Set `min_queries == max_queries == K`. Everything is vectorised (one numpy expansion + one join),
  so `N*W` in the millions is fine.

**`sample_log_uniform_durations`** — `:55`. Takes an `rng`, not a seed.
**`read_query_codes`** — `sample_tasks.py:419`. Resolves a code universe from a list, YAML, or
`{dir}/metadata/codes.parquet`. Dedups + sorts for determinism.
**`derive_seed(*parts)`** — `utils/seeds.py`. blake2b, cross-process stable, feeds
`np.random.default_rng`.

### Labeling

**`label_binary_occurrence`** — `sample_query_sequences.py:195` — is step 4, and it is efficient.

```python
left  = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts")).sort(sid, q, "_pts")
right = events_df.rename({code: q}).select(sid, q, time).sort(sid, q, time)
joined = left.join_asof(right, by=[sid, q], left_on="_pts", right_on=time, strategy="forward")
answer = pl.col(time).is_not_null() & (pl.col(time) < pl.col(pt) + pl.duration(days=pl.col(d)))
```

**One `join_asof` labels every query regardless of how many distinct codes are in play.** No
per-code groupby, no per-row loop. 2 sorts + a merge. Then it reassembles into `QuerySeqSchema` list
rows via `group_by(_ctx_id, maintain_order=True).agg(...)` (`:253-263`).

Semantics (v2 design — this matters):

- `answer` is a **plain non-nullable boolean**. No censoring, no nulls.
- An event you couldn't observe because the record ends first is `False`, not null.
- Right-truncation is expressed **as a query**: `TIMELINE//END` (`seq_dataset.py:48`) is a real MEDS
  code at each subject's last event, so `(TIMELINE//END, d)` is `True` iff the record ends within `d`.
- Window is `(t, t+d)` — **open at both ends**. Strict lower via the `+1µs` asof shift, strict
  upper via `<`: an event charted exactly on `t+d` does **not** count as an occurrence. The same
  rule applies to event-bounded queries, where the window is `(t, boundary)`, so "Sepsis before
  discharge" excludes a Sepsis code sharing the discharge instant. The docstrings (`schema.py:110`,
  `seq_dataset.py:10`) state the same interval; `tests/test_window_bounds_contract.py` is where the
  rule lives and drives every labeller through one table.

**`_read_event_shard`** — `sample_tasks.py:392`. Mandatory before labeling. The two casts are
correctness-critical, not cosmetic: `code → Utf8` (upstream may store Categorical *or integer vocab
indices*, either of which silently produces zero join matches) and `time → Datetime("us")` (at
millisecond precision the `+1µs` shift rounds to zero and strict `>` silently becomes `>=`).

### Writing

`QuerySeqSchema` (`src/every_query/data/schema.py:100`):

| column | arrow dtype |
|---|---|
| `subject_id` | `int64` |
| `prediction_time` | `timestamp[us]` |
| `queries` | `large_list<large_string>` |
| `durations` | `large_list<float32>` |
| `answers` | `large_list<bool>` |

Write with `QuerySeqSchema.align(df.to_arrow())` → `pl.from_arrow(...)` → `_atomic_write_parquet`
(`sample_tasks.py:487`). Extra columns are allowed by the `flexible_schema` base — relevant to Gap 2.

### Inference

`EQ_predict_sequences` (`src/every_query/predict/predict_sequences.py`) reuses the *training* config
wholesale — `setup_model` (`utils/model_loader.py:24`) loads `resolved_config.yaml` and only
`task_labels_dir` / `batch_size` are overridden (`:101-105`). Output is flat, one row per query
position: `(subject_id, prediction_time, position, query, duration_days, answer, answer_prob)`.

**Inference is already teacher-forced and already gives you exactly what you asked for.** Confirmed
in the mask (`conditional_model.py:126-128`) and read-out (`:322-325`):

```python
allowed  = (b_k < b_i) | ((b_k == b_i) & (t_k < TOKEN_ANSWER))
allowed |= (b_k == b_i) & (t_i == TOKEN_ANSWER) & (t_k == TOKEN_ANSWER)
...
answer_hidden = dec_out[:, TOKEN_DURATION::TOKENS_PER_QUERY, :]   # read at the DURATION token
```

Each query is a 3-token block `[code_j, duration_j, answer_j]`. The prediction for `A_j` is read off
the **duration** token, which attends to the patient encoding, all strictly-earlier blocks
*including their teacher-forced answers*, and `Q_j` itself — but never `A_j`. So position `K-1` of a
length-`K` sequence is precisely `P(A_K | patient, Q_1..Q_K, A_1..A_{K-1})`.

`A_K`'s answer token is in the input stream but nothing except itself can attend to it. No leak.
Tested: `tests/test_conditional_queries.py:43-75` (mask) and `:143-263`
(`test_own_answer_does_not_change_own_logit`, `test_prior_answer_changes_later_logit`).

**Efficiency note:** you only asked for position `K-1`, but the model emits all `K` positions in the
same forward pass at no extra cost, each conditioned on a different amount of prior context. If
"how does discriminability change with conditioning depth?" is anywhere in scope, that's free — just
don't filter it away.

---

## Gaps — what you'd actually have to write

**Gap 1 — no working entry point takes a supplied index df.** `run_worker:266` always calls
`sample_contexts(...)`, which samples its own prediction times; there's no `contexts_path` knob. The
archive's `build_task_from_inputs.py` was meant to fill this and is broken (above). So: a thin
module that reads your parquet, replicates it `N`×, and calls the two existing functions. The
functions it wraps do all the real work.

**Gap 2 — the `N` replicates are indistinguishable in the output.** `predictions_to_df`
(`predict_sequences.py:66-78`) emits only `(subject_id, prediction_time, position, …)`. With `N`
replicates per context that key is non-unique `N`× over. Row *order* is preserved and asserted
(`:61-64` row-count check + the `SequentialSampler` guard at `:111-113`), so replicates are
recoverable positionally — but that's fragile and silently wrong if anything reorders. Carry a
`_rep`/`_ctx_id` column through instead. `QuerySeqSchema` allows extra columns and
`labels_df` (`seq_dataset.py:171-184`) reads a fixed column list, so it rides along on disk
harmlessly — but `predictions_to_df` selects explicitly and needs a small edit to emit it.

**Gap 3 — shard routing for the events.** `label_binary_occurrence` needs an `events_df` covering
your cohort's subjects, and your cohort is an arbitrary subject set, not a shard. Use the prefilter
pattern from `build_task_from_inputs.py` (quoted above). **Do not naively concatenate all shards** —
`scripts/eval_macro_position.py:238-241` records that the union is "tens of millions of rows and
would OOM".

**Gap 4 — the query vocabulary is a hard constraint.** `encode_query` (`seq_dataset.py:186-194`)
**raises `KeyError`** on any code outside the model's vocabulary — deliberately, unlike v1 which
silently PAD-encodes. And unlike `EQ_predict`, `EQ_predict_sequences` has **no vocab pre-flight**, so
OOV fails late, inside `collate`, mid-run. Sample codes from
`data/tensorized_cohort/metadata/codes.parquet` (columns `code`, `code/vocab_index`; 11,958 codes),
or add a pre-flight. `TIMELINE//END` is present in that vocab (it's index 11884 per the training
config) — worth asserting anyway, since `__init__` only *warns* if it's missing (`:162-167`).

**Gap 5 — split semantics, and a silent row-drop.** `EQ_predict_sequences` accepts only
`tuning` / `held_out` (`predict_sequences.py:40-43,89-90`); `train` is rejected because that
dataloader shuffles. Your subjects must live in whichever split you pass. Note `task_labels_fps`
globs **every** parquet under `tasks_dir` recursively, ignoring split subdirs; the split filter
happens later via the schema_df semi-join in `get_task_seq_bounds_and_labels`
(`seq_dataset.py:120-133`), which **drops rows whose `subject_id` is absent from the split with no
error**. Assert `len(dataset) == N*W` before trusting any output.

**Gap 6 — `predict_sequences` is not streaming.** `EQ_predict` uses a `BasePredictionWriter` so
memory is flat in cohort size; `EQ_predict_sequences` calls `trainer.predict(...)` with default
`return_predictions=True` and materialises everything before `predictions_to_df` (`:117-119`). At
`N*W` sequences × `K` positions this is the most likely OOM. Chunk the task dir, or port the
streaming writer.

---

## Decisions still open

- **Durations.** The sketch never mentions them, but every query is `(code, duration)` — there is no
  code-only query here. To match `big_v2` training: log-uniform over `[1, 365]`. The
  `duration_mode="same"`/`"nondecreasing"` alternatives use a Python loop over contexts (`:164-171`),
  the only non-vectorised path — `N*W` iterations of it would hurt.
- **`K`.** Capped at 8 by `max_queries`; training used exactly 5. `K=5` is the in-distribution choice.
- **Whether to keep intermediate positions** (see the efficiency note) — costs nothing at generation
  time, unrecoverable afterwards.
- **Environment.** No `.venv` in this checkout. The archive README says to build from the pinned
  lockfile in its own snapshot (`cd code/EveryQuery-conditional && uv sync --frozen`); the original
  training venv at `/home/mmd/MIMIC_experiments/venvs/eq` is not group-readable.

---

## Landmines

- **Three stale scripts that `ImportError`**: `examples/build_task_from_inputs.py` (in the archive),
  `scripts/make_clinical_task_sequences.py`, `scripts/make_position_probe.py`. All still import
  `label_sequence_index_df` / `CENSOR_QUERY_CODE`, deleted in `01c34e1`. Design references only.
  `make_clinical_task_sequences.py:145-161` has the clean vectorised
  `anchors.with_row_index(CTX_ID_COL).join(spec_frame, how="cross")` pattern.
- **Don't resurrect the censored 3-valued sequence labeler.** Deleted for cause:
  `CONDITIONAL_QUERIES.md:55-61` — a same-horizon censor query at position 0 leaks terminal-event
  labels; 30-day mortality AUROC hit 0.991, *of which 0.996 came from the censor answer alone*. If
  you genuinely need censoring-aware labels the supported path is
  `evaluate_index_df(..., id_cols=(CTX_ID_COL, POSITION_COL))` (`sample_tasks.py:267`) +
  `compute_max_time_per_subject` (`:256`), with the same group_by reassembly — but read the
  post-mortem first.
- **Three docs disagree on the headline result.** `CONDITIONAL_QUERIES.md:157-171` and the archive
  README both quote the step-50k numbers (slope +0.00295/pos, ρ=1.0); `reports/README.md:7-16` has
  the converged `big_v2` numbers with a **4× smaller** effect (+0.00069/pos, CI [+0.00029,+0.00109],
  ρ=0.90). The converged ones are the real ones.
- **Macro, not pooled, metrics.** `CONDITIONAL_QUERIES.md:127-152` — pooled AUROC reads ≈0.91 vs an
  honest macro ≈0.77. Only relevant if you compute metrics rather than just emitting probabilities.

---

## Closest working templates

| script | what to copy |
|---|---|
| `scripts/eval_v2.py:104` | `build_labeled_sequences(ctx_events, spec_fn)` — cleanest index-df → N sequences → labels loop in the repo |
| `scripts/eval_v2.py:138` | `score_last(...)` — teacher-forced scoring of *the last query only*, i.e. literally your inference step, incl. `force_prior ∈ {None,0,1}` counterfactual overrides |
| `scripts/eval_macro_position.py:41` | `occurrence_labels(flat, events)` — row-wise labeling returning a flat `occurred` column in input order, if you'd rather not have list columns |
| `scripts/eval_occurs_uncensored.py:16` | `build_uncensored_triples(...)` — the only explicit uncensored filtering |
| `scripts/generate_mimic_sequences.py` | shard-parallel driver; `ProcessPoolExecutor(mp_context="spawn")` (fork deadlocks against live polars/torch pools), `POLARS_MAX_THREADS=4` per worker |
| `examples/build_task_from_inputs.py` (archive) | the subject-prefiltered shard scan (Gap 3) — broken as written |

`scripts/eval_v2.py` is the one to read first: it already does index-df → sequences → labels →
score-the-last-query, just with its own context sampling rather than a supplied cohort.
