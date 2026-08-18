# Conditional-model integration plan (`conditional-v2`)

Port the conditional query-sequence work from the fork's `main` onto current upstream
`payalchandak/EveryQuery` `main`, restructured to the 5-stage sampler redesign.

This branch is currently **upstream/main plus this document only**. Nothing has been ported yet.

---

## 1. Starting state

| | ref | date |
|---|---|---|
| Merge base | `a13912b` ("fig size") | 2026-05-06 |
| Fork `main` head (22 commits ahead) | `0a7d57c` | 2026-07-25 |
| `upstream/main` head (54 commits ahead) | `8f83423` | 2026-08-11 |

```bash
git remote add upstream https://github.com/payalchandak/EveryQuery.git   # already added
git fetch upstream main
```

Port source files with `git checkout main -- <path>`. The fork's 22 commits stay reachable
on `main`; nothing is lost by not replaying them.

### What diverged

Upstream's 54 commits are dominated by the **5-stage task-sampler redesign** (~20 commits,
`sample_tasks.py` rewritten, +1725 lines) plus an August training-infra wave: bf16-mixed with
fp32 sigmoid, exact mid-epoch resume, `warmup_ratio`, model sized from data, in-training task
AUROC, per-task gradient norms.

The fork's 22 commits are the conditional pipeline (v1 -> v2 binary/EOS redesign -> eval_v3
dense grids). Of ~9,700 added lines, **~9,550 are in files upstream never touched** — a clean
parallel namespace. Only 8 files overlap:

```
conftest.py                                     tests/test_generate_tasks.py
pyproject.toml                                  tests/test_predict_cli.py
README.md                                       src/every_query/utils/model_loader.py
src/every_query/generate_tasks/README.md        src/every_query/generate_tasks/sample_tasks.py
```

### Two findings that shrink the work

1. **`id_cols` on `evaluate_index_df` is dead code.** Nothing ever calls it. The fork's entire
   `sample_tasks.py` diff can be dropped.
2. **`label_binary_occurrence` is self-contained.** The conditional labeler does its own asof
   join with no censoring/null — censoring is represented explicitly as the `TIMELINE//END`
   query in the sequence. It never calls `evaluate_index_df`, so upstream's censored-first
   semantics and the conditional labels are independent.

### Do not port `sample_tasks.py`

The fork's `db3abfc` ("occurrence overrides censoring") is **superseded by upstream `9bd85a1`**
("Stage 4 labeling: censoring beats occurrence; death is fully observed", #278) — which is
authored by the same person and handles death explicitly via `_truncate_at_death` rather than
bluntly overriding censoring. Take upstream's version wholesale.

---

## 2. Decisions already made

These were decided deliberately. **Do not re-litigate them during execution.**

| Decision | Choice |
|---|---|
| History strategy | Clean port off `upstream/main` — no rebase, no merge |
| Context sampling | **Full driver + fan-out**: Stage 0 -> 2 -> 3' in driver, Stage 4' per shard |
| Query drawing | **Adopt `QueryDistribution`**, extend with sequence-specific knobs |
| Eval-grid contexts | **Dual source**: supplied cohort if given, else Stage 0/2 sampling |
| Artifact layout | **Adopt upstream's `_artifacts` sibling layout** |
| Research artifacts | **Port everything as-is** — `scripts/`, PDFs, `reports/`, markdown docs |
| Definition of done | Test suite green **+ demo smoke run**. No real-cohort runs on this branch. |

---

## 3. Consequence: retraining is required

Two decisions each shift the training distribution:

- **Global contexts.** Upstream samples contexts globally across the split, weighted by each
  subject's prediction-time count. The fork drew `n_contexts` per shard independently. Different
  context distribution.
- **Float durations.** `QueryDistribution.sample` draws `duration_days` as **continuous floats**
  (`np.exp(rng.uniform(log lo, log hi))`, no rounding). The fork's `sample_log_uniform_durations`
  returns `round(exp(U(...)))` — **integer days**. The existing checkpoint was trained on
  integer-day durations.

`QuerySeqSchema` already stores durations as `Float32`, so nothing breaks type-wise — it is
purely a distribution shift. But **the numbers in `reports/EveryQuery_Conditional_Report_FINAL.pdf`
describe runs the new sampler cannot reproduce.** Those PDFs are a record of superseded
experiments, not a baseline to diff against. Regeneration and retraining are a follow-up branch,
out of scope here.

---

## 4. Phases

### Phase 1 — Port the additive files verbatim (1 commit)

**The source is the local `main` branch of this same repository — not a remote, not upstream.**
Bring each file across with:

```bash
git checkout main -- <path>
```

Nothing needs to be fetched, cloned, or downloaded. `main` is the fork's own head (`0a7d57c`)
and already holds every file listed below; `git checkout main -- <path>` copies the file from
that branch into the working tree and stages it, leaving `main` itself untouched. Do **not**
look for these files on `origin`, on `upstream`, or in a PR — `upstream` has never contained
any of them.

To enumerate the full set instead of working from the abbreviated list below:

```bash
git diff --name-only a13912b main          # merge base is fixed; verified to list 46 files
```

Equivalently, `git diff --name-only $(git merge-base main upstream/main) main`.

That lists all 46 files the fork changed. Subtract the 8 overlapping files named in section 1
(handled in Phases 4 and 5, not here) to get the 38 additive ones, and drop `sample_tasks.py`.

The 38 include `scripts/` (~10 eval + report builders), the 3 PDFs (~1.2MB), `reports/`,
`CONDITIONAL_QUERIES.md`, `COHORT_INFERENCE_NOTES.md`, and `HANDOFF.md`.

Key source files:

```
src/every_query/model/conditional_model.py
src/every_query/model/conditional_lightning.py
src/every_query/data/seq_dataset.py
src/every_query/data/schema.py                            (QuerySeqSchema additions)
src/every_query/generate_tasks/sample_query_sequences.py
src/every_query/generate_tasks/sample_evaluation_query_sequences.py
src/every_query/predict/predict_sequences.py
src/every_query/evaluate/evaluate_sequences.py
src/every_query/train/configs/conditional_config.yaml
src/every_query/train/configs/_demo_train_conditional.yaml
src/every_query/{generate_tasks,predict,evaluate}/configs/*sequences*.yaml
tests/test_conditional_queries.py
tests/test_conditional_cli.py
```

**Do not port**: `sample_tasks.py`, `tests/test_sample_tasks.py`, `tests/test_run_id.py`
(upstream dropped `run_id` in `f3d50eb` and replaced the sampler tests with a `tests/sampler/`
package).

This commit will not import cleanly. That is expected — it is the raw payload.

### Phase 2 — Rebuild `sample_query_sequences.py` as a 5-stage pipeline

**This is most of the effort.** The fork is shard-local: one `run_worker` per
`(input_shard, task_shard)` that samples contexts from that shard's own events and does
everything inline. The target mirrors upstream's `run()`.

| Stage | Upstream source | Work |
|---|---|---|
| 0 | `build_prediction_times` | reuse **as-is** |
| 1' | `QueryDistribution` | **extend** — draw `K` specs per sequence, add `eos_first_fraction`, `duration_mode` |
| 2 | `sample_patient_contexts` | reuse **as-is**; draw `n = num_sequences` |
| 3' | `build_index` | **rewrite** as `build_sequence_index` |
| 4' | fork's `label_binary_occurrence` | logic unchanged, wrapped in a per-shard worker |

**Stage 3' is the one genuinely new function.** Upstream's `build_index` does
`np.repeat(queries, num_contexts_per_query)` — one query per context. The conditional pipeline
needs `K` queries sharing one `(subject_id, prediction_time)`, tagged with `_ctx_id` / `_position`,
then the same per-shard join against `_prediction_times/{shard}.parquet` on
`(subject_id, prediction_time_index)`.

Preserve from upstream's `build_index`:
- the join-key dtype-mismatch guard (silent all-null joins otherwise),
- left-join-then-raise on null `prediction_time` (the join is total by design),
- `sort("shard")` + `group_by(maintain_order=True)` for deterministic shard order,
- one `read_parquet` per shard to keep driver memory flat.

Other notes:
- **Delete** `sample_log_uniform_durations` (replaced by `QueryDistribution`).
- **Keep `label_binary_occurrence` byte-identical.** It is self-contained and correct. Its
  doctest is a good regression anchor.
- **Seeding**: adopt upstream's axis convention — `derive_seed(seed, "queries")` and
  `derive_seed(seed, "contexts")` — replacing the fork's per-shard
  `derive_seed(seed, "seq_queries", stem, task_shard)`.
- Fan out Stage 4' via `ProcessPoolExecutor` sized by `resolve_workers(cfg.max_workers)`,
  matching `label_shards`.

### Phase 3 — Eval-grid sampler: dual context source

In `sample_evaluation_query_sequences.py`:

- `contexts_path` set -> `read_supplied_contexts` (**keep exactly as-is** so existing cohort
  parquets keep working),
- `contexts_path` unset -> Stage 0 + Stage 2 sampling.

Drop the `_resolve_path` import — removed upstream in the MEICAR path-handling overhaul (#235).

### Phase 4 — Configs: env fallbacks are gone, not renamed

Upstream **removed `${oc.env:...}` fallbacks entirely** (#235); path roots are mandatory Hydra
args. The fork's 5 conditional configs currently read `${oc.env:FINAL_DATA_DIR}`,
`${oc.env:TASK_DIR}`, `${oc.env:OUTPUT_DIR}`. Convert to `???` and pass on the CLI:

```bash
data_dir=$TOKENIZED_EVENTS_DIR out_dir=$TRAINING_TASKS_DIR query_codes=$TENSORIZED_COHORT_DIR
```

Key renames (model the config on `sample_training_tasks_config.yaml`):

| Old | New | Note |
|---|---|---|
| `min_context_per_subject` | `min_prediction_times_per_subject` | **semantic change** — counts distinct prediction times, not events. Revisit the value. |
| `num_workers` | `max_workers` | `null` => `resolve_workers()` uses cores-on-node |
| `FINAL_DATA_DIR` / `PROCESSED_DATA_DIR` | `TENSORIZED_COHORT_DIR` | env vars collapsed in `4d19294` |

Adopt the two-root artifact layout (invariant 7 — disjoint, never nested):

```
out_dir/{split}/{shard}.parquet              <- final outputs ONLY
{name}_artifacts/{split}/
    _prediction_time_counts.parquet          <- Stage 0 summary
    _prediction_times/{shard}.parquet        <- Stage 0 map
    _index/{shard}.parquet                   <- Stage 3' index
```

This invalidates existing run directories; the ported eval scripts have hardcoded paths that
will need updating before any follow-up runs.

### Phase 5 — Small re-applications

- `src/every_query/utils/model_loader.py`: re-apply the ~10-line `module_cls=None` parameter
  (defaults to `EveryQueryLightningModule`; conditional runs pass
  `ConditionalQueryLightningModule`).
- `pyproject.toml`: re-add the 4 entry points —
  `EQ_generate_query_sequences`, `EQ_generate_evaluation_query_sequences`,
  `EQ_predict_sequences`, `EQ_evaluate_sequences`.
- `conftest.py`, `README.md`, `src/every_query/generate_tasks/README.md`: merge the fork's
  additions into upstream's rewritten versions **by hand** — upstream substantially rewrote all three.
- Re-check whether the fork's `tests/test_generate_tasks.py` and `tests/test_predict_cli.py`
  edits are still needed; upstream rewrote both.

All core symbols the conditional code imports **still exist upstream** and need no shims:
`EveryQueryLightningModule`, `_dict_to_factory`, `MLP`, `_validate_tasks_dir`, `setup_model`,
`_atomic_write_parquet`, `_read_event_shard`, `read_query_codes`.

Gone and needing attention: `_resolve_path` (removed), `sample_contexts` (redesigned into
`sample_patient_contexts`).

### Phase 6 — Tests

`test_conditional_queries.py` and `test_conditional_cli.py` port as files but need updating for
the new stage structure and config keys.

Add the **distribution-parity test**: pin the sequence query/duration draws against the main
sampler's `QueryDistribution` so future drift in either surfaces as a failure. This is what
`9bf36b3` ("Match sampled-eval-grid defaults to the training query distribution") was chasing
by hand.

### Phase 7 — Definition of done

1. Full test suite green (upstream's + the conditional files).
2. Demo smoke run on the tiny demo cohort:
   `EQ_generate_query_sequences` -> `EQ_train` (few steps) -> `EQ_predict_sequences`
   -> `EQ_evaluate_sequences`.

Real-cohort regeneration, retraining, and eval_v3 re-runs are a **follow-up branch**.

---

## 5. Risks

1. **Retraining is required** (section 3). Budget for it before planning experiments.
2. **Phase 2 is a control-flow rewrite**, not a function swap. Phases 1, 4, 5 are mechanical.
3. **`min_prediction_times_per_subject` changed meaning.** It counts distinct prediction times,
   not events. Carrying the old numeric value across is a silent behavior change.
4. **Ported eval scripts will break** on the `_artifacts` layout change.
5. **Porting research artifacts diverges from upstream convention.** Upstream moved research-only
   code to a private `EveryQueryExperiments` repo (`09c4a06`) and core ships only `every_query`.
   Carrying `scripts/` and 1.2MB of PDFs here is a deliberate choice for self-containment; it
   makes a future upstream PR messier.

## 6. Data-handling reminder

Per the global instructions in `~/.claude/CLAUDE.md`: raw patient data must never enter model
context. Redirect all job output to log files, filter logs before reading
(`grep -E 'Error|epoch|loss'`), and never `print` / `.head()` / `.describe()` a cohort or task
frame. The demo-cohort smoke run in Phase 7 is the only data-touching step on this branch.
