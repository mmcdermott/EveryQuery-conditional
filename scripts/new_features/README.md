# `scripts/new_features` — tiny all-features train + evaluate

Trains a **tiny** conditional query-sequence model on the MIMIC tensorized cohort with all three
of the new features turned on, then measures held-out AUROC on 20 evaluation tasks of each query
type at sequence length 1 and length 3.

Everything the run writes lands under one root, `$NF_ROOT` (`$EQ_EXP_ROOT/new_features_test`), so
the whole experiment is one directory to inspect or delete.

## The three features, and where each is switched on

| feature | data side | model side | turned on in |
|---|---|---|---|
| RoPE time encoding | `datamodule.dataset_kwargs.strip_delta_tokens=true` | `lightning_module.model.use_rope_time=true` | `04_train.sh` |
| Event-bounded durations | `eventbound_fraction=0.5` on the sampler; the dataset then auto-detects the `bound_events` column | none — `bound_marker` is always allocated | `02_…` (labels) + `04_train.sh` |
| DAG-aware embeddings & queries | `ontology_dir=` on both samplers and `datamodule.dataset_kwargs.ontology_dir` | `lightning_module.model.ontology_dir` | `01_…`, `02_…`, `04_…`, `05_…` |

Two of these pairings behave very differently when they drift:

* `use_rope_time` ⇔ `strip_delta_tokens` is **checked at the first forward** and raises in either
  direction. You cannot get it wrong silently.
* The two `ontology_dir` keys are **not** cross-checked anywhere. If they differ, queries address
  the wrong embedding rows and the run completes normally with meaningless ancestor semantics.
  Both are therefore set from a single shell variable in `04_train.sh`, on purpose.

## Order of operations

```bash
bash scripts/new_features/01_build_ontology.sh                        # DAG artifacts (freeze after this)
bash scripts/new_features/02_sample_training_sequences.sh tuning 20000
bash scripts/new_features/02_sample_training_sequences.sh train 300000
bash scripts/new_features/03_make_eval_specs.sh                       # 60 designed specs, len 1 and 3
bash scripts/new_features/04_train.sh 00:00:16:00                     # tiny model, W&B, wall-clock capped
bash scripts/new_features/05_make_eval_labels.sh len1 4000
bash scripts/new_features/05_make_eval_labels.sh len3 4000
bash scripts/new_features/06_predict.sh "$RUN_DIR" len1
bash scripts/new_features/06_predict.sh "$RUN_DIR" len3
bash scripts/new_features/08_score.sh                                 # per-task + macro AUROC
```

`run_all.sh` chains all of it.

**Freeze `$NF_ONTOLOGY_DIR` after step 01.** The checkpoint stores the ontology *path*, not the
matrix, and re-reads it at every load — rebuilding in place silently changes the embeddings used
at inference. Sweep `decay` into separate directories instead.

## The evaluation set

`03_make_eval_specs.py` draws 20 tasks of each type **randomly from the model's own query
universe**, subject to an occurrence floor (a code nobody ever has yields no positives and an
undefined AUROC, which measures nothing):

* `dur_00…19` — a leaf code with a log-uniform horizon in `[1, 731]` days
* `evt_00…19` — a leaf code bounded by the next occurrence of a frequent ancestor event,
  `[code, -1, bound_event]`. Pairs where the boundary is an ancestor-or-self of the query are
  rejected: those are unconditionally False and carry no signal.
* `anc_00…19` — an ancestor node, which fires on any descendant. `HOSPITAL_ADMISSION` is pinned
  as `anc_00` (a pure ancestor over 70 child codes).

The length-3 file keeps the **same target at position 2** behind two randomly drawn filler
queries, so length 1 and length 3 differ only in conditioning depth.

Spec names are category+index only. The code strings live in the YAML on disk and are never
printed, so the metrics tables are safe to read.

## Why `08_score.sh` exists rather than `EQ_evaluate_sequences`

The shipped evaluator has no macro-average, no support gate, and groups `by_query` without
grouping by position — so at length 3 it pools the task under test with the random filler
queries. `07_score.py` reconstructs each row's spec identity from row order and **asserts that
reconstruction against the spec YAML** before scoring, then reports per-task AUROC with a
Hanley–McNeil CI and a per-category macro with a sign test against the 0.5 null.

`scripts/eval_v3.py` cannot be used here at all: it hand-builds the dataset with no
`dataset_kwargs`, so ancestor queries `KeyError` and rope-time raises on the missing
`time_pos_ids`.

## Data handling

No script prints patient rows, subject ids, or code strings. `verify_labels.py`,
`probe_*.py` and `07_score.py` emit aggregates and category-indexed names only.
