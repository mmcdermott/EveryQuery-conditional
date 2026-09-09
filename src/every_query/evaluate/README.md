# `evaluate/`

Evaluation stage of the EveryQuery pipeline. Two evaluators ship here, one per pipeline:
`EQ_evaluate` for the single-query model and `EQ_evaluate_multitask` for the multitask one.
They consume different prediction schemas and do not substitute for each other.

`EQ_evaluate` was rewired in Phase 2.5 ([#131](https://github.com/payalchandak/EveryQuery/pull/131))
to point at the consolidated `evaluate.py` (single-stage, no model instantiation). The
legacy four-stage evaluator (`eval.py`, `gen_index_times.py`, `gen_task.py`,
`select_model.py`) has been deleted; recover from git history if needed. Cross-model
comparison (what the old `EQ_select_model` did) lives in the `EveryQueryExperiments` repo
— tracked on [#83](https://github.com/payalchandak/EveryQuery/issues/83).

## Consolidated pipeline (`evaluate/evaluate.py`)

```
predict/ predictions.parquet  ──►  EQ_evaluate  ──►  metrics.parquet
(PredictionSchema)                                  (per-(query, duration_days): n_rows,
                                                     n_occurs_labeled, n_positive,
                                                     occurs_auroc, censor_auroc,
                                                     prevalence)
```

```bash
EQ_evaluate \
	predictions_parquet="$TRAINING_OUTPUT_DIR/predictions.parquet" \
	metrics_parquet="$TRAINING_OUTPUT_DIR/metrics.parquet"
```

One Hydra main. No model instantiation, no trainer loop, no multi-model orchestration.

## Multitask pipeline (`evaluate/evaluate_multitask.py`)

`EQ_predict_multitask` writes one row per evaluation-grid row, not per
`(query, duration_days)` pair, so `EQ_evaluate` cannot read it. `EQ_evaluate_multitask`
groups those rows by the query **specification** — the five window list columns
`(queries, durations, start_durations, start_events, bound_events)` — which is what
`EQ_generate_evaluation_query_sequences` resolves once and labels at every context. Each
group is therefore one task scored over the whole cohort, and the headline is a macro
average over those cells rather than a pooled AUROC, which would measure cross-query
base-rate separation instead.

```
predict/ predictions.parquet  ──►  EQ_evaluate_multitask  ──►  <stem>.by_task.parquet
(one row per grid row)                                          (one row per query spec)
                                                            ──►  <stem>.summary.parquet
                                                                 (macro AUROC + 3 CI pairs)
```

```bash
EQ_evaluate_multitask \
	predictions_parquet="$TRAINING_OUTPUT_DIR/predictions.parquet" \
	metrics_stem="$TRAINING_OUTPUT_DIR/metrics"
```

Bootstrap confidence intervals are always emitted, per cell and on the macro; `n_resamples`
trades runtime for resolution but is not a way to switch them off. The resampling unit is the
**subject**, not the row, since `prediction_times_per_subject` may exceed 1. One subject index
is drawn per replicate and shared across every cell, so a single `n_cells x n_resamples` AUROC
grid yields all four intervals — the per-cell one plus three macro variants that differ in what
they resample: `_subjects` (patients), `_tasks` (specs, comparable to what the training-time
callback logs), and `_nested` (both, and the one to quote). `n_tasks_null` is reported beside
`n_tasks_scored` so a macro over a handful of scorable cells cannot pass as a macro over all of
them.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.2 — `EQ_predict` (the producer for the new pipeline): [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in [#99](https://github.com/payalchandak/EveryQuery/pull/99))
- Phase 2.4 — consolidated `evaluate.py` landed: [#100](https://github.com/payalchandak/EveryQuery/pull/100)
- Phase 2.5 — `EQ_evaluate` rewired to new main: [#131](https://github.com/payalchandak/EveryQuery/pull/131)
- Cross-model leaderboard (lives in `EveryQueryExperiments`): [#83](https://github.com/payalchandak/EveryQuery/issues/83)
