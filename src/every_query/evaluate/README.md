# `evaluate/`

Evaluation stage of the EveryQuery pipeline. Consumes trained model checkpoints + eval
task parquets (legacy) or a `PredictionSchema` parquet from `EQ_predict` (new), produces
per-code AUCs and (legacy-only) a model-selection ranking.

> [!IMPORTANT]
> **Dual-path state.** The new consolidated `evaluate.py` landed in
> [#100](https://github.com/payalchandak/EveryQuery/pull/100) as `Phase 2.4`'s core
> contract — a single Hydra main over `PredictionSchema` parquets. But the
> `[project.scripts]` `EQ_evaluate` entry point still targets the legacy four-stage
> `evaluate.eval:main`. The `[project.scripts]` rewire + legacy-file deletions are
> tracked as the remaining work on [#83](https://github.com/payalchandak/EveryQuery/issues/83).

## New consolidated pipeline (`evaluate/evaluate.py`)

Reach it today as `python -m every_query.evaluate.evaluate predictions_parquet=<path> metrics_parquet=<path>`:

```
predict/ predictions.parquet  ──►  evaluate.py  ──►  metrics.parquet
(PredictionSchema)                                  (per-(query, duration_days): n_rows,
                                                     n_occurs_labeled, n_positive,
                                                     occurs_auroc, censor_auroc)
```

One Hydra main. No model instantiation, no trainer loop, no multi-model orchestration.
Cross-model comparison (what the old `EQ_select_model` did) moves to
`paper_experiments/leaderboard/` — tracked on #83 (answer #3).

## Legacy four-stage pipeline (`evaluate/eval.py` etc.)

Still what `EQ_evaluate` runs today until the #83 rewire lands. Run in order:

1. **`EQ_gen_eval_index`** — sample prediction-time `(subject, time)` tuples into a
    deterministic eval index. Config: `conf/gen_index_times_config.yaml`.
2. **`EQ_gen_eval_tasks`** — slice per-duration task matrices by `(code, duration)` using
    that index. Config: `conf/gen_tasks_config.yaml`.
3. **`EQ_evaluate`** — run a trained checkpoint against each sliced task, write per-code
    AUCs. Config: `conf/eval_config.yaml`. (Targets `eval.py`; will be switched to the
    new `evaluate.py` above when #83 closes out.)
4. **`EQ_select_model`** — rank models by pairwise win rate across the (code, duration)
    grid. Config: `conf/select_model_config.yaml`.

## Known seam

`EQ_gen_eval_tasks` currently expects per-duration wide task parquets at
`$TASK_DIR/{duration}/{split}/*.parquet`. Its former producer
(`EQ_generate_tasks_exhaustive`) was removed in
[#76](https://github.com/payalchandak/EveryQuery/pull/76). Phase 2 of
[#54](https://github.com/payalchandak/EveryQuery/issues/54) replaced this with
`EQ_predict` → `PredictionSchema` (the new consolidated pipeline above).

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.2 — `EQ_predict` (the producer for the new pipeline): [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in [#99](https://github.com/payalchandak/EveryQuery/pull/99))
- Phase 2.3 — inventory + classify each current file: [#82](https://github.com/payalchandak/EveryQuery/issues/82)
- Phase 2.4 — consolidation: [#83](https://github.com/payalchandak/EveryQuery/issues/83) (partial — new main landed via [#100](https://github.com/payalchandak/EveryQuery/pull/100); rewire deferred)
- `eval_config.yaml` stale `model_run_dirs`: [#66](https://github.com/payalchandak/EveryQuery/issues/66)
