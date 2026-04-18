# `evaluate/`

Evaluation stage of the EveryQuery pipeline. Consumes trained model checkpoints + eval task
parquets, produces per-code AUCs and a model-selection ranking.

> [!IMPORTANT]
> Today `evaluate/` is a collection of **four** Hydra entry points that were split up when
> the pipeline was stitched together organically. Phase 2.4 of the refactor umbrella
> ([#83](https://github.com/payalchandak/EveryQuery/issues/83)) collapses them into a single
> `EQ_evaluate` that consumes predictions written by `EQ_predict`. The file moves landed
> here (#90) are pure scaffolding — no behavior change yet — so future structural diffs are
> contained to this one submodule.

## Current four-stage pipeline

Run in this order, each as an installed console script:

1. **`EQ_gen_eval_index`** — sample prediction-time `(subject, time)` tuples into a
    deterministic eval index. Config: `conf/gen_index_times_config.yaml`.
2. **`EQ_gen_eval_tasks`** — slice per-duration task matrices by `(code, duration)` using
    that index. Config: `conf/gen_tasks_config.yaml`.
3. **`EQ_evaluate`** — run a trained checkpoint against each sliced task, write per-code
    AUCs. Config: `conf/eval_config.yaml`.
4. **`EQ_select_model`** — rank models by pairwise win rate across the (code, duration)
    grid. Config: `conf/select_model_config.yaml`.

## Known seam

`EQ_gen_eval_tasks` currently expects per-duration wide task parquets at
`$TASK_DIR/{duration}/{split}/*.parquet`. Its former producer
(`EQ_generate_tasks_exhaustive`) was removed in
[#76](https://github.com/payalchandak/EveryQuery/pull/76). Phase 2 of
[#54](https://github.com/payalchandak/EveryQuery/issues/54) replaces this whole seam with a
`FlexibleSchema`-driven `EQ_predict` that takes a single task-specifying parquet.

## Pipeline position (planned, post-Phase-2)

```
predict/ predictions parquet  ──►  evaluate/  ──►  metrics + model ranking
```

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.3 — inventory + classify each current file: [#82](https://github.com/payalchandak/EveryQuery/issues/82)
- Phase 2.4 — consolidation: [#83](https://github.com/payalchandak/EveryQuery/issues/83)
- Phase 5 — `eval_config.yaml` stale `model_run_dirs`: [#66](https://github.com/payalchandak/EveryQuery/issues/66)
