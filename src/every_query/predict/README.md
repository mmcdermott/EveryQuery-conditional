# `predict/`

Inference stage of the EveryQuery pipeline — everything that consumes a trained model
checkpoint and produces per-`(subject_id, prediction_time, code, duration_days)`
probabilities.

## Layout

```
predict/
├── predict.py            → EQ_predict (inference-only Hydra main)
├── schema.py             → PredictionSchema (TaskQuerySchema + predicted_boolean_probability)
├── configs/
│   └── predict.yaml      → required: model_run_dir, tasks_parquet, output_parquet
└── external_tasks/       → convert + aggregate tasks outside EQ's native vocabulary
    ├── aces_to_eq.py     → ACES task parquet → EQ task parquet
    ├── process_composite.py
    ├── get_per_code_from_composite.py
    └── configs/
```

## Pipeline position

```
generate_tasks/  +  train/ best_model.ckpt
       │                     │
       ▼                     ▼
     tasks.parquet  ──►  predict/  ──►  predictions.parquet  ──►  evaluate/
     (TaskQuerySchema)                  (PredictionSchema)
```

`EQ_predict` takes a `TaskQuerySchema`-conformant parquet of rows
`(subject_id, prediction_time, code, duration_days)` and writes a
`PredictionSchema`-conformant parquet adding the model's
`predicted_boolean_probability` per row.  No AUCs, no model selection — that's `evaluate/`
(Phase 2.4, #83).

## External tasks

See [`external_tasks/README.md`](external_tasks/README.md) for the ACES / composite-code
aggregation utilities.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.1 — task-query schema: [#80](https://github.com/payalchandak/EveryQuery/issues/80)
- Phase 2.2 — `EQ_predict` implementation: [#81](https://github.com/payalchandak/EveryQuery/issues/81) (this PR)
- Phase 3 — external-tasks promotion: [#62](https://github.com/payalchandak/EveryQuery/issues/62)
