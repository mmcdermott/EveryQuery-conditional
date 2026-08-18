# `predict/`

Inference stage of the EveryQuery pipeline — everything that consumes a trained model
checkpoint and produces per-`(subject_id, prediction_time, query, duration_days)`
probabilities.

## Layout

```python
>>> from pretty_print_directory import PrintConfig
>>> print_directory("src/every_query/predict", config=PrintConfig(ignore_regex="__pycache__"))
├── README.md
├── __init__.py
├── configs
│   ├── predict.yaml
│   └── predict_sequences.yaml
├── external_tasks
│   ├── README.md
│   ├── __init__.py
│   ├── aces_to_eq.py
│   ├── configs
│   │   ├── aces_to_eq.yaml
│   │   ├── get_per_code_from_composite_config.yaml
│   │   └── process_composite_config.yaml
│   ├── get_per_code_from_composite.py
│   └── process_composite.py
├── predict.py
├── predict_sequences.py
└── schema.py

```

Key files:

- `predict.py` — `EQ_predict` (inference-only Hydra main).
- `predict_sequences.py` — `EQ_predict_sequences`: the conditional counterpart. Consumes
  `QuerySeqSchema` parquets plus a `ConditionalQueryLightningModule` checkpoint and runs
  teacher-forced inference, emitting one flat row per query *position*
  (`subject_id`, `prediction_time`, `position`, `query`, `duration_days`, `answer`, `answer_prob`).
- `schema.py` — `PredictionSchema` (`TaskQuerySchema` + `censor_prob` + `occurs_prob`).
- `configs/predict_sequences.yaml` — same required trio as `predict.yaml`
  (`model_run_dir`, `tasks_dir`, `output_parquet`), pointed at a conditional training run.
- `configs/predict.yaml` — required: `model_run_dir`, `tasks_dir`, `output_parquet`; optional: `ckpt_name`, `split` (`held_out` | `tuning`), `overwrite` (default `false` — refuses to clobber an existing `output_parquet`; pass `overwrite=true` to replace).
- `external_tasks/` — convert + aggregate tasks outside EQ's native vocabulary (`aces_to_eq.py`, `process_composite.py`, `get_per_code_from_composite.py`).

## Pipeline position

```
generate_tasks/  +  train/ best_model.ckpt
       │                     │
       ▼                     ▼
     tasks_dir/    ──►  predict/  ──►  predictions.parquet  ──►  evaluate/
     *.parquet                         (PredictionSchema)
     (TaskQuerySchema)
```

`EQ_predict` takes a directory of `TaskQuerySchema`-conformant parquet files — rows of
`(subject_id, prediction_time, query, duration_days)` plus optional inherited label
columns — and writes a `PredictionSchema`-conformant parquet adding the model's two-head
probabilities per row: `censor_prob` (P(row is censored)) and `occurs_prob`
(P(event occurred | not censored)). No AUCs, no model selection — that's
`evaluate/` (Phase 2.4, #83).

See [#129](https://github.com/payalchandak/EveryQuery/issues/129) for the post-refactor
discussion on generalizing `occurs_prob` → `label_prob` for non-occurrence task types.

## External tasks

See [`external_tasks/README.md`](external_tasks/README.md) for the ACES / composite-code
aggregation utilities.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.1 — task-query schema: [#80](https://github.com/payalchandak/EveryQuery/issues/80)
- Phase 2.2 — `EQ_predict` implementation: [#81](https://github.com/payalchandak/EveryQuery/issues/81) (this PR)
- Phase 3 — external-tasks promotion: [#62](https://github.com/payalchandak/EveryQuery/issues/62)
