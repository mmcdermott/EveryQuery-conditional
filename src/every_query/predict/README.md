# `predict/`

Inference stage of the EveryQuery pipeline — everything that consumes a trained model
checkpoint and produces per-`(subject_id, prediction_time, task_query)` probabilities.

> [!NOTE]
> This submodule is **scaffolded** as part of #90 but its main entry point (`EQ_predict`) is
> not implemented yet — tracked in [#81](https://github.com/payalchandak/EveryQuery/issues/81).
> What lives here today is the external-tasks layer (`external_tasks/`) that was previously
> at the top level as `aces_to_eq/` and `process_composite/`.

## Layout

```
predict/
└── external_tasks/           → convert + aggregate tasks that aren't in EQ's native vocabulary
    ├── aces_to_eq.py         → ACES task parquet → EQ task parquet
    ├── process_composite.py  → aggregate per-code predictions for disjunction tasks
    ├── get_per_code_from_composite.py
    └── configs/
```

## Pipeline position (planned)

```
model/ checkpoint   +   data/ batch     ──►  predict/  ──►  predictions parquet
                                                             (consumed by evaluate/)
```

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.2 — `EQ_predict`: [#81](https://github.com/payalchandak/EveryQuery/issues/81)
- Phase 2.1 — schema: [#80](https://github.com/payalchandak/EveryQuery/issues/80)
- Phase 3 — external-tasks promotion: [#62](https://github.com/payalchandak/EveryQuery/issues/62)
