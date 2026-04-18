# `data/`

The EveryQuery data layer: the PyTorch `Dataset` contract, the `EveryQueryBatch` named-tuple,
and the `QueryData` query-schema type. All query-specific data-shaping lives here, separate
from the model architecture.

## What lives here

- **`dataset.py`** — `EveryQueryPytorchDataset`, `EveryQueryBatch`, `QueryData`. The PyTorch
    `Dataset` implementation that maps tensorized MEDS shards + a task-labels parquet into the
    query-aware batches the model consumes. Shared between the `train/` stage (training loop)
    and the future `predict/` stage (inference).

Call through the package so stage submodules don't need to know the file layout:

```python
from every_query.data import EveryQueryPytorchDataset, EveryQueryBatch, QueryData
```

Hydra `_target_` strings in configs use the fully-qualified path
(`every_query.data.dataset.EveryQueryPytorchDataset`) for explicitness.

## Why `data/` is separate from `model/`

The data layer evolves on a different schedule than the model architecture.
Issue [#80](https://github.com/payalchandak/EveryQuery/issues/80) will reshape what
`EveryQueryBatch` carries — likely toward a MEDS `LabelSchema`-derived shape — with no touch
to the `nn.Module`. Keeping them in distinct submodules means a schema PR diffs only `data/`
and its two call-site edges (`generate_tasks/` output wiring + `train/` dataloader wiring).

## Pipeline position

```
generate_tasks/ ─► data/  ──►  model/  ─►  train/ / predict/
                  (batch      (architecture)
                   contract)
```

`data/` depends only on `meds_torchdata` for the upstream `MEDSTorchDataConfig` — it has no
dependency on `model/`, `train/`, or any stage submodule.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 1 submodule restructure: [#79](https://github.com/payalchandak/EveryQuery/issues/79) (this PR)
- Batch-schema evolution: [#80](https://github.com/payalchandak/EveryQuery/issues/80)
