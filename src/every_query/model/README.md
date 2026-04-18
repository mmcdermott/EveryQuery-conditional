# `model/`

Shared model-definition code: *what the model is*, independent of *when it runs*.
Imported by the `train/` stage (for training) and eventually by `predict/` (for inference).
Nothing in this submodule is stage-specific — no Hydra entry points, no CLI, no config files.

## What lives here

- **`model.py`** — `EveryQueryModel` (the ModernBERT-style encoder `nn.Module`) and
    `EveryQueryOutput` (the forward-pass output dataclass). This is the core architecture.
- **`lightning_module.py`** — `EveryQueryLightningModule`. Wraps `EveryQueryModel` for
    PyTorch Lightning with `training_step` / `validation_step` / `predict_step`. Shared between
    training and inference — at predict time the same LightningModule's `predict_step` produces
    predictions.
- **`dataset.py`** — `EveryQueryPytorchDataset`, `EveryQueryBatch`, `QueryData`. The PyTorch
    `Dataset` contract that maps tensorized MEDS shards + a task-labels parquet into batches the
    model consumes. Also shared across training and inference.

The public API is re-exported via `__init__.py`, so call sites stay terse:

```python
from every_query.model import (
    EveryQueryModel,
    EveryQueryLightningModule,
    EveryQueryPytorchDataset,
)
```

Fully-qualified module paths (`every_query.model.dataset.EveryQueryPytorchDataset`, etc.) are
used in Hydra `_target_` strings for explicitness, since a config reader wants to know exactly
which file the class lives in.

## Pipeline position

```
preprocessing/ → generate_tasks/ → train/   ┐
                                            ├──►  model/  (used by both)
                                 predict/   ┘
```

`model/` has no dependency on any stage submodule — it's the shared core. Stage submodules
import from here, never the other way around.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 1 submodule restructure: [#79](https://github.com/payalchandak/EveryQuery/issues/79) (this PR)
