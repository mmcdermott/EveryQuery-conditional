"""Model submodule: the EveryQuery encoder, its PyTorch Dataset contract, and the Lightning wrapper.

Everything under this submodule is *shared* across the ``train/`` and ``predict/`` stages —
it defines *what the model is* independently of *when it runs*.  Callers import through this
package (``from every_query.model import EveryQueryModel``) so stage submodules do not need to
know the internal file layout.
"""

from every_query.model.dataset import EveryQueryBatch, EveryQueryPytorchDataset, QueryData
from every_query.model.lightning_module import EveryQueryLightningModule
from every_query.model.model import EveryQueryModel, EveryQueryOutput

__all__ = [
    "EveryQueryBatch",
    "EveryQueryLightningModule",
    "EveryQueryModel",
    "EveryQueryOutput",
    "EveryQueryPytorchDataset",
    "QueryData",
]
