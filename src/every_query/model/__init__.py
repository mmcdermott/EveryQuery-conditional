"""Model submodule: the EveryQuery encoder + its Lightning wrapper.

The *architecture* of the EveryQuery model.  Pure PyTorch + Lightning, independent of the
data-layer shape (which lives in ``every_query.data``) and of the Hydra-driven stage entry
points (``train/``, ``predict/``).  Call through this package so stage submodules do not need
to know the internal file layout::

    from every_query.model import EveryQueryModel, EveryQueryLightningModule
"""

from every_query.model.lightning_module import EveryQueryLightningModule
from every_query.model.model import EveryQueryModel, EveryQueryOutput

__all__ = [
    "EveryQueryLightningModule",
    "EveryQueryModel",
    "EveryQueryOutput",
]
