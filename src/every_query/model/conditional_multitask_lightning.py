"""Lightning wrapper for :class:`ConditionalMultitaskARModel`."""

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Literal

import torch
from meds import held_out_split, train_split, tuning_split

from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.model.conditional_multitask_ar_model import (
    ConditionalMultitaskARModel,
    ConditionalMultitaskOutput,
)
from every_query.model.lightning_module import EveryQueryLightningModule, _dict_to_factory


class ConditionalMultitaskLightningModule(EveryQueryLightningModule):
    """Train and restore the all-vocabulary multitask architecture."""

    def __init__(
        self,
        model: ConditionalMultitaskARModel,
        optimizer: Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer] | None = None,
        LR_scheduler: Callable[..., Any] | None = None,
        warmup_ratio: float = 0.0,
        grad_norm_log_every_n_steps: int = 1000,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
            warmup_ratio=warmup_ratio,
            grad_norm_log_every_n_steps=grad_norm_log_every_n_steps,
        )
        # Full-vocabulary macro metrics are deliberately deferred.  Keep the parent's epoch
        # hooks and optimizer plumbing, but give them no scalar-model metrics to update.
        self.metrics = {train_split: {}, tuning_split: {}, held_out_split: {}}

    def _log_metrics(
        self,
        loss: torch.Tensor,
        outputs: ConditionalMultitaskOutput,
        batch: MultitaskBoundaryBatch,
        split: Literal[train_split, tuning_split, held_out_split],
    ):
        is_train = split == train_split
        sync_dist = not is_train and torch.distributed.is_available() and torch.distributed.is_initialized()
        self.log(
            f"{split}/loss",
            loss.item(),
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=sync_dist,
        )

    def training_step(self, batch: MultitaskBoundaryBatch) -> torch.Tensor:
        """Run the multitask loss without the scalar parent's two-head gradient logic."""
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, train_split)
        return loss

    @torch.no_grad()
    def predict_step(self, batch: MultitaskBoundaryBatch) -> dict[str, torch.Tensor]:
        """Return probabilities, targets, conditions, masks, and all window specifications."""
        _, outputs = self.model(batch)
        start_durations, start_codes = self.model._start_fields(batch)
        return {
            "probs": outputs.probs.detach().cpu(),
            "q_mask": batch.q_mask.detach().cpu(),
            "q_start_durations": start_durations.detach().cpu(),
            "q_start_codes": start_codes.detach().cpu(),
            "q_durations": batch.q_durations.detach().cpu(),
            "q_bound_codes": batch.q_bound_codes.detach().cpu(),
            "targets": batch.targets.detach().cpu(),
            "condition_codes": batch.condition_codes.detach().cpu(),
            "condition_answers": batch.condition_answers.detach().cpu(),
        }

    ARCHITECTURES: ClassVar[dict[str, type]] = {
        "conditional_multitask_ar": ConditionalMultitaskARModel,
    }

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str | None = None) -> "ConditionalMultitaskLightningModule":
        """Restore a multitask checkpoint using its mandatory architecture discriminator."""
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        for key in ("model", "optimizer", "LR_scheduler"):
            if key not in hparams:
                raise KeyError(
                    f"Checkpoint does not contain {key} hyperparameters. Got {list(hparams.keys())}"
                )

        model_hparams = dict(hparams["model"]) if isinstance(hparams.get("model"), dict) else None
        if model_hparams is None:
            raise TypeError("Checkpoint model hyperparameters must be a dictionary")
        architecture = model_hparams.pop("architecture")
        if architecture not in cls.ARCHITECTURES:
            raise KeyError(
                f"Checkpoint declares unknown multitask architecture {architecture!r}; "
                f"expected one of {sorted(cls.ARCHITECTURES)}."
            )
        model = cls.ARCHITECTURES[architecture](**model_hparams)
        optimizer = _dict_to_factory(hparams["optimizer"])
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])

        return super(EveryQueryLightningModule, cls).load_from_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
            warmup_ratio=hparams.get("warmup_ratio", 0.0),
        )
