"""LightningModule for :class:`~every_query.model.conditional_model.ConditionalQueryModel`.

Reuses the optimizer / LR-scheduler factory plumbing of
:class:`~every_query.model.lightning_module.EveryQueryLightningModule`; overrides the metric and
step logic for the per-position answer-logit output shape.

Metric conventions:
- ``censor_auc``: AUROC of position-0 answers (the always-observed censor query).
- ``occurs_auc``: AUROC over positions >= 1, restricted to observed (non-censored, non-padding)
  answers — the conditional-query analogue of the single-query model's occurs head.
"""

import logging
from typing import Literal

import torch
from meds import held_out_split, train_split, tuning_split
from torchmetrics.classification import BinaryAUROC

from every_query.data.seq_dataset import ConditionalQueryBatch
from every_query.model.conditional_model import ANSWER_YES, ConditionalQueryModel, ConditionalQueryOutput
from every_query.model.lightning_module import EveryQueryLightningModule, _dict_to_factory

logger = logging.getLogger(__name__)


class ConditionalQueryLightningModule(EveryQueryLightningModule):
    """Lightning wrapper for the conditional query-sequence model."""

    def __init__(self, model: ConditionalQueryModel, optimizer=None, LR_scheduler=None):
        super().__init__(model=model, optimizer=optimizer, LR_scheduler=LR_scheduler)
        # A single binary answer head produces every per-query logit.  Training-time AUROCs are
        # pooled (across all codes) and therefore base-rate inflated — kept only to track
        # dynamics/stability; the trustworthy within-query numbers are computed at held-out
        # evaluation.
        #   answer_auc       — pooled over every (non-padding) query position;
        #   answer_auc_pos{j}— position j alone.  Later positions condition on strictly more
        #                      teacher-forced answers; per-position lets us watch that trend.
        self._n_positions = model.max_queries

        def _metric_set():
            m = {"answer_auc": BinaryAUROC().cpu()}
            for j in range(self._n_positions):
                m[f"answer_auc_pos{j}"] = BinaryAUROC().cpu()
            return m

        self.metrics = {train_split: {}, tuning_split: _metric_set(), held_out_split: _metric_set()}

    def _log_metrics(
        self,
        loss: torch.Tensor,
        outputs: ConditionalQueryOutput,
        batch: ConditionalQueryBatch,
        split: Literal[train_split, tuning_split, held_out_split],
    ):
        batch_size = batch.batch_size
        is_train = split == train_split
        sync_dist = not is_train and torch.distributed.is_available() and torch.distributed.is_initialized()

        self.log(
            f"{split}/loss",
            loss.item(),
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        if is_train or outputs.answer_logits is None:
            return

        probs = outputs.answer_logits.detach().cpu().sigmoid().float()
        targets = (batch.q_answers == ANSWER_YES).detach().cpu().long()
        observed = outputs.valid_mask.detach().cpu()

        # Overall: every real query position pooled.
        if observed.any():
            self._update_metric(
                name="answer_auc", split=split, preds=probs[observed], target=targets[observed]
            )

        # Per-position AUROC: position j is real where observed[:, j] is True.  Sequences
        # shorter than j+1 simply contribute no rows at position j.
        n_pos = min(self._n_positions, observed.shape[1])
        for j in range(n_pos):
            sel_j = observed[:, j]
            if sel_j.any():
                self._update_metric(
                    name=f"answer_auc_pos{j}",
                    split=split,
                    preds=probs[:, j][sel_j],
                    target=targets[:, j][sel_j],
                )

    def training_step(self, batch: ConditionalQueryBatch) -> torch.Tensor:
        """Forward pass and metric logging for one training batch.

        Overrides the parent rather than inheriting it: upstream's ``training_step`` logs *per-task*
        encoder gradient norms by calling ``_encoder_grad_norm`` on ``outputs.occurs_loss`` and
        ``outputs.censor_loss``.  That split only exists for the two-headed single-query model.  The
        conditional model has **one** loss — every query position is a binary answer, and censoring
        is carried by an ordinary ``TIMELINE//END`` query rather than a separate head — so
        :class:`ConditionalQueryOutput` has neither attribute and the inherited hook raises
        ``AttributeError`` on the first step.

        The equivalent stability signal is the total pre-clipping gradient norm already logged by
        :meth:`on_before_optimizer_step`; there is no second task to compare against.
        """
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, train_split)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log the total gradient L2 norm (pre-clipping) as a training-stability signal."""
        from lightning.pytorch.utilities import grad_norm

        norms = grad_norm(self, norm_type=2)
        total = norms.get("grad_2.0_norm_total")
        if total is not None:
            self.log("train/grad_norm", float(total), on_step=True, on_epoch=False)

    @torch.no_grad()
    def predict_step(self, batch: ConditionalQueryBatch) -> dict[str, torch.Tensor]:
        """Per-position probabilities + the batch's query tensors for downstream stitching."""
        _, outputs = self.model(batch)
        return {
            "answer_probs": outputs.answer_probs.detach().cpu(),
            "q_mask": batch.q_mask.detach().cpu(),
            "q_answers": batch.q_answers.detach().cpu(),
            "q_codes": batch.q_codes.detach().cpu(),
            "q_durations": batch.q_durations.detach().cpu(),
        }

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str | None = None) -> "ConditionalQueryLightningModule":
        """Restore from a Lightning checkpoint (conditional-model variant of the parent's loader)."""
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        for k in ["model", "optimizer", "LR_scheduler"]:
            if k not in hparams:
                raise KeyError(f"Checkpoint does not contain {k} hyperparameters. Got {list(hparams.keys())}")

        model = (
            ConditionalQueryModel(**hparams["model"])
            if isinstance(hparams.get("model"), dict)
            else ConditionalQueryModel()
        )
        optimizer = _dict_to_factory(hparams["optimizer"])
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])

        # Skip the immediate parent's loader (it hardcodes EveryQueryModel) and call Lightning's.
        return super(EveryQueryLightningModule, cls).load_from_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
        )
