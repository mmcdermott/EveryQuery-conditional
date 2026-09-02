"""LightningModule for the conditional query-sequence models (both architectures).

Wraps either :class:`~every_query.model.conditional_model.ConditionalQueryEncoderDecoderModel`
or :class:`~every_query.model.conditional_ar_model.ConditionalQueryARModel` — the two share the
``(loss, ConditionalQueryOutput)`` forward contract, so one Lightning module covers both; the
checkpoint records which architecture it holds via the model's ``architecture`` hparam
(absent on encoder-decoder checkpoints, old and new, which is itself the discriminator).

Reuses the optimizer / LR-scheduler factory plumbing of
:class:`~every_query.model.lightning_module.EveryQueryLightningModule`; overrides the metric and
step logic for the per-position answer-logit output shape.

Metric conventions:
- ``censor_auc``: AUROC of position-0 answers (the always-observed censor query).
- ``occurs_auc``: AUROC over positions >= 1, restricted to observed (non-censored, non-padding)
  answers — the conditional-query analogue of the single-query model's occurs head.
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Literal

import torch
from meds import held_out_split, train_split, tuning_split
from torchmetrics.classification import BinaryAUROC

from every_query.data.seq_dataset import ConditionalQueryBatch
from every_query.model.conditional_ar_model import ConditionalQueryARModel
from every_query.model.conditional_model import (
    ANSWER_YES,
    ConditionalQueryEncoderDecoderModel,
    ConditionalQueryOutput,
)
from every_query.model.lightning_module import EveryQueryLightningModule, _dict_to_factory

logger = logging.getLogger(__name__)


class ConditionalQueryLightningModule(EveryQueryLightningModule):
    """Lightning wrapper for the conditional query-sequence model.

    Shares the parent's optimizer/LR-scheduler plumbing, so ``warmup_ratio`` means exactly what it
    means upstream: the fraction of total optimizer steps spent in LR warmup, with the step counts
    derived from ``trainer.estimated_stepping_batches`` at fit time (see the parent's
    ``configure_optimizers``).  It is validated by the parent and must lie in ``[0.0, 1.0]``.

    ``grad_norm_log_every_n_steps`` is reinterpreted here.  Upstream it throttles two
    ``torch.autograd.grad`` passes (one per task head), which is why its default is 1000.  The
    conditional model has a single loss, so there is nothing to compare and no extra autograd pass:
    :meth:`on_before_optimizer_step` just reads the ``.grad`` tensors Lightning already populated.
    That is cheap, so the default here is every step.
    """

    def __init__(
        self,
        model: ConditionalQueryEncoderDecoderModel | ConditionalQueryARModel,
        optimizer: Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer] | None = None,
        LR_scheduler: Callable[..., Any] | None = None,
        warmup_ratio: float = 0.0,
        grad_norm_log_every_n_steps: int = 1,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
            warmup_ratio=warmup_ratio,
            grad_norm_log_every_n_steps=grad_norm_log_every_n_steps,
        )
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
        """Log the total gradient L2 norm (pre-clipping) as a training-stability signal.

        Fires every ``grad_norm_log_every_n_steps`` optimizer steps (including step 0, for an early
        baseline).  This is the conditional model's stand-in for upstream's per-task encoder
        gradient norms: with one answer head there is no second task to ratio against, so the total
        pre-clipping norm is the whole signal.
        """
        if self.global_step % self.grad_norm_log_every_n_steps != 0:
            return

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

    #: Restore-time dispatch table.  The ``architecture`` model-hparam names the class; its
    #: absence means an encoder-decoder checkpoint — every checkpoint written before the
    #: decoder-only architecture existed (when the class was still named
    #: ``ConditionalQueryModel``) lacks the key, so those load unchanged under the new name.
    ARCHITECTURES: ClassVar[dict[str, type]] = {
        "encoder_decoder": ConditionalQueryEncoderDecoderModel,
        "autoregressive": ConditionalQueryARModel,
    }

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str | None = None) -> "ConditionalQueryLightningModule":
        """Restore from a Lightning checkpoint (conditional-model variant of the parent's loader)."""
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        for k in ["model", "optimizer", "LR_scheduler"]:
            if k not in hparams:
                raise KeyError(f"Checkpoint does not contain {k} hyperparameters. Got {list(hparams.keys())}")

        model_hparams = dict(hparams["model"]) if isinstance(hparams.get("model"), dict) else None
        arch = (model_hparams or {}).pop("architecture", "encoder_decoder")
        if arch not in cls.ARCHITECTURES:
            raise KeyError(
                f"Checkpoint declares unknown conditional architecture {arch!r}; "
                f"expected one of {sorted(cls.ARCHITECTURES)}."
            )
        model_cls = cls.ARCHITECTURES[arch]
        model = model_cls(**model_hparams) if model_hparams is not None else model_cls()
        optimizer = _dict_to_factory(hparams["optimizer"])
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])

        # Skip the immediate parent's loader (it hardcodes EveryQueryModel) and call Lightning's.
        return super(EveryQueryLightningModule, cls).load_from_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            LR_scheduler=LR_scheduler,
            # Checkpoints predating warmup_ratio simply had no warmup.  ``grad_norm_log_every_n_steps``
            # is deliberately absent from ``save_hyperparameters`` upstream (a logging knob only), so
            # it is not restored here either and falls back to the constructor default.
            warmup_ratio=hparams.get("warmup_ratio", 0.0),
        )
