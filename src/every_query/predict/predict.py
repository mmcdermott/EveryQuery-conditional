"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a directory of task-query parquets
(:class:`TaskQuerySchema`), runs per-row inference, writes a
:class:`PredictionSchema`-conformant parquet.  No AUCs, no model selection, no
multi-model orchestration — all of that is ``evaluate/`` or ``paper_experiments/``
territory.

Implementation is a single, order-preserving pass through
:class:`EveryQueryPytorchDataset`:

1. Validate ``tasks_dir`` (must be a directory of ``.parquet`` files; warn if more
   than one file is present) and point MTD's ``task_labels_dir`` at it directly.
   No rewrite, no tmpdir, no input-side type coercion — if the parquets are
   malformed the dataset will surface that, and the output gets a canonical dtype
   pass via ``PredictionSchema.align`` at write.
2. ``trainer.predict(dataloaders=D.test_dataloader())`` — the dataset handles mixed
   ``(query, duration_days)`` rows natively (``_seeded_getitem`` prepends the
   row's own query token; ``collate`` builds per-item tensors).
3. Order-preserving ``hstack`` of ``tasks_df`` + the ``(censor_prob, occurs_prob)``
   columns.  ``tasks_df`` is built by reading the same sorted
   ``rglob("*.parquet")`` MTD uses internally, so both frames share the same row
   order.  A row-count mismatch (e.g. a task subject not present in the tensorized
   held_out split) raises loudly up front.

Upstream note: this module re-reads the task parquets that MTD already read
internally to build ``self.schema_df`` (we need ``tasks_df`` for the ``hstack``
that preserves inherited ``LabelSchema`` columns).  That duplicate read would
disappear if MTD exposed an ordered view of the raw input rows — tracked
separately so we can collapse this once it's available.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")


def _task_parquets(tasks_dir: Path) -> list[Path]:
    """Validate ``tasks_dir`` and return the sorted list of task parquet files.

    Raises ``NotADirectoryError`` if ``tasks_dir`` isn't a directory and ``ValueError``
    if it contains any non-parquet files.  Warns if more than one parquet is present
    — the typical inference use case is a single flat parquet; multiple files usually
    indicates the caller pointed us at the training task-labels directory by mistake.

    Sorted-rglob ordering matches ``MEDSTorchDataConfig.task_labels_fps`` exactly, so
    callers reading via this function can trust their row order lines up with the
    dataset's.
    """
    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"tasks_dir must be a directory, got {tasks_dir}")

    all_files = sorted(p for p in tasks_dir.rglob("*") if p.is_file())
    non_parquet = [p for p in all_files if p.suffix != ".parquet"]
    if non_parquet:
        raise ValueError(
            f"tasks_dir {tasks_dir} contains non-parquet files: "
            f"{[str(p.relative_to(tasks_dir)) for p in non_parquet[:5]]}"
            f"{'...' if len(non_parquet) > 5 else ''}.  Point EQ_predict at a directory "
            f"containing only TaskQuerySchema-conformant parquet files."
        )

    parquets = [p for p in all_files if p.suffix == ".parquet"]
    if len(parquets) > 1:
        logger.warning(
            f"tasks_dir {tasks_dir} contains {len(parquets)} parquet files; all will be "
            f"concatenated.  The typical inference use case is a single flat parquet."
        )
    return parquets


def _warn_out_of_vocab(tasks_df: pl.DataFrame, train_cfg: DictConfig) -> None:
    """Warn if any task-query codes are missing from the trained model's vocabulary.

    Out-of-vocab query codes survive the predict loop (``encode_query`` silently falls
    back to ``PAD_INDEX``) but produce effectively-uniform predictions.  Log a warning
    at startup so the caller notices rather than silently getting garbage probabilities.
    """
    metadata_fp = Path(train_cfg.datamodule.config.tensorized_cohort_dir) / "metadata" / "codes.parquet"
    if not metadata_fp.is_file():
        logger.warning(f"Cannot resolve training vocabulary — no codes metadata at {metadata_fp}")
        return
    training_vocab = set(pl.read_parquet(metadata_fp, columns=["code"])["code"].to_list())
    task_codes = set(tasks_df[TaskQuerySchema.query_name].unique().to_list())
    missing = task_codes - training_vocab
    if missing:
        logger.warning(
            f"{len(missing)} of {len(task_codes)} task-query codes are not in the model's training "
            f"vocabulary and will be PAD-encoded (predictions will be near-uniform for these rows): "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )


def _gather_probabilities(pred_batches: list[dict[str, torch.Tensor]]) -> pl.DataFrame:
    """Flatten Lightning's per-batch ``predict_step`` dicts into a two-column probabilities frame.

    Returns ``(censor_prob, occurs_prob)`` in dataset iteration order.  Column names
    come from :class:`PredictionSchema` so a rename on the schema flows through.

    ``logits_to_probs`` does a trailing ``.squeeze()``, so a single-item batch emits a
    0-d scalar tensor that ``torch.cat`` refuses to stack — ``reshape(-1)`` on every
    per-batch tensor makes the concat well-defined regardless of batch size.
    """
    if not pred_batches:
        return pl.DataFrame(
            schema={
                PredictionSchema.censor_prob_name: pl.Float32,
                PredictionSchema.occurs_prob_name: pl.Float32,
            }
        )

    def cat(key: str) -> pl.Series:
        # ``PredictionSchema.{censor,occurs}_prob`` are ``pa.float32()`` — cast here
        # so the final ``PredictionSchema.align`` doesn't have to coerce f64 → f32.
        return pl.Series(torch.cat([b[key].reshape(-1) for b in pred_batches]).numpy()).cast(pl.Float32)

    return pl.DataFrame(
        {
            PredictionSchema.censor_prob_name: cat("censor_probs"),
            PredictionSchema.occurs_prob_name: cat("occurs_probs"),
        }
    )


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict")
def main(cfg: DictConfig) -> None:
    """Run inference and write :class:`PredictionSchema`-conformant output."""
    model_run_dir = Path(cfg.model_run_dir)
    tasks_dir = Path(cfg.tasks_dir)
    output_parquet = Path(cfg.output_parquet)
    ckpt_name = cfg.get("ckpt_name")

    parquets = _task_parquets(tasks_dir)
    logger.info(f"Loading tasks from {tasks_dir} ({len(parquets)} parquet file(s))")
    # ``pl.concat`` on an empty list raises; guard.
    if not parquets:
        raise ValueError(f"tasks_dir {tasks_dir} contains no parquet files")
    # Match MTD's sorted-rglob ordering so hstack alignment is guaranteed.
    tasks_df = pl.concat([pl.read_parquet(fp) for fp in parquets], how="vertical")
    logger.info(
        f"Loaded {tasks_df.height} tasks across {tasks_df[TaskQuerySchema.query_name].n_unique()} query codes"
    )

    train_cfg, model, trainer = setup_model(model_run_dir, ckpt_name=ckpt_name)
    _warn_out_of_vocab(tasks_df, train_cfg)

    train_cfg.datamodule.config.task_labels_dir = str(tasks_dir)
    D = instantiate(train_cfg.datamodule)
    # Pass ``test_dataloader()`` directly — MTD's ``Datamodule`` has no
    # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the base
    # class's ``MisconfigurationException``.  ``test_dataloader`` is the held_out
    # loader with ``shuffle=False``.
    pred_batches = trainer.predict(model=model, dataloaders=D.test_dataloader(), ckpt_path=None)

    probs = _gather_probabilities(pred_batches)

    # Order-preservation check: the dataset reads ``sorted(tasks_dir.rglob("*.parquet"))``
    # exactly like ``_task_parquets`` above, and the dataloader is ``shuffle=False``, so
    # predictions come back 1:1 matched with ``tasks_df`` rows.  A mismatch means the
    # dataset filtered some rows out — typically a task subject not in the tensorized
    # held_out split — and downstream index-based pairing would silently misalign.
    # Fail loudly.
    if probs.height != tasks_df.height:
        raise RuntimeError(
            f"Prediction row count ({probs.height}) doesn't match task row count "
            f"({tasks_df.height}) — {tasks_df.height - probs.height} task rows were "
            f"dropped by the dataset, most likely because those subjects aren't present "
            f"in the tensorized held_out split at "
            f"{train_cfg.datamodule.config.tensorized_cohort_dir}.  Predict only on "
            f"subjects whose tensorized data lives in the held_out split."
        )

    # Horizontal concat — ``probs`` is in dataset order, which matches ``tasks_df``.
    # No key-based join, so all inherited ``LabelSchema`` columns on ``tasks_df``
    # carry through unchanged.
    out = tasks_df.hstack(probs)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    # Final canonicalization pass — ``align`` casts / reorders columns to match the
    # schema's canonical arrow layout, so the written parquet is schema-conformant
    # regardless of whatever dtypes the input happened to have.
    aligned = PredictionSchema.align(out.to_arrow())
    pl.from_arrow(aligned).write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} predictions to {output_parquet}")


if __name__ == "__main__":
    main()
