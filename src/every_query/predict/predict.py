"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a task-query parquet (:class:`TaskQuerySchema`),
runs per-row inference, writes a :class:`PredictionSchema`-conformant parquet.  No AUCs, no
model selection, no multi-model orchestration — all of that is ``evaluate/`` or
``paper_experiments/`` territory.

Implementation is a single pass through :class:`EveryQueryPytorchDataset`:

1. Materialize the caller's flat ``TaskQuerySchema`` parquet under ``{tmpdir}/held_out/`` —
   that's the split whose ``_seeded_getitem`` injects ``subject_id`` / ``prediction_time``
   into the batch, which ``predict_step`` needs to label the output rows.
2. ``trainer.predict(dataloaders=D.test_dataloader())`` on the whole flat frame — the
   dataset handles mixed ``(query, duration_days)`` rows natively (``_seeded_getitem``
   prepends the row's own query token; ``collate`` builds per-item query / duration
   tensors).
3. Gather ``(censor_prob, occurs_prob)`` per row from ``predict_step``'s output.  Row
   identifiers (``subject_id`` / ``prediction_time`` / ``query`` / ``duration_days``)
   come from the dataset's ``schema_df`` rather than the batch — same row order
   (``test_dataloader`` uses ``shuffle=False``) but avoids the integer-query round-trip,
   which would PAD-encode any out-of-vocab input codes.
4. Join back to the original tasks frame on the full key so inherited label columns
   carry through, then ``align`` to ``PredictionSchema`` and write.
"""

import logging
import tempfile
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from meds import held_out_split
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")


def _read_and_validate_tasks(tasks_parquet: Path) -> pl.DataFrame:
    """Load a task-query parquet aligned to :class:`TaskQuerySchema`.

    Uses ``align()`` rather than ``validate()`` so the returned frame is guaranteed
    type-coerced + column-ordered — the caller gets the conversion benefit and can
    skip any defensive casting downstream.  ``align()`` still errors on missing or
    extra columns or unreconcilable type drift, so the contract is upheld.
    """
    return pl.from_arrow(TaskQuerySchema.align(pq.read_table(tasks_parquet)))


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
    """Flatten Lightning's per-batch ``predict_step`` dicts into one probabilities frame.

    Returns ``(subject_id, prediction_time, censor_prob, occurs_prob)`` in dataset
    iteration order.  Identifier columns ``query`` / ``duration_days`` come from the
    dataset's ``schema_df`` in ``main`` — the integer-encoded ``query`` in the batch
    is PAD for any out-of-vocab input code, so it isn't round-trip-safe on its own.

    ``logits_to_probs`` does a trailing ``.squeeze()``, so a single-item batch emits a
    0-d scalar tensor that ``torch.cat`` refuses to stack — ``reshape(-1)`` on every
    per-batch tensor makes the concat well-defined regardless of batch size.
    """
    if not pred_batches:
        return pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "censor_prob": pl.Float32,
                "occurs_prob": pl.Float32,
            }
        )

    def cat(key: str) -> torch.Tensor:
        return torch.cat([b[key].reshape(-1) for b in pred_batches])

    return pl.DataFrame(
        {
            "subject_id": cat("subject_id").numpy(),
            "prediction_time": cat("prediction_time").numpy(),
            "censor_prob": cat("censor_probs").numpy(),
            "occurs_prob": cat("occurs_probs").numpy(),
        }
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict")
def main(cfg: DictConfig) -> None:
    """Run inference and write :class:`PredictionSchema`-conformant output."""
    model_run_dir = Path(cfg.model_run_dir)
    tasks_parquet = Path(cfg.tasks_parquet)
    output_parquet = Path(cfg.output_parquet)
    ckpt_name = cfg.get("ckpt_name")

    logger.info(f"Loading tasks from {tasks_parquet}")
    tasks_df = _read_and_validate_tasks(tasks_parquet)
    logger.info(
        f"Loaded {tasks_df.height} tasks across {tasks_df[TaskQuerySchema.query_name].n_unique()} query codes"
    )

    # Guard against duplicate task keys.  The final join back to ``tasks_df`` keys on
    # ``(subject_id, prediction_time, query, duration_days)`` — a duplicate there
    # would produce a many-to-many join that silently multiplies output rows.  Fail
    # loudly up front rather than emit a malformed predictions parquet.
    key_cols = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]
    n_dupes = tasks_df.height - tasks_df.select(key_cols).unique().height
    if n_dupes:
        raise ValueError(
            f"Input tasks parquet contains {n_dupes} duplicate "
            f"(subject_id, prediction_time, query, duration_days) rows — each tuple must "
            f"appear at most once so the join back to labels carries a 1:1 mapping."
        )

    train_cfg, model, trainer = setup_model(model_run_dir, ckpt_name=ckpt_name)
    _warn_out_of_vocab(tasks_df, train_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write under {tmpdir}/held_out/ — that split's ``_seeded_getitem`` injects
        # subject_id / prediction_time into the batch, which ``predict_step`` needs.
        split_dir = Path(tmpdir) / held_out_split
        split_dir.mkdir()
        tasks_df.write_parquet(split_dir / "0.parquet")

        train_cfg.datamodule.config.task_labels_dir = tmpdir
        D = instantiate(train_cfg.datamodule)
        # Pass test_dataloader directly — MTD's ``Datamodule`` has no
        # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the
        # base class's ``MisconfigurationException``.  test_dataloader() is the
        # held-out loader.
        pred_batches = trainer.predict(model=model, dataloaders=D.test_dataloader(), ckpt_path=None)

        probs = _gather_probabilities(pred_batches)

        # Pull the identifier columns from the dataset's own schema_df — same row
        # order as the predictions (test_dataloader uses shuffle=False), and carries
        # the original ``query`` string + ``duration_days`` directly, sidestepping
        # the integer-query round-trip (which would PAD-encode out-of-vocab codes).
        ds = D.test_dataset
        identifiers = pl.DataFrame(
            {
                TaskQuerySchema.subject_id_name: ds.schema_df[TaskQuerySchema.subject_id_name],
                TaskQuerySchema.prediction_time_name: ds.schema_df[TaskQuerySchema.prediction_time_name],
                TaskQuerySchema.query_name: ds.query,
                TaskQuerySchema.duration_days_name: ds.duration_days,
            }
        ).with_columns(
            # ``EveryQueryPytorchDataset.__init__`` converts held_out prediction_time to
            # int64 microseconds for the batch tensor; convert back to match the tasks_df.
            pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Datetime("us")),
            pl.col(TaskQuerySchema.duration_days_name).cast(pl.Float32),
        )

    if probs.is_empty() or identifiers.height != probs.height:
        raise RuntimeError(
            f"No predictions produced, or shape mismatch between identifiers "
            f"({identifiers.height}) and prediction batches ({probs.height}).  "
            f"Input: {tasks_parquet} with {tasks_df.height} rows.  "
            f"Check that the task subjects exist in the model's held_out split."
        )

    # Horizontal concat in dataset order — ``probs`` comes out of ``trainer.predict``
    # in ``test_dataloader(shuffle=False)`` order, which matches ``schema_df`` row order.
    # Drop the duplicate identifiers from ``probs`` before concat.
    predictions = pl.concat(
        [identifiers, probs.drop([TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name])],
        how="horizontal",
    )

    # Join back to the original tasks_df so inherited label columns carry through.
    out = tasks_df.join(
        predictions,
        on=[
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            TaskQuerySchema.query_name,
            TaskQuerySchema.duration_days_name,
        ],
        how="left",
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    aligned = PredictionSchema.align(out.to_arrow())
    pl.from_arrow(aligned).write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} predictions to {output_parquet}")


if __name__ == "__main__":
    main()
