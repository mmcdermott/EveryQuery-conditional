"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a task-query parquet (:class:`TaskQuerySchema`),
runs per-row inference, writes a :class:`PredictionSchema`-conformant parquet.  No AUCs, no
model selection, no multi-model orchestration — all of that is ``evaluate/`` or
``paper_experiments/`` territory.

Implementation is a single, order-preserving pass through
:class:`EveryQueryPytorchDataset`:

1. Load + align the caller's flat ``TaskQuerySchema`` parquet, write it into a tmpdir
   that we point ``task_labels_dir`` at.  The tmpdir is there only because MTD's
   ``MEDSTorchDataConfig`` requires ``task_labels_dir`` to be a directory it can
   ``rglob("*.parquet")`` — we use an empty tmpdir so the glob finds exactly our
   aligned file and nothing else.  The dataset scans across all parquets under
   ``task_labels_dir`` regardless of subdir layout, so we don't need a per-split
   subdir; the ``held_out`` split comes from ``Datamodule.test_dataset``.
2. ``trainer.predict(dataloaders=D.test_dataloader())`` — the dataset handles mixed
   ``(query, duration_days)`` rows natively (``_seeded_getitem`` prepends the row's
   own query token; ``collate`` builds per-item tensors).
3. Order-preserving ``hstack`` of ``tasks_df`` + the ``(censor_prob, occurs_prob)``
   columns: ``test_dataloader`` uses ``shuffle=False`` and the dataset reads a single
   shard in sorted file order, so per-row probabilities come back in the same order
   as ``tasks_df`` rows.  A row-count mismatch (e.g., a task subject not present in
   the tensorized held_out split) raises loudly up front.
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
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")


def _read_and_validate_tasks(tasks_parquet: Path) -> pl.DataFrame:
    """Load a task-query parquet aligned to :class:`TaskQuerySchema`.

    Uses ``align()`` so the returned frame is guaranteed type-coerced + column-ordered;
    ``align()`` still errors on missing/extra columns or unreconcilable type drift so
    the contract is upheld.
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
    tasks_parquet = Path(cfg.tasks_parquet)
    output_parquet = Path(cfg.output_parquet)
    ckpt_name = cfg.get("ckpt_name")

    logger.info(f"Loading tasks from {tasks_parquet}")
    tasks_df = _read_and_validate_tasks(tasks_parquet)
    logger.info(
        f"Loaded {tasks_df.height} tasks across {tasks_df[TaskQuerySchema.query_name].n_unique()} query codes"
    )

    train_cfg, model, trainer = setup_model(model_run_dir, ckpt_name=ckpt_name)
    _warn_out_of_vocab(tasks_df, train_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty tmpdir is required only because ``MEDSTorchDataConfig.task_labels_dir``
        # must be a directory (it's ``rglob``'d for ``*.parquet``).  The dataset
        # doesn't care about subdir structure — a single file at the top level is
        # fine.  Writing the already-aligned ``tasks_df`` rather than symlinking the
        # user's file ensures canonical dtypes land on disk before the dataset reads.
        tasks_df.write_parquet(Path(tmpdir) / "tasks.parquet")

        train_cfg.datamodule.config.task_labels_dir = tmpdir
        D = instantiate(train_cfg.datamodule)
        # Pass ``test_dataloader()`` directly — MTD's ``Datamodule`` has no
        # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the
        # base class's ``MisconfigurationException``.  ``test_dataloader`` is the
        # held_out loader with ``shuffle=False``.
        pred_batches = trainer.predict(model=model, dataloaders=D.test_dataloader(), ckpt_path=None)
        dataset_height = len(D.test_dataset)

    probs = _gather_probabilities(pred_batches)

    # Order-preservation check: the dataset reads our single task parquet in file
    # order and the dataloader is ``shuffle=False``, so predictions come back 1:1
    # matched with ``tasks_df`` rows.  A mismatch means the dataset filtered some
    # rows out — typically a task subject that isn't in the tensorized held_out
    # split — and downstream index-based pairing would silently misalign.  Fail loudly.
    if probs.height != tasks_df.height:
        raise RuntimeError(
            f"Prediction row count ({probs.height}) doesn't match task row count "
            f"({tasks_df.height}) — {tasks_df.height - dataset_height} task rows were "
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
    aligned = PredictionSchema.align(out.to_arrow())
    pl.from_arrow(aligned).write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} predictions to {output_parquet}")


if __name__ == "__main__":
    main()
