"""Inference entry point for conditional query sequences — ``EQ_predict_sequences``.

Takes a trained conditional-model run directory and a directory of
:class:`~every_query.data.schema.QuerySeqSchema` parquets, runs teacher-forced inference (each
query's prediction is conditioned on the *ground-truth* answers of all prior queries in its
sequence), and writes a flat per-query-position parquet with columns:

- ``subject_id``, ``prediction_time`` — the patient context;
- ``position`` — 0-based index in the query sequence (0 = censor query);
- ``query`` — MEDS code string (position 0 holds the censor sentinel);
- ``duration_days`` — the query horizon;
- ``answer`` — ground-truth nullable boolean (null = censored);
- ``answer_prob`` — the model's predicted probability that the answer is True.

Row order: input sequence-row order, exploded by position — so consumers can reassemble
sequences by grouping on ``(subject_id, prediction_time)`` in order.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import torch
from hydra.utils import instantiate
from meds import held_out_split, tuning_split
from omegaconf import DictConfig

from every_query.data.seq_dataset import ANSWERS_COL, DURATIONS_COL, QUERIES_COL
from every_query.model.conditional_lightning import ConditionalQueryLightningModule
from every_query.predict.predict import _validate_tasks_dir
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")

_SPLIT_TO_DATAMODULE_ATTRS: dict[str, tuple[str, str]] = {
    tuning_split: ("val_dataset", "val_dataloader"),
    held_out_split: ("test_dataset", "test_dataloader"),
}


def predictions_to_df(schema_df: pl.DataFrame, predictions: list[dict[str, torch.Tensor]]) -> pl.DataFrame:
    """Stitch batch prediction dicts back onto the dataset's sequence rows and explode.

    ``schema_df`` rows (with list columns ``queries`` / ``durations`` / ``answers``) are in
    dataloader order; ``predictions`` is the corresponding list of ``predict_step`` outputs.
    The per-batch ``answer_probs`` are unpadded via ``q_mask`` so each sequence contributes
    exactly ``len(queries)`` probabilities.
    """
    probs_per_row: list[list[float]] = []
    for batch_pred in predictions:
        probs = batch_pred["answer_probs"]
        mask = batch_pred["q_mask"]
        for i in range(probs.shape[0]):
            probs_per_row.append(probs[i][mask[i]].tolist())

    if len(probs_per_row) != schema_df.height:
        raise RuntimeError(
            f"Prediction row count ({len(probs_per_row)}) != dataset row count ({schema_df.height})."
        )

    out = schema_df.select(
        "subject_id",
        "prediction_time",
        pl.col(QUERIES_COL).alias("query"),
        pl.col(DURATIONS_COL).alias("duration_days"),
        pl.col(ANSWERS_COL).alias("answer"),
    ).with_columns(pl.Series("answer_prob", probs_per_row, dtype=pl.List(pl.Float32)))

    n_q = pl.col("query").list.len()
    out = out.with_columns(pl.int_ranges(0, n_q).alias("position"))
    return out.explode("position", "query", "duration_days", "answer", "answer_prob").select(
        "subject_id", "prediction_time", "position", "query", "duration_days", "answer", "answer_prob"
    )


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict_sequences")
def main(cfg: DictConfig) -> None:
    model_run_dir = Path(cfg.model_run_dir)
    tasks_dir = Path(cfg.tasks_dir)
    output_parquet = Path(cfg.output_parquet)
    split = cfg.get("split", held_out_split)
    overwrite = bool(cfg.get("overwrite", False))

    if split not in _SPLIT_TO_DATAMODULE_ATTRS:
        raise ValueError(f"split must be one of {sorted(_SPLIT_TO_DATAMODULE_ATTRS)}, got {split!r}.")
    if output_parquet.exists() and not overwrite:
        raise FileExistsError(f"output_parquet {output_parquet} already exists; pass overwrite=true.")

    _validate_tasks_dir(tasks_dir)
    logger.info(f"Loading query sequences from {tasks_dir} (split={split})")

    train_cfg, model, trainer = setup_model(
        model_run_dir, ckpt_name=cfg.get("ckpt_name"), module_cls=ConditionalQueryLightningModule
    )

    batch_size_override = cfg.get("batch_size")
    if batch_size_override is not None:
        train_cfg.datamodule.batch_size = int(batch_size_override)
    train_cfg.datamodule.config.task_labels_dir = str(tasks_dir)
    D = instantiate(train_cfg.datamodule)

    dataset_attr, dataloader_attr = _SPLIT_TO_DATAMODULE_ATTRS[split]
    dataset = getattr(D, dataset_attr)
    dataloader = getattr(D, dataloader_attr)()

    sampler_cls = type(getattr(dataloader, "sampler", None)).__name__
    if sampler_cls != "SequentialSampler":
        raise RuntimeError(f"{dataloader_attr} must use SequentialSampler; got {sampler_cls!r}.")

    logger.info(f"Loaded {len(dataset)} query sequences")

    predictions = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None)

    out = predictions_to_df(dataset.schema_df, predictions)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} per-position predictions to {output_parquet}")


if __name__ == "__main__":
    main()
