"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a task-query parquet (:class:`TaskQuerySchema`),
runs per-row inference, writes a :class:`PredictionSchema`-conformant parquet.  No AUCs, no
model selection, no multi-model orchestration — all of that is ``evaluate/`` or
``paper_experiments/`` territory.

The implementation groups the input tasks by ``(query, duration_days)`` and dispatches each
group through the existing ``EveryQueryPytorchDataset`` by materializing a temporary
task-labels parquet in the shape the dataset expects (``{tmpdir}/held_out/0.parquet`` —
``held_out`` because that's the split whose ``_seeded_getitem`` injects ``subject_id`` /
``prediction_time`` into the batch, which ``predict_step`` needs to label the output rows).
This is an MVP approach — a future optimization would teach the dataset to ingest a
:class:`TaskQuerySchema` parquet directly, avoiding the tmpdir round-trip — see the
design-doc comment on #81 for context.
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
from every_query.evaluate.eval import _setup_model
from every_query.predict.schema import PredictionSchema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")


def _read_and_validate_tasks(tasks_parquet: Path) -> pl.DataFrame:
    """Load a task-query parquet and validate against :class:`TaskQuerySchema`.

    Raises with a readable message if the schema doesn't align.
    """
    table = pq.read_table(tasks_parquet)
    TaskQuerySchema.align(table)  # raises with field diff on mismatch
    return pl.from_arrow(table)


def _run_one_group(
    train_cfg: DictConfig,
    model,
    trainer,
    group_df: pl.DataFrame,
    query: str,
    duration_days: float,
) -> pl.DataFrame:
    """Run ``trainer.predict`` on one ``(query, duration_days)`` slice.

    Materializes a temporary task-labels parquet in the shape the existing
    ``EveryQueryPytorchDataset`` expects (``subject_id, prediction_time, query,
    duration_days, boolean_value`` — ``boolean_value`` is nullable per
    ``TaskQuerySchema``; at predict time we leave it null since no ground-truth label
    is needed for inference).  Points the datamodule at the tmpfile, predicts, and
    returns a frame with the model's two-head probabilities.
    """
    # Shape the slice into what the dataset expects.  `boolean_value` is left null
    # (censored-sentinel value) since it's not consumed during inference — the
    # dataset still splits it into batch.censor / batch.occurs but those are ignored
    # by the predict path.
    tmp_rows = group_df.select("subject_id", "prediction_time").with_columns(
        pl.lit(query).alias("query"),
        pl.lit(duration_days).cast(pl.Float32).alias("duration_days"),
        pl.lit(None, dtype=pl.Boolean).alias("boolean_value"),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Must land under {tmpdir}/held_out/ so the dataset loads it as the held_out
        # split — only that split's ``_seeded_getitem`` injects subject_id /
        # prediction_time into the batch, which ``predict_step`` needs for labeling.
        split_dir = Path(tmpdir) / held_out_split
        split_dir.mkdir()
        tmp_rows.write_parquet(split_dir / "0.parquet")

        train_cfg.datamodule.config.task_labels_dir = tmpdir
        D = instantiate(train_cfg.datamodule)

        # Pass the test_dataloader directly instead of the datamodule — the MTD
        # ``Datamodule`` doesn't define ``predict_dataloader``, so
        # ``trainer.predict(datamodule=D)`` would hit the base class's
        # ``MisconfigurationException``.  ``test_dataloader()`` is the held-out loader.
        pred_batches = trainer.predict(model=model, dataloaders=D.test_dataloader(), ckpt_path=None)

        s_ids, p_times, censor_probs, occurs_probs = [], [], [], []
        for b in pred_batches:
            s_ids.append(b["subject_id"].reshape(-1))
            p_times.append(b["prediction_time"].reshape(-1))
            # ``logits_to_probs`` does a trailing ``.squeeze()``, so a single-item batch
            # emits a 0-d scalar tensor that ``torch.cat`` refuses to stack.  Reshape to
            # ``(N,)`` so per-batch results always concatenate cleanly.
            censor_probs.append(b["censor_probs"].reshape(-1))
            occurs_probs.append(b["occurs_probs"].reshape(-1))
        if not s_ids:
            return pl.DataFrame(
                schema={
                    "subject_id": pl.Int64,
                    "prediction_time": pl.Datetime("us"),
                    "censor_prob": pl.Float32,
                    "occurs_prob": pl.Float32,
                }
            )

        return pl.DataFrame(
            {
                "subject_id": torch.cat(s_ids).numpy(),
                "prediction_time": torch.cat(p_times).numpy(),
                "censor_prob": torch.cat(censor_probs).numpy(),
                "occurs_prob": torch.cat(occurs_probs).numpy(),
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))


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
    task_codes = set(tasks_df["query"].unique().to_list())
    missing = task_codes - training_vocab
    if missing:
        logger.warning(
            f"{len(missing)} of {len(task_codes)} task-query codes are not in the model's training "
            f"vocabulary and will be PAD-encoded (predictions will be near-uniform for these rows): "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
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
    logger.info(f"Loaded {tasks_df.height} tasks across {tasks_df['query'].n_unique()} query codes")

    train_cfg, model, trainer = _setup_model(model_run_dir, ckpt_name=ckpt_name)
    _warn_out_of_vocab(tasks_df, train_cfg)

    # Group by (query, duration_days) — the dataset expects a single-query/duration slice.
    per_group_results: list[pl.DataFrame] = []
    for (query, duration_days), group_df in tasks_df.group_by(["query", "duration_days"]):
        logger.info(f"Predicting for query={query!r} duration_days={duration_days}")
        preds = _run_one_group(train_cfg, model, trainer, group_df, query, float(duration_days))
        if preds.is_empty():
            logger.warning(f"No predictions produced for query={query!r} duration={duration_days}")
            continue
        # Tag each result with the grouping key so the join back is unambiguous.
        preds = preds.with_columns(
            pl.lit(query).alias("query"),
            pl.lit(float(duration_days)).cast(pl.Float32).alias("duration_days"),
        )
        per_group_results.append(preds)

    if not per_group_results:
        raise RuntimeError(
            f"No predictions produced — every (query, duration) group was empty.  "
            f"Input: {tasks_parquet} with {tasks_df.height} rows."
        )

    predictions = pl.concat(per_group_results, how="vertical_relaxed")

    # Join back to the original tasks_df so inherited label columns carry through.
    out = tasks_df.join(
        predictions,
        on=["subject_id", "prediction_time", "query", "duration_days"],
        how="left",
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    PredictionSchema.align(pq.read_table(output_parquet))  # fail loudly if shape drifted
    logger.info(f"Wrote {out.height} predictions to {output_parquet}")


if __name__ == "__main__":
    main()
