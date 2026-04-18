"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a task-query parquet (:class:`TaskQuerySchema`),
runs per-row inference, writes a :class:`PredictionSchema`-conformant parquet.  No AUCs, no
model selection, no multi-model orchestration — all of that is ``evaluate/`` or
``paper_experiments/`` territory.

The implementation groups the input tasks by ``(code, duration_days)`` and dispatches each
group through the existing ``EveryQueryPytorchDataset`` by materializing a temporary
task-labels parquet in the shape the dataset expects.  This is an MVP approach — a future
optimization would teach the dataset to ingest a :class:`TaskQuerySchema` parquet directly,
avoiding the tmpdir round-trip — see the design-doc comment on #81 for context.
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
    code: str,
    duration_days: float,
) -> pl.DataFrame:
    """Run ``trainer.predict`` on one (code, duration) slice.

    Materializes a temporary task-labels parquet in the shape the existing
    ``EveryQueryPytorchDataset`` expects (columns ``subject_id, prediction_time, query,
    duration_days, boolean_value, occurs``), points the datamodule at it, predicts, and
    returns a frame with the model's per-row probabilities.
    """
    # Shape the slice into what the dataset expects.
    tmp_rows = group_df.select("subject_id", "prediction_time").with_columns(
        pl.lit(code).alias("query"),
        pl.lit(int(round(duration_days))).cast(pl.Int64).alias("duration_days"),
        pl.lit(False).alias("boolean_value"),  # placeholder — ignored at predict time
        pl.lit(False).alias("occurs"),  # placeholder
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_parquet = Path(tmpdir) / "tasks.parquet"
        tmp_rows.write_parquet(tmp_parquet)

        train_cfg.datamodule.config.task_labels_dir = tmpdir
        D = instantiate(train_cfg.datamodule)

        pred_batches = trainer.predict(model=model, datamodule=D, ckpt_path=None)

        s_ids, p_times, probs = [], [], []
        for b in pred_batches:
            s_ids.append(b["subject_id"])
            p_times.append(b["prediction_time"])
            probs.append(b["occurs_probs"])
        if not s_ids:
            return pl.DataFrame(
                schema={
                    "subject_id": pl.Int64,
                    "prediction_time": pl.Datetime("us"),
                    "predicted_boolean_probability": pl.Float32,
                }
            )

        return pl.DataFrame(
            {
                "subject_id": torch.cat(s_ids).numpy(),
                "prediction_time": torch.cat(p_times).numpy(),
                "predicted_boolean_probability": torch.cat(probs).numpy(),
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
    logger.info(f"Loaded {tasks_df.height} tasks across {tasks_df['code'].n_unique()} codes")

    train_cfg, model, trainer = _setup_model(model_run_dir, ckpt_name=ckpt_name)

    # Group by (code, duration_days) — the dataset expects a single-code/duration slice.
    per_group_results: list[pl.DataFrame] = []
    for (code, duration_days), group_df in tasks_df.group_by(["code", "duration_days"]):
        logger.info(f"Predicting for code={code!r} duration_days={duration_days}")
        preds = _run_one_group(train_cfg, model, trainer, group_df, code, float(duration_days))
        if preds.is_empty():
            logger.warning(f"No predictions produced for code={code!r} duration={duration_days}")
            continue
        # Tag each result with the grouping key so the join back is unambiguous.
        preds = preds.with_columns(
            pl.lit(code).alias("code"),
            pl.lit(float(duration_days)).cast(pl.Float32).alias("duration_days"),
        )
        per_group_results.append(preds)

    if not per_group_results:
        raise RuntimeError(
            f"No predictions produced — every (code, duration) group was empty.  "
            f"Input: {tasks_parquet} with {tasks_df.height} rows."
        )

    predictions = pl.concat(per_group_results, how="vertical_relaxed")

    # Join back to the original tasks_df so inherited label columns carry through.
    out = tasks_df.join(
        predictions,
        on=["subject_id", "prediction_time", "code", "duration_days"],
        how="left",
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    PredictionSchema.align(pq.read_table(output_parquet))  # fail loudly if shape drifted
    logger.info(f"Wrote {out.height} predictions to {output_parquet}")


if __name__ == "__main__":
    main()
