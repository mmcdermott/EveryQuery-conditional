"""Metrics for conditional query-sequence predictions — ``EQ_evaluate_sequences``.

Consumes the flat per-query-position parquet written by ``EQ_predict_sequences`` and emits two
metric tables:

- ``<out stem>.by_position.parquet`` — AUROC / counts / prevalence grouped by sequence position
  (position 0 = the censor query; positions >= 1 are occurrence queries whose predictions are
  conditioned on all earlier ground-truth answers).
- ``<out stem>.by_query.parquet`` — AUROC / counts / prevalence grouped by ``(query,
  duration_bucket)`` over all non-censor positions, the conditional analogue of the single-query
  evaluator's per-task table.

Censored rows (``answer`` null) are dropped from occurrence AUROCs, mirroring the single-query
evaluator.  The censor query's own label is always observed.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from every_query.data.seq_dataset import CENSOR_QUERY_CODE
from every_query.evaluate.metrics import _auroc_or_none

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "evaluate" / "configs")

# (upper-exclusive edge in days, bucket label) pairs for the by-query table.
DURATION_BUCKETS = [(2, "1d"), (8, "2-7d"), (31, "8-30d"), (91, "31-90d"), (181, "91-180d"), (366, "181-365d")]


def _bucket_expr(col: str = "duration_days") -> pl.Expr:
    """Map a duration to a human-readable horizon bucket label."""
    hi, label = DURATION_BUCKETS[0]
    expr = pl.when(pl.col(col) < hi).then(pl.lit(label))
    for hi, label in DURATION_BUCKETS[1:]:
        expr = expr.when(pl.col(col) < hi).then(pl.lit(label))
    return expr.otherwise(pl.lit(">365d")).alias("duration_bucket")


def _group_metrics(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """AUROC / counts / prevalence per group, over observed (non-null-answer) rows."""
    rows = []
    for key, group in df.group_by(group_cols, maintain_order=True):
        observed = group.filter(pl.col("answer").is_not_null())
        y = observed["answer"].to_list()
        p = observed["answer_prob"].to_list()
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                "n_rows": group.height,
                "n_observed": observed.height,
                "n_positive": int(sum(y)),
                "prevalence": (sum(y) / len(y)) if y else None,
                "auroc": _auroc_or_none(y, p),
            }
        )
    return pl.DataFrame(rows).sort(group_cols)


def compute_sequence_metrics(predictions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute ``(by_position, by_query)`` metric tables from per-position predictions.

    Examples:
        >>> from datetime import datetime
        >>> preds = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "prediction_time": [datetime(2024, 1, 1)] * 4,
        ...     "position": [0, 1, 0, 1],
        ...     "query": ["__CENSOR__", "A", "__CENSOR__", "A"],
        ...     "duration_days": [30.0, 7.0, 30.0, 7.0],
        ...     "answer": [True, True, False, False],
        ...     "answer_prob": [0.9, 0.8, 0.2, 0.3],
        ... })
        >>> by_pos, by_query = compute_sequence_metrics(preds)
        >>> by_pos["auroc"].to_list()
        [1.0, 1.0]
        >>> by_query["query"].to_list()
        ['A']
    """
    by_position = _group_metrics(predictions, ["position"])

    non_censor = predictions.filter(pl.col("query") != CENSOR_QUERY_CODE).with_columns(_bucket_expr())
    by_query = _group_metrics(non_censor, ["query", "duration_bucket"])

    return by_position, by_query


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="evaluate_sequences")
def main(cfg: DictConfig) -> None:
    predictions_parquet = Path(cfg.predictions_parquet)
    out_stem = Path(cfg.metrics_stem)

    predictions = pl.read_parquet(predictions_parquet)
    logger.info(f"Loaded {predictions.height} per-position predictions from {predictions_parquet}")

    by_position, by_query = compute_sequence_metrics(predictions)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    by_position.write_parquet(out_stem.with_suffix(".by_position.parquet"))
    by_query.write_parquet(out_stem.with_suffix(".by_query.parquet"))
    logger.info(
        f"Wrote {by_position.height} position rows and {by_query.height} query rows to "
        f"{out_stem}.by_position/.by_query.parquet"
    )


if __name__ == "__main__":
    main()
