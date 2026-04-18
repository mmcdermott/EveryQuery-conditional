"""Metrics computation over :class:`PredictionSchema`-conformant parquets.

Pure-numpy pipeline — no Lightning trainer, no datamodule.  Consumes what ``EQ_predict``
writes (a parquet of `(subject_id, prediction_time, code, duration_days,
predicted_boolean_probability, boolean_value)`) and produces a per-group metrics table.

Initial metric is AUROC computed per `(code, duration_days)` group via ``sklearn``.  Extends
naturally to other metrics as the paper's needs solidify (calibration, Brier score, etc.) —
one-line additions in :func:`_metrics_for_group`.
"""

import logging

import polars as pl
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _auroc_or_none(y_true: list[bool], y_score: list[float]) -> float | None:
    """Return AUROC, or ``None`` if the label is single-class (AUROC undefined)."""
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _metrics_for_group(group_df: pl.DataFrame) -> dict[str, float | int | None]:
    """Compute metrics for one ``(code, duration_days)`` slice of predictions.

    Input frame carries at least ``boolean_value`` and ``predicted_boolean_probability`` for
    a single query.  Output row has ``n_rows``, ``n_positive``, and one AUROC column.  Adding
    more metrics is a one-liner here — keep metrics definitions in one place rather than
    scattered across eval configs.
    """
    # Drop rows with no ground-truth label (shouldn't be common post-#80 but guard anyway).
    labeled = group_df.filter(pl.col("boolean_value").is_not_null())
    if labeled.is_empty():
        return {"n_rows": 0, "n_positive": 0, "auroc": None}

    y_true = labeled["boolean_value"].to_list()
    y_score = labeled["predicted_boolean_probability"].to_list()
    return {
        "n_rows": labeled.height,
        "n_positive": int(sum(y_true)),
        "auroc": _auroc_or_none(y_true, y_score),
    }


def compute_metrics(predictions: pl.DataFrame) -> pl.DataFrame:
    """Group :class:`PredictionSchema`-shaped rows by ``(code, duration_days)`` and compute metrics.

    Returns a dataframe with columns ``code``, ``duration_days``, ``n_rows``, ``n_positive``,
    ``auroc``.  Groups with an all-same-class label return ``auroc = null``.

    Raises:
        ValueError: if ``predictions`` is missing the ``boolean_value`` column (ground-truth
            labels are required to compute any metric).

    Examples:
        >>> import polars as pl
        >>> preds = pl.DataFrame({
        ...     "code": ["A", "A", "B", "B"],
        ...     "duration_days": pl.Series([30.0, 30.0, 30.0, 30.0], dtype=pl.Float32),
        ...     "boolean_value": [True, False, True, True],
        ...     "predicted_boolean_probability": [0.9, 0.1, 0.8, 0.7],
        ... })
        >>> out = compute_metrics(preds)
        >>> out.sort("code")[["code", "n_rows", "n_positive", "auroc"]].to_dicts()
        [{'code': 'A', 'n_rows': 2, 'n_positive': 1, 'auroc': 1.0}, {'code': 'B', 'n_rows': 2, 'n_positive': 2, 'auroc': None}]
        >>> out["duration_days"].dtype
        Float32
    """
    if "boolean_value" not in predictions.columns:
        raise ValueError(
            "predictions is missing the 'boolean_value' column — ground-truth labels are "
            "required to compute metrics.  Run EQ_predict on labelled task data (i.e. a "
            "TaskQuerySchema parquet that already has boolean_value filled in)."
        )

    rows = []
    for (code, duration_days), group_df in predictions.group_by(["code", "duration_days"]):
        metrics = _metrics_for_group(group_df)
        rows.append({"code": code, "duration_days": float(duration_days), **metrics})

    if not rows:
        return pl.DataFrame(
            schema={
                "code": pl.Utf8,
                "duration_days": pl.Float32,
                "n_rows": pl.Int64,
                "n_positive": pl.Int64,
                "auroc": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("duration_days").cast(pl.Float32))
