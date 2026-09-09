"""``query_sequence_labeling.resolve_prediction_times``: the two guards on the Stage 3 join.

``resolve_prediction_times`` turns a Stage 2 context frame
(``subject_id``, ``shard``, ``prediction_time_index``) into real timestamps by joining the Stage 0
``_prediction_times/{shard}.parquet`` map on ``(subject_id, prediction_time_index)``.  It is a
shared seam: ``sample_multitask_sequences`` imports it and calls it as
``EQ_generate_multitask_sequences``' Stage 3 join, and ``build_sequence_index`` calls it too.

Both of the guards below exist because the *silent* failure is the dangerous one:

* a **join-key dtype mismatch** does not raise in polars — it matches nothing, and every row comes
  back with a null ``prediction_time``;
* a **left join** is used deliberately so unresolvable ``(subject_id, prediction_time_index)``
  pairs surface as nulls and raise, where an inner join would quietly drop those contexts and
  train on a silently smaller cohort.

These tests were lost when the conditional query-sequence pipeline was deleted; the function they
cover was not.
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from every_query.generate_tasks.query_sequence_labeling import resolve_prediction_times
from every_query.generate_tasks.sample_tasks import prediction_times_path


def _write_prediction_times_map(artifacts: Path, split: str, shard: str, rows: list[tuple]) -> None:
    """Write a Stage 0 ``_prediction_times/{shard}.parquet`` map with upstream's exact dtypes."""
    fp = prediction_times_path(artifacts, split, shard)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "prediction_time_index": [r[1] for r in rows],
            "time": [r[2] for r in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "prediction_time_index": pl.Int64,
            "time": pl.Datetime("us"),
        },
    ).write_parquet(fp)


def _stage2_contexts(rows: list[tuple]) -> pl.DataFrame:
    """A Stage 2 output frame: ``(subject_id, shard, prediction_time_index)``."""
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "shard": [r[1] for r in rows],
            "prediction_time_index": [r[2] for r in rows],
        },
        schema={"subject_id": pl.Int64, "shard": pl.Utf8, "prediction_time_index": pl.Int64},
    )


def test_resolve_prediction_times_raises_on_unresolvable_rank(tmp_path: Path):
    """The join is total by design — a null timestamp is a hard error, never a silent drop."""
    artifacts, split = tmp_path / "a", "train"
    _write_prediction_times_map(artifacts, split, "0", [(1, 0, datetime(2024, 1, 1))])
    with pytest.raises(ValueError, match="null prediction_time"):
        resolve_prediction_times(_stage2_contexts([(1, "0", 99)]), artifacts, split, "0")


def test_resolve_prediction_times_raises_on_join_key_dtype_drift(tmp_path: Path):
    """A dtype mismatch silently produces an all-null join in polars; it must fail loudly."""
    artifacts, split = tmp_path / "a", "train"
    _write_prediction_times_map(artifacts, split, "0", [(1, 0, datetime(2024, 1, 1))])
    drifted = _stage2_contexts([(1, "0", 0)]).with_columns(pl.col("subject_id").cast(pl.UInt32))
    with pytest.raises(ValueError, match="dtype mismatch"):
        resolve_prediction_times(drifted, artifacts, split, "0")
