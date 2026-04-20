"""Subprocess integration test for ``EQ_generate_evaluation_tasks``.

Complements ``test_generate_tasks.py`` (pretraining-shape, scattered tasks) by
exercising the new evaluation-shape endpoint: a dense grid of
``subjects x sampled_times x codes x durations``.  This is what ``EQ_predict``
consumes to produce held-out predictions that ``EQ_evaluate`` then scores.

Checks:
1. Invokes the CLI against the session's preprocessed cohort.
2. Verifies the output parquet is ``TaskQuerySchema``-conformant.
3. Verifies the dense-grid shape (row count = sampled_times x |codes| x |durations|
   subjects permitting).
4. Verifies determinism — two runs with the same seed produce identical output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check
from every_query.data.schema import TaskQuerySchema

if TYPE_CHECKING:
    from pathlib import Path


def test_eq_generate_evaluation_tasks_end_to_end(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """``EQ_generate_evaluation_tasks`` writes a TaskQuerySchema-conformant dense-grid parquet."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path / "eval_tasks"

    # Small but nontrivial: two codes, three durations, two prediction times per
    # subject → at most 2 * 2 * 3 = 12 rows per subject.
    codes = ["HR", "TEMP"]
    durations = [1, 7, 30]
    pt_per_subject = 2

    run_and_check(
        [
            "EQ_generate_evaluation_tasks",
            f"data_dir={intermediate!s}",
            f"codes_dir={eq_preprocessed_dataset!s}",
            f"out_dir={out_dir!s}",
            "split=tuning",
            "input_shard=0",
            f"prediction_times_per_subject={pt_per_subject}",
            "min_context_per_subject=1",
            f"codes=[{','.join(codes)}]",
            f"durations=[{','.join(str(d) for d in durations)}]",
            "seed=42",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=120.0,
    )

    written = out_dir / "eval" / "tuning" / "0.parquet"
    assert written.is_file(), f"expected {written} to exist; got {list(out_dir.rglob('*.parquet'))}"

    # Schema conformance.
    TaskQuerySchema.align(pq.read_table(written))

    df = pl.read_parquet(written)
    assert df.height > 0, "dense grid should be non-empty for the tuning split of the demo cohort"

    # Dense-grid shape: for every sampled prediction_time, we emitted one row per
    # (code x duration) pair.  Count distinct (subject_id, prediction_time) pairs
    # in the output and confirm the grid expansion factor is exactly |codes| x |durations|.
    n_unique_contexts = (
        df.select(TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name).unique().height
    )
    assert df.height == n_unique_contexts * len(codes) * len(durations), (
        f"expected dense grid height {n_unique_contexts * len(codes) * len(durations)} "
        f"({n_unique_contexts} contexts x {len(codes)} codes x {len(durations)} durations), "
        f"got {df.height}"
    )

    # At most ``prediction_times_per_subject`` prediction times per subject.
    per_subject_times = (
        df.select(TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name)
        .unique()
        .group_by(TaskQuerySchema.subject_id_name)
        .len()
    )
    assert per_subject_times["len"].max() <= pt_per_subject, (
        f"per-subject prediction-time count exceeded cap: {per_subject_times['len'].to_list()}"
    )

    # Every (query, duration_days) cell covers the same set of (subject, time) pairs.
    cell_sizes = df.group_by(TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name).len()
    assert cell_sizes["len"].n_unique() == 1, f"cell sizes should all be equal (dense grid); got {cell_sizes}"


def test_eq_generate_evaluation_tasks_deterministic(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """Two runs with the same seed produce byte-identical output parquets."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    common_args: list[str] = [
        f"data_dir={intermediate!s}",
        f"codes_dir={eq_preprocessed_dataset!s}",
        "split=tuning",
        "input_shard=0",
        "prediction_times_per_subject=3",
        "min_context_per_subject=1",
        "codes=[HR,TEMP]",
        "durations=[1,7,30]",
        "seed=7",
    ]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_and_check(
            ["EQ_generate_evaluation_tasks", f"out_dir={out!s}", *common_args],
            env=ENSURE_ENV_PLACEHOLDERS,
            timeout=120.0,
        )

    df_a = pl.read_parquet(out_a / "eval" / "tuning" / "0.parquet").sort(
        [
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            TaskQuerySchema.query_name,
            TaskQuerySchema.duration_days_name,
        ]
    )
    df_b = pl.read_parquet(out_b / "eval" / "tuning" / "0.parquet").sort(
        [
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            TaskQuerySchema.query_name,
            TaskQuerySchema.duration_days_name,
        ]
    )
    assert df_a.equals(df_b), "EQ_generate_evaluation_tasks should be deterministic in (seed, input_shard)"
