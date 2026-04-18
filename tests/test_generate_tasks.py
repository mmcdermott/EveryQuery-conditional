"""End-to-end tests for the ``EQ_generate_tasks`` CLI (sample_tasks).

Runs the real console script against a preprocessed cohort (via the
``eq_sampled_tasks_dir`` fixture) and verifies the label schema, label semantics,
reproducibility, and that the ``n_tasks`` knob is actually honored.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl

from conftest import run_and_check

if TYPE_CHECKING:
    from pathlib import Path


# Shared CLI overrides matching the foundation fixture.  Duplicated (not factored into a helper)
# because the fixture's values live in conftest.py and the per-test-run invocations need to diverge
# from them on specific fields (n_tasks, out_dir) — a helper abstraction would hide the differences
# we are explicitly testing for.
_DURATION_MIN = 1
_DURATION_MAX = 30
_MIN_CONTEXT_PER_SUBJECT = 1


def _read_train_labels(tasks_dir: Path) -> pl.DataFrame:
    """Load + concatenate all labeled parquets under the train split of *tasks_dir*."""
    fps = sorted((tasks_dir / "train").glob("*.parquet"))
    assert fps, f"no labeled parquets found under {tasks_dir / 'train'}"
    return pl.concat([pl.read_parquet(fp) for fp in fps], how="vertical")


def test_sampler_output_schema(eq_sampled_tasks_dir: Path) -> None:
    """The labeled shards have exactly the columns MTD expects, with the right dtypes."""
    df = _read_train_labels(eq_sampled_tasks_dir)
    expected_schema = {
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime,
        "boolean_value": pl.Boolean,
        "occurs": pl.Boolean,
        "query": pl.Utf8,
        "duration_days": pl.Int64,
    }
    # Compare by (name, outer-dtype) so Datetime subtypes (μs vs ns) don't break the test.
    actual = {c: (type(dt) if dt.base_type() == pl.Datetime else dt) for c, dt in df.schema.items()}
    expected = {
        c: (type(dt) if isinstance(dt, type) and dt is pl.Datetime else dt)
        for c, dt in expected_schema.items()
    }
    for col, expected_dtype in expected.items():
        assert col in actual, f"column {col!r} missing from sampler output: {df.columns}"
        if expected_dtype is type(pl.Datetime):
            assert df[col].dtype.base_type() == pl.Datetime, f"{col} is not Datetime: {df[col].dtype}"
        else:
            assert df[col].dtype == expected_dtype, (
                f"column {col!r} has dtype {df[col].dtype}, expected {expected_dtype}"
            )


def test_label_semantics(eq_sampled_tasks_dir: Path) -> None:
    """Labels obey the documented invariants: duration>0, no nulls in key cols."""
    df = _read_train_labels(eq_sampled_tasks_dir)
    assert df.height > 0, "sampler produced an empty labeled shard"
    # Key columns are never null — the sampler guarantees this per issue #80 design.
    for col in ("subject_id", "prediction_time", "query", "duration_days"):
        nulls = df[col].null_count()
        assert nulls == 0, f"column {col!r} has {nulls} null values in sampler output"
    # Durations are strictly positive integers within the configured sampling bounds.
    assert df["duration_days"].min() >= _DURATION_MIN
    assert df["duration_days"].max() <= _DURATION_MAX


def test_reproducible_with_same_seed(
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory,
) -> None:
    """Two sampler runs with identical seed + sampling params produce identical labels.

    Regression guard: non-deterministic sampling (e.g., forgetting to seed numpy / polars before
    a shuffle) would silently change eval labels between runs and produce spurious AUROC drift.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    rerun_out = tmp_path_factory.mktemp("eq_tasks_rerun")
    run_and_check(
        [
            "EQ_generate_tasks",
            f"data_dir={intermediate!s}",
            f"codes_dir={eq_preprocessed_dataset!s}",
            f"out_dir={rerun_out!s}",
            "split=train",
            "input_shard=0",
            "task_shard=0",
            "n_tasks=8",
            "contexts_per_task=2",
            f"duration_min={_DURATION_MIN}",
            f"duration_max={_DURATION_MAX}",
            f"min_context_per_subject={_MIN_CONTEXT_PER_SUBJECT}",
            "seed=1",
        ],
        timeout=120.0,
    )

    fixture_labels = _read_train_labels(eq_sampled_tasks_dir)
    rerun_labels = pl.read_parquet(sorted((rerun_out / "train").glob("*.parquet")))

    # Compare by row content (sorted) since file-byte equality is flaky across polars versions.
    # The invariant we care about: same seed → same labeled rows.
    sort_cols = ["subject_id", "prediction_time", "query", "duration_days"]
    a = fixture_labels.filter(pl.col("subject_id").is_not_null()).sort(sort_cols)
    b = rerun_labels.filter(pl.col("subject_id").is_not_null()).sort(sort_cols)
    assert a.equals(b), (
        "two EQ_generate_tasks runs with identical seed produced different labeled output.\n"
        f"fixture hash: {hashlib.sha256(a.write_parquet_buffered().getbuffer()).hexdigest()[:16]}  "
        f"rerun hash: {hashlib.sha256(b.write_parquet_buffered().getbuffer()).hexdigest()[:16]}"
    )


def test_n_tasks_knob_is_honored(
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory,
) -> None:
    """Differential: raising ``n_tasks`` produces more index rows (``n_tasks * contexts_per_task``).

    Catches silent flag-drop — a shape-only test would pass even if ``n_tasks`` were ignored.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    big_out = tmp_path_factory.mktemp("eq_tasks_big")
    run_and_check(
        [
            "EQ_generate_tasks",
            f"data_dir={intermediate!s}",
            f"codes_dir={eq_preprocessed_dataset!s}",
            f"out_dir={big_out!s}",
            "split=train",
            "input_shard=0",
            "task_shard=0",
            "n_tasks=16",  # 2x the fixture's n_tasks=8
            "contexts_per_task=2",
            f"duration_min={_DURATION_MIN}",
            f"duration_max={_DURATION_MAX}",
            f"min_context_per_subject={_MIN_CONTEXT_PER_SUBJECT}",
        ],
        timeout=120.0,
    )

    small_df = _read_train_labels(eq_sampled_tasks_dir)
    big_df = pl.read_parquet(sorted((big_out / "train").glob("*.parquet")))

    assert big_df.height > small_df.height, (
        f"n_tasks=16 did not produce more labeled rows than n_tasks=8 "
        f"(big={big_df.height}, small={small_df.height}).  The n_tasks knob may be ignored."
    )
    # Expected: big ≈ 2x small, since contexts_per_task is the same.  Tolerant assertion since some
    # rows can be filtered out during the join_asof labeling step.
    assert big_df.height >= int(1.5 * small_df.height), (
        f"n_tasks=16 only produced {big_df.height} rows vs. {small_df.height} at n_tasks=8; "
        f"expected roughly 2x (allowing 25%% label-drop tolerance)"
    )
