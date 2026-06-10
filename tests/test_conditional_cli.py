"""End-to-end CLI tests for the conditional query-sequence pipeline.

Chain: ``EQ_generate_query_sequences`` → ``EQ_train --config-name=_demo_train_conditional`` →
``EQ_predict_sequences`` → ``EQ_evaluate_sequences``, all as real subprocesses against the
session fixture cohort (mirrors the single-query CLI chain in ``conftest.py``).
"""

from pathlib import Path

import polars as pl
import pytest
from meds import train_split, tuning_split

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check

NEW_CLIS = ["EQ_generate_query_sequences", "EQ_predict_sequences", "EQ_evaluate_sequences"]


@pytest.mark.parametrize("cli", NEW_CLIS)
def test_cli_help_exits_zero(cli):
    # run_and_check prepends the active venv's bin to PATH and raises on nonzero exit.
    run_and_check([cli, "--help"], timeout=60.0)


@pytest.fixture(scope="session")
def cq_sequence_tasks_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    """Runs ``EQ_generate_query_sequences`` for train + tuning splits on the fixture cohort."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("cq_seq_tasks")
    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_query_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"split={split}",
                "input_shard=0",
                "task_shard=0",
                "n_contexts=8",
                "min_queries=1",
                "max_queries=5",
                "duration_min=1",
                "duration_max=30",
                "min_context_per_subject=1",
                "seed=1",
            ],
            env={"PROCESSED": str(eq_preprocessed_dataset)},
            timeout=120.0,
        )
    return out_dir


def test_generated_sequences_conform(cq_sequence_tasks_dir: Path):
    fp = cq_sequence_tasks_dir / train_split / "0__0000.parquet"
    assert fp.exists()
    df = pl.read_parquet(fp)
    assert df.height > 0
    assert set(df.columns) >= {"subject_id", "prediction_time", "queries", "durations", "answers"}

    lens_q = df["queries"].list.len()
    lens_d = df["durations"].list.len()
    lens_a = df["answers"].list.len()
    assert (lens_q == lens_d).all() and (lens_q == lens_a).all()
    assert int(lens_q.min()) >= 1 and int(lens_q.max()) <= 5

    # Binary observed-occurrence answers: every element non-null, no privileged first query.
    assert df["answers"].explode().null_count() == 0


@pytest.fixture(scope="session")
def cq_trained_model_dir(
    eq_preprocessed_dataset: Path, cq_sequence_tasks_dir: Path, tmp_path_factory
) -> Path:
    output_dir = tmp_path_factory.mktemp("cq_train_out")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={cq_sequence_tasks_dir!s}",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=300.0,
    )
    return output_dir


def test_conditional_train_produces_checkpoint(cq_trained_model_dir: Path):
    assert (cq_trained_model_dir / "checkpoints" / "last.ckpt").exists()
    assert (cq_trained_model_dir / "resolved_config.yaml").exists()


def test_conditional_predict_and_evaluate(
    cq_trained_model_dir: Path, cq_sequence_tasks_dir: Path, tmp_path: Path
):
    # Predict on the tuning split sequences (the fixture cohort has no held_out labels here).
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    src = cq_sequence_tasks_dir / tuning_split / "0__0000.parquet"
    (tasks_dir / "tasks.parquet").write_bytes(src.read_bytes())

    predictions_fp = tmp_path / "predictions.parquet"
    run_and_check(
        [
            "EQ_predict_sequences",
            f"model_run_dir={cq_trained_model_dir!s}",
            f"tasks_dir={tasks_dir!s}",
            f"output_parquet={predictions_fp!s}",
            f"split={tuning_split}",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=300.0,
    )

    preds = pl.read_parquet(predictions_fp)
    tasks = pl.read_parquet(tasks_dir / "tasks.parquet")
    assert preds.height == int(tasks["queries"].list.len().sum()), "one output row per query position"
    assert preds["answer_prob"].is_between(0.0, 1.0).all()
    # Binary observed-occurrence answers are always present (never null).
    assert preds["answer"].null_count() == 0

    metrics_stem = tmp_path / "metrics"
    run_and_check(
        [
            "EQ_evaluate_sequences",
            f"predictions_parquet={predictions_fp!s}",
            f"metrics_stem={metrics_stem!s}",
        ],
        timeout=120.0,
    )
    by_pos = pl.read_parquet(metrics_stem.with_suffix(".by_position.parquet"))
    by_query = pl.read_parquet(metrics_stem.with_suffix(".by_query.parquet"))
    assert by_pos.height >= 1 and by_pos["position"].min() == 0
    assert {"n_rows", "n_observed", "prevalence", "auroc"} <= set(by_pos.columns)
    assert by_query.height >= 1
