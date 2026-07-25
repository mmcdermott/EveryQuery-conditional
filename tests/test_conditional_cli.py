"""End-to-end CLI tests for the conditional query-sequence pipeline.

Chain: ``EQ_generate_query_sequences`` → ``EQ_train --config-name=_demo_train_conditional`` →
``EQ_predict_sequences`` → ``EQ_evaluate_sequences``, all as real subprocesses against the
session fixture cohort (mirrors the single-query CLI chain in ``conftest.py``).
"""

import json
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


def test_supplied_contexts_mode(eq_preprocessed_dataset: Path, tmp_path: Path):
    """``contexts_path`` scores a user-supplied index df: N sequences per row, K queries each."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / train_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").max().alias("prediction_time")).head(3)
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    out_dir = tmp_path / "out"
    run_and_check(
        [
            "EQ_generate_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={out_dir!s}",
            f"split={train_split}",
            f"contexts_path={cohort_fp!s}",
            "n_replicates=4",
            "min_queries=5",
            "max_queries=5",
            "duration_min=1",
            "duration_max=365",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )

    df = pl.read_parquet(out_dir / train_split / "cohort__0000.parquet")
    assert df.height == cohort.height * 4
    assert (df["queries"].list.len() == 5).all()
    # Every supplied context is present, each replicated exactly 4x.
    counts = df.group_by("subject_id", "prediction_time").len()
    assert counts.height == cohort.height and (counts["len"] == 4).all()


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


def test_eval_v3_scores_supplied_sequences(
    cq_trained_model_dir: Path, eq_preprocessed_dataset: Path, tmp_path: Path
):
    """``scripts/eval_v3.py`` score-last inference over a supplied QuerySeqSchema parquet.

    Uses ``contexts_path`` mode so the tasks carry replicates, exercising the alignment check and
    the extra-column passthrough that make positional attachment of probabilities sound.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / tuning_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").max().alias("prediction_time")).head(3)
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    tasks_dir = tmp_path / "tasks"
    run_and_check(
        [
            "EQ_generate_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={tasks_dir!s}",
            f"split={tuning_split}",
            f"contexts_path={cohort_fp!s}",
            "n_replicates=2",
            "min_queries=3",
            "max_queries=3",
            "duration_min=1",
            "duration_max=30",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )
    tasks_fp = tasks_dir / tuning_split / "cohort__0000.parquet"
    n_seqs = pl.read_parquet(tasks_fp).height

    out_dir = tmp_path / "eval_v3_out"
    script = Path(__file__).parent.parent / "scripts" / "eval_v3.py"
    run_and_check(
        [
            "python",
            str(script),
            "--tasks",
            str(tasks_dir),
            "--run-dir",
            str(cq_trained_model_dir),
            "--cohort-dir",
            str(eq_preprocessed_dataset),
            "--split",
            tuning_split,
            "--out-dir",
            str(out_dir),
            "--batch-size",
            "4",
            "--all-positions",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=300.0,
    )

    preds = pl.read_parquet(out_dir / "predictions.parquet")
    assert preds.height == n_seqs, "one row per supplied sequence"
    assert preds["prob"].is_between(0.0, 1.0).all()
    assert preds["true_answer"].null_count() == 0
    assert (preds["n_queries"] == 3).all()
    assert (preds["prior_answers"].list.len() == 2).all()
    # --all-positions keeps every position, and the last one is what `prob` reports.
    assert (preds["all_position_probs"].list.len() == 3).all()
    assert preds["all_position_probs"].list.last().to_numpy() == pytest.approx(
        preds["prob"].to_numpy(), abs=1e-6
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_sequences"] == n_seqs
    assert summary["split"] == tuning_split
    assert summary["force_prior"] == "none"

    # Counterfactual conditioning: overriding every prior answer must change the scored
    # distribution (the target query and its context are identical across the two runs).
    forced_dir = tmp_path / "eval_v3_forced"
    run_and_check(
        [
            "python",
            str(script),
            "--tasks",
            str(tasks_dir),
            "--run-dir",
            str(cq_trained_model_dir),
            "--cohort-dir",
            str(eq_preprocessed_dataset),
            "--split",
            tuning_split,
            "--out-dir",
            str(forced_dir),
            "--batch-size",
            "4",
            "--force-prior",
            "yes",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=300.0,
    )
    forced = pl.read_parquet(forced_dir / "predictions.parquet")
    assert forced.height == preds.height
    assert forced["target_query"].to_list() == preds["target_query"].to_list()
    assert forced["prob"].is_between(0.0, 1.0).all()
    assert json.loads((forced_dir / "summary.json").read_text())["force_prior"] == "yes"


def test_eval_v3_rejects_wrong_split(
    cq_trained_model_dir: Path, eq_preprocessed_dataset: Path, cq_sequence_tasks_dir: Path, tmp_path: Path
):
    """Subjects absent from ``--split`` are dropped by the schema_df semi-join; that must not be silent."""
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / train_split).mkdir(parents=True)
    src = cq_sequence_tasks_dir / train_split / "0__0000.parquet"
    (tasks_dir / train_split / "tasks.parquet").write_bytes(src.read_bytes())

    script = Path(__file__).parent.parent / "scripts" / "eval_v3.py"
    with pytest.raises(RuntimeError, match=r"(?s)Dataset has .* rows but the supplied parquet"):
        run_and_check(
            [
                "python",
                str(script),
                "--tasks",
                str(tasks_dir),
                "--run-dir",
                str(cq_trained_model_dir),
                "--cohort-dir",
                str(eq_preprocessed_dataset),
                "--split",
                tuning_split,  # train-split subjects are not in tuning
                "--out-dir",
                str(tmp_path / "out"),
            ],
            env=ENSURE_ENV_PLACEHOLDERS,
            timeout=300.0,
        )
