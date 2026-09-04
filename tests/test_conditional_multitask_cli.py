"""CLI integration for multitask sampling, training, and checkpoint restoration."""

import filecmp
from pathlib import Path

import pytest
import yaml
from meds import train_split, tuning_split

from conftest import run_and_check


def test_conditional_multitask_config_help():
    run_and_check(["EQ_train", "--config-name=_demo_train_conditional_multitask_ar", "--help"], timeout=60.0)


@pytest.fixture(scope="session")
def conditional_multitask_labels_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    """Generate issue-#24 labels for both splits needed by Lightning's fit loop."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("conditional_multitask_labels")
    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_multitask_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                "num_training_examples=8",
                "num_bounds=5",
                "duration_min=0.01",
                "duration_max=2",
                "eventbound_fraction=0.5",
                "eventstart_fraction=0.25",
                "prediction_time_start_fraction=0.25",
                "start_duration_min=0.01",
                "start_duration_max=2",
                "min_prediction_times_per_subject=1",
                "max_workers=1",
                "label_chunk_rows=2",
                "seed=1",
            ],
            timeout=180.0,
        )
    return out_dir


@pytest.fixture(scope="session")
def conditional_multitask_trained_dir(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path_factory
) -> Path:
    output_dir = tmp_path_factory.mktemp("conditional_multitask_train")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional_multitask_ar",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={conditional_multitask_labels_dir!s}",
        ],
        timeout=300.0,
    )
    return output_dir


def test_conditional_multitask_train_checkpoint_and_reload(
    conditional_multitask_trained_dir: Path,
):
    checkpoint = conditional_multitask_trained_dir / "checkpoints" / "last.ckpt"
    resolved = conditional_multitask_trained_dir / "resolved_config.yaml"
    assert checkpoint.exists() and resolved.exists()

    cfg = yaml.safe_load(resolved.read_text())
    model_cfg = cfg["lightning_module"]["model"]
    assert model_cfg["_target_"].endswith("ConditionalMultitaskARModel")
    expected = cfg["datamodule"]["config"]["max_seq_len"] + 3 * model_cfg["max_windows"]
    assert model_cfg["config_overrides"]["max_position_embeddings"] == expected

    from every_query.model.conditional_multitask_ar_model import ConditionalMultitaskARModel
    from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
    from every_query.utils.model_loader import setup_model

    loaded_cfg, module, trainer = setup_model(
        conditional_multitask_trained_dir, module_cls=ConditionalMultitaskLightningModule
    )
    assert isinstance(module.model, ConditionalMultitaskARModel)
    assert loaded_cfg.lightning_module.model.max_windows == 5
    assert trainer is not None


def _train_cmd(cohort_dir: Path, labels_dir: Path, output_dir: Path, *overrides: str) -> list[str]:
    return [
        "EQ_train",
        "--config-name=_demo_train_conditional_multitask_ar",
        f"output_dir={output_dir!s}",
        f"datamodule.config.tensorized_cohort_dir={cohort_dir!s}",
        f"datamodule.config.task_labels_dir={labels_dir!s}",
        *overrides,
    ]


def test_logger_false_drops_the_lr_monitor(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path: Path
):
    """Every production config ships a ``LearningRateMonitor``; ``trainer.logger=false`` must not crash on it
    at train start (Lightning refuses the monitor without a logger)."""
    output_dir = tmp_path / "nologger"
    run_and_check(
        _train_cmd(
            eq_preprocessed_dataset,
            conditional_multitask_labels_dir,
            output_dir,
            "trainer.logger=false",
            "+trainer.callbacks.learning_rate_monitor="
            "{_target_: lightning.pytorch.callbacks.LearningRateMonitor}",
        ),
        timeout=300.0,
    )
    assert (output_dir / "best_model.ckpt").is_file()


@pytest.fixture(scope="module")
def max_steps_before_first_validation_dir(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path_factory
) -> Path:
    """One optimizer step under a fractional val cadence, i.e. training ends before the first validation ever
    records a best checkpoint; logged to CSV so hparams.yaml can be inspected."""
    output_dir = tmp_path_factory.mktemp("multitask_max_steps")
    run_and_check(
        _train_cmd(
            eq_preprocessed_dataset,
            conditional_multitask_labels_dir,
            output_dir,
            "~trainer.logger",
            "+trainer.logger={_target_: lightning.pytorch.loggers.CSVLogger, "
            "save_dir: ${trainer.default_root_dir}/loggers}",
            "trainer.check_val_every_n_epoch=1",
            "trainer.val_check_interval=0.5",
            "+trainer.max_steps=1",
        ),
        timeout=300.0,
    )
    return output_dir


def test_max_steps_before_first_validation_still_publishes_best_model(
    max_steps_before_first_validation_dir: Path,
):
    out = max_steps_before_first_validation_dir
    assert (out / "best_model.ckpt").is_file()
    # No validation ran, so there is no "best"; last.ckpt is what gets published.
    assert filecmp.cmp(out / "best_model.ckpt", out / "checkpoints" / "last.ckpt", shallow=False)


def test_csv_logger_logs_best_ckpt_path_as_a_plain_string(max_steps_before_first_validation_dir: Path):
    """``best_ckpt_path`` is logged as ``str``: a ``!!python/object`` PosixPath tag is not
    ``yaml.safe_load``-able.

    (The datamodule's own ``MEDSTorchDataConfig`` paths still land as
    tags in the same file; only the entry train.py writes is checked here.)
    """
    hparams = sorted(max_steps_before_first_validation_dir.rglob("hparams.yaml"))
    assert hparams, "CSVLogger wrote no hparams.yaml"
    lines = [ln for ln in hparams[-1].read_text().splitlines() if ln.startswith("best_ckpt_path:")]
    assert len(lines) == 1, lines
    loaded = yaml.safe_load(lines[0])
    assert isinstance(loaded["best_ckpt_path"], str)
    assert Path(loaded["best_ckpt_path"]).is_file()


# ---------------------------------------------------------------------------
# Issue #28: EQ_predict_multitask over a QuerySeqSchema grid with active starts
# ---------------------------------------------------------------------------


def test_predict_multitask_scores_a_queryseq_grid_with_active_starts(
    conditional_multitask_trained_dir: Path, eq_preprocessed_dataset: Path, tmp_path: Path
):
    """End to end: EQ_generate_evaluation_query_sequences (designed sequences with duration and
    event starts, on a supplied cohort) -> EQ_predict_multitask, one scalar prediction per row,
    with no legacy sidecar anywhere."""
    import polars as pl

    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / tuning_split).rglob("*.parquet")))
    # Prediction time = each subject's first timed event, so windows opening later have data to see.
    cohort = shard.group_by("subject_id").agg(pl.col("time").drop_nulls().min().alias("prediction_time"))
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    specs = {
        "post_admission": [
            {"query": "DISCHARGE", "start_event": "ADMISSION//PULMONARY", "duration_days": 30}
        ],
        "delayed_then_bounded": [
            {"query": "HR//value_[119.8,inf)", "start_duration_days": 1, "duration_days": 30},
            {
                "query": "DISCHARGE",
                "start_event": "ADMISSION//PULMONARY",
                "duration_days": -1,
                "bound_event": "TIMELINE//END",
            },
        ],
        "single": [["TIMELINE//END", 1]],
    }
    specs_fp = tmp_path / "specs.yaml"
    specs_fp.write_text(yaml.safe_dump(specs))
    grid_dir = tmp_path / "grid"
    run_and_check(
        [
            "EQ_generate_evaluation_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={grid_dir!s}",
            f"query_codes={eq_preprocessed_dataset!s}",
            f"split={tuning_split}",
            f"contexts_path={cohort_fp!s}",
            f"sequences_path={specs_fp!s}",
        ],
        timeout=180.0,
    )
    grid = pl.concat(
        [pl.read_parquet(fp) for fp in sorted((grid_dir / "eval" / tuning_split).glob("*.parquet"))]
    )
    assert grid.height == cohort.height * len(specs)
    assert {"start_durations", "start_events"} <= set(grid.columns)
    for name in ("_multitask_manifest.json", "eval_meta", "eval_tasks.parquet"):
        assert not list(grid_dir.parent.rglob(name)), name
    assert not list(grid_dir.parent.rglob("*.labels.npy"))

    predictions_fp = tmp_path / "predictions.parquet"
    run_and_check(
        [
            "EQ_predict_multitask",
            f"model_run_dir={conditional_multitask_trained_dir!s}",
            f"tasks_dir={grid_dir / 'eval'!s}",
            f"output_parquet={predictions_fp!s}",
            f"split={tuning_split}",
        ],
        timeout=300.0,
    )
    preds = pl.read_parquet(predictions_fp)
    assert preds.height == grid.height
    assert preds.columns == [
        "subject_id",
        "prediction_time",
        "queries",
        "start_durations",
        "start_events",
        "durations",
        "bound_events",
        "answers",
        "target_code",
        "label",
        "prob",
    ]
    assert preds["prob"].is_between(0.0, 1.0).all()
    assert preds["target_code"].to_list() == [q[-1] for q in preds["queries"].to_list()]
    assert preds["label"].to_list() == [a[-1] for a in preds["answers"].to_list()]
    # The grid rows come back verbatim, active starts included.
    key = ["subject_id", "prediction_time", "queries"]
    joined = preds.join(grid, on=key, how="inner", suffix="_grid")
    assert joined.height == grid.height
    for col in ("answers", "durations", "bound_events", "start_durations", "start_events"):
        assert joined[col].to_list() == joined[f"{col}_grid"].to_list(), col
    starts = preds.explode("start_durations", "start_events")
    assert (starts["start_events"] == "ADMISSION//PULMONARY").sum() == 2 * cohort.height
    assert (starts["start_durations"] == 1.0).sum() == cohort.height
    # Both label values are represented, so the contract is exercised on a real positive and negative.
    assert preds["label"].any() and not preds["label"].all()
    # Nothing legacy was written by inference either.
    for name in ("_multitask_manifest.json", "eval_meta", "eval_tasks.parquet"):
        assert not list(tmp_path.rglob(name)), name
    assert not list(tmp_path.rglob("*.labels.npy"))
