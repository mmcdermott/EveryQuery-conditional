"""CLI integration for multitask sampling, training, and checkpoint restoration."""

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
