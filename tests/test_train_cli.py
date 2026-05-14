"""CLI-level integration tests for ``every_query.train``.

Tests exercise the full ``every_query.train.train:main`` Hydra entry-point (via subprocess)
using the ``_demo_train.yaml`` config against sampler-shaped task labels.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_VENV_BIN = str(Path(sys.executable).parent)


# Env vars `ensure_env()` used to gate on, before #184 replaced the blind presence
# check with config-aware ``validate_training_config``.  ``_demo_train.yaml`` passes
# every path on the CLI and sets ``trainer.logger: false``, so none of these are
# needed — `_run_train_subprocess` scrubs them by default to prove that.
_FORMERLY_GATED_ENV_VARS = ("PROJECT_DIR", "OUTPUT_DIR", "TASK_DIR", "FINAL_DATA_DIR", "WANDB_ENTITY")


def _run_train_subprocess(
    task_labels_dir: Path,
    tensorized_cohort_dir: Path,
    output_dir: Path,
    *,
    do_resume: bool = False,
    do_overwrite: bool = True,
    extra_overrides: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``python -m every_query.train.train`` as a subprocess with the demo config.

    No path env vars are set — the demo config + CLI overrides supply every path, and
    #184 made `EQ_train` validate the *resolved* config rather than env-var presence.
    Any of the formerly-gated vars inherited from the test runner's environment are
    scrubbed so this stays an honest "CLI-only, zero env vars" exercise.
    """
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    for var in _FORMERLY_GATED_ENV_VARS:
        env.pop(var, None)

    overrides = [
        f"output_dir={output_dir}",
        f"datamodule.config.task_labels_dir={task_labels_dir}",
        f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir}",
        f"do_resume={str(do_resume).lower()}",
        f"do_overwrite={str(do_overwrite).lower()}",
        f"hydra.run.dir={output_dir}/.hydra_run",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [
        sys.executable,
        "-m",
        "every_query.train.train",
        "--config-name=_demo_train",
        *overrides,
    ]

    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)


class TestTrainCliRuns:
    """Full training run via subprocess with ``_demo_train.yaml``."""

    @pytest.fixture(scope="class")
    def training_output(self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_train")
        result = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert result.returncode == 0, (
            f"train.py failed (rc={result.returncode}).\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return output_dir

    def test_config_yaml_written(self, training_output):
        assert (training_output / "config.yaml").is_file()

    def test_resolved_config_yaml_written(self, training_output):
        assert (training_output / "resolved_config.yaml").is_file()

    def test_checkpoints_directory_has_ckpt(self, training_output):
        ckpt_dir = training_output / "checkpoints"
        assert ckpt_dir.is_dir()
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        assert len(ckpts) >= 1

    def test_best_model_ckpt_exists(self, training_output):
        assert (training_output / "best_model.ckpt").is_file()


class TestTrainResume:
    """Resuming from an existing checkpoint completes without error."""

    @pytest.fixture(scope="class")
    def resumed_output(self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_resume")

        initial = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert initial.returncode == 0, (
            f"Initial training failed (rc={initial.returncode}).\nstderr:\n{initial.stderr}"
        )

        resumed = _run_train_subprocess(
            task_labels_dir,
            tensorized_cohort_dir,
            output_dir,
            do_resume=True,
            do_overwrite=False,
            extra_overrides=["trainer.max_steps=4"],
        )
        assert resumed.returncode == 0, (
            f"Resumed training failed (rc={resumed.returncode}).\nstderr:\n{resumed.stderr}"
        )
        return output_dir

    def test_checkpoints_present_after_resume(self, resumed_output):
        ckpt_dir = resumed_output / "checkpoints"
        assert ckpt_dir.is_dir()
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        assert len(ckpts) >= 1

    def test_best_model_present_after_resume(self, resumed_output):
        assert (resumed_output / "best_model.ckpt").is_file()


class TestTrainOverwriteReRun:
    """Re-running ``EQ_train`` with ``do_overwrite=True`` into a populated output_dir must leave a readable
    ``config.yaml`` + ``resolved_config.yaml`` behind.

    Regression guard for #31: the
    pre-fix code wiped the dir via ``shutil.rmtree`` but only re-saved the config on the fresh-dir
    branch, so overwrite re-runs silently produced runs that downstream tools (notably
    ``eval.py``, which loads ``resolved_config.yaml``) couldn't inspect.
    """

    @pytest.fixture(scope="class")
    def overwrite_output(self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_overwrite")

        first = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert first.returncode == 0, (
            f"Initial training failed (rc={first.returncode}).\nstderr:\n{first.stderr}"
        )
        assert (output_dir / "config.yaml").is_file()

        overwritten = _run_train_subprocess(
            task_labels_dir,
            tensorized_cohort_dir,
            output_dir,
            do_overwrite=True,
            do_resume=False,
        )
        assert overwritten.returncode == 0, (
            f"Overwrite re-run failed (rc={overwritten.returncode}).\nstderr:\n{overwritten.stderr}"
        )
        return output_dir

    def test_config_yaml_present_after_overwrite(self, overwrite_output):
        assert (overwrite_output / "config.yaml").is_file()

    def test_resolved_config_yaml_present_after_overwrite(self, overwrite_output):
        assert (overwrite_output / "resolved_config.yaml").is_file()


class TestTrainOverwriteAndResumeBothTrue:
    """When ``do_overwrite=True`` AND ``do_resume=True`` are set together, the warning at the top of ``main``
    documents that *overwrite wins* and the directory is cleared.

    Config-writing should
    therefore also behave as if we were overwriting — not silently skipped the way a pure resume
    would do.  Without the ``cfg.do_overwrite`` guard on the save, this combination produces an
    output dir whose config.yaml was wiped and never rewritten.  Regression guard for the edge
    case caught in Copilot's review of #31.
    """

    @pytest.fixture(scope="class")
    def both_flags_output(self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_both_flags")

        first = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert first.returncode == 0, (
            f"Initial training failed (rc={first.returncode}).\nstderr:\n{first.stderr}"
        )

        # Both flags simultaneously — overwrite should win (the warning at the top of main() says so).
        both = _run_train_subprocess(
            task_labels_dir,
            tensorized_cohort_dir,
            output_dir,
            do_overwrite=True,
            do_resume=True,
        )
        assert both.returncode == 0, (
            f"Run with do_overwrite=do_resume=True failed (rc={both.returncode}).\nstderr:\n{both.stderr}"
        )
        return output_dir

    def test_config_yaml_present_after_both_flags(self, both_flags_output):
        assert (both_flags_output / "config.yaml").is_file()

    def test_resolved_config_yaml_present_after_both_flags(self, both_flags_output):
        assert (both_flags_output / "resolved_config.yaml").is_file()


class TestTrainResumeRejectsStructuralDrift:
    """``do_resume=True`` must refuse to proceed when the new invocation disagrees with the resumed-from
    config on a structural key (anything not in ``ALLOWED_DIFFERENCE_KEYS``).

    Regression guard for #91.  Note ``TestTrainResume`` above covers the complementary case —
    resume with a bumped ``trainer.max_steps`` (an ALLOWED_DIFFERENCE key) succeeds.
    """

    def test_max_seq_len_drift_is_rejected(
        self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory
    ) -> None:
        output_dir = tmp_path_factory.mktemp("cli_resume_drift")

        first = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert first.returncode == 0, (
            f"Initial training failed (rc={first.returncode}).\nstderr:\n{first.stderr}"
        )

        drifted = _run_train_subprocess(
            task_labels_dir,
            tensorized_cohort_dir,
            output_dir,
            do_resume=True,
            do_overwrite=False,
            extra_overrides=["datamodule.config.max_seq_len=32"],
        )
        assert drifted.returncode != 0, (
            f"Expected resume with drifted max_seq_len to fail, but it succeeded.\n"
            f"stdout:\n{drifted.stdout}\nstderr:\n{drifted.stderr}"
        )
        assert "max_seq_len" in drifted.stderr, (
            f"Expected error to name 'max_seq_len', got:\n{drifted.stderr}"
        )


class TestTrainConfigValidation:
    """``EQ_train`` validates the *resolved* config, not env-var presence (#184).

    ``TestTrainCliRuns`` already proves the happy path runs with **zero** path env
    vars set (``_run_train_subprocess`` scrubs ``_FORMERLY_GATED_ENV_VARS``, and
    ``_demo_train.yaml`` sets ``trainer.logger: false`` so no ``WANDB_ENTITY`` is
    needed).  These tests cover the rejection path: a path that *is* supplied but
    doesn't point at a real directory must fail early with a clear message.
    """

    def test_missing_cohort_dir_rejected(self, task_labels_dir, tmp_path_factory) -> None:
        output_dir = tmp_path_factory.mktemp("cli_bad_cohort")
        bogus_cohort = output_dir / "does_not_exist_cohort"
        result = _run_train_subprocess(task_labels_dir, bogus_cohort, output_dir)
        assert result.returncode != 0, (
            f"Expected a nonexistent tensorized_cohort_dir to be rejected, but the run "
            f"succeeded.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "tensorized_cohort_dir" in result.stderr, (
            f"Expected the error to name 'tensorized_cohort_dir', got:\n{result.stderr}"
        )

    def test_missing_task_labels_dir_rejected(self, tensorized_cohort_dir, tmp_path_factory) -> None:
        output_dir = tmp_path_factory.mktemp("cli_bad_tasks")
        bogus_tasks = output_dir / "does_not_exist_tasks"
        result = _run_train_subprocess(bogus_tasks, tensorized_cohort_dir, output_dir)
        assert result.returncode != 0, (
            f"Expected a nonexistent task_labels_dir to be rejected, but the run "
            f"succeeded.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "task_labels_dir" in result.stderr, (
            f"Expected the error to name 'task_labels_dir', got:\n{result.stderr}"
        )
