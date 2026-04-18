"""End-to-end tests for the ``EQ_train`` CLI.

Covers the ``eq_trained_model_dir`` fixture (from #105) plus a resume-differential that
catches silent ``do_resume=True`` regressions — a reinitialize-on-resume bug would leave
the global_step unchanged and fail this test.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check

if TYPE_CHECKING:
    from pathlib import Path


def _highest_step_checkpoint(output_dir: Path) -> tuple[Path, dict]:
    """Return the (path, loaded_ckpt_dict) for the checkpoint with the largest ``global_step``.

    Lightning's ``ModelCheckpoint(save_last=True)`` versions ``last.ckpt`` as ``last-v1.ckpt``
    when resuming into a directory that already has a ``last.ckpt`` — it keeps the old one in
    place.  Scanning all ``*.ckpt`` and picking the max-step one is robust to that behavior.
    """
    ckpts = sorted((output_dir / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"no checkpoints under {output_dir / 'checkpoints'}"
    best_path, best_ckpt, best_step = None, None, -1
    for fp in ckpts:
        c = torch.load(fp, map_location="cpu", weights_only=False)
        if c["global_step"] > best_step:
            best_path, best_ckpt, best_step = fp, c, c["global_step"]
    return best_path, best_ckpt


def test_resolved_config_captures_overrides(eq_trained_model_dir: Path) -> None:
    """`resolved_config.yaml` records the fully-interpolated config with CLI overrides baked in.

    Catches regressions where Hydra overrides are silently dropped between CLI parse and on-disk persistence —
    downstream tools (evaluate, reproduce-this-run scripts) depend on this file being faithful.
    """
    resolved = eq_trained_model_dir / "resolved_config.yaml"
    assert resolved.exists(), f"resolved_config.yaml not written to {eq_trained_model_dir}"
    cfg = OmegaConf.load(resolved)
    # output_dir override → must reflect the actual tmp run dir, not the ??? sentinel.
    assert cfg.output_dir == str(eq_trained_model_dir), (
        f"resolved_config.output_dir={cfg.output_dir!r} does not match actual run dir "
        f"{eq_trained_model_dir!r}"
    )
    # task_labels_dir / tensorized_cohort_dir also came through overrides.
    assert cfg.datamodule.config.task_labels_dir, "task_labels_dir missing from resolved config"
    assert cfg.datamodule.config.tensorized_cohort_dir, "tensorized_cohort_dir missing from resolved config"


def test_checkpoint_metadata(eq_trained_model_dir: Path) -> None:
    """Lightning checkpoint carries the expected training-state keys and matches ``max_steps=2``."""
    _, ckpt = _highest_step_checkpoint(eq_trained_model_dir)
    for key in ("state_dict", "optimizer_states", "global_step", "epoch", "hyper_parameters"):
        assert key in ckpt, f"checkpoint missing expected key {key!r}"
    # Demo config has max_steps=2; global_step advances once per optimizer step.
    assert ckpt["global_step"] == 2, (
        f"demo training ran for {ckpt['global_step']} steps, expected 2 per _demo_train.yaml"
    )
    # State dict is non-empty and contains ModernBERT backbone parameters.
    assert len(ckpt["state_dict"]) > 0
    assert any("HF_model" in k or "model." in k for k in ckpt["state_dict"]), (
        f"state_dict doesn't appear to contain the model backbone: keys={list(ckpt['state_dict'])[:3]}"
    )


def test_resume_advances_global_step(
    eq_trained_model_dir: Path,
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory,
) -> None:
    """`do_resume=True` picks up where training left off and advances training.

    MEICAR-style differential.  Copy the fixture's trained run to a fresh dir, resume with
    higher ``max_steps``, assert the resumed checkpoint has advanced past the starting
    global_step.  A silent reinit-on-resume bug (pre-#86 style) would leave global_step at
    the original value and fail this test.
    """
    # Stage the fixture's outputs into a new dir that we're free to mutate without polluting
    # the session-scoped fixture for other tests.
    resume_dir = tmp_path_factory.mktemp("eq_train_resume")
    for item in ("config.yaml", "resolved_config.yaml", "checkpoints"):
        src = eq_trained_model_dir / item
        dst = resume_dir / item
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    # NOTE: copying config.yaml is load-bearing — train.py uses its presence (cfg_path.exists())
    # to decide whether the output dir is "populated" and to enter the do_resume branch.  Without
    # it, do_resume=True would be silently ignored and a fresh training run would start instead.

    _, start_ckpt = _highest_step_checkpoint(resume_dir)
    starting_step = start_ckpt["global_step"]
    assert starting_step == 2, f"sanity: fixture trained to step 2, got {starting_step}"

    # Resume for 2 more steps.  do_overwrite must be False — the demo config defaults it to True,
    # and the train entry point prioritizes overwrite when both are set.
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train",
            f"output_dir={resume_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={eq_sampled_tasks_dir!s}",
            "do_overwrite=False",
            "do_resume=True",
            "trainer.max_steps=4",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=300.0,
    )

    _, resumed_ckpt = _highest_step_checkpoint(resume_dir)
    assert resumed_ckpt["global_step"] > starting_step, (
        f"do_resume=True did not advance global_step "
        f"(starting={starting_step}, after resume={resumed_ckpt['global_step']}).  "
        f"Likely silent reinit-on-resume regression."
    )
    assert resumed_ckpt["global_step"] == 4, (
        f"resume with max_steps=4 advanced to global_step={resumed_ckpt['global_step']}, expected 4"
    )
