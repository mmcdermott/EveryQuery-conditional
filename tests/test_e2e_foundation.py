"""Foundation-level integration tests — just verify the CLI-chain fixtures wire up.

Each real per-stage test (``test_process_data.py``, ``test_generate_tasks.py``,
``test_train.py``, etc.) will land as its own PR under the #104 umbrella.  This file is
the minimal sanity check that the ``eq_preprocessed_dataset`` → ``eq_sampled_tasks_dir``
→ ``eq_trained_model_dir`` chain actually runs end-to-end on ``simple_static_MEDS``.

Kept small on purpose: the per-stage assertion logic lives in its own PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_preprocess_produces_metadata(eq_preprocessed_dataset: Path) -> None:
    """EQ_process_data writes a vocabulary / codes metadata parquet."""
    assert (eq_preprocessed_dataset / "metadata" / "codes.parquet").exists()
    # Tokenization artifacts should be present — downstream MTD consumers need them.
    assert (eq_preprocessed_dataset / "tokenization" / "event_seqs").exists()


def test_generate_tasks_writes_both_splits(eq_sampled_tasks_dir: Path) -> None:
    """EQ_generate_tasks produces at least one labeled parquet per split."""
    import polars as pl

    for split in ("train", "tuning"):
        fps = list((eq_sampled_tasks_dir / split).glob("*.parquet"))
        assert fps, f"no labeled parquet found under {eq_sampled_tasks_dir / split}"
        df = pl.read_parquet(fps[0])
        assert set(df.columns) >= {
            "subject_id",
            "prediction_time",
            "boolean_value",
            "occurs",
            "query",
            "duration_days",
        }, f"unexpected columns in {fps[0]}: {df.columns}"


def test_train_produces_checkpoint(eq_trained_model_dir: Path) -> None:
    """EQ_train --config-name=_demo_train produces a checkpoint and resolved config."""
    ckpts = list((eq_trained_model_dir / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"no checkpoint under {eq_trained_model_dir / 'checkpoints'}"
