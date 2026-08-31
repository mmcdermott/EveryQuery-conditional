"""End-to-end CLI smoke test with the ported features switched on.

The per-feature suites test the mechanisms; this one tests that the *pipeline* survives them —
that ``EQ_build_ontology`` → ``EQ_generate_query_sequences`` → ``EQ_train`` actually runs with
RoPE time positions, event bounds and ancestor queries enabled at once, through real
subprocesses against the fixture cohort.

Marked ``slow``: it trains a model.  Run with ``pytest -m slow tests/test_features_e2e_cli.py``.

Boundary codes are read from the fixture's own vocabulary at runtime rather than hardcoded, so
this does not encode any particular cohort's spellings.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from meds import train_split, tuning_split

from conftest import run_and_check


@pytest.fixture(scope="module")
def ontology_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ontology")
    run_and_check(
        [
            # Invoked as a module rather than by console script: the venv's entry points were
            # installed before this CLI existed, and re-installing is not this test's job.
            sys.executable,
            "-m",
            "every_query.data.build_ontology",
            f"tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"out_dir={out!s}",
            "decay=0.5",
        ],
        timeout=120.0,
    )
    return out


@pytest.mark.slow
def test_build_ontology_writes_its_three_artifacts(ontology_dir: Path):
    for name in ("ontology_vocab.parquet", "embedding_mix.parquet", "event_to_query_nodes.parquet"):
        assert (ontology_dir / name).is_file(), f"{name} missing"
    nodes = pl.read_parquet(ontology_dir / "ontology_vocab.parquet")
    assert nodes.height > 0
    assert set(nodes.columns) == {"node_name", "token_id", "is_observed_code"}


@pytest.fixture(scope="module")
def featured_tasks_dir(eq_preprocessed_dataset: Path, ontology_dir: Path, tmp_path_factory) -> Path:
    """Generate query sequences with event bounds and ancestor queries both enabled."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("featured_seq_tasks")

    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_query_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                "num_training_sequence_examples=64",
                "min_queries=2",
                "max_queries=4",
                "duration_min=1",
                "duration_max=30",
                "min_prediction_times_per_subject=1",
                "seed=1",
                "eventbound_fraction=0.3",
                f"ontology_dir={ontology_dir!s}",
            ],
            timeout=300.0,
        )
    return out_dir


@pytest.mark.slow
def test_generation_emits_every_query_form(featured_tasks_dir: Path, ontology_dir: Path):
    """The generated labels really do contain bounds and ancestors."""
    shards = sorted((featured_tasks_dir / train_split).glob("*.parquet"))
    assert shards, "no output shards"
    df = pl.concat([pl.read_parquet(fp) for fp in shards])

    assert "bound_events" in df.columns, "event bounds were requested but no column was written"
    assert df["bound_events"].explode().null_count() < df["bound_events"].explode().len(), (
        "no query was actually event-bounded"
    )

    # Ancestor queries name a node that is not a leaf of the cohort vocabulary.
    nodes = pl.read_parquet(ontology_dir / "ontology_vocab.parquet")
    ancestors = set(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    queries = df["queries"].explode().to_list()
    assert ancestors, "the fixture ontology produced no ancestor nodes"
    assert any(q in ancestors for q in queries), "no ancestor query was generated"
    # Boundaries come from the same universe, so ancestor nodes bound queries too.
    bounds = df["bound_events"].explode().drop_nulls().to_list()
    assert any(b in ancestors for b in bounds), "no ancestor node was drawn as a boundary"

    # Answers stay binary and aligned no matter which forms are mixed in.
    assert df["answers"].explode().null_count() == 0
    assert (df["queries"].list.len() == df["answers"].list.len()).all()
    assert (df["queries"].list.len() == df["bound_events"].list.len()).all()


@pytest.mark.slow
def test_train_runs_with_every_feature_enabled(
    eq_preprocessed_dataset: Path, featured_tasks_dir: Path, ontology_dir: Path, tmp_path_factory
):
    """The decisive check: a real training run with all three features switched on."""
    out = tmp_path_factory.mktemp("featured_train")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional",
            f"output_dir={out!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={featured_tasks_dir!s}",
            "lightning_module.model.use_rope_time=true",
            f"lightning_module.model.ontology_dir={ontology_dir!s}",
            "trainer.limit_val_batches=1",
        ],
        timeout=900.0,
    )
    ckpts = list(out.rglob("*.ckpt"))
    assert ckpts, "training produced no checkpoint"
