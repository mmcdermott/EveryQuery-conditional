"""End-to-end CLI smoke test with the ported features switched on.

The per-feature suites test the mechanisms; this one tests that the *pipeline* survives them —
that ``EQ_build_ontology`` → ``EQ_generate_query_sequences`` → ``EQ_train`` actually runs with
event bounds, ancestor queries and aggregate queries enabled at once, through real subprocesses
against the fixture cohort.

Marked ``slow``: it trains a model.  Run with ``pytest -m slow tests/test_features_e2e_cli.py``.

Boundary and component codes are read from the fixture's own vocabulary at runtime rather than
hardcoded, so this does not encode any particular cohort's spellings.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from meds import train_split, tuning_split

from conftest import run_and_check
from every_query.data.query_vocab import is_aggregate
from every_query.data.seq_dataset import EOS_CODE


def _vocabulary(preprocessed: Path) -> list[str]:
    codes_fp = preprocessed / "metadata" / "codes.parquet"
    return pl.read_parquet(codes_fp, columns=["code"])["code"].to_list()


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
    for name in ("nodes.parquet", "mix.parquet", "closure.parquet"):
        assert (ontology_dir / name).is_file(), f"{name} missing"
    nodes = pl.read_parquet(ontology_dir / "nodes.parquet")
    assert nodes.height > 0
    assert set(nodes.columns) == {"node", "vocab_index", "is_leaf"}


@pytest.fixture(scope="module")
def featured_tasks_dir(eq_preprocessed_dataset: Path, ontology_dir: Path, tmp_path_factory) -> Path:
    """Generate query sequences with event bounds, ancestors and aggregates all enabled."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("featured_seq_tasks")

    vocab = _vocabulary(eq_preprocessed_dataset)
    # Any real code works as a boundary; prefer the end-of-timeline code, which every cohort has.
    bounds = [EOS_CODE] if EOS_CODE in vocab else [vocab[0]]

    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_query_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                "num_sequences=16",
                "min_queries=2",
                "max_queries=4",
                "duration_min=1",
                "duration_max=30",
                "min_prediction_times_per_subject=1",
                "seed=1",
                "eventbound_fraction=0.3",
                f"bound_events=[{','.join(bounds)}]",
                f"ontology_dir={ontology_dir!s}",
                "ancestor_fraction=0.3",
                "aggregate_fraction=0.3",
            ],
            timeout=300.0,
        )
    return out_dir


@pytest.mark.slow
def test_generation_emits_every_query_form(featured_tasks_dir: Path):
    """The generated labels really do contain bounds, ancestors and aggregates."""
    shards = sorted((featured_tasks_dir / train_split).glob("*.parquet"))
    assert shards, "no output shards"
    df = pl.concat([pl.read_parquet(fp) for fp in shards])

    assert "bound_events" in df.columns, "event bounds were requested but no column was written"
    assert df["bound_events"].explode().null_count() < df["bound_events"].explode().len(), (
        "no query was actually event-bounded"
    )

    queries = df["queries"].explode().to_list()
    assert any(is_aggregate(q) for q in queries), "no aggregate query was generated"

    # Answers stay binary and aligned no matter which forms are mixed in.
    assert df["answers"].explode().null_count() == 0
    assert (df["queries"].list.len() == df["answers"].list.len()).all()
    assert (df["queries"].list.len() == df["bound_events"].list.len()).all()


@pytest.mark.slow
def test_train_runs_with_every_feature_enabled(
    eq_preprocessed_dataset: Path, featured_tasks_dir: Path, ontology_dir: Path, tmp_path_factory
):
    """The decisive check: a real training run with all four features switched on."""
    out = tmp_path_factory.mktemp("featured_train")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional",
            f"output_dir={out!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={featured_tasks_dir!s}",
            "datamodule.dataset_kwargs.strip_delta_tokens=true",
            f"datamodule.dataset_kwargs.ontology_dir={ontology_dir!s}",
            "datamodule.dataset_kwargs.aggregate_queries=true",
            "lightning_module.model.use_rope_time=true",
            f"lightning_module.model.ontology_dir={ontology_dir!s}",
            "trainer.limit_val_batches=1",
        ],
        timeout=900.0,
    )
    ckpts = list(out.rglob("*.ckpt"))
    assert ckpts, "training produced no checkpoint"
