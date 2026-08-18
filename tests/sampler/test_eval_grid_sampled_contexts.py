"""The evaluation grid's second cohort source: contexts sampled from the split.

``sample_evaluation_query_sequences`` used to *require* a caller-supplied ``contexts_path`` — it
raised when the key was unset, which its own config defaulted to, so the endpoint was unrunnable out
of the box.  It now falls back to the same upstream Stage 0 + Stage 2 draw the training sampler
uses, seeded on the same ``"contexts"`` axis, and writes Stage 0's intermediates to the
``{name}_artifacts`` sibling of ``out_dir``.

These tests cover that sampled branch only; the supplied branch (unchanged) is exercised in
``tests/test_conditional_queries.py``.

Fixture cohort: the shared ``synthetic_events`` frame (3 subjects x 30 distinct times) split across
two shards — ``"0"`` holds subjects 1 and 2, ``"1"`` holds subject 3 — so the per-shard rank ->
timestamp resolution is exercised rather than assumed.
"""

import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from every_query.generate_tasks.sample_evaluation_query_sequences import (
    run_worker,
    sample_grid_contexts,
)
from every_query.generate_tasks.sample_tasks import (
    prediction_time_counts_path,
    sample_patient_contexts,
)
from every_query.utils.seeds import derive_seed

SPLIT = "held_out"
MIN_PT = 5  # every fixture subject has 30 distinct times, so all three stay eligible


@pytest.fixture
def cohort(tmp_path: Path, synthetic_events: pl.DataFrame, write_split_shards) -> Path:
    """Write ``synthetic_events`` as two shards and return the MEDS root (``data_dir``)."""
    return write_split_shards(
        tmp_path,
        {
            "0": synthetic_events.filter(pl.col("subject_id") != 3),
            "1": synthetic_events.filter(pl.col("subject_id") == 3),
        },
        split=SPLIT,
    )


@pytest.fixture
def ranked_times(synthetic_events: pl.DataFrame) -> pl.DataFrame:
    """An independent ``(subject_id, time, rank)`` oracle for Stage 0's ``prediction_time_index``.

    Rebuilt straight off the events rather than read back from the artifacts tree, so a test using
    it checks the resolution rather than restating it.
    """
    return (
        synthetic_events.select("subject_id", "time")
        .unique()
        .sort("subject_id", "time")
        .with_columns(pl.int_range(pl.len()).over("subject_id").alias("rank"))
    )


# ── sample_grid_contexts ────────────────────────────────────────────────


def test_sampled_contexts_are_shaped_like_a_supplied_cohort(tmp_path: Path, cohort: Path):
    """Nothing downstream may be able to tell the two cohort sources apart."""
    ctx = sample_grid_contexts(cohort, tmp_path / "art", SPLIT, 40, MIN_PT, derive_seed(1, "contexts"))

    assert ctx.columns == ["subject_id", "prediction_time"]
    assert ctx.schema["subject_id"] == pl.Int64
    assert ctx.schema["prediction_time"] == pl.Datetime("us")
    # A dense grid asks each sequence about each context exactly once, so the with-replacement draw
    # is deduped; `n_contexts` is a ceiling, not an exact count.
    assert ctx.height == ctx.unique().height
    assert 0 < ctx.height <= 40
    assert ctx.equals(ctx.sort("subject_id", "prediction_time"))


def test_sampled_contexts_resolve_to_real_eligible_event_times(
    tmp_path: Path, cohort: Path, ranked_times: pl.DataFrame
):
    """Stage 2 returns Int64 *ranks*; every one must land on a real timestamp past the eligibility floor."""
    ctx = sample_grid_contexts(cohort, tmp_path / "art", SPLIT, 60, MIN_PT, derive_seed(1, "contexts"))

    joined = ctx.join(
        ranked_times,
        left_on=["subject_id", "prediction_time"],
        right_on=["subject_id", "time"],
        how="left",
    )
    assert joined["rank"].null_count() == 0, "every context must be a real distinct event time"
    assert joined["rank"].min() >= MIN_PT, "min_prediction_times_per_subject is the draw floor"


def test_sampled_contexts_are_the_upstream_stage_2_draw(
    tmp_path: Path, cohort: Path, ranked_times: pl.DataFrame
):
    """The draw is upstream's, not a reimplementation: same rows as ``sample_patient_contexts``.

    The expected frame is built by looking each drawn ``prediction_time_index`` up in the
    events-derived rank oracle, so this pins the *values* the sampler must produce rather than
    replaying its own resolution step.
    """
    artifacts = tmp_path / "art"
    seed = derive_seed(3, "contexts")
    ctx = sample_grid_contexts(cohort, artifacts, SPLIT, 50, MIN_PT, seed)

    counts = pl.read_parquet(prediction_time_counts_path(artifacts, SPLIT)).sort("subject_id")
    drawn = sample_patient_contexts(counts, 50, MIN_PT, np.random.default_rng(seed))
    expected = (
        drawn.select("subject_id", "prediction_time_index")
        .join(
            ranked_times,
            left_on=["subject_id", "prediction_time_index"],
            right_on=["subject_id", "rank"],
            how="left",
        )
        .select("subject_id", pl.col("time").alias("prediction_time"))
        .unique()
        .sort("subject_id", "prediction_time")
    )
    assert ctx.equals(expected)


def test_sampled_contexts_are_deterministic_and_seed_sensitive(tmp_path: Path, cohort: Path):
    a = sample_grid_contexts(cohort, tmp_path / "a", SPLIT, 40, MIN_PT, derive_seed(7, "contexts"))
    b = sample_grid_contexts(cohort, tmp_path / "b", SPLIT, 40, MIN_PT, derive_seed(7, "contexts"))
    c = sample_grid_contexts(cohort, tmp_path / "c", SPLIT, 40, MIN_PT, derive_seed(8, "contexts"))
    assert a.equals(b)
    assert not a.equals(c)


def test_sampled_contexts_reject_an_empty_budget(tmp_path: Path, cohort: Path):
    with pytest.raises(ValueError, match="n_contexts must be >= 1"):
        sample_grid_contexts(cohort, tmp_path / "art", SPLIT, 0, MIN_PT, 0)


# ── run_worker's sampled branch ─────────────────────────────────────────


def _run(cohort: Path, out_dir: Path, codes: list[str], **overrides) -> list[Path]:
    kwargs = {
        "data_dir": cohort,
        "out_dir": out_dir,
        "split": SPLIT,
        "query_codes": codes,
        "n_sequences": 3,
        "min_queries": 2,
        "max_queries": 2,
        "n_contexts": 20,
        "min_prediction_times_per_subject": MIN_PT,
        "seed": 1,
    }
    kwargs.update(overrides)
    return run_worker(**kwargs)


def test_run_worker_samples_a_cohort_when_none_is_supplied(
    tmp_path: Path, cohort: Path, synthetic_query_codes: list[str]
):
    """The endpoint is runnable with no cohort file: the grid's contexts come from the split."""
    out_dir = tmp_path / "eval_tasks"
    written = _run(cohort, out_dir, synthetic_query_codes)

    # The cohort tag records the *requested* draw size (output paths are resolved before any
    # cohort is read), and the spec tag the number of sampled sequences.
    assert written == [out_dir / SPLIT / "sampled20ctx__sampled3.parquet"]

    labeled = pl.read_parquet(written[0])
    assert labeled.columns == ["subject_id", "prediction_time", "queries", "durations", "answers"]
    n_contexts = labeled.select("subject_id", "prediction_time").n_unique()
    assert 0 < n_contexts <= 20
    assert labeled.height == n_contexts * 3, "dense grid: every context gets every sequence"
    assert labeled["queries"].list.len().unique().to_list() == [2]
    assert labeled["answers"].list.len().unique().to_list() == [2]


def test_sampled_run_writes_stage0_artifacts_to_the_sibling_tree(
    tmp_path: Path, cohort: Path, synthetic_query_codes: list[str]
):
    """Invariant 7: intermediates land beside the output root, never inside it."""
    out_dir = tmp_path / "eval_tasks"
    _run(cohort, out_dir, synthetic_query_codes)

    artifacts = tmp_path / "eval_tasks_artifacts"
    assert (artifacts / SPLIT / "_prediction_time_counts.parquet").exists()
    assert sorted(p.name for p in (artifacts / SPLIT / "_prediction_times").iterdir()) == [
        "0.parquet",
        "1.parquet",
    ]
    assert [p.name for p in out_dir.iterdir()] == [SPLIT], "out_dir holds final outputs only"


def test_a_supplied_cohort_is_only_a_different_source_for_the_same_contexts(
    tmp_path: Path, cohort: Path, synthetic_query_codes: list[str]
):
    """Feed the sampled cohort back in as ``contexts_path`` and the labels must be identical.

    This is what makes the branch purely additive: the two sources differ in provenance, and in the
    output file name, and in nothing else.
    """
    (sampled_fp,) = _run(cohort, tmp_path / "sampled", synthetic_query_codes, seed=5)
    sampled = pl.read_parquet(sampled_fp)

    cohort_fp = tmp_path / "cohort.parquet"
    sampled.select("subject_id", "prediction_time").unique().sort(
        "subject_id", "prediction_time"
    ).write_parquet(cohort_fp)

    (supplied_fp,) = _run(
        cohort, tmp_path / "supplied", synthetic_query_codes, seed=5, contexts_path=cohort_fp
    )
    assert supplied_fp.name == "cohort__sampled3.parquet"
    assert pl.read_parquet(supplied_fp).equals(sampled)


def test_sampled_outputs_are_skipped_unless_overwritten(
    tmp_path: Path, cohort: Path, synthetic_query_codes: list[str]
):
    out_dir = tmp_path / "eval_tasks"
    (fp,) = _run(cohort, out_dir, synthetic_query_codes)
    assert _run(cohort, out_dir, synthetic_query_codes) == []
    assert _run(cohort, out_dir, synthetic_query_codes, overwrite=True) == [fp]


def test_per_spec_dirs_works_on_the_sampled_path(
    tmp_path: Path, cohort: Path, synthetic_query_codes: list[str]
):
    out_dir = tmp_path / "eval_tasks"
    written = _run(cohort, out_dir, synthetic_query_codes, per_spec_dirs=True, n_sequences=2)

    assert [p.relative_to(out_dir).as_posix() for p in written] == [
        f"seq_0000/{SPLIT}/tasks.parquet",
        f"seq_0001/{SPLIT}/tasks.parquet",
    ]
    # Each per-spec file is the same cohort asked one sequence, so the heights agree.
    assert len({pl.read_parquet(p).height for p in written}) == 1


# ── config <-> code wiring ──────────────────────────────────────────────


def test_sampled_cohort_config_keys_match_run_worker_defaults():
    """The two new knobs must be present in the YAML and agree with the Python defaults.

    Programmatic callers bypass Hydra, so a drifted default would give them a different grid than
    the CLI produces — the same failure mode
    ``test_eval_sampling_defaults_stay_in_training_distribution`` guards for the query knobs.
    """
    configs = Path(eval_seq.CONFIGS)
    eval_cfg = yaml.safe_load((configs / "sample_evaluation_query_sequences_config.yaml").read_text())
    train_cfg = yaml.safe_load((configs / "sample_query_sequences_config.yaml").read_text())
    defaults = inspect.signature(eval_seq.run_worker).parameters

    for key in ("n_contexts", "min_prediction_times_per_subject"):
        assert defaults[key].default == eval_cfg[key], f"run_worker default for {key} != config"

    # Unset is the sampled path, not an error — on both sides, and by default.
    assert eval_cfg["contexts_path"] is None
    assert defaults["contexts_path"].default is None

    # The eligibility bound governs which prediction times exist at all; a sampled evaluation grid
    # drawn under a different bound is scored at contexts training never saw.
    assert eval_cfg["min_prediction_times_per_subject"] == train_cfg["min_prediction_times_per_subject"]
