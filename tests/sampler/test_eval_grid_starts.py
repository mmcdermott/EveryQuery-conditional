"""Explicit window starts in the dense evaluation grid generator (issue #27): specs, sampling, output.

Covered, in pipeline order:

1. ``SequenceSpec`` — every valid and invalid start representation.
2. Designed specs — the YAML mapping-entry form, JSON, the bare list, and the long-format parquet
   with optional ``start_duration_days`` / ``start_event`` columns; unknown start codes rejected.
3. Sampled specs — the start component is drawn on its own three seed axes, so (a) the default
   knobs reproduce the pre-#27 specs exactly and (b) any start knob leaves the query / duration /
   end-bound draw untouched; validation mirrors the multitask sampler's.
4. The grid — default output carries no start columns (schema-compatible with today); an active
   start puts both columns in the index and the parquet; the provenance fingerprint sees starts so
   a start-knob change relabels rather than serving a stale shard.

Parity reasoning, stated once (the issue asks for it here): the query / duration / end draw of the
eval grid is parity-anchored to ``sample_query_sequences.py`` and must not move; only the start
component mirrors ``sample_multitask_sequences.py``'s ``BoundaryDistribution``.  The tests therefore
pin *both* halves — the legacy draw byte-for-byte, and the start draw's independence from it —
without claiming the whole sequence equals a multitask draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml
from hydra import compose, initialize_config_dir

from every_query.data.schema import QuerySeqSchema
from every_query.data.seq_dataset import EVENT_BOUND_DURATION_SENTINEL as SENTINEL
from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from every_query.generate_tasks.sample_evaluation_query_sequences import (
    SequenceSpec,
    _sample_starts,
    _specs_fingerprint,
    build_dense_sequence_index_df,
    read_sequence_specs,
    sample_sequence_specs,
    validate_spec_codes,
)
from every_query.generate_tasks.sample_query_sequences import (
    BOUND_COL,
    START_DURATION_COL,
    START_EVENT_COL,
)

SPLIT = "held_out"
SHARDS = ["0", "1"]


# ---------------------------------------------------------------------------
# 1. SequenceSpec representations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_durations", "start_events", "active"),
    [
        ((), (), False),
        ((0.0, 0.0), (None, None), False),
        ((0, 7), (None, None), True),
        ((SENTINEL, 0.0), ("ADMIT", None), True),
        ((SENTINEL, SENTINEL), ("ADMIT", "DISCHARGE"), True),
        ((0.5, SENTINEL), (None, "ADMIT"), True),
    ],
)
def test_valid_start_representations(start_durations, start_events, active):
    spec = SequenceSpec(
        "s", ("A", "B"), (1.0, 2.0), start_durations=start_durations, start_events=start_events
    )
    assert spec.has_active_starts is active
    for i in range(2):
        sd, se = spec.start_at(i)
        assert isinstance(sd, float)
        assert (se is None) == (sd != SENTINEL)


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"start_durations": (7.0, 0.0)}, ValueError, "give both or neither"),
        ({"start_events": (None, None)}, ValueError, "give both or neither"),
        ({"start_durations": (7.0,), "start_events": (None,)}, ValueError, "2 queries but 1 start"),
        (
            {"start_durations": (7.0, 0.0), "start_events": ("ADMIT", None)},
            ValueError,
            "must be the -1.0 sentinel",
        ),
        (
            {"start_durations": (SENTINEL, 0.0), "start_events": (None, None)},
            ValueError,
            "finite number >= 0",
        ),
        ({"start_durations": (-2.0, 0.0), "start_events": (None, None)}, ValueError, "finite number >= 0"),
        (
            {"start_durations": (float("inf"), 0.0), "start_events": (None, None)},
            ValueError,
            "finite number >= 0",
        ),
        (
            {"start_durations": (float("nan"), 0.0), "start_events": (None, None)},
            ValueError,
            "finite number >= 0",
        ),
        ({"start_durations": (True, 0.0), "start_events": (None, None)}, TypeError, "must be a number"),
        (
            {"start_durations": (SENTINEL, 0.0), "start_events": ("", None)},
            ValueError,
            "non-empty string or null",
        ),
        (
            {"start_durations": (SENTINEL, 0.0), "start_events": (3, None)},
            ValueError,
            "non-empty string or null",
        ),
    ],
)
def test_invalid_start_representations(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        SequenceSpec("bad", ("A", "B"), (1.0, 2.0), **kwargs)


def test_spec_name_sanitising_keeps_the_starts():
    spec = SequenceSpec("a/b", ("A",), (1.0,), start_durations=(7.0,), start_events=(None,))
    (safe,) = eval_seq._sanitise_names([spec])
    assert safe.name == "a_b" and safe.start_durations == (7.0,) and safe.start_events == (None,)


# ---------------------------------------------------------------------------
# 2. Designed specs
# ---------------------------------------------------------------------------

DESIGNED = {
    "post_admission": [{"query": "LAB//X", "start_event": "HOSPITAL_ADMISSION", "duration_days": 30}],
    "delayed": [{"query": "ICD//I10", "start_duration_days": 7, "duration_days": 30}],
    "between_events": [
        {
            "query": "PROCEDURE//X",
            "start_event": "HOSPITAL_ADMISSION",
            "duration_days": -1,
            "bound_event": "HOSPITAL_DISCHARGE",
        }
    ],
    "mixed": [
        ["TIMELINE//END", 1],
        ["SEPSIS", -1, "HOSPITAL_DISCHARGE"],
        {"query": "LAB//X", "duration_days": 3},
    ],
}


def _expected_designed() -> dict[str, tuple]:
    return {
        "post_admission": (("LAB//X",), (30.0,), (), (SENTINEL,), ("HOSPITAL_ADMISSION",)),
        "delayed": (("ICD//I10",), (30.0,), (), (7.0,), (None,)),
        "between_events": (
            ("PROCEDURE//X",),
            (-1.0,),
            ("HOSPITAL_DISCHARGE",),
            (SENTINEL,),
            ("HOSPITAL_ADMISSION",),
        ),
        "mixed": (
            ("TIMELINE//END", "SEPSIS", "LAB//X"),
            (1.0, -1.0, 3.0),
            (None, "HOSPITAL_DISCHARGE", None),
            (),
            (),
        ),
    }


def _check_designed(specs: list[SequenceSpec]) -> None:
    got = {s.name: (s.queries, s.durations, s.bounds, s.start_durations, s.start_events) for s in specs}
    assert got == _expected_designed()


def test_yaml_mapping_form(tmp_path: Path):
    fp = tmp_path / "designed.yaml"
    fp.write_text(yaml.safe_dump(DESIGNED))
    _check_designed(read_sequence_specs(fp))


def test_json_form(tmp_path: Path):
    fp = tmp_path / "designed.json"
    fp.write_text(json.dumps(DESIGNED))
    _check_designed(read_sequence_specs(fp))


def test_bare_list_form(tmp_path: Path):
    fp = tmp_path / "designed.yaml"
    fp.write_text(yaml.safe_dump(list(DESIGNED.values())))
    specs = read_sequence_specs(fp)
    assert [s.name for s in specs] == [f"seq_{i:04d}" for i in range(4)]
    expected = list(_expected_designed().values())
    assert [(s.queries, s.durations, s.bounds, s.start_durations, s.start_events) for s in specs] == expected


def test_long_format_parquet_with_start_columns(tmp_path: Path):
    rows = [
        {
            "seq_id": "post_admission",
            "position": 0,
            "query": "LAB//X",
            "duration_days": 30.0,
            "bound_event": None,
            "start_duration_days": None,
            "start_event": "HOSPITAL_ADMISSION",
        },
        {
            "seq_id": "delayed",
            "position": 0,
            "query": "ICD//I10",
            "duration_days": 30.0,
            "bound_event": None,
            "start_duration_days": 7.0,
            "start_event": None,
        },
        {
            "seq_id": "between_events",
            "position": 0,
            "query": "PROCEDURE//X",
            "duration_days": -1.0,
            "bound_event": "HOSPITAL_DISCHARGE",
            "start_duration_days": -1.0,
            "start_event": "HOSPITAL_ADMISSION",
        },
        {
            "seq_id": "mixed",
            "position": 1,
            "query": "SEPSIS",
            "duration_days": -1.0,
            "bound_event": "HOSPITAL_DISCHARGE",
            "start_duration_days": 0.0,
            "start_event": None,
        },
        {
            "seq_id": "mixed",
            "position": 0,
            "query": "TIMELINE//END",
            "duration_days": 1.0,
            "bound_event": None,
            "start_duration_days": None,
            "start_event": None,
        },
        {
            "seq_id": "mixed",
            "position": 2,
            "query": "LAB//X",
            "duration_days": 3.0,
            "bound_event": None,
            "start_duration_days": 0.0,
            "start_event": None,
        },
    ]
    fp = tmp_path / "designed.parquet"
    pl.DataFrame(rows).with_columns(pl.col("start_duration_days").cast(pl.Float64)).write_parquet(fp)
    _check_designed(read_sequence_specs(fp))


def test_long_format_parquet_without_start_columns_is_unchanged(tmp_path: Path):
    rows = [
        {"seq_id": "a", "position": 0, "query": "X", "duration_days": 3.0},
        {"seq_id": "a", "position": 1, "query": "Y", "duration_days": 5.0},
    ]
    fp = tmp_path / "designed.parquet"
    pl.DataFrame(rows).write_parquet(fp)
    (spec,) = read_sequence_specs(fp)
    assert (spec.queries, spec.durations, spec.bounds, spec.start_durations, spec.start_events) == (
        ("X", "Y"),
        (3.0, 5.0),
        (),
        (),
        (),
    )


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"query": "A", "duration_days": 3, "typo": 1}, "unknown key"),
        ({"query": "A"}, "missing required key"),
        ({"query": "A", "duration_days": 3, "start_event": "ADMIT", "start_duration_days": 7}, "sentinel"),
        ({"query": "A", "duration_days": 3, "start_duration_days": -1}, "finite number >= 0"),
        ({"query": "A", "duration_days": 30, "bound_event": "X"}, "sentinel"),
    ],
)
def test_contradictory_mapping_entries_are_rejected(tmp_path: Path, entry, match):
    fp = tmp_path / "bad.yaml"
    fp.write_text(yaml.safe_dump({"s": [entry]}))
    with pytest.raises(ValueError, match=match):
        read_sequence_specs(fp)


def test_unknown_start_codes_are_rejected_against_the_vocabulary():
    specs = [SequenceSpec("s", ("A",), (1.0,), start_durations=(SENTINEL,), start_events=("NOPE",))]
    with pytest.raises(ValueError, match="NOPE"):
        validate_spec_codes(specs, {"A", "B"})
    validate_spec_codes(specs, {"A", "NOPE"})


# ---------------------------------------------------------------------------
# 3. Sampled specs
# ---------------------------------------------------------------------------

CODES = ["A", "B", "C", "TIMELINE//END"]


def _draw(**kw) -> list[SequenceSpec]:
    base = {
        "n_sequences": 12,
        "query_codes": CODES,
        "min_queries": 1,
        "max_queries": 4,
        "duration_low": 1,
        "duration_high": 365,
        "seed": 7,
        "eventbound_fraction": 0.5,
    }
    base.update(kw)
    return sample_sequence_specs(**base)


def _legacy_view(specs: list[SequenceSpec]) -> list[tuple]:
    return [(s.queries, s.durations, s.bounds) for s in specs]


def _start_view(specs: list[SequenceSpec]) -> list[tuple]:
    return [tuple(s.start_at(i) for i in range(len(s))) for s in specs]


def test_default_start_knobs_reproduce_the_previous_draw_exactly():
    """Every window opens at the prediction time, the specs carry no start tuples, and the legacy (queries,
    durations, bounds) draw is the one the pre-#27 signature produced."""
    plain = _draw()
    spelled = _draw(eventstart_fraction=0.0, prediction_time_start_fraction=1.0)
    assert plain == spelled
    assert all(not s.start_durations and not s.start_events and not s.has_active_starts for s in plain)


# Literal output of ``sample_sequence_specs(3, ["A", "B", "C", "TIMELINE//END"], 1, 4, 1, 365, 7,
# eventbound_fraction=0.5)`` on the pre-#27 function (commit 494d7f5 of
# ``feat/prevalence-weighted-boundaries``), recorded so the draw is pinned against a *fixed* value
# rather than against the new function itself: rewiring the start component onto a legacy RNG axis
# would perturb these numbers and fail here, where a self-comparison would still pass.
_GOLDEN_LEGACY_DRAW = [
    (("A",), (47.47083986883709,), ()),
    (("A",), (-1.0,), ("A",)),
    (
        ("TIMELINE//END", "B", "TIMELINE//END", "TIMELINE//END"),
        (1.1336861389346837, -1.0, 11.378642731393592, 359.8491603781334),
        (None, "TIMELINE//END", None, None),
    ),
]


@pytest.mark.parametrize(
    "start_kwargs",
    [
        {},
        {"eventstart_fraction": 0.4, "prediction_time_start_fraction": 0.3, "start_event_codes": ["A", "B"]},
        {"eventstart_fraction": 0.0, "prediction_time_start_fraction": 0.0, "start_duration_max": 10.0},
    ],
    ids=["default-starts", "mixed-starts", "all-duration-starts"],
)
def test_legacy_draw_matches_the_pre_start_golden(start_kwargs):
    """The (queries, durations, bounds) draw is byte-identical to the pre-#27 sampler's, with the start knobs
    at their defaults and with every start form switched on."""
    specs = sample_sequence_specs(
        3, ["A", "B", "C", "TIMELINE//END"], 1, 4, 1, 365, 7, eventbound_fraction=0.5, **start_kwargs
    )
    assert _legacy_view(specs) == _GOLDEN_LEGACY_DRAW


def test_start_knobs_do_not_perturb_the_legacy_draw():
    plain = _draw()
    started = _draw(eventstart_fraction=0.3, prediction_time_start_fraction=0.3, start_event_codes=["A", "B"])
    assert _legacy_view(started) == _legacy_view(plain)
    assert [len(s) for s in started] == [len(s) for s in plain]
    assert any(s.has_active_starts for s in started)
    forms = {
        ("event" if e is not None else "pt" if d == 0 else "delay")
        for s in started
        for d, e in _start_view([s])[0]
    }
    assert forms == {"event", "pt", "delay"}


def test_legacy_knobs_do_not_perturb_the_start_component():
    """Changing an end-side knob changes the end draw only; the per-slot start forms/values stay put slot for
    slot (the start streams are drawn over the same total query count, so lengths must agree — which they do
    because lengths come from the untouched structure axis)."""
    a = _draw(eventstart_fraction=0.3, prediction_time_start_fraction=0.3, eventbound_fraction=0.0)
    b = _draw(eventstart_fraction=0.3, prediction_time_start_fraction=0.3, eventbound_fraction=1.0)
    assert _start_view(a) == _start_view(b)
    assert _legacy_view(a) != _legacy_view(b)


def test_each_start_axis_is_independent():
    base = {
        "eventstart_fraction": 0.4,
        "prediction_time_start_fraction": 0.2,
        "start_event_codes": ["A", "B"],
    }
    ref = _draw(**base)
    other_pool = _draw(**{**base, "start_event_codes": ["C"]})
    other_range = _draw(**{**base, "start_duration_min": 100, "start_duration_max": 200})

    def forms(specs):
        return [
            [("event" if e is not None else "pt" if d == 0 else "delay") for d, e in row]
            for row in _start_view(specs)
        ]

    # The pool only moves which event code is drawn; forms and delay durations are untouched.
    assert forms(other_pool) == forms(ref)
    assert [[d for d, e in row if e is None] for row in _start_view(other_pool)] == [
        [d for d, e in row if e is None] for row in _start_view(ref)
    ]
    assert {e for row in _start_view(other_pool) for _, e in row if e is not None} == {"C"}
    # The duration range only moves the delays; forms and event codes are untouched.
    assert forms(other_range) == forms(ref)
    assert [[e for _, e in row] for row in _start_view(other_range)] == [
        [e for _, e in row] for row in _start_view(ref)
    ]
    delays = [d for row in _start_view(other_range) for d, e in row if e is None and d > 0]
    assert delays and all(100 <= d <= 200 for d in delays)


def test_sampled_start_draw_is_deterministic():
    kw = {"eventstart_fraction": 0.5, "prediction_time_start_fraction": 0.25}
    assert _draw(**kw) == _draw(**kw)
    assert _draw(**kw) != _draw(**kw, seed=8)


def test_null_start_pool_is_the_query_universe():
    specs = _draw(eventstart_fraction=1.0, prediction_time_start_fraction=0.0, n_sequences=60)
    codes = {e for s in specs for e in s.start_events}
    assert codes == set(CODES)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"eventstart_fraction": 1.2}, "eventstart_fraction must be in"),
        ({"eventstart_fraction": -0.1}, "eventstart_fraction must be in"),
        ({"prediction_time_start_fraction": 1.5}, "prediction_time_start_fraction must be in"),
        ({"eventstart_fraction": 0.6, "prediction_time_start_fraction": 0.6}, "must be <= 1"),
        (
            {"eventstart_fraction": 0.5, "prediction_time_start_fraction": 0.5, "start_event_codes": []},
            "non-empty start_event_codes",
        ),
        ({"start_duration_min": 0}, "start_duration_min must be > 0"),
        ({"start_duration_min": 10, "start_duration_max": 5}, "start_duration_max"),
        ({"start_duration_distribution": "normal"}, "start_duration_distribution"),
        ({"start_event_codes": ["A", "NOPE"]}, "outside the query universe"),
    ],
)
def test_start_sampling_validation_mirrors_the_multitask_sampler(kw, match):
    with pytest.raises(ValueError, match=match):
        _draw(**kw)


def test_sample_starts_cumulative_split_matches_the_multitask_form_rule():
    """Same ``u`` thresholds as ``BoundaryDistribution.sample``: event below ``e``, prediction time below ``e
    + p``, delay above."""
    rng_u = np.random.default_rng(3)
    u = rng_u.random(500)
    d, e = _sample_starts(
        500,
        ["X"],
        0.2,
        0.3,
        1.0,
        5.0,
        "uniform",
        np.random.default_rng(3),
        np.random.default_rng(4),
        np.random.default_rng(5),
    )
    assert ((e != None) == (u < 0.2)).all()  # noqa: E711
    assert ((d == 0.0) == ((u >= 0.2) & (u < 0.5))).all()
    assert (d[(u >= 0.5)] >= 1.0).all() and (d[(u >= 0.5)] <= 5.0).all()


# ---------------------------------------------------------------------------
# 4. The grid
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path, synthetic_events: pl.DataFrame, write_split_shards) -> Path:
    return write_split_shards(
        tmp_path,
        {
            "0": synthetic_events.filter(pl.col("subject_id") != 3),
            "1": synthetic_events.filter(pl.col("subject_id") == 3),
        },
        split=SPLIT,
    )


@pytest.fixture
def codes_yaml(tmp_path: Path, synthetic_query_codes: list[str]) -> Path:
    fp = tmp_path / "codes.yaml"
    fp.write_text(yaml.safe_dump(synthetic_query_codes))
    return fp


def _run(data_dir: Path, out_dir: Path, codes_yaml: Path, **overrides) -> None:
    kwargs = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "query_codes": codes_yaml,
        "split": SPLIT,
        "num_evaluation_sequences": 4,
        "min_queries": 2,
        "max_queries": 2,
        "prediction_times_per_subject": 3,
        "min_context_per_subject": 5,
        "seed": 1,
    }
    kwargs.update(overrides)
    with initialize_config_dir(config_dir=eval_seq.CONFIGS, version_base=None):
        cfg = compose(
            config_name="sample_evaluation_query_sequences_config",
            overrides=[f"{k}={v}" for k, v in kwargs.items()],
        )
    eval_seq.main.__wrapped__(cfg)


def _labels(out_dir: Path, shard: str) -> pl.DataFrame:
    return pl.read_parquet(out_dir / "eval" / SPLIT / f"{shard}.parquet")


def test_config_ships_the_start_keys_with_the_prediction_time_defaults():
    cfg = yaml.safe_load(
        (Path(eval_seq.CONFIGS) / "sample_evaluation_query_sequences_config.yaml").read_text()
    )
    assert cfg["eventstart_fraction"] == 0.0
    assert cfg["prediction_time_start_fraction"] == 1.0
    assert cfg["start_duration_min"] == 1 and cfg["start_duration_max"] == 180
    assert cfg["start_duration_distribution"] == "log-uniform"
    assert cfg["start_event_codes"] is None


def test_default_grid_carries_no_start_columns(tmp_path: Path, data_dir: Path, codes_yaml: Path):
    """Schema-compatible with today: the default config writes exactly the pre-#27 columns."""
    out_dir = tmp_path / "grid"
    _run(data_dir, out_dir, codes_yaml)
    for shard in SHARDS:
        df = _labels(out_dir, shard)
        assert df.columns == [
            "subject_id",
            "prediction_time",
            "queries",
            "durations",
            "answers",
            "bound_events",
        ]
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))


def test_default_grid_specs_are_the_previous_draw(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_query_codes
):
    """The rows the default config writes are the specs the pre-#27 sampler drew on this seed."""
    from every_query.utils.seeds import derive_seed

    out_dir = tmp_path / "grid"
    _run(data_dir, out_dir, codes_yaml)
    # The duration bounds come from the shipped config so this test tracks its defaults.
    shipped = yaml.safe_load(
        (Path(eval_seq.CONFIGS) / "sample_evaluation_query_sequences_config.yaml").read_text()
    )
    expected = sample_sequence_specs(
        4,
        synthetic_query_codes,
        2,
        2,
        float(shipped["duration_min"]),
        float(shipped["duration_max"]),
        derive_seed(1, "eval_seq_specs", SPLIT),
        eventbound_fraction=float(shipped["eventbound_fraction"]),
    )
    df = _labels(out_dir, "0").head(4)
    assert [tuple(q) for q in df["queries"].to_list()] == [s.queries for s in expected]
    # The parquet stores float32 horizons; the spec holds the float64 draw.
    got = [d for row in df["durations"].to_list() for d in row]
    assert got == [float(np.float32(d)) for s in expected for d in s.durations]
    assert [tuple(b) for b in df["bound_events"].to_list()] == [s.bounds for s in expected]


def test_sampled_starts_reach_the_output_and_label_through_the_start_path(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    out_dir = tmp_path / "grid"
    _run(
        data_dir,
        out_dir,
        codes_yaml,
        eventstart_fraction=0.5,
        prediction_time_start_fraction=0.0,
        num_evaluation_sequences=8,
    )
    for shard in SHARDS:
        df = _labels(out_dir, shard)
        assert {"start_durations", "start_events"} <= set(df.columns)
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))
        flat = df.explode("start_durations", "start_events")
        assert ((flat["start_durations"] == SENTINEL) == flat["start_events"].is_not_null()).all()
        assert (flat.filter(pl.col("start_events").is_null())["start_durations"] > 0).all()
        assert df["answers"].explode().null_count() == 0


def test_designed_starts_are_labeled_per_the_rule(
    tmp_path: Path, data_dir: Path, codes_yaml: Path, synthetic_events: pl.DataFrame
):
    """End to end through ``main`` on the synthetic cohort, checked row by row against the plain-Python oracle
    in ``tests/test_queryseq_starts.py`` (which shares no code with the labeler).

    The specs exercise every start/end form at once: an event start with an event end, an event
    start with a duration end, a duration start with an event end, and a start event that never
    occurs.  Comparing whole ``answers`` columns to the oracle — rather than asserting a few
    aggregate properties — is what makes a labeler that silently ignored the starts fail here.
    """
    from tests.test_queryseq_starts import _oracle

    # synthetic_events: per subject, codes cycle A01, B02, C03, D04, E05 every 10 days.
    specs = {
        "between": [
            {"query": "ICD//C03", "start_event": "ICD//B02", "duration_days": -1, "bound_event": "ICD//C03"},
            {"query": "MED//D04", "start_event": "ICD//B02", "duration_days": -1, "bound_event": "MED//E05"},
        ],
        "after_b": [{"query": "ICD//C03", "start_event": "ICD//B02", "duration_days": 15}],
        "delayed": [
            {"query": "ICD//A01", "start_duration_days": 25, "duration_days": -1, "bound_event": "MED//E05"},
            {"query": "ICD//B02", "start_duration_days": 25, "duration_days": 30},
        ],
        "never": [
            {"query": "ICD//A01", "start_event": "MED//E05", "duration_days": -1, "bound_event": "NOPE//X"}
        ],
    }
    fp = tmp_path / "specs.yaml"
    fp.write_text(yaml.safe_dump(specs))
    out_dir = tmp_path / "grid"
    with pytest.raises(ValueError, match="absent from the query vocabulary"):
        _run(data_dir, out_dir, codes_yaml, sequences_path=fp)
    specs["never"][0]["bound_event"] = "MED//D04"
    fp.write_text(yaml.safe_dump(specs))
    _run(data_dir, out_dir, codes_yaml, sequences_path=fp)
    df = pl.concat([_labels(out_dir, s) for s in SHARDS])
    assert df.height == df.select("subject_id", "prediction_time").n_unique() * len(specs)

    events = [tuple(r) for r in synthetic_events.select("subject_id", "time", "code").rows()]
    seen_forms: set[tuple] = set()
    for row in df.iter_rows(named=True):
        positions = zip(
            row["queries"],
            row["durations"],
            row["bound_events"],
            row["start_durations"],
            row["start_events"],
            row["answers"],
            strict=True,
        )
        for query, duration, bound, start_duration, start_event, answer in positions:
            spec = (query, float(duration), bound, float(start_duration), start_event)
            expected = _oracle(events, row["subject_id"], row["prediction_time"], spec)
            assert answer == expected, (row["subject_id"], row["prediction_time"], spec)
            seen_forms.add((start_event is not None, bound is not None))
    # Every start/end form combination was actually exercised.
    assert seen_forms == {(True, True), (True, False), (False, True), (False, False)}
    # The oracle is not vacuous: both answers occur across the grid.
    flat = df.explode("answers")
    assert flat["answers"].any() and not flat["answers"].all()


def test_yaml_null_start_keys_read_as_absent(tmp_path: Path):
    """An explicit ``start_duration_days: null`` is "absent", exactly as the parquet reader treats a
    null cell: an event start gets the sentinel, a duration-less entry the prediction time."""
    fp = tmp_path / "designed.yaml"
    fp.write_text(
        yaml.safe_dump(
            {
                "s": [
                    {"query": "A", "duration_days": 3, "start_duration_days": None, "start_event": "ADMIT"},
                    {"query": "B", "duration_days": 3, "start_duration_days": None, "start_event": None},
                ]
            }
        )
    )
    (spec,) = read_sequence_specs(fp)
    assert spec.start_durations == (SENTINEL, 0.0) and spec.start_events == ("ADMIT", None)


def test_start_knob_change_relabels_instead_of_serving_stale_shards(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    out_dir = tmp_path / "grid"
    _run(data_dir, out_dir, codes_yaml)
    files = [out_dir / "eval" / SPLIT / f"{s}.parquet" for s in SHARDS]
    stamps = {fp: fp.stat().st_mtime_ns for fp in files}
    _run(data_dir, out_dir, codes_yaml)  # same knobs: reused
    assert {fp: fp.stat().st_mtime_ns for fp in files} == stamps
    _run(data_dir, out_dir, codes_yaml, eventstart_fraction=0.5, prediction_time_start_fraction=0.5)
    assert all(fp.stat().st_mtime_ns != t for fp, t in stamps.items())
    assert all("start_events" in _labels(out_dir, s).columns for s in SHARDS)


def test_fingerprint_is_unchanged_by_spelled_out_default_starts_and_changed_by_active_ones():
    plain = [SequenceSpec("x", ("A", "B"), (1.0, 30.0))]
    spelled = [
        SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, 0.0), start_events=(None, None))
    ]
    delayed = [
        SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, 7.0), start_events=(None, None))
    ]
    event = [
        SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, SENTINEL), start_events=(None, "A"))
    ]
    assert _specs_fingerprint(plain) == _specs_fingerprint(spelled)
    assert len({_specs_fingerprint(plain), _specs_fingerprint(delayed), _specs_fingerprint(event)}) == 3


def test_dense_index_carries_start_columns_only_for_active_starts():
    from datetime import datetime

    ctx = pl.DataFrame(
        {"subject_id": [1], "prediction_time": [datetime(2024, 1, 1)]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    )
    plain = build_dense_sequence_index_df(ctx, [SequenceSpec("a", ("X", "Y"), (1.0, 2.0))])
    assert START_DURATION_COL not in plain.columns and START_EVENT_COL not in plain.columns
    spelled = build_dense_sequence_index_df(
        ctx,
        [SequenceSpec("a", ("X", "Y"), (1.0, 2.0), start_durations=(0.0, 0.0), start_events=(None, None))],
    )
    assert spelled.equals(plain)
    started = build_dense_sequence_index_df(
        ctx,
        [
            SequenceSpec(
                "a", ("X", "Y"), (1.0, 2.0), start_durations=(7.0, SENTINEL), start_events=(None, "Z")
            ),
            SequenceSpec("b", ("X",), (-1.0,), bounds=("W",)),
        ],
    )
    assert started[START_DURATION_COL].to_list() == [7.0, SENTINEL, 0.0]
    assert started[START_EVENT_COL].to_list() == [None, "Z", None]
    assert started[BOUND_COL].to_list() == [None, None, "W"]
    assert started.schema[START_DURATION_COL] == pl.Float32
    empty = build_dense_sequence_index_df(
        ctx.head(0), [SequenceSpec("a", ("X",), (1.0,), start_durations=(7.0,), start_events=(None,))]
    )
    assert empty.height == 0 and START_EVENT_COL in empty.columns
