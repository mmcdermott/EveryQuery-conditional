"""End-to-end correctness tests for sampled and supplied query-sequence pipelines.

Unlike the broader CLI smoke tests, these tests pin every value written against a tiny cohort whose
labels can be worked out by hand.  The sampled tests use the real Stage 0 prediction-time map,
Stage 1' query/bound sampling, Stage 2 context sampling, Stage 3' index partitioning, and Stage 4'
process worker.  The supplied-sequence test also drives the dense evaluation entry point over a
designed YAML file and supplied cohort.
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml
from omegaconf import OmegaConf

from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    build_event_to_query_nodes,
    build_ontology,
)
from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks import (
    sample_evaluation_query_sequences as eval_sqs,
)
from every_query.generate_tasks import (
    sample_query_sequences as sqs,
)

QUERY_CODES = [
    "DX//SEPSIS",
    "LAB//LACTATE",
    "HOSPITAL//DISCHARGE",
    "MED//ANTIBIOTIC",
    "FOLLOWUP//VISIT",
]


def _write_designed_cohort(tmp_path: Path) -> Path:
    """Write one three-subject MEDS shard with hand-checkable futures after each sampled context.

    Each subject has eight distinct timestamps.  Seed 260 samples subject 101 at index 0, subject
    303 at index 3, and subject 202 at index 2.  ``ANCHOR`` supplies otherwise inert prediction
    times; it is deliberately absent from the query vocabulary.

    The resulting event frame is::

        shape: (24, 3)
        ┌────────────┬─────────────────────┬─────────────────────┐
        │ subject_id ┆ time                ┆ code                │
        │ ---        ┆ ---                 ┆ ---                 │
        │ i64        ┆ datetime[μs]        ┆ str                 │
        ╞════════════╪═════════════════════╪═════════════════════╡
        │ 101        ┆ 2024-01-01 00:00:00 ┆ ANCHOR              │
        │ 101        ┆ 2024-01-02 00:00:00 ┆ DX//SEPSIS          │
        │ 101        ┆ 2024-01-03 00:00:00 ┆ ANCHOR              │
        │ 101        ┆ 2024-01-04 00:00:00 ┆ ANCHOR              │
        │ 101        ┆ 2024-01-05 00:00:00 ┆ ANCHOR              │
        │ 101        ┆ 2024-01-06 00:00:00 ┆ ANCHOR              │
        │ 101        ┆ 2024-01-07 00:00:00 ┆ ANCHOR              │
        │ 101        ┆ 2024-01-21 00:00:00 ┆ ANCHOR              │
        │ 202        ┆ 2024-02-01 00:00:00 ┆ ANCHOR              │
        │ 202        ┆ 2024-02-02 00:00:00 ┆ ANCHOR              │
        │ 202        ┆ 2024-02-03 00:00:00 ┆ ANCHOR              │
        │ 202        ┆ 2024-02-04 00:00:00 ┆ DX//SEPSIS          │
        │ 202        ┆ 2024-02-05 00:00:00 ┆ LAB//LACTATE        │
        │ 202        ┆ 2024-02-06 00:00:00 ┆ MED//ANTIBIOTIC     │
        │ 202        ┆ 2024-02-07 00:00:00 ┆ ANCHOR              │
        │ 202        ┆ 2024-02-21 00:00:00 ┆ ANCHOR              │
        │ 303        ┆ 2024-03-01 00:00:00 ┆ ANCHOR              │
        │ 303        ┆ 2024-03-02 00:00:00 ┆ LAB//LACTATE        │
        │ 303        ┆ 2024-03-03 00:00:00 ┆ ANCHOR              │
        │ 303        ┆ 2024-03-04 00:00:00 ┆ ANCHOR              │
        │ 303        ┆ 2024-03-05 00:00:00 ┆ MED//ANTIBIOTIC     │
        │ 303        ┆ 2024-03-06 00:00:00 ┆ FOLLOWUP//VISIT     │
        │ 303        ┆ 2024-03-07 00:00:00 ┆ HOSPITAL//DISCHARGE │
        │ 303        ┆ 2024-03-21 00:00:00 ┆ DX//SEPSIS          │
        └────────────┴─────────────────────┴─────────────────────┘
    """
    schedules = {
        101: (
            datetime(2024, 1, 1),
            ["ANCHOR", "DX//SEPSIS", "ANCHOR", "ANCHOR", "ANCHOR", "ANCHOR", "ANCHOR", "ANCHOR"],
        ),
        202: (
            datetime(2024, 2, 1),
            [
                "ANCHOR",
                "ANCHOR",
                "ANCHOR",
                "DX//SEPSIS",
                "LAB//LACTATE",
                "MED//ANTIBIOTIC",
                "ANCHOR",
                "ANCHOR",
            ],
        ),
        303: (
            datetime(2024, 3, 1),
            [
                "ANCHOR",
                "LAB//LACTATE",
                "ANCHOR",
                "ANCHOR",
                "MED//ANTIBIOTIC",
                "FOLLOWUP//VISIT",
                "HOSPITAL//DISCHARGE",
                "DX//SEPSIS",
            ],
        ),
    }
    offsets = (0, 1, 2, 3, 4, 5, 6, 20)
    rows = [
        {"subject_id": subject_id, "time": base + timedelta(days=offset), "code": code}
        for subject_id, (base, codes) in schedules.items()
        for offset, code in zip(offsets, codes, strict=True)
    ]
    events = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))

    data_dir = tmp_path / "intermediate"
    shard_dir = data_dir / "data" / "train"
    shard_dir.mkdir(parents=True)
    events.write_parquet(shard_dir / "0.parquet")
    return data_dir


def _write_designed_ontology(tmp_path: Path) -> Path:
    """Build the real three-file ontology, including declared non-prefix DAG parents.

    The declared edges make the supplied ancestor names genuinely DAG-derived rather than merely
    abbreviations inferred from the leaves' ``//``-separated spelling.

    The ``codes.parquet``-shaped input frame is::

        shape: (5, 3)
        ┌─────────────────────┬──────────────────┬───────────────────────────────┐
        │ code                ┆ code/vocab_index ┆ parent_codes                  │
        │ ---                 ┆ ---              ┆ ---                           │
        │ str                 ┆ i64              ┆ list[str]                     │
        ╞═════════════════════╪══════════════════╪═══════════════════════════════╡
        │ DX//SEPSIS          ┆ 1                ┆ ["CLINICAL//INFECTION"]       │
        │ LAB//LACTATE        ┆ 2                ┆ ["CLINICAL//BIOMARKER"]       │
        │ HOSPITAL//DISCHARGE ┆ 3                ┆ ["ENCOUNTER//END"]            │
        │ MED//ANTIBIOTIC     ┆ 4                ┆ ["TREATMENT//ANTI_INFECTIVE"] │
        │ FOLLOWUP//VISIT     ┆ 5                ┆ ["ENCOUNTER//FOLLOWUP"]       │
        └─────────────────────┴──────────────────┴───────────────────────────────┘
    """
    codes = pl.DataFrame(
        {
            "code": QUERY_CODES,
            "code/vocab_index": list(range(1, len(QUERY_CODES) + 1)),
            "parent_codes": [
                ["CLINICAL//INFECTION"],
                ["CLINICAL//BIOMARKER"],
                ["ENCOUNTER//END"],
                ["TREATMENT//ANTI_INFECTIVE"],
                ["ENCOUNTER//FOLLOWUP"],
            ],
        }
    )
    nodes, mix = build_ontology(codes)
    ontology_dir = tmp_path / "ontology"
    ontology_dir.mkdir()
    nodes.write_parquet(ontology_dir / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(ontology_dir / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(ontology_dir / EVENT_TO_QUERY_NODES_FILE)
    return ontology_dir


def test_run_labels_every_query_across_designed_mixed_bound_sequences(tmp_path: Path) -> None:
    """All three complete sampled output rows agree with their hand-derived truth tables.

    Seed 260 samples one five-query sequence for every subject.  In sampled output order::

        subject  context      query                 bound                 horizon  expected
        101      2024-01-01   DX//SEPSIS            FOLLOWUP//VISIT            -1  True
        101      2024-01-01   DX//SEPSIS            <none>                     10  True
        101      2024-01-01   HOSPITAL//DISCHARGE   FOLLOWUP//VISIT            -1  False
        101      2024-01-01   DX//SEPSIS            <none>                     10  True
        101      2024-01-01   HOSPITAL//DISCHARGE   FOLLOWUP//VISIT            -1  False

        303      2024-03-04   MED//ANTIBIOTIC       MED//ANTIBIOTIC            -1  False
        303      2024-03-04   FOLLOWUP//VISIT       <none>                     10  True
        303      2024-03-04   FOLLOWUP//VISIT       HOSPITAL//DISCHARGE        -1  True
        303      2024-03-04   MED//ANTIBIOTIC       DX//SEPSIS                 -1  True
        303      2024-03-04   LAB//LACTATE          FOLLOWUP//VISIT            -1  False

        202      2024-02-03   DX//SEPSIS            LAB//LACTATE               -1  True
        202      2024-02-03   HOSPITAL//DISCHARGE   <none>                     10  False
        202      2024-02-03   HOSPITAL//DISCHARGE   <none>                     10  False
        202      2024-02-03   MED//ANTIBIOTIC       HOSPITAL//DISCHARGE        -1  True
        202      2024-02-03   LAB//LACTATE          FOLLOWUP//VISIT            -1  True

    The first row is the requested degenerate-bound case: subject 101 never has a future
    ``FOLLOWUP//VISIT``, but does have future ``DX//SEPSIS``, so the open-ended event-bounded answer
    is ``True``.  Subject 202 also exercises missing-boundary behavior, while subject 303 exercises
    a real future boundary, a target before its boundary, a target after its boundary, and a
    self-bounded query whose target and boundary occur simultaneously.
    """
    data_dir = _write_designed_cohort(tmp_path)
    out_dir = tmp_path / "sequence_tasks"
    cfg = OmegaConf.create(
        {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "query_codes": QUERY_CODES,
            "split": "train",
            "seed": 260,
            "num_sequences": 3,
            "min_queries": 5,
            "max_queries": 5,
            "duration_min": 10,
            "duration_max": 10,
            "duration_distribution": "uniform",
            "min_prediction_times_per_subject": 0,
            "max_workers": 1,
            "eos_first_fraction": 0.0,
            "duration_mode": "random",
            "eventbound_fraction": 0.5,
            "ontology_dir": None,
            "overwrite": False,
        }
    )

    sqs.run(cfg)

    shard_files = sorted((out_dir / "train").glob("*.parquet"))
    assert [fp.name for fp in shard_files] == ["0.parquet"]
    actual = pl.read_parquet(shard_files[0])
    QuerySeqSchema.validate(actual.to_arrow())

    expected = pl.DataFrame(
        {
            "subject_id": pl.Series([101, 303, 202], dtype=pl.Int64),
            "prediction_time": pl.Series(
                [datetime(2024, 1, 1), datetime(2024, 3, 4), datetime(2024, 2, 3)],
                dtype=pl.Datetime("us"),
            ),
            "queries": [
                [
                    "DX//SEPSIS",
                    "DX//SEPSIS",
                    "HOSPITAL//DISCHARGE",
                    "DX//SEPSIS",
                    "HOSPITAL//DISCHARGE",
                ],
                [
                    "MED//ANTIBIOTIC",
                    "FOLLOWUP//VISIT",
                    "FOLLOWUP//VISIT",
                    "MED//ANTIBIOTIC",
                    "LAB//LACTATE",
                ],
                [
                    "DX//SEPSIS",
                    "HOSPITAL//DISCHARGE",
                    "HOSPITAL//DISCHARGE",
                    "MED//ANTIBIOTIC",
                    "LAB//LACTATE",
                ],
            ],
            "durations": pl.Series(
                [
                    [-1.0, 10.0, -1.0, 10.0, -1.0],
                    [-1.0, 10.0, -1.0, -1.0, -1.0],
                    [-1.0, 10.0, 10.0, -1.0, -1.0],
                ],
                dtype=pl.List(pl.Float32),
            ),
            "answers": [
                [True, True, False, True, False],
                [False, True, True, True, False],
                [True, False, False, True, True],
            ],
            "bound_events": [
                ["FOLLOWUP//VISIT", None, "FOLLOWUP//VISIT", None, "FOLLOWUP//VISIT"],
                ["MED//ANTIBIOTIC", None, "HOSPITAL//DISCHARGE", "DX//SEPSIS", "FOLLOWUP//VISIT"],
                ["LAB//LACTATE", None, None, "HOSPITAL//DISCHARGE", "FOLLOWUP//VISIT"],
            ],
        }
    )

    assert actual.equals(expected), (
        "The full Stage 0-4' output disagrees with the designed fixture truth table.\n"
        f"expected: {expected.to_dicts()}\n"
        f"actual:   {actual.to_dicts()}"
    )


def test_run_samples_and_labels_dag_nodes_end_to_end(tmp_path: Path) -> None:
    """The training sampler draws DAG nodes and labels them through descendant events.

    This is the ontology-enabled counterpart of the leaf-only sampled-pipeline test above.  Stage
    1' samples from the full leaf-plus-ancestor universe; Stage 4' expands raw leaf events through
    the closure before answering both ancestor targets and ancestor boundaries.  Seed 1 yields::

        subject  context      pos  query                    duration  boundary                    answer
        202      2024-02-01     0  ENCOUNTER//FOLLOWUP            -1  FOLLOWUP                    False
        202      2024-02-01     1  CLINICAL//BIOMARKER            -1  ENCOUNTER                   True
        202      2024-02-01     2  HOSPITAL//DISCHARGE            -1  HOSPITAL                    False
        202      2024-02-01     3  DX                              10  <none>                      True
        202      2024-02-01     4  DX                              -1  CLINICAL//BIOMARKER         True

        101      2024-01-05     0  LAB//LACTATE                   -1  ENCOUNTER//END              False
        101      2024-01-05     1  HOSPITAL                       -1  DX                           False
        101      2024-01-05     2  DX//SEPSIS                     10  <none>                      False
        101      2024-01-05     3  FOLLOWUP//VISIT                10  <none>                      False
        101      2024-01-05     4  ENCOUNTER//FOLLOWUP            10  <none>                      False

        303      2024-03-03     0  LAB//LACTATE                   -1  CLINICAL//INFECTION         False
        303      2024-03-03     1  HOSPITAL                       10  <none>                      True
        303      2024-03-03     2  ENCOUNTER                      10  <none>                      True
        303      2024-03-03     3  FOLLOWUP//VISIT                10  <none>                      True
        303      2024-03-03     4  TREATMENT                      -1  LAB                          True

    The raw cohort never contains ``CLINICAL//BIOMARKER``, ``DX``, ``HOSPITAL``, ``ENCOUNTER``,
    or ``TREATMENT`` events.  Their positive labels therefore prove that the complete sampled path
    is DAG-aware rather than merely accepting ancestor strings in its output schema.
    """
    data_dir = _write_designed_cohort(tmp_path)
    ontology_dir = _write_designed_ontology(tmp_path)
    out_dir = tmp_path / "dag_sequence_tasks"
    cfg = OmegaConf.create(
        {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "query_codes": QUERY_CODES,
            "split": "train",
            "seed": 1,
            "num_sequences": 3,
            "min_queries": 5,
            "max_queries": 5,
            "duration_min": 10,
            "duration_max": 10,
            "duration_distribution": "uniform",
            "min_prediction_times_per_subject": 0,
            "max_workers": 1,
            "eos_first_fraction": 0.0,
            "duration_mode": "random",
            "eventbound_fraction": 0.5,
            "ontology_dir": str(ontology_dir),
            "overwrite": False,
        }
    )

    sqs.run(cfg)

    shard_files = sorted((out_dir / "train").glob("*.parquet"))
    assert [fp.name for fp in shard_files] == ["0.parquet"]
    actual = pl.read_parquet(shard_files[0])
    QuerySeqSchema.validate(actual.to_arrow())

    expected = pl.DataFrame(
        {
            "subject_id": pl.Series([202, 101, 303], dtype=pl.Int64),
            "prediction_time": pl.Series(
                [datetime(2024, 2, 1), datetime(2024, 1, 5), datetime(2024, 3, 3)],
                dtype=pl.Datetime("us"),
            ),
            "queries": [
                [
                    "ENCOUNTER//FOLLOWUP",
                    "CLINICAL//BIOMARKER",
                    "HOSPITAL//DISCHARGE",
                    "DX",
                    "DX",
                ],
                ["LAB//LACTATE", "HOSPITAL", "DX//SEPSIS", "FOLLOWUP//VISIT", "ENCOUNTER//FOLLOWUP"],
                ["LAB//LACTATE", "HOSPITAL", "ENCOUNTER", "FOLLOWUP//VISIT", "TREATMENT"],
            ],
            "durations": pl.Series(
                [
                    [-1.0, -1.0, -1.0, 10.0, -1.0],
                    [-1.0, -1.0, 10.0, 10.0, 10.0],
                    [-1.0, 10.0, 10.0, 10.0, -1.0],
                ],
                dtype=pl.List(pl.Float32),
            ),
            "answers": [
                [False, True, False, True, True],
                [False, False, False, False, False],
                [False, True, True, True, True],
            ],
            "bound_events": [
                ["FOLLOWUP", "ENCOUNTER", "HOSPITAL", None, "CLINICAL//BIOMARKER"],
                ["ENCOUNTER//END", "DX", None, None, None],
                ["CLINICAL//INFECTION", None, None, None, "LAB"],
            ],
        }
    )

    assert actual.equals(expected), (
        "The ontology-enabled Stage 0-4' output disagrees with the DAG truth table.\n"
        f"expected: {expected.to_dicts()}\n"
        f"actual:   {actual.to_dicts()}"
    )


def test_supplied_sequences_path_labels_complete_dense_grid(tmp_path: Path) -> None:
    """A designed YAML sequence file is labeled correctly at all three supplied contexts.

    The supplied sequence file contains::

        sequence              pos  query                 duration  boundary
        duration_windows        0  DX//SEPSIS                  10  <none>
        duration_windows        1  HOSPITAL//DISCHARGE         10  <none>
        duration_windows        2  MED//ANTIBIOTIC             10  <none>

        mixed_event_bounds      0  DX//SEPSIS                  -1  FOLLOWUP//VISIT
        mixed_event_bounds      1  MED//ANTIBIOTIC             10  <none>
        mixed_event_bounds      2  LAB//LACTATE                -1  FOLLOWUP//VISIT

        boundary_edge_cases     0  MED//ANTIBIOTIC             -1  MED//ANTIBIOTIC
        boundary_edge_cases     1  FOLLOWUP//VISIT             -1  HOSPITAL//DISCHARGE
        boundary_edge_cases     2  HOSPITAL//DISCHARGE         -1  FOLLOWUP//VISIT

        dag_ancestor_queries     0  CLINICAL//INFECTION         10  <none>
        dag_ancestor_queries     1  TREATMENT//ANTI_INFECTIVE   -1  ENCOUNTER//END
        dag_ancestor_queries     2  CLINICAL//BIOMARKER         -1  ENCOUNTER//FOLLOWUP

    This drives the full dense evaluation-sequence entry point: parse ``sequences_path``, validate
    every query and boundary against the vocabulary, partition the supplied cohort by shard,
    cross-join its three contexts with the four designed sequences, label every position, align
    to ``QuerySeqSchema``, and write the output and unique-context parquets.

    The supplied contexts frame is::

        shape: (3, 2)
        ┌────────────┬─────────────────────┐
        │ subject_id ┆ prediction_time     │
        │ ---        ┆ ---                 │
        │ i64        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╡
        │ 101        ┆ 2024-01-01 00:00:00 │
        │ 202        ┆ 2024-02-03 00:00:00 │
        │ 303        ┆ 2024-03-04 00:00:00 │
        └────────────┴─────────────────────┘

    The complete answer truth table is::

        sequence                 subject 101   subject 202   subject 303
        duration_windows         [T, F, F]     [T, F, T]     [F, T, T]
        mixed_event_bounds       [T, F, F]     [T, T, T]     [F, T, F]
        boundary_edge_cases      [F, F, F]     [F, F, F]     [F, T, F]
        dag_ancestor_queries      [T, F, F]     [T, T, T]     [F, T, F]

    In ``mixed_event_bounds`` position 0, subject 101 has future ``DX//SEPSIS`` but no future
    ``FOLLOWUP//VISIT`` boundary, so its answer is ``True``.  The same designed query is ``False``
    for subject 303 because its follow-up occurs before its later sepsis event.

    ``dag_ancestor_queries`` names only declared ontology parents: none of those strings occur in
    the raw event frame.  Its labels can be correct only if descendant target events and descendant
    boundary events are expanded through the DAG before labeling.
    """
    data_dir = _write_designed_cohort(tmp_path)
    ontology_dir = _write_designed_ontology(tmp_path)

    contexts_path = tmp_path / "contexts.parquet"
    contexts = pl.DataFrame(
        {
            "subject_id": pl.Series([101, 202, 303], dtype=pl.Int64),
            "prediction_time": pl.Series(
                [datetime(2024, 1, 1), datetime(2024, 2, 3), datetime(2024, 3, 4)],
                dtype=pl.Datetime("us"),
            ),
        }
    )
    contexts.write_parquet(contexts_path)

    sequences_path = tmp_path / "designed_sequences.yaml"
    sequences_path.write_text(
        yaml.safe_dump(
            {
                "duration_windows": [
                    ["DX//SEPSIS", 10],
                    ["HOSPITAL//DISCHARGE", 10],
                    ["MED//ANTIBIOTIC", 10],
                ],
                "mixed_event_bounds": [
                    ["DX//SEPSIS", -1, "FOLLOWUP//VISIT"],
                    ["MED//ANTIBIOTIC", 10],
                    ["LAB//LACTATE", -1, "FOLLOWUP//VISIT"],
                ],
                "boundary_edge_cases": [
                    ["MED//ANTIBIOTIC", -1, "MED//ANTIBIOTIC"],
                    ["FOLLOWUP//VISIT", -1, "HOSPITAL//DISCHARGE"],
                    ["HOSPITAL//DISCHARGE", -1, "FOLLOWUP//VISIT"],
                ],
                "dag_ancestor_queries": [
                    ["CLINICAL//INFECTION", 10],
                    ["TREATMENT//ANTI_INFECTIVE", -1, "ENCOUNTER//END"],
                    ["CLINICAL//BIOMARKER", -1, "ENCOUNTER//FOLLOWUP"],
                ],
            },
            sort_keys=False,
        )
    )

    out_dir = tmp_path / "evaluation_sequence_tasks"
    cfg = OmegaConf.create(
        {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "query_codes": QUERY_CODES,
            "split": "train",
            "seed": 1,
            "prediction_times_per_subject": 99,
            "min_context_per_subject": 99,
            "subject_subsample_fraction": None,
            "contexts_path": str(contexts_path),
            "write_unique_prediction_times": True,
            "sequences_path": str(sequences_path),
            # These sampling knobs are deliberately irrelevant when sequences_path is supplied.
            "n_sequences": 99,
            "min_queries": 1,
            "max_queries": 1,
            "duration_min": 1,
            "duration_max": 1,
            "duration_distribution": "uniform",
            "eos_first_fraction": 0.0,
            "duration_mode": "random",
            "eventbound_fraction": 0.0,
            "ontology_dir": str(ontology_dir),
            "overwrite": False,
        }
    )

    eval_sqs.main.__wrapped__(cfg)

    shard_files = sorted((out_dir / "eval" / "train").glob("*.parquet"))
    assert [fp.name for fp in shard_files] == ["0.parquet"]
    actual = pl.read_parquet(shard_files[0])
    QuerySeqSchema.validate(actual.to_arrow())

    duration_queries = ["DX//SEPSIS", "HOSPITAL//DISCHARGE", "MED//ANTIBIOTIC"]
    mixed_queries = ["DX//SEPSIS", "MED//ANTIBIOTIC", "LAB//LACTATE"]
    edge_queries = ["MED//ANTIBIOTIC", "FOLLOWUP//VISIT", "HOSPITAL//DISCHARGE"]
    dag_queries = ["CLINICAL//INFECTION", "TREATMENT//ANTI_INFECTIVE", "CLINICAL//BIOMARKER"]
    mixed_bounds = ["FOLLOWUP//VISIT", None, "FOLLOWUP//VISIT"]
    edge_bounds = ["MED//ANTIBIOTIC", "HOSPITAL//DISCHARGE", "FOLLOWUP//VISIT"]
    dag_bounds = [None, "ENCOUNTER//END", "ENCOUNTER//FOLLOWUP"]

    expected = pl.DataFrame(
        {
            "subject_id": pl.Series([101] * 4 + [202] * 4 + [303] * 4, dtype=pl.Int64),
            "prediction_time": pl.Series(
                [datetime(2024, 1, 1)] * 4
                + [datetime(2024, 2, 3)] * 4
                + [datetime(2024, 3, 4)] * 4,
                dtype=pl.Datetime("us"),
            ),
            "queries": [duration_queries, mixed_queries, edge_queries, dag_queries] * 3,
            "durations": pl.Series(
                [
                    [10.0, 10.0, 10.0],
                    [-1.0, 10.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [10.0, -1.0, -1.0],
                ]
                * 3,
                dtype=pl.List(pl.Float32),
            ),
            "answers": [
                [True, False, False],
                [True, False, False],
                [False, False, False],
                [True, False, False],
                [True, False, True],
                [True, True, True],
                [False, False, False],
                [True, True, True],
                [False, True, True],
                [False, True, False],
                [False, True, False],
                [False, True, False],
            ],
            "bound_events": [
                [None, None, None],
                mixed_bounds,
                edge_bounds,
                dag_bounds,
                [None, None, None],
                mixed_bounds,
                edge_bounds,
                dag_bounds,
                [None, None, None],
                mixed_bounds,
                edge_bounds,
                dag_bounds,
            ],
        }
    )

    assert actual.equals(expected), (
        "The supplied-sequence dense grid disagrees with the designed fixture truth table.\n"
        f"expected: {expected.to_dicts()}\n"
        f"actual:   {actual.to_dicts()}"
    )

    unique = pl.read_parquet(out_dir / "eval_unique" / "train" / "0.parquet")
    assert unique.equals(contexts), "the unique-context output must reproduce the supplied cohort"
