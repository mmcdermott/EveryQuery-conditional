"""Sampling-first query-*sequence* label generator for conditional pre-training.

Sibling of :mod:`~every_query.generate_tasks.sample_tasks`, running the *same* 5-stage pipeline but
emitting, per sampled patient context, an ordered *sequence* of queries rather than one scattered
``(code, duration)``::

    Stage 0   build + cache the canonical prediction-time map and subject counts (reused as-is)
    Stage 1'  sample ``num_sequences`` variable-length query sequences (QuerySequenceDistribution)
    Stage 2   sample ``num_sequences`` patient contexts (reused as-is)
    Stage 3'  resolve ``prediction_time_index -> prediction_time``, zip, write per-shard index
    Stage 4'  binary-label each index shard independently and write the final dataset parquet

Stages 0-3' run sequentially in the driver (:func:`run`); Stage 4' fans out one
:func:`label_one_sequence_shard` worker per shard via ``ProcessPoolExecutor``.  Only the primed
stages differ from ``sample_tasks``: Stage 1' draws ``L ~ Uniform{min_queries..max_queries}``
queries per sequence instead of one, and Stage 3' tags each row with ``_ctx_id`` / ``_position`` so
Stage 4' can reassemble the list columns.

Every query is an ordinary vocabulary code drawn uniformly - including the end-of-timeline code
``TIMELINE//END``.  There is **no** privileged censor position and **no** null answer: censoring is
represented explicitly as a ``TIMELINE//END`` query whose answer is ``True`` exactly when the record
ends inside the window.  The model trained on these sequences learns
``P(A_j | patient, Q_1..Q_j, A_1..A_{j-1})``.

Output rows follow :class:`~every_query.data.schema.QuerySeqSchema`.

Two optional, default-off sweep knobs reweight sequence *structure* without touching the base
code/duration draw: ``eos_first_fraction`` (force position 0 to ``TIMELINE//END``) and
``duration_mode`` (``random`` | ``same`` | ``nondecreasing``).

A second, independent entry path labels a **supplied** ``(subject_id, prediction_time)`` cohort -
see :func:`run_worker`.  It bypasses Stages 0/2/3' entirely (the contexts are given, not sampled)
and is selected by passing ``contexts_path=`` on the CLI.
"""

import os

# Pin polars to a single thread BEFORE importing polars (or anything that transitively imports it -
# meds, every_query.data.schema), mirroring ``sample_tasks``.  Stage 4' workers inherit this env;
# with process-level fan-out already saturating cores, intra-op polars threads would oversubscribe.
# A transitive ``import polars`` above this line would silently defeat the setting.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import json
import logging
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.data.seq_dataset import EOS_CODE
from every_query.generate_tasks.sample_tasks import (
    INDEX_DIRNAME,
    LABELED_DIRNAME,
    QueryDistribution,
    QuerySpec,
    _atomic_write_json,
    _atomic_write_parquet,
    _clean_stale_temps,
    _index_fingerprint,
    _prune_stale_outputs,
    _read_event_shard,
    _require_path_arg,
    build_prediction_times,
    default_artifacts_dir,
    index_path,
    prediction_time_counts_path,
    prediction_times_path,
    read_query_codes,
    resolve_workers,
    sample_patient_contexts,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)

CTX_ID_COL = "_ctx_id"
POSITION_COL = "_position"

INDEX_COLUMNS = [
    CTX_ID_COL,
    POSITION_COL,
    TaskQuerySchema.subject_id_name,
    TaskQuerySchema.prediction_time_name,
    TaskQuerySchema.query_name,
    TaskQuerySchema.duration_days_name,
]


# ---------------------------------------------------------------------------
# Stage 1' - the sequence query distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySequenceDistribution(QueryDistribution):
    """Stage 1' - draw variable-length *sequences* of :class:`QuerySpec` s.

    Extends :class:`~every_query.generate_tasks.sample_tasks.QueryDistribution` rather than
    reimplementing it: the base ``sample`` call is what actually draws every code and duration, so
    a sequence run and a training-task run with the same ``rng`` and the same total query count
    produce **identical** ``(code, duration_days)`` draws.  That equality is the distribution-parity
    contract between the two samplers (durations are floats here too - no day-rounding).

    Sequence *structure* - how many queries each sequence gets, and the two sweep reweightings -
    is drawn from a **separate** generator (``structure_rng``) so it never perturbs the base draw.

    Args:
        min_queries: Minimum queries per sequence (``>= 1``).
        max_queries: Maximum queries per sequence (``>= min_queries``).
        eos_first_fraction: Probability a sequence's position 0 is forced to the end-of-timeline
            code ``TIMELINE//END`` (upweights the censoring-control pattern).  ``0.0`` = pure random.
        duration_mode: Within-sequence duration coupling.  ``"random"`` (independent, default);
            ``"same"`` (every query reuses the sequence's first duration); ``"nondecreasing"``
            (durations sorted ascending) - the latter two reflect that many censoring-style asks
            reuse one horizon.

    Examples:
        >>> import numpy as np
        >>> dist = QuerySequenceDistribution(
        ...     ["A", "B", EOS_CODE], min_duration=1.0, max_duration=365.0,
        ...     duration_distribution="log-uniform", min_queries=1, max_queries=4)
        >>> seqs = dist.sample_sequences(5, np.random.default_rng(0), np.random.default_rng(1))
        >>> len(seqs)
        5
        >>> all(1 <= len(s) <= 4 for s in seqs)
        True
        >>> all(q.code in {"A", "B", EOS_CODE} for s in seqs for q in s)
        True

        Parity with the base distribution - the flattened draw is identical to what
        ``QueryDistribution.sample`` yields for the same generator and total:

        >>> total = sum(len(s) for s in seqs)
        >>> base = QueryDistribution(["A", "B", EOS_CODE], 1.0, 365.0, "log-uniform")
        >>> [q for s in seqs for q in s] == base.sample(total, np.random.default_rng(0))
        True

        ``eos_first_fraction=1.0`` forces every position 0:

        >>> forced = QuerySequenceDistribution(
        ...     ["A", "B", EOS_CODE], 1.0, 365.0, "log-uniform", 2, 2, eos_first_fraction=1.0)
        >>> seqs = forced.sample_sequences(4, np.random.default_rng(0), np.random.default_rng(1))
        >>> {s[0].code for s in seqs}
        {'TIMELINE//END'}

        ``duration_mode="same"`` collapses each sequence to one horizon:

        >>> same = QuerySequenceDistribution(
        ...     ["A", "B"], 1.0, 365.0, "log-uniform", 3, 3, duration_mode="same")
        >>> seqs = same.sample_sequences(3, np.random.default_rng(0), np.random.default_rng(1))
        >>> all(len({q.duration_days for q in s}) == 1 for s in seqs)
        True

        ``num_sequences=0`` is valid and returns an empty list:

        >>> dist.sample_sequences(0, np.random.default_rng(0), np.random.default_rng(1))
        []
    """

    min_queries: int = 1
    max_queries: int = 5
    eos_first_fraction: float = 0.0
    duration_mode: str = "random"

    _VALID_DURATION_MODES = ("random", "same", "nondecreasing")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_queries < 1:
            raise ValueError(f"min_queries must be >= 1 (got {self.min_queries})")
        if self.max_queries < self.min_queries:
            raise ValueError(f"max_queries ({self.max_queries}) must be >= min_queries ({self.min_queries})")
        if not 0.0 <= self.eos_first_fraction <= 1.0:
            raise ValueError(f"eos_first_fraction must be in [0, 1] (got {self.eos_first_fraction})")
        if self.duration_mode not in self._VALID_DURATION_MODES:
            raise ValueError(
                f"duration_mode must be one of {self._VALID_DURATION_MODES} (got {self.duration_mode!r})"
            )
        # Forcing a code that is not in the sampling universe would emit queries the model's
        # vocabulary cannot encode - a config error worth failing on rather than discovering as an
        # out-of-vocab crash three stages later.
        if self.eos_first_fraction > 0 and EOS_CODE not in self.query_codes:
            raise ValueError(
                f"eos_first_fraction={self.eos_first_fraction} forces position 0 to {EOS_CODE!r}, "
                "but that code is not in query_codes"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "QuerySequenceDistribution":
        """Build from the sequence sampler's Hydra config plus :func:`read_query_codes`.

        Reads the duration bounds from ``duration_min`` / ``duration_max`` - the conditional
        configs' historical key names, deliberately kept (rather than renamed to the base class's
        ``min_duration`` / ``max_duration``) so this config and
        ``sample_evaluation_query_sequences_config.yaml`` stay key-identical; a test asserts that
        equality, since a training/eval mismatch in the duration draw silently invalidates eval.
        """
        return cls(
            query_codes=read_query_codes(cfg.query_codes),
            min_duration=float(cfg.duration_min),
            max_duration=float(cfg.duration_max),
            duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
            min_queries=int(cfg.min_queries),
            max_queries=int(cfg.max_queries),
            eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
            duration_mode=str(cfg.get("duration_mode", "random")),
        )

    def sample_sequences(
        self,
        num_sequences: int,
        query_rng: np.random.Generator,
        structure_rng: np.random.Generator,
    ) -> list[list[QuerySpec]]:
        """Draw ``num_sequences`` sequences of ``L ~ Uniform{min_queries..max_queries}`` queries.

        Two caller-owned generators keep the axes independent (mirroring the redesign's invariant
        5): ``query_rng`` feeds *only* the inherited :meth:`QueryDistribution.sample` (seeded via
        ``derive_seed(seed, "queries")``, matching the training sampler), while ``structure_rng``
        (``derive_seed(seed, "sequences")``) draws lengths and applies the two sweep reweightings.
        Draws happen in a fixed order, so output is deterministic for fixed generators.

        Raises:
            ValueError: If ``num_sequences < 0``.
        """
        if num_sequences < 0:
            raise ValueError(f"num_sequences must be >= 0 (got {num_sequences})")
        if num_sequences == 0:
            return []

        lengths = structure_rng.integers(self.min_queries, self.max_queries + 1, size=num_sequences)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        # The single base draw - this is the parity anchor with the training sampler.
        specs = self.sample(int(offsets[-1]), query_rng)
        sequences = [list(specs[offsets[i] : offsets[i + 1]]) for i in range(num_sequences)]

        if self.eos_first_fraction > 0:
            forced = structure_rng.random(num_sequences) < self.eos_first_fraction
            for i in np.flatnonzero(forced):
                head = sequences[i][0]
                sequences[i][0] = QuerySpec(code=EOS_CODE, duration_days=head.duration_days)

        if self.duration_mode == "same":
            sequences = [[QuerySpec(q.code, s[0].duration_days) for q in s] for s in sequences]
        elif self.duration_mode == "nondecreasing":
            sequences = [
                [QuerySpec(q.code, d) for q, d in zip(s, sorted(x.duration_days for x in s), strict=True)]
                for s in sequences
            ]

        return sequences


def _expand_sequences(sequences: list[list[QuerySpec]]) -> pl.DataFrame:
    """Flatten sequences into one row per query: ``(_ctx_id, _position, query, duration_days)``.

    ``_ctx_id`` is the sequence's position in ``sequences`` (so it is unique across the whole run,
    not just within a shard) and ``_position`` is the query's zero-based rank inside its sequence.
    Rows come out in ``(_ctx_id, _position)`` order.

    Examples:
        >>> seqs = [[QuerySpec("A", 1.0), QuerySpec("B", 2.0)], [QuerySpec("C", 3.0)]]
        >>> _expand_sequences(seqs).to_dicts() == [
        ...     {"_ctx_id": 0, "_position": 0, "query": "A", "duration_days": 1.0},
        ...     {"_ctx_id": 0, "_position": 1, "query": "B", "duration_days": 2.0},
        ...     {"_ctx_id": 1, "_position": 0, "query": "C", "duration_days": 3.0}]
        True
    """
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    if lengths.size and lengths.min() < 1:
        raise ValueError("every sequence must contain at least one query")

    ctx_ids = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    positions = (
        np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])
        if lengths.size
        else np.empty(0, dtype=np.int64)
    )
    return pl.DataFrame(
        {
            CTX_ID_COL: pl.Series(ctx_ids, dtype=pl.UInt32),
            POSITION_COL: pl.Series(positions, dtype=pl.Int64),
            TaskQuerySchema.query_name: pl.Series([q.code for s in sequences for q in s], dtype=pl.Utf8),
            TaskQuerySchema.duration_days_name: pl.Series(
                [q.duration_days for s in sequences for q in s], dtype=pl.Float32
            ),
        }
    )


def _attach_queries_to_contexts(contexts: pl.DataFrame, per_query: pl.DataFrame) -> pl.DataFrame:
    """Gather one context row per query row and hstack the query columns.

    A row-gather by ``_ctx_id`` (which *is* the context row index) rather than a join: it is
    positional, order-preserving, and needs no join key on a frame the size of the fan-out.  Polars
    joins carry no order guarantee, so a join here would silently scramble ``_position`` order.
    """
    return contexts[per_query[CTX_ID_COL].to_numpy()].with_columns(
        per_query[CTX_ID_COL],
        per_query[POSITION_COL],
        per_query[TaskQuerySchema.query_name],
        per_query[TaskQuerySchema.duration_days_name],
    )


def build_sequence_index_df(
    contexts: pl.DataFrame,
    query_codes: list[str],
    min_queries: int,
    max_queries: int,
    duration_low: float,
    duration_high: float,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    duration_distribution: str = "log-uniform",
) -> pl.DataFrame:
    """Expand *already-resolved* contexts into a flat per-query index frame (in-memory variant).

    Used by the supplied-cohort path and by the evaluation-grid sampler, where the
    ``(subject_id, prediction_time)`` pairs are given rather than drawn from Stage 0/2 - so there
    is no ``prediction_time_index`` to resolve and no per-shard partitioning to do.  The sampled
    path uses :func:`build_sequence_index` instead.

    Returns:
        DataFrame ``(_ctx_id, _position, subject_id, prediction_time, query, duration_days)``.

    Examples:
        >>> from datetime import datetime
        >>> ctx = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        ... })
        >>> idx = build_sequence_index_df(ctx, ["A", "B", EOS_CODE], 1, 4, 1, 365, seed=0)
        >>> idx.group_by("_ctx_id").len()["len"].max() <= 4
        True
        >>> bool(idx["query"].is_in(["A", "B", EOS_CODE]).all())
        True

        Durations are floats, not rounded to whole days (they come from ``QueryDistribution``):

        >>> idx = build_sequence_index_df(ctx, ["A"], 8, 8, 1, 365, seed=0)
        >>> bool((idx["duration_days"] != idx["duration_days"].round()).any())
        True

        Determinism:

        >>> build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, 7).equals(
        ...     build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, 7))
        True

        ``eos_first_fraction=1.0`` forces position 0 to the end-of-timeline query:

        >>> idx = build_sequence_index_df(
        ...     ctx, ["A", "B", EOS_CODE], 2, 2, 1, 365, 0, eos_first_fraction=1.0)
        >>> idx.filter(pl.col("_position") == 0)["query"].unique().to_list()
        ['TIMELINE//END']
    """
    dist = QuerySequenceDistribution(
        query_codes=list(query_codes),
        min_duration=float(duration_low),
        max_duration=float(duration_high),
        duration_distribution=duration_distribution,
        min_queries=min_queries,
        max_queries=max_queries,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
    )

    if contexts.height == 0:
        return contexts.head(0).select(
            pl.lit(0, dtype=pl.UInt32).alias(CTX_ID_COL),
            pl.lit(0, dtype=pl.Int64).alias(POSITION_COL),
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            pl.lit("", dtype=pl.Utf8).alias(TaskQuerySchema.query_name),
            pl.lit(0.0, dtype=pl.Float32).alias(TaskQuerySchema.duration_days_name),
        )

    sequences = dist.sample_sequences(
        contexts.height,
        np.random.default_rng(derive_seed(seed, "queries")),
        np.random.default_rng(derive_seed(seed, "sequences")),
    )
    per_query = _expand_sequences(sequences)
    return _attach_queries_to_contexts(contexts, per_query).select(INDEX_COLUMNS)


# ---------------------------------------------------------------------------
# Stage 3' - zip sequences with sampled contexts, resolve times, write the index
# ---------------------------------------------------------------------------


def resolve_prediction_times(
    contexts: pl.DataFrame,
    training_task_artifacts_dir: Path,
    split: str,
    shard: str,
) -> pl.DataFrame:
    """Resolve one shard's ``prediction_time_index`` ranks into real ``prediction_time`` stamps.

    Stage 2 returns ``(subject_id, shard, prediction_time_index)`` where the index is an ``Int64``
    *rank*, not a datetime; everything downstream needs the timestamp.  Upstream buries this join
    inside :func:`~every_query.generate_tasks.sample_tasks.build_index`, which is not reusable here
    (it zips one query per context), so it lives on its own — both :func:`build_sequence_index` and
    the evaluation-grid sampler's sampled-context branch join exactly this way.

    ``contexts`` must be a **single shard's** rows: the join reads only that shard's Stage 0
    ``_prediction_times/{shard}.parquet`` map, which keeps the driver holding one payload-free map
    at a time so memory stays flat as the shard count grows.

    Preserves the two upstream guards verbatim:

    - a **join-key dtype check** — a mismatch (e.g. ``Int64`` vs ``UInt32`` ``subject_id``) makes
      polars produce a silent all-null join, so it fails loudly instead of being cast over;
    - a **left-join-then-raise on null** ``prediction_time`` — the join is total by design (the
      contexts carry the same eligibility bound Stage 0 wrote the map with), so a null is a hard
      error, reported with a small sample of offending keys.  An inner join would silently drop the
      rows instead.

    Returns:
        ``contexts`` with the map's ``time`` column joined on and renamed to ``prediction_time``
        (``prediction_time_index`` is retained; select it away if unwanted).

    Raises:
        ValueError: On a join-key dtype mismatch, or if any row resolves to a null timestamp.
    """
    join_keys = ["subject_id", "prediction_time_index"]
    pt_map = pl.read_parquet(prediction_times_path(training_task_artifacts_dir, split, str(shard)))

    for key in join_keys:
        ctx_dtype, map_dtype = contexts.schema[key], pt_map.schema[key]
        if ctx_dtype != map_dtype:
            raise ValueError(
                f"Join key {key!r} dtype mismatch: contexts has {ctx_dtype}, the "
                f"_prediction_times map has {map_dtype}. A mismatch silently produces null "
                "prediction_times; fix the dtype upstream."
            )

    joined = contexts.join(pt_map, on=join_keys, how="left").rename(
        {"time": TaskQuerySchema.prediction_time_name}
    )

    null_rows = joined.filter(pl.col(TaskQuerySchema.prediction_time_name).is_null())
    if null_rows.height > 0:
        sample = null_rows.select(join_keys).head(5).to_dicts()
        raise ValueError(
            f"Shard {shard}: {null_rows.height} contexts have null prediction_time after join. "
            "The _prediction_times map may be stale or contexts reference invalid "
            f"(subject_id, prediction_time_index) pairs. Sample of offending rows: {sample}"
        )

    return joined


def build_sequence_index(
    sequences: list[list[QuerySpec]],
    contexts: pl.DataFrame,
    training_task_artifacts_dir: Path,
    split: str,
) -> int:
    """Stage 3': zip sequences with contexts, resolve prediction times, write partitioned index.

    The sequence analogue of :func:`~every_query.generate_tasks.sample_tasks.build_index`.  Where
    that function ``np.repeat`` s one query across ``num_contexts_per_query`` contexts, this one
    expands each context into the ``L`` queries of its sequence, tagged with ``_ctx_id`` /
    ``_position`` so Stage 4' can reassemble the list columns.  ``sequences[i]`` belongs to
    ``contexts`` row ``i`` - the two are drawn with the same length and zipped positionally.

    Everything else matches ``build_index`` deliberately: the rank -> timestamp resolution runs one
    shard at a time through :func:`resolve_prediction_times` (one ``read_parquet`` per shard, so
    driver memory stays flat, with its dtype guard and raise-on-null intact), and ``sort("shard")``
    + ``group_by(maintain_order=True)`` give deterministic shard order.

    Every query of a sequence shares one context, hence one shard, so no sequence is ever split
    across index partitions.

    Args:
        sequences: Stage 1' output - one list of :class:`QuerySpec` per sequence.
        contexts: Stage 2 output - ``(subject_id, shard, prediction_time_index)``, same height.
        training_task_artifacts_dir: Intermediate-artifacts root.
        split: Dataset split name (e.g. ``"train"``).

    Returns:
        ``n_shards`` - the number of index partitions written.

    Raises:
        ValueError: If ``len(sequences) != contexts.height``, or a context fails to resolve a
            prediction time (null after join).
        AssertionError: If ``sequences`` or ``contexts`` is empty - there is no supported
            empty-budget path through this pipeline.
    """
    if len(sequences) != contexts.height:
        raise ValueError(
            f"len(sequences) ({len(sequences)}) must equal contexts.height ({contexts.height}); "
            "Stage 1' and Stage 2 are drawn with the same length and zipped positionally"
        )

    assert sequences, "sequences must be non-empty"
    assert contexts.height > 0, "contexts must be non-empty"

    per_query = _expand_sequences(sequences)
    combined = _attach_queries_to_contexts(contexts, per_query)

    index_dir = training_task_artifacts_dir / split / INDEX_DIRNAME
    if index_dir.exists():
        shutil.rmtree(index_dir)

    n_shards = 0
    # Sort + ``maintain_order=True`` make the order shards are processed (and logged) deterministic;
    # ``group_by`` alone is not order-preserving.  Output content is unaffected (shards are
    # independent), but the deterministic order keeps reruns and logs stable.
    combined = combined.sort("shard")
    for shard_key, shard_group in combined.group_by("shard", maintain_order=True):
        (shard_name,) = shard_key
        joined = resolve_prediction_times(shard_group, training_task_artifacts_dir, split, str(shard_name))

        _atomic_write_parquet(
            joined.select(INDEX_COLUMNS).sort(CTX_ID_COL, POSITION_COL),
            index_path(training_task_artifacts_dir, split, str(shard_name)),
        )
        n_shards += 1

    return n_shards


# ---------------------------------------------------------------------------
# Stage 4' - per-shard binary labeling
# ---------------------------------------------------------------------------


def label_binary_occurrence(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label every query with a binary *observed-occurrence* answer and reassemble into list rows.

    ``answer = True`` iff an event whose code equals the query occurs strictly within
    ``(prediction_time, prediction_time + duration_days)`` **and is present in the record**.  There
    is no censoring/null: an event we cannot observe (because the record ends first) is ``False``.
    Censoring is captured separately by the ``TIMELINE//END`` query - which, being the last event
    of each record, answers ``True`` exactly when the record ends within the window.

    The strict ``>`` lower bound is enforced by shifting the asof key ``+1us`` (microsecond-precision
    datetimes), mirroring ``sample_tasks.evaluate_index_df``.

    Returns:
        DataFrame ``(subject_id, prediction_time, queries, durations, answers)``, one row per
        ``_ctx_id``, lists ordered by ``_position``.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [datetime(2024, 1, 1), datetime(2024, 1, 5), datetime(2024, 1, 20)],
        ...     "code": ["A", "B", "TIMELINE//END"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": [0, 0, 0],
        ...     "_position": [0, 1, 2],
        ...     "subject_id": [1, 1, 1],
        ...     "prediction_time": [datetime(2024, 1, 2)] * 3,
        ...     "query": ["B", "A", "TIMELINE//END"],
        ...     "duration_days": [10.0, 10.0, 10.0],
        ... }).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("duration_days").cast(pl.Float32),
        ...     pl.col("_ctx_id").cast(pl.UInt32))
        >>> row = label_binary_occurrence(idx, events).row(0, named=True)
        >>> row["answers"]  # B in (2,12]=yes; A at day1 not >day2 =no; END at day20 not in window =no
        [True, False, False]
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    left = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts")).sort(
        sid, q, "_pts"
    )
    right = (
        events_df.rename({DataSchema.code_name: q})
        .select(sid, q, DataSchema.time_name)
        .sort(sid, q, DataSchema.time_name)
    )
    joined = left.join_asof(
        right, by=[sid, q], left_on="_pts", right_on=DataSchema.time_name, strategy="forward"
    )
    window_end = pl.col(pt) + pl.duration(days=pl.col(d))
    answer = pl.col(DataSchema.time_name).is_not_null() & (pl.col(DataSchema.time_name) < window_end)

    flat = joined.with_columns(answer.alias("answer")).sort(CTX_ID_COL, POSITION_COL)
    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(sid).first(),
            pl.col(pt).first(),
            pl.col(q).alias("queries"),
            pl.col(d).alias("durations"),
            pl.col("answer").alias("answers"),
        )
        .drop(CTX_ID_COL)
    )


def label_query_sequences(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Stage 4' labeling entry point: answer every query in ``index_df`` against ``events_df``.

    The single seam through which *all* sequence labeling flows — both the sharded training
    path (:func:`label_one_sequence_shard`) and the dense evaluation grid — so a query form
    added here is answered identically in training data and eval grids.  A form that labels one
    way in training and another in evaluation produces a grid that looks fine and silently
    measures the wrong thing.

    Today every query is a plain occurrence question and this delegates unchanged to
    :func:`label_binary_occurrence`, which stays the regression anchor for that semantics.

    This must remain a **module-level function**: Stage 4' fans shards out through a
    ``ProcessPoolExecutor`` with ``mp_context="spawn"``, so a closure or a locally-defined
    callable would fail to pickle.

    Args:
        index_df: Flat per-query index frame (one row per query of each sequence).
        events_df: The event stream to answer against.

    Returns:
        One row per sequence, with the ``queries``/``durations``/``answers`` list columns.
    """
    return label_binary_occurrence(index_df, events_df)


def label_one_sequence_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Stage 4' worker: label one index partition and write the final ``QuerySeqSchema`` parquet.

    The sequence analogue of :func:`~every_query.generate_tasks.sample_tasks.label_one_shard`,
    including its fingerprint-keyed skip: an existing output is reused only when the recorded
    :func:`~every_query.generate_tasks.sample_tasks._index_fingerprint` matches the current Stage 3'
    partition, so a changed sampling config relabels even under ``overwrite=False``.  A
    missing/unreadable fingerprint is treated as stale.

    Returns:
        ``(shard, status)`` where status is ``"skipped"`` or ``"labeled"``.
    """
    final = out_dir / f"{shard}.parquet"
    fingerprint_fp = index_dir.parent / LABELED_DIRNAME / f"{shard}.json"

    index_df = pl.read_parquet(index_dir / f"{shard}.parquet")
    current_fingerprint = _index_fingerprint(index_df)

    if not overwrite and final.exists():
        try:
            recorded = json.loads(fingerprint_fp.read_text()).get("index_fingerprint")
        except (OSError, json.JSONDecodeError, AttributeError):
            recorded = None
        if recorded == current_fingerprint:
            return shard, "skipped"

    _clean_stale_temps(out_dir, shard)

    events_df = _read_event_shard(data_dir / f"{shard}.parquet")

    labeled = label_query_sequences(index_df, events_df)
    aligned = QuerySeqSchema.align(labeled.to_arrow())

    _atomic_write_parquet(pl.from_arrow(aligned), final)
    # Record the index fingerprint *after* the output is committed so a present sidecar always
    # describes a present, complete output (the parquet is the value, the sidecar its provenance).
    _atomic_write_json({"index_fingerprint": current_fingerprint}, fingerprint_fp)
    return shard, "labeled"


def _label_sequence_shards(
    shards: list[str],
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool,
    n_workers: int,
) -> None:
    """Fan one :func:`label_one_sequence_shard` worker out per shard via a spawn-based pool.

    ``"spawn"``, not the Linux default ``"fork"``: by Stage 4' the driver has already run polars
    (which starts a rayon threadpool), and forking while those threads hold locks leaves the child
    with inherited-but-locked mutexes, deadlocking the worker the moment it touches polars (#210).
    """
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
        futs = {
            ex.submit(label_one_sequence_shard, s, index_dir, data_dir, out_dir, overwrite): s for s in shards
        }
        for fut in as_completed(futs):
            fut.result()  # re-raise so a failed shard aborts the run loudly


def _validate_sequence_count(out_files: list[Path], expected: int) -> int:
    """Sum the final shards' row counts and check they account for every sampled sequence.

    One output row is one sequence, so the union of the split's shards must have exactly
    ``num_sequences`` rows; a mismatch means a dropped context or a partially-written shard.  Uses
    the parquet metadata row count, so no payload is read into memory.
    """
    written = sum(pl.scan_parquet(fp).select(pl.len()).collect().item() for fp in out_files)
    if written != expected:
        raise ValueError(
            f"Expected {expected:,} labeled sequences across {len(out_files):,} shard(s) but found "
            f"{written:,}. The output directory may hold a partial run; rerun with overwrite=true."
        )
    return written


def label_sequence_shards(
    cfg: DictConfig,
    path_to_data: Path,
    seq_tasks_dir: Path,
    seq_task_artifacts_dir: Path,
    total_sequences: int,
) -> int:
    """Stage 4': fan one labeling worker out per Stage 3' index shard; return sequences written.

    Shards are exactly the Stage 3' index partitions; workers receive ids/paths (never DataFrames)
    and write their own atomic output.  The driver creates ``out_dir`` once, before the pool.
    """
    index_dir = seq_task_artifacts_dir / cfg.split / INDEX_DIRNAME
    labeled_dir = seq_task_artifacts_dir / cfg.split / LABELED_DIRNAME
    data_dir = path_to_data / "data" / cfg.split
    out_dir = seq_tasks_dir / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(p.stem for p in index_dir.glob("*.parquet"))
    _prune_stale_outputs(out_dir, labeled_dir, set(shards))

    n_workers = resolve_workers(cfg.get("max_workers"))
    logger.info("Stage 4': labeling %s shard(s) across %s worker(s).", f"{len(shards):,}", f"{n_workers:,}")
    _label_sequence_shards(shards, index_dir, data_dir, out_dir, bool(cfg.overwrite), n_workers)

    out_files = sorted(out_dir.glob("*.parquet"))
    written = _validate_sequence_count(out_files, total_sequences)
    logger.info("Pipeline complete: wrote %s query sequences to %s.", f"{written:,}", out_dir)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Execute the 5-stage sequence pipeline for a fully-resolved config (no Hydra side effects).

    Mirrors :func:`~every_query.generate_tasks.sample_tasks.run`: Stages 0-3' run sequentially in
    this driver process and produce the partitioned Stage 3' index; Stage 4' then fans out one
    worker per shard via a ``ProcessPoolExecutor`` sized by
    :func:`~every_query.generate_tasks.sample_tasks.resolve_workers`.  The query, context, and
    sequence-structure axes are seeded independently via
    :func:`~every_query.utils.seeds.derive_seed` so they reproduce separately for a fixed
    ``cfg.seed``.

    Path roots come from ``cfg.data_dir`` / ``cfg.out_dir``; the intermediate-artifacts root has no
    key of its own and is always the ``{name}_artifacts`` sibling of ``out_dir`` (invariant 7).
    """
    path_to_data = _require_path_arg(cfg.get("data_dir"), "data_dir")
    seq_tasks_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    seq_task_artifacts_dir = default_artifacts_dir(seq_tasks_dir)

    # Stage 0: precompute & cache subject prediction_time_indexes and per-subject counts.
    n_subjects = build_prediction_times(
        path_to_data=path_to_data,
        training_task_artifacts_dir=seq_task_artifacts_dir,
        split=cfg.split,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        overwrite=cfg.overwrite,
    )
    logger.info("Stage 0: %s eligible subject(s) for split=%s.", f"{n_subjects:,}", cfg.split)

    num_sequences = int(cfg.num_sequences)

    # Independent RNG streams per axis (invariant 5): "queries" matches the training sampler's draw
    # exactly (parity), "contexts" matches its context draw, and "sequences" carries the
    # structure-only draws that have no counterpart there.
    query_rng = np.random.default_rng(derive_seed(cfg.seed, "queries"))
    context_rng = np.random.default_rng(derive_seed(cfg.seed, "contexts"))
    structure_rng = np.random.default_rng(derive_seed(cfg.seed, "sequences"))

    # Stage 1': sample one variable-length query sequence per context-to-be.
    query_dist = QuerySequenceDistribution.from_config(cfg)
    sequences = query_dist.sample_sequences(num_sequences, query_rng, structure_rng)
    n_queries = sum(len(s) for s in sequences)
    logger.info(
        "Stage 1': sampled %s sequence(s) totaling %s quer%s from a %s-code universe "
        "(%s durations over [%g, %g] days, lengths ~ U{%d..%d}, eos_first_fraction=%g, "
        "duration_mode=%s).",
        f"{num_sequences:,}",
        f"{n_queries:,}",
        "y" if n_queries == 1 else "ies",
        f"{query_dist.query_universe_size:,}",
        query_dist.duration_distribution,
        query_dist.min_duration,
        query_dist.max_duration,
        query_dist.min_queries,
        query_dist.max_queries,
        query_dist.eos_first_fraction,
        query_dist.duration_mode,
    )

    # Stage 2: one patient context per sequence.  Re-sort by subject_id so the subject_idx ->
    # subject_id mapping is independent of parquet round-trip order (see sample_tasks.run).
    prediction_time_counts_df = pl.read_parquet(
        prediction_time_counts_path(seq_task_artifacts_dir, cfg.split)
    ).sort("subject_id")
    contexts = sample_patient_contexts(
        prediction_time_counts=prediction_time_counts_df,
        n=num_sequences,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        rng=context_rng,
    )
    logger.info("Stage 2: sampled %s patient context(s).", f"{contexts.height:,}")

    # Stage 3': zip sequences with contexts, resolve prediction times, write the per-shard index.
    n_index_shards = build_sequence_index(
        sequences=sequences,
        contexts=contexts,
        training_task_artifacts_dir=seq_task_artifacts_dir,
        split=cfg.split,
    )
    logger.info(
        "Stage 3': wrote partitioned index for split=%s (%s query rows across %s shard(s)).",
        cfg.split,
        f"{n_queries:,}",
        f"{n_index_shards:,}",
    )

    # Stage 4': fan one labeling worker out per shard (one Stage 3' index partition each).
    label_sequence_shards(cfg, path_to_data, seq_tasks_dir, seq_task_artifacts_dir, num_sequences)


# ---------------------------------------------------------------------------
# Supplied-cohort path (contexts are given, not sampled)
# ---------------------------------------------------------------------------


def read_supplied_contexts(contexts_path: str | Path, n_replicates: int) -> pl.DataFrame:
    """Read a supplied ``(subject_id, prediction_time)`` index parquet, repeated ``n_replicates`` times.

    Extra columns are dropped; ``prediction_time`` is cast to ``Datetime("us")`` for the same reason
    ``_read_event_shard`` casts event times (the ``+1us`` strict-``>`` shift in
    :func:`label_binary_occurrence` rounds to zero at millisecond precision).

    Each output row becomes one independently-sampled query sequence, so ``n_replicates=N`` yields
    ``N`` sequences per supplied row.  Replicate ``r`` of supplied row ``i`` is at row ``r*W + i``.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1 (got {n_replicates})")
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name

    df = pl.read_parquet(contexts_path)
    missing = {sid, pt} - set(df.columns)
    if missing:
        raise ValueError(f"{contexts_path} is missing required column(s) {sorted(missing)}")
    df = df.select(pl.col(sid).cast(pl.Int64), pl.col(pt).cast(pl.Datetime("us")))
    return pl.concat([df] * n_replicates) if n_replicates > 1 else df


def read_events_for_subjects(split_dir: Path, subjects: pl.Series) -> pl.DataFrame:
    """Gather the events of ``subjects`` from every shard under ``split_dir``.

    A supplied cohort is an arbitrary subject set rather than one shard, so its events are spread
    across shards.  The per-shard prefilter keeps peak memory at the cohort's own events - the
    unfiltered union of a real split is tens of millions of rows.
    """
    wanted = subjects.unique()
    frames = [
        ev
        for fp in sorted(split_dir.rglob("*.parquet"))
        if (ev := _read_event_shard(fp).filter(pl.col(DataSchema.subject_id_name).is_in(wanted))).height
    ]
    if not frames:
        raise ValueError(f"No events under {split_dir} for any of the {wanted.len()} supplied subjects.")
    events_df = pl.concat(frames, how="vertical").sort([DataSchema.subject_id_name, DataSchema.time_name])
    n_found = events_df[DataSchema.subject_id_name].n_unique()
    if n_found < wanted.len():
        # Silent here would mean all-False answers now and a silent row-drop in EQ_predict_sequences
        # later (its schema_df semi-join drops subjects absent from the split without erroring).
        raise ValueError(
            f"{wanted.len() - n_found} of {wanted.len()} supplied subjects have no events under "
            f"{split_dir}; check that `split` matches the cohort."
        )
    return events_df


def run_worker(
    data_dir: Path,
    out_dir: Path,
    query_codes: list[str],
    split: str,
    contexts_path: str | Path,
    task_shard: int = 0,
    seed: int = 1,
    min_queries: int = 1,
    max_queries: int = 5,
    duration_min: float = 1,
    duration_max: float = 731,
    duration_distribution: str = "log-uniform",
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    n_replicates: int = 1,
    overwrite: bool = False,
) -> Path | None:
    """Label a **supplied** ``(subject_id, prediction_time)`` cohort with sampled query sequences.

    The non-sampled entry path: contexts come from ``contexts_path`` (repeated ``n_replicates``
    times, one sampled sequence each) rather than from Stages 0/2, so no prediction-time map is
    built and no ``_artifacts`` tree is written.  Events are gathered across every shard of
    ``split`` because a supplied cohort is an arbitrary subject set.  The output is a single parquet
    named after the cohort file stem: ``{out_dir}/{split}/{stem}__{task_shard:04d}.parquet``.

    For the sampled path - the 5-stage pipeline over a whole split - call :func:`run` instead.

    Returns:
        The written path, or ``None`` if the output already existed and ``overwrite`` is false.
    """
    stem = Path(contexts_path).stem
    labels_fp = Path(out_dir) / split / f"{stem}__{task_shard:04d}.parquet"
    if labels_fp.exists() and not overwrite:
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    contexts = read_supplied_contexts(contexts_path, n_replicates)
    subjects = contexts[TaskQuerySchema.subject_id_name]
    events_df = read_events_for_subjects(Path(data_dir) / "data" / split, subjects)
    logger.info(
        "Loaded %d contexts from %s (x%d) and %d events from %s/data/%s",
        contexts.height,
        contexts_path,
        n_replicates,
        events_df.height,
        data_dir,
        split,
    )

    index_df = build_sequence_index_df(
        contexts=contexts,
        query_codes=query_codes,
        min_queries=min_queries,
        max_queries=max_queries,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=derive_seed(seed, stem, task_shard),
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        duration_distribution=duration_distribution,
    )

    labeled = label_query_sequences(index_df, events_df)

    aligned = QuerySeqSchema.align(labeled.to_arrow())
    labels_fp.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(pl.from_arrow(aligned), labels_fp)
    logger.info("Wrote %d labeled query sequences to %s", labeled.height, labels_fp)
    return labels_fp


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``EQ_generate_query_sequences``); dispatches on ``contexts_path``.

    ``contexts_path`` unset (the default) runs the sampled 5-stage pipeline over the whole split via
    :func:`run`.  ``contexts_path=<parquet>`` labels that supplied cohort via :func:`run_worker`.

    Path roots are required args with no ``.env``/env-var fallback (removed upstream in #235):
    ``data_dir``, ``out_dir``, and ``query_codes`` are mandatory, resolved through the same
    :func:`~every_query.generate_tasks.sample_tasks._require_path_arg` guard the training sampler
    uses so an unexported ``$VAR`` fails with one clear message instead of a literal ``None`` path.
    """
    if cfg.get("contexts_path") is None:
        run(cfg)
        return

    data_dir = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    query_codes = read_query_codes(cfg.get("query_codes"))

    run_worker(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=query_codes,
        split=str(cfg.split),
        contexts_path=cfg.contexts_path,
        task_shard=int(cfg.task_shard),
        seed=int(cfg.seed),
        min_queries=int(cfg.min_queries),
        max_queries=int(cfg.max_queries),
        duration_min=float(cfg.duration_min),
        duration_max=float(cfg.duration_max),
        duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
        eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
        duration_mode=str(cfg.get("duration_mode", "random")),
        n_replicates=int(cfg.get("n_replicates", 1)),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
