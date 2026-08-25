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

Labeling a **supplied** ``(subject_id, prediction_time)`` cohort is the eval sampler's job
(:mod:`~every_query.generate_tasks.sample_evaluation_query_sequences`, ``contexts_path=``); this
module only samples.
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
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from every_query.data import query_vocab
from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.data.seq_dataset import EOS_CODE, EVENT_BOUND_DURATION_SENTINEL
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

# Per-query boundary code in the flat index frame; becomes the ``bound_events`` list column.
BOUND_COL = "bound_event"

# Documented starting point for ``bound_events``, from the upstream experiment's MIMIC-IV
# vocabulary.  NOT a silent fallback: ``eventbound_fraction > 0`` requires the caller to pass
# ``bound_events`` explicitly, because these strings are cohort-specific and a boundary code that
# does not exist in the vocabulary would otherwise be dropped, silently reducing the pool.
EXAMPLE_BOUNDARY_CODES = (
    "TIMELINE//END",
    "MEDS_DEATH",
    "HOSPITAL_DISCHARGE//HOME",
    "HOSPITAL_DISCHARGE//UNK",
    "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM",
    "ICU_DISCHARGE//Medical Intensive Care Unit (MICU)",
    "ICU_ADMISSION//Medical Intensive Care Unit (MICU)",
)


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
            # When an ontology is configured, ancestor nodes are mixed into the *universe* the
            # codes are drawn uniformly from, so the draw itself — and its parity with
            # sample_tasks — is untouched.  See build_query_universe.
            query_codes=build_query_universe(
                read_query_codes(cfg.query_codes),
                ontology_dir=cfg.get("ontology_dir"),
                ancestor_fraction=float(cfg.get("ancestor_fraction", 0.0) or 0.0),
                seed=int(cfg.get("seed", 0)),
            ),
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

    ``bound_event`` rides along when present, so an event-bounded draw survives the gather.
    """
    attached = [
        per_query[CTX_ID_COL],
        per_query[POSITION_COL],
        per_query[TaskQuerySchema.query_name],
        per_query[TaskQuerySchema.duration_days_name],
    ]
    if BOUND_COL in per_query.columns:
        attached.append(per_query[BOUND_COL])
    return contexts[per_query[CTX_ID_COL].to_numpy()].with_columns(*attached)


# Ancestors of the TIMELINE//* namespace are tautological as queries ("did any timeline event
# occur"), so they teach nothing and would inflate a macro ancestor AUROC with free positives.
_TAUTOLOGICAL_ANCESTOR_PREFIXES = ("TIMELINE",)


def build_query_universe(
    query_codes: Sequence[str],
    ontology_dir: str | Path | None = None,
    ancestor_fraction: float = 0.0,
    seed: int = 0,
) -> list[str]:
    """Mix ontology ancestor nodes into the query universe, by multiplicity.

    Returns ``query_codes`` unchanged when the feature is off.  Otherwise returns a list in which
    **every leaf code appears exactly once** and ancestor nodes are added often enough that they
    make up ``ancestor_fraction`` of the list — so uniform sampling from it yields that mixture.

    Two properties matter here and neither is negotiable:

    - *No leaf is ever dropped.*  The sampler draws codes uniformly from this list, so a code
      missing from it is a code the model never sees asked about.  An earlier version built the
      universe by sampling a fixed number of slots, which silently lost a long tail of the
      vocabulary — and, on a real cohort, lost ``TIMELINE//END`` itself, the code this model's
      entire censoring mechanism runs through.
    - *The draw is not perturbed.*  Reshaping the universe rather than biasing the draw leaves
      the code/duration RNG stream untouched, which is what keeps parity with ``sample_tasks``.

    The cost is that a leaf's marginal probability is no longer exactly ``1/|vocab|`` — which is
    the point, since the universe is the thing being varied.

    Examples:
        Off by default, and off without an ontology:

        >>> build_query_universe(["A", "B"]) == ["A", "B"]
        True
    """
    if not ontology_dir or ancestor_fraction <= 0.0:
        return list(query_codes)
    if not 0.0 <= ancestor_fraction < 1.0:
        raise ValueError(
            f"ancestor_fraction must be in [0, 1), got {ancestor_fraction}; every leaf code stays "
            "in the universe, so ancestors cannot be 100% of it."
        )

    from every_query.data.ontology import load_nodes

    nodes = load_nodes(ontology_dir)
    ancestors = sorted(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    ancestors = [
        a
        for a in ancestors
        if not a.startswith(_TAUTOLOGICAL_ANCESTOR_PREFIXES) and not query_vocab.has_reserved_chars(a)
    ]
    if not ancestors:
        logger.warning(
            "ancestor_fraction=%g but the ontology at %s contributes no usable ancestor nodes; "
            "the query universe is unchanged.",
            ancestor_fraction,
            ontology_dir,
        )
        return list(query_codes)

    leaves = list(query_codes)
    # f = anc / (leaves + anc)  =>  anc = leaves * f / (1 - f)
    n_anc_slots = round(len(leaves) * ancestor_fraction / (1.0 - ancestor_fraction))
    if n_anc_slots == 0:
        return leaves

    rng = np.random.default_rng(derive_seed(seed, "ancestor_universe"))
    if n_anc_slots <= len(ancestors):
        picks = sorted(rng.choice(len(ancestors), size=n_anc_slots, replace=False).tolist())
        anc_slots = [ancestors[i] for i in picks]
    else:
        # More slots than ancestors: give every ancestor the same base multiplicity, then hand
        # the remainder to a seeded subset, so the target fraction is hit exactly.
        reps, remainder = divmod(n_anc_slots, len(ancestors))
        anc_slots = ancestors * reps
        if remainder:
            extra = sorted(rng.choice(len(ancestors), size=remainder, replace=False).tolist())
            anc_slots += [ancestors[i] for i in extra]

    universe = leaves + anc_slots
    logger.info(
        "Ontology: query universe is %d slot(s) — %d leaf code(s) plus %d ancestor slot(s) drawn "
        "from %d ancestor node(s), an ancestor share of %.1f%%.",
        len(universe),
        len(leaves),
        len(anc_slots),
        len(ancestors),
        100.0 * len(anc_slots) / len(universe),
    )
    return universe


def resolve_bound_events(cfg: DictConfig, query_codes: Sequence[str]) -> list[str]:
    """Resolve and validate the boundary-code pool for event-bounded queries.

    ``bound_events`` is a **required** key once ``eventbound_fraction > 0``: the codes are
    cohort-specific strings (see :data:`EXAMPLE_BOUNDARY_CODES` for the upstream MIMIC-IV set),
    and there is no defensible default that works across cohorts.

    A boundary code absent from ``query_codes`` is a hard error rather than a silent drop.
    Dropping would shrink the pool without saying so — in the limit to nothing — and the
    resulting model would look trained while never having seen the boundary you asked for.

    Returns:
        The validated boundary codes, or ``[]`` when the feature is off.
    """
    fraction = float(cfg.get("eventbound_fraction", 0.0) or 0.0)
    if fraction <= 0.0:
        return []

    raw = cfg.get("bound_events")
    if not raw:
        raise ValueError(
            "eventbound_fraction > 0 requires `bound_events`, either a list of boundary codes or "
            "a path to a YAML list of them, present in this cohort's vocabulary.  These strings "
            f"are cohort-specific; the upstream MIMIC-IV set was {list(EXAMPLE_BOUNDARY_CODES)}."
        )
    # Accept a file path as well as a literal list.  Real boundary codes carry spaces, periods
    # and parentheses ("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", "ICU_DISCHARGE//...(MICU)"),
    # which Hydra's override grammar cannot parse as a bare list on the command line — so
    # requiring a literal list would make the documented codes unusable from the CLI.
    configured = read_query_codes(raw)

    unknown = query_vocab.unknown_codes(configured, set(query_codes))
    if unknown:
        raise ValueError(
            f"{len(unknown)} bound_events code(s) are not in the query vocabulary: {unknown}.  "
            "A boundary code the encoder never saw cannot define a window; fix the spelling or "
            "drop it explicitly rather than letting the pool shrink silently."
        )
    logger.info(
        "Event bounds: %.0f%% of queries will be bounded by one of %d boundary code(s): %s",
        100.0 * fraction,
        len(configured),
        configured,
    )
    return configured


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
    eventbound_fraction: float = 0.0,
    bound_events: Sequence[str] | None = None,
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

        ``eventbound_fraction`` converts that share of queries into event-bounded ones, which
        carry a boundary code and the duration sentinel instead of a horizon:

        >>> idx = build_sequence_index_df(
        ...     ctx, ["A", "B"], 4, 4, 1, 365, 0,
        ...     eventbound_fraction=1.0, bound_events=["DISCHARGE"])
        >>> idx["bound_event"].unique().to_list()
        ['DISCHARGE']
        >>> idx["duration_days"].unique().to_list()
        [-1.0]

        Left at its default the column is absent entirely, so the frame is byte-identical to a
        run from before the feature existed:

        >>> "bound_event" in build_sequence_index_df(ctx, ["A"], 2, 2, 1, 365, 0).columns
        False
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

    bounded = eventbound_fraction > 0.0
    columns = [*INDEX_COLUMNS, BOUND_COL] if bounded else INDEX_COLUMNS

    if contexts.height == 0:
        empty = contexts.head(0).select(
            pl.lit(0, dtype=pl.UInt32).alias(CTX_ID_COL),
            pl.lit(0, dtype=pl.Int64).alias(POSITION_COL),
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            pl.lit("", dtype=pl.Utf8).alias(TaskQuerySchema.query_name),
            pl.lit(0.0, dtype=pl.Float32).alias(TaskQuerySchema.duration_days_name),
        )
        if bounded:
            empty = empty.with_columns(pl.lit(None, dtype=pl.Utf8).alias(BOUND_COL))
        return empty

    sequences = dist.sample_sequences(
        contexts.height,
        np.random.default_rng(derive_seed(seed, "queries")),
        np.random.default_rng(derive_seed(seed, "sequences")),
    )
    per_query = _expand_sequences(sequences)
    if bounded:
        # A fourth, independent seed axis.  Drawing bounds from the "queries" generator would
        # shift every later code/duration draw and break the distribution-parity contract with
        # sample_tasks; drawing from "sequences" would perturb eos_first_fraction/duration_mode.
        per_query = assign_event_bounds(
            per_query,
            list(bound_events or ()),
            eventbound_fraction,
            np.random.default_rng(derive_seed(seed, "bounds")),
        )
    return _attach_queries_to_contexts(contexts, per_query).select(columns)


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
    eventbound_fraction: float = 0.0,
    bound_events: Sequence[str] | None = None,
    seed: int = 0,
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
    bounded = eventbound_fraction > 0.0
    if bounded:
        # Fourth seed axis — see build_sequence_index_df for why it must not share the query or
        # sequence generators.
        per_query = assign_event_bounds(
            per_query,
            list(bound_events or ()),
            eventbound_fraction,
            np.random.default_rng(derive_seed(seed, "bounds")),
        )
    columns = [*INDEX_COLUMNS, BOUND_COL] if bounded else INDEX_COLUMNS
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
            joined.select(columns).sort(CTX_ID_COL, POSITION_COL),
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
    ``(prediction_time, prediction_time + duration_days)`` **and is present in the record**.  The
    interval is **open at both ends**: an occurrence exactly at ``prediction_time`` is outside it,
    and so is one landing exactly on the horizon ``prediction_time + duration_days`` (the
    comparison below is ``<``, not ``<=``).  There
    is no censoring/null: an event we cannot observe (because the record ends first) is ``False``.
    Censoring is captured separately by the ``TIMELINE//END`` query - which, being the last event
    of each record, answers ``True`` exactly when the record ends within the window.

    The strict ``>`` lower bound is enforced by shifting the asof key ``+1us`` (microsecond-precision
    datetimes); the strict ``<`` upper bound is compared directly.  Both ends are open, so the rule
    is symmetric -- neither endpoint instant belongs to the window.  Every other window decider in
    the repo labels this same interval the same way; ``tests/test_window_bounds_contract.py`` is
    where that agreement is pinned.

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
        >>> row["answers"]  # B in (2,12)=yes; A at day1 not >day2 =no; END at day20 not in window =no
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


def _first_occurrence_after(
    left: pl.DataFrame, events_df: pl.DataFrame, code_col: str, out_col: str
) -> pl.DataFrame:
    """Forward-asof each row of ``left`` to the first event whose code matches ``code_col``.

    Attaches ``out_col`` — the timestamp of the first occurrence strictly after the row's
    ``_pts`` key — or null when the code never occurs again.  Rows whose ``code_col`` is null
    (a time-bounded query has no boundary) simply find no match.
    """
    sid = TaskQuerySchema.subject_id_name
    time_col = DataSchema.time_name

    right = (
        events_df.rename({DataSchema.code_name: code_col})
        .select(sid, code_col, time_col)
        .sort(sid, code_col, time_col)
    )
    return (
        left.sort(sid, code_col, "_pts")
        .join_asof(right, by=[sid, code_col], left_on="_pts", right_on=time_col, strategy="forward")
        .rename({time_col: out_col})
    )


def label_with_event_bounds(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label a mixed frame of time-bounded and event-bounded queries.

    A query with a null ``bound_event`` behaves exactly as
    :func:`label_binary_occurrence` — occurrence strictly inside the open interval
    ``(prediction_time, prediction_time + duration_days)``, the horizon instant itself excluded,
    matching :attr:`~every_query.data.schema.QuerySeqSchema.answers`.

    A query with a boundary code is answered over ``(prediction_time, boundary)`` instead,
    where ``boundary`` is the **first occurrence of the boundary code strictly after the
    prediction time**.  The search for that boundary is itself strict at the bottom: an
    occurrence of the boundary code *at* ``prediction_time`` does not close the window.

    **A query code occurring at the exact same instant as the boundary event does NOT count.**
    The upper bound is open here for the same reason it is open on the horizon — one rule, not
    two — and it is what makes an event-bounded query read as *"the query code, strictly after
    the prediction time and strictly before the boundary"*.  The consequence is much larger here
    than on the horizon, though, because MEDS clusters many codes onto a single timestamp: a
    discharge and the codes charted with it routinely share an instant, so this decides real rows
    rather than a measure-zero edge.  It is deliberately not configurable;
    ``test_a_query_at_the_exact_boundary_instant_does_not_count`` in
    ``tests/test_event_bounds_oracle.py`` is the one test that pins it, so making the boundary
    inclusive again is a one-line change with one test to flip.

    One corollary worth stating outright: when ``bound_event`` equals the query code the answer is
    unconditionally ``False``, because the first occurrence after the prediction time is
    simultaneously the boundary and the earliest candidate, and nothing can be strictly before
    itself.

    **When the boundary never occurs again, the window runs to the end of the record**, which
    degenerates the query into "does this code ever occur again".  That is the upstream
    experiment's semantics, kept deliberately so results stay comparable — but it is a real
    trap for boundary codes that do not recur (``MEDS_DEATH`` most obviously), so
    :func:`label_query_sequences` logs how often it fires, per boundary code.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "time": [datetime(2024, 1, 3), datetime(2024, 1, 6), datetime(2024, 1, 9),
        ...              datetime(2024, 1, 20)],
        ...     "code": ["A", "DISCHARGE", "A", "TIMELINE//END"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": [0, 0],
        ...     "_position": [0, 1],
        ...     "subject_id": [1, 1],
        ...     "prediction_time": [datetime(2024, 1, 1)] * 2,
        ...     "query": ["A", "A"],
        ...     "duration_days": [30.0, -1.0],
        ...     "bound_event": [None, "DISCHARGE"],
        ... }).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("duration_days").cast(pl.Float32),
        ...     pl.col("_ctx_id").cast(pl.UInt32))
        >>> row = label_with_event_bounds(idx, events).row(0, named=True)

        Time-bounded: A occurs on the 3rd, inside 30 days.  Event-bounded: A on the 3rd is
        before the DISCHARGE on the 6th, so also True.

        >>> row["answers"]
        [True, True]
        >>> row["bound_events"]
        [None, 'DISCHARGE']

        The bound is strict, and only occurrences *before* it count.  Asking about a code that
        occurs only after the boundary answers False even though it is well inside 30 days:

        >>> idx2 = idx.head(1).with_columns(
        ...     pl.lit("LATE").alias("query"),
        ...     pl.lit(-1.0).cast(pl.Float32).alias("duration_days"),
        ...     pl.lit("DISCHARGE").alias("bound_event"))
        >>> late_events = pl.concat([events, pl.DataFrame({
        ...     "subject_id": [1], "time": [datetime(2024, 1, 8)], "code": ["LATE"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))])
        >>> label_with_event_bounds(idx2, late_events).row(0, named=True)["answers"]
        [False]
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    left = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts"))
    joined = _first_occurrence_after(left, events_df, q, "_q_time")
    joined = _first_occurrence_after(joined, events_df, BOUND_COL, "_b_time")

    is_bounded = pl.col(BOUND_COL).is_not_null()
    # A boundary that never recurs leaves the window open to the end of the record, so any
    # occurrence at all counts.  Expressed as "no upper bound" rather than as a far-future
    # sentinel date, which would need a tz-aware literal to compare against these naive columns.
    window_open = is_bounded & pl.col("_b_time").is_null()
    window_end = (
        pl.when(is_bounded).then(pl.col("_b_time")).otherwise(pl.col(pt) + pl.duration(days=pl.col(d)))
    )
    answer = pl.col("_q_time").is_not_null() & (window_open | (pl.col("_q_time") < window_end))

    flat = joined.with_columns(answer.alias("answer")).sort(CTX_ID_COL, POSITION_COL)
    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(sid).first(),
            pl.col(pt).first(),
            pl.col(q).alias("queries"),
            pl.col(d).alias("durations"),
            pl.col("answer").alias("answers"),
            pl.col(BOUND_COL).alias("bound_events"),
        )
        .drop(CTX_ID_COL)
    )


def log_degenerate_bounds(index_df: pl.DataFrame, events_df: pl.DataFrame) -> dict[str, float]:
    """Report, per boundary code, the fraction of event-bounded rows whose boundary never fires.

    Those rows are silently relabelled "does this code ever occur again" (see
    :func:`label_with_event_bounds`).  A boundary that degenerates most of the time is not
    really being learned as a boundary, and the resulting AUROC would say otherwise — so the
    rate is logged at generation time, when it is still cheap to notice.
    """
    if BOUND_COL not in index_df.columns:
        return {}
    bounded = index_df.filter(pl.col(BOUND_COL).is_not_null())
    if bounded.height == 0:
        return {}

    left = bounded.with_columns(
        (pl.col(TaskQuerySchema.prediction_time_name) + pl.duration(microseconds=1)).alias("_pts")
    )
    joined = _first_occurrence_after(left, events_df, BOUND_COL, "_b_time")
    rates = (
        joined.group_by(BOUND_COL)
        .agg(pl.col("_b_time").is_null().mean().alias("degenerate_rate"), pl.len().alias("n"))
        .sort(BOUND_COL)
    )
    out: dict[str, float] = {}
    for row in rates.iter_rows(named=True):
        out[row[BOUND_COL]] = float(row["degenerate_rate"])
        logger.info(
            "Event bound %r: %.1f%% of %d queries have no boundary occurrence after the "
            "prediction time (window runs to end of record — the query degenerates to "
            "'does this code ever occur again').",
            row[BOUND_COL],
            100.0 * row["degenerate_rate"],
            row["n"],
        )
    return out


def maybe_explode_to_closure(events_df: pl.DataFrame, ontology_dir: str | Path | None):
    """Repeat each event under its ancestor node names, so ancestor queries label normally.

    A no-op without an ontology.  With one, "did any descendant of X occur" becomes an ordinary
    occurrence question about X, so :func:`label_binary_occurrence` needs no ancestor awareness
    at all — it re-sorts its own right frame, so the join-scrambled row order is irrelevant.

    Note this multiplies the event frame by the mean closure size, which raises peak memory in
    each Stage 4' worker.
    """
    if not ontology_dir:
        return events_df
    from every_query.data.ontology import explode_events_to_closure, load_closure_map

    return explode_events_to_closure(events_df, load_closure_map(ontology_dir))


def label_query_sequences(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Stage 4' labeling entry point: answer every query in ``index_df`` against ``events_df``.

    The single seam through which *all* sequence labeling flows — both the sharded training
    path (:func:`label_one_sequence_shard`) and the dense evaluation grid — so a query form
    added here is answered identically in training data and eval grids.  A form that labels one
    way in training and another in evaluation produces a grid that looks fine and silently
    measures the wrong thing.

    Dispatch is on the *frame*, not on a config flag: an index carrying a ``bound_event``
    column is labelled with :func:`label_with_event_bounds`, anything else with
    :func:`label_binary_occurrence`, which stays the regression anchor for plain occurrence
    semantics.  Keying on the data means an eval grid built with bounds is always scored with
    bound-aware labels, even if some caller forgets to pass the flag.

    This must remain a **module-level function**: Stage 4' fans shards out through a
    ``ProcessPoolExecutor`` with ``mp_context="spawn"``, so a closure or a locally-defined
    callable would fail to pickle.

    Args:
        index_df: Flat per-query index frame (one row per query of each sequence).
        events_df: The event stream to answer against.

    Returns:
        One row per sequence, with the ``queries``/``durations``/``answers`` list columns
        (plus ``bound_events`` when the index carried bounds).
    """
    if BOUND_COL in index_df.columns:
        log_degenerate_bounds(index_df, events_df)
        return label_with_event_bounds(index_df, events_df)
    return label_binary_occurrence(index_df, events_df)


def assign_event_bounds(
    index_df: pl.DataFrame,
    boundary_codes: Sequence[str],
    eventbound_fraction: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Convert a fraction of the per-query rows into event-bounded queries.

    Draws i.i.d. **per query**, not per sequence, so a single sequence mixes time- and
    event-bounded asks — which is the point: the model has to read the slot to know which
    kind it is being given.  Chosen rows get a boundary code and the
    ``EVENT_BOUND_DURATION_SENTINEL`` in place of their sampled horizon; the rest keep their
    horizon and a null bound.

    ``rng`` is passed in rather than a seed so the caller can keep this on its own seed axis:
    drawing bounds from the query generator would shift every subsequent code/duration draw
    and break the sampler's distribution-parity contract with ``sample_tasks``.

    Examples:
        >>> import numpy as np
        >>> idx = pl.DataFrame({
        ...     "query": ["A", "B", "C", "D"],
        ...     "duration_days": [30.0, 30.0, 30.0, 30.0],
        ... }).with_columns(pl.col("duration_days").cast(pl.Float32))

        ``0.0`` adds the column but bounds nothing, so the frame is unchanged apart from an
        all-null ``bound_event``:

        >>> out = assign_event_bounds(idx, ["X"], 0.0, np.random.default_rng(0))
        >>> out["bound_event"].to_list(), out["duration_days"].to_list()
        ([None, None, None, None], [30.0, 30.0, 30.0, 30.0])

        ``1.0`` bounds every row and replaces every horizon with the sentinel:

        >>> out = assign_event_bounds(idx, ["X"], 1.0, np.random.default_rng(0))
        >>> out["bound_event"].to_list(), out["duration_days"].to_list()
        (['X', 'X', 'X', 'X'], [-1.0, -1.0, -1.0, -1.0])

        Bounds are drawn from the supplied pool only:

        >>> out = assign_event_bounds(idx, ["X", "Y"], 1.0, np.random.default_rng(3))
        >>> set(out["bound_event"].to_list()) <= {"X", "Y"}
        True
    """
    if not boundary_codes:
        raise ValueError(
            "assign_event_bounds requires at least one boundary code.  Pass `bound_events=[...]` "
            "with codes present in the cohort vocabulary."
        )
    if not 0.0 <= eventbound_fraction <= 1.0:
        raise ValueError(f"eventbound_fraction must be in [0, 1], got {eventbound_fraction}")

    n = index_df.height
    if n == 0:
        return index_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(BOUND_COL))

    chosen = rng.random(n) < eventbound_fraction
    picks = rng.integers(0, len(boundary_codes), size=n)
    # A plain list of ``str | None``, not a numpy object array: polars cannot cast Object to Utf8,
    # which only shows up once the draw is actually mixed (all-None and all-bound both survive it).
    bounds = [boundary_codes[p] if c else None for c, p in zip(chosen, picks, strict=True)]

    return index_df.with_columns(
        pl.Series(BOUND_COL, bounds, dtype=pl.Utf8),
    ).with_columns(
        pl.when(pl.col(BOUND_COL).is_not_null())
        .then(pl.lit(EVENT_BOUND_DURATION_SENTINEL))
        .otherwise(pl.col(TaskQuerySchema.duration_days_name))
        .cast(pl.Float32)
        .alias(TaskQuerySchema.duration_days_name)
    )


def label_one_sequence_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    ontology_dir: str | None = None,
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
    events_df = maybe_explode_to_closure(events_df, ontology_dir)

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
    ontology_dir: str | None = None,
) -> None:
    """Fan one :func:`label_one_sequence_shard` worker out per shard via a spawn-based pool.

    ``"spawn"``, not the Linux default ``"fork"``: by Stage 4' the driver has already run polars
    (which starts a rayon threadpool), and forking while those threads hold locks leaves the child
    with inherited-but-locked mutexes, deadlocking the worker the moment it touches polars (#210).
    """
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
        futs = {
            ex.submit(label_one_sequence_shard, s, index_dir, data_dir, out_dir, overwrite, ontology_dir): s
            for s in shards
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
    _label_sequence_shards(
        shards,
        index_dir,
        data_dir,
        out_dir,
        bool(cfg.overwrite),
        n_workers,
        ontology_dir=cfg.get("ontology_dir"),
    )

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
    # Validate the boundary pool before any sampling work: a bad code should fail in seconds,
    # not after Stage 0 has rebuilt the prediction-time map.
    bound_events = resolve_bound_events(cfg, query_dist.query_codes)
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
        eventbound_fraction=float(cfg.get("eventbound_fraction", 0.0) or 0.0),
        bound_events=bound_events,
        seed=cfg.seed,
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
# Supplied-cohort helpers (used by the eval-grid sampler)
# ---------------------------------------------------------------------------


def read_supplied_contexts(contexts_path: str | Path) -> pl.DataFrame:
    """Read a supplied ``(subject_id, prediction_time)`` index parquet.

    Extra columns are dropped; ``prediction_time`` is cast to ``Datetime("us")`` for the same reason
    ``_read_event_shard`` casts event times (the ``+1us`` strict-``>`` shift in
    :func:`label_binary_occurrence` rounds to zero at millisecond precision).
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name

    df = pl.read_parquet(contexts_path)
    missing = {sid, pt} - set(df.columns)
    if missing:
        raise ValueError(f"{contexts_path} is missing required column(s) {sorted(missing)}")
    return df.select(pl.col(sid).cast(pl.Int64), pl.col(pt).cast(pl.Datetime("us")))


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


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``EQ_generate_query_sequences``): the sampled 5-stage pipeline via :func:`run`.

    Path roots are required args with no ``.env``/env-var fallback (removed upstream in #235):
    ``data_dir``, ``out_dir``, and ``query_codes`` are mandatory, resolved through the same
    :func:`~every_query.generate_tasks.sample_tasks._require_path_arg` guard the training sampler
    uses so an unexported ``$VAR`` fails with one clear message instead of a literal ``None`` path.
    """
    run(cfg)


if __name__ == "__main__":
    main()
