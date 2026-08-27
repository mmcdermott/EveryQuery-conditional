"""Dense-grid evaluation-task generator for conditional query *sequences*.

Sibling of :mod:`~every_query.generate_tasks.sample_query_sequences` (scattered shape: every
context draws its own independent query sequence).  This module produces the **dense** shape:
a fixed set of ``N`` query sequences, each labeled at **every** context of one cohort.  For a
given sequence the only thing varying across its rows is the patient, which is what per-sequence
metrics need — pooling across heterogeneous sequences instead measures cross-query base-rate
separation rather than within-task skill.

Relationship to ``sample_evaluation_tasks``: **same knobs, same semantics, different task axis.**
That module cross-joins its cohort with a ``codes x durations`` grid and emits flat
``TaskQuerySchema`` rows for ``EQ_predict``; this one cross-joins the *same* cohort with ``N``
ordered :class:`SequenceSpec` s and emits :class:`QuerySeqSchema` rows (``queries`` /
``durations`` / ``answers`` list columns) for ``EQ_predict_sequences``.  Everything on the cohort
side is imported from ``sample_evaluation_tasks`` and driven by its knobs and seed axes, so for
the same ``(seed, split, prediction_times_per_subject, min_context_per_subject,
subject_subsample_fraction)`` the two generators score the **identical** ``(subject, time)`` set
— the flat grid and the sequence grid are directly comparable.

Pipeline, one worker per shard of ``{data_dir}/data/{split}/*.parquet`` (all shards, no knob):

    1. Resolve the ``N`` :class:`SequenceSpec` s **once**: read designed ones from
       ``sequences_path``, or draw ``n_sequences`` from the training query distribution
       (:func:`sample_sequence_specs`, seeded on ``(seed, "eval_seq_specs", split)`` alone so the
       same specs come out for any cohort).  Validate every code against the model vocabulary.
    2. Per shard: read its events; take the cohort either from ``contexts_path`` (this shard's
       subjects only) or by sampling it exactly as ``sample_evaluation_tasks`` does —
       :func:`~every_query.generate_tasks.sample_evaluation_tasks.subsample_subject_ids` then
       :func:`~every_query.generate_tasks.sample_evaluation_tasks.sample_prediction_times_per_subject`
       on the same ``("subject_subsample" | "prediction_times", split, shard)`` seed axes.
    3. Cross-join cohort x specs into the flat per-query index frame
       (:func:`build_dense_sequence_index_df`), label with
       :func:`~every_query.generate_tasks.sample_query_sequences.label_query_sequences`, align to
       ``QuerySeqSchema``, write.

Ontology support mirrors the training sampler on both of its halves, because a grid that mirrors
only one measures the wrong thing without ever failing: ``ontology_dir`` extends the query
universe (:func:`~every_query.generate_tasks.sample_query_sequences.build_query_universe`) so
ancestor nodes are drawn - as queries and as boundaries - and accepted as designed-spec codes, and
it explodes the event stream through the closure
(:func:`~every_query.generate_tasks.sample_query_sequences.maybe_expand_to_matching_query_nodes`) so an
ancestor query is labeled by ordinary occurrence.  Without the explosion an ancestor query is
labeled ``False`` at every context — the ancestor's *name* is in no event stream, only its
descendants' are — which is a well-formed parquet of wrong answers, not an error.

Output layout, the same as ``sample_evaluation_tasks``':

    ``{out_dir}/eval/{split}/{shard}.parquet``          the labeled grid, one file per shard
    ``{out_dir}/eval_unique/{split}/{shard}.parquet``   deduped ``(subject_id, prediction_time)``
                                                        (``write_unique_prediction_times``)

``{out_dir}/eval`` is directly consumable as ``EQ_predict_sequences tasks_dir=...`` (MEDS-TorchData
rglobs it, so point it at ``eval/`` — never at ``out_dir`` itself, or the ``eval_unique/`` frames
are read as labels).  Give this generator and ``EQ_generate_evaluation_tasks`` distinct ``out_dir``
roots: both write ``eval/{split}/{shard}.parquet``, in incompatible schemas.

Within one shard file spec identity is recoverable two ways: the ``queries``/``durations`` columns
*are* the spec (group on them), and row order is context-major — row ``i`` is
``(contexts[i // N], specs[i % N])``.

Answers follow the module-wide sequence contract: binary, never null; an unobservable occurrence
(record ends before the window does) is ``False``, and censoring is carried by an explicit
``TIMELINE//END`` query rather than a null answer.  Unlike ``sample_evaluation_tasks`` there is
therefore no censored-row filter here.  If you need censored windows *excluded* from metrics rather
than counted as negatives, that filtering belongs downstream (see
``scripts/eval_occurs_uncensored.py``).
"""

import json
import logging
import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.data.seq_dataset import EVENT_BOUND_DURATION_SENTINEL
from every_query.generate_tasks.sample_evaluation_tasks import (
    _labels_fp,
    _unique_fp,
    sample_prediction_times_per_subject,
    subsample_subject_ids,
)
from every_query.generate_tasks.sample_query_sequences import (
    BOUND_COL,
    CTX_ID_COL,
    POSITION_COL,
    QuerySequenceDistribution,
    build_query_universe,
    label_query_sequences,
    maybe_expand_to_matching_query_nodes,
)
from every_query.generate_tasks.sample_tasks import (
    LABELED_DIRNAME,
    _atomic_write_json,
    _atomic_write_parquet,
    _read_event_shard,
    _require_path_arg,
    _split_shards,
    default_artifacts_dir,
    read_query_codes,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)

SPEC_ID_COL = "_spec_id"

# Spec names become directory names under ``per_spec_dirs``, and YAML keys / parquet ``seq_id``
# values are free-form (MEDS codes contain ``/``).  Anything outside this set is replaced with
# ``_``; collisions after sanitising are rejected rather than silently overwritten.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SequenceSpec:
    """One query sequence to evaluate at every cohort context.

    Attributes:
        name: Identifier, used as the output directory name under ``per_spec_dirs``.
        queries: Ordered MEDS code strings.
        durations: Per-query horizons in days, aligned with ``queries``.

    Examples:
        >>> SequenceSpec("mortality_30d", ("TIMELINE//END", "MEDS_DEATH"), (1.0, 30.0))
        SequenceSpec(name='mortality_30d', queries=('TIMELINE//END', 'MEDS_DEATH'),
                     durations=(1.0, 30.0), bounds=())

        A position may instead be bounded by an *event*: its window runs to the next occurrence
        of that code, so it carries the duration sentinel rather than a horizon.

        >>> spec = SequenceSpec("sepsis_before_discharge", ("SEPSIS",), (-1.0,),
        ...                         ("HOSPITAL_DISCHARGE//HOME",))
        >>> spec.bound_at(0)
        'HOSPITAL_DISCHARGE//HOME'

        A bounded position with a real horizon is a contradiction and is rejected:

        >>> SequenceSpec("bad", ("SEPSIS",), (30.0,), ("DISCHARGE",))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' position 0 is bounded by 'DISCHARGE', so its duration must be
        the -1.0 sentinel, got 30.0

        Ragged or empty specs are rejected at construction:

        >>> SequenceSpec("bad", ("A", "B"), (1.0,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' has 2 queries but 1 duration(s)
        >>> SequenceSpec("bad", (), ())
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' is empty; a sequence needs at least one query
        >>> SequenceSpec("bad", ("A",), (0.0,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' duration 0.0 at position 0 must be a finite number > 0
    """

    name: str
    queries: tuple[str, ...]
    durations: tuple[float, ...]
    bounds: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError(f"spec {self.name!r} is empty; a sequence needs at least one query")
        if len(self.queries) != len(self.durations):
            raise ValueError(
                f"spec {self.name!r} has {len(self.queries)} queries but {len(self.durations)} duration(s)"
            )
        for i, q in enumerate(self.queries):
            if not isinstance(q, str) or not q:
                raise ValueError(f"spec {self.name!r} query at position {i} must be a non-empty string")
        if self.bounds and len(self.bounds) != len(self.queries):
            raise ValueError(
                f"spec {self.name!r} has {len(self.queries)} queries but {len(self.bounds)} bound(s)"
            )
        for i, d in enumerate(self.durations):
            # ``bool`` is an ``int`` subclass, so ``True`` would otherwise become ``1.0`` days.
            if isinstance(d, bool) or not isinstance(d, int | float):
                raise TypeError(
                    f"spec {self.name!r} duration at position {i} must be a number, "
                    f"got {type(d).__name__}: {d!r}"
                )
            # An event-bounded position has no horizon — its window ends at a boundary event —
            # so it carries the negative sentinel instead.  That is the *only* licence for a
            # non-positive duration; an unbounded position must still name a real horizon.
            if self.bound_at(i) is not None:
                if float(d) != EVENT_BOUND_DURATION_SENTINEL:
                    raise ValueError(
                        f"spec {self.name!r} position {i} is bounded by {self.bound_at(i)!r}, so its "
                        f"duration must be the {EVENT_BOUND_DURATION_SENTINEL} sentinel, got {float(d)}"
                    )
                continue
            if not math.isfinite(d) or d <= 0:
                raise ValueError(
                    f"spec {self.name!r} duration {float(d)} at position {i} must be a finite number > 0"
                )

    def bound_at(self, i: int) -> str | None:
        """Boundary code at position ``i``, or ``None`` when that position is time-bounded."""
        return self.bounds[i] if self.bounds else None

    def __len__(self) -> int:
        return len(self.queries)


def _sanitise_names(specs: list[SequenceSpec]) -> list[SequenceSpec]:
    """Make every spec name filesystem-safe, rejecting post-sanitisation collisions."""
    seen: dict[str, str] = {}
    out: list[SequenceSpec] = []
    for spec in specs:
        safe = _SAFE_NAME_RE.sub("_", spec.name).strip("._") or "seq"
        if safe in seen:
            raise ValueError(
                f"spec names {seen[safe]!r} and {spec.name!r} both sanitise to {safe!r}; "
                f"rename one so per-spec output directories stay distinct."
            )
        seen[safe] = spec.name
        out.append(
            spec if safe == spec.name else SequenceSpec(safe, spec.queries, spec.durations, spec.bounds)
        )
    return out


def _specs_from_pairs(name: str, pairs: object) -> SequenceSpec:
    """Build one spec from a ``[[code, duration], ...]`` list."""
    if not isinstance(pairs, Sequence) or isinstance(pairs, str):
        raise ValueError(f"sequence {name!r} must be a list of [code, duration] pairs, got {pairs!r}")
    queries: list[str] = []
    durations: list[float] = []
    bounds: list[str | None] = []
    for i, pair in enumerate(pairs):
        if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) not in (2, 3):
            raise ValueError(
                f"sequence {name!r} entry {i} must be a [code, duration] pair or a "
                f"[code, duration, bound_event] triple, got {pair!r}"
            )
        bound = pair[2] if len(pair) == 3 else None
        if bound is not None and not isinstance(bound, str):
            # Guards the shape `[code, duration, 9]`, which would otherwise be read as a
            # boundary rather than rejected as a malformed [code, duration] pair.
            raise ValueError(
                f"sequence {name!r} entry {i} must be a [code, duration] pair or a "
                f"[code, duration, bound_event] triple with a string bound, got {pair!r}"
            )
        queries.append(pair[0])
        durations.append(pair[1])
        bounds.append(bound)
    return SequenceSpec(
        name=name,
        queries=tuple(queries),
        durations=tuple(durations),
        bounds=tuple(bounds) if any(b is not None for b in bounds) else (),
    )


def read_sequence_specs(path: str | Path) -> list[SequenceSpec]:
    """Read designed query sequences from a YAML/JSON or parquet file.

    YAML/JSON accepts either a mapping of ``name -> [[code, duration], ...]``::

        mortality_30d:
          - [TIMELINE//END, 1]
          - [MEDS_DEATH, 30]

    or a bare list of sequences, in which case names are generated (``seq_0000``, ...)::

        - [[TIMELINE//END, 1], [MEDS_DEATH, 30]]

    A parquet is read as long-format ``(seq_id, position, query, duration_days)``, one row per
    query; ``seq_id`` becomes the spec name and ``position`` fixes the within-sequence order.
    That form is the convenient one when the specs are themselves generated by a script.

    Raises:
        ValueError: If the file is empty, has an unsupported suffix, or any sequence is malformed.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(p)
        # Accept both ``position`` and the internal ``_position`` spelling, so a frame dumped
        # straight out of ``build_dense_sequence_index_df`` can be fed back in.
        pos_col = "position" if "position" in df.columns else POSITION_COL
        required = {"seq_id", pos_col, "query", "duration_days"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing required column(s) {sorted(missing)}")
        grouped = (
            df.select(
                pl.col("seq_id").cast(pl.Utf8),
                pl.col(pos_col).cast(pl.Int64),
                pl.col("query").cast(pl.Utf8),
                pl.col("duration_days").cast(pl.Float64),
            )
            .sort("seq_id", pos_col)
            .group_by("seq_id", maintain_order=True)
            .agg(pl.col("query"), pl.col("duration_days"))
        )
        specs = [
            SequenceSpec(
                name=row["seq_id"],
                queries=tuple(row["query"]),
                durations=tuple(row["duration_days"]),
            )
            for row in grouped.iter_rows(named=True)
        ]
    elif p.suffix in {".yaml", ".yml", ".json"}:
        if p.suffix == ".json":
            data = json.loads(p.read_text())
        else:
            import yaml

            data = yaml.safe_load(p.read_text())
        if isinstance(data, dict) and "sequences" in data:
            data = data["sequences"]
        if isinstance(data, dict):
            specs = [_specs_from_pairs(str(name), pairs) for name, pairs in data.items()]
        elif isinstance(data, list):
            specs = [_specs_from_pairs(f"seq_{i:04d}", pairs) for i, pairs in enumerate(data)]
        else:
            raise ValueError(
                f"{p} must contain a mapping of name -> [[code, duration], ...] or a list of "
                f"such sequences, got {type(data).__name__}"
            )
    else:
        raise ValueError(f"{p} must be a .yaml/.yml/.json or .parquet file (got suffix {p.suffix!r})")

    if not specs:
        raise ValueError(f"{p} contains no sequences")
    return _sanitise_names(specs)


def sample_sequence_specs(
    n_sequences: int,
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
) -> list[SequenceSpec]:
    """Draw ``n_sequences`` specs from the *training* query distribution, once.

    Delegates to
    :class:`~every_query.generate_tasks.sample_query_sequences.QuerySequenceDistribution` on the
    same three seed axes the scattered sampler uses, so the code/duration/bound draw is
    byte-for-byte the one training saw — an evaluation grid drawn from a subtly different
    distribution than training is the kind of drift that shows up as unexplained metric shifts.
    No contexts are involved: these specs are then applied to *every* real context.

    Examples:
        >>> specs = sample_sequence_specs(3, ["A", "B", "TIMELINE//END"], 2, 2, 1, 365, seed=0)
        >>> len(specs)
        3
        >>> [len(s) for s in specs]
        [2, 2, 2]
        >>> all(q in {"A", "B", "TIMELINE//END"} for s in specs for q in s.queries)
        True

        Determinism, and independence from the cohort it will be applied to:

        >>> sample_sequence_specs(3, ["A", "B"], 1, 3, 1, 365, 7) == sample_sequence_specs(
        ...     3, ["A", "B"], 1, 3, 1, 365, 7)
        True

        ``eos_first_fraction=1.0`` forces position 0 to the end-of-timeline query:

        >>> specs = sample_sequence_specs(
        ...     2, ["A", "B", "TIMELINE//END"], 2, 2, 1, 365, 0, eos_first_fraction=1.0)
        >>> {s.queries[0] for s in specs}
        {'TIMELINE//END'}

        ``eventbound_fraction`` converts that share of queries into event-bounded ones, carrying a
        boundary drawn from the same universe and the duration sentinel instead of a horizon:

        >>> (spec,) = sample_sequence_specs(1, ["A", "B"], 4, 4, 1, 365, 0, eventbound_fraction=1.0)
        >>> set(spec.bounds) <= {"A", "B"}, set(spec.durations)
        (True, {-1.0})
    """
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1 (got {n_sequences})")

    dist = QuerySequenceDistribution(
        query_codes=list(query_codes),
        min_duration=float(duration_low),
        max_duration=float(duration_high),
        duration_distribution=duration_distribution,
        min_queries=min_queries,
        max_queries=max_queries,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        eventbound_fraction=eventbound_fraction,
    )
    sequences = dist.sample_sequences(
        n_sequences,
        np.random.default_rng(derive_seed(seed, "queries")),
        np.random.default_rng(derive_seed(seed, "sequences")),
        np.random.default_rng(derive_seed(seed, "bounds")),
    )
    return [
        SequenceSpec(
            name=f"seq_{i:04d}",
            queries=tuple(q.code for q in seq),
            durations=tuple(float(q.duration_days) for q in seq),
            bounds=tuple(q.bound_event for q in seq) if any(q.bound_event for q in seq) else (),
        )
        for i, seq in enumerate(sequences)
    ]


def build_dense_sequence_index_df(
    contexts: pl.DataFrame,
    specs: list[SequenceSpec],
) -> pl.DataFrame:
    """Cross-join every context with every spec into the flat per-query index frame.

    The output is shaped exactly like ``build_sequence_index``'s — ``(_ctx_id, _position,
    subject_id, prediction_time, query, duration_days)`` — so
    :func:`~every_query.generate_tasks.sample_query_sequences.label_binary_occurrence` consumes it
    unchanged.  ``_ctx_id = context_row * len(specs) + spec_index``, so sorting by ``_ctx_id`` puts
    the frame in context-major order: labeled row ``i`` is ``(contexts[i // N], specs[i % N])``.

    Args:
        contexts: ``(subject_id, prediction_time)`` rows; extra columns are ignored.
        specs: The ``N`` sequences to evaluate at every context.

    Returns:
        DataFrame with ``contexts.height * sum(len(s) for s in specs)`` rows.

    Examples:
        >>> from datetime import datetime
        >>> ctx = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        ... }, schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
        >>> specs = [
        ...     SequenceSpec("a", ("TIMELINE//END", "X"), (1.0, 30.0)),
        ...     SequenceSpec("b", ("Y",), (7.0,)),
        ... ]
        >>> idx = build_dense_sequence_index_df(ctx, specs)
        >>> idx.height  # 2 contexts x (2 + 1) queries
        6
        >>> idx.columns
        ['_ctx_id', '_position', 'subject_id', 'prediction_time', 'query', 'duration_days']

        Every context gets every spec, and ``_ctx_id`` is context-major:

        >>> idx.select("_ctx_id", "subject_id", "query").rows()
        [(0, 1, 'TIMELINE//END'), (0, 1, 'X'), (1, 1, 'Y'), (2, 2, 'TIMELINE//END'), (2, 2, 'X'), (3, 2, 'Y')]

        Empty cohort yields an empty frame with the right schema:

        >>> build_dense_sequence_index_df(ctx.head(0), specs).height
        0
    """
    if not specs:
        raise ValueError("specs must be non-empty")

    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    # The bound column appears only when some spec is event-bounded, so a bound-free grid stays
    # byte-identical to one built before the feature existed.
    any_bounds = any(s.bounds for s in specs)

    out_schema = {
        CTX_ID_COL: pl.UInt32,
        POSITION_COL: pl.Int64,
        sid: contexts.schema.get(sid, pl.Int64),
        pt: contexts.schema.get(pt, pl.Datetime("us")),
        q: pl.Utf8,
        d: pl.Float32,
    }
    if any_bounds:
        out_schema[BOUND_COL] = pl.Utf8
    if contexts.height == 0:
        return pl.DataFrame(schema=out_schema)

    n_specs = len(specs)
    n_sequences = contexts.height * n_specs
    # ``_ctx_id`` is UInt32 to match ``build_sequence_index`` (whose ids come from
    # ``with_row_index``); a silently-wrapped id would merge two contexts' sequences into one
    # labeled row, so check rather than trusting the cast.
    if n_sequences > (1 << 32) - 1:
        raise ValueError(
            f"{contexts.height} contexts x {n_specs} specs = {n_sequences} sequences exceeds the "
            f"UInt32 _ctx_id range; shard the cohort across multiple runs."
        )

    spec_cols = {
        SPEC_ID_COL: [i for i, s in enumerate(specs) for _ in s.queries],
        POSITION_COL: [p for s in specs for p in range(len(s))],
        q: [c for s in specs for c in s.queries],
        d: [float(dd) for s in specs for dd in s.durations],
    }
    spec_schema = {SPEC_ID_COL: pl.Int64, POSITION_COL: pl.Int64, q: pl.Utf8, d: pl.Float32}
    if any_bounds:
        spec_cols[BOUND_COL] = [s.bound_at(p) for s in specs for p in range(len(s))]
        spec_schema[BOUND_COL] = pl.Utf8

    spec_frame = pl.DataFrame(spec_cols, schema=spec_schema)

    return (
        contexts.select(sid, pt)
        # Cast to Int64 before the multiply: ``with_row_index`` yields UInt32, and
        # ``row * n_specs`` on UInt32 would wrap before the range check above could help.
        .with_row_index("_row")
        .with_columns(pl.col("_row").cast(pl.Int64))
        .join(spec_frame, how="cross")
        .with_columns((pl.col("_row") * n_specs + pl.col(SPEC_ID_COL)).cast(pl.UInt32).alias(CTX_ID_COL))
        .select(list(out_schema))
        .sort(CTX_ID_COL, POSITION_COL)
    )


def resolve_specs(
    sequences_path: str | Path | None,
    query_codes: list[str],
    n_sequences: int,
    min_queries: int,
    max_queries: int,
    duration_min: float,
    duration_max: float,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    duration_distribution: str = "log-uniform",
    eventbound_fraction: float = 0.0,
) -> list[SequenceSpec]:
    """Read designed specs from ``sequences_path``, or sample them when it is ``None``."""
    if sequences_path is not None:
        specs = read_sequence_specs(sequences_path)
        logger.info("Read %d designed query sequence(s) from %s", len(specs), sequences_path)
        return specs
    specs = sample_sequence_specs(
        n_sequences=n_sequences,
        query_codes=query_codes,
        min_queries=min_queries,
        max_queries=max_queries,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=seed,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        duration_distribution=duration_distribution,
        eventbound_fraction=eventbound_fraction,
    )
    logger.info("Sampled %d query sequence(s) from the training query distribution", len(specs))
    return specs


def model_query_vocab(query_codes: Sequence[str], ontology_dir: str | Path | None) -> set[str]:
    """Every name ``ConditionalQueryPytorchDataset.encode_query`` will accept for this run.

    Built by the same :func:`~every_query.data.ontology.extend_code_map` the dataset uses, so a
    spec that validates here is a spec the dataset can encode — and vice versa.  Note this is
    *wider* than the sampling universe from ``build_query_universe`` (which drops ``TIMELINE``
    ancestors): a designed spec does not sample, it only needs the name to resolve.
    """
    if not ontology_dir:
        return set(query_codes)

    from every_query.data.ontology import extend_code_map

    return set(extend_code_map(dict.fromkeys(query_codes, 0), ontology_dir))


def validate_spec_codes(specs: list[SequenceSpec], vocab: Collection[str]) -> None:
    """Fail fast on spec codes outside the model's query vocabulary.

    ``ConditionalQueryPytorchDataset.encode_query`` raises ``KeyError`` on an unknown code, which
    surfaces deep inside ``collate`` partway through an inference run — long after the minutes of
    labeling and model loading this check precedes.  Hand-written designed specs are exactly where
    a typo'd or wrong-vocabulary MEDS code enters.
    """
    # Boundary events are checked too — a boundary the encoder never saw cannot define a window,
    # and an unchecked one would surface as a KeyError deep inside collate, after the labeling
    # and model load.
    mentioned = [q for s in specs for q in s.queries]
    mentioned += [b for s in specs for b in s.bounds if b is not None]
    unknown = sorted(set(mentioned) - set(vocab))
    if unknown:
        shown = ", ".join(repr(c) for c in unknown[:10])
        more = f" (and {len(unknown) - 10} more)" if len(unknown) > 10 else ""
        raise ValueError(
            f"{len(unknown)} spec query code(s) are absent from the query vocabulary: {shown}{more}. "
            f"Check that `query_codes` points at the same codes.parquet the model was trained on."
        )


# ---------------------------------------------------------------------------
# The cohort
# ---------------------------------------------------------------------------


def read_supplied_contexts(contexts_path: str | Path) -> pl.DataFrame:
    """Read a supplied ``(subject_id, prediction_time)`` cohort parquet, deduplicated.

    Extra columns are dropped; ``prediction_time`` is cast to ``Datetime("us")`` for the same reason
    ``_read_event_shard`` casts event times (the ``+1us`` strict-``>`` shift in
    :func:`label_binary_occurrence` rounds to zero at millisecond precision).  Duplicate rows are
    dropped with a warning: a dense grid asks every sequence about every context once, and a
    duplicated cohort row would silently double-weight that patient in the metrics.

    Raises:
        ValueError: If a required column is missing, or the file has no rows.
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name

    df = pl.read_parquet(contexts_path)
    missing = {sid, pt} - set(df.columns)
    if missing:
        raise ValueError(f"{contexts_path} is missing required column(s) {sorted(missing)}")
    contexts = df.select(pl.col(sid).cast(pl.Int64), pl.col(pt).cast(pl.Datetime("us")))
    unique = contexts.unique(maintain_order=True)
    if unique.height < contexts.height:
        logger.warning(
            "Dropped %d duplicate (subject_id, prediction_time) row(s) from %s; %d unique contexts remain.",
            contexts.height - unique.height,
            contexts_path,
            unique.height,
        )
    if unique.height == 0:
        raise ValueError(f"{contexts_path} has no rows; nothing to evaluate.")
    return unique


def assert_subjects_in_split(data_dir: Path, split: str, shards: list[str], subjects: pl.Series) -> None:
    """Fail fast if any supplied-cohort subject has no shard in ``split``.

    Silent here would mean all-``False`` answers now and a silent row-drop in
    ``EQ_predict_sequences`` later (its schema_df semi-join drops subjects absent from the split
    without erroring).  Only the ``subject_id`` column is scanned, so this is cheap even on a real
    split, and it runs before any shard is labeled rather than after the last one.
    """
    sid = TaskQuerySchema.subject_id_name
    present = pl.concat(
        [
            pl.scan_parquet(data_dir / "data" / split / f"{shard}.parquet").select(pl.col(sid).cast(pl.Int64))
            for shard in shards
        ]
    ).collect()[sid]
    wanted = subjects.unique()
    missing = wanted.filter(~wanted.is_in(present.implode()))
    if missing.len():
        raise ValueError(
            f"{missing.len()} of {subjects.n_unique()} supplied subjects have no events under "
            f"{data_dir / 'data' / split} (e.g. {missing.head(5).to_list()}); "
            f"check that `split` matches the cohort."
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _frame_digest(df: pl.DataFrame) -> str:
    """Serialization-independent digest of a frame's logical rows: ``"{height}:{hash16}"``.

    The same construction :func:`~every_query.generate_tasks.sample_tasks._index_fingerprint` uses
    for Stage 4's index partitions — polars' vectorized ``hash_rows`` summed over rows, combined
    with the row count — so a rewritten-but-identical parquet digests the same.
    """
    total = int(df.hash_rows(seed=0).sum()) if df.height else 0
    return f"{df.height}:{total & 0xFFFFFFFFFFFFFFFF:016x}"


def _ontology_fingerprint(ontology_dir: str | Path | None, query_codes: Sequence[str]) -> str | None:
    """Digest of the two ontology-derived inputs to a grid's labels, or ``None`` with no ontology.

    The output path encodes the shard, never the ontology, so a leaf-only grid and an ancestor grid
    land on the same file and an existence-only skip would silently keep the first one's labels.
    The two things that would make those labels wrong are covered here:

    - the **closure**, which decides whether an ancestor query labels ``True`` at all, and
    - the **query universe**, which is where the ontology's ancestor nodes (and the leaf
      vocabulary) land after
      :func:`~every_query.generate_tasks.sample_query_sequences.build_query_universe`; the
      universe is digested with its slot index, since order steers the draw.

    ``None`` when no ontology is configured, so an output written without a sidecar compares equal
    and is still skipped — the ontology-off path writes no sidecar at all.
    """
    if not ontology_dir:
        return None
    from every_query.data.ontology import load_event_to_query_nodes

    universe = pl.DataFrame(
        {"slot": list(range(len(query_codes))), "code": list(query_codes)},
        schema={"slot": pl.Int64, "code": pl.String},
    )
    return f"{_frame_digest(load_event_to_query_nodes(ontology_dir))}|{_frame_digest(universe)}"


def _provenance_path(out_dir: Path, fp: Path) -> Path:
    """Sidecar recording the ontology a written output was labeled under.

    Mirrors :func:`~every_query.generate_tasks.sample_tasks.labeled_fingerprint_path`: provenance
    lives in the ``{name}_artifacts`` sibling, never in the final-output root, so the output tree
    keeps holding nothing but the parquets ``EQ_predict_sequences`` rglobs (invariant 7).  The
    output's path *below* ``out_dir`` is kept as the sidecar's own, so every shard of every split
    gets its own and none can collide.

    Examples:
        >>> _provenance_path(Path("/x/grid"), Path("/x/grid/eval/held_out/0.parquet"))
        PosixPath('/x/grid_artifacts/_labeled/eval/held_out/0.json')
    """
    rel = fp.relative_to(out_dir).with_suffix(".json")
    return default_artifacts_dir(out_dir) / LABELED_DIRNAME / rel


def _recorded_fingerprint(out_dir: Path, fp: Path) -> str | None:
    """Read back :func:`_ontology_fingerprint` for an existing output; ``None`` if absent/unreadable."""
    try:
        return json.loads(_provenance_path(out_dir, fp).read_text()).get("ontology_fingerprint")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _output_is_current(out_dir: Path, fp: Path, fingerprint: str | None) -> bool:
    """Whether an existing output was labeled under the ontology this run is configured with.

    An unreadable or missing sidecar reads as ``None``, i.e. "no ontology", matching
    :func:`label_one_sequence_shard`'s treatment of a missing fingerprint as stale for every case
    but the one that is genuinely current.
    """
    return fp.exists() and _recorded_fingerprint(out_dir, fp) == fingerprint


def _write(labeled: pl.DataFrame, fp: Path, out_dir: Path, fingerprint: str | None) -> None:
    """Align to ``QuerySeqSchema``, atomically write one output parquet, and record its provenance.

    The ontology sidecar is written *after* the parquet is committed, so a present sidecar always
    describes a present output — the ordering :func:`label_one_sequence_shard` uses for the same
    reason.  With no ontology no sidecar is written and any stale one is removed, which keeps "no
    sidecar" meaning exactly "labeled without an ontology".
    """
    aligned = QuerySeqSchema.align(labeled.to_arrow())
    fp.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(pl.from_arrow(aligned), fp)
    provenance = _provenance_path(out_dir, fp)
    if fingerprint is None:
        provenance.unlink(missing_ok=True)
    else:
        _atomic_write_json({"ontology_fingerprint": fingerprint}, provenance)
    logger.info("Wrote %d labeled query sequences to %s", labeled.height, fp)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def run_worker(
    data_dir: Path,
    out_dir: Path,
    split: str,
    input_shard: str,
    specs: list[SequenceSpec],
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    seed: int,
    overwrite: bool = False,
    subject_subsample_fraction: float | None = None,
    contexts: pl.DataFrame | None = None,
    ontology_dir: str | None = None,
    fingerprint: str | None = None,
    write_unique_prediction_times: bool = True,
    unique_out_dir: Path | None = None,
) -> Path | None:
    """Label ``specs`` at every context of one shard, writing one ``QuerySeqSchema`` parquet.

    The sequence analogue of :func:`~every_query.generate_tasks.sample_evaluation_tasks.run_worker`,
    step for step: same output paths (``out_dir`` is already the ``eval/`` root), same skip rule,
    same event reader, and — when ``contexts`` is ``None`` — the same cohort sampler on the same
    seed axes, so the two workers draw the identical cohort for one ``(seed, split, shard)``.

    Args:
        data_dir: MEDS root; this shard's events are at ``{data_dir}/data/{split}/{input_shard}.parquet``.
        out_dir: The ``eval/`` output root.  Provenance sidecars are keyed on the path below its
            *parent*, so ``out_dir.parent`` is what :func:`_provenance_path` receives.
        split: Split whose shard to label.
        input_shard: Shard stem.
        specs: The ``N`` sequences to evaluate at every context — resolved once by the caller and
            shared across shards.
        prediction_times_per_subject: ``K`` prediction times to sample per subject.  Ignored when
            ``contexts`` is supplied.
        min_context_per_subject: Prior events a subject needs before a time is a candidate.
            Ignored when ``contexts`` is supplied.
        seed: Top-level seed; the cohort axes derive on ``(split, input_shard)`` from it.
        overwrite: Regenerate outputs that already exist.
        subject_subsample_fraction: Optional per-subject hash-threshold subsample.  Ignored when
            ``contexts`` is supplied.
        contexts: A supplied ``(subject_id, prediction_time)`` cohort; only this shard's subjects
            are taken from it.  ``None`` samples the cohort from the shard instead.
        ontology_dir: Directory of ``EQ_build_ontology`` artifacts.  Explodes the event stream
            through the closure before labeling, exactly as the training sampler's Stage 4' does.
            ``None`` is a no-op.
        fingerprint: :func:`_ontology_fingerprint` for this run, recorded beside the output so
            "already exists" means "already labeled under this ontology".
        write_unique_prediction_times: Also write the deduped cohort under ``unique_out_dir``.
        unique_out_dir: The ``eval_unique/`` root.

    Returns:
        The written parquet path, or ``None`` if the output existed and ``overwrite=False``.
    """
    root = out_dir.parent
    labels_fp = _labels_fp(out_dir, split, input_shard)
    unique_fp = (
        _unique_fp(unique_out_dir, split, input_shard)
        if write_unique_prediction_times and unique_out_dir is not None
        else None
    )
    if (
        not overwrite
        and _output_is_current(root, labels_fp, fingerprint)
        and (unique_fp is None or unique_fp.exists())
    ):
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name

    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    if contexts is not None:
        # Subjects are disjoint across shards, so every supplied context lands in exactly one file.
        shard_contexts = contexts.filter(pl.col(sid).is_in(events_df[sid].unique().implode())).sort(sid, pt)
    else:
        if subject_subsample_fraction is not None:
            subj_seed = derive_seed(seed, "subject_subsample", split, input_shard)
            events_df = subsample_subject_ids(events_df, subject_subsample_fraction, subj_seed)
            logger.info(
                "Subsampled to %d events / %d subjects (fraction=%.4f)",
                events_df.height,
                events_df[sid].n_unique(),
                subject_subsample_fraction,
            )
        pt_seed = derive_seed(seed, "prediction_times", split, input_shard)
        shard_contexts = sample_prediction_times_per_subject(
            events_df=events_df,
            k=prediction_times_per_subject,
            min_context_per_subject=min_context_per_subject,
            seed=pt_seed,
        )
    logger.info(
        "Cohort: %d contexts across %d subjects x %d sequences = %d rows to label",
        shard_contexts.height,
        shard_contexts[sid].n_unique() if shard_contexts.height else 0,
        len(specs),
        shard_contexts.height * len(specs),
    )

    # After the cohort draw (the flat sampler samples from raw events too) and once per shard,
    # mirroring the training sampler's Stage 4' worker.
    events_df = maybe_expand_to_matching_query_nodes(events_df, ontology_dir)

    # An empty cohort still writes an empty, well-formed parquet, so downstream sees a complete
    # split even when a sparse shard yields no eligible prediction time.
    index_df = build_dense_sequence_index_df(shard_contexts, specs)
    labeled = label_query_sequences(index_df, events_df)
    _write(labeled, labels_fp, root, fingerprint)

    if unique_fp is not None:
        unique_df = labeled.select(sid, pt).unique().sort(sid, pt)
        _atomic_write_parquet(unique_df, unique_fp)
        logger.info("Wrote %d unique (subject_id, prediction_time) rows to %s", unique_df.height, unique_fp)

    return labels_fp


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_evaluation_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Produce one evaluation-sequences parquet per shard of the chosen split.

    Shards are discovered from ``{data_dir}/data/{split}/*.parquet`` and all of them are processed
    in this one invocation, exactly as ``EQ_generate_evaluation_tasks`` does.

    Usage (``N`` sequences drawn from the training distribution at ``K`` sampled times per
    subject — the zero-argument default beyond the three path roots):

        EQ_generate_evaluation_query_sequences \\
            data_dir=... out_dir=... query_codes=... split=held_out \\
            prediction_times_per_subject=2 n_sequences=64

    Usage (designed sequences on a supplied cohort):

        EQ_generate_evaluation_query_sequences \\
            data_dir=... out_dir=... query_codes=... split=held_out \\
            contexts_path=cohort.parquet sequences_path=tasks.yaml
    """
    data_dir = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    split = str(cfg.split)
    seed = int(cfg.seed)
    ontology_dir = cfg.get("ontology_dir")

    # The same two arguments, read the same way, as ``QuerySequenceDistribution.from_config``: an
    # eval grid drawn from a *narrower* universe than training's is the drift this whole module
    # exists to avoid, and the ancestor half of the universe is no exception.
    query_codes = build_query_universe(read_query_codes(cfg.get("query_codes")), ontology_dir=ontology_dir)

    # The sequence axis, once: the specs are shared by every shard, and are seeded independently of
    # the cohort so the same ``(seed, split)`` yields the same ``N`` sequences for any cohort.
    sequences_path = cfg.get("sequences_path")
    specs = resolve_specs(
        sequences_path=sequences_path,
        query_codes=query_codes,
        n_sequences=int(cfg.n_sequences),
        min_queries=int(cfg.min_queries),
        max_queries=int(cfg.max_queries),
        duration_min=float(cfg.duration_min),
        duration_max=float(cfg.duration_max),
        seed=derive_seed(seed, "eval_seq_specs", split),
        eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
        duration_mode=str(cfg.get("duration_mode", "random")),
        duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
        eventbound_fraction=float(cfg.get("eventbound_fraction", 0.0) or 0.0),
    )
    validate_spec_codes(specs, model_query_vocab(query_codes, ontology_dir))
    # Only *designed* specs get the whole-day check: sampled ones draw continuous float durations
    # by design.  A designed spec's fractional duration labels correctly but is usually a slip.
    if sequences_path is not None:
        non_integer = sorted({float(d) for s in specs for d in s.durations if not float(d).is_integer()})
        if non_integer:
            logger.warning(
                "%d designed spec duration(s) are not whole days (e.g. %s); labeling honours the "
                "fraction — check this is intended.",
                len(non_integer),
                non_integer[:5],
            )
    fingerprint = _ontology_fingerprint(ontology_dir, query_codes)

    # Reject booleans up front: Hydra/OmegaConf parses ``subject_subsample_fraction=true`` as a
    # Python ``True``, which would otherwise become ``1.0`` and silently disable subsampling.
    ssf_raw = cfg.get("subject_subsample_fraction")
    if isinstance(ssf_raw, bool) or (ssf_raw is not None and not isinstance(ssf_raw, int | float)):
        raise TypeError(
            f"cfg.subject_subsample_fraction must be a number in (0, 1] or null, "
            f"got {type(ssf_raw).__name__}: {ssf_raw!r}"
        )
    subject_subsample_fraction = None if ssf_raw is None else float(ssf_raw)

    shards = _split_shards(data_dir, split)

    contexts_path = cfg.get("contexts_path")
    contexts = None
    if contexts_path is not None:
        contexts = read_supplied_contexts(str(contexts_path))
        assert_subjects_in_split(data_dir, split, shards, contexts[TaskQuerySchema.subject_id_name])
        logger.info("Read %d supplied context(s) from %s", contexts.height, contexts_path)

    write_unique_prediction_times = bool(cfg.get("write_unique_prediction_times", True))
    for input_shard in shards:
        run_worker(
            data_dir=data_dir,
            out_dir=Path(out_dir) / "eval",
            split=split,
            input_shard=input_shard,
            specs=specs,
            prediction_times_per_subject=int(cfg.prediction_times_per_subject),
            min_context_per_subject=int(cfg.min_context_per_subject),
            seed=seed,
            overwrite=bool(cfg.get("overwrite", False)),
            subject_subsample_fraction=subject_subsample_fraction,
            contexts=contexts,
            ontology_dir=ontology_dir,
            fingerprint=fingerprint,
            write_unique_prediction_times=write_unique_prediction_times,
            unique_out_dir=Path(out_dir) / "eval_unique" if write_unique_prediction_times else None,
        )


if __name__ == "__main__":
    main()
