"""Dense-grid evaluation-task generator for conditional query *sequences*.

Sibling of :mod:`~every_query.generate_tasks.sample_query_sequences` (scattered shape: every
context draws its own independent query sequence).  This module produces the **dense** shape:
a fixed set of ``N`` query sequences, each labeled at **every** context of one cohort.  For a
given sequence the only thing varying across its rows is the patient, which is what per-sequence
metrics need — pooling across heterogeneous sequences instead measures cross-query base-rate
separation rather than within-task skill.

Relationship to ``sample_evaluation_tasks``: same *motivation* (dense grid for evaluation),
different *row shape*.  ``sample_evaluation_tasks`` emits flat ``TaskQuerySchema`` rows — one
scalar ``boolean_value`` per ``(subject, time, code, duration)`` — for the single-query model's
``EQ_predict`` → ``EQ_evaluate`` path.  This module emits :class:`QuerySeqSchema` rows with
``queries``/``durations``/``answers`` list columns for ``EQ_predict_sequences``.  A dense grid of
*sequences* is a cross product of contexts with ordered ``(code, duration)`` **tuples**, not a
``codes x durations`` cross join, so the two share no index-building code.

Pipeline (every stage but the grid build is imported from ``sample_query_sequences``):

    1. Resolve the cohort — either read it from ``contexts_path`` (a parquet with at least
       ``(subject_id, prediction_time)``), or (when that is null) sample ``n_contexts`` of them
       from ``split`` itself via upstream Stages 0 + 2 (:func:`sample_grid_contexts`).  Duplicate
       contexts are dropped either way: a dense grid asks each sequence about each context once.
    2. Resolve the ``N`` :class:`SequenceSpec` s: read designed ones from ``sequences_path``, or
       (when that is null) draw ``n_sequences`` once from the same
       ``uniform codes x log-uniform durations`` distribution the training sampler uses.
    3. Cross-join cohort x specs into the flat per-query index frame
       (:func:`build_dense_sequence_index_df`).
    4. Label with
       :func:`~every_query.generate_tasks.sample_query_sequences.label_binary_occurrence`,
       align to ``QuerySeqSchema``, write.

Both axes carry a tag into the combined file's name: ``contexts_tag`` is the cohort file's stem or
``sampled{n_contexts}ctx``, ``specs_tag`` the spec file's stem or ``sampled{N}``.

Because the cohort spans arbitrary subjects rather than one shard, events are gathered across
every shard of ``split`` via ``read_events_for_subjects`` — which raises if any supplied subject
is absent from the split, rather than silently labeling them all-``False``.  That holds on the
sampled path too: it stays a single driver-side pass rather than a per-shard fan-out, because the
combined output's defining contract is one context-major file (row ``i`` is
``(contexts[i // N], specs[i % N])``), which sharding the output would break — and because
``n_contexts`` bounds the cohort to the same size the supplied path already handles.

Output layout, either

    ``{out_dir}/{split}/{contexts_tag}__{specs_tag}.parquet``  (default, one combined file), or
    ``{out_dir}/{spec_name}/{split}/tasks.parquet``            (``per_spec_dirs=true``),

both directly consumable as ``EQ_predict_sequences tasks_dir=...`` — MEDS-TorchData rglobs the
task-labels dir, so per-spec dirs let you score (and compute metrics for) one designed task at a
time, while the combined file needs a single inference pass.

In the combined file, spec identity is recoverable two ways: the ``queries``/``durations`` columns
*are* the spec (group on them), and row order is context-major — row ``i`` is
``(contexts[i // N], specs[i % N])``.

Answers follow the module-wide sequence contract: binary, never null; an unobservable occurrence
(record ends before the window does) is ``False``, and censoring is carried by an explicit
``TIMELINE//END`` query rather than a null answer.  If you need censored windows *excluded* from
metrics rather than counted as negatives, that filtering belongs downstream (see
``scripts/eval_occurs_uncensored.py``).
"""

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    build_sequence_index_df,
    label_binary_occurrence,
    read_events_for_subjects,
    read_supplied_contexts,
    resolve_prediction_times,
)
from every_query.generate_tasks.sample_tasks import (
    _atomic_write_parquet,
    _require_path_arg,
    build_prediction_times,
    default_artifacts_dir,
    prediction_time_counts_path,
    read_query_codes,
    sample_patient_contexts,
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
        SequenceSpec(name='mortality_30d', queries=('TIMELINE//END', 'MEDS_DEATH'), durations=(1.0, 30.0))

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
        for i, d in enumerate(self.durations):
            # ``bool`` is an ``int`` subclass, so ``True`` would otherwise become ``1.0`` days.
            if isinstance(d, bool) or not isinstance(d, int | float):
                raise TypeError(
                    f"spec {self.name!r} duration at position {i} must be a number, "
                    f"got {type(d).__name__}: {d!r}"
                )
            if not math.isfinite(d) or d <= 0:
                raise ValueError(
                    f"spec {self.name!r} duration {float(d)} at position {i} must be a finite number > 0"
                )

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
        out.append(spec if safe == spec.name else SequenceSpec(safe, spec.queries, spec.durations))
    return out


def _specs_from_pairs(name: str, pairs: object) -> SequenceSpec:
    """Build one spec from a ``[[code, duration], ...]`` list."""
    if not isinstance(pairs, Sequence) or isinstance(pairs, str):
        raise ValueError(f"sequence {name!r} must be a list of [code, duration] pairs, got {pairs!r}")
    queries: list[str] = []
    durations: list[float] = []
    for i, pair in enumerate(pairs):
        if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"sequence {name!r} entry {i} must be a [code, duration] pair, got {pair!r}")
        queries.append(pair[0])
        durations.append(pair[1])
    return SequenceSpec(name=name, queries=tuple(queries), durations=tuple(durations))


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
) -> list[SequenceSpec]:
    """Draw ``n_sequences`` specs from the *training* query distribution, once.

    Delegates to :func:`~every_query.generate_tasks.sample_query_sequences.build_sequence_index_df`
    against a dummy context frame so the code/duration distribution is byte-for-byte the one the
    scattered sampler uses — an evaluation grid drawn from a subtly different distribution than
    training is the kind of drift that shows up as unexplained metric shifts.  The contexts are
    dummies precisely because these specs are then applied to *every* real context.

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
    """
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1 (got {n_sequences})")

    # The context values are irrelevant (only the row *count* drives how many sequences are
    # drawn), so the prediction times are the epoch, built by cast rather than by a naive
    # ``datetime`` literal.
    dummy = pl.DataFrame(
        {
            TaskQuerySchema.subject_id_name: [0] * n_sequences,
            TaskQuerySchema.prediction_time_name: [0] * n_sequences,
        },
        schema={
            TaskQuerySchema.subject_id_name: pl.Int64,
            TaskQuerySchema.prediction_time_name: pl.Int64,
        },
    ).with_columns(pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Datetime("us")))
    index_df = build_sequence_index_df(
        contexts=dummy,
        query_codes=query_codes,
        min_queries=min_queries,
        max_queries=max_queries,
        duration_low=duration_low,
        duration_high=duration_high,
        seed=seed,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        duration_distribution=duration_distribution,
    )
    grouped = index_df.group_by(CTX_ID_COL, maintain_order=True).agg(
        pl.col(TaskQuerySchema.query_name),
        pl.col(TaskQuerySchema.duration_days_name),
    )
    return [
        SequenceSpec(
            name=f"seq_{i:04d}",
            queries=tuple(row[TaskQuerySchema.query_name]),
            durations=tuple(float(d) for d in row[TaskQuerySchema.duration_days_name]),
        )
        for i, row in enumerate(grouped.iter_rows(named=True))
    ]


def build_dense_sequence_index_df(
    contexts: pl.DataFrame,
    specs: list[SequenceSpec],
) -> pl.DataFrame:
    """Cross-join every context with every spec into the flat per-query index frame.

    The output is shaped exactly like ``build_sequence_index_df``'s — ``(_ctx_id, _position,
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

    out_schema = {
        CTX_ID_COL: pl.UInt32,
        POSITION_COL: pl.Int64,
        sid: contexts.schema.get(sid, pl.Int64),
        pt: contexts.schema.get(pt, pl.Datetime("us")),
        q: pl.Utf8,
        d: pl.Float32,
    }
    if contexts.height == 0:
        return pl.DataFrame(schema=out_schema)

    n_specs = len(specs)
    n_sequences = contexts.height * n_specs
    # ``_ctx_id`` is UInt32 to match ``build_sequence_index_df`` (whose ids come from
    # ``with_row_index``); a silently-wrapped id would merge two contexts' sequences into one
    # labeled row, so check rather than trusting the cast.
    if n_sequences > (1 << 32) - 1:
        raise ValueError(
            f"{contexts.height} contexts x {n_specs} specs = {n_sequences} sequences exceeds the "
            f"UInt32 _ctx_id range; shard the cohort across multiple runs."
        )

    spec_frame = pl.DataFrame(
        {
            SPEC_ID_COL: [i for i, s in enumerate(specs) for _ in s.queries],
            POSITION_COL: [p for s in specs for p in range(len(s))],
            q: [c for s in specs for c in s.queries],
            d: [float(dd) for s in specs for dd in s.durations],
        },
        schema={SPEC_ID_COL: pl.Int64, POSITION_COL: pl.Int64, q: pl.Utf8, d: pl.Float32},
    )

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
    )
    logger.info("Sampled %d query sequence(s) from the training query distribution", len(specs))
    return specs


def validate_spec_codes(specs: list[SequenceSpec], query_codes: list[str]) -> None:
    """Fail fast on spec codes outside the model's query vocabulary.

    ``ConditionalQueryPytorchDataset.encode_query`` raises ``KeyError`` on an unknown code, which
    surfaces deep inside ``collate`` partway through an inference run — long after the minutes of
    labeling and model loading this check precedes.  Hand-written designed specs are exactly where
    a typo'd or wrong-vocabulary MEDS code enters.
    """
    vocab = set(query_codes)
    unknown = sorted({q for s in specs for q in s.queries if q not in vocab})
    if unknown:
        shown = ", ".join(repr(c) for c in unknown[:10])
        more = f" (and {len(unknown) - 10} more)" if len(unknown) > 10 else ""
        raise ValueError(
            f"{len(unknown)} spec query code(s) are absent from the query vocabulary: {shown}{more}. "
            f"Check that `query_codes` points at the same codes.parquet the model was trained on."
        )


# ---------------------------------------------------------------------------
# The cohort, source 2 of 2: sampled from the split via upstream Stages 0 + 2
# ---------------------------------------------------------------------------


def sample_grid_contexts(
    data_dir: Path,
    artifacts_dir: Path,
    split: str,
    n_contexts: int,
    min_prediction_times_per_subject: int,
    seed: int,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Draw the grid's cohort from ``split`` itself, for when no ``contexts_path`` is supplied.

    The sampled counterpart to
    :func:`~every_query.generate_tasks.sample_query_sequences.read_supplied_contexts`, assembled
    entirely from upstream pieces so a sampled evaluation grid is scored at contexts from the
    *same* distribution the training sampler draws from (``sample_query_sequences.run`` Stages 0
    and 2, seeded on the same ``"contexts"`` axis):

    1. :func:`~every_query.generate_tasks.sample_tasks.build_prediction_times` (Stage 0) writes /
       reuses the ``_prediction_times`` map and ``_prediction_time_counts`` summary under
       ``artifacts_dir`` — the ``{name}_artifacts`` sibling of ``out_dir`` (invariant 7).
    2. :func:`~every_query.generate_tasks.sample_tasks.sample_patient_contexts` (Stage 2) draws
       ``n_contexts`` ``(subject_id, shard, prediction_time_index)`` rows.
    3. :func:`~every_query.generate_tasks.sample_query_sequences.resolve_prediction_times` turns
       each shard's ``prediction_time_index`` *ranks* into real timestamps, one shard at a time --
       ``build_dense_sequence_index_df`` and ``label_binary_occurrence`` both need a datetime, and
       the per-shard call keeps that function's dtype guard and raise-on-null-join intact.

    Stage 2 draws subjects **with replacement**, so the raw draw repeats contexts; a dense grid asks
    each sequence about each context exactly once, so the draw is deduplicated here.  ``n_contexts``
    is therefore a ceiling on the grid's width rather than an exact count — the same interpretation
    the supplied path gives a cohort file that contains duplicate rows.

    Args:
        data_dir: MEDS root; the split's event shards live beneath it.
        artifacts_dir: Intermediate-artifacts root (Stage 0 writes here, step 3 reads it back).
        split: Split to sample from.
        n_contexts: Number of contexts to draw, before deduplication.
        min_prediction_times_per_subject: Stage 0 eligibility bound and Stage 2 draw floor.
        seed: **Already-derived** seed for the context axis — pass
            ``derive_seed(seed, "contexts")`` so the draw shares the training sampler's axis.
        overwrite: Forwarded to Stage 0; forces a rebuild of a cached prediction-time map.

    Returns:
        Deduplicated ``(subject_id, prediction_time)`` rows sorted by both, shaped exactly like
        ``read_supplied_contexts``' output so nothing downstream can tell the two sources apart.

    Raises:
        ValueError: If ``n_contexts < 1``, or a drawn context fails to resolve a timestamp.
    """
    if n_contexts < 1:
        raise ValueError(f"n_contexts must be >= 1 (got {n_contexts})")

    n_subjects = build_prediction_times(
        path_to_data=data_dir,
        training_task_artifacts_dir=artifacts_dir,
        split=split,
        min_prediction_times_per_subject=min_prediction_times_per_subject,
        overwrite=overwrite,
    )
    logger.info("Stage 0: %s eligible subject(s) for split=%s.", f"{n_subjects:,}", split)

    # Re-sort by subject_id so the subject_idx -> subject_id mapping is independent of parquet
    # round-trip order (the same reason ``sample_query_sequences.run`` re-sorts).
    counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, split)).sort("subject_id")
    drawn = sample_patient_contexts(
        prediction_time_counts=counts,
        n=n_contexts,
        min_prediction_times_per_subject=min_prediction_times_per_subject,
        rng=np.random.default_rng(seed),
    )
    unique_ctx = drawn.unique(maintain_order=True)
    if unique_ctx.height < drawn.height:
        logger.info(
            "Stage 2: drew %s context(s) with replacement; %s unique remain after dedup.",
            f"{drawn.height:,}",
            f"{unique_ctx.height:,}",
        )

    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    # ``sort("shard")`` + ``maintain_order=True`` make the shard iteration order deterministic; the
    # final ``sort(sid, pt)`` then makes the cohort order independent of it entirely.
    resolved = pl.concat(
        [
            resolve_prediction_times(group, artifacts_dir, split, str(shard_key[0]))
            for shard_key, group in unique_ctx.sort("shard").group_by("shard", maintain_order=True)
        ]
    )
    return (
        resolved.select(pl.col(sid).cast(pl.Int64), pl.col(pt).cast(pl.Datetime("us")))
        .unique(maintain_order=True)
        .sort(sid, pt)
    )


def _spec_fps(
    out_dir: Path,
    split: str,
    stem: str,
    specs: list[SequenceSpec],
    per_spec_dirs: bool,
) -> list[Path]:
    """Resolve the output parquet path(s) for one run."""
    if per_spec_dirs:
        return [out_dir / s.name / split / "tasks.parquet" for s in specs]
    return [out_dir / split / f"{stem}.parquet"]


def run_worker(
    data_dir: Path,
    out_dir: Path,
    split: str,
    query_codes: list[str],
    contexts_path: str | Path | None = None,
    sequences_path: str | Path | None = None,
    n_sequences: int = 64,
    # Fixed length 3, inside training's 1..5 range: one length per grid keeps per-position
    # comparisons clean without going off-distribution.
    min_queries: int = 3,
    max_queries: int = 3,
    # The duration bounds mirror ``sample_query_sequences``' defaults so horizons come from the
    # distribution the model was pretrained on.  Keep in sync with
    # ``configs/sample_query_sequences_config.yaml``.
    duration_min: float = 1,
    duration_max: float = 731,
    duration_distribution: str = "log-uniform",
    seed: int = 1,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    n_contexts: int = 512,
    min_prediction_times_per_subject: int = 50,
    per_spec_dirs: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Label ``N`` shared query sequences across every context of a cohort.

    Both axes of the grid are resolved the same way — supply a path, or leave it ``None`` and it is
    sampled:

    ===================  ====================  =============================================
    axis                 supplied              sampled (path is ``None``)
    ===================  ====================  =============================================
    the ``N`` sequences  ``sequences_path``    ``n_sequences`` from the training distribution
    the cohort           ``contexts_path``     ``n_contexts`` via :func:`sample_grid_contexts`
    ===================  ====================  =============================================

    A supplied cohort is read verbatim (every one of its subjects must be present in ``split``); a
    sampled one comes from upstream Stages 0 + 2 over ``split`` itself, whose intermediates land in
    the ``{name}_artifacts`` sibling of ``out_dir`` (invariant 7).  Either way the cohort reaches
    the grid build as plain ``(subject_id, prediction_time)`` rows, so the two sources differ only
    in provenance — and in the output file's ``{contexts_tag}``: the cohort file's stem, or
    ``sampled{n_contexts}ctx``.

    Args:
        data_dir: Root whose ``{data_dir}/data/{split}/*.parquet`` shards hold the cohort's events.
        out_dir: Output root (see the module docstring for the two layouts).
        split: Split whose shards to gather events from; must contain every cohort subject.
        query_codes: The model's query vocabulary; the sampling pool when ``sequences_path`` is
            ``None``, and the validation set for designed specs either way.
        contexts_path: Parquet with at least ``(subject_id, prediction_time)`` — the cohort.
            ``None`` samples ``n_contexts`` contexts from ``split`` instead.
        sequences_path: Designed specs (YAML/JSON/parquet).  ``None`` samples ``n_sequences``.
        n_contexts: Contexts to draw when ``contexts_path`` is ``None`` — a ceiling, since the draw
            is with replacement and then deduplicated.  Unused for a supplied cohort.
        min_prediction_times_per_subject: Stage 0 eligibility bound and Stage 2 draw floor for the
            sampled cohort.  Unused for a supplied cohort.
        per_spec_dirs: Write one ``{out_dir}/{spec_name}/{split}/tasks.parquet`` per spec instead
            of a single combined parquet.
        overwrite: Regenerate outputs that already exist.

    Returns:
        The written parquet path(s) — empty when every output existed and ``overwrite=False``.
    """
    specs = resolve_specs(
        sequences_path=sequences_path,
        query_codes=query_codes,
        n_sequences=n_sequences,
        min_queries=min_queries,
        max_queries=max_queries,
        duration_min=duration_min,
        duration_max=duration_max,
        seed=derive_seed(seed, "eval_seq_specs", split),
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        duration_distribution=duration_distribution,
    )
    validate_spec_codes(specs, query_codes)

    # Only *designed* specs get the whole-day check.  Sampled specs come from `QueryDistribution`,
    # which draws continuous float durations by design (the training sampler does too, since the
    # 5-stage port), so warning about them would fire on essentially every sampled grid.  A designed
    # spec's fractional duration is still worth flagging: it labels correctly (polars'
    # `duration(days=...)` honours the fraction) but is usually a hand-authoring slip.
    if sequences_path is not None:
        non_integer = sorted({float(d) for s in specs for d in s.durations if not float(d).is_integer()})
        if non_integer:
            logger.warning(
                "%d designed spec duration(s) are not whole days (e.g. %s); labeling honours the "
                "fraction — check this is intended.",
                len(non_integer),
                non_integer[:5],
            )

    specs_tag = Path(sequences_path).stem if sequences_path else f"sampled{len(specs)}"
    # ``contexts_path`` is None on the sampled path, so the cohort tag falls back to the *requested*
    # draw size rather than the post-dedup height: output paths are resolved (and the
    # already-exists check made) before any cohort is read.
    contexts_tag = Path(contexts_path).stem if contexts_path is not None else f"sampled{n_contexts}ctx"
    stem = f"{contexts_tag}__{specs_tag}"
    out_fps = _spec_fps(out_dir, split, stem, specs, per_spec_dirs)
    if not overwrite and all(fp.exists() for fp in out_fps):
        logger.info("All %d output(s) already exist (e.g. %s), skipping.", len(out_fps), out_fps[0])
        return []

    if contexts_path is not None:
        contexts = read_supplied_contexts(contexts_path, n_replicates=1)
        n_raw = contexts.height
        contexts = contexts.unique(maintain_order=True)
        if contexts.height < n_raw:
            # A dense grid asks every sequence about every context once; duplicated cohort rows
            # would silently double-weight those patients in the metrics.
            logger.warning(
                "Dropped %d duplicate (subject_id, prediction_time) row(s) from %s; "
                "%d unique contexts remain.",
                n_raw - contexts.height,
                contexts_path,
                contexts.height,
            )
        if contexts.height == 0:
            raise ValueError(f"{contexts_path} has no rows; nothing to evaluate.")
    else:
        contexts = sample_grid_contexts(
            data_dir=Path(data_dir),
            # The artifacts root has no config key of its own: it is always the
            # ``{name}_artifacts`` sibling of out_dir, so the final-output and intermediate trees
            # stay disjoint and never-nested (invariant 7).
            artifacts_dir=default_artifacts_dir(Path(out_dir)),
            split=split,
            n_contexts=n_contexts,
            min_prediction_times_per_subject=min_prediction_times_per_subject,
            # The ``"contexts"`` axis, matching the training sampler; the spec draw above keeps its
            # own deliberate ``("eval_seq_specs", split)`` axis.
            seed=derive_seed(seed, "contexts"),
            overwrite=overwrite,
        )
        logger.info(
            "Sampled %d unique context(s) from split=%s (requested %d).",
            contexts.height,
            split,
            n_contexts,
        )

    events_df = read_events_for_subjects(data_dir / "data" / split, contexts[TaskQuerySchema.subject_id_name])
    logger.info(
        "Loaded %d contexts x %d sequences = %d labeled rows to build, from %d events",
        contexts.height,
        len(specs),
        contexts.height * len(specs),
        events_df.height,
    )

    written: list[Path] = []
    if per_spec_dirs:
        for spec, fp in zip(specs, out_fps, strict=True):
            if fp.exists() and not overwrite:
                logger.info("Labels already exist at %s, skipping.", fp)
                continue
            index_df = build_dense_sequence_index_df(contexts, [spec])
            _write(label_binary_occurrence(index_df, events_df), fp)
            written.append(fp)
    else:
        index_df = build_dense_sequence_index_df(contexts, specs)
        _write(label_binary_occurrence(index_df, events_df), out_fps[0])
        written.append(out_fps[0])
    return written


def _write(labeled: pl.DataFrame, fp: Path) -> None:
    """Align to ``QuerySeqSchema`` and atomically write one output parquet."""
    aligned = QuerySeqSchema.align(labeled.to_arrow())
    fp.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(pl.from_arrow(aligned), fp)
    logger.info("Wrote %d labeled query sequences to %s", labeled.height, fp)


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_evaluation_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point; path roots are required args, mirroring ``sample_query_sequences.main``.

    Usage (designed sequences, one output dir per task):

        EQ_generate_evaluation_query_sequences \\
            contexts_path=cohort.parquet sequences_path=tasks.yaml \\
            split=held_out per_spec_dirs=true data_dir=... out_dir=... query_codes=...

    Usage (``N`` sequences drawn from the training distribution, one combined parquet).  The
    sampling defaults mirror ``sample_query_sequences_config.yaml``; if the checkpoint was
    trained with overrides, pass the same ones here:

        EQ_generate_evaluation_query_sequences \\
            contexts_path=cohort.parquet n_sequences=64 \\
            split=held_out data_dir=... out_dir=... query_codes=...

    Usage (no cohort file at all — the grid's contexts are sampled from the split too, the way
    ``EQ_generate_query_sequences`` samples its own).  This is the zero-argument default, so the
    endpoint is runnable with nothing but the three path roots:

        EQ_generate_evaluation_query_sequences \\
            n_contexts=512 n_sequences=64 \\
            split=held_out data_dir=... out_dir=... query_codes=...
    """
    data_dir = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    query_codes = read_query_codes(cfg.get("query_codes"))

    # Unset ``contexts_path`` is the sampled-cohort path (Stages 0 + 2 over the split), not an
    # error: the two sources are peers, exactly as in ``sample_query_sequences.main``.
    contexts_path = cfg.get("contexts_path")

    run_worker(
        data_dir=data_dir,
        out_dir=out_dir,
        split=str(cfg.split),
        query_codes=query_codes,
        contexts_path=None if contexts_path is None else str(contexts_path),
        sequences_path=cfg.get("sequences_path"),
        n_sequences=int(cfg.n_sequences),
        min_queries=int(cfg.min_queries),
        max_queries=int(cfg.max_queries),
        duration_min=float(cfg.duration_min),
        duration_max=float(cfg.duration_max),
        duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
        seed=int(cfg.seed),
        eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
        duration_mode=str(cfg.get("duration_mode", "random")),
        n_contexts=int(cfg.n_contexts),
        min_prediction_times_per_subject=int(cfg.min_prediction_times_per_subject),
        per_spec_dirs=bool(cfg.get("per_spec_dirs", False)),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
