"""Dense-grid evaluation-task generator for conditional query *sequences*.

Sibling of :mod:`~every_query.generate_tasks.sample_query_sequences` (scattered shape: every
context draws its own independent query sequence).  This module produces the **dense** shape:
a fixed set of ``N`` query sequences, each labeled at **every** context of a caller-supplied
cohort.  For a given sequence the only thing varying across its rows is the patient, which is
what per-sequence metrics need — pooling across heterogeneous sequences instead measures
cross-query base-rate separation rather than within-task skill.

Relationship to ``sample_evaluation_tasks``: same *motivation* (dense grid for evaluation),
different *row shape*.  ``sample_evaluation_tasks`` emits flat ``TaskQuerySchema`` rows — one
scalar ``boolean_value`` per ``(subject, time, code, duration)`` — for the single-query model's
``EQ_predict`` → ``EQ_evaluate`` path.  This module emits :class:`QuerySeqSchema` rows with
``queries``/``durations``/``answers`` list columns for ``EQ_predict_sequences``.  A dense grid of
*sequences* is a cross product of contexts with ordered ``(code, duration)`` **tuples**, not a
``codes x durations`` cross join, so the two share no index-building code.

Pipeline (every stage but the grid build is imported from ``sample_query_sequences``):

    1. Read the supplied cohort index from ``contexts_path`` — a parquet with at least
       ``(subject_id, prediction_time)``; duplicate contexts are dropped.
    2. Resolve the ``N`` :class:`SequenceSpec` s: read designed ones from ``sequences_path``, or
       (when that is null) draw ``n_sequences`` once from the same
       ``uniform codes x log-uniform durations`` distribution the training sampler uses.
    3. Cross-join cohort x specs into the flat per-query index frame
       (:func:`build_dense_sequence_index_df`).
    4. Label with
       :func:`~every_query.generate_tasks.sample_query_sequences.label_binary_occurrence`,
       align to ``QuerySeqSchema``, write.

Because the cohort spans arbitrary subjects rather than one shard, events are gathered across
every shard of ``split`` via ``read_events_for_subjects`` — which raises if any supplied subject
is absent from the split, rather than silently labeling them all-``False``.

Output layout, either

    ``{out_dir}/{split}/{contexts_stem}__{specs_tag}.parquet``  (default, one combined file), or
    ``{out_dir}/{spec_name}/{split}/tasks.parquet``             (``per_spec_dirs=true``),

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
)
from every_query.generate_tasks.sample_tasks import (
    _atomic_write_parquet,
    _resolve_path,
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
    duration_low: int,
    duration_high: int,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
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
    duration_min: int,
    duration_max: int,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
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
    contexts_path: str | Path,
    query_codes: list[str],
    sequences_path: str | Path | None = None,
    n_sequences: int = 64,
    min_queries: int = 5,
    max_queries: int = 5,
    duration_min: int = 1,
    duration_max: int = 365,
    seed: int = 1,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    per_spec_dirs: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Label ``N`` shared query sequences across every context of a supplied cohort.

    Args:
        data_dir: Root whose ``{data_dir}/data/{split}/*.parquet`` shards hold the cohort's events.
        out_dir: Output root (see the module docstring for the two layouts).
        split: Split whose shards to gather events from; must contain every cohort subject.
        contexts_path: Parquet with at least ``(subject_id, prediction_time)`` — the cohort.
        query_codes: The model's query vocabulary; the sampling pool when ``sequences_path`` is
            ``None``, and the validation set for designed specs either way.
        sequences_path: Designed specs (YAML/JSON/parquet).  ``None`` samples ``n_sequences``.
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
    )
    validate_spec_codes(specs, query_codes)

    non_integer = sorted({float(d) for s in specs for d in s.durations if not float(d).is_integer()})
    if non_integer:
        # Fractional horizons label correctly (polars' `duration(days=...)` honours the fraction),
        # but the training sampler only ever emits whole days, so the model is being asked about a
        # horizon shape it never saw.  Worth knowing; not worth blocking.
        logger.warning(
            "%d spec duration(s) are not whole days (e.g. %s); labeling honours the fraction, but "
            "training durations are always whole days.",
            len(non_integer),
            non_integer[:5],
        )

    specs_tag = Path(sequences_path).stem if sequences_path else f"sampled{len(specs)}"
    stem = f"{Path(contexts_path).stem}__{specs_tag}"
    out_fps = _spec_fps(out_dir, split, stem, specs, per_spec_dirs)
    if not overwrite and all(fp.exists() for fp in out_fps):
        logger.info("All %d output(s) already exist (e.g. %s), skipping.", len(out_fps), out_fps[0])
        return []

    contexts = read_supplied_contexts(contexts_path, n_replicates=1)
    n_raw = contexts.height
    contexts = contexts.unique(maintain_order=True)
    if contexts.height < n_raw:
        # A dense grid asks every sequence about every context once; duplicated cohort rows would
        # silently double-weight those patients in the metrics.
        logger.warning(
            "Dropped %d duplicate (subject_id, prediction_time) row(s) from %s; %d unique contexts remain.",
            n_raw - contexts.height,
            contexts_path,
            contexts.height,
        )
    if contexts.height == 0:
        raise ValueError(f"{contexts_path} has no rows; nothing to evaluate.")

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
    """Hydra entry point; path fallbacks mirror ``sample_query_sequences.main``.

    Usage (designed sequences, one output dir per task):

        EQ_generate_evaluation_query_sequences \\
            contexts_path=cohort.parquet sequences_path=tasks.yaml \\
            split=held_out per_spec_dirs=true data_dir=... out_dir=... query_codes=...

    Usage (``N`` sequences drawn from the training distribution, one combined parquet):

        EQ_generate_evaluation_query_sequences \\
            contexts_path=cohort.parquet n_sequences=64 min_queries=5 max_queries=5 \\
            split=held_out data_dir=... out_dir=... query_codes=...
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = _resolve_path(cfg.get("data_dir"), "INTERMEDIATE", "data_dir")
    out_dir = _resolve_path(cfg.get("out_dir"), "TASK_DIR", "out_dir")
    query_codes = read_query_codes(cfg.get("query_codes"))

    contexts_path = cfg.get("contexts_path")
    if contexts_path is None:
        raise ValueError(
            "contexts_path must be set: this endpoint labels a supplied cohort, so pass "
            "contexts_path=/path/to/cohort.parquet (a parquet with subject_id + prediction_time). "
            "To sample contexts from a shard instead, use EQ_generate_query_sequences."
        )

    run_worker(
        data_dir=data_dir,
        out_dir=out_dir,
        split=str(cfg.split),
        contexts_path=str(contexts_path),
        query_codes=query_codes,
        sequences_path=cfg.get("sequences_path"),
        n_sequences=int(cfg.n_sequences),
        min_queries=int(cfg.min_queries),
        max_queries=int(cfg.max_queries),
        duration_min=int(cfg.duration_min),
        duration_max=int(cfg.duration_max),
        seed=int(cfg.seed),
        eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
        duration_mode=str(cfg.get("duration_mode", "random")),
        per_spec_dirs=bool(cfg.get("per_spec_dirs", False)),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
