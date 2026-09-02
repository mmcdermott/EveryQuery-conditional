"""All-vocabulary, multi-bound *multitask* label generator (issue #20).

For every sampled patient context the sampler draws a fixed sequence of ``K = num_bounds`` boundaries
and produces one boolean target per **base-vocabulary code** per boundary::

    target[i, k, v] = prediction_time[i] < t_v < resolved_boundary[i, k]

where ``t_v`` is the first occurrence of code ``v`` strictly after the prediction time.  Each
boundary is either *duration-bounded* (``prediction_time + duration_days``) or *event-bounded* (the
first occurrence of ``bound_event`` strictly after the prediction time, ``+inf`` if it never recurs).
The window is open at both ends, exactly as the scalar
:func:`~every_query.generate_tasks.sample_query_sequences.label_with_event_bounds` path defines it;
that function is the correctness oracle the tests compare against.

The staged sampler architecture is reused::

    Stage 0    build + cache the canonical prediction-time map (reused as-is)
    Stage 1M   sample one sequence of K boundaries per future context (BoundaryDistribution)
    Stage 2    sample patient contexts (reused as-is)
    Stage 3M   zip boundaries with contexts, resolve prediction times, partition by event shard,
               sort each partition by (subject_id, prediction_time, _ctx_id)
    Stage 4M   per shard: build the interval table, resolve the (N, K) boundary matrix, label every
               vocabulary code with the subject-sorted interval-range kernel, pack, and write

Output, per event shard of the split::

    out_dir/{split}/{shard}.parquet             MultitaskBoundarySchema metadata, one row per context
    out_dir/{split}/{shard}.labels.npy          uint8 (rows, K, ceil(V/8)), little bit order,
                                                row-aligned with the parquet
    out_dir/{split}/_multitask_manifest.json    split-level manifest, written by the driver alone

Targets are bit-packed and written incrementally through a temporary ``open_memmap``; no unpacked
shard-wide target tensor is ever allocated, and no worker holds shard-wide ``(context, code,
next_time)`` triples.

**MVP scope**: the target vocabulary is the cohort's ``codes.parquet`` - observable codes treated as
ontology leaves, bits aligned to the unchanged ``code/vocab_index``.  A non-null ``ontology_dir`` is
a hard error.  The three seams :func:`build_target_vocabulary`, :func:`prepare_events_for_labeling`
and :func:`resolve_event_boundaries` are where ancestor support will plug in later.
"""

from __future__ import annotations

import os

# Pin polars to a single thread BEFORE importing polars, mirroring the sibling samplers: Stage 4M
# workers inherit this env, and process-level fan-out already saturates cores.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import hashlib
import json
import logging
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig, ListConfig

from every_query.data.schema import MultitaskBoundarySchema, TaskQuerySchema
from every_query.data.seq_dataset import EVENT_BOUND_DURATION_SENTINEL
from every_query.generate_tasks.interval_table import (
    INF,
    IntervalTable,
    build_interval_table,
    iter_packed_label_chunks,
    resolve_bound_times,
)
from every_query.generate_tasks.sample_query_sequences import resolve_prediction_times
from every_query.generate_tasks.sample_tasks import (
    INDEX_DIRNAME,
    LABELED_DIRNAME,
    _atomic_write_json,
    _atomic_write_parquet,
    _index_fingerprint,
    _read_event_shard,
    _require_path_arg,
    _unique_tmp_path,
    build_prediction_times,
    default_artifacts_dir,
    index_path,
    prediction_time_counts_path,
    resolve_workers,
    sample_patient_contexts,
)
from every_query.utils.seeds import derive_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MANIFEST_NAME = "_multitask_manifest.json"
LABELS_SUFFIX = ".labels.npy"
ONTOLOGY_MODE_NONE = "none"
BITORDER = "little"
WINDOW_SEMANTICS = "open_open"
MISSING_EVENT_BOUNDARY = "inf"
DATETIME_UNIT = "us"

CTX_ID_COL = "_ctx_id"
DURATIONS_COL = "durations"
BOUND_EVENTS_COL = "bound_events"
SID = TaskQuerySchema.subject_id_name
PT = TaskQuerySchema.prediction_time_name

INDEX_COLUMNS = [CTX_ID_COL, SID, PT, DURATIONS_COL, BOUND_EVENTS_COL]
METADATA_COLUMNS = [SID, PT, DURATIONS_COL, BOUND_EVENTS_COL]

ONTOLOGY_NOT_SUPPORTED = (
    "The multitask sampler currently supports observable leaf codes only. "
    "Ontology-expanded targets and boundaries will be added separately."
)


def reject_ontology(ontology_dir: object) -> None:
    """Hard-fail on any non-null ``ontology_dir`` (MVP is leaf-only)."""
    if ontology_dir is not None:
        raise NotImplementedError(ONTOLOGY_NOT_SUPPORTED)


# ---------------------------------------------------------------------------
# Vocabulary (extension seam 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetVocabulary:
    """The cohort's base vocabulary, bit-aligned to its unchanged ``code/vocab_index``.

    Attributes:
        codes: Codes sorted by vocabulary index.
        indices: ``int64`` vocabulary index per code (same order).  Unique, ``>= 0``.
        size: ``V = max(index) + 1`` - the bit width of every packed boundary row.
        fingerprint: Deterministic digest of the ordered ``(index, code)`` mapping.

    Examples:
        >>> v = TargetVocabulary.from_pairs(["B", "A", "PAD"], [2, 1, 0])
        >>> v.codes, v.indices.tolist(), v.size
        (('PAD', 'A', 'B'), [0, 1, 2], 3)
        >>> v.boundary_candidates()
        ['A', 'B']
        >>> v.fingerprint == TargetVocabulary.from_pairs(["A", "B", "PAD"], [1, 2, 0]).fingerprint
        True
    """

    codes: tuple[str, ...]
    indices: np.ndarray
    size: int
    fingerprint: str
    packed_width: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "packed_width", (self.size + 7) // 8)

    @classmethod
    def from_pairs(cls, codes: Sequence[str], indices: Sequence[int]) -> TargetVocabulary:
        if len(codes) != len(indices):
            raise ValueError("codes and indices must have the same length")
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("the target vocabulary is empty")
        if idx.min() < 0:
            raise ValueError("code/vocab_index must be non-negative")
        if len(np.unique(idx)) != idx.size:
            raise ValueError("code/vocab_index values must be unique")
        if len(set(codes)) != len(codes):
            raise ValueError("codes must be unique")
        order = np.argsort(idx, kind="stable")
        ordered_codes = tuple(str(codes[i]) for i in order)
        ordered_idx = idx[order]
        size = int(ordered_idx[-1]) + 1
        h = hashlib.sha256()
        h.update(f"multitask-vocab-v{FORMAT_VERSION}:{size}\n".encode())
        for i, c in zip(ordered_idx.tolist(), ordered_codes, strict=True):
            h.update(f"{i}\t{c}\n".encode())
        return cls(codes=ordered_codes, indices=ordered_idx, size=size, fingerprint=h.hexdigest())

    def code_to_index(self) -> dict[str, int]:
        return dict(zip(self.codes, self.indices.tolist(), strict=True))

    def boundary_candidates(self) -> list[str]:
        """Codes usable as event boundaries: every base code except PAD / index zero."""
        return [c for c, i in zip(self.codes, self.indices.tolist(), strict=True) if i != 0]


def _codes_parquet_path(source: object) -> Path:
    p = Path(str(source))
    if p.is_dir():
        p = p / "metadata" / "codes.parquet"
    if p.suffix != ".parquet":
        raise ValueError(
            f"query_codes must be a metadata root dir or a codes.parquet path (got {source!r}); the "
            "multitask sampler aligns bits to code/vocab_index, which an explicit code list lacks."
        )
    return p


def build_target_vocabulary(source: object, ontology_dir: object = None) -> TargetVocabulary:
    """Extension seam 1: the vocabulary whose codes are targets (and boundary candidates).

    MVP: the base cohort vocabulary from ``codes.parquet`` (``code`` + ``code/vocab_index``), every
    code an ontology leaf.  Codes absent from a given split are neither removed nor renumbered; their
    bits simply stay false for that split.  A non-null ``ontology_dir`` raises.
    """
    reject_ontology(ontology_dir)
    if isinstance(source, list | tuple | ListConfig):
        raise ValueError(
            "query_codes must be a metadata root dir or a codes.parquet path; the multitask sampler "
            "aligns bits to code/vocab_index, which an explicit code list lacks."
        )
    if source is None or not str(source).strip():
        raise ValueError("query_codes is unset; pass a metadata root dir or a codes.parquet path.")
    fp = _codes_parquet_path(source)
    df = pl.read_parquet(fp, columns=["code", "code/vocab_index"]).filter(
        pl.col("code/vocab_index").is_not_null() & pl.col("code").is_not_null()
    )
    if df.height == 0:
        raise ValueError(f"{fp} holds no indexed codes")
    return TargetVocabulary.from_pairs(df["code"].to_list(), df["code/vocab_index"].to_list())


def read_boundary_codes(spec: object, vocab: TargetVocabulary) -> list[str]:
    """Resolve the boundary-code pool: ``None`` => all base codes (index >= 1); else a list / YAML path.

    Every listed code must be in the vocabulary and must not be PAD; unknown codes are hard errors.
    Order-preserving dedup.
    """
    if spec is None:
        return vocab.boundary_candidates()
    if isinstance(spec, list | tuple | ListConfig):
        raw = list(spec)
    else:
        from every_query.generate_tasks.sample_tasks import read_query_codes

        raw = read_query_codes(str(spec))
    seen: set[str] = set()
    codes = [c for c in raw if not (c in seen or seen.add(c))]
    if not codes:
        raise ValueError("boundary_codes resolved to an empty list")
    c2i = vocab.code_to_index()
    unknown = [c for c in codes if c not in c2i]
    if unknown:
        raise ValueError(f"{len(unknown)} boundary code(s) are not in the base vocabulary: {unknown[:10]}")
    pad = [c for c in codes if c2i[c] == 0]
    if pad:
        raise ValueError(f"boundary code(s) at vocab index 0 (PAD) are not allowed: {pad}")
    return codes


# ---------------------------------------------------------------------------
# Stage 1M - the boundary-sequence distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundarySample:
    """``num_contexts x K`` boundary draws: ``durations`` (float32) and ``bound_events`` (object, None)."""

    durations: np.ndarray
    bound_events: np.ndarray

    @property
    def n(self) -> int:
        return int(self.durations.shape[0])

    @property
    def k(self) -> int:
        return int(self.durations.shape[1])


@dataclass(frozen=True)
class BoundaryDistribution:
    """Stage 1M: draw one fixed-length sequence of ``K`` boundaries per context.

    Every slot is drawn independently: a Bernoulli(``eventbound_fraction``) form draw, a duration from
    the configured distribution, and a boundary code iid uniform with replacement over
    ``boundary_codes``.  The three axes use three caller-owned generators and every stream is drawn
    in full for all ``n x K`` slots before masking, so changing ``eventbound_fraction`` (or the code
    pool) perturbs neither the duration stream nor the code stream, and vice versa.

    Examples:
        >>> dist = BoundaryDistribution(num_bounds=3, min_duration=1.0, max_duration=10.0,
        ...     duration_distribution="uniform", eventbound_fraction=0.5, boundary_codes=("A", "B"))
        >>> rngs = lambda: [np.random.default_rng(i) for i in range(3)]
        >>> s = dist.sample(4, *rngs())
        >>> s.durations.shape, s.bound_events.shape, s.durations.dtype
        ((4, 3), (4, 3), dtype('float32'))
        >>> bool(((s.durations == -1.0) == (s.bound_events != None)).all())  # noqa: E711
        True

        Durations of duration-bounded slots are unaffected by the event fraction:

        >>> off = BoundaryDistribution(3, 1.0, 10.0, "uniform", 0.0, ("A", "B")).sample(4, *rngs())
        >>> keep = s.bound_events == None  # noqa: E711
        >>> bool((s.durations[keep] == off.durations[keep]).all())
        True
    """

    num_bounds: int
    min_duration: float
    max_duration: float
    duration_distribution: str
    eventbound_fraction: float
    boundary_codes: tuple[str, ...]

    _VALID_DISTRIBUTIONS = ("uniform", "log-uniform")

    def __post_init__(self) -> None:
        if self.num_bounds < 1:
            raise ValueError(f"num_bounds must be >= 1 (got {self.num_bounds})")
        if self.min_duration <= 0:
            raise ValueError(f"min_duration must be > 0 (got {self.min_duration})")
        if self.max_duration < self.min_duration:
            raise ValueError(
                f"max_duration ({self.max_duration}) must be >= min_duration ({self.min_duration})"
            )
        if self.duration_distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"duration_distribution must be one of {self._VALID_DISTRIBUTIONS} "
                f"(got {self.duration_distribution!r})"
            )
        if not 0.0 <= self.eventbound_fraction <= 1.0:
            raise ValueError(f"eventbound_fraction must be in [0, 1] (got {self.eventbound_fraction})")
        if self.eventbound_fraction > 0 and not self.boundary_codes:
            raise ValueError("eventbound_fraction > 0 requires a non-empty boundary_codes pool")

    @classmethod
    def from_config(cls, cfg: DictConfig, boundary_codes: Sequence[str]) -> BoundaryDistribution:
        return cls(
            num_bounds=int(cfg.get("num_bounds", 5)),
            min_duration=float(cfg.duration_min),
            max_duration=float(cfg.duration_max),
            duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
            eventbound_fraction=float(cfg.get("eventbound_fraction", 0.0) or 0.0),
            boundary_codes=tuple(boundary_codes),
        )

    def sample(
        self,
        num_contexts: int,
        form_rng: np.random.Generator,
        duration_rng: np.random.Generator,
        code_rng: np.random.Generator,
    ) -> BoundarySample:
        if num_contexts < 0:
            raise ValueError(f"num_contexts must be >= 0 (got {num_contexts})")
        shape = (num_contexts, self.num_bounds)
        is_event = form_rng.random(shape) < self.eventbound_fraction
        if self.duration_distribution == "log-uniform":
            durations = np.exp(
                duration_rng.uniform(np.log(self.min_duration), np.log(self.max_duration), shape)
            )
        else:
            durations = duration_rng.uniform(self.min_duration, self.max_duration, shape)
        durations = durations.astype(np.float32)
        bound_events = np.full(shape, None, dtype=object)
        if self.boundary_codes:
            picks = code_rng.integers(0, len(self.boundary_codes), size=shape)
            pool = np.array(self.boundary_codes, dtype=object)
            bound_events[is_event] = pool[picks[is_event]]
        durations[is_event] = EVENT_BOUND_DURATION_SENTINEL
        return BoundarySample(durations=durations, bound_events=bound_events)


# ---------------------------------------------------------------------------
# Stage 3M - zip with contexts, resolve prediction times, partition and sort
# ---------------------------------------------------------------------------


def _bounds_to_columns(sample: BoundarySample) -> dict[str, pl.Series]:
    """``(N, K)`` arrays -> ``List`` columns via a flat series + reshape: no per-slot Python objects."""
    n, k = sample.durations.shape
    durations = pl.Series(DURATIONS_COL, sample.durations.ravel(), dtype=pl.Float32)
    # A flat list of references to the (shared) pool strings; an object ndarray would be inferred as
    # polars Object when every slot is None.
    bound_events = pl.Series(BOUND_EVENTS_COL, sample.bound_events.ravel().tolist(), dtype=pl.Utf8)
    return {
        DURATIONS_COL: durations.reshape((n, k)).arr.to_list(),
        BOUND_EVENTS_COL: bound_events.reshape((n, k)).arr.to_list(),
    }


def sort_index_for_labeling(index_df: pl.DataFrame) -> pl.DataFrame:
    """Give every context a stable id and sort by ``(subject_id, prediction_time, _ctx_id)``.

    A missing ``_ctx_id`` is filled with the row position, so a supplied index labels the same way as
    a sampled one; the sort is what lets Stage 4M scan each subject's interval slice once per chunk.
    """
    if CTX_ID_COL not in index_df.columns:
        index_df = index_df.with_row_index(CTX_ID_COL).with_columns(pl.col(CTX_ID_COL).cast(pl.Int64))
    return index_df.with_columns(pl.col(CTX_ID_COL).cast(pl.Int64)).sort(SID, PT, CTX_ID_COL)


def build_multitask_index(
    sample: BoundarySample,
    contexts: pl.DataFrame,
    artifacts_dir: Path,
    split: str,
) -> int:
    """Stage 3M: zip boundary sequences with contexts, resolve prediction times, write per-shard index.

    ``sample`` row ``i`` belongs to ``contexts`` row ``i``.  Each partition is written sorted by
    ``(subject_id, prediction_time, _ctx_id)``; ``_ctx_id`` is the global sampling position, so the
    original order is recoverable but need not be preserved downstream.
    """
    if sample.n != contexts.height:
        raise ValueError(f"sample has {sample.n} rows but contexts has {contexts.height}")
    if sample.n == 0:
        raise ValueError("no contexts to index")

    combined = contexts.with_columns(
        pl.Series(CTX_ID_COL, np.arange(sample.n, dtype=np.int64)), **_bounds_to_columns(sample)
    )
    index_dir = artifacts_dir / split / INDEX_DIRNAME
    if index_dir.exists():
        shutil.rmtree(index_dir)

    n_shards = 0
    combined = combined.sort("shard")
    for shard_key, group in combined.group_by("shard", maintain_order=True):
        (shard_name,) = shard_key
        joined = resolve_prediction_times(group, artifacts_dir, split, str(shard_name))
        joined = joined.with_columns(pl.col(PT).cast(pl.Datetime("us")))
        _atomic_write_parquet(
            sort_index_for_labeling(joined.select(INDEX_COLUMNS)),
            index_path(artifacts_dir, split, str(shard_name)),
        )
        n_shards += 1
    return n_shards


# ---------------------------------------------------------------------------
# Manifest + fingerprints
# ---------------------------------------------------------------------------


def _sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_fingerprint(dist: BoundaryDistribution, vocab: TargetVocabulary) -> str:
    """Digest of everything that changes the *meaning* of a stored label bit.

    A changed duration distribution, event-bound fraction, number of boundaries, boundary pool, base
    vocabulary or window semantics changes this digest and so invalidates existing labels.
    """
    return _sha256_json(
        {
            "format_version": FORMAT_VERSION,
            "num_bounds": dist.num_bounds,
            "duration_min": dist.min_duration,
            "duration_max": dist.max_duration,
            "duration_distribution": dist.duration_distribution,
            "eventbound_fraction": dist.eventbound_fraction,
            "boundary_codes": _sha256_json(list(dist.boundary_codes)),
            "vocab_fingerprint": vocab.fingerprint,
            "vocab_size": vocab.size,
            "window": WINDOW_SEMANTICS,
            "missing_event_boundary": MISSING_EVENT_BOUNDARY,
            "datetime_unit": DATETIME_UNIT,
            "ontology_mode": ONTOLOGY_MODE_NONE,
        }
    )


def build_manifest(dist: BoundaryDistribution, vocab: TargetVocabulary) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "num_bounds": dist.num_bounds,
        "vocab_size": vocab.size,
        "packed_width_bytes": vocab.packed_width,
        "bitorder": BITORDER,
        "window": WINDOW_SEMANTICS,
        "missing_event_boundary": MISSING_EVENT_BOUNDARY,
        "datetime_unit": DATETIME_UNIT,
        "event_bound_duration_sentinel": EVENT_BOUND_DURATION_SENTINEL,
        "vocab_fingerprint": vocab.fingerprint,
        "ontology_mode": ONTOLOGY_MODE_NONE,
        "config_fingerprint": config_fingerprint(dist, vocab),
        "labels_suffix": LABELS_SUFFIX,
    }


def manifest_path(split_dir: Path) -> Path:
    return split_dir / MANIFEST_NAME


def validate_manifest(manifest: dict) -> dict:
    """Check the fields every reader relies on; return the manifest for chaining."""
    required = {
        "format_version",
        "num_bounds",
        "vocab_size",
        "packed_width_bytes",
        "bitorder",
        "window",
        "missing_event_boundary",
        "datetime_unit",
        "vocab_fingerprint",
        "ontology_mode",
        "config_fingerprint",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"multitask manifest is missing {missing}")
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported manifest format_version {manifest['format_version']!r}")
    if manifest["bitorder"] != BITORDER:
        raise ValueError(f"manifest bitorder must be {BITORDER!r}, got {manifest['bitorder']!r}")
    if manifest["ontology_mode"] != ONTOLOGY_MODE_NONE:
        raise ValueError(
            f"manifest ontology_mode must be {ONTOLOGY_MODE_NONE!r}, got {manifest['ontology_mode']!r}"
        )
    if manifest["packed_width_bytes"] != (int(manifest["vocab_size"]) + 7) // 8:
        raise ValueError("manifest packed_width_bytes disagrees with vocab_size")
    if int(manifest["num_bounds"]) < 1:
        raise ValueError("manifest num_bounds must be >= 1")
    return manifest


def write_manifest(split_dir: Path, manifest: dict) -> dict:
    """Driver-only: atomically write the split manifest, then read it back and validate it."""
    validate_manifest(manifest)
    split_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest, manifest_path(split_dir))
    return read_manifest(split_dir)


def read_manifest(split_dir: Path) -> dict:
    fp = manifest_path(split_dir)
    if not fp.exists():
        raise FileNotFoundError(f"no multitask manifest at {fp}")
    return validate_manifest(json.loads(fp.read_text()))


# ---------------------------------------------------------------------------
# Stage 4M - labeling kernel
# ---------------------------------------------------------------------------


def prepare_events_for_labeling(events_df: pl.DataFrame, ontology_dir: object = None) -> pl.DataFrame:
    """Extension seam 2: the event stream the interval table is built from.

    MVP: the original stream, unexpanded (no closure expansion).  A non-null ``ontology_dir`` raises.
    """
    reject_ontology(ontology_dir)
    return events_df


def resolve_event_boundaries(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    durations: np.ndarray,
    bound_code_index: np.ndarray,
    ontology_dir: object = None,
) -> np.ndarray:
    """Extension seam 3: the ``(N, K)`` boundary-time matrix.

    MVP: leaf-code event bounds via :func:`~every_query.generate_tasks.interval_table.resolve_bound_times`.
    """
    reject_ontology(ontology_dir)
    return resolve_bound_times(table, subject_ids, prediction_times, durations, bound_code_index)


def validate_index(index_df: pl.DataFrame, num_bounds: int) -> None:
    """Check the Stage 3M / supplied index contract: K slots per row, one active representation each."""
    for col in (SID, PT, DURATIONS_COL, BOUND_EVENTS_COL):
        if col not in index_df.columns:
            raise ValueError(f"index is missing required column {col!r}")
    if index_df.height == 0:
        return
    lens = index_df.select(
        pl.col(DURATIONS_COL).list.len().alias("d"), pl.col(BOUND_EVENTS_COL).list.len().alias("b")
    )
    if not ((lens["d"] == num_bounds) & (lens["b"] == num_bounds)).all():
        raise ValueError(f"every index row must carry exactly {num_bounds} durations and bound_events")
    flat = index_df.select(
        pl.col(DURATIONS_COL).explode().alias("d"), pl.col(BOUND_EVENTS_COL).explode().alias("b")
    )
    bounded = flat["b"].is_not_null()
    ok = (bounded & (flat["d"] == EVENT_BOUND_DURATION_SENTINEL)) | (~bounded & (flat["d"] >= 0))
    if not ok.all():
        raise ValueError(
            f"each boundary slot must be either duration >= 0 with a null bound_event, or duration == "
            f"{EVENT_BOUND_DURATION_SENTINEL} with a non-null bound_event"
        )


def _encode_events(
    events_df: pl.DataFrame, vocab: TargetVocabulary
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(subject_id, time_us, code_index)`` of the non-null-time, in-vocabulary events + drop count.

    Index 0 (PAD) is excluded before the table is built, so target bit 0 is false by construction.
    """
    codes_df = pl.DataFrame({"code": list(vocab.codes), "_code_index": vocab.indices})
    ev = (
        events_df.select(SID, DataSchema.time_name, DataSchema.code_name)
        .filter(pl.col(DataSchema.time_name).is_not_null())
        .with_columns(pl.col(DataSchema.code_name).cast(pl.Utf8))
        .join(codes_df, on="code", how="left")
    )
    n_unknown = int(ev["_code_index"].null_count())
    ev = ev.filter(pl.col("_code_index") > 0)  # drops unknown (null) and PAD (0)
    sid = ev[SID].to_numpy().astype(np.int64)
    t = ev[DataSchema.time_name].cast(pl.Datetime("us")).cast(pl.Int64).to_numpy().astype(np.int64)
    ci = ev["_code_index"].to_numpy().astype(np.int64)
    return sid, t, ci, n_unknown


def _encode_bounds(
    index_df: pl.DataFrame, vocab: TargetVocabulary, num_bounds: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(durations (N, K) float32, bound_code_index (N, K) int64 with -1 for duration slots)``."""
    n = index_df.height
    durations = np.asarray(index_df[DURATIONS_COL].explode().to_numpy(), dtype=np.float32).reshape(
        n, num_bounds
    )
    bound_events = index_df[BOUND_EVENTS_COL].explode()
    c2i = vocab.code_to_index()
    distinct = bound_events.drop_nulls().unique().to_list()
    unknown = sorted(c for c in distinct if c not in c2i)
    if unknown:
        raise ValueError(f"{len(unknown)} boundary code(s) are not in the base vocabulary: {unknown[:10]}")
    pad = [c for c in distinct if c2i[c] == 0]
    if pad:
        raise ValueError(f"boundary code(s) at vocab index 0 (PAD) are not allowed: {pad}")
    mapping = pl.DataFrame(
        {"code": distinct, "_i": [c2i[c] for c in distinct]}, schema={"code": pl.Utf8, "_i": pl.Int64}
    )
    encoded = (
        pl.DataFrame({"code": bound_events.cast(pl.Utf8)})
        .join(mapping, on="code", how="left", maintain_order="left")["_i"]
        .fill_null(-1)
        .to_numpy()
        .astype(np.int64)
        .reshape(n, num_bounds)
    )
    return durations, encoded


@dataclass
class LabelStats:
    n_events: int = 0
    n_unknown_code_events: int = 0
    n_intervals: int = 0
    n_contexts: int = 0
    vocab_size: int = 0
    packed_width: int = 0
    num_bounds: int = 0
    build_seconds: float = 0.0
    bound_seconds: float = 0.0
    label_seconds: float = 0.0
    contexts_per_second: float = 0.0
    n_event_bounds: int = 0
    frac_event_bounds_inf: float = 0.0
    mean_positives_per_context_boundary: float = 0.0
    output_bytes: int = 0
    peak_rss_bytes: int | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(kb) * 1024
    except Exception:  # pragma: no cover - platform without getrusage
        return None


def label_multitask_index(
    index_df: pl.DataFrame,
    events_df: pl.DataFrame,
    vocab: TargetVocabulary,
    num_bounds: int,
    *,
    chunk_rows: int = 2000,
    out: np.ndarray | None = None,
    ontology_dir: object = None,
) -> tuple[pl.DataFrame, np.ndarray, LabelStats]:
    """Stage 4M kernel: label a (possibly supplied) index against an event stream.

    Returns ``(metadata_df, packed, stats)``: ``metadata_df`` is the index sorted by
    ``(subject_id, prediction_time, _ctx_id)`` and reduced to :data:`METADATA_COLUMNS`; ``packed`` is
    the row-aligned ``(N, K, ceil(V/8))`` uint8 target array (``out`` when supplied - typically a
    writable memmap - else a freshly allocated packed array).  No dense ``(N, K, V)`` tensor is ever
    built; scratch is ``chunk_rows x K x V`` booleans.

    Sampling policy never enters here: everything about a boundary is fixed by the index rows.

    Examples:
        >>> from datetime import datetime
        >>> vocab = TargetVocabulary.from_pairs(["A", "B", "END"], [1, 2, 3])
        >>> events = pl.DataFrame({"subject_id": [1, 1, 1],
        ...     "time": [datetime(2024, 1, 3), datetime(2024, 1, 6), datetime(2024, 1, 9)],
        ...     "code": ["A", "B", "END"]}).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({"subject_id": [1], "prediction_time": [datetime(2024, 1, 1)],
        ...     "durations": [[30.0, -1.0]], "bound_events": [[None, "B"]]}).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("durations").cast(pl.List(pl.Float32)),
        ...     pl.col("bound_events").cast(pl.List(pl.Utf8)))
        >>> meta, packed, stats = label_multitask_index(idx, events, vocab, 2)
        >>> np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").tolist()
        [[[0, 1, 1, 1], [0, 1, 0, 0]]]
    """
    reject_ontology(ontology_dir)
    validate_index(index_df, num_bounds)
    stats = LabelStats(vocab_size=vocab.size, packed_width=vocab.packed_width, num_bounds=num_bounds)

    index_df = sort_index_for_labeling(index_df)
    n = index_df.height
    stats.n_contexts = n

    t0 = time.perf_counter()
    events_df = prepare_events_for_labeling(events_df, ontology_dir)
    ev_sid, ev_t, ev_ci, n_unknown = _encode_events(events_df, vocab)
    stats.n_events = int(ev_sid.size)
    stats.n_unknown_code_events = n_unknown
    table = build_interval_table(ev_sid, ev_t, ev_ci, vocab_size=vocab.size)
    stats.n_intervals = table.n_rows
    stats.build_seconds = time.perf_counter() - t0

    subject_ids = index_df[SID].to_numpy().astype(np.int64)
    prediction_times = index_df[PT].cast(pl.Datetime("us")).cast(pl.Int64).to_numpy().astype(np.int64)
    durations, bound_code_index = _encode_bounds(index_df, vocab, num_bounds)

    t1 = time.perf_counter()
    bound_times = resolve_event_boundaries(
        table, subject_ids, prediction_times, durations, bound_code_index, ontology_dir
    )
    stats.bound_seconds = time.perf_counter() - t1
    is_event = bound_code_index >= 0
    stats.n_event_bounds = int(is_event.sum())
    stats.frac_event_bounds_inf = float((bound_times[is_event] == INF).mean()) if is_event.any() else 0.0

    shape = (n, num_bounds, vocab.packed_width)
    if out is None:
        out = np.zeros(shape, dtype=np.uint8)
    elif tuple(out.shape) != shape or out.dtype != np.uint8:
        raise ValueError(f"out must be uint8 with shape {shape}, got {out.dtype} {out.shape}")

    t2 = time.perf_counter()
    positives = 0
    for lo, hi, packed in iter_packed_label_chunks(
        table, subject_ids, prediction_times, bound_times, vocab.size, chunk_rows
    ):
        out[lo:hi] = packed
        positives += int(np.bitwise_count(packed).sum())  # popcount on packed bytes: no unpacked scratch
    stats.label_seconds = time.perf_counter() - t2
    stats.contexts_per_second = n / stats.label_seconds if stats.label_seconds > 0 else float("inf")
    stats.mean_positives_per_context_boundary = positives / (n * num_bounds) if n else 0.0
    stats.output_bytes = int(np.prod(shape))
    stats.peak_rss_bytes = _peak_rss_bytes()

    metadata = index_df.select(METADATA_COLUMNS)
    return metadata, out, stats


# ---------------------------------------------------------------------------
# Stage 4M - per-shard worker with atomic sidecars and fingerprint-keyed reuse
# ---------------------------------------------------------------------------


def labels_path(out_dir: Path, shard: str) -> Path:
    return out_dir / f"{shard}{LABELS_SUFFIX}"


def _clean_stale_multitask_temps(out_dir: Path, shard: str) -> int:
    removed = 0
    for pattern in (f".{shard}.parquet.tmp.*", f".{shard}{LABELS_SUFFIX}.tmp.*"):
        for tmp in out_dir.glob(pattern):
            tmp.unlink(missing_ok=True)
            removed += 1
    return removed


def _packed_shape(fp: Path) -> tuple[int, ...] | None:
    try:
        return tuple(np.load(fp, mmap_mode="r").shape)
    except Exception:
        return None


def output_is_reusable(
    final_parquet: Path,
    final_labels: Path,
    sidecar_fp: Path,
    index_fingerprint: str,
    manifest: dict,
) -> bool:
    """The full reuse gate: both files, agreeing row counts, right packed shape, all fingerprints."""
    if not (final_parquet.exists() and final_labels.exists()):
        return False
    try:
        recorded = json.loads(sidecar_fp.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("ontology_mode") != ONTOLOGY_MODE_NONE:
        return False
    if recorded.get("index_fingerprint") != index_fingerprint:
        return False
    if recorded.get("vocab_fingerprint") != manifest["vocab_fingerprint"]:
        return False
    if recorded.get("config_fingerprint") != manifest["config_fingerprint"]:
        return False
    if recorded.get("ontology_mode") != ONTOLOGY_MODE_NONE:
        return False
    shape = _packed_shape(final_labels)
    if shape is None:
        return False
    try:
        n_rows = pl.scan_parquet(final_parquet).select(pl.len()).collect().item()
    except Exception:
        return False
    return shape == (n_rows, int(manifest["num_bounds"]), int(manifest["packed_width_bytes"]))


def write_labeled_shard(
    metadata: pl.DataFrame | None,
    out_dir: Path,
    shard: str,
    *,
    labels_tmp: Path | None = None,
) -> None:
    """Commit a shard: rename the flushed temp labels file, then atomically write the parquet.

    Order matters for restart safety - labels first, parquet second, fingerprint sidecar (written by
    the caller) last - so a present sidecar always describes a present, complete pair.
    """
    if labels_tmp is not None:
        os.replace(labels_tmp, labels_path(out_dir, shard))
    aligned = MultitaskBoundarySchema.align(metadata.to_arrow())
    _atomic_write_parquet(pl.from_arrow(aligned), out_dir / f"{shard}.parquet")


def label_one_multitask_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    labeled_dir: Path,
    codes_source: str,
    manifest: dict,
    overwrite: bool = False,
    chunk_rows: int = 2000,
) -> tuple[str, str, dict]:
    """Stage 4M worker: label one index partition; write ``{shard}.labels.npy`` + ``{shard}.parquet``.

    Module-level so it pickles under ``spawn``.  Reads the manifest the driver already wrote (it never
    writes one).  Returns ``(shard, status, stats)`` with status ``"skipped"`` or ``"labeled"``.
    """
    validate_manifest(manifest)
    num_bounds = int(manifest["num_bounds"])
    final_parquet = out_dir / f"{shard}.parquet"
    final_labels = labels_path(out_dir, shard)
    sidecar_fp = labeled_dir / f"{shard}.json"

    index_df = pl.read_parquet(index_dir / f"{shard}.parquet")
    current_fingerprint = _index_fingerprint(index_df)

    if not overwrite and output_is_reusable(
        final_parquet, final_labels, sidecar_fp, current_fingerprint, manifest
    ):
        return shard, "skipped", {}

    _clean_stale_multitask_temps(out_dir, shard)
    vocab = build_target_vocabulary(codes_source)
    if vocab.fingerprint != manifest["vocab_fingerprint"] or vocab.size != int(manifest["vocab_size"]):
        raise ValueError(
            f"shard {shard}: the vocabulary at {codes_source} (size {vocab.size}, {vocab.fingerprint[:12]}) "
            f"does not match the manifest (size {manifest['vocab_size']}, "
            f"{manifest['vocab_fingerprint'][:12]})"
        )

    events_df = _read_event_shard(data_dir / f"{shard}.parquet")

    shape = (index_df.height, num_bounds, vocab.packed_width)
    labels_tmp = _unique_tmp_path(final_labels)
    try:
        if index_df.height == 0:
            metadata = sort_index_for_labeling(index_df).select(METADATA_COLUMNS)
            # np.save would append '.npy' to a suffix-less temp name; write through the handle.
            with open(labels_tmp, "wb") as f:
                np.save(f, np.zeros(shape, dtype=np.uint8))
            stats = LabelStats(vocab_size=vocab.size, packed_width=vocab.packed_width, num_bounds=num_bounds)
        else:
            mm = np.lib.format.open_memmap(labels_tmp, mode="w+", dtype=np.uint8, shape=shape)
            try:
                metadata, _, stats = label_multitask_index(
                    index_df, events_df, vocab, num_bounds, chunk_rows=chunk_rows, out=mm
                )
                mm.flush()
            finally:
                del mm
        write_labeled_shard(metadata, out_dir, shard, labels_tmp=labels_tmp)
    except Exception:
        labels_tmp.unlink(missing_ok=True)
        raise

    _atomic_write_json(
        {
            "index_fingerprint": current_fingerprint,
            "vocab_fingerprint": vocab.fingerprint,
            "config_fingerprint": manifest["config_fingerprint"],
            "ontology_mode": ONTOLOGY_MODE_NONE,
            "n_rows": index_df.height,
            "stats": stats.as_dict(),
        },
        sidecar_fp,
    )
    return shard, "labeled", stats.as_dict()


def _log_shard_stats(shard: str, stats: dict) -> None:
    if not stats:
        return
    rss = stats.get("peak_rss_bytes")
    logger.info(
        "Stage 4M shard %s: events=%s (unknown-code dropped=%s) intervals=%s contexts=%s V=%s "
        "packed_width=%s "
        "build=%.2fs bounds=%.2fs label+pack=%.2fs (%.0f ctx/s) event_bounds=%s inf_frac=%.3f "
        "mean_pos/ctx/bound=%.2f output=%s bytes peak_rss=%s",
        shard,
        f"{stats['n_events']:,}",
        f"{stats['n_unknown_code_events']:,}",
        f"{stats['n_intervals']:,}",
        f"{stats['n_contexts']:,}",
        stats["vocab_size"],
        stats["packed_width"],
        stats["build_seconds"],
        stats["bound_seconds"],
        stats["label_seconds"],
        stats["contexts_per_second"],
        f"{stats['n_event_bounds']:,}",
        stats["frac_event_bounds_inf"],
        stats["mean_positives_per_context_boundary"],
        f"{stats['output_bytes']:,}",
        f"{rss / 2**20:.0f} MiB" if rss else "n/a",
    )


def _prune_stale_multitask_outputs(out_dir: Path, labeled_dir: Path, current: set[str]) -> None:
    for fp in out_dir.glob("*.parquet"):
        if fp.stem not in current:
            fp.unlink()
    for fp in out_dir.glob(f"*{LABELS_SUFFIX}"):
        shard = fp.name[: -len(LABELS_SUFFIX)]
        if shard not in current:
            fp.unlink()
    if labeled_dir.exists():
        for fp in labeled_dir.glob("*.json"):
            if fp.stem not in current:
                fp.unlink()


def _label_multitask_shards(
    shards: list[str],
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    labeled_dir: Path,
    codes_source: str,
    manifest: dict,
    overwrite: bool,
    n_workers: int,
    chunk_rows: int,
) -> dict[str, str]:
    """Fan one worker per shard through a ``spawn`` pool (fork would inherit polars' locked threads)."""
    mp_context = multiprocessing.get_context("spawn")
    statuses: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
        futs = {
            ex.submit(
                label_one_multitask_shard,
                s,
                index_dir,
                data_dir,
                out_dir,
                labeled_dir,
                codes_source,
                manifest,
                overwrite,
                chunk_rows,
            ): s
            for s in shards
        }
        for fut in as_completed(futs):
            shard, status, stats = fut.result()
            statuses[shard] = status
            _log_shard_stats(shard, stats)
    return statuses


def _validate_context_count(out_dir: Path, expected: int, manifest: dict) -> int:
    written = 0
    for fp in sorted(out_dir.glob("*.parquet")):
        n = pl.scan_parquet(fp).select(pl.len()).collect().item()
        shape = _packed_shape(labels_path(out_dir, fp.stem))
        want = (n, int(manifest["num_bounds"]), int(manifest["packed_width_bytes"]))
        if shape != want:
            raise ValueError(f"{fp.stem}: packed labels have shape {shape}, expected {want}")
        written += n
    if written != expected:
        raise ValueError(
            f"Expected {expected:,} labeled contexts but found {written:,} across {out_dir}. "
            "The output directory may hold a partial run; rerun with overwrite=true."
        )
    return written


def label_multitask_shards(
    cfg: DictConfig,
    path_to_data: Path,
    out_root: Path,
    artifacts_dir: Path,
    manifest: dict,
    total_contexts: int,
) -> int:
    """Stage 4M driver half: prune stale outputs, fan out workers, validate the final row count."""
    index_dir = artifacts_dir / cfg.split / INDEX_DIRNAME
    labeled_dir = artifacts_dir / cfg.split / LABELED_DIRNAME
    data_dir = path_to_data / "data" / cfg.split
    out_dir = out_root / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(p.stem for p in index_dir.glob("*.parquet"))
    _prune_stale_multitask_outputs(out_dir, labeled_dir, set(shards))

    n_workers = resolve_workers(cfg.get("max_workers"))
    chunk_rows = int(cfg.get("label_chunk_rows", 2000))
    logger.info(
        "Stage 4M: labeling %s shard(s) across %s worker(s), chunk_rows=%s (scratch ~%.0f MiB/worker).",
        f"{len(shards):,}",
        f"{n_workers:,}",
        f"{chunk_rows:,}",
        chunk_rows * int(manifest["num_bounds"]) * int(manifest["vocab_size"]) / 2**20,
    )
    statuses = _label_multitask_shards(
        shards,
        index_dir,
        data_dir,
        out_dir,
        labeled_dir,
        str(cfg.query_codes),
        manifest,
        bool(cfg.overwrite),
        n_workers,
        chunk_rows,
    )
    n_skipped = sum(s == "skipped" for s in statuses.values())
    written = _validate_context_count(out_dir, total_contexts, manifest)
    logger.info(
        "Pipeline complete: %s contexts x %s boundaries x %s codes in %s (%s shard(s) reused).",
        f"{written:,}",
        manifest["num_bounds"],
        f"{manifest['vocab_size']:,}",
        out_dir,
        n_skipped,
    )
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Execute Stages 0-4M for a fully-resolved config (no Hydra side effects)."""
    # Leaf-only MVP: fail before any Stage 0 work.
    reject_ontology(cfg.get("ontology_dir"))

    path_to_data = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_root = _require_path_arg(cfg.get("out_dir"), "out_dir")
    artifacts_dir = default_artifacts_dir(out_root)

    vocab = build_target_vocabulary(cfg.get("query_codes"))
    boundary_codes = read_boundary_codes(cfg.get("boundary_codes"), vocab)
    dist = BoundaryDistribution.from_config(cfg, boundary_codes)
    logger.info(
        "Vocabulary: %s codes, V=%s (packed width %s bytes), fingerprint %s; %s boundary candidate(s).",
        f"{len(vocab.codes):,}",
        f"{vocab.size:,}",
        vocab.packed_width,
        vocab.fingerprint[:12],
        f"{len(boundary_codes):,}",
    )

    n_subjects = build_prediction_times(
        path_to_data=path_to_data,
        training_task_artifacts_dir=artifacts_dir,
        split=cfg.split,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        overwrite=cfg.overwrite,
    )
    logger.info("Stage 0: %s eligible subject(s) for split=%s.", f"{n_subjects:,}", cfg.split)

    num_contexts = int(cfg.num_training_examples)
    context_rng = np.random.default_rng(derive_seed(cfg.seed, "contexts"))
    form_rng = np.random.default_rng(derive_seed(cfg.seed, "bound_forms"))
    duration_rng = np.random.default_rng(derive_seed(cfg.seed, "bound_durations"))
    code_rng = np.random.default_rng(derive_seed(cfg.seed, "bound_codes"))

    sample = dist.sample(num_contexts, form_rng, duration_rng, code_rng)
    n_event = int((sample.bound_events != None).sum())  # noqa: E711
    logger.info(
        "Stage 1M: sampled %s boundary sequence(s) of K=%d (%s event-bounded slots, %.1f%%; %s durations "
        "over [%g, %g] days).",
        f"{num_contexts:,}",
        dist.num_bounds,
        f"{n_event:,}",
        100.0 * n_event / max(num_contexts * dist.num_bounds, 1),
        dist.duration_distribution,
        dist.min_duration,
        dist.max_duration,
    )

    counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, cfg.split)).sort("subject_id")
    contexts = sample_patient_contexts(
        prediction_time_counts=counts,
        n=num_contexts,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        rng=context_rng,
    )
    logger.info("Stage 2: sampled %s patient context(s).", f"{contexts.height:,}")

    n_shards = build_multitask_index(sample, contexts, artifacts_dir, cfg.split)
    logger.info(
        "Stage 3M: wrote partitioned index for split=%s across %s shard(s).", cfg.split, f"{n_shards:,}"
    )

    # The driver - never a worker - owns the split manifest, written and validated before the pool.
    manifest = write_manifest(out_root / cfg.split, build_manifest(dist, vocab))
    label_multitask_shards(cfg, path_to_data, out_root, artifacts_dir, manifest, num_contexts)


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_multitask_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``EQ_generate_multitask_sequences``)."""
    run(cfg)


if __name__ == "__main__":
    main()
