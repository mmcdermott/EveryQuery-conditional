"""Dataset + batch type for the all-vocabulary multi-bound multitask labels (issue #20).

On disk, ``EQ_generate_multitask_sequences`` writes per event shard of a split::

    {task_labels_dir}/{split}/{shard}.parquet             MultitaskBoundarySchema metadata rows
    {task_labels_dir}/{split}/{shard}.labels.npy          uint8 (rows, K, ceil(V/8)), bitorder little
    {task_labels_dir}/{split}/_multitask_manifest.json    vocabulary + semantics the bits were built under

The parquet and the ``.npy`` are row-aligned; that alignment **is** the contract.  This dataset

1. loads only the metadata parquets at init (never the packed labels);
2. opens each ``.labels.npy`` with ``mmap_mode="r"`` (read-only, lazily, per process);
3. tags every metadata row with its *source shard key* and *physical source row* while reading the
   parquet and carries both through the upstream task-sequence join, so the sidecar row is looked up
   explicitly rather than inferred from a concatenation order;
4. gathers packed rows in :meth:`MultitaskBoundaryPytorchDataset.collate` and unpacks **once per batch**
   with ``bitorder="little"``; ``__getitem__`` never unpacks;
5. keeps targets boolean until the loss casts them;
6. loads the ``K-1`` sampler-materialized conditioning codes/answers (issue #22) and, in ``collate``,
   checks every stored answer against the unpacked target bit.  It never samples either.

It fails loudly at init when the cohort's ``codes.parquet`` (or an explicit ``expected_vocab_size`` /
``expected_vocab_fingerprint``) disagrees with the manifest.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import MEDSTorchBatch

from every_query.data.seq_dataset import EVENT_BOUND_DURATION_SENTINEL, NO_BOUND_INDEX

logger = logging.getLogger(__name__)

DURATIONS_COL = "durations"
BOUND_EVENTS_COL = "bound_events"
CONDITION_CODES_COL = "condition_codes"
CONDITION_ANSWERS_COL = "condition_answers"
MULTITASK_LABEL_COLS = (DURATIONS_COL, BOUND_EVENTS_COL, CONDITION_CODES_COL, CONDITION_ANSWERS_COL)

SOURCE_SHARD_COL = "_source_shard"
SOURCE_ROW_COL = "_source_row"
_INTERNAL_COLS = (SOURCE_SHARD_COL, SOURCE_ROW_COL)

MANIFEST_NAME = "_multitask_manifest.json"
LABELS_SUFFIX = ".labels.npy"
BITORDER = "little"


@dataclass
class MultitaskBoundaryBatch(MEDSTorchBatch):
    """MEDS batch extended with ``K`` boundaries per sample and all-vocabulary boolean targets.

    Attributes:
        q_durations: ``(B, K)`` float - horizon in days; ``EVENT_BOUND_DURATION_SENTINEL`` at
            event-bounded slots.
        q_bound_codes: ``(B, K)`` long - boundary vocabulary index; ``NO_BOUND_INDEX`` (0) for a
            duration-bounded slot.
        q_mask: ``(B, K)`` bool - True at real slots (all True: ``K`` is fixed; kept for the model
            contract).
        targets: ``(B, K, V)`` bool - ``targets[b, k, v]`` is "code ``v`` occurs strictly inside the
            open window of boundary ``k``".  Bit ``0`` (PAD) is always False and must be masked from
            the loss.
        condition_codes: ``(B, K-1)`` long - vocabulary index of the sampler-drawn conditioning code
            for boundaries ``0..K-2`` (never PAD).
        condition_answers: ``(B, K-1)`` bool - ``targets[b, j, condition_codes[b, j]]``, materialized
            by the sampler and verified in ``collate``.

    Examples:
        >>> batch = MultitaskBoundaryBatch(
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        ...     numeric_value=torch.zeros(2, 3),
        ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 3),
        ...     q_durations=torch.tensor([[30.0, -1.0], [7.0, 2.0]]),
        ...     q_bound_codes=torch.tensor([[0, 4], [0, 0]]),
        ...     q_mask=torch.ones(2, 2, dtype=torch.bool),
        ...     targets=torch.zeros(2, 2, 6, dtype=torch.bool),
        ...     condition_codes=torch.tensor([[3], [5]]),
        ...     condition_answers=torch.zeros(2, 1, dtype=torch.bool),
        ... )
        >>> batch.num_bounds, batch.vocab_size
        (2, 6)

        Mismatched shapes raise:

        >>> MultitaskBoundaryBatch(
        ...     code=torch.tensor([[1, 2], [3, 4]]),
        ...     numeric_value=torch.zeros(2, 2),
        ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 2),
        ...     q_durations=torch.tensor([[30.0, -1.0], [7.0, 2.0]]),
        ...     q_bound_codes=torch.tensor([[0, 4], [0, 0]]),
        ...     q_mask=torch.ones(2, 2, dtype=torch.bool),
        ...     targets=torch.zeros(2, 3, 6, dtype=torch.bool),
        ...     condition_codes=torch.tensor([[3], [5]]),
        ...     condition_answers=torch.zeros(2, 1, dtype=torch.bool),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected targets to have shape (2, 2, V), but got torch.Size([2, 3, 6])!
    """

    q_durations: torch.FloatTensor | None = None
    q_bound_codes: torch.LongTensor | None = None
    q_mask: torch.BoolTensor | None = None
    targets: torch.BoolTensor | None = None
    condition_codes: torch.LongTensor | None = None
    condition_answers: torch.BoolTensor | None = None
    # Per-patient-token elapsed hours for rotary time encoding; ``None`` unless the dataset strips
    # delta tokens (mirrors ``ConditionalQueryBatch.time_pos_ids``).
    time_pos_ids: torch.LongTensor | None = None

    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = (
        "boolean_value",
        "q_durations",
        "q_bound_codes",
        "targets",
        "condition_codes",
        "condition_answers",
    )

    @property
    def num_bounds(self) -> int | None:
        return None if self.q_durations is None else self.q_durations.shape[1]

    @property
    def vocab_size(self) -> int | None:
        return None if self.targets is None else self.targets.shape[-1]

    def __post_init__(self):
        super().__post_init__()
        if self.q_durations is None:
            return
        expected = (self.batch_size, self.q_durations.shape[1])
        for name in ("q_durations", "q_bound_codes", "q_mask"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != expected:
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {expected} for {name}, but got {got}!")
        if self.targets is None or self.targets.dim() != 3 or tuple(self.targets.shape[:2]) != expected:
            got = None if self.targets is None else self.targets.shape
            raise ValueError(
                f"Expected targets to have shape {(*expected, 'V')}, but got {got}!".replace("'", "")
            )
        if self.targets.dtype != torch.bool:
            raise TypeError(f"targets must be boolean, got {self.targets.dtype}")
        cond_shape = (self.batch_size, self.q_durations.shape[1] - 1)
        for name in ("condition_codes", "condition_answers"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != cond_shape:
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {cond_shape} for {name}, but got {got}!")
        if self.condition_answers.dtype != torch.bool:
            raise TypeError(f"condition_answers must be boolean, got {self.condition_answers.dtype}")


def read_manifest(split_dir: Path) -> dict:
    fp = split_dir / MANIFEST_NAME
    if not fp.exists():
        raise FileNotFoundError(
            f"No multitask manifest at {fp}. Generate labels with EQ_generate_multitask_sequences; the "
            "driver writes the manifest before any shard is labeled."
        )
    manifest = json.loads(fp.read_text())
    for key in (
        "num_bounds",
        "vocab_size",
        "packed_width_bytes",
        "bitorder",
        "vocab_fingerprint",
        "ontology_mode",
    ):
        if key not in manifest:
            raise ValueError(f"multitask manifest {fp} is missing {key!r}")
    if manifest["bitorder"] != BITORDER:
        raise ValueError(f"multitask manifest bitorder must be {BITORDER!r}, got {manifest['bitorder']!r}")
    return manifest


def _vocab_fingerprint_from_codes_parquet(fp: Path) -> tuple[int, str]:
    """``(V, fingerprint)`` of a cohort ``codes.parquet``, computed exactly as the sampler does."""
    from every_query.generate_tasks.sample_multitask_sequences import build_target_vocabulary

    vocab = build_target_vocabulary(fp)
    return vocab.size, vocab.fingerprint


class MultitaskBoundaryPytorchDataset(MEDSPytorchDataset):
    """MEDS dataset over multitask-boundary metadata parquets + packed ``.labels.npy`` sidecars."""

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Upstream seq-bounds computation + hstack of the boundary columns and the source keys.

        The upstream implementation preserves ``label_df`` input order for surviving rows (subjects
        present in ``schema_df``), so the extras are hstacked from a semi-filtered copy.  The two
        internal source columns ride along here: they are what lets ``collate`` find the packed row.
        """
        base = super().get_task_seq_bounds_and_labels(label_df, schema_df)
        extras = [
            c for c in (*MULTITASK_LABEL_COLS, *_INTERNAL_COLS) if c in label_df.collect_schema().names()
        ]
        if not extras:
            return base
        sid = DataSchema.subject_id_name
        surviving = label_df.join(schema_df.lazy().select(sid).unique().collect(), on=sid, how="semi")
        return base.hstack(surviving.select(extras))

    def __init__(
        self,
        cfg: MEDSTorchDataConfig,
        split: str,
        *,
        expected_vocab_size: int | None = None,
        expected_vocab_fingerprint: str | None = None,
        check_cohort_vocabulary: bool = True,
        strip_delta_tokens: bool = False,
    ):
        """Build the dataset.

        Args:
            cfg: Upstream MEDS torchdata config; ``task_labels_dir`` must point at the sampler's
                ``out_dir`` (the split subdirectory holds the manifest and sidecars).
            split: MEDS split name.
            expected_vocab_size: The model's output width; must equal the manifest's ``vocab_size``.
            expected_vocab_fingerprint: The vocabulary fingerprint the model was built against; must
                equal the manifest's.
            check_cohort_vocabulary: Also recompute the fingerprint from the cohort's
                ``codes.parquet`` (``cfg.code_metadata_fp``) and require it to match the manifest.
            strip_delta_tokens: As in :class:`~every_query.data.seq_dataset.ConditionalQueryPytorchDataset`.
        """
        self._split_dir = Path(cfg.task_labels_dir) / split if cfg.task_labels_dir is not None else None
        if self._split_dir is None:
            raise ValueError("MultitaskBoundaryPytorchDataset requires task_labels_dir")
        self.manifest = read_manifest(self._split_dir)
        self.num_bounds = int(self.manifest["num_bounds"])
        self.vocab_size = int(self.manifest["vocab_size"])
        self.packed_width = int(self.manifest["packed_width_bytes"])
        self.vocab_fingerprint = str(self.manifest["vocab_fingerprint"])
        if expected_vocab_size is not None and expected_vocab_size != self.vocab_size:
            raise ValueError(
                f"vocab_size mismatch: the model expects {expected_vocab_size} but the multitask manifest at "
                f"{self._split_dir} was built with V={self.vocab_size}. Regenerate the labels or fix the "
                "model."
            )
        if expected_vocab_fingerprint is not None and expected_vocab_fingerprint != self.vocab_fingerprint:
            raise ValueError(
                f"vocabulary fingerprint mismatch: expected {expected_vocab_fingerprint[:12]}... but the "
                f"manifest records {self.vocab_fingerprint[:12]}.... The labels were built against a "
                "different "
                "codes.parquet."
            )
        if check_cohort_vocabulary:
            v, fp = _vocab_fingerprint_from_codes_parquet(cfg.code_metadata_fp)
            if v != self.vocab_size or fp != self.vocab_fingerprint:
                raise ValueError(
                    f"The cohort vocabulary at {cfg.code_metadata_fp} (V={v}, {fp[:12]}...) does not match "
                    f"the multitask manifest (V={self.vocab_size}, {self.vocab_fingerprint[:12]}...). The "
                    "labels were "
                    "generated against a different codes.parquet; regenerate them for this cohort."
                )

        self._label_files: dict[str, Path] = {}
        self._memmaps: dict[str, np.ndarray] = {}

        super().__init__(cfg, split)

        schema_cols = self.schema_df.collect_schema().names()
        missing = [c for c in (*MULTITASK_LABEL_COLS, *_INTERNAL_COLS) if c not in schema_cols]
        if missing:
            raise ValueError(
                f"MultitaskBoundaryPytorchDataset requires columns {[*MULTITASK_LABEL_COLS]} (plus internal "
                f"source keys); missing {missing}. Generate labels with EQ_generate_multitask_sequences."
            )

        code_meta = pl.read_parquet(
            cfg.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True
        )
        self.code_to_index: dict[str, int] = {
            c: int(i)
            for c, i in zip(code_meta["code"].to_list(), code_meta["code/vocab_index"].to_list(), strict=True)
        }

        n = self.schema_df.height
        durations = self.schema_df[DURATIONS_COL]
        bound_events = self.schema_df[BOUND_EVENTS_COL]
        if (
            n
            and not (
                (durations.list.len() == self.num_bounds) & (bound_events.list.len() == self.num_bounds)
            ).all()
        ):
            raise ValueError(f"every label row must carry exactly {self.num_bounds} durations/bound_events")
        self._q_durations = (
            np.asarray(durations.explode().to_numpy(), dtype=np.float32).reshape(n, self.num_bounds)
            if n
            else np.zeros((0, self.num_bounds), dtype=np.float32)
        )
        # Map codes -> indices in polars (no per-slot Python objects): null slots -> NO_BOUND_INDEX,
        # non-null codes outside the vocabulary -> -1, which is an error.
        flat_bounds = bound_events.explode()
        mapped = pl.select(
            pl.when(flat_bounds.is_null())
            .then(NO_BOUND_INDEX)
            .otherwise(flat_bounds.replace_strict(self.code_to_index, default=-1, return_dtype=pl.Int64))
        ).to_series()
        unknown = flat_bounds.filter(mapped == -1).unique().sort().to_list()
        if unknown:
            raise ValueError(
                f"{len(unknown)} boundary code(s) in the labels are not in this cohort's vocabulary: "
                f"{unknown[:10]}. Regenerate the labels against this cohort's codes.parquet."
            )
        self._q_bound_codes = mapped.to_numpy().astype(np.int64).reshape(n, self.num_bounds)
        sentinel_ok = (self._q_bound_codes != NO_BOUND_INDEX) == (
            self._q_durations == EVENT_BOUND_DURATION_SENTINEL
        )
        if not sentinel_ok.all():
            raise ValueError("durations/bound_events disagree on which slots are event-bounded")

        # Issue #22: K-1 sampler-materialized conditioning codes/answers. Loaded and validated only;
        # never sampled here. Answers are cross-checked against the unpacked bits in ``collate``.
        kc = self.num_bounds - 1
        cond_codes = self.schema_df[CONDITION_CODES_COL]
        cond_answers = self.schema_df[CONDITION_ANSWERS_COL]
        if n and not ((cond_codes.list.len() == kc) & (cond_answers.list.len() == kc)).all():
            raise ValueError(f"every label row must carry exactly {kc} condition_codes/condition_answers")
        if n and kc:
            flat_cond = cond_codes.explode()
            cond_mapped = flat_cond.replace_strict(self.code_to_index, default=-1, return_dtype=pl.Int64)
            bad = flat_cond.filter(cond_mapped.is_null() | (cond_mapped <= 0)).unique().sort().to_list()
            if bad:
                raise ValueError(
                    f"{len(bad)} condition code(s) in the labels are null, PAD, or not in this cohort's "
                    f"vocabulary: {bad[:10]}. Regenerate the labels against this cohort's codes.parquet."
                )
            self._condition_codes = cond_mapped.to_numpy().astype(np.int64).reshape(n, kc)
            flat_ans = cond_answers.explode()
            if flat_ans.null_count():
                raise ValueError("condition_answers must not contain nulls")
            self._condition_answers = flat_ans.to_numpy().astype(bool).reshape(n, kc)
        else:
            self._condition_codes = np.zeros((n, kc), dtype=np.int64)
            self._condition_answers = np.zeros((n, kc), dtype=bool)

        # Shard keys are stored once; each row carries a compact integer id, not a Python string.
        self._shard_keys: list[str] = sorted(self._label_files)
        self._source_shard = (
            self.schema_df[SOURCE_SHARD_COL]
            .replace_strict({k: i for i, k in enumerate(self._shard_keys)}, return_dtype=pl.Int32)
            .to_numpy()
        )
        self._source_row = self.schema_df[SOURCE_ROW_COL].to_numpy().astype(np.int64)

        # Every referenced sidecar must exist and have the manifest's packed shape; check the header
        # only (np.load with mmap_mode reads no payload).
        for shard_id in np.unique(self._source_shard).tolist():
            key = self._shard_keys[shard_id]
            fp = self._label_files[key]
            if not fp.exists():
                raise FileNotFoundError(f"packed labels sidecar {fp} for metadata shard {key!r} is missing")
            shape = tuple(np.load(fp, mmap_mode="r").shape)
            if shape[1:] != (self.num_bounds, self.packed_width):
                raise ValueError(
                    f"{fp} has packed shape {shape}, expected (rows, {self.num_bounds}, {self.packed_width})"
                )
            rows = self._source_row[self._source_shard == shard_id]
            if rows.size and int(rows.max()) >= shape[0]:
                raise ValueError(
                    f"{fp} has {shape[0]} rows but the metadata references row {int(rows.max())}"
                )

        self.strip_delta_tokens = strip_delta_tokens
        from every_query.data import rope_time

        self.delta_ids = rope_time.delta_vocab_ids(self.code_to_index)

    # -- label loading ---------------------------------------------------------------------------

    @property
    def labels_df(self) -> pl.DataFrame:
        """Metadata rows only, each tagged with its source shard key and physical row offset."""
        if not self.has_task_index:
            return None
        required = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        frames = []
        # Only this split's files: the manifest and sidecars live in the split subdirectory, and the
        # upstream rglob would otherwise also read every other split's metadata just to drop it.
        fps = [fp for fp in self.config.task_labels_fps if self._split_dir in fp.parents]
        if not fps:
            raise FileNotFoundError(f"no metadata parquets under {self._split_dir}")
        for fp in fps:
            key = self._source_key(fp)
            self._label_files[key] = fp.with_name(fp.name[: -len(fp.suffix)] + LABELS_SUFFIX)
            missing = [c for c in (*required, *MULTITASK_LABEL_COLS) if c not in pl.read_parquet_schema(fp)]
            if missing:
                raise ValueError(
                    f"{fp} is missing required column(s) {missing}; regenerate the labels with the "
                    "current EQ_generate_multitask_sequences."
                )
            df = pl.read_parquet(fp, columns=[*required, *MULTITASK_LABEL_COLS], use_pyarrow=True)
            frames.append(
                df.with_row_index(SOURCE_ROW_COL).with_columns(
                    pl.col(SOURCE_ROW_COL).cast(pl.Int64),
                    pl.lit(key).alias(SOURCE_SHARD_COL),
                )
            )
        logger.info(f"Reading multitask boundary metadata from {fps}")
        return pl.concat(frames, how="vertical")

    def _source_key(self, fp: Path) -> str:
        """The metadata parquet's identity: its path relative to ``task_labels_dir``, minus the suffix.

        Relative to the labels root (not the split dir) so ``train/0`` and ``tuning/0`` never collide.
        """
        return str(fp.relative_to(Path(self.config.task_labels_dir)).with_suffix(""))

    def _memmap(self, key: str) -> np.ndarray:
        mm = self._memmaps.get(key)
        if mm is None:
            mm = np.load(self._label_files[key], mmap_mode="r")
            self._memmaps[key] = mm
        return mm

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_memmaps"] = {}  # DataLoader workers reopen their own read-only maps
        return state

    # -- items + collate -------------------------------------------------------------------------

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        out = super()._seeded_getitem(idx, seed)
        idx = range(len(self._source_row))[idx]
        out["q_durations"] = self._q_durations[idx]
        out["q_bound_codes"] = self._q_bound_codes[idx]
        out["condition_codes"] = self._condition_codes[idx]
        out["condition_answers"] = self._condition_answers[idx]
        out[SOURCE_SHARD_COL] = self._shard_keys[self._source_shard[idx]]
        out[SOURCE_ROW_COL] = int(self._source_row[idx])
        return out

    def gather_packed(self, batch: list[dict]) -> np.ndarray:
        """``(B, K, ceil(V/8))`` uint8 rows gathered from the read-only memmaps, one slice per shard."""
        B = len(batch)
        packed = np.empty((B, self.num_bounds, self.packed_width), dtype=np.uint8)
        keys = np.asarray([item[SOURCE_SHARD_COL] for item in batch], dtype=object)
        rows = np.asarray([item[SOURCE_ROW_COL] for item in batch], dtype=np.int64)
        for key in dict.fromkeys(keys.tolist()):
            sel = np.flatnonzero(keys == key)
            packed[sel] = self._memmap(key)[rows[sel]]
        return packed

    def unpack_targets(self, packed: np.ndarray) -> torch.BoolTensor:
        dense = np.unpackbits(packed, axis=-1, count=self.vocab_size, bitorder=BITORDER)
        return torch.from_numpy(dense.astype(bool, copy=False))

    def collate(self, batch: list[dict]) -> MultitaskBoundaryBatch:
        out = dict(super().collate(batch).items())
        out.pop("boolean_value", None)

        time_pos_ids = None
        if self.strip_delta_tokens:
            from every_query.data.seq_dataset import ConditionalQueryPytorchDataset

            time_pos_ids = ConditionalQueryPytorchDataset._apply_rope_time(self, out)

        B = len(batch)
        q_durations = torch.from_numpy(np.stack([item["q_durations"] for item in batch]).astype(np.float32))
        q_bound_codes = torch.from_numpy(np.stack([item["q_bound_codes"] for item in batch]).astype(np.int64))
        q_mask = torch.ones(B, self.num_bounds, dtype=torch.bool)
        targets = self.unpack_targets(self.gather_packed(batch))
        condition_codes = torch.from_numpy(
            np.stack([item["condition_codes"] for item in batch]).astype(np.int64)
        )
        condition_answers = torch.from_numpy(np.stack([item["condition_answers"] for item in batch]))

        # The stored answer must be the target bit of its code at its boundary (the sampler's contract).
        kc = self.num_bounds - 1
        if kc:
            expect = targets[:, :kc].gather(2, condition_codes.unsqueeze(-1)).squeeze(-1)
            if not torch.equal(expect, condition_answers):
                bad = (expect != condition_answers).nonzero()[:5].tolist()
                raise ValueError(
                    f"condition_answers disagree with the packed targets at (batch, slot) {bad}; the "
                    "metadata and .labels.npy sidecar are inconsistent. Regenerate the labels."
                )

        return MultitaskBoundaryBatch(
            **out,
            q_durations=q_durations,
            q_bound_codes=q_bound_codes,
            q_mask=q_mask,
            targets=targets,
            condition_codes=condition_codes,
            condition_answers=condition_answers,
            time_pos_ids=time_pos_ids,
        )
