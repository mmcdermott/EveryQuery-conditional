"""Dataset + batch types for conditional query sequences.

The on-disk label rows follow :class:`~every_query.data.schema.QuerySeqSchema`: one row per
``(subject_id, prediction_time)`` context carrying three aligned list columns —

- ``queries``: list of MEDS code strings (any vocabulary code, in random order — including the
  end-of-timeline code :data:`EOS_CODE`, which is an ordinary code, not a special sentinel);
- ``durations``: list of float horizon lengths (days);
- ``answers``: list of booleans.  ``answers[j]`` is simply *"was ``queries[j]`` observed in
  ``(prediction_time, prediction_time + durations[j]]``?"* — binary, never null.  An event we
  could not observe (because the record ends first) is ``False``; censoring is expressed by a
  separate ``TIMELINE//END`` query rather than a null answer.

The dataset tensorizes each sequence into fixed-position tensors padded to the batch's longest
query list:

- ``q_codes``     (B, L) long — vocab indices; padding is 0;
- ``q_durations`` (B, L) float — horizon in days, 0 at padding;
- ``q_answers``   (B, L) long — teacher-forcing classes ``ANSWER_NO`` / ``ANSWER_YES`` (padding
  holds ``ANSWER_NO``; it is ignored under ``q_mask``);
- ``q_mask``      (B, L) bool — True at real (non-padding) query positions.

Unlike :class:`~every_query.data.dataset.EveryQueryPytorchDataset`, **no query token is prepended
to the patient event sequence** — the patient sequence feeds the bidirectional encoder unchanged,
and queries live exclusively in the decoder stream.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import MEDSTorchBatch

from every_query.model.conditional_model import ANSWER_NO, ANSWER_YES

logger = logging.getLogger(__name__)

# End-of-timeline code: a real MEDS vocabulary code emitted once per subject at the record's last
# timestamp.  A query ``(EOS_CODE, d)`` therefore answers "does the record end within d?" — the
# mechanism by which the conditional model handles censoring (see conditional_model docstring).
EOS_CODE = "TIMELINE//END"

QUERIES_COL = "queries"
DURATIONS_COL = "durations"
ANSWERS_COL = "answers"
SEQ_LABEL_COLS = (QUERIES_COL, DURATIONS_COL, ANSWERS_COL)


@dataclass
class ConditionalQueryBatch(MEDSTorchBatch):
    """MEDS batch extended with per-sample query-sequence tensors.

    Examples:
        >>> batch = ConditionalQueryBatch(
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        ...     numeric_value=torch.zeros(2, 3),
        ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 3),
        ...     q_codes=torch.tensor([[9, 7], [9, 0]]),
        ...     q_durations=torch.tensor([[30.0, 7.0], [365.0, 0.0]]),
        ...     q_answers=torch.tensor([[1, 0], [0, 0]]),
        ...     q_mask=torch.tensor([[True, True], [True, False]]),
        ... )
        >>> batch.n_queries
        2
        >>> batch.q_mask.sum().item()
        3

        Mismatched query-tensor shapes raise:

        >>> ConditionalQueryBatch(
        ...     code=torch.tensor([[1, 2], [3, 4]]),
        ...     numeric_value=torch.zeros(2, 2),
        ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 2),
        ...     q_codes=torch.tensor([[9, 7]]),
        ...     q_durations=torch.tensor([[30.0, 7.0]]),
        ...     q_answers=torch.tensor([[1, 0]]),
        ...     q_mask=torch.tensor([[True, True]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (2, 2) for q_codes, but got torch.Size([1, 2])!
    """

    q_codes: torch.LongTensor | None = None
    q_durations: torch.FloatTensor | None = None
    q_answers: torch.LongTensor | None = None
    q_mask: torch.BoolTensor | None = None

    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = ("boolean_value", "q_codes", "q_durations", "q_answers")

    @property
    def n_queries(self) -> int | None:
        return None if self.q_codes is None else self.q_codes.shape[1]

    def __post_init__(self):
        super().__post_init__()
        if self.q_codes is None:
            return
        expected = (self.batch_size, self.q_codes.shape[1])
        for name in ("q_codes", "q_durations", "q_answers", "q_mask"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != expected:
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {expected} for {name}, but got {got}!")


class ConditionalQueryPytorchDataset(MEDSPytorchDataset):
    """MEDS dataset over query-sequence label parquets (see module docstring for tensor shapes)."""

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Upstream seq-bounds computation + hstack of the query-sequence list columns.

        Mirrors ``EveryQueryPytorchDataset.get_task_seq_bounds_and_labels``: the upstream
        implementation preserves ``label_df`` input order for surviving rows (subjects present in
        ``schema_df``), so the extras can be hstacked from a semi-filtered copy of ``label_df``.
        """
        base = super().get_task_seq_bounds_and_labels(label_df, schema_df)
        extras = [c for c in SEQ_LABEL_COLS if c in label_df.collect_schema().names()]
        if not extras:
            return base
        sid = DataSchema.subject_id_name
        surviving = label_df.join(schema_df.lazy().select(sid).unique().collect(), on=sid, how="semi")
        return base.hstack(surviving.select(extras))

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__(cfg, split)

        schema_cols = self.schema_df.collect_schema().names()
        missing = [c for c in SEQ_LABEL_COLS if c not in schema_cols]
        if missing:
            raise ValueError(
                f"ConditionalQueryPytorchDataset requires query-sequence label columns "
                f"{list(SEQ_LABEL_COLS)}; missing {missing}.  Generate labels with "
                f"EQ_generate_query_sequences."
            )

        self.queries = self.schema_df[QUERIES_COL]
        self.durations = self.schema_df[DURATIONS_COL]
        self.answers = self.schema_df[ANSWERS_COL]

        code_meta = pl.read_parquet(
            self.config.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True
        )
        self.code_to_index: dict[str, int] = {
            c: int(i)
            for c, i in zip(code_meta["code"].to_list(), code_meta["code/vocab_index"].to_list(), strict=True)
        }
        if EOS_CODE not in self.code_to_index:
            raise ValueError(
                f"End-of-timeline code {EOS_CODE!r} is not in the cohort vocabulary; the conditional "
                f"model handles censoring by querying it, so it must be a real code."
            )
        self.eos_query_index: int = self.code_to_index[EOS_CODE]

    @property
    def labels_df(self) -> pl.DataFrame:
        """Task label rows incl. the query-sequence list columns, in MEDS Label schema order."""
        if not self.has_task_index:
            return None

        required = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        def read_df(fp: Path) -> pl.DataFrame:
            available = pq.read_schema(fp).names
            extras = [c for c in SEQ_LABEL_COLS if c in available]
            return pl.read_parquet(fp, columns=[*required, *extras], use_pyarrow=True)

        logger.info(f"Reading query-sequence tasks from {self.config.task_labels_fps}")
        return pl.concat([read_df(fp) for fp in self.config.task_labels_fps], how="vertical")

    def encode_query(self, code_name: str) -> int:
        """Map a query code string to its vocab index.

        Unknown codes raise rather than silently PAD-encode — a query against a code the encoder
        never saw would produce an arbitrary answer.  ``TIMELINE//END`` is an ordinary code here.
        """
        if code_name not in self.code_to_index:
            raise KeyError(f"Query code {code_name!r} is not in the training vocabulary.")
        return self.code_to_index[code_name]

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        out = super()._seeded_getitem(idx, seed)
        out["queries"] = self.queries[idx].to_list()
        out["durations"] = self.durations[idx].to_list()
        out["answers"] = self.answers[idx].to_list()
        return out

    def collate(self, batch: list[dict]) -> ConditionalQueryBatch:
        out = dict(super().collate(batch).items())
        # The base batch type doesn't know about the seq label columns; drop anything
        # super() didn't consume and rebuild the query tensors below.
        out.pop("boolean_value", None)

        max_q = max(len(item["queries"]) for item in batch)
        B = len(batch)

        q_codes = torch.zeros(B, max_q, dtype=torch.long)
        q_durations = torch.zeros(B, max_q, dtype=torch.float)
        q_answers = torch.full((B, max_q), ANSWER_NO, dtype=torch.long)
        q_mask = torch.zeros(B, max_q, dtype=torch.bool)

        for i, item in enumerate(batch):
            codes = [self.encode_query(c) for c in item["queries"]]
            n = len(codes)
            q_codes[i, :n] = torch.as_tensor(codes, dtype=torch.long)
            q_durations[i, :n] = torch.as_tensor(item["durations"], dtype=torch.float)
            q_answers[i, :n] = torch.as_tensor(
                [ANSWER_YES if a else ANSWER_NO for a in item["answers"]],
                dtype=torch.long,
            )
            q_mask[i, :n] = True

        return ConditionalQueryBatch(
            **out, q_codes=q_codes, q_durations=q_durations, q_answers=q_answers, q_mask=q_mask
        )
