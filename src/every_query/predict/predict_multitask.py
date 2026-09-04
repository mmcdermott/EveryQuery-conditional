"""Inference over a ``QuerySeqSchema`` evaluation grid with the multitask model — ``EQ_predict_multitask``.

The evaluation flow is::

    EQ_generate_evaluation_query_sequences -> QuerySeqSchema eval grid -> EQ_predict_multitask

Takes a trained :class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`
run directory and the ``eval/`` root written by ``EQ_generate_evaluation_query_sequences`` (the same
grid ``EQ_predict_sequences`` consumes, plus the explicit window starts only this model can read),
and writes **one scalar prediction per grid row**: the probability that the row's *final* query
occurs in its window, conditioned on the patient and on the earlier queries with their true answers.

Per real grid row the :class:`~every_query.data.multitask_eval_dataset.QuerySeqMultitaskEvalDataset`
adapter maps::

    q_durations        <- durations
    q_bound_codes      <- bound_events        (no-bound index where null / absent)
    q_start_durations  <- start_durations     (0.0 where absent)
    q_start_codes      <- start_events        (no-bound index where null / absent)
    condition_codes    <- queries[:-1]
    condition_answers  <- answers[:-1]
    scored_code        <- queries[-1]
    label              <- answers[-1]

``[:-1]`` / ``[-1]`` are the row's own real queries, never the padded batch width, and the final
query is not teacher-forced into its own prediction.  The model scores only that code at only that
window (``ConditionalMultitaskARModel.score_final_query``): the same hidden state, tied embedding row
and bias the training forward uses, without ever materializing ``(B, K, V)`` logits or dense targets.

Nothing here needs, reads or writes a ``.labels.npy`` sidecar, a multitask manifest, ``eval_meta``
or ``eval_tasks.parquet``.  The checkpoint's cohort / sequence settings (``tensorized_cohort_dir``,
``max_seq_len``, ``seq_sampling_strategy``, ...) are reused from its ``resolved_config.yaml``; its
datamodule is *not* instantiated, since that one expects the training sampler's packed layout.

Output columns, one row per input sequence, in dataset (= dataloader = input) order::

    subject_id, prediction_time,
    queries, start_durations, start_events, durations, bound_events, answers,
    target_code, label, prob

The complete lists identify the conditional task (``target_code == queries[-1]``,
``label == answers[-1]``).  ``start_durations`` / ``start_events`` / ``bound_events`` are always
written, normalized to their defaults (``0.0`` / null / null) when the grid lacked the column.  The
old evaluation-only ``task_id``, ``task_group``, ``start_resolved``, ``end_resolved`` and
``window_days`` sidecar fields are intentionally not recreated.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from meds import held_out_split, tuning_split
from omegaconf import DictConfig  # noqa: TC002 - Hydra resolves this at runtime
from torch.utils.data import DataLoader, SequentialSampler

from every_query.data.multitask_eval_dataset import QuerySeqMultitaskEvalDataset
from every_query.data.seq_dataset import (
    ANSWERS_COL,
    BOUND_EVENTS_COL,
    DURATIONS_COL,
    QUERIES_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
)
from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")

# "train" is disallowed: the grid is an evaluation artifact and scoring must stay sequential.
_ALLOWED_SPLITS = (tuning_split, held_out_split)

OUTPUT_COLUMNS = [
    "subject_id",
    "prediction_time",
    QUERIES_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
    DURATIONS_COL,
    BOUND_EVENTS_COL,
    ANSWERS_COL,
    "target_code",
    "label",
    "prob",
]


def build_eval_dataset(
    train_cfg: DictConfig,
    tasks_dir: Path,
    split: str,
    *,
    expected_vocab_size: int,
    use_rope_time: bool,
    max_windows: int | None = None,
) -> QuerySeqMultitaskEvalDataset:
    """Build the QuerySeq adapter on the checkpoint's cohort settings, pointed at the grid.

    ``train_cfg.datamodule.config`` (a ``MEDSTorchDataConfig``) is instantiated with its
    ``task_labels_dir`` swapped for ``tasks_dir``; everything else - ``tensorized_cohort_dir``,
    ``max_seq_len``, ``seq_sampling_strategy``, ``static_inclusion_mode``, ``batch_mode`` - is
    the checkpoint's.  ``strip_delta_tokens`` is read from the datamodule's ``dataset_kwargs`` and
    must agree with the model's ``use_rope_time``: a mismatch would feed a RoPE-time model
    token-index positions (or vice versa) and score garbage without an error.  The cohort's
    vocabulary width must equal the model's tied-embedding width (``train.py`` sizes the latter from
    the former, and the multitask model has no ontology extension), so a checkpoint pointed at a
    different cohort fails here rather than scoring codes through the wrong embedding rows.
    """
    strip_delta_tokens = bool(train_cfg.datamodule.get("dataset_kwargs", {}).get("strip_delta_tokens", False))
    if strip_delta_tokens != bool(use_rope_time):
        raise ValueError(
            f"the checkpoint's datamodule strips delta tokens={strip_delta_tokens} but its model has "
            f"use_rope_time={use_rope_time}; resolved_config.yaml is inconsistent."
        )
    data_cfg = instantiate(train_cfg.datamodule.config, task_labels_dir=str(tasks_dir))
    if int(data_cfg.vocab_size) != int(expected_vocab_size):
        raise ValueError(
            f"the cohort at {data_cfg.tensorized_cohort_dir} has vocab_size={data_cfg.vocab_size} but the "
            f"checkpoint's tied embedding table is {expected_vocab_size} wide; the model was trained on a "
            "different codes.parquet."
        )
    return QuerySeqMultitaskEvalDataset(
        data_cfg,
        split=split,
        strip_delta_tokens=strip_delta_tokens,
        expected_vocab_size=int(expected_vocab_size),
        max_windows=max_windows,
    )


def build_dataloader(
    dataset: QuerySeqMultitaskEvalDataset, batch_size: int, num_workers: int = 0
) -> DataLoader:
    """A sequential loader: row ``i`` of the output must be grid row ``i``."""
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=int(num_workers),
    )
    if not isinstance(loader.sampler, SequentialSampler):
        raise RuntimeError(
            f"the evaluation loader must use SequentialSampler; got {type(loader.sampler).__name__}"
        )
    return loader


def _to_device(batch, device: torch.device):
    """Move every tensor field of a batch dataclass onto ``device``, in place."""
    for f in fields(batch):
        v = getattr(batch, f.name, None)
        if isinstance(v, torch.Tensor):
            setattr(batch, f.name, v.to(device, non_blocking=True))
    return batch


@torch.no_grad()
def run_inference(
    model: ConditionalMultitaskLightningModule,
    dataloader: DataLoader,
    device: torch.device,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the grid; return ``(probs, labels)``, one float32 / bool per grid row.

    Each batch costs one backbone pass plus ``B`` dot products - nothing of shape ``(B, K, V)``
    exists at any point.  The loader must yield exactly ``n_rows`` rows, or the output could not be
    row-aligned with the grid.
    """
    model = model.to(device).eval()
    probs_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    seen = 0
    for batch in dataloader:
        batch = _to_device(batch, device)
        logits = model.model.score_final_query(batch, batch.scored_codes)
        probs_out.append(torch.sigmoid(logits).float().cpu().numpy())
        labels_out.append(batch.labels.cpu().numpy().astype(bool))
        seen += int(logits.shape[0])
    if seen != n_rows:
        raise RuntimeError(f"dataloader yielded {seen} prediction(s) but the grid has {n_rows} row(s)")
    if not probs_out:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
    return np.concatenate(probs_out).astype(np.float32), np.concatenate(labels_out)


def predictions_to_df(
    dataset: QuerySeqMultitaskEvalDataset, probs: np.ndarray, labels: np.ndarray
) -> pl.DataFrame:
    """One output row per grid row, in dataset order, with the normalized window lists.

    ``labels`` (collated by the loader) must equal ``answers[-1]`` read back from the dataset's own
    rows: the two are produced by different code paths, so disagreement means the loader's row
    order drifted from the dataset's, and the probabilities would be attached to the wrong rows.
    """
    schema_df = dataset.schema_df
    n = schema_df.height
    if probs.shape != (n,) or labels.shape != (n,):
        raise RuntimeError(
            f"got {probs.shape[0]} prediction(s) / {labels.shape[0]} label(s) for {n} grid row(s)"
        )
    queries = schema_df[QUERIES_COL]
    answers = schema_df[ANSWERS_COL]
    durations = schema_df[DURATIONS_COL]
    lengths = queries.list.len()
    if n and (lengths == 0).any():
        raise ValueError("every grid row needs at least one query")

    target_code = queries.list.last()
    label = answers.list.last()
    if n and not np.array_equal(label.to_numpy().astype(bool), labels):
        bad = int(np.flatnonzero(label.to_numpy().astype(bool) != labels)[0])
        raise RuntimeError(
            f"the collated final-query label disagrees with answers[-1] at grid row {bad}; the loader's "
            "row order does not match the dataset's."
        )

    def default_list(fill, dtype: pl.DataType) -> pl.Series:
        return pl.Series([[fill] * int(k) for k in lengths.to_list()], dtype=pl.List(dtype))

    start_durations = schema_df[START_DURATIONS_COL] if dataset.has_starts else default_list(0.0, pl.Float32)
    start_events = schema_df[START_EVENTS_COL] if dataset.has_starts else default_list(None, pl.Utf8)
    bound_events = schema_df[BOUND_EVENTS_COL] if dataset.has_bound_events else default_list(None, pl.Utf8)

    return pl.DataFrame(
        {
            "subject_id": schema_df["subject_id"],
            "prediction_time": schema_df["prediction_time"],
            QUERIES_COL: queries,
            START_DURATIONS_COL: start_durations.cast(pl.List(pl.Float32)).alias(START_DURATIONS_COL),
            START_EVENTS_COL: start_events.cast(pl.List(pl.Utf8)).alias(START_EVENTS_COL),
            DURATIONS_COL: durations.cast(pl.List(pl.Float32)),
            BOUND_EVENTS_COL: bound_events.cast(pl.List(pl.Utf8)).alias(BOUND_EVENTS_COL),
            ANSWERS_COL: answers,
            "target_code": target_code,
            "label": label.cast(pl.Boolean),
            "prob": pl.Series(probs, dtype=pl.Float32),
        }
    ).select(OUTPUT_COLUMNS)


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict_multitask")
def main(cfg: DictConfig) -> None:
    model_run_dir = Path(cfg.model_run_dir)
    tasks_dir = Path(cfg.tasks_dir)
    output_parquet = Path(cfg.output_parquet)
    split = cfg.get("split", held_out_split)
    overwrite = bool(cfg.get("overwrite", False))

    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}.")
    if output_parquet.exists() and not overwrite:
        raise FileExistsError(f"output_parquet {output_parquet} already exists; pass overwrite=true.")
    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"tasks_dir must be the grid's eval/ root, got {tasks_dir}")

    train_cfg, model, _ = setup_model(
        model_run_dir, ckpt_name=cfg.get("ckpt_name"), module_cls=ConditionalMultitaskLightningModule
    )

    dataset = build_eval_dataset(
        train_cfg,
        tasks_dir,
        split,
        expected_vocab_size=model.model.vocab_size,
        use_rope_time=model.model.use_rope_time,
        max_windows=model.model.max_windows,
    )
    logger.info(f"Loaded {len(dataset)} grid rows from {tasks_dir} (split={split})")

    batch_size = cfg.get("batch_size")
    if batch_size is None:
        batch_size = int(train_cfg.datamodule.batch_size)
    num_workers = cfg.get("num_workers")
    if num_workers is None:
        num_workers = train_cfg.datamodule.get("num_workers", 0) or 0
    dataloader = build_dataloader(dataset, int(batch_size), num_workers=int(num_workers))

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Scoring the final query of {len(dataset)} rows on {device}")
    probs, labels = run_inference(model, dataloader, device, len(dataset))

    out = predictions_to_df(dataset, probs, labels)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} final-query predictions to {output_parquet}")


if __name__ == "__main__":
    main()
