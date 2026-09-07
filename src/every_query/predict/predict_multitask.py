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
or ``eval_tasks.parquet``.  The prediction datamodule
(:class:`~every_query.data.conditional_multitask_datamodule.ConditionalMultitaskDataModule`) is
built from the checkpoint's cohort / sequence settings (``tensorized_cohort_dir``, ``max_seq_len``,
``seq_sampling_strategy``, ...) as recorded in its ``resolved_config.yaml``, with only the label
root swapped for the grid's ``eval/`` root; the training labels are never read, and the run
directory's own datamodule node is never instantiated (its ``config`` names the training labels
root, which ``MEDSTorchDataConfig`` requires to exist and which need not be on the inference
machine; older nodes also have no predict loader).  The Lightning predict loop owns device placement
and batching: ``Trainer.predict`` moves each batch to the accelerator and calls
:meth:`~every_query.model.conditional_multitask_lightning.ConditionalMultitaskLightningModule.predict_step`,
which scores through ``score_final_query`` and never touches the dense forward or ``batch.targets``.

Prediction is **single-device** by construction.  Row ``i`` of the output is grid row ``i`` because
the loader is sequential and the per-batch outputs are concatenated in loader order; a multi-device
or distributed strategy would shard and interleave rows across ranks and break that alignment, so
the inference trainer is pinned to one device (``device`` on the command line picks which) and
refuses anything else (issue #30 non-goal).

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
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import SingleDeviceStrategy
from meds import held_out_split
from omegaconf import DictConfig  # noqa: TC002 - Hydra resolves this at runtime

from every_query.data.conditional_multitask_datamodule import EVAL_SPLITS, ConditionalMultitaskDataModule
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

if TYPE_CHECKING:
    from every_query.data.multitask_eval_dataset import QuerySeqMultitaskEvalDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")

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


# --- data ------------------------------------------------------------------------------------------


def build_predict_datamodule(
    train_cfg: DictConfig,
    tasks_dir: Path,
    split: str,
    *,
    expected_vocab_size: int,
    use_rope_time: bool,
    max_windows: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> ConditionalMultitaskDataModule:
    """The prediction datamodule: the checkpoint's cohort settings, pointed at the grid.

    Reads ``train_cfg.datamodule`` - the ``datamodule`` node of the run's ``resolved_config.yaml`` -
    for its ``config`` (a ``MEDSTorchDataConfig``), ``dataset_kwargs``, ``batch_size``,
    ``num_workers`` and ``pin_memory``.  Only those keys are read, never the node's ``_target_``, so
    a run trained before issue #30 (whose node names ``ResumableDatamodule`` + a ``data_class``) and
    a current one (``ConditionalMultitaskDataModule`` with ``eval_tasks_dir: null``) are handled
    alike; the node itself is deliberately not instantiated (its ``config`` would require the training
    labels root to exist on this machine, and the former has no predict loader at all).

    ``config`` is instantiated with its ``task_labels_dir`` replaced by ``tasks_dir``: the training
    labels root recorded in the checkpoint need not exist on the machine running inference, and
    nothing here reads it.  Everything else - ``tensorized_cohort_dir``, ``max_seq_len``,
    ``seq_sampling_strategy``, ``static_inclusion_mode``, ``batch_mode`` - is the checkpoint's.

    Two consistency checks guard the checkpoint against the cohort.  ``strip_delta_tokens`` (from
    ``dataset_kwargs``) must agree with the model's ``use_rope_time``: a mismatch would feed a
    RoPE-time model token-index positions (or vice versa) and score garbage without an error.  The
    cohort's vocabulary width must equal the model's tied-embedding width (``train.py`` sizes the
    latter from the former, and the multitask model has no ontology extension), so a checkpoint
    pointed at a different cohort fails here rather than scoring codes through the wrong embedding
    rows.

    Args:
        train_cfg: The OmegaConf of the run's ``resolved_config.yaml`` (``setup_model`` returns it).
        tasks_dir: The grid's ``eval/`` root.
        split: The grid split to score; one of ``EVAL_SPLITS`` (the datamodule refuses ``train``).
        expected_vocab_size: The model's tied-embedding width (``model.vocab_size``).
        use_rope_time: The model's ``use_rope_time``.
        max_windows: The model's window budget, forwarded so an over-long grid row is rejected when
            the dataset is built rather than inside a forward pass.
        batch_size: Overrides the checkpoint's ``datamodule.batch_size`` when given.
        num_workers: Overrides the checkpoint's ``datamodule.num_workers`` when given.
    """
    dm_cfg = train_cfg.datamodule
    dataset_kwargs = dm_cfg.get("dataset_kwargs") or {}
    strip_delta_tokens = bool(dataset_kwargs.get("strip_delta_tokens", False))
    if strip_delta_tokens != bool(use_rope_time):
        raise ValueError(
            f"the checkpoint's datamodule strips delta tokens={strip_delta_tokens} but its model has "
            f"use_rope_time={use_rope_time}; resolved_config.yaml is inconsistent."
        )
    data_cfg = instantiate(dm_cfg.config, task_labels_dir=str(tasks_dir))
    if int(data_cfg.vocab_size) != int(expected_vocab_size):
        raise ValueError(
            f"the cohort at {data_cfg.tensorized_cohort_dir} has vocab_size={data_cfg.vocab_size} but the "
            f"checkpoint's tied embedding table is {expected_vocab_size} wide; the model was trained on a "
            "different codes.parquet."
        )
    if batch_size is None:
        batch_size = dm_cfg.batch_size
    if num_workers is None:
        num_workers = dm_cfg.get("num_workers", 0) or 0
    return ConditionalMultitaskDataModule(
        data_cfg,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=dm_cfg.get("pin_memory"),
        dataset_kwargs={
            "strip_delta_tokens": strip_delta_tokens,
            "expected_vocab_size": int(expected_vocab_size),
        },
        eval_tasks_dir=tasks_dir,
        max_windows=max_windows,
        predict_split=split,
    )


def build_eval_dataset(
    train_cfg: DictConfig,
    tasks_dir: Path,
    split: str,
    *,
    expected_vocab_size: int,
    use_rope_time: bool,
    max_windows: int | None = None,
) -> QuerySeqMultitaskEvalDataset:
    """The prediction datamodule's ``predict_dataset``; see :func:`build_predict_datamodule`."""
    return build_predict_datamodule(
        train_cfg,
        tasks_dir,
        split,
        expected_vocab_size=expected_vocab_size,
        use_rope_time=use_rope_time,
        max_windows=max_windows,
    ).predict_dataset


# --- trainer ---------------------------------------------------------------------------------------


def resolve_accelerator(device: str | None) -> tuple[str, int | list[int]]:
    """Map the CLI ``device`` string onto Lightning's ``(accelerator, devices)`` for exactly one device.

    ``None`` lets Lightning pick (``"auto"``, one device: the first GPU when there is one, else the
    CPU).  A torch device string names the accelerator - ``cpu``, ``cuda`` (the default GPU),
    ``cuda:N`` (GPU ``N``) or ``mps`` - and always resolves to a single device.

    Examples:
        >>> resolve_accelerator(None)
        ('auto', 1)
        >>> resolve_accelerator("cpu"), resolve_accelerator("cuda"), resolve_accelerator("mps")
        (('cpu', 1), ('gpu', 1), ('mps', 1))
        >>> resolve_accelerator("cuda:2")
        ('gpu', [2])
        >>> resolve_accelerator("xpu")
        Traceback (most recent call last):
            ...
        ValueError: device must be null, cpu, cuda, cuda:N or mps; got 'xpu'
    """
    if device is None:
        return "auto", 1
    try:
        parsed = torch.device(str(device))
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"device must be null, cpu, cuda, cuda:N or mps; got {device!r}") from e
    match parsed.type:
        case "cpu":
            return "cpu", 1
        case "cuda":
            return "gpu", (1 if parsed.index is None else [int(parsed.index)])
        case "mps":
            return "mps", 1
        case _:
            raise ValueError(f"device must be null, cpu, cuda, cuda:N or mps; got {device!r}")


def check_single_device(trainer: Trainer) -> None:
    """Refuse a trainer that would run prediction on more than one device or process.

    Raises:
        ValueError: If the trainer's strategy is not a ``SingleDeviceStrategy``, or it spans more
            than one device, node or process.
    """
    single = (
        isinstance(trainer.strategy, SingleDeviceStrategy)
        and trainer.num_devices == 1
        and trainer.num_nodes == 1
        and trainer.world_size == 1
    )
    if not single:
        raise ValueError(
            "EQ_predict_multitask prediction must run on exactly one device: the inference trainer "
            f"resolved to strategy={type(trainer.strategy).__name__}, num_devices={trainer.num_devices}, "
            f"num_nodes={trainer.num_nodes}, world_size={trainer.world_size}. Outputs are concatenated in "
            "loader order and must stay row-aligned with dataset.schema_df; multi-device / distributed "
            "prediction would shard rows across ranks and is not supported (issue #30 non-goal). Pass "
            "device=cpu, device=cuda:N or device=mps (one device) and do not launch through a "
            "multi-process launcher (torchrun / srun --ntasks>1) that would set up a distributed world."
        )


DEFAULT_PREDICT_PRECISION = "bf16-mixed"


def build_predict_trainer(
    device: str | None = None,
    *,
    precision: str = DEFAULT_PREDICT_PRECISION,
    enable_progress_bar: bool = True,
) -> Trainer:
    """A single-device inference ``Trainer``: no logger, no checkpointing, inference mode.

    This is not the training trainer of the run's ``resolved_config.yaml`` (callbacks, checkpoint
    directories, a logger); it is built from scratch on the device :func:`resolve_accelerator` maps
    ``device`` to.  The single-device guard (:func:`check_single_device`) lives in
    :func:`run_inference`, which applies it to whatever trainer it is handed.

    ``precision`` is a Lightning precision string and defaults to ``bf16-mixed``, the production
    training precision, so the backbone pass runs under the same autocast the checkpoint was trained
    with.  Pass ``32-true`` for full-precision scoring (e.g. to compare against a hand-computed
    ``score_final_query``); the two agree to bf16 rounding (~1e-2 on the probabilities), not exactly.
    """
    accelerator, devices = resolve_accelerator(device)
    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
        strategy="auto",
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(enable_progress_bar),
        enable_model_summary=False,
        inference_mode=True,
    )


# --- inference -------------------------------------------------------------------------------------


def run_inference(
    model: ConditionalMultitaskLightningModule,
    datamodule: ConditionalMultitaskDataModule,
    trainer: Trainer,
) -> tuple[np.ndarray, np.ndarray]:
    """``trainer.predict`` over the datamodule's predict loader; ``(probs, labels)``, one per grid row.

    The Lightning predict loop moves each batch to the trainer's device and collects the
    ``predict_step`` dictionaries (``probs`` / ``labels`` / ``scored_codes``, CPU tensors of shape
    ``(B,)``), which are concatenated here in loader order into a float32 and a bool array.  Each
    batch costs one backbone pass plus ``B`` dot products - nothing of shape ``(B, K, V)`` exists at
    any point.  The loader must yield exactly ``len(datamodule.predict_dataset)`` rows, or the output
    could not be row-aligned with the grid.

    Raises:
        ValueError: If ``trainer`` is not single-device (:func:`check_single_device`).
        RuntimeError: If the loader yielded a different number of rows than the grid has.
    """
    check_single_device(trainer)
    n_rows = len(datamodule.predict_dataset)
    outputs = trainer.predict(model, datamodule=datamodule, return_predictions=True) or []
    probs_out = [out["probs"] for out in outputs]
    labels_out = [out["labels"] for out in outputs]
    seen = sum(int(p.shape[0]) for p in probs_out)
    if seen != n_rows:
        raise RuntimeError(f"dataloader yielded {seen} prediction(s) but the grid has {n_rows} row(s)")
    if not probs_out:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
    probs = torch.cat(probs_out).float().cpu().numpy().astype(np.float32, copy=False)
    labels = torch.cat(labels_out).cpu().numpy().astype(bool)
    return probs, labels


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

    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(EVAL_SPLITS)}, got {split!r}.")
    if output_parquet.exists() and not overwrite:
        raise FileExistsError(f"output_parquet {output_parquet} already exists; pass overwrite=true.")
    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"tasks_dir must be the grid's eval/ root, got {tasks_dir}")

    train_cfg, model, _ = setup_model(
        model_run_dir, ckpt_name=cfg.get("ckpt_name"), module_cls=ConditionalMultitaskLightningModule
    )

    datamodule = build_predict_datamodule(
        train_cfg,
        tasks_dir,
        split,
        expected_vocab_size=model.model.vocab_size,
        use_rope_time=model.model.use_rope_time,
        max_windows=model.model.max_windows,
        batch_size=cfg.get("batch_size"),
        num_workers=cfg.get("num_workers"),
    )
    dataset = datamodule.predict_dataset
    logger.info(f"Loaded {len(dataset)} grid rows from {tasks_dir} (split={split})")

    trainer = build_predict_trainer(
        cfg.get("device"),
        precision=str(cfg.get("precision") or DEFAULT_PREDICT_PRECISION),
        enable_progress_bar=bool(cfg.get("enable_progress_bar", True)),
    )
    logger.info(f"Scoring the final query of {len(dataset)} rows on {trainer.strategy.root_device}")
    probs, labels = run_inference(model, datamodule, trainer)

    out = predictions_to_df(dataset, probs, labels)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} final-query predictions to {output_parquet}")


if __name__ == "__main__":
    main()
