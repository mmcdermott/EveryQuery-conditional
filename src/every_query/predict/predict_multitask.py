"""Inference over a multitask evaluation grid — ``EQ_predict_multitask``.

Takes a trained :class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`
run directory and a grid written by ``EQ_generate_evaluation_multitask_sequences``, and writes one
flat row per ``(context, task, window)`` with the model's probability and the ground-truth label for
that window's **scored code**.

Which code is scored, and why
-----------------------------
During training every window's hidden state is projected onto the whole vocabulary, so one forward
pass yields ``(B, K, V)`` probabilities.  A *task*, though, asks about one code.  For window ``k``
the scored code is the one the query stream already names:

- ``k < K-1``: the conditioning code ``C_k``.  Its answer ``A_k`` is teacher-forced into the stream,
  but only at the position *after* ``W_k``, so the causal mask keeps it out of window ``k``'s own
  prediction — ``probs[:, k, C_k]`` is an honest prediction, not a read-back of its own label.
- ``k == K-1``: the task's ``target_code``.  This is the query the grid exists to answer, and the
  one the primary metric is computed on; rows carry ``is_final`` so a consumer can select it.

Writing every window rather than only the last costs nothing (the probabilities are already
computed) and multiplies the rows available for stratified analysis by ``K``.

Output columns
--------------
``subject_id``, ``prediction_time``, ``task_id``, ``task_group``, ``position`` (0-based window
index), ``is_final``, ``scored_code``, ``prob``, ``label``, plus the window-resolution diagnostics
carried through from the grid's sidecar: ``start_resolved``, ``end_resolved``, ``window_days``.

Row order is dataloader order exploded by window, so consumers can regroup on
``(subject_id, prediction_time, task_id)``.
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

from every_query.data.multitask_dataset import SOURCE_ROW_COL, SOURCE_SHARD_COL
from every_query.generate_tasks.sample_evaluation_multitask_sequences import (
    END_RESOLVED_COL,
    GROUP_COL,
    START_RESOLVED_COL,
    TARGET_CODE_COL,
    TASK_ID_COL,
    WINDOW_DAYS_COL,
)
from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")

_SPLIT_TO_DATAMODULE_ATTRS: dict[str, tuple[str, str]] = {
    tuning_split: ("val_dataset", "val_dataloader"),
    held_out_split: ("test_dataset", "test_dataloader"),
}

SIDECAR_COLUMNS = [
    TASK_ID_COL,
    GROUP_COL,
    TARGET_CODE_COL,
    START_RESOLVED_COL,
    END_RESOLVED_COL,
    WINDOW_DAYS_COL,
]


def read_sidecars(meta_dir: Path, split: str) -> pl.DataFrame:
    """Load the grid's evaluation sidecars, keyed the way the dataset names its source rows.

    ``MultitaskBoundaryPytorchDataset`` tags every metadata row with ``_source_shard`` (the label
    parquet's path relative to ``task_labels_dir``, without the suffix) and ``_source_row`` (its
    physical row in that file).  The sidecar is written row-for-row alongside that parquet, so
    re-deriving the same two keys here is what re-attaches a task to a model output — no ordering
    assumption, no join on floats.
    """
    split_dir = meta_dir / split
    fps = sorted(split_dir.glob("*.parquet"))
    if not fps:
        raise FileNotFoundError(f"no evaluation sidecars under {split_dir}")
    frames = []
    for fp in fps:
        frames.append(
            pl.read_parquet(fp)
            .with_row_index(SOURCE_ROW_COL)
            .with_columns(
                pl.col(SOURCE_ROW_COL).cast(pl.Int64),
                pl.lit(f"{split}/{fp.stem}").alias(SOURCE_SHARD_COL),
            )
        )
    return pl.concat(frames, how="vertical")


def align_sidecar(schema_df: pl.DataFrame, sidecar: pl.DataFrame) -> pl.DataFrame:
    """Reorder ``sidecar`` into dataset row order, failing loudly if a row is unmatched."""
    keys = [SOURCE_SHARD_COL, SOURCE_ROW_COL]
    missing = [c for c in keys if c not in schema_df.columns]
    if missing:
        raise ValueError(f"the dataset's schema frame lacks {missing}; it is not a multitask dataset")
    joined = (
        schema_df.select(*keys)
        .with_row_index("_pos")
        .join(sidecar.select(*keys, *SIDECAR_COLUMNS), on=keys, how="left")
        .sort("_pos")
        .drop("_pos")
    )
    if joined.height != schema_df.height:
        raise RuntimeError(
            f"sidecar join changed the row count ({schema_df.height} -> {joined.height}); the "
            "sidecars and label parquets are inconsistent."
        )
    n_null = joined[TASK_ID_COL].null_count()
    if n_null:
        raise RuntimeError(
            f"{n_null} labeled row(s) have no evaluation sidecar entry; regenerate the grid so the "
            "sidecar and the label parquets are written together."
        )
    return joined


def scored_code_matrix(
    dataset, sidecar: pl.DataFrame, num_bounds: int
) -> tuple[np.ndarray, list[list[str]]]:
    """``(N, K)`` vocabulary indices of the scored code per window, plus the code strings.

    Windows ``0..K-2`` score their conditioning code; window ``K-1`` scores the task's target code.
    """
    cond = np.stack(dataset._condition_codes).astype(np.int64)
    if cond.shape[1] != num_bounds - 1:
        raise ValueError(f"expected {num_bounds - 1} conditioning codes per row, got {cond.shape[1]}")
    target_codes = sidecar[TARGET_CODE_COL].to_list()
    unknown = sorted({c for c in target_codes if c not in dataset.code_to_index})
    if unknown:
        raise ValueError(f"{len(unknown)} target code(s) are outside the cohort vocabulary: {unknown[:5]}")
    target_idx = np.array([dataset.code_to_index[c] for c in target_codes], dtype=np.int64)

    idx = np.empty((cond.shape[0], num_bounds), dtype=np.int64)
    idx[:, : num_bounds - 1] = cond
    idx[:, num_bounds - 1] = target_idx

    index_to_code = {i: c for c, i in dataset.code_to_index.items()}
    codes = [[index_to_code.get(int(v), "") for v in row] for row in idx]
    return idx, codes


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
    dataloader,
    scored_idx: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the grid, keeping only the scored code's probability and label per window.

    The model emits ``(B, K, V)`` logits; at V ~ 14k that is ~70 MB per batch of 256, so gathering
    down to ``(B, K)`` inside the loop — rather than accumulating batches the way
    ``Trainer.predict`` does — is what keeps the whole grid in memory.
    """
    model = model.to(device).eval()
    probs_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    row = 0
    for batch in dataloader:
        batch = _to_device(batch, device)
        _, outputs = model.model(batch)
        b = outputs.logits.shape[0]
        sel = torch.from_numpy(scored_idx[row : row + b]).to(device).unsqueeze(-1)
        probs_out.append(outputs.probs.gather(2, sel).squeeze(-1).float().cpu().numpy())
        labels_out.append(batch.targets.gather(2, sel).squeeze(-1).cpu().numpy())
        row += b
    if row != scored_idx.shape[0]:
        raise RuntimeError(f"dataloader yielded {row} rows but the grid has {scored_idx.shape[0]}")
    return np.concatenate(probs_out), np.concatenate(labels_out)


def predictions_to_df(
    schema_df: pl.DataFrame,
    sidecar: pl.DataFrame,
    scored_codes: list[list[str]],
    probs: np.ndarray,
    labels: np.ndarray,
) -> pl.DataFrame:
    """Explode the ``(N, K)`` arrays into one row per ``(context, task, window)``."""
    k = probs.shape[1]
    out = pl.DataFrame(
        {
            "subject_id": schema_df["subject_id"],
            "prediction_time": schema_df["prediction_time"],
            TASK_ID_COL: sidecar[TASK_ID_COL],
            GROUP_COL: sidecar[GROUP_COL],
            "scored_code": scored_codes,
            "prob": [row.tolist() for row in probs],
            "label": [row.tolist() for row in labels],
            START_RESOLVED_COL: sidecar[START_RESOLVED_COL],
            END_RESOLVED_COL: sidecar[END_RESOLVED_COL],
            WINDOW_DAYS_COL: sidecar[WINDOW_DAYS_COL],
        }
    ).with_columns(pl.int_ranges(0, k).alias("position"))
    exploded = [
        "position",
        "scored_code",
        "prob",
        "label",
        START_RESOLVED_COL,
        END_RESOLVED_COL,
        WINDOW_DAYS_COL,
    ]
    return (
        out.explode(*exploded)
        .with_columns((pl.col("position") == k - 1).alias("is_final"))
        .select(
            "subject_id",
            "prediction_time",
            TASK_ID_COL,
            GROUP_COL,
            "position",
            "is_final",
            "scored_code",
            pl.col("prob").cast(pl.Float32),
            pl.col("label").cast(pl.Boolean),
            START_RESOLVED_COL,
            END_RESOLVED_COL,
            WINDOW_DAYS_COL,
        )
    )


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict_multitask")
def main(cfg: DictConfig) -> None:
    model_run_dir = Path(cfg.model_run_dir)
    tasks_dir = Path(cfg.tasks_dir)
    meta_dir = Path(cfg.eval_meta_dir) if cfg.get("eval_meta_dir") else tasks_dir.parent / "eval_meta"
    output_parquet = Path(cfg.output_parquet)
    split = cfg.get("split", held_out_split)
    overwrite = bool(cfg.get("overwrite", False))

    if split not in _SPLIT_TO_DATAMODULE_ATTRS:
        raise ValueError(f"split must be one of {sorted(_SPLIT_TO_DATAMODULE_ATTRS)}, got {split!r}.")
    if output_parquet.exists() and not overwrite:
        raise FileExistsError(f"output_parquet {output_parquet} already exists; pass overwrite=true.")

    train_cfg, model, _ = setup_model(
        model_run_dir, ckpt_name=cfg.get("ckpt_name"), module_cls=ConditionalMultitaskLightningModule
    )

    if cfg.get("batch_size") is not None:
        train_cfg.datamodule.batch_size = int(cfg.batch_size)
    train_cfg.datamodule.config.task_labels_dir = str(tasks_dir)
    D = instantiate(train_cfg.datamodule)

    dataset_attr, dataloader_attr = _SPLIT_TO_DATAMODULE_ATTRS[split]
    dataset = getattr(D, dataset_attr)
    dataloader = getattr(D, dataloader_attr)()
    sampler_cls = type(getattr(dataloader, "sampler", None)).__name__
    if sampler_cls != "SequentialSampler":
        raise RuntimeError(f"{dataloader_attr} must use SequentialSampler; got {sampler_cls!r}.")
    logger.info(f"Loaded {len(dataset)} grid rows from {tasks_dir} (split={split})")

    sidecar = align_sidecar(dataset.schema_df, read_sidecars(meta_dir, split))
    scored_idx, scored_codes = scored_code_matrix(dataset, sidecar, dataset.num_bounds)

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Running inference on {device} over {len(dataset)} rows x K={dataset.num_bounds}")
    probs, labels = run_inference(model, dataloader, scored_idx, device)

    out = predictions_to_df(dataset.schema_df, sidecar, scored_codes, probs, labels)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} per-window predictions to {output_parquet}")


if __name__ == "__main__":
    main()
