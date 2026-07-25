#!/usr/bin/env python
"""Score-last inference over a **supplied** QuerySeqSchema parquet.

``eval_v2.py`` builds its own probe sequences (it samples contexts from the event shards, then
constructs designed ``[C,d]`` / ``[EOS,d][C,d]`` / ``[C,7d][C,30d]`` specs).  This script does the
same *inference* — teacher-forced scoring of the **last** query of each sequence, with the optional
counterfactual override of all prior answers — but takes the sequences as input instead of
generating them.  Build them however you like; ``EQ_generate_query_sequences contexts_path=...``
is the supported path for a user-supplied cohort.

The model is fed ``q_1..q_K`` plus the ground-truth answers to ``q_1..q_{K-1}``; the reported
``prob`` is its predicted probability for ``q_K``.  (Inference here is teacher-forced, not
generative — the prior answers are inputs, not predictions.)

Differences from ``eval_v2.score_last`` worth knowing:

- **Data config is derived from the checkpoint**, not hardcoded.  ``eval_v2`` pins
  ``max_seq_len=256`` / ``static_inclusion_mode=omit`` / ``seq_sampling_strategy=to_end``, which
  happen to match ``big_v2``; a differently-configured run would be scored under silently wrong
  settings.  Here they come from ``resolved_config.yaml``.
- **Vocabulary pre-flight.**  ``encode_query`` raises ``KeyError`` on an out-of-vocabulary code,
  and neither ``EQ_predict_sequences`` nor ``eval_v2`` checks up front — so OOV fails late, inside
  ``collate``, mid-run.  This checks every query code before loading the model.
- **Row alignment is verified, not assumed.**  The dataset's schema_df semi-join silently drops
  rows whose ``subject_id`` is absent from ``split``.  Probabilities are attached positionally
  (as in ``eval_v2``), so this asserts row count *and* per-row key equality first, and carries any
  extra columns (e.g. ``_ctx_id``) through from the supplied parquet.

Usage:

    python scripts/eval_v3.py \\
        --tasks cohort_tasks \\
        --run-dir  /experiments/EQ_conditional_experiments/model/big_v2 \\
        --cohort-dir /experiments/EQ_conditional_experiments/data/tensorized_cohort \\
        --split held_out \\
        --out-dir results/my_cohort

``--tasks`` accepts either a directory laid out as ``<dir>/<split>/*.parquet`` (what
``EQ_generate_query_sequences`` writes) or a single parquet file.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from every_query.data.schema import QuerySeqSchema
from every_query.data.seq_dataset import (
    ANSWER_NO,
    ANSWER_YES,
    ANSWERS_COL,
    DURATIONS_COL,
    QUERIES_COL,
    ConditionalQueryPytorchDataset,
)
from every_query.model.conditional_lightning import ConditionalQueryLightningModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KEY_COLS = ("subject_id", "prediction_time")
SEQ_COLS = (QUERIES_COL, DURATIONS_COL, ANSWERS_COL)
FORCE_PRIOR = {"none": None, "no": ANSWER_NO, "yes": ANSWER_YES}


# ── Inputs ──────────────────────────────────────────────────────────────


def resolve_tasks_dir(tasks: Path, split: str, stack: list) -> Path:
    """Return a ``task_labels_dir`` for ``tasks``, which may be a directory or a single parquet.

    A single file is exposed as ``<tmp>/<split>/<name>`` via symlink; the temporary directory is
    kept alive by appending it to ``stack`` (the caller holds it for the process lifetime).
    """
    if tasks.is_file():
        tmp = tempfile.TemporaryDirectory(prefix="eval_v3_tasks_")
        stack.append(tmp)
        d = Path(tmp.name) / split
        d.mkdir(parents=True)
        (d / tasks.name).symlink_to(tasks.resolve())
        return Path(tmp.name)
    if not tasks.is_dir():
        raise NotADirectoryError(f"--tasks {tasks} is neither a file nor a directory")
    if not any(tasks.rglob("*.parquet")):
        raise ValueError(f"--tasks {tasks} contains no parquet files")
    return tasks


def validate_supplied(df: pl.DataFrame, source: str) -> None:
    """Fail loudly on a frame that is not QuerySeqSchema-conformant or has ragged list columns."""
    missing = [c for c in (*KEY_COLS, *SEQ_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s) {missing}; expected QuerySeqSchema.")
    if not df.height:
        raise ValueError(f"{source} has no rows.")

    lens = df.select(
        pl.col(QUERIES_COL).list.len().alias("q"),
        pl.col(DURATIONS_COL).list.len().alias("d"),
        pl.col(ANSWERS_COL).list.len().alias("a"),
    )
    ragged = lens.filter((pl.col("q") != pl.col("d")) | (pl.col("q") != pl.col("a")))
    if ragged.height:
        raise ValueError(
            f"{source}: {ragged.height} row(s) have mismatched queries/durations/answers lengths."
        )
    if lens["q"].min() < 1:
        raise ValueError(f"{source}: every sequence needs at least one query.")
    if df[ANSWERS_COL].explode().is_null().any():
        raise ValueError(
            f"{source}: `answers` contains nulls.  Answers are binary in this design — an event "
            "that could not be observed is False, and censoring is carried by a TIMELINE//END "
            "query.  See conditional_model.py's module docstring."
        )
    # Cheap schema conformance check; raises on dtype drift the column check above would miss.
    QuerySeqSchema.align(df.select(*KEY_COLS, *SEQ_COLS).to_arrow())


def preflight_vocab(df: pl.DataFrame, code_metadata_fp: Path) -> None:
    """Raise on any query code outside the model's vocabulary, before the model is loaded.

    ``ConditionalQueryPytorchDataset.encode_query`` raises on OOV codes deliberately, but does so
    inside ``collate`` — i.e. after model load, mid-run.  Checking here turns that into a fast,
    actionable failure listing the offending codes.
    """
    vocab = set(pl.read_parquet(code_metadata_fp, columns=["code"])["code"].to_list())
    used = set(df[QUERIES_COL].explode().unique().drop_nulls().to_list())
    missing = sorted(used - vocab)
    if missing:
        shown = ", ".join(repr(c) for c in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise KeyError(
            f"{len(missing)} query code(s) are not in the model vocabulary ({code_metadata_fp}): "
            f"{shown}{more}"
        )


# ── Model + data config ─────────────────────────────────────────────────


def load_model(run_dir: Path, ckpt_name: str | None = None) -> ConditionalQueryLightningModule:
    """Load the conditional module from ``run_dir`` (explicit ckpt → best_model → checkpoints/last)."""
    if ckpt_name:
        ckpt = run_dir / "checkpoints" / f"{ckpt_name}.ckpt"
    else:
        ckpt = run_dir / "best_model.ckpt"
        if not ckpt.is_file():
            ckpt = run_dir / "checkpoints" / "last.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint found for {run_dir} (looked at {ckpt})")
    print(f"loading {ckpt}")
    return ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt)).eval().to(DEVICE)


def build_data_config(run_dir: Path, cohort_dir: Path, tasks_dir: Path):
    """Instantiate the run's own ``MEDSTorchDataConfig``, repointed at this cohort + task dir.

    Deriving rather than hardcoding keeps ``max_seq_len`` / ``static_inclusion_mode`` /
    ``seq_sampling_strategy`` in lockstep with whatever the checkpoint was trained under.
    """
    cfg_fp = run_dir / "resolved_config.yaml"
    if not cfg_fp.is_file():
        raise FileNotFoundError(f"{cfg_fp} not found; needed to derive the data config.")
    train_cfg = OmegaConf.load(cfg_fp)
    data_cfg = train_cfg.datamodule.config
    print(
        "data config from checkpoint: "
        f"max_seq_len={data_cfg.get('max_seq_len')} "
        f"static_inclusion_mode={data_cfg.get('static_inclusion_mode')} "
        f"seq_sampling_strategy={data_cfg.get('seq_sampling_strategy')}"
    )
    return instantiate(data_cfg, tensorized_cohort_dir=str(cohort_dir), task_labels_dir=str(tasks_dir))


def read_supplied(cfg) -> pl.DataFrame:
    """Read the task parquets in the dataset's own file order, so positional alignment is meaningful."""
    fps = list(cfg.task_labels_fps)
    if not fps:
        raise ValueError("MEDSTorchDataConfig.task_labels_fps resolved to no files.")
    return pl.concat([pl.read_parquet(fp) for fp in fps], how="vertical_relaxed")


def check_alignment(schema_df: pl.DataFrame, supplied: pl.DataFrame, split: str) -> None:
    """Verify the dataset kept every supplied row, in order, before attaching probs positionally."""
    if schema_df.height != supplied.height:
        raise RuntimeError(
            f"Dataset has {schema_df.height} rows but the supplied parquet(s) have "
            f"{supplied.height}.  {supplied.height - schema_df.height} row(s) were dropped — the "
            f"schema_df semi-join silently discards subjects absent from split={split!r}.  Check "
            "that your cohort's subjects live in that split."
        )
    for col in KEY_COLS:
        if not schema_df[col].equals(supplied[col]):
            raise RuntimeError(
                f"Row order diverged from the supplied parquet on {col!r}; positional attachment "
                "of probabilities would be wrong.  Do not trust this output."
            )


# ── Scoring ─────────────────────────────────────────────────────────────


@torch.no_grad()
def score_last(model, ds, batch_size: int, force_prior: int | None, keep_all_positions: bool):
    """Teacher-forced probability for the last query of each sequence, in dataset order.

    ``force_prior`` in ``{None, ANSWER_NO, ANSWER_YES}``: when set, every *prior* real position's
    teacher-forced answer is overridden to that class, giving the counterfactual
    ``P(q_K | all priors = <class>)`` on otherwise identical contexts.

    Returns ``(last_probs, all_probs)``; ``all_probs`` is None unless ``keep_all_positions``.
    """
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate)
    last_probs: list[float] = []
    all_probs: list[list[float]] | None = [] if keep_all_positions else None

    n_batches = (len(ds) + batch_size - 1) // batch_size
    for bi, batch in enumerate(dl):
        batch = move_data_to_device(batch, DEVICE)
        if force_prior is not None:
            qa = batch.q_answers.clone()
            lengths = batch.q_mask.sum(dim=1)
            for i in range(qa.shape[0]):
                if lengths[i] > 1:
                    qa[i, : lengths[i] - 1] = force_prior  # all but the last real position
            batch.q_answers = qa

        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, out = model.model(batch)

        probs = out.answer_probs.float().cpu()
        mask = batch.q_mask.cpu()
        for i in range(probs.shape[0]):
            idxs = mask[i].nonzero().flatten()
            last_probs.append(float(probs[i, idxs[-1]]))
            if all_probs is not None:
                all_probs.append(probs[i, idxs].tolist())

        if bi % 50 == 0:
            print(f"  batch {bi + 1}/{n_batches}", flush=True)

    return last_probs, all_probs


def macro_within_query(df: pl.DataFrame, prob_col="prob", min_pos=10, min_neg=10):
    """Per-target-code AUROC, macro-averaged.

    Pooled AUROC is base-rate inflated (≈0.91 vs an honest macro ≈0.77 on ``big_v2``); see
    ``CONDITIONAL_QUERIES.md:127-152``.
    """
    rows = []
    for (code,), g in df.group_by(["target_query"]):
        y = g["true_answer"].to_list()
        npos = int(sum(y))
        if npos >= min_pos and (len(y) - npos) >= min_neg:
            auroc = None
            if len(np.unique(y)) > 1:
                auroc = float(roc_auc_score(y, g[prob_col].to_list()))
            rows.append({"query": code, "n": len(y), "prevalence": npos / len(y), "auroc": auroc})
    if not rows:
        return pl.DataFrame(), None
    t = pl.DataFrame(rows).sort("query")
    scored = t.filter(pl.col("auroc").is_not_null())
    return t, (float(scored["auroc"].mean()) if scored.height else None)


# ── Driver ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, type=Path, help="QuerySeqSchema parquet, or dir of them")
    ap.add_argument(
        "--run-dir", required=True, type=Path, help="training run dir (resolved_config.yaml + ckpt)"
    )
    ap.add_argument("--cohort-dir", required=True, type=Path, help="tensorized cohort dir")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="held_out", help="split holding your subjects (default: held_out)")
    ap.add_argument("--ckpt-name", default=None, help="explicit checkpoint stem under checkpoints/")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--force-prior",
        default="none",
        choices=sorted(FORCE_PRIOR),
        help="override all prior teacher-forced answers to this class (counterfactual conditioning)",
    )
    ap.add_argument("--all-positions", action="store_true", help="also emit every position's probability")
    ap.add_argument(
        "--no-metrics", action="store_true", help="skip AUROC (use when answers are placeholders)"
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_stack: list = []
    tasks_dir = resolve_tasks_dir(args.tasks, args.split, tmp_stack)

    cfg = build_data_config(args.run_dir, args.cohort_dir, tasks_dir)
    supplied = read_supplied(cfg)
    validate_supplied(supplied, str(args.tasks))
    print(f"{supplied.height} supplied sequences from {args.tasks}")

    preflight_vocab(supplied, Path(cfg.code_metadata_fp))
    print(f"vocab pre-flight OK ({supplied[QUERIES_COL].explode().n_unique()} distinct query codes)")

    ds = ConditionalQueryPytorchDataset(cfg, split=args.split)
    check_alignment(ds.schema_df, supplied, args.split)
    print(f"dataset: {len(ds)} sequences, alignment verified; device {DEVICE}")

    model = load_model(args.run_dir, args.ckpt_name)
    force_prior = FORCE_PRIOR[args.force_prior]
    if force_prior is not None:
        print(f"force_prior={args.force_prior}: all prior answers overridden")

    last_probs, all_probs = score_last(model, ds, args.batch_size, force_prior, args.all_positions)

    # Extra columns the caller carried on the parquet (e.g. _ctx_id identifying replicates) ride
    # through positionally — the alignment check above is what makes that sound.
    extras = [c for c in supplied.columns if c not in (*KEY_COLS, *SEQ_COLS)]
    out = ds.schema_df.select(
        *KEY_COLS,
        pl.col(QUERIES_COL).list.len().alias("n_queries"),
        pl.col(QUERIES_COL).list.last().alias("target_query"),
        pl.col(DURATIONS_COL).list.last().alias("target_duration_days"),
        pl.col(ANSWERS_COL).list.last().alias("true_answer"),
        pl.col(ANSWERS_COL).list.slice(0, pl.col(ANSWERS_COL).list.len() - 1).alias("prior_answers"),
    ).with_columns(pl.Series("prob", last_probs, dtype=pl.Float64))
    if extras:
        out = out.hstack(supplied.select(extras))
        print(f"carried through extra column(s): {extras}")
    if all_probs is not None:
        out = out.with_columns(pl.Series("all_position_probs", all_probs, dtype=pl.List(pl.Float64)))

    preds_fp = args.out_dir / "predictions.parquet"
    out.write_parquet(preds_fp)
    print(f"wrote {out.height} rows to {preds_fp}")

    summary = {
        "tasks": str(args.tasks),
        "run_dir": str(args.run_dir),
        "split": args.split,
        "force_prior": args.force_prior,
        "n_sequences": out.height,
        "mean_prob": float(out["prob"].mean()),
        "target_prevalence": float(np.mean(out["true_answer"].to_list())),
    }

    if not args.no_metrics:
        tbl, macro = macro_within_query(out)
        if tbl.height:
            tbl.write_parquet(args.out_dir / "by_query.parquet")
        summary["macro_within_query_auroc"] = macro
        summary["n_codes_scored"] = tbl.height
        print(f"macro within-query AUROC: {macro} over {tbl.height} codes")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
