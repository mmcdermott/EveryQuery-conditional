#!/usr/bin/env python
"""Full evaluation of a trained conditional-query model + figure/metric generation for the report.

Produces, under ``--out-dir``:
  random_predictions.parquet            per-position predictions on the held-out random sequences
  metrics.by_position.parquet           AUROC by sequence position (random eval)
  metrics.by_query.parquet              AUROC by (query, horizon bucket) (random eval)
  clinical/<task>.predictions.parquet   predictions for each designed clinical task
  clinical_summary.parquet              per-clinical-task target AUROC / prevalence / counts
  conditioning_effect.parquet           singleton-vs-conditional comparison on matched queries
  figs/*.png                            figures for the report
  summary.json                          headline numbers

This script imports the pipeline functions directly (no subprocess) so it can also build the
"singleton" counterfactual (each query asked alone, i.e. [censor, q]) for the conditioning study.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from sklearn.metrics import roc_auc_score

from every_query.data.seq_dataset import (
    ANSWER_YES,
    CENSOR_QUERY_CODE,
    ConditionalQueryPytorchDataset,
)
from every_query.evaluate.evaluate_sequences import compute_sequence_metrics
from every_query.model.conditional_lightning import ConditionalQueryLightningModule
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import OmegaConf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _auroc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def load_model(run_dir: Path) -> ConditionalQueryLightningModule:
    ckpt = run_dir / "best_model.ckpt"
    if not ckpt.is_file():
        ckpt = run_dir / "checkpoints" / "last.ckpt"
    m = ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt))
    m.eval().to(DEVICE)
    return m


def predict_dataset(model, dataset, batch_size=256) -> pl.DataFrame:
    """Teacher-forced per-position predictions, stitched onto the dataset's sequence rows."""
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
    probs_per_row: list[list[float]] = []
    with torch.no_grad():
        for batch in dl:
            batch = move_data_to_device(batch, DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                _, out = model.model(batch)
            probs = out.answer_probs.float().cpu()
            mask = batch.q_mask.cpu()
            for i in range(probs.shape[0]):
                probs_per_row.append(probs[i][mask[i]].tolist())

    sdf = dataset.schema_df
    out = sdf.select(
        "subject_id",
        "prediction_time",
        pl.col("queries").alias("query"),
        pl.col("durations").alias("duration_days"),
        pl.col("answers").alias("answer"),
    ).with_columns(pl.Series("answer_prob", probs_per_row, dtype=pl.List(pl.Float32)))
    n_q = pl.col("query").list.len()
    out = out.with_columns(pl.int_ranges(0, n_q).alias("position"))
    return out.explode("position", "query", "duration_days", "answer", "answer_prob").select(
        "subject_id", "prediction_time", "position", "query", "duration_days", "answer", "answer_prob"
    )


def make_dataset(cohort_dir, tasks_dir, split, max_seq_len=256):
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cohort_dir),
        task_labels_dir=str(tasks_dir),
        max_seq_len=max_seq_len,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    return ConditionalQueryPytorchDataset(cfg, split=split)


# ── Conditioning study: singleton vs in-context ─────────────────────────


def build_singleton_dataset(cohort_dir, tasks_dir, split, tmp_dir, position=1, n_seq=20000):
    """Write a tasks dir where each sequence is [censor, q_j] for the j-th query of each
    original sequence — the same query, asked in isolation — so we can measure how much the
    preceding answers change the prediction for that query."""
    from every_query.data.schema import QuerySeqSchema

    src = make_dataset(cohort_dir, tasks_dir, split)
    sdf = src.schema_df
    # Keep sequences with at least `position+1` queries; take the query at `position`.
    sdf = sdf.filter(pl.col("queries").list.len() > position).head(n_seq)

    singleton = sdf.select(
        "subject_id",
        "prediction_time",
        pl.concat_list(
            pl.col("queries").list.first(),  # censor
            pl.col("queries").list.get(position),
        ).alias("queries"),
        pl.concat_list(
            pl.col("durations").list.first(),
            pl.col("durations").list.get(position),
        ).alias("durations"),
        pl.concat_list(
            pl.col("answers").list.first(),
            pl.col("answers").list.get(position),
        ).alias("answers"),
    )

    out_dir = Path(tmp_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned = QuerySeqSchema.align(singleton.to_arrow())
    pl.from_arrow(aligned).write_parquet(out_dir / "singletons.parquet")
    return Path(tmp_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument("--tasks-dir", required=True, type=Path)
    p.add_argument("--clinical-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="held_out")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figs").mkdir(exist_ok=True)
    (args.out_dir / "clinical").mkdir(exist_ok=True)
    summary: dict = {}

    model = load_model(args.run_dir)
    print(f"loaded model from {args.run_dir} on {DEVICE}")
    summary["_arch"] = {
        "params_m": sum(p.numel() for p in model.model.parameters()) / 1e6,
        "encoder_layers": model.model.HF_model_config.num_hidden_layers,
        "hidden_size": model.model.HF_model_config.hidden_size,
    }

    # 1. Random held-out sequences ------------------------------------------------------
    ds = make_dataset(args.cohort_dir, args.tasks_dir, args.split)
    print(f"random eval: {len(ds)} sequences")
    rnd = predict_dataset(model, ds, args.batch_size)
    rnd.write_parquet(args.out_dir / "random_predictions.parquet")

    by_pos, by_query = compute_sequence_metrics(rnd)
    by_pos.write_parquet(args.out_dir / "metrics.by_position.parquet")
    by_query.write_parquet(args.out_dir / "metrics.by_query.parquet")

    # Headline: censor-query AUROC (position 0) and pooled occurrence AUROC (positions >=1).
    censor = rnd.filter(pl.col("position") == 0)
    occ = rnd.filter((pl.col("position") >= 1) & pl.col("answer").is_not_null())
    summary["random"] = {
        "n_sequences": ds.schema_df.height,
        "n_query_positions": rnd.height,
        "censor_auroc": _auroc(censor["answer"].to_list(), censor["answer_prob"].to_list()),
        "censor_prevalence": float(censor["answer"].mean()),
        "occurs_auroc_pooled": _auroc(occ["answer"].to_list(), occ["answer_prob"].to_list()),
        "occurs_prevalence": float(occ["answer"].mean()),
        "occurs_censored_frac": float(
            rnd.filter(pl.col("position") >= 1)["answer"].null_count()
            / max(1, rnd.filter(pl.col("position") >= 1).height)
        ),
    }

    # 2. Clinical designed tasks --------------------------------------------------------
    clinical_rows = []
    for task_dir in sorted(p for p in args.clinical_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        cds = make_dataset(args.cohort_dir, task_dir, args.split)
        preds = predict_dataset(model, cds, args.batch_size)
        preds.write_parquet(args.out_dir / "clinical" / f"{task}.predictions.parquet")

        last_pos = int(preds["position"].max())
        target = preds.filter(pl.col("position") == last_pos)
        obs = target.filter(pl.col("answer").is_not_null())
        clinical_rows.append(
            {
                "task": task,
                "n_sequences": cds.schema_df.height,
                "target_position": last_pos,
                "target_query": target["query"][0],
                "target_auroc": _auroc(obs["answer"].to_list(), obs["answer_prob"].to_list()),
                "target_prevalence": float(obs["answer"].mean()) if obs.height else None,
                "n_observed": obs.height,
                "censored_frac": float(target["answer"].null_count() / max(1, target.height)),
                "censor_auroc": _auroc(
                    preds.filter(pl.col("position") == 0)["answer"].to_list(),
                    preds.filter(pl.col("position") == 0)["answer_prob"].to_list(),
                ),
            }
        )
        print(f"clinical {task}: AUROC {clinical_rows[-1]['target_auroc']}")
    clinical = pl.DataFrame(clinical_rows)
    clinical.write_parquet(args.out_dir / "clinical_summary.parquet")
    summary["clinical"] = clinical_rows

    # 3. Conditioning study: same query, in-context vs singleton ------------------------
    tmp = args.out_dir / "_singleton_tasks"
    cond_rows = []
    cond_pairs: dict[int, pl.DataFrame] = {}
    for position in (1, 2):
        sdir = build_singleton_dataset(args.cohort_dir, args.tasks_dir, args.split, tmp, position=position)
        sds = make_dataset(args.cohort_dir, sdir, args.split)
        s_pred = predict_dataset(model, sds, args.batch_size)
        # Singleton: target is position 1 of the [censor, q] sequence.
        s_tgt = s_pred.filter(pl.col("position") == 1).rename({"answer_prob": "prob_singleton"})

        # In-context: same (subject, prediction_time, query, duration) at the original position.
        ctx_tgt = rnd.filter(pl.col("position") == position).rename({"answer_prob": "prob_incontext"})

        joined = ctx_tgt.join(
            s_tgt.select(
                "subject_id", "prediction_time", "query", "duration_days", "prob_singleton"
            ),
            on=["subject_id", "prediction_time", "query", "duration_days"],
            how="inner",
        ).filter(pl.col("answer").is_not_null())
        if joined.height < 10:
            continue
        cond_pairs[position] = joined
        cond_rows.append(
            {
                "position": position,
                "n_matched": joined.height,
                "auroc_singleton": _auroc(
                    joined["answer"].to_list(), joined["prob_singleton"].to_list()
                ),
                "auroc_incontext": _auroc(
                    joined["answer"].to_list(), joined["prob_incontext"].to_list()
                ),
                "mean_abs_prob_shift": float(
                    (joined["prob_incontext"] - joined["prob_singleton"]).abs().mean()
                ),
                "corr_probs": float(
                    np.corrcoef(
                        joined["prob_incontext"].to_numpy(), joined["prob_singleton"].to_numpy()
                    )[0, 1]
                ),
            }
        )
    cond = pl.DataFrame(cond_rows)
    cond.write_parquet(args.out_dir / "conditioning_effect.parquet")
    summary["conditioning"] = cond_rows
    if cond_pairs:
        best_pos = min(cond_pairs)
        cond_pairs[best_pos].write_parquet(args.out_dir / "conditioning_pairs.parquet")

    # 4. Figures ------------------------------------------------------------------------
    _make_figures(args.out_dir, by_pos, by_query, clinical, cond, rnd, cond_pairs)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote summary.json")
    print(json.dumps(summary, indent=2))


def _make_figures(out_dir, by_pos, by_query, clinical, cond, rnd, cond_pairs):
    figs = out_dir / "figs"

    # AUROC by position.
    bp = by_pos.filter(pl.col("auroc").is_not_null())
    if bp.height:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(bp["position"].to_list(), bp["auroc"].to_list(), color="#4C72B0")
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlabel("query position in sequence (0 = censor query)")
        ax.set_ylabel("AUROC")
        ax.set_title("Held-out AUROC by sequence position")
        ax.set_ylim(0.4, 1.0)
        fig.tight_layout()
        fig.savefig(figs / "auroc_by_position.png", dpi=150)
        plt.close(fig)

    # AUROC by horizon bucket (averaged over queries).
    bq = by_query.filter(pl.col("auroc").is_not_null())
    if bq.height:
        agg = (
            bq.group_by("duration_bucket")
            .agg(pl.col("auroc").mean().alias("auroc"), pl.len().alias("n"))
        )
        order = ["1d", "2-7d", "8-30d", "31-90d", "91-180d", "181-365d", ">365d"]
        agg = agg.with_columns(
            pl.col("duration_bucket").cast(pl.Enum(order)).alias("_o")
        ).sort("_o")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(agg["duration_bucket"].to_list(), agg["auroc"].to_list(), color="#55A868")
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlabel("query horizon bucket")
        ax.set_ylabel("mean AUROC over query codes")
        ax.set_title("Held-out AUROC by query horizon")
        ax.set_ylim(0.4, 1.0)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(figs / "auroc_by_horizon.png", dpi=150)
        plt.close(fig)

    # Clinical task AUROCs.
    cl = clinical.filter(pl.col("target_auroc").is_not_null()).sort("target_auroc")
    if cl.height:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(cl["task"].to_list(), cl["target_auroc"].to_list(), color="#C44E52")
        ax.axvline(0.5, ls="--", c="grey", lw=1)
        for i, (a, pv) in enumerate(zip(cl["target_auroc"], cl["target_prevalence"], strict=True)):
            ax.text(a + 0.01, i, f"{a:.3f} (p={pv:.2%})", va="center", fontsize=8)
        ax.set_xlim(0.4, 1.05)
        ax.set_xlabel("target-query AUROC")
        ax.set_title("Designed clinical conditional tasks (held-out)")
        fig.tight_layout()
        fig.savefig(figs / "clinical_auroc.png", dpi=150)
        plt.close(fig)

    # Conditioning scatter: in-context vs singleton probability for the same query.
    if cond_pairs:
        pos = min(cond_pairs)
        j = cond_pairs[pos]
        s = j["prob_singleton"].to_numpy()
        c = j["prob_incontext"].to_numpy()
        ans = j["answer"].to_numpy()
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        for label, color in [(True, "#C44E52"), (False, "#4C72B0")]:
            m = ans == label
            ax.scatter(s[m], c[m], s=6, alpha=0.3, c=color, label=f"answer={label}")
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        ax.set_xlabel("P(answer) asked alone  [censor, q]")
        ax.set_ylabel("P(answer) asked in context  [..., q]")
        ax.set_title(f"Effect of conditioning on prior answers (pos {pos})")
        ax.legend(fontsize=8, markerscale=2)
        fig.tight_layout()
        fig.savefig(figs / "conditioning_scatter.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
