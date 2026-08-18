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
"singleton" counterfactual (each query asked alone, i.e. [EOS, q]) for the conditioning study.

Status (conditional-v2): this script reads **position 0 as the censoring query** throughout —
``censor_auroc``/``censor_prevalence``, the ``[EOS, q]`` singleton prefix, and the by-position
figures all assume it.  That was guaranteed in v1 (every sequence began with the ``__CENSOR__``
sentinel); it is *not* guaranteed now.  The v2 sampler draws fully random sequences and only puts
the end-of-record query first when ``eos_first_fraction`` says so, so :func:`require_eos_first`
below hard-fails on a tasks dir that was not generated with ``eos_first_fraction=1.0`` rather than
silently reporting a random query's AUROC as the censoring number.

Two further v1 assumptions are now inert rather than wrong: answers are non-null booleans, so every
``*censored_frac`` reads 0.0 and ``answer.is_not_null()`` filters nothing.  Censoring is visible
only through the EOS query itself.  ``eval_v3.py`` is the current evaluator; this script and its
partner ``build_report.py`` are kept as the record of the v1 report.
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
    EOS_CODE,
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


def per_group_auroc(df: pl.DataFrame, group_cols: list[str], min_pos: int = 10, min_neg: int = 10):
    """Within-group AUROC over observed (non-null answer) rows.

    Returns a frame with one row per group: counts and AUROC (null when a group has
    fewer than ``min_pos`` positives or ``min_neg`` negatives — too few for a stable
    within-group estimate).  Computing AUROC *inside* a group (e.g. a single query code)
    means every positive/negative pair shares that group's base rate, so the estimate is
    not inflated by cross-group base-rate separation the way a pooled AUROC is.
    """
    rows = []
    for key, g in df.group_by(group_cols, maintain_order=True):
        obs = g.filter(pl.col("answer").is_not_null())
        y = obs["answer"].to_list()
        n_pos = int(sum(y))
        n_neg = len(y) - n_pos
        au = _auroc(y, obs["answer_prob"].to_list()) if (n_pos >= min_pos and n_neg >= min_neg) else None
        rows.append({**dict(zip(group_cols, key, strict=True)), "n": len(y), "n_pos": n_pos, "auroc": au})
    return pl.DataFrame(rows)


def macro_auroc(group_df: pl.DataFrame):
    """Mean within-group AUROC over groups that had a defined estimate (+ how many qualified)."""
    valid = group_df.filter(pl.col("auroc").is_not_null())
    return (float(valid["auroc"].mean()) if valid.height else None, valid.height)


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
    # ``seq_id`` is a per-sequence (per schema_df row) identifier.  It is load-bearing for the
    # position probe, where several sequences share the same (subject_id, prediction_time) but
    # place the target code at different positions — without it they'd be indistinguishable
    # after the explode below.
    out = out.with_row_index("seq_id")
    n_q = pl.col("query").list.len()
    out = out.with_columns(pl.int_ranges(0, n_q).alias("position"))
    return out.explode("position", "query", "duration_days", "answer", "answer_prob").select(
        "seq_id", "subject_id", "prediction_time", "position", "query", "duration_days", "answer", "answer_prob"
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


def require_eos_first(ds, what: str) -> None:
    """Abort unless position 0 of every sequence in ``ds`` is the end-of-record query.

    Everything below treats position 0 as "the censoring query".  In v1 that held by construction;
    in v2 it holds only for sequences generated with ``eos_first_fraction=1.0``.  Without this check
    the script runs to completion and reports an arbitrary query code's AUROC under the key
    ``censor_auroc`` — a wrong number that looks right.
    """
    first = ds.schema_df.select(pl.col("queries").list.first().alias("q"))["q"]
    offenders = first.filter(first != EOS_CODE)
    if offenders.len():
        raise SystemExit(
            f"{what}: {offenders.len()} of {first.len()} sequences do not start with "
            f"{EOS_CODE!r} (e.g. {offenders[0]!r}).  This script reads position 0 as the censoring "
            "query, which the v2 sampler only guarantees when the sequences were generated with "
            "`EQ_generate_query_sequences ... eos_first_fraction=1.0`.  Regenerate the tasks with "
            "that setting, or use scripts/eval_v3.py, which makes no position-0 assumption."
        )


# ── Conditioning study: singleton vs in-context ─────────────────────────


def build_singleton_dataset(cohort_dir, tasks_dir, split, tmp_dir, position=1, n_seq=20000):
    """Write a tasks dir where each sequence is [EOS, q_j] for the j-th query of each
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
            pl.col("queries").list.first(),  # the end-of-record (censoring) query
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
    p.add_argument("--probe-dir", type=Path, default=None,
                   help="dir with probe/tasks.parquet from make_position_probe.py (matched-code "
                        "per-position conditioning probe); skipped if omitted")
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
    require_eos_first(ds, f"random eval tasks ({args.tasks_dir})")
    print(f"random eval: {len(ds)} sequences")
    rnd = predict_dataset(model, ds, args.batch_size)
    rnd.write_parquet(args.out_dir / "random_predictions.parquet")

    by_pos, by_query = compute_sequence_metrics(rnd)
    by_pos.write_parquet(args.out_dir / "metrics.by_position.parquet")
    by_query.write_parquet(args.out_dir / "metrics.by_query.parquet")

    censor = rnd.filter(pl.col("position") == 0)
    occ = rnd.filter((pl.col("position") >= 1) & pl.col("answer").is_not_null())

    # Pooled occurrence AUROC — over all positions>=1 query codes at once.  This is the
    # base-rate-INFLATED number: most positive/negative pairs are cross-query, separable
    # just by per-code prevalence, so it overstates within-query skill.  Kept only as a
    # contrast against the macro (within-query) AUROC below.
    occ_pooled = _auroc(occ["answer"].to_list(), occ["answer_prob"].to_list())

    # Within-query AUROC: compute AUROC inside each occurrence query code, then macro-average.
    pg_query = per_group_auroc(occ, ["query"], min_pos=10, min_neg=10)
    pg_query.write_parquet(args.out_dir / "occ_auroc_by_query.parquet")
    macro_q, n_q_groups = macro_auroc(pg_query)

    # Within-query AUROC broken out by sequence position (positions >=1).  Macro-averaging the
    # per-(query, position) AUROCs within each position is the clean conditioning test: same
    # query distribution at every position, only the count of prior teacher-forced answers differs.
    pg_qp = per_group_auroc(occ, ["query", "position"], min_pos=8, min_neg=8)
    pg_qp.write_parquet(args.out_dir / "occ_auroc_by_query_position.parquet")
    by_pos_macro = []
    for pos in sorted(pg_qp["position"].unique().to_list()):
        m, ng = macro_auroc(pg_qp.filter(pl.col("position") == pos))
        by_pos_macro.append({"position": pos, "macro_auroc": m, "n_query_groups": ng})
    pl.DataFrame(by_pos_macro).write_parquet(args.out_dir / "macro_auroc_by_position.parquet")

    summary["random"] = {
        "n_sequences": ds.schema_df.height,
        "n_query_positions": rnd.height,
        "censor_auroc": _auroc(censor["answer"].to_list(), censor["answer_prob"].to_list()),
        "censor_prevalence": float(censor["answer"].mean()),
        "occurs_auroc_pooled_inflated": occ_pooled,
        "occurs_auroc_macro_per_query": macro_q,
        "n_query_groups_macro": n_q_groups,
        "occurs_prevalence": float(occ["answer"].mean()),
        "occurs_censored_frac": float(
            rnd.filter(pl.col("position") >= 1)["answer"].null_count()
            / max(1, rnd.filter(pl.col("position") >= 1).height)
        ),
        "macro_auroc_by_position": by_pos_macro,
    }

    # 2. Clinical designed tasks --------------------------------------------------------
    clinical_rows = []
    for task_dir in sorted(p for p in args.clinical_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        cds = make_dataset(args.cohort_dir, task_dir, args.split)
        require_eos_first(cds, f"clinical task {task!r} ({task_dir})")
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
        # Within-query macro AUROC for each setting (avoids the cross-query base-rate inflation),
        # computed over the same matched rows so singleton vs in-context is a fair comparison.
        macro_s, _ = macro_auroc(
            per_group_auroc(joined.rename({"prob_singleton": "answer_prob"}), ["query"], 8, 8)
        )
        macro_c, _ = macro_auroc(
            per_group_auroc(joined.rename({"prob_incontext": "answer_prob"}), ["query"], 8, 8)
        )
        cond_rows.append(
            {
                "position": position,
                "n_matched": joined.height,
                "macro_auroc_singleton": macro_s,
                "macro_auroc_incontext": macro_c,
                "auroc_singleton_pooled": _auroc(
                    joined["answer"].to_list(), joined["prob_singleton"].to_list()
                ),
                "auroc_incontext_pooled": _auroc(
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

    # 4. Dense matched-code position probe ----------------------------------------------
    # The cleanest conditioning curve: for ~20 curated codes the *same* code sits at
    # positions 1..P across many patients, so per-(code, position) AUROC has enough
    # positives and a fixed code compared across positions varies only the number of
    # prior teacher-forced answers it conditions on.  Target = each sequence's last query.
    probe_macro = []
    if args.probe_dir is not None and (args.probe_dir / "probe" / "tasks.parquet").exists():
        pds = make_dataset(args.cohort_dir, args.probe_dir / "probe", args.split)
        ppred = predict_dataset(model, pds, args.batch_size)
        # Target of each probe sequence is its own last position; key on the per-sequence id so
        # sequences sharing a (subject, prediction_time) but different target positions stay distinct.
        last = ppred.group_by("seq_id").agg(pl.col("position").max().alias("_lp"))
        ptgt = ppred.join(last, on="seq_id").filter(
            pl.col("position") == pl.col("_lp")
        ).rename({"query": "target_query", "position": "target_position"})
        ptgt.write_parquet(args.out_dir / "probe_predictions.parquet")

        pg_probe = per_group_auroc(ptgt, ["target_query", "target_position"], min_pos=10, min_neg=10)
        pg_probe.write_parquet(args.out_dir / "probe_auroc_by_query_position.parquet")
        for pos in sorted(pg_probe["target_position"].unique().to_list()):
            sub = pg_probe.filter(pl.col("target_position") == pos)
            m, ng = macro_auroc(sub)
            probe_macro.append({"position": pos, "macro_auroc": m, "n_codes": ng})
        # Per-code curve across positions (codes present at every position with a defined AUROC).
        pl.DataFrame(probe_macro).write_parquet(args.out_dir / "probe_macro_by_position.parquet")
        summary["probe"] = {
            "n_sequences": pds.schema_df.height,
            "macro_auroc_by_position": probe_macro,
        }
        print("probe macro by position:", probe_macro)

    # 5. Figures ------------------------------------------------------------------------
    _make_figures(args.out_dir, by_pos, by_query, clinical, cond, rnd, cond_pairs, summary)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote summary.json")
    print(json.dumps(summary, indent=2))


def _make_figures(out_dir, by_pos, by_query, clinical, cond, rnd, cond_pairs, summary):
    figs = out_dir / "figs"

    # Pooled-vs-within-query AUROC by position: the pooled (base-rate-inflated) per-position
    # AUROC next to the macro within-query per-position AUROC, so the inflation is visible and
    # the clean conditioning trend is the macro line.
    bp = by_pos.filter((pl.col("auroc").is_not_null()) & (pl.col("position") >= 1))
    macro_bp = pl.DataFrame(summary["random"]["macro_auroc_by_position"]).filter(
        pl.col("macro_auroc").is_not_null()
    )
    if bp.height or macro_bp.height:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        if bp.height:
            ax.plot(bp["position"], bp["auroc"], "-o", c="#BBBBBB", label="pooled (base-rate inflated)")
        if macro_bp.height:
            ax.plot(macro_bp["position"], macro_bp["macro_auroc"], "-o", c="#C44E52",
                    label="macro within-query (clean)")
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlabel("occurrence-query position (# of prior answers conditioned on)")
        ax.set_ylabel("AUROC")
        ax.set_title("Occurrence AUROC by position: pooled vs. within-query")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "auroc_pooled_vs_macro_by_position.png", dpi=150)
        plt.close(fig)

    # Matched-code probe: per-code curves across positions + macro mean.
    probe_fp = out_dir / "probe_auroc_by_query_position.parquet"
    if probe_fp.exists():
        pg = pl.read_parquet(probe_fp).filter(pl.col("auroc").is_not_null())
        if pg.height:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            for code, g in pg.group_by("target_query"):
                g = g.sort("target_position")
                ax.plot(g["target_position"], g["auroc"], "-", c="#CCCCCC", lw=0.7, alpha=0.8)
            mac = pl.read_parquet(out_dir / "probe_macro_by_position.parquet").filter(
                pl.col("macro_auroc").is_not_null()
            )
            ax.plot(mac["position"], mac["macro_auroc"], "-o", c="#C44E52", lw=2.0,
                    label="macro mean over codes")
            ax.set_xlabel("target-query position (# of prior answers)")
            ax.set_ylabel("within-code AUROC")
            ax.set_title("Matched-code probe: same code at positions 1..P\n(grey = each code; red = macro mean)")
            ax.set_xticks(sorted(pg["target_position"].unique().to_list()))
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figs / "probe_by_position.png", dpi=150)
            plt.close(fig)

    # Legacy pooled AUROC by position (incl. censor at 0) — kept for reference.
    bp = by_pos.filter(pl.col("auroc").is_not_null())
    if bp.height:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(bp["position"].to_list(), bp["auroc"].to_list(), color="#4C72B0")
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlabel("query position in sequence (0 = censor query)")
        ax.set_ylabel("pooled AUROC")
        ax.set_title("Pooled AUROC by sequence position (incl. censor)")
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
