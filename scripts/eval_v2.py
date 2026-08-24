#!/usr/bin/env python
"""Evaluation for the v2 conditional model (binary observed-occurrence; censoring via EOS query).

Critically reviews the query forms the architecture makes available and measures each:

A. **Marginal occurrence** ``[C, d]`` — P(C observed in (t,t+d)).  Headline = within-query AUROC
   (computed inside each code, macro-averaged) since pooled AUROC is base-rate inflated.

B. **EOS-conditioned censoring control** ``[TIMELINE//END, d] [C, d]`` — the same target C is scored
   three ways on identical (patient, t) contexts by overriding the teacher-forced EOS answer:
     - EOS forced NO  → P(C | data continue past t+d)  ≈ original EveryQuery's implicit question;
     - EOS forced YES → P(C | record ends within d)    ≈ the actionable form for terminal events;
     - marginal       → P(C) with no EOS prefix.
   For a terminal code (MEDS_DEATH) we expect P(death | EOS=NO) ≈ 0 and P(death | EOS=YES) large —
   i.e. the conditioning *recovers* death prediction that the marginal/old-EQ form cannot express.

C. **Informative-prior conditioning (nested horizons)** ``[C, d_short] [C, d_long]`` — teacher-force
   the *true* C@d_short answer; if the model uses conditioning it must respect the logical
   entailment C@d_short=YES ⇒ C@d_long=YES (P→1) and separate the two strata.  This is the test the
   v1 random-filler probe could not do (its priors were uninformative).

All three are run on held-out subjects; sequences are built + binary-labeled here from the
intermediate event shards, then scored with the trained encoder/decoder.

Status (conditional-v2): repaired against the 5-stage sampler.  ``sample_eval_contexts`` no longer
draws contexts shard-locally — see its docstring for the two semantic changes (global context
budget; ``min_prediction_times`` counts prediction times, not events).  The numbers in
``EveryQuery_Conditional_Report_v2.pdf`` were produced by the *old* sampler and by a checkpoint
trained on integer-day durations; they cannot be reproduced from this file (see
``docs/history/2026-08-18-conditional-v2-integration-plan.md`` §3).  ``eval_v3.py`` supersedes this script's coarse
per-query view with dense per-conditional grids.
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
from meds_torchdata.config import MEDSTorchDataConfig
from sklearn.metrics import roc_auc_score

from every_query.data.schema import QuerySeqSchema
from every_query.data.seq_dataset import ANSWER_NO, ANSWER_YES, EOS_CODE, ConditionalQueryPytorchDataset
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    label_binary_occurrence,
    resolve_prediction_times,
)
from every_query.generate_tasks.sample_tasks import (
    _read_event_shard,
    build_prediction_times,
    prediction_time_counts_path,
    sample_patient_contexts,
)
from every_query.model.conditional_lightning import ConditionalQueryLightningModule
from every_query.utils.seeds import derive_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Curated, prevalence-spanning target codes (resolved against the cohort vocab at runtime).
CURATED_PREFIXES = [
    ("MEDS_DEATH", 1), ("HOSPITAL_ADMISSION//", 2), ("HOSPITAL_DISCHARGE//", 2),
    ("ICU_ADMISSION//", 2), ("MEDICATION//", 3), ("LAB//", 3), ("DIAGNOSIS//", 2),
]


def _auroc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(p)))


def curated_codes(processed: Path) -> list[str]:
    df = pl.read_parquet(processed / "metadata" / "codes.parquet", columns=["code", "code/n_subjects"])
    out: list[str] = []
    for pref, k in CURATED_PREFIXES:
        sub = df.filter(pl.col("code").str.starts_with(pref) | (pl.col("code") == pref))
        out += sub.sort("code/n_subjects", descending=True).head(k)["code"].to_list()
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def load_model(run_dir: Path) -> ConditionalQueryLightningModule:
    ckpt = run_dir / "best_model.ckpt"
    if not ckpt.is_file():
        ckpt = run_dir / "checkpoints" / "last.ckpt"
    m = ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt))
    return m.eval().to(DEVICE)


# ── Build + label probe sequences from the event shards ─────────────────


DEFAULT_MIN_PREDICTION_TIMES = 10


def log_uniform_durations(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` log-uniform horizons in days over ``[lo, hi]`` as **floats**.

    Replaces ``sample_query_sequences.sample_log_uniform_durations``, removed in the Phase 2
    rewrite.  That function returned ``round(exp(U(log lo, log hi)))`` — *integer* days; the
    5-stage sampler's ``QueryDistribution.sample`` draws continuous floats and no longer rounds
    (``docs/history/2026-08-18-conditional-v2-integration-plan.md`` §3), so this matches what the current sampler puts in
    the training distribution.  Kept here as a plain draw rather than routed through
    ``QueryDistribution.sample`` because that also draws a *code* per duration, which would fork the
    rng stream these scripts pair with their own code draws.
    """
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


def sample_eval_contexts(
    intermediate: Path,
    split: str,
    n_contexts: int,
    seed: int,
    artifacts_dir: Path,
    min_prediction_times: int = DEFAULT_MIN_PREDICTION_TIMES,
    limit_shards: int | None = None,
) -> dict:
    """Sample ``n_contexts`` (subject, prediction_time) pairs; return them plus per-shard events.

    Rebuilt on the 5-stage sampler.  The removed ``sample_tasks.sample_contexts`` drew contexts
    **shard-locally** straight off an events frame; the replacement is Stage 0
    (:func:`build_prediction_times`) -> Stage 2 (:func:`sample_patient_contexts`) -> Stage 3'
    (:func:`resolve_prediction_times`), and it differs in two ways a caller must not paper over:

    * ``n_contexts`` is a **global** budget across the whole split — the eligible subject universe
      spans every shard — not a per-shard count.  The old ``--n-per-shard`` flag was renamed rather
      than silently reinterpreted, so an old command line now fails instead of sampling
      ``n_shards``x fewer contexts.
    * ``min_prediction_times`` counts **distinct prediction times**; the old
      ``min_context_per_subject`` counted *events*.  A subject with 10 events rarely has 10 distinct
      timestamps, so carrying the same number across is a strictly stronger eligibility filter.
      The default is unchanged only because these evals are qualitative; treat it as a new knob.

    Stage 0 scans the split once and caches its map under ``artifacts_dir``.  Point
    ``artifacts_dir`` at the ``_artifacts`` tree ``EQ_generate_query_sequences`` already wrote for
    this split and ``min_prediction_times`` and the scan collapses to a cache read.
    ``limit_shards`` is applied to the eligible universe *before* the draw, so it still subsamples
    cheaply without changing how many contexts come back.

    Returns:
        ``{"contexts": DataFrame(subject_id, prediction_time, _shard), "events": {shard: events}}``.
        ``_shard`` is the shard **name** (a string), not an enumeration index — derive any per-shard
        seed with :func:`derive_seed`, not integer arithmetic.
    """
    build_prediction_times(
        path_to_data=intermediate,
        training_task_artifacts_dir=artifacts_dir,
        split=split,
        min_prediction_times_per_subject=min_prediction_times,
    )
    counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, split))
    if limit_shards:
        keep = sorted(counts["shard"].unique().to_list())[:limit_shards]
        # Re-sort by subject_id: Stage 2 gathers by row position, so the counts table must stay
        # sorted by subject_id after any filtering.
        counts = counts.filter(pl.col("shard").is_in(keep)).sort("subject_id")

    contexts = sample_patient_contexts(
        counts,
        n=n_contexts,
        min_prediction_times_per_subject=min_prediction_times,
        rng=np.random.default_rng(derive_seed(seed, "contexts")),
    )

    ctxs, events_by_shard = [], {}
    for shard, part in contexts.group_by("shard"):
        shard = shard[0] if isinstance(shard, tuple) else shard
        resolved = resolve_prediction_times(part, artifacts_dir, split, shard)
        ctxs.append(
            resolved.select("subject_id", "prediction_time").with_columns(
                pl.lit(shard).alias("_shard")
            )
        )
        events_by_shard[shard] = _read_event_shard(intermediate / "data" / split / f"{shard}.parquet")
    return {"contexts": pl.concat(ctxs, how="vertical"), "events": events_by_shard}


def add_context_sampling_args(ap: argparse.ArgumentParser, default_n_contexts: int) -> None:
    """Register the flags :func:`contexts_from_args` reads (shared by the ``eval_*`` scripts)."""
    ap.add_argument(
        "--n-contexts", type=int, default=default_n_contexts,
        help="Global number of (subject, prediction_time) contexts to draw across the split. "
             "Replaces --n-per-shard, which was a per-shard count.",
    )
    ap.add_argument(
        "--min-prediction-times", type=int, default=DEFAULT_MIN_PREDICTION_TIMES,
        help="Minimum distinct prior prediction times a subject needs to be eligible. Replaces "
             "--min-context-per-subject, which counted events.",
    )
    ap.add_argument(
        "--artifacts-dir", type=Path, default=None,
        help="Where the Stage 0 prediction-time map lives. Defaults to "
             "<out-dir>/_prediction_time_artifacts; point it at the sampler's _artifacts tree for "
             "this split to reuse that cache instead of rescanning.",
    )
    ap.add_argument(
        "--limit-shards", type=int, default=None,
        help="Restrict the eligible subject universe to the first N shards (cheap subsampling).",
    )


def contexts_from_args(args, out_dir: Path, seed: int) -> dict:
    """Call :func:`sample_eval_contexts` from an ``argparse`` namespace built by the helper above."""
    return sample_eval_contexts(
        args.intermediate,
        args.split,
        args.n_contexts,
        seed,
        artifacts_dir=args.artifacts_dir or (Path(out_dir) / "_prediction_time_artifacts"),
        min_prediction_times=args.min_prediction_times,
        limit_shards=args.limit_shards,
    )


def build_labeled_sequences(ctx_events: dict, spec_fn) -> pl.DataFrame:
    """Build + binary-label QuerySeqSchema sequences. ``spec_fn(rng)`` -> list[(code, dur)] per ctx."""
    frames = []
    for si, ev in ctx_events["events"].items():
        ctx = ctx_events["contexts"].filter(pl.col("_shard") == si).drop("_shard")
        if not ctx.height:
            continue
        rng = np.random.default_rng(derive_seed(1000, si))
        rows = []
        for cid, c in enumerate(ctx.iter_rows(named=True)):
            for pos, (code, dur) in enumerate(spec_fn(rng)):
                rows.append((cid, pos, c["subject_id"], c["prediction_time"], code, float(dur)))
        idf = pl.DataFrame(
            rows, orient="row",
            schema=[CTX_ID_COL, POSITION_COL, "subject_id", "prediction_time", "query", "duration_days"],
        ).with_columns(
            pl.col(CTX_ID_COL).cast(pl.UInt32), pl.col(POSITION_COL).cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")), pl.col("duration_days").cast(pl.Float32),
        )
        frames.append(label_binary_occurrence(idf, ev))
    return pl.concat(frames, how="vertical")


def write_tasks(df: pl.DataFrame, out_dir: Path, split: str) -> Path:
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    pl.from_arrow(QuerySeqSchema.align(df.to_arrow())).write_parquet(d / "tasks.parquet")
    return out_dir


# ── Scoring with optional teacher-forced-prior override ─────────────────


@torch.no_grad()
def score_last(model, cohort_dir, tasks_dir, split, force_prior=None, batch_size=256):
    """Return a frame of (subject_id, prediction_time, target_query, target_dur, true_answer, prob)
    for the LAST query of each sequence.  ``force_prior`` in {None, 0, 1}: if set, all prior-position
    teacher-forced answers are overridden to that class (counterfactual conditioning)."""
    from torch.utils.data import DataLoader

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cohort_dir), task_labels_dir=str(tasks_dir), max_seq_len=256,
        seq_sampling_strategy="to_end", static_inclusion_mode="omit", batch_mode="SM",
    )
    ds = ConditionalQueryPytorchDataset(cfg, split=split)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate)
    last_probs = []
    for batch in dl:
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

    sdf = ds.schema_df
    return sdf.select(
        "subject_id", "prediction_time",
        pl.col("queries").list.last().alias("target_query"),
        pl.col("durations").list.last().alias("target_dur"),
        pl.col("answers").list.last().alias("true_answer"),
    ).with_columns(pl.Series("prob", last_probs, dtype=pl.Float64))


def macro_within_query(df: pl.DataFrame, prob_col="prob", min_pos=10, min_neg=10):
    rows = []
    for (code,), g in df.group_by(["target_query"]):
        y = g["true_answer"].to_list()
        npos = int(sum(y))
        if npos >= min_pos and (len(y) - npos) >= min_neg:
            rows.append({"query": code, "n": len(y), "prev": npos / len(y),
                         "auroc": _auroc(y, g[prob_col].to_list())})
    t = pl.DataFrame(rows)
    macro = float(t.filter(pl.col("auroc").is_not_null())["auroc"].mean()) if t.height else None
    return t, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cohort-dir", required=True, type=Path)
    ap.add_argument("--intermediate", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--batch-size", type=int, default=256)
    add_context_sampling_args(ap, default_n_contexts=4096)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figs").mkdir(exist_ok=True)
    work = args.out_dir / "_tasks"
    model = load_model(args.run_dir)
    codes = curated_codes(args.processed)
    print(f"{len(codes)} curated codes; device {DEVICE}")
    summary: dict = {"curated_codes": codes}

    ce = contexts_from_args(args, args.out_dir, seed=7)
    print(f"sampled {ce['contexts'].height} contexts")

    def sd(model, td):
        return score_last(model, args.cohort_dir, td, args.split, batch_size=args.batch_size)

    # ── A. Marginal occurrence: [C, 30d] singletons over curated codes ──
    D = 30.0
    def marginal_spec(rng):
        return [(codes[int(rng.integers(len(codes)))], D)]
    td_marg = write_tasks(build_labeled_sequences(ce, marginal_spec), work / "marginal", args.split)
    marg = sd(model, td_marg)
    marg_tbl, marg_macro = macro_within_query(marg)
    marg_tbl.write_parquet(args.out_dir / "marginal_by_query.parquet")
    summary["A_marginal"] = {"macro_within_query_auroc": marg_macro, "n_codes": marg_tbl.height,
                             "n_sequences": marg.height}
    print("A marginal macro within-query AUROC:", marg_macro)

    # ── B. EOS-conditioned: [EOS 30d][C 30d], same (patient, code) ──
    def eos_spec(rng):
        return [(EOS_CODE, D), (codes[int(rng.integers(len(codes)))], D)]
    td_eos = write_tasks(build_labeled_sequences(ce, eos_spec), work / "eos", args.split)
    eos_true = score_last(model, args.cohort_dir, td_eos, args.split, force_prior=None, batch_size=args.batch_size)
    eos_no = score_last(model, args.cohort_dir, td_eos, args.split, force_prior=ANSWER_NO, batch_size=args.batch_size)
    eos_yes = score_last(model, args.cohort_dir, td_eos, args.split, force_prior=ANSWER_YES, batch_size=args.batch_size)
    # align by row order (same dataset, same order)
    b = eos_true.rename({"prob": "prob_true"}).with_columns(
        pl.Series("prob_eosNO", eos_no["prob"]), pl.Series("prob_eosYES", eos_yes["prob"]))
    b.write_parquet(args.out_dir / "eos_conditioned.parquet")
    brows = []
    for (code,), g in b.group_by(["target_query"]):
        y = g["true_answer"].to_list()
        brows.append({
            "query": code, "n": len(y), "prevalence": float(np.mean(y)),
            "mean_P_eosNO": float(g["prob_eosNO"].mean()), "mean_P_eosYES": float(g["prob_eosYES"].mean()),
            "auroc_true_eos": _auroc(y, g["prob_true"].to_list()),
        })
    b_tbl = pl.DataFrame(brows).sort("query")
    b_tbl.write_parquet(args.out_dir / "eos_by_query.parquet")
    summary["B_eos"] = b_tbl.to_dicts()
    death = b_tbl.filter(pl.col("query") == "MEDS_DEATH")
    if death.height:
        r = death.row(0, named=True)
        summary["B_death_highlight"] = {"P_death_given_data_continues": r["mean_P_eosNO"],
                                        "P_death_given_record_ends": r["mean_P_eosYES"],
                                        "prevalence": r["prevalence"]}
        print("B death: P(death|EOS=NO)=%.4f  P(death|EOS=YES)=%.4f" % (r["mean_P_eosNO"], r["mean_P_eosYES"]))

    # ── C. Informative prior (nested horizons): [C 7d][C 30d] ──
    DS, DL = 7.0, 30.0
    def nested_spec(rng):
        c = codes[int(rng.integers(len(codes)))]
        return [(c, DS), (c, DL)]
    td_nest = write_tasks(build_labeled_sequences(ce, nested_spec), work / "nested", args.split)
    # need the prior (C@7d) answer too: re-derive from the labeled tasks (first answer)
    nest = score_last(model, args.cohort_dir, td_nest, args.split, force_prior=None, batch_size=args.batch_size)
    nest_tasks = pl.read_parquet(work / "nested" / args.split / "tasks.parquet")
    nest = nest.with_columns(pl.Series("prior_answer", nest_tasks["answers"].list.first()))
    nest.write_parquet(args.out_dir / "nested_horizons.parquet")
    # monotonicity: among prior(C@7d)=YES, true target(C@30d) must be YES; measure model P(target)
    yes = nest.filter(pl.col("prior_answer"))
    no = nest.filter(~pl.col("prior_answer"))
    summary["C_nested"] = {
        "n": nest.height,
        "frac_prior_yes": float(nest["prior_answer"].mean()),
        "mean_P_target_given_prior_yes": float(yes["prob"].mean()) if yes.height else None,
        "mean_P_target_given_prior_no": float(no["prob"].mean()) if no.height else None,
        "true_target_rate_given_prior_yes": float(np.mean(yes["true_answer"].to_list())) if yes.height else None,
    }
    print("C nested: P(C@30|C@7=YES)=%s  P(C@30|C@7=NO)=%s" % (
        summary["C_nested"]["mean_P_target_given_prior_yes"], summary["C_nested"]["mean_P_target_given_prior_no"]))

    _figures(args.out_dir, b_tbl, marg_tbl, nest)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", args.out_dir / "summary.json")


def _figures(out_dir, b_tbl, marg_tbl, nest):
    figs = out_dir / "figs"
    # B: P(C|EOS=YES) vs P(C|EOS=NO) per code
    bt = b_tbl.sort("mean_P_eosYES", descending=True)
    if bt.height:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * bt.height)))
        yloc = np.arange(bt.height)
        ax.barh(yloc + 0.2, bt["mean_P_eosYES"].to_list(), height=0.4, color="#C44E52", label="EOS=YES (record ends)")
        ax.barh(yloc - 0.2, bt["mean_P_eosNO"].to_list(), height=0.4, color="#4C72B0", label="EOS=NO (data continue)")
        ax.set_yticks(yloc)
        ax.set_yticklabels([q[:38] for q in bt["query"].to_list()], fontsize=7)
        ax.set_xlabel("mean predicted P(code observed in 30d)")
        ax.set_title("Censoring control via EOS conditioning: P(C | record ends) vs P(C | data continue)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs / "eos_conditioning.png", dpi=150); plt.close(fig)
    # C: nested-horizon conditioning histogram
    if nest.height and nest["prior_answer"].sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(nest.filter(pl.col("prior_answer"))["prob"].to_numpy(), bins=30, alpha=0.6,
                color="#C44E52", label="C@7d = YES", density=True)
        ax.hist(nest.filter(~pl.col("prior_answer"))["prob"].to_numpy(), bins=30, alpha=0.6,
                color="#4C72B0", label="C@7d = NO", density=True)
        ax.set_xlabel("model P(C observed in 30d)")
        ax.set_ylabel("density")
        ax.set_title("Informative-prior conditioning (nested horizons)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs / "nested_conditioning.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

