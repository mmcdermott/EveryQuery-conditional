#!/usr/bin/env python
"""Per-task AUROC and macro averages of one prediction parquet (the old suite's 50_analyze.py, generalized).

``--variant stored`` reads the 2026-09-05 baseline parquets of multitask-canonical-eval (``preds/`` for
the canonical grid, ``preds_horizon/`` for the horizon grid) so the baseline is re-scored by exactly
this code; every other variant reads ``preds/<variant>/<grid>_<split>/<size>.parquet`` of this
experiment.  Metrics land in ``metrics/<variant>/<grid>_<split>/<size>/`` (``task_metrics.parquet``,
``summary.json``).  Only aggregates are written or printed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

EXP = Path("/home/gkondas/eq-new-exps/dev-lightning-refactor-check-in")
OLD_EXP = Path("/home/gkondas/eq-new-exps/multitask-canonical-eval")
TASK_COLS = ["queries", "start_durations", "start_events", "durations", "bound_events"]


def preds_path(variant: str, grid: str, split: str, size: str) -> Path:
    if variant == "stored":
        if split != "tuning":
            raise ValueError("the stored baseline only exists on the tuning split")
        return OLD_EXP / ("preds" if grid == "canonical" else "preds_horizon") / f"{size}.parquet"
    return EXP / "preds" / variant / f"{grid}_{split}" / f"{size}.parquet"


def task_key(row: dict) -> str:
    parts = []
    for col in TASK_COLS:
        parts.append("|".join("∅" if v is None else str(v) for v in row[col]))
    return "//".join(parts)


def window_forms(row: dict) -> tuple[str, str]:
    sd, se = row["start_durations"][-1], row["start_events"][-1]
    if se is not None:
        start = "event"
    elif float(sd) == 0.0:
        start = "prediction_time"
    else:
        start = "delayed"
    end = "event" if row["bound_events"][-1] is not None else "duration"
    return start, end


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def task_table(preds: pl.DataFrame) -> pl.DataFrame:
    """One row per task (in first-appearance order) with its AUROC / AUPRC; identical to the old suite."""
    keys, starts, ends = [], [], []
    for row in preds.select(TASK_COLS).iter_rows(named=True):
        keys.append(task_key(row))
        s, e = window_forms(row)
        starts.append(s)
        ends.append(e)
    preds = preds.with_columns(
        pl.Series("task", keys), pl.Series("start_form", starts), pl.Series("end_form", ends)
    )
    records = []
    for task, g in preds.group_by("task", maintain_order=True):
        (task,) = task
        y = g["label"].to_numpy().astype(bool)
        p = g["prob"].to_numpy().astype(np.float64)
        n_pos = int(y.sum())
        defined = 0 < n_pos < y.size
        first = g.row(0, named=True)
        records.append(
            {
                "task": task,
                "task_index": len(records),
                "start_form": first["start_form"],
                "end_form": first["end_form"],
                "target_code": first["target_code"],
                "final_start_duration": float(first["start_durations"][-1]),
                "final_start_event": first["start_events"][-1],
                "final_duration": float(first["durations"][-1]),
                "final_bound_event": first["bound_events"][-1],
                "n_contexts": int(y.size),
                "n_pos": n_pos,
                "base_rate": float(y.mean()),
                "mean_prob": float(p.mean()),
                "auroc": float(roc_auc_score(y, p)) if defined else float("nan"),
                "auprc": float(average_precision_score(y, p)) if defined else float("nan"),
                "auroc_defined": defined,
            }
        )
    return pl.DataFrame(records)


def summarize(df: pl.DataFrame) -> dict:
    a = df["auroc"].to_numpy().astype(np.float64)
    d = a[np.isfinite(a)]
    lo, hi = bootstrap_ci(a)
    return {
        "n_tasks": int(df.height),
        "n_tasks_defined": int(d.size),
        "macro_auroc": float(d.mean()) if d.size else float("nan"),
        "macro_auroc_ci95": [lo, hi],
        "median_auroc": float(np.median(d)) if d.size else float("nan"),
        "macro_auprc": float(np.nanmean(df["auprc"].to_numpy())) if d.size else float("nan"),
        "mean_base_rate": float(df["base_rate"].mean()),
    }


def build_summary(preds: pl.DataFrame, tasks: pl.DataFrame, **ident) -> dict:
    summary = {
        **ident,
        "n_rows": int(preds.height),
        "n_contexts": int(preds.select("subject_id", "prediction_time").n_unique()),
        "overall": summarize(tasks),
        "by_start_form": {
            k[0]: summarize(g) for k, g in tasks.group_by("start_form", maintain_order=True)
        },
        "by_end_form": {k[0]: summarize(g) for k, g in tasks.group_by("end_form", maintain_order=True)},
        "by_window_type": {
            f"{k[0]}_start/{k[1]}_end": summarize(g)
            for k, g in tasks.sort("start_form", "end_form").group_by(
                "start_form", "end_form", maintain_order=True
            )
        },
    }
    duration_ended = tasks.filter(pl.col("end_form") == "duration")
    if duration_ended.height and duration_ended["final_duration"].n_unique() <= 12:
        summary["by_final_horizon_days"] = {
            f"{int(k[0])}": summarize(g)
            for k, g in duration_ended.sort("final_duration").group_by("final_duration", maintain_order=True)
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="stored | new_bf16 | new_fp32 | old_rerun")
    ap.add_argument("--grid", required=True, choices=["canonical", "horizon"])
    ap.add_argument("--split", required=True, choices=["tuning", "held_out"])
    ap.add_argument("--size", required=True, help="small | base | large, or a retrained run's name")
    args = ap.parse_args()
    preds = pl.read_parquet(preds_path(args.variant, args.grid, args.split, args.size))
    out_dir = EXP / "metrics" / args.variant / f"{args.grid}_{args.split}" / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_table(preds)
    tasks.write_parquet(out_dir / "task_metrics.parquet")
    summary = build_summary(preds, tasks, variant=args.variant, grid=args.grid, split=args.split, size=args.size)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    o = summary["overall"]
    print(
        f"{args.variant}/{args.grid}_{args.split}/{args.size}: {summary['n_rows']} rows, "
        f"{summary['n_contexts']} contexts, {o['n_tasks_defined']}/{o['n_tasks']} tasks with a defined "
        f"AUROC; macro AUROC {o['macro_auroc']:.4f} [{o['macro_auroc_ci95'][0]:.4f}, "
        f"{o['macro_auroc_ci95'][1]:.4f}], median {o['median_auroc']:.4f}, macro AUPRC {o['macro_auprc']:.4f}"
    )


if __name__ == "__main__":
    main()
