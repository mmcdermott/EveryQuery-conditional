#!/usr/bin/env python
"""Row-for-row and task-for-task comparison of the prediction variants against the stored baseline.

For every (grid, size) on the tuning split and every pair below that has both sides scored, the two
prediction parquets are first checked to be the SAME grid in the SAME row order (subject, prediction
time, the five window lists, target code, label), then compared on the probabilities (distribution
of |delta prob|) and on the per-task AUROCs from ``metrics/<variant>/.../task_metrics.parquet``
(delta per task, delta of the macro mean).  Writes ``metrics/comparison.json`` and
``metrics/comparison.md``; only aggregates are written.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

EXP = Path("/home/gkondas/eq-new-exps/dev-lightning-refactor-check-in")
_spec = importlib.util.spec_from_file_location("analyze", EXP / "scripts" / "20_analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

SIZES = ["small", "base", "large"]
GRIDS = ["canonical", "horizon"]
PAIRS = [
    ("new_fp32", "stored", "dev code at fp32 vs the stored baseline (the refactor, at the baseline's precision)"),
    ("new_bf16", "stored", "dev code at its new default bf16-mixed vs the stored baseline (what the CLI now returns)"),
    ("old_rerun", "stored", "old code re-run today vs the stored baseline (environment: torch 2.11 vs 2.14, run-to-run noise)"),
    ("new_fp32", "old_rerun", "dev code at fp32 vs old code, same venv (the code change alone)"),
    ("new_bf16", "new_fp32", "bf16-mixed vs fp32 under the dev code (the precision default alone)"),
]
ALIGN_COLS = [
    "subject_id", "prediction_time", "queries", "start_durations", "start_events", "durations",
    "bound_events", "target_code", "label",
]
THRESHOLDS = [1e-6, 1e-4, 1e-3, 1e-2, 5e-2]


def load_preds(variant: str, grid: str, size: str) -> pl.DataFrame | None:
    path = analyze.preds_path(variant, grid, "tuning", size)
    return pl.read_parquet(path) if path.is_file() else None


def load_tasks(variant: str, grid: str, size: str) -> pl.DataFrame | None:
    path = EXP / "metrics" / variant / f"{grid}_tuning" / size / "task_metrics.parquet"
    return pl.read_parquet(path).sort("task_index") if path.is_file() else None


def check_aligned(a: pl.DataFrame, b: pl.DataFrame) -> dict:
    """The two frames must be the same grid rows in the same order; report which columns differ."""
    out = {"n_rows_a": int(a.height), "n_rows_b": int(b.height), "same_height": a.height == b.height, "columns_differing": []}
    if not out["same_height"]:
        return out
    for col in ALIGN_COLS:
        xs, ys = a[col].to_list(), b[col].to_list()
        n_bad = sum(1 for x, y in zip(xs, ys, strict=True) if x != y)
        if n_bad:
            out["columns_differing"].append({"column": col, "n_rows_differing": int(n_bad)})
    out["aligned"] = not out["columns_differing"]
    return out


def compare_probs(pa: np.ndarray, pb: np.ndarray) -> dict:
    d = np.abs(pa.astype(np.float64) - pb.astype(np.float64))
    return {
        "n_rows": int(d.size),
        "frac_rows_bitwise_equal": float((d == 0).mean()),
        "max_abs_dprob": float(d.max()),
        "mean_abs_dprob": float(d.mean()),
        "median_abs_dprob": float(np.median(d)),
        "p99_abs_dprob": float(np.quantile(d, 0.99)),
        "frac_rows_abs_dprob_gt": {f"{t:g}": float((d > t).mean()) for t in THRESHOLDS},
        "pearson_prob": float(np.corrcoef(pa, pb)[0, 1]),
    }


def compare_tasks(ta: pl.DataFrame, tb: pl.DataFrame) -> dict:
    if ta["task"].to_list() != tb["task"].to_list():
        raise RuntimeError("task tables list different tasks or a different order")
    aa, ab = ta["auroc"].to_numpy(), tb["auroc"].to_numpy()
    ok = np.isfinite(aa) & np.isfinite(ab)
    d = aa[ok] - ab[ok]
    worst = int(np.argmax(np.abs(d))) if d.size else -1
    worst_row = ta.filter(pl.Series(ok)).row(worst, named=True) if d.size else {}
    return {
        "n_tasks": int(ta.height),
        "n_tasks_defined_both": int(ok.sum()),
        "n_tasks_defined_a": int(np.isfinite(aa).sum()),
        "n_tasks_defined_b": int(np.isfinite(ab).sum()),
        "macro_auroc_a": float(np.nanmean(aa)),
        "macro_auroc_b": float(np.nanmean(ab)),
        "delta_macro_auroc": float(np.nanmean(aa) - np.nanmean(ab)),
        "median_auroc_a": float(np.nanmedian(aa)),
        "median_auroc_b": float(np.nanmedian(ab)),
        "macro_auprc_a": float(np.nanmean(ta["auprc"].to_numpy())),
        "macro_auprc_b": float(np.nanmean(tb["auprc"].to_numpy())),
        "delta_macro_auprc": float(np.nanmean(ta["auprc"].to_numpy()) - np.nanmean(tb["auprc"].to_numpy())),
        "max_abs_dauroc_task": float(np.abs(d).max()) if d.size else float("nan"),
        "mean_dauroc_task": float(d.mean()) if d.size else float("nan"),
        "n_tasks_abs_dauroc_gt": {f"{t:g}": int((np.abs(d) > t).sum()) for t in (1e-4, 1e-3, 5e-3, 1e-2)},
        "n_tasks_a_higher": int((d > 0).sum()),
        "n_tasks_b_higher": int((d < 0).sum()),
        "worst_task": {
            "task_index": int(worst_row.get("task_index", -1)),
            "start_form": worst_row.get("start_form"),
            "end_form": worst_row.get("end_form"),
            "n_pos": worst_row.get("n_pos"),
            "auroc_a": float(aa[ok][worst]) if d.size else None,
            "auroc_b": float(ab[ok][worst]) if d.size else None,
        },
    }


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    results: dict = {}
    md = ["# Refactor check: prediction variants vs the stored 2026-09-05 baseline (tuning split)", ""]
    md.append("Pairs (a vs b):")
    for a, b, why in PAIRS:
        md.append(f"- `{a}` vs `{b}`: {why}")
    md.append("")
    for grid in GRIDS:
        key = f"{grid}_tuning"
        results[key] = {}
        md += [f"## {grid} grid", "", "| size | a | b | aligned | rows bitwise equal | max abs dprob | mean abs dprob | p99 abs dprob | rows abs dprob > 1e-2 | max abs dAUROC (task) | tasks abs dAUROC > 5e-3 | macro AUROC a | macro AUROC b | d macro AUROC | d macro AUPRC |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for size in SIZES:
            results[key][size] = {}
            preds = {v: load_preds(v, grid, size) for v in {p for pair in PAIRS for p in pair[:2]}}
            tasks = {v: load_tasks(v, grid, size) for v in preds}
            for a, b, _ in PAIRS:
                if preds[a] is None or preds[b] is None or tasks[a] is None or tasks[b] is None:
                    continue
                align = check_aligned(preds[a], preds[b])
                entry = {"alignment": align}
                if align.get("aligned"):
                    entry["probs"] = compare_probs(preds[a]["prob"].to_numpy(), preds[b]["prob"].to_numpy())
                    entry["tasks"] = compare_tasks(tasks[a], tasks[b])
                results[key][size][f"{a} vs {b}"] = entry
                if align.get("aligned"):
                    p, t = entry["probs"], entry["tasks"]
                    md.append(
                        f"| {size} | {a} | {b} | yes | {fmt_pct(p['frac_rows_bitwise_equal'])} | {p['max_abs_dprob']:.2e} | "
                        f"{p['mean_abs_dprob']:.2e} | {p['p99_abs_dprob']:.2e} | {fmt_pct(p['frac_rows_abs_dprob_gt']['0.01'])} | "
                        f"{t['max_abs_dauroc_task']:.2e} | {t['n_tasks_abs_dauroc_gt']['0.005']}/{t['n_tasks_defined_both']} | "
                        f"{t['macro_auroc_a']:.4f} | {t['macro_auroc_b']:.4f} | {t['delta_macro_auroc']:+.5f} | {t['delta_macro_auprc']:+.5f} |"
                    )
                else:
                    md.append(f"| {size} | {a} | {b} | NO ({align}) | | | | | | | | | | | |")
        md.append("")
    (EXP / "metrics" / "comparison.json").write_text(json.dumps(results, indent=2))
    (EXP / "metrics" / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    write_per_task()


def write_per_task() -> None:
    """Per-task AUROC of every variant next to the stored baseline, one table per (grid, size)."""
    variants = ["stored", "new_fp32", "new_bf16", "old_rerun"]
    md = ["# Per-task AUROC by variant (tuning split)", ""]
    for grid in GRIDS:
        for size in SIZES:
            tables = {v: load_tasks(v, grid, size) for v in variants}
            tables = {v: t for v, t in tables.items() if t is not None}
            if "stored" not in tables:
                continue
            base = tables["stored"].select(
                "task_index", "start_form", "end_form", "final_duration", "n_pos", "base_rate",
                pl.col("auroc").alias("auroc_stored"),
            )
            for v, t in tables.items():
                if v == "stored":
                    continue
                base = base.join(
                    t.select("task_index", pl.col("auroc").alias(f"auroc_{v}")), on="task_index", how="left"
                ).with_columns((pl.col(f"auroc_{v}") - pl.col("auroc_stored")).alias(f"d_{v}"))
            cols = base.columns
            md += [f"## {grid} / {size}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for row in base.sort("task_index").iter_rows():
                cells = []
                for c, v in zip(cols, row, strict=True):
                    if isinstance(v, float):
                        cells.append("n/a" if v != v else (f"{v:+.4f}" if c.startswith("d_") else f"{v:.4f}"))
                    else:
                        cells.append("∅" if v is None else str(v))
                md.append("| " + " | ".join(cells) + " |")
            md.append("")
    (EXP / "metrics" / "per_task.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
