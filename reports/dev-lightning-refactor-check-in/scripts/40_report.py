#!/usr/bin/env python
"""Updated evaluation results with the dev code, next to the stored baseline: ``metrics/summary.md``.

Reads every ``metrics/<variant>/<grid>_<split>/<size>/summary.json`` this experiment produced plus the
old suite's own ``summary.json`` files (its 2026-09-05 numbers, computed by its 50_analyze.py under the
experiment venv) and writes one Markdown report: the updated headline table per evaluation set, the
per-stratum breakdowns for the dev default vs the baseline, and the old-vs-new macro deltas.
"""

import json
from pathlib import Path

EXP = Path("/home/gkondas/eq-new-exps/dev-lightning-refactor-check-in")
OLD_EXP = Path("/home/gkondas/eq-new-exps/multitask-canonical-eval")
SIZES = ["small", "base", "large"]
SETS = ["canonical_tuning", "horizon_tuning", "canonical_held_out", "horizon_held_out"]
VARIANTS = ["new_bf16", "new_fp32", "old_rerun", "stored"]


def load(variant: str, grid_split: str, size: str) -> dict | None:
    p = EXP / "metrics" / variant / grid_split / size / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else None


def load_old_suite(grid_split: str, size: str) -> dict | None:
    if grid_split == "canonical_tuning":
        p = OLD_EXP / "metrics" / size / "summary.json"
    elif grid_split == "horizon_tuning":
        p = OLD_EXP / "metrics_horizon" / size / "summary.json"
    else:
        return None
    return json.loads(p.read_text()) if p.is_file() else None


def fmt(s: dict | None) -> str:
    if s is None or s["n_tasks_defined"] == 0:
        return "n/a"
    lo, hi = s["macro_auroc_ci95"]
    ci = f" [{lo:.3f}, {hi:.3f}]" if lo == lo else ""
    return f"{s['macro_auroc']:.4f}{ci}"


def main() -> None:
    lines = ["# Updated multitask evaluation results with the dev code (#30 Lightning refactor)", ""]
    lines.append(
        "Same three checkpoints (small 128x3, base 256x6, large 384x12; seed 140799, trained 2026-09-05) "
        "and the same four evaluation grids as multitask-canonical-eval, re-scored with `EQ_predict_multitask` "
        "from origin/dev (0b90790).  `new_bf16` is the CLI's new default (`Trainer.predict`, precision "
        "bf16-mixed); `new_fp32` passes `precision=32-true`; `old_rerun` is the pre-refactor code (de0f5ee) "
        "re-run in an identical venv; `stored` is the 2026-09-05 baseline parquet re-scored by this "
        "experiment's analysis script; `old suite` is the number the old suite itself reported.  Macro AUROC "
        "= mean over tasks of the within-task AUROC of the final query; 95% CIs bootstrap over tasks."
    )
    for grid_split in SETS:
        any_summary = next((load(v, grid_split, s) for v in VARIANTS for s in SIZES if load(v, grid_split, s)), None)
        if any_summary is None:
            continue
        o = any_summary["overall"]
        lines += ["", f"## {grid_split.replace('_', ' / ', 1)}", ""]
        lines.append(
            f"{o['n_tasks']} tasks x {any_summary['n_contexts']} contexts = {any_summary['n_rows']} scored rows per model."
        )
        lines += ["", "| size | variant | macro AUROC [95% CI] | tasks defined | median AUROC | macro AUPRC |", "|---|---|---|---|---|---|"]
        for size in SIZES:
            for variant in VARIANTS:
                s = load(variant, grid_split, size)
                if s is None:
                    continue
                o = s["overall"]
                lines.append(
                    f"| {size} | {variant} | {fmt(o)} | {o['n_tasks_defined']}/{o['n_tasks']} | "
                    f"{o['median_auroc']:.4f} | {o['macro_auprc']:.4f} |"
                )
            old = load_old_suite(grid_split, size)
            if old is not None:
                o = old["overall"]
                lines.append(
                    f"| {size} | old suite (2026-09-05 report) | {fmt(o)} | {o['n_tasks_defined']}/{o['n_tasks']} | "
                    f"{o['median_auroc']:.4f} | {o['macro_auprc']:.4f} |"
                )
        # Per-stratum breakdown: dev default vs stored (tuning) or dev default alone (held_out).
        for section, key in (
            ("By window type (start form / end form)", "by_window_type"),
            ("By start form", "by_start_form"),
            ("By end form", "by_end_form"),
            ("By horizon of the scored window (days)", "by_final_horizon_days"),
        ):
            if key not in any_summary:
                continue
            groups = list(any_summary[key])
            has_stored = any(load("stored", grid_split, s) for s in SIZES)
            head = "| stratum (tasks) | " + " | ".join(
                f"{s} new_bf16" + (f" | {s} stored | {s} delta" if has_stored else "") for s in SIZES
            ) + " |"
            lines += ["", f"### {section}", "", head, "|---|" + "---|" * (len(SIZES) * (3 if has_stored else 1))]
            for g in groups:
                cells = []
                for s in SIZES:
                    new, stored = load("new_bf16", grid_split, s), load("stored", grid_split, s)
                    n = new[key][g]["macro_auroc"] if new and g in new.get(key, {}) else float("nan")
                    cells.append(f"{n:.4f}")
                    if has_stored:
                        b = stored[key][g]["macro_auroc"] if stored and g in stored.get(key, {}) else float("nan")
                        cells += [f"{b:.4f}", f"{n - b:+.4f}"]
                n_tasks = any_summary[key][g]["n_tasks"]
                lines.append(f"| {g} ({n_tasks}) | " + " | ".join(cells) + " |")
    cmp_path = EXP / "metrics" / "comparison.md"
    if cmp_path.is_file():
        lines += ["", "---", ""] + cmp_path.read_text().splitlines()
    (EXP / "metrics" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
