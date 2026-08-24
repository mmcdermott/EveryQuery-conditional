# Reports

`EveryQuery_Conditional_Report_FINAL.pdf` — the final report on the converged `big_v2` run
(28.79M unique fixed-length-5 random query sequences, one no-repeat epoch, ~299k steps). Covers the
model, training stability/convergence, evaluation methodology, and three result families:

1. **Macro per-task discrimination & the conditioning trend** (`results/macro_patient_final.json`):
   patient-uniform, occurrence-driven macro AUROC = 0.792 → 0.7945 across positions 0→4 (slope
   +0.00069/pos, 95% CI [+0.00029, +0.00109], excludes 0; Spearman ρ = 0.90). Conditioning helps,
   small but statistically significant.
2. **Designed clinical tasks** (`results/clinical_v2.json`): single-query AUROC across post-admission,
   post-discharge, and random-time anchors (e.g. 30d mortality 0.86 post-admission, 0.91 random-time,
   0.81 post-discharge; ED readmission 0.74; MICU 7d 0.77), plus teacher-forced conditioning demos
   (e.g. forcing MICU-in-7d raises 30d mortality 2.8% → 10.3%).
3. **Original-EveryQuery-comparable uncensored occurs-AUROC** (`results/uncens.json`): macro 0.794
   ([0.790, 0.798]) on contexts where the record does not end within the horizon.

## Reproduce

```bash
cd scripts
EXP=../../experiments
../.venv/bin/python build_report_final.py \
  --train-csv $EXP/runs/big_v2/loggers/csv/version_0/metrics.csv \
  --macro    $EXP/macro_patient_final/summary.json \
  --clinical $EXP/clinical_v2/summary.json \
  --uncens   $EXP/uncens/summary.json \
  --out      ../reports/EveryQuery_Conditional_Report_FINAL.pdf
```

The three summary JSONs are produced by `eval_macro_position.py`, `eval_clinical.py`, and
`eval_occurs_uncensored.py` respectively (see `../docs/CONDITIONAL_QUERIES.md`).
