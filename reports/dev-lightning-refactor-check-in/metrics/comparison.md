# Refactor check: prediction variants vs the stored 2026-09-05 baseline (tuning split)

Pairs (a vs b):
- `new_fp32` vs `stored`: dev code at fp32 vs the stored baseline (the refactor, at the baseline's precision)
- `new_bf16` vs `stored`: dev code at its new default bf16-mixed vs the stored baseline (what the CLI now returns)
- `old_rerun` vs `stored`: old code re-run today vs the stored baseline (environment: torch 2.11 vs 2.14, run-to-run noise)
- `new_fp32` vs `old_rerun`: dev code at fp32 vs old code, same venv (the code change alone)
- `new_bf16` vs `new_fp32`: bf16-mixed vs fp32 under the dev code (the precision default alone)

## canonical grid

| size | a | b | aligned | rows bitwise equal | max abs dprob | mean abs dprob | p99 abs dprob | rows abs dprob > 1e-2 | max abs dAUROC (task) | tasks abs dAUROC > 5e-3 | macro AUROC a | macro AUROC b | d macro AUROC | d macro AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | new_fp32 | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8425 | 0.8425 | +0.00000 | +0.00000 |
| small | new_bf16 | stored | yes | 0.0% | 9.44e-03 | 1.17e-04 | 1.63e-03 | 0.0% | 1.40e-03 | 0/40 | 0.8425 | 0.8425 | -0.00000 | -0.00005 |
| small | old_rerun | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8425 | 0.8425 | +0.00000 | +0.00000 |
| small | new_fp32 | old_rerun | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8425 | 0.8425 | +0.00000 | +0.00000 |
| small | new_bf16 | new_fp32 | yes | 0.0% | 9.44e-03 | 1.17e-04 | 1.63e-03 | 0.0% | 1.40e-03 | 0/40 | 0.8425 | 0.8425 | -0.00000 | -0.00005 |
| base | new_fp32 | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8615 | 0.8615 | +0.00000 | +0.00000 |
| base | new_bf16 | stored | yes | 0.0% | 1.70e-02 | 1.12e-04 | 1.81e-03 | 0.0% | 7.09e-04 | 0/40 | 0.8615 | 0.8615 | +0.00001 | +0.00004 |
| base | old_rerun | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8615 | 0.8615 | +0.00000 | +0.00000 |
| base | new_fp32 | old_rerun | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8615 | 0.8615 | +0.00000 | +0.00000 |
| base | new_bf16 | new_fp32 | yes | 0.0% | 1.70e-02 | 1.12e-04 | 1.81e-03 | 0.0% | 7.09e-04 | 0/40 | 0.8615 | 0.8615 | +0.00001 | +0.00004 |
| large | new_fp32 | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8591 | 0.8591 | +0.00000 | +0.00000 |
| large | new_bf16 | stored | yes | 0.0% | 1.12e-02 | 7.75e-05 | 1.15e-03 | 0.0% | 2.02e-04 | 0/40 | 0.8591 | 0.8591 | -0.00000 | +0.00001 |
| large | old_rerun | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8591 | 0.8591 | +0.00000 | +0.00000 |
| large | new_fp32 | old_rerun | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/40 | 0.8591 | 0.8591 | +0.00000 | +0.00000 |
| large | new_bf16 | new_fp32 | yes | 0.0% | 1.12e-02 | 7.75e-05 | 1.15e-03 | 0.0% | 2.02e-04 | 0/40 | 0.8591 | 0.8591 | -0.00000 | +0.00001 |

## horizon grid

| size | a | b | aligned | rows bitwise equal | max abs dprob | mean abs dprob | p99 abs dprob | rows abs dprob > 1e-2 | max abs dAUROC (task) | tasks abs dAUROC > 5e-3 | macro AUROC a | macro AUROC b | d macro AUROC | d macro AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | new_fp32 | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8692 | 0.8692 | +0.00000 | +0.00000 |
| small | new_bf16 | stored | yes | 0.0% | 2.36e-02 | 1.64e-04 | 2.27e-03 | 0.0% | 2.42e-04 | 0/50 | 0.8692 | 0.8692 | -0.00001 | -0.00003 |
| small | old_rerun | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8692 | 0.8692 | +0.00000 | +0.00000 |
| small | new_fp32 | old_rerun | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8692 | 0.8692 | +0.00000 | +0.00000 |
| small | new_bf16 | new_fp32 | yes | 0.0% | 2.36e-02 | 1.64e-04 | 2.27e-03 | 0.0% | 2.42e-04 | 0/50 | 0.8692 | 0.8692 | -0.00001 | -0.00003 |
| base | new_fp32 | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8857 | 0.8857 | +0.00000 | +0.00000 |
| base | new_bf16 | stored | yes | 0.0% | 8.61e-03 | 1.22e-04 | 1.77e-03 | 0.0% | 1.46e-04 | 0/50 | 0.8857 | 0.8857 | +0.00002 | +0.00006 |
| base | old_rerun | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8857 | 0.8857 | +0.00000 | +0.00000 |
| base | new_fp32 | old_rerun | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8857 | 0.8857 | +0.00000 | +0.00000 |
| base | new_bf16 | new_fp32 | yes | 0.0% | 8.61e-03 | 1.22e-04 | 1.77e-03 | 0.0% | 1.46e-04 | 0/50 | 0.8857 | 0.8857 | +0.00002 | +0.00006 |
| large | new_fp32 | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8893 | 0.8893 | +0.00000 | +0.00000 |
| large | new_bf16 | stored | yes | 0.0% | 3.24e-02 | 1.13e-04 | 1.12e-03 | 0.2% | 1.88e-04 | 0/50 | 0.8893 | 0.8893 | -0.00001 | -0.00003 |
| large | old_rerun | stored | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8893 | 0.8893 | +0.00000 | +0.00000 |
| large | new_fp32 | old_rerun | yes | 100.0% | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% | 0.00e+00 | 0/50 | 0.8893 | 0.8893 | +0.00000 | +0.00000 |
| large | new_bf16 | new_fp32 | yes | 0.0% | 3.24e-02 | 1.13e-04 | 1.12e-03 | 0.2% | 1.88e-04 | 0/50 | 0.8893 | 0.8893 | -0.00001 | -0.00003 |
