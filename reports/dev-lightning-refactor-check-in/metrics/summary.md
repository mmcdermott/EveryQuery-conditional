# Updated multitask evaluation results with the dev code (#30 Lightning refactor)

Same three checkpoints (small 128x3, base 256x6, large 384x12; seed 140799, trained 2026-09-05) and the same four evaluation grids as multitask-canonical-eval, re-scored with `EQ_predict_multitask` from origin/dev (0b90790).  `new_bf16` is the CLI's new default (`Trainer.predict`, precision bf16-mixed); `new_fp32` passes `precision=32-true`; `old_rerun` is the pre-refactor code (de0f5ee) re-run in an identical venv; `stored` is the 2026-09-05 baseline parquet re-scored by this experiment's analysis script; `old suite` is the number the old suite itself reported.  Macro AUROC = mean over tasks of the within-task AUROC of the final query; 95% CIs bootstrap over tasks.

## canonical / tuning

40 tasks x 14132 contexts = 565280 scored rows per model.

| size | variant | macro AUROC [95% CI] | tasks defined | median AUROC | macro AUPRC |
|---|---|---|---|---|---|
| small | new_bf16 | 0.8425 [0.814, 0.868] | 40/40 | 0.8486 | 0.1551 |
| small | new_fp32 | 0.8425 [0.814, 0.868] | 40/40 | 0.8485 | 0.1552 |
| small | old_rerun | 0.8425 [0.814, 0.868] | 40/40 | 0.8485 | 0.1552 |
| small | stored | 0.8425 [0.814, 0.868] | 40/40 | 0.8485 | 0.1552 |
| small | old suite (2026-09-05 report) | 0.8425 [0.814, 0.868] | 40/40 | 0.8485 | 0.1552 |
| base | new_bf16 | 0.8615 [0.837, 0.885] | 40/40 | 0.8856 | 0.1757 |
| base | new_fp32 | 0.8615 [0.837, 0.885] | 40/40 | 0.8856 | 0.1757 |
| base | old_rerun | 0.8615 [0.837, 0.885] | 40/40 | 0.8856 | 0.1757 |
| base | stored | 0.8615 [0.837, 0.885] | 40/40 | 0.8856 | 0.1757 |
| base | old suite (2026-09-05 report) | 0.8615 [0.837, 0.885] | 40/40 | 0.8856 | 0.1757 |
| large | new_bf16 | 0.8591 [0.836, 0.882] | 40/40 | 0.8634 | 0.1739 |
| large | new_fp32 | 0.8591 [0.836, 0.882] | 40/40 | 0.8634 | 0.1739 |
| large | old_rerun | 0.8591 [0.836, 0.882] | 40/40 | 0.8634 | 0.1739 |
| large | stored | 0.8591 [0.836, 0.882] | 40/40 | 0.8634 | 0.1739 |
| large | old suite (2026-09-05 report) | 0.8591 [0.836, 0.882] | 40/40 | 0.8634 | 0.1739 |

### By window type (start form / end form)

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| delayed_start/duration_end (3) | 0.7672 | 0.7671 | +0.0002 | 0.7958 | 0.7958 | -0.0001 | 0.8128 | 0.8128 | -0.0000 |
| delayed_start/event_end (7) | 0.8274 | 0.8274 | +0.0000 | 0.8490 | 0.8490 | +0.0000 | 0.8482 | 0.8482 | -0.0000 |
| event_start/duration_end (14) | 0.8586 | 0.8587 | -0.0001 | 0.8740 | 0.8740 | +0.0000 | 0.8643 | 0.8643 | -0.0000 |
| event_start/event_end (6) | 0.8400 | 0.8399 | +0.0001 | 0.8666 | 0.8666 | +0.0000 | 0.8642 | 0.8641 | +0.0001 |
| prediction_time_start/duration_end (6) | 0.8604 | 0.8604 | -0.0000 | 0.8793 | 0.8793 | +0.0000 | 0.8789 | 0.8790 | -0.0000 |
| prediction_time_start/event_end (4) | 0.8456 | 0.8456 | +0.0000 | 0.8546 | 0.8546 | -0.0000 | 0.8569 | 0.8569 | -0.0000 |

### By start form

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| prediction_time (10) | 0.8545 | 0.8545 | -0.0000 | 0.8694 | 0.8694 | +0.0000 | 0.8701 | 0.8701 | -0.0000 |
| event (20) | 0.8530 | 0.8530 | -0.0000 | 0.8718 | 0.8718 | +0.0000 | 0.8643 | 0.8643 | +0.0000 |
| delayed (10) | 0.8094 | 0.8093 | +0.0001 | 0.8330 | 0.8330 | -0.0000 | 0.8376 | 0.8376 | -0.0000 |

### By end form

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| event (17) | 0.8362 | 0.8361 | +0.0000 | 0.8565 | 0.8565 | +0.0000 | 0.8559 | 0.8559 | +0.0000 |
| duration (23) | 0.8471 | 0.8472 | -0.0000 | 0.8652 | 0.8652 | +0.0000 | 0.8614 | 0.8614 | -0.0000 |

## horizon / tuning

50 tasks x 14132 contexts = 706600 scored rows per model.

| size | variant | macro AUROC [95% CI] | tasks defined | median AUROC | macro AUPRC |
|---|---|---|---|---|---|
| small | new_bf16 | 0.8692 [0.848, 0.890] | 50/50 | 0.8938 | 0.2182 |
| small | new_fp32 | 0.8692 [0.848, 0.890] | 50/50 | 0.8938 | 0.2183 |
| small | old_rerun | 0.8692 [0.848, 0.890] | 50/50 | 0.8938 | 0.2183 |
| small | stored | 0.8692 [0.848, 0.890] | 50/50 | 0.8938 | 0.2183 |
| small | old suite (2026-09-05 report) | 0.8692 [0.848, 0.890] | 50/50 | 0.8938 | 0.2183 |
| base | new_bf16 | 0.8857 [0.867, 0.904] | 50/50 | 0.8989 | 0.2488 |
| base | new_fp32 | 0.8857 [0.867, 0.904] | 50/50 | 0.8990 | 0.2487 |
| base | old_rerun | 0.8857 [0.867, 0.904] | 50/50 | 0.8990 | 0.2487 |
| base | stored | 0.8857 [0.867, 0.904] | 50/50 | 0.8990 | 0.2487 |
| base | old suite (2026-09-05 report) | 0.8857 [0.867, 0.904] | 50/50 | 0.8990 | 0.2487 |
| large | new_bf16 | 0.8893 [0.872, 0.906] | 50/50 | 0.9078 | 0.2531 |
| large | new_fp32 | 0.8893 [0.872, 0.906] | 50/50 | 0.9078 | 0.2532 |
| large | old_rerun | 0.8893 [0.872, 0.906] | 50/50 | 0.9078 | 0.2532 |
| large | stored | 0.8893 [0.872, 0.906] | 50/50 | 0.9078 | 0.2532 |
| large | old suite (2026-09-05 report) | 0.8893 [0.872, 0.906] | 50/50 | 0.9078 | 0.2532 |

### By window type (start form / end form)

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| delayed_start/duration_end (13) | 0.8598 | 0.8598 | -0.0000 | 0.8714 | 0.8714 | +0.0000 | 0.8782 | 0.8782 | -0.0000 |
| event_start/duration_end (18) | 0.8928 | 0.8927 | +0.0000 | 0.9039 | 0.9039 | +0.0000 | 0.9032 | 0.9032 | -0.0000 |
| prediction_time_start/duration_end (19) | 0.8533 | 0.8534 | -0.0000 | 0.8782 | 0.8781 | +0.0000 | 0.8837 | 0.8837 | +0.0000 |

### By start form

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| delayed (13) | 0.8598 | 0.8598 | -0.0000 | 0.8714 | 0.8714 | +0.0000 | 0.8782 | 0.8782 | -0.0000 |
| event (18) | 0.8928 | 0.8927 | +0.0000 | 0.9039 | 0.9039 | +0.0000 | 0.9032 | 0.9032 | -0.0000 |
| prediction_time (19) | 0.8533 | 0.8534 | -0.0000 | 0.8782 | 0.8781 | +0.0000 | 0.8837 | 0.8837 | +0.0000 |

### By end form

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| duration (50) | 0.8692 | 0.8692 | -0.0000 | 0.8857 | 0.8857 | +0.0000 | 0.8893 | 0.8893 | -0.0000 |

### By horizon of the scored window (days)

| stratum (tasks) | small new_bf16 | small stored | small delta | base new_bf16 | base stored | base delta | large new_bf16 | large stored | large delta |
|---|---|---|---|---|---|---|---|---|---|
| 30 (7) | 0.8466 | 0.8467 | -0.0000 | 0.8648 | 0.8647 | +0.0000 | 0.8746 | 0.8747 | -0.0000 |
| 90 (11) | 0.8932 | 0.8933 | -0.0000 | 0.9124 | 0.9124 | -0.0000 | 0.9107 | 0.9107 | -0.0000 |
| 180 (12) | 0.8726 | 0.8727 | -0.0000 | 0.8885 | 0.8885 | -0.0000 | 0.8892 | 0.8892 | -0.0000 |
| 365 (9) | 0.8595 | 0.8595 | +0.0000 | 0.8789 | 0.8788 | +0.0000 | 0.8895 | 0.8895 | -0.0000 |
| 730 (6) | 0.8589 | 0.8589 | +0.0000 | 0.8704 | 0.8704 | +0.0000 | 0.8728 | 0.8728 | +0.0000 |
| 1095 (5) | 0.8693 | 0.8693 | +0.0000 | 0.8801 | 0.8800 | +0.0001 | 0.8825 | 0.8825 | -0.0000 |

## canonical / held_out

40 tasks x 14211 contexts = 568440 scored rows per model.

| size | variant | macro AUROC [95% CI] | tasks defined | median AUROC | macro AUPRC |
|---|---|---|---|---|---|
| small | new_bf16 | 0.8455 [0.820, 0.871] | 40/40 | 0.8566 | 0.1623 |
| base | new_bf16 | 0.8611 [0.836, 0.884] | 40/40 | 0.8762 | 0.1855 |
| large | new_bf16 | 0.8597 [0.835, 0.883] | 40/40 | 0.8679 | 0.1828 |

### By window type (start form / end form)

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| delayed_start/duration_end (3) | 0.7945 | 0.8159 | 0.8147 |
| delayed_start/event_end (7) | 0.8308 | 0.8480 | 0.8468 |
| event_start/duration_end (14) | 0.8568 | 0.8751 | 0.8687 |
| event_start/event_end (6) | 0.8471 | 0.8568 | 0.8610 |
| prediction_time_start/duration_end (6) | 0.8576 | 0.8748 | 0.8742 |
| prediction_time_start/event_end (4) | 0.8492 | 0.8547 | 0.8602 |

### By start form

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| prediction_time (10) | 0.8542 | 0.8667 | 0.8686 |
| event (20) | 0.8539 | 0.8696 | 0.8664 |
| delayed (10) | 0.8199 | 0.8384 | 0.8372 |

### By end form

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| event (17) | 0.8409 | 0.8527 | 0.8550 |
| duration (23) | 0.8489 | 0.8673 | 0.8631 |

## horizon / held_out

50 tasks x 14211 contexts = 710550 scored rows per model.

| size | variant | macro AUROC [95% CI] | tasks defined | median AUROC | macro AUPRC |
|---|---|---|---|---|---|
| small | new_bf16 | 0.8613 [0.841, 0.882] | 50/50 | 0.8834 | 0.2171 |
| base | new_bf16 | 0.8791 [0.861, 0.897] | 50/50 | 0.8941 | 0.2489 |
| large | new_bf16 | 0.8823 [0.865, 0.899] | 50/50 | 0.8954 | 0.2553 |

### By window type (start form / end form)

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| delayed_start/duration_end (13) | 0.8569 | 0.8684 | 0.8715 |
| event_start/duration_end (18) | 0.8756 | 0.8907 | 0.8917 |
| prediction_time_start/duration_end (19) | 0.8509 | 0.8755 | 0.8809 |

### By start form

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| delayed (13) | 0.8569 | 0.8684 | 0.8715 |
| event (18) | 0.8756 | 0.8907 | 0.8917 |
| prediction_time (19) | 0.8509 | 0.8755 | 0.8809 |

### By end form

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| duration (50) | 0.8613 | 0.8791 | 0.8823 |

### By horizon of the scored window (days)

| stratum (tasks) | small new_bf16 | base new_bf16 | large new_bf16 |
|---|---|---|---|
| 30 (7) | 0.8416 | 0.8608 | 0.8701 |
| 90 (11) | 0.8822 | 0.9020 | 0.9022 |
| 180 (12) | 0.8688 | 0.8894 | 0.8878 |
| 365 (9) | 0.8488 | 0.8661 | 0.8749 |
| 730 (6) | 0.8459 | 0.8602 | 0.8630 |
| 1095 (5) | 0.8661 | 0.8762 | 0.8791 |

---

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
