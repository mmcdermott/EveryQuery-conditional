# Training-side check: retrained small models (canonical grid, tuning split, dev CLI bf16-mixed)

| run | params | tuning/loss trajectory (4 validations) | final train/loss_epoch | macro AUROC | median AUROC | macro AUPRC |
|---|---|---|---|---|---|---|
| small_old_a | 2600149 | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418 | 0.8425 | 0.8485 | 0.1552 |
| small_old_b | 2600149 | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418 | 0.8424 | 0.8485 | 0.1552 |
| small_new_a | 2600149 | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418 | 0.8425 | 0.8485 | 0.1552 |
| small_stored | 2600149 | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418 | 0.8425 | 0.8486 | 0.1551 |

| pair | why | d macro AUROC [95% CI over tasks] | max abs dAUROC (task) | mean abs dAUROC (task) | a higher / b higher |
|---|---|---|---|---|---|
| small_old_a - small_old_b | old code vs old code, same seed (run-to-run noise floor) | +0.0001 [-0.0000, +0.0002] | 0.0011 | 0.0001 | 20/20 of 40 |
| small_new_a - small_old_a | dev code vs old code, same seed | +0.0000 [-0.0001, +0.0002] | 0.0019 | 0.0002 | 24/14 of 40 |
| small_new_a - small_old_b | dev code vs old code (second run), same seed | +0.0001 [-0.0000, +0.0003] | 0.0029 | 0.0002 | 24/16 of 40 |
| small_new_a - small_stored | dev code vs the stored 2026-09-05 small run | +0.0001 [-0.0001, +0.0003] | 0.0036 | 0.0002 | 25/15 of 40 |
| small_old_a - small_stored | old code today vs the stored 2026-09-05 small run | +0.0000 [-0.0001, +0.0002] | 0.0017 | 0.0002 | 19/21 of 40 |
