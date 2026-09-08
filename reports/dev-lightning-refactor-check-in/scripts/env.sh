# shellcheck shell=bash
# Shared paths for the dev-branch (#30 Lightning refactor) evaluation check-in.  Source from every script.
# Nothing is trained here: the three checkpoints and the four evaluation grids of the 2026-09-05
# multitask-canonical-eval experiment are re-scored, and the new predictions are compared with the
# stored ones row for row and task for task.
export EXP=/home/gkondas/eq-new-exps/dev-lightning-refactor-check-in
export OLD_EXP=/home/gkondas/eq-new-exps/multitask-canonical-eval   # checkpoints, grids, stored preds/metrics
export REPO=/home/gkondas/EveryQuery-conditional
# origin/dev 0b90790 (PR #31, the #30 refactor) on branch exp/dev-lightning-refactor-check-in
export BIN_NEW=${REPO}/.claude/worktrees/dev-lightning-refactor-check-in/.venv/bin
# de0f5ee = exp/multitask-canonical-eval, the code the stored 2026-09-05 predictions were made with
export BIN_OLD=${REPO}/.claude/worktrees/old-eval-baseline/.venv/bin
export COHORT=/home/gkondas/eq-new-exps/cohort                       # preprocessed MIMIC-IV MEDS (intermediate + processed)
export SEED=140799
export WANDB_MODE=offline
export WANDB_ENTITY=${WANDB_ENTITY:-gregkondas9-columbia-university}
export WANDB_PROJECT=${WANDB_PROJECT:-eqc-dev-refactor-check}
