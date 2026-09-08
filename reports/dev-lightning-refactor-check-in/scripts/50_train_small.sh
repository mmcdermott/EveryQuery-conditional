#!/usr/bin/env bash
# Retrain the SMALL model (128 x 3, the 6-minute size) with the old or the dev code on the identical
# labels, seed, batch size and one-epoch budget as multitask-canonical-eval/scripts/30_train.sh, so the
# fit path of the refactor (ConditionalMultitaskDataModule + explicit validation_step) can be compared
# with the pre-refactor one.  GPU training is not bit-reproducible (deterministic=false), so the old code
# is trained twice to measure run-to-run noise.  Usage: 50_train_small.sh <old|new> <run_name> [seed]
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
CODE=${1:?old|new}; NAME=${2:?run_name}; SEED_=${3:-${SEED}}
case "$CODE" in
    new) BIN="${BIN_NEW}" ;;
    old) BIN="${BIN_OLD}" ;;
    *) echo "unknown code ${CODE}" >&2; exit 1 ;;
esac
H=128; L=3; A=2; F=512
echo "=== $(date -Is) train small with ${CODE} code -> runs/${NAME} (seed ${SEED_})"
"${BIN}/EQ_train" --config-name=conditional_multitask_ar_config \
    output_dir="${EXP}/runs/${NAME}" \
    seed="${SEED_}" \
    datamodule.config.tensorized_cohort_dir="${COHORT}/processed" \
    datamodule.config.task_labels_dir="${OLD_EXP}/labels" \
    datamodule.batch_size=96 \
    datamodule.num_workers=8 \
    lightning_module.model.ontology_dir=null \
    lightning_module.model.config_overrides.hidden_size="${H}" \
    lightning_module.model.config_overrides.num_hidden_layers="${L}" \
    lightning_module.model.config_overrides.num_attention_heads="${A}" \
    lightning_module.model.config_overrides.num_key_value_heads="${A}" \
    lightning_module.model.config_overrides.intermediate_size="${F}" \
    trainer.val_check_interval=0.25 \
    trainer.limit_val_batches=25 \
    trainer.logger.project="${WANDB_PROJECT}" \
    trainer.logger.entity="${WANDB_ENTITY}" \
    trainer.logger.name="${NAME}"
echo "=== $(date -Is) train ${NAME} done"
