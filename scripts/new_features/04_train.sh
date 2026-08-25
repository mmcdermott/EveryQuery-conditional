#!/usr/bin/env bash
#
# Step 4 — train the TINY conditional query-sequence model with ALL THREE new features ON.
#
#   Feature 1  rope time encoding
#       datamodule.dataset_kwargs.strip_delta_tokens=true   (drops TIMELINE//DELTA* tokens,
#                                                            emits per-token elapsed-hour time_pos_ids)
#       lightning_module.model.use_rope_time=true           (consumes them as rotary positions)
#       These two are checked against each other at the first forward -- a mismatch is a hard
#       ValueError in either direction, so this pairing cannot silently go wrong.
#
#   Feature 2  event-bounded durations
#       No model/data key: the dataset detects the `bound_events` column in the labels parquet
#       and `bound_marker` is always allocated.  The feature is turned on upstream, by
#       `eventbound_fraction=0.5` in step 02.
#
#   Feature 3  DAG-aware embeddings and queries
#       datamodule.dataset_kwargs.ontology_dir=$NF_ONTOLOGY_DIR   (ancestor names -> token ids)
#       lightning_module.model.ontology_dir=$NF_ONTOLOGY_DIR      (embeds as (A @ W)[ids])
#       NOTE there is NO cross-check between these two keys.  If they drift, indices address the
#       wrong embedding rows and the run succeeds with meaningless ancestor semantics -- so they
#       are set from ONE shell variable here, deliberately.
#
# W&B override mechanics: Hydra `=` on a dict node MERGES.  Swapping only `_target_` leaves the
# CSVLogger's `flush_logs_every_n_steps`, which WandbLogger forwards into `wandb.init()` and
# which raises TypeError at the first `.experiment` access -- hence the explicit `~` deletion.
#
# Usage:  bash scripts/new_features/04_train.sh [max_time HH:MM:SS] [run_name]

# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

MAX_TIME="${1:-00:00:16:00}"   # DD:HH:MM:SS -- hard wall-clock cap on fit()
export RUN_NAME="${2:-cq-tiny-allfeat}"

LOG="${NF_LOG_DIR}/04_train.log"
OUT_BASE="${NF_TRAIN_OUT_DIR}/${RUN_NAME}"

# SMOKE=1: a few steps, same overrides (fast_dev_run would disable the logger and prove nothing).
SMOKE_ARGS=()
if [[ -n "${SMOKE:-}" ]]; then
    RUN_NAME="${RUN_NAME}-smoke"
    OUT_BASE="${NF_TRAIN_OUT_DIR}/${RUN_NAME}"
    LOG="${NF_LOG_DIR}/04_train_smoke.log"
    SMOKE_ARGS=(+trainer.limit_train_batches=6 trainer.val_check_interval=5 trainer.limit_val_batches=2)
fi

echo "run_name : $RUN_NAME"
echo "max_time : $MAX_TIME"
echo "out_base : $OUT_BASE"
echo "wandb    : ${WANDB_ENTITY}/${WANDB_PROJECT}  (mode=${WANDB_MODE})"
echo "log      : $LOG"

START=$SECONDS
STATUS=0

"$EQ_PY" -m every_query.train.train \
    --config-name=conditional_config \
    output_dir="$OUT_BASE" \
    datamodule.config.tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    datamodule.config.task_labels_dir="$NF_TRAIN_TASKS_DIR" \
    datamodule.dataset_kwargs.strip_delta_tokens=true \
    datamodule.dataset_kwargs.ontology_dir="$NF_ONTOLOGY_DIR" \
    lightning_module.model.use_rope_time=true \
    lightning_module.model.ontology_dir="$NF_ONTOLOGY_DIR" \
    lightning_module.model.num_hidden_layers=4 \
    lightning_module.model.decoder_layers=2 \
    lightning_module.model.decoder_heads=4 \
    lightning_module.model.config_overrides.hidden_size=256 \
    lightning_module.model.config_overrides.num_attention_heads=4 \
    lightning_module.model.config_overrides.intermediate_size=1024 \
    datamodule.batch_size=64 \
    datamodule.num_workers=8 \
    trainer.limit_val_batches=20 \
    trainer.val_check_interval=0.05 \
    trainer.log_every_n_steps=10 \
    +trainer.num_sanity_val_steps=0 \
    +trainer.max_time="$MAX_TIME" \
    trainer.logger._target_=lightning.pytorch.loggers.WandbLogger \
    '~trainer.logger.flush_logs_every_n_steps' \
    trainer.logger.name="$RUN_NAME" \
    +trainer.logger.project="$WANDB_PROJECT" \
    +trainer.logger.entity="$WANDB_ENTITY" \
    +trainer.logger.offline=false \
    +trainer.logger.log_model=false \
    "${SMOKE_ARGS[@]}" \
    > "$LOG" 2>&1 || STATUS=$?

echo "elapsed: $((SECONDS - START))s   exit=$STATUS"

echo
echo "--- feature evidence ---"
grep -E 'RoPE time: stripping|no TIMELINE//DELTA|sizing the encoder to V_ext|query vocabulary extended|Train dataset contains' "$LOG" || true
echo
echo "--- errors ---"
grep -E 'Error|Traceback|CRITICAL|raise ' "$LOG" | head -20 || true
echo
echo "--- run dir ---"
"$EQ_PY" - <<'PY'
import os, pathlib
b = pathlib.Path(os.environ["NF_TRAIN_OUT_DIR"]) / os.environ.get("RUN_NAME", "cq-tiny-allfeat")
cands = [p for p in b.glob("*/*") if (p / "resolved_config.yaml").is_file()]
print(max(cands, key=lambda p: p.stat().st_mtime) if cands else "NO RUN DIR")
PY
exit $STATUS
