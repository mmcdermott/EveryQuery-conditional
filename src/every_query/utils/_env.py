"""Load ``.env`` so machine-specific paths reach Hydra ``${oc.env:...}`` interpolations.

This module deliberately does *not* validate env-var presence.  A CLI override —
``EQ_train datamodule.config.tensorized_cohort_dir=/path`` — supplies a path without the
backing env var ever being set, so a blind presence check would reject valid runs.
Validation of the *resolved* config (do the input directories exist? is a wandb entity
set when the wandb logger is actually active?) happens after Hydra composition, in
:func:`every_query.train.train.validate_training_config`.  See #184 for the history.
"""

from dotenv import load_dotenv


def load_env() -> None:
    """Load ``.env`` into ``os.environ`` so ``${oc.env:...}`` interpolations can resolve.

    A thin wrapper over :func:`dotenv.load_dotenv` — it neither requires nor validates
    any specific variable.  Config-aware path/identity validation lives in
    :func:`every_query.train.train.validate_training_config` (see #184).
    """
    load_dotenv()
