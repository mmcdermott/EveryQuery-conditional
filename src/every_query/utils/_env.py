"""Load .env and validate that required environment variables are set.

Imported early by both train.py and tasks.py so that a missing .env on a fresh machine surfaces a single clear
error rather than a mid-run KeyError or Hydra InterpolationResolutionError.
"""

import os
import sys

from dotenv import load_dotenv

REQUIRED_ENV_VARS = (
    "PROJECT_DIR",
    "OUTPUT_DIR",
    "TASK_DIR",
    "FINAL_DATA_DIR",
    "WANDB_ENTITY",
)
# `PROCESSED` and `INTERMEDIATE` are used by the preprocessing pipeline (the dirs it
# writes to) and by the generate_tasks stage as dotenv fallbacks inside
# `sample_tasks.py::_resolve_path` / `sample_evaluation_tasks.py::_resolve_path`.
# They're not gated by this `REQUIRED_ENV_VARS` check because:
#   1. `_resolve_path` already tolerates a missing env var when the caller supplies the
#      path directly (the normal CLI / test path).
#   2. No Hydra config interpolates `${oc.env:PROCESSED}` or `${oc.env:INTERMEDIATE}`,
#      so demo fixtures / `--help` invocations don't need throwaway placeholders.
# If you're running a full fresh-machine setup that includes preprocessing, set both
# in `.env` (see `.env.example`).  See #117 for the history.


def ensure_env() -> None:
    load_dotenv()
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        joined = ", ".join(missing)
        print(
            f"ERROR: required environment variables not set: {joined}\n"
            "Copy .env.example to .env and fill in machine-specific paths, "
            "or export these variables before running.",
            file=sys.stderr,
        )
        sys.exit(1)
