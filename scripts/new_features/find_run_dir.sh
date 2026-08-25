#!/usr/bin/env bash
# Print the newest Hydra run dir for a run name (the timestamped ${output_dir}/<date>/<time>).
# shellcheck source=scripts/new_features/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" > /dev/null
export RUN_NAME="${1:-cq-tiny-allfeat}"
"$EQ_PY" - <<'PY'
import os, pathlib, sys
b = pathlib.Path(os.environ["NF_TRAIN_OUT_DIR"]) / os.environ["RUN_NAME"]
c = [p for p in b.glob("*/*") if (p / "resolved_config.yaml").is_file()]
if not c:
    sys.exit("NO RUN DIR")
print(max(c, key=lambda p: p.stat().st_mtime))
PY
