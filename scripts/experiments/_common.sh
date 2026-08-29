# shellcheck shell=bash
#
# Shared header for the experiment scripts.  Source it, do not execute it.
#
# Two things every script here needs, both of which have silently produced wrong results before:
#
# 1. **The interpreter.**  The venv lives in the main checkout and is shared by every worktree;
#    there is no `.venv` inside a worktree.
#
# 2. **The import path.**  The editable install is a `.pth` file naming ONE absolute path — the
#    main checkout's `src`.  A worktree shares that venv, so a plain `python script.py` launched
#    from a worktree imports `every_query` from the *main checkout*, which is on a different
#    branch.  `pyproject.toml`'s `pythonpath = ["src"]` fixes this for pytest and ONLY for pytest.
#    A measurement script gets no such protection: this is how a blast-radius measurement in this
#    very session compared the pre-fix code against itself and reported "0 rows changed".
#    So put this checkout's `src` at the FRONT of PYTHONPATH, always.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source ./env.sh

EQ_VENV="${EQ_VENV:-/home/gkondas/EveryQuery-conditional/.venv}"
EQ_PY="${EQ_VENV}/bin/python"
if [[ ! -x "$EQ_PY" ]]; then
    echo "No interpreter at $EQ_PY. Set EQ_VENV to the shared venv." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$EQ_LOG_DIR"

# Fail loudly rather than silently measuring the wrong branch.
ACTUAL_SRC="$("$EQ_PY" -c 'import every_query, pathlib; print(pathlib.Path(every_query.__file__).parent.parent)')"
if [[ "$ACTUAL_SRC" != "${REPO_ROOT}/src" ]]; then
    echo "every_query resolves to ${ACTUAL_SRC}, not ${REPO_ROOT}/src — refusing to run." >&2
    exit 1
fi

echo "checkout : $REPO_ROOT  ($(git rev-parse --short HEAD), $(git branch --show-current))"
echo "python   : $EQ_PY"
