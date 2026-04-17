"""Subprocess smoke tests for the ``EQ_*`` console script entry points.

Each endpoint is invoked via its installed console script name (the contract
introduced in PR #61) with ``--help``.  A successful exit proves four things
at once:

1. The ``[project.scripts]`` entry in ``pyproject.toml`` resolved and the
   installed shim is runnable.
2. Hydra accepted the ``config_path=`` argument (this PR's switch from
   fragile relative paths to ``importlib.resources.files()``).
3. Hydra could locate and load the named config file under that path — if
   the package-data glob or the resource resolution regressed, ``--help``
   would fail with ``MissingConfigException``.
4. Module-level imports don't blow up in a fresh interpreter.

Three endpoints (``EQ_train``, ``EQ_evaluate``, ``EQ_gen_eval_tasks``)
compose a ``{train,eval}_codes`` default group whose canonical file is
generated out-of-band (see ``src/every_query/sample_codes/``) and is not
checked in.  For those we point Hydra at a throwaway ``--config-dir`` that
supplies an empty-codes smoke variant, which preserves the rest of the
compose path.

Subprocess coverage follows the pytest-cov recipe:
https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_VENV_BIN = str(Path(sys.executable).parent)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# (console_script, extra_args) — extras inject a smoke code-group for configs
# whose defaults pull an out-of-tree YAML.
_ENTRYPOINTS: list[tuple[str, list[str]]] = [
    ("EQ_train", ["train_codes=smoke"]),
    ("EQ_evaluate", ["eval_codes=smoke"]),
    ("EQ_generate_tasks", []),
    ("EQ_generate_tasks_exhaustive", []),
    ("EQ_gen_eval_index", []),
    ("EQ_gen_eval_tasks", ["eval_codes=smoke"]),
    ("EQ_select_model", []),
]


@pytest.fixture(scope="module")
def smoke_config_dir(tmp_path_factory) -> Path:
    """Temp Hydra search dir supplying empty ``train_codes`` / ``eval_codes``."""
    d = tmp_path_factory.mktemp("eq_smoke_cfg")
    (d / "train_codes").mkdir()
    (d / "train_codes" / "smoke.yaml").write_text("codes: []\n")
    (d / "eval_codes").mkdir()
    (d / "eval_codes" / "smoke.yaml").write_text("id: []\nood: []\n")
    return d


@pytest.fixture(scope="module")
def cli_env(tmp_path_factory) -> dict[str, str]:
    """Subprocess env with venv ``PATH`` and pytest-cov subprocess hook.

    Writes an ephemeral ``sitecustomize.py`` that calls
    ``coverage.process_startup()`` and points the child at it via
    ``PYTHONPATH`` + ``COVERAGE_PROCESS_START``.  This is a no-op when
    coverage is not installed or not active; when it is,
    ``[tool.coverage.run] parallel = true`` in ``pyproject.toml`` makes
    the child-process data files merge cleanly with the parent's.
    """
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    sitecustom = tmp_path_factory.mktemp("eq_cov_sitecustomize")
    (sitecustom / "sitecustomize.py").write_text(
        "try:\n    import coverage\n    coverage.process_startup()\nexcept Exception:\n    pass\n",
    )
    env["PYTHONPATH"] = str(sitecustom) + os.pathsep + env.get("PYTHONPATH", "")
    env["COVERAGE_PROCESS_START"] = str(_REPO_ROOT / "pyproject.toml")

    return env


@pytest.mark.parametrize(("script", "extra_args"), _ENTRYPOINTS, ids=[e[0] for e in _ENTRYPOINTS])
def test_entrypoint_help(script, extra_args, cli_env, smoke_config_dir):
    """``<script> --help`` exits 0 and produces Hydra's help signature."""
    cmd = [script, f"--config-dir={smoke_config_dir}", *extra_args, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=cli_env, timeout=60)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{script} --help failed (rc={result.returncode})\n"
        f"cmd: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Powered by Hydra" in combined, (
        f"{script} --help lacks Hydra help signature; output was:\n{combined}"
    )
