"""Smoke tests for the ``EQ_*`` console script entry points and the ``scripts/`` drivers.

Each endpoint is invoked via its installed console script name (the contract
introduced in PR #61) with ``--help`` and must exit 0.  A successful exit
proves the ``[project.scripts]`` entry resolved, the package config
directory resolved via ``importlib.resources.files()``, and module-level
imports don't blow up in a fresh interpreter.

``test_script_imports`` covers the un-installed research drivers under ``scripts/``, which
``--ignore=scripts`` keeps out of collection entirely.

Child-process coverage is picked up automatically via
``[tool.coverage.run] patch = ["subprocess"]`` in ``pyproject.toml`` — no
per-subprocess env wiring required.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_VENV_BIN = str(Path(sys.executable).parent)
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

_ENTRYPOINTS: list[str] = [
    "EQ_process_data",
    "EQ_train",
    "EQ_generate_training_tasks",
    "EQ_generate_evaluation_tasks",
    "EQ_predict",
    "EQ_evaluate",
    # Conditional query-sequence pipeline.
    "EQ_generate_query_sequences",
    "EQ_generate_evaluation_query_sequences",
    "EQ_predict_sequences",
    "EQ_evaluate_sequences",
    # All-vocabulary multi-bound multitask sampler (#20).
    "EQ_generate_multitask_sequences",
]

# Third-party modules a ``scripts/`` driver needs that this project does not depend on.  The script
# raises an informative ``ModuleNotFoundError`` naming the install command; skipping is honest here
# because a missing optional dep is not API drift, which is what this test exists to catch.
_OPTIONAL_SCRIPT_DEPS: dict[str, str] = {
    "build_report.py": "reportlab",
    "build_report_v2.py": "reportlab",
    "build_report_final.py": "reportlab",
}


@pytest.fixture(scope="module")
def cli_env() -> dict[str, str]:
    """Subprocess env with venv ``PATH`` prepended so console scripts resolve."""
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    return env


@pytest.mark.parametrize("script", _ENTRYPOINTS)
def test_entrypoint_help(script, cli_env):
    """``<script> --help`` exits 0."""
    cmd = [script, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=cli_env, timeout=60)
    assert result.returncode == 0, (
        f"{script} --help failed (rc={result.returncode})\n"
        f"cmd: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("script_path", sorted(_SCRIPTS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_script_imports(script_path: Path):
    """Every ``scripts/*.py`` driver still imports against the current library API.

    ``pyproject.toml`` sets ``--ignore=scripts``, so ``--doctest-modules`` never imports these.
    That exclusion is deliberate — they are one-off research drivers with heavy imports and
    ``argparse`` mains, not a doctest surface — but it is also what let several of them sit broken
    behind stale ``every_query`` imports across a whole sampler refactor, invisible to CI.  This is
    the cheap replacement: import only.  No cohort, no checkpoint, no ``main()``.
    """
    optional = _OPTIONAL_SCRIPT_DEPS.get(script_path.name)
    if optional is not None:
        pytest.importorskip(optional, reason=f"{script_path.name} needs the optional {optional!r} dep")

    # Several drivers import their siblings (``from eval_v2 import ...``), which resolves only with
    # scripts/ on the path — exactly how they run (``python scripts/eval_position_effect.py``).
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"_scripts_smoke_{script_path.stem}", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
