#!/usr/bin/env python3
"""Generate Ethos-tokenized MEDS shards from the public MIMIC-IV-demo dataset.

Goal: produce a real Ethos preprocessing artifact (timeline shards + the
``StaticDataCollector`` pickle) so we can:

  1. Inspect the actual format of Ethos's static-data side-channel — required to lift
     the parquet-only restriction in
     ``src/every_query/paper_experiments/ethos_compat/reprocess.py::load_static_table``.
  2. Run the slow integration test
     ``tests/test_ethos_compat.py::test_real_ethos_output_recombines_and_mtd_ingests``
     against real Ethos output (set ``ETHOS_DEMO_DIR`` to the result of this script).

The MIMIC-IV-demo dataset is **publicly accessible** (no PhysioNet credentials
required for the demo specifically — see https://physionet.org/content/mimic-iv-demo/).

This script is a **best-effort** orchestration of the upstream tools.  Each external
tool (curl, ``meds_etl_mimic`` or equivalent, ``ethos`` CLI) has its own setup
requirements that may have drifted; the script logs each step and exits with
diagnostic info if a step fails.  Read the printed errors and finish the failing step
manually if needed — every intermediate is materialized on disk so you can resume
from where it broke.

Usage:

    python scripts/setup_ethos_demo_data.py --workdir /tmp/ethos-demo

Output layout (under ``workdir``):

    workdir/
    ├── mimic_iv_demo/                   ← raw MIMIC-IV-demo CSVs from PhysioNet
    ├── meds/                            ← MEDS-converted shards (via meds_etl_mimic)
    ├── ethos_repo/                      ← cloned ipolharvard/ethos-ares
    └── ethos_output/                    ← Ethos-tokenized shards + static pickle
        ├── data/<split>/*.parquet
        ├── metadata/codes.parquet
        └── (whatever Ethos's StaticDataCollector emits — that's what we want to inspect)

Set ``ETHOS_DEMO_DIR=workdir/ethos_output`` after a successful run.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger("setup_ethos_demo_data")


MIMIC_DEMO_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-demo-2.2.zip"
MIMIC_DEMO_ZIP_NAME = "mimic-iv-demo-2.2.zip"
ETHOS_REPO_URL = "https://github.com/ipolharvard/ethos-ares.git"


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a subprocess command, streaming output.

    Raises on non-zero exit.
    """
    logger.info("$ %s%s", " ".join(cmd), f"  (cwd={cwd})" if cwd else "")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"Inspect the logs above; finish this step manually if needed."
        )


def step_download_mimic_demo(workdir: Path) -> Path:
    """Download MIMIC-IV-demo zip and unpack into ``workdir/mimic_iv_demo/``."""
    target = workdir / "mimic_iv_demo"
    if target.is_dir() and any(target.iterdir()):
        logger.info("MIMIC-IV-demo already unpacked at %s — skipping download.", target)
        return target

    workdir.mkdir(parents=True, exist_ok=True)
    zip_path = workdir / MIMIC_DEMO_ZIP_NAME
    if not zip_path.is_file():
        logger.info("Downloading MIMIC-IV-demo from %s …", MIMIC_DEMO_URL)
        urllib.request.urlretrieve(MIMIC_DEMO_URL, zip_path)
        logger.info("Downloaded %.1f MB → %s", zip_path.stat().st_size / 1e6, zip_path)

    logger.info("Unpacking %s → %s", zip_path, target)
    shutil.unpack_archive(str(zip_path), str(target))
    return target


def step_convert_to_meds(mimic_dir: Path, workdir: Path) -> Path:
    """Convert MIMIC-IV-demo to MEDS via ``meds_etl`` (or fall back to manual).

    Several wrapper names exist depending on install version; tries each in order.
    """
    target = workdir / "meds"
    if target.is_dir() and any(target.iterdir()):
        logger.info("MEDS conversion already done at %s — skipping.", target)
        return target

    target.mkdir(parents=True, exist_ok=True)
    mimic_actual = next((p for p in mimic_dir.rglob("hosp/admissions.csv*")), None)
    if mimic_actual is None:
        raise SystemExit(
            f"Could not locate hosp/admissions.csv under {mimic_dir} — the unpacked "
            f"MIMIC-IV-demo layout looks unexpected."
        )
    mimic_root = mimic_actual.parent.parent
    logger.info("MIMIC-IV-demo root resolved to %s", mimic_root)

    candidates = [
        ["meds_etl_mimic", str(mimic_root), str(target)],
        ["meds_etl", "mimic", str(mimic_root), str(target)],
        ["python", "-m", "meds_etl.mimic", str(mimic_root), str(target)],
    ]
    last_err: Exception | None = None
    for cmd in candidates:
        if shutil.which(cmd[0]) is None and cmd[0] != "python":
            continue
        try:
            _run(cmd)
            return target
        except SystemExit as e:
            last_err = e
            logger.warning("Failed: %s — trying next candidate", " ".join(cmd))

    raise SystemExit(
        "Could not run any meds_etl variant.  Install meds_etl "
        "(``uv pip install meds-etl``) and rerun, or convert MIMIC-IV-demo to MEDS "
        f"manually under {target!s}.\nLast error: {last_err}"
    )


def step_clone_ethos(workdir: Path) -> Path:
    """Clone ``ipolharvard/ethos-ares`` into ``workdir/ethos_repo/``."""
    target = workdir / "ethos_repo"
    if target.is_dir() and (target / ".git").is_dir():
        logger.info("Ethos repo already cloned at %s — pulling latest.", target)
        _run(["git", "-C", str(target), "pull", "--ff-only"])
        return target

    workdir.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", ETHOS_REPO_URL, str(target)])
    return target


def step_run_ethos_tokenization(meds_dir: Path, ethos_repo: Path, workdir: Path) -> Path:
    """Invoke Ethos's tokenization on the MEDS shards.

    Best-effort: the exact CLI / module name varies across upstream commits.  Tries
    ``python -m ethos.tokenize.run_tokenization`` first.
    """
    target = workdir / "ethos_output"
    if target.is_dir() and any(target.iterdir()):
        logger.info("Ethos output already exists at %s — skipping.", target)
        return target

    target.mkdir(parents=True, exist_ok=True)

    logger.info("Installing ethos from %s …", ethos_repo)
    _run(["uv", "pip", "install", "-e", str(ethos_repo)])

    cmd = [
        "python",
        "-m",
        "ethos.tokenize.run_tokenization",
        f"input_dir={meds_dir!s}",
        f"output_dir={target!s}",
        "dataset=mimic",
    ]
    try:
        _run(cmd, cwd=ethos_repo)
    except SystemExit as e:
        raise SystemExit(
            f"Ethos tokenization failed.  This is the most fragile step — the upstream "
            f"CLI may have moved.  Look in {ethos_repo}/src/ethos/tokenize/ for the "
            f"current entry point and run it manually with input={meds_dir} and "
            f"output={target}.\nOriginal error: {e}"
        ) from e
    return target


def step_inspect_ethos_output(ethos_output: Path) -> None:
    """Walk the Ethos output dir and report what's there — especially any pickle.

    The pickle is the static-data side-channel; reporting its Python type + structure
    is the whole point of running this script (so we can update ``load_static_table``
    to handle the real format).
    """
    logger.info("=" * 72)
    logger.info("Ethos output inspection: %s", ethos_output)
    logger.info("=" * 72)
    for entry in sorted(ethos_output.rglob("*")):
        if entry.is_file():
            size_kb = entry.stat().st_size / 1024
            logger.info("  %s  (%.1f KB)", entry.relative_to(ethos_output), size_kb)

    pickles = list(ethos_output.rglob("*.pkl")) + list(ethos_output.rglob("*.pickle"))
    if not pickles:
        logger.warning(
            "No pickle files found under %s — Ethos may emit statics in a different "
            "format on this version.  Look at the parquet files instead.",
            ethos_output,
        )
        return

    import pickle as _pickle

    for pkl in pickles:
        logger.info("--- Inspecting pickle: %s ---", pkl)
        try:
            with pkl.open("rb") as f:
                data = _pickle.load(f)
        except Exception as e:
            logger.warning("Could not load pickle %s: %s", pkl, e)
            continue

        logger.info("  type: %s", type(data).__name__)
        if isinstance(data, dict):
            logger.info("  keys (up to 5): %s", list(data.keys())[:5])
            sample_key = next(iter(data), None)
            if sample_key is not None:
                sample_val = data[sample_key]
                logger.info(
                    "  sample value (key=%s): type=%s value=%r",
                    sample_key,
                    type(sample_val).__name__,
                    sample_val if not isinstance(sample_val, list) else sample_val[:3],
                )
        else:
            logger.info("  value (truncated): %r", str(data)[:500])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workdir",
        required=True,
        type=Path,
        help="Working directory.  All intermediates land here.",
    )
    parser.add_argument(
        "--skip-ethos",
        action="store_true",
        help="Skip the Ethos clone + tokenization step (just download + convert MEDS).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Workdir: %s", workdir)

    mimic_dir = step_download_mimic_demo(workdir)
    meds_dir = step_convert_to_meds(mimic_dir, workdir)

    if args.skip_ethos:
        logger.info("--skip-ethos set; stopping after MEDS conversion.")
        logger.info("MEDS output: %s", meds_dir)
        return 0

    ethos_repo = step_clone_ethos(workdir)
    ethos_output = step_run_ethos_tokenization(meds_dir, ethos_repo, workdir)
    step_inspect_ethos_output(ethos_output)

    print()
    print("=" * 72)
    print("Done.  To run the integration test against this output:")
    print(f"  ETHOS_DEMO_DIR={ethos_output} pytest -m slow tests/test_ethos_compat.py")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
