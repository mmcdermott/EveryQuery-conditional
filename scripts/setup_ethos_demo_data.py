#!/usr/bin/env python3
"""Generate Ethos-tokenized MEDS shards from the public MIMIC-IV-demo dataset.

Goal: produce a real Ethos preprocessing artifact (post-tokenization MEDS-shape
shards + the ``StaticDataCollector`` pickle) so we can:

  1. Run ``EQ_reprocess_ethos`` against real Ethos output.
  2. Run ``MTD_preprocess`` on the recombined directory to verify ingestibility.
  3. Power the slow integration test
     ``tests/test_ethos_compat.py::test_real_ethos_output_recombines_and_mtd_ingests``
     (set ``ETHOS_DEMO_DIR`` to the result of this script).

This script has been **executed end-to-end against MIMIC-IV-demo-MEDS** during the
PR-#176 development.  The resulting outputs are the ground truth for:

  * Ethos's static pickle schema (handled by
    :func:`every_query.paper_experiments.ethos_compat.reprocess._load_ethos_static_pickle`).
  * The ICD/CM, ATC, and ICD/PCS prefix patterns used by the recombiner.

## Source data

The MIMIC-IV-demo dataset in MEDS format is publicly accessible at
https://physionet.org/content/mimic-iv-demo-meds/0.0.1/ (no PhysioNet credentials
required for the demo).  This script downloads the project zip via the ``get-zip``
endpoint, so no manual login is needed.

## Steps

1. Download + unpack ``mimic-iv-demo-meds-0.0.1.zip`` from PhysioNet.
2. Create an isolated venv (Ethos pins ``meds_transforms==0.1.1`` /
   ``polars~=1.26`` which conflict with EveryQuery's pins) and install
   ``ipolharvard/ethos-ares`` into it.
3. Run ``ethos_tokenize`` on each split (``train``, ``tuning``, ``held_out``).
4. Build a MEDS-shape directory by collecting the post-tokenization
   ``inject_time_intervals`` parquet from each split — this is the input shape
   ``EQ_reprocess_ethos`` consumes.
5. Inspect ``static_data.pickle`` and report its schema.

## Usage

```bash
python scripts/setup_ethos_demo_data.py --workdir /tmp/ethos-demo
```

After a successful run set:

```bash
export ETHOS_DEMO_DIR=/tmp/ethos-demo/ethos_meds_shape
pytest -m slow tests/test_ethos_compat.py::test_real_ethos_output_recombines_and_mtd_ingests
```

The Ethos pickle (for static reinjection) lands at
``/tmp/ethos-demo/ethos_output/train/static_data.pickle``.
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


# Public MIMIC-IV-demo-MEDS — no PhysioNet credentials required for the demo.
MIMIC_DEMO_MEDS_URL = "https://physionet.org/content/mimic-iv-demo-meds/get-zip/0.0.1/"
MIMIC_DEMO_MEDS_ZIP = "mimic-iv-demo-meds-0.0.1.zip"
ETHOS_REPO_URL = "https://github.com/ipolharvard/ethos-ares.git"

# Ethos numbers its MEDS-transforms stages dynamically.  ``inject_time_intervals`` is
# the last MEDS-shape stage before the safetensors tokenizer; we glob on suffix.
INJECT_TIME_INTERVALS_GLOB = "*inject_time_intervals/0.parquet"


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a subprocess command, streaming output.

    Raises on non-zero exit.
    """
    logger.info("$ %s%s", " ".join(cmd), f"  (cwd={cwd})" if cwd else "")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"Inspect logs above; finish manually if needed."
        )


def step_download_meds(workdir: Path) -> Path:
    """Download MIMIC-IV-demo-MEDS and unpack it into ``workdir/meds/``."""
    target = workdir / "meds"
    if target.is_dir() and any(target.iterdir()):
        logger.info("MIMIC-IV-demo-MEDS already unpacked at %s — skipping download.", target)
        return target

    workdir.mkdir(parents=True, exist_ok=True)
    zip_path = workdir / MIMIC_DEMO_MEDS_ZIP
    if not zip_path.is_file():
        logger.info("Downloading %s → %s", MIMIC_DEMO_MEDS_URL, zip_path)
        # Some PhysioNet endpoints reject the default urllib User-Agent; mimic curl.
        req = urllib.request.Request(MIMIC_DEMO_MEDS_URL, headers={"User-Agent": "curl/7.81"})
        with urllib.request.urlopen(req, timeout=600) as r:
            zip_path.write_bytes(r.read())
        logger.info("Downloaded %.1f MB", zip_path.stat().st_size / 1e6)

    logger.info("Unpacking %s …", zip_path)
    raw = workdir / "_unpacked"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir()
    shutil.unpack_archive(str(zip_path), str(raw))

    # The zip contains one inner directory whose name encodes dataset version.  Move
    # it to the canonical ``meds/`` location.
    inner = next(p for p in raw.iterdir() if p.is_dir())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(inner), str(target))
    shutil.rmtree(raw)
    logger.info("MEDS data resolved to %s", target)
    return target


def step_install_ethos(workdir: Path) -> Path:
    """Clone ``ipolharvard/ethos-ares`` and install into a dedicated venv.

    Returns the venv's ``ethos_tokenize`` executable path.
    """
    repo = workdir / "ethos_repo"
    if not (repo / ".git").is_dir():
        _run(["git", "clone", "--depth", "1", ETHOS_REPO_URL, str(repo)])
    else:
        logger.info("Ethos repo already cloned at %s", repo)

    venv_dir = workdir / "venv"
    if not (venv_dir / "bin" / "python").exists():
        _run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    ethos_bin = venv_dir / "bin" / "ethos_tokenize"
    if not ethos_bin.exists():
        env = {**__import__("os").environ, "VIRTUAL_ENV": str(venv_dir)}
        _run(["uv", "pip", "install", "-e", str(repo)], env=env)
    return ethos_bin


def step_tokenize_each_split(meds_dir: Path, ethos_bin: Path, workdir: Path) -> Path:
    """Run ``ethos_tokenize`` per split (train/tuning/held_out).

    The first split (train) builds a vocabulary; subsequent splits reuse it via
    ``vocab=`` so cross-split codes line up.
    """
    out = workdir / "ethos_output"
    splits = ["train", "tuning", "held_out"]
    for split in splits:
        if (out / split / "static_data.pickle").exists():
            logger.info("Ethos output for %s already present — skipping.", split)
            continue
        cmd = [
            str(ethos_bin),
            f"input_dir={meds_dir / 'data' / split}",
            f"output_dir={out}",
            f"out_fn={split}",
        ]
        if split != "train":
            cmd.append(f"vocab={out / 'train'}")
        _run(cmd, cwd=workdir)
    return out


def step_build_meds_shape_dir(meds_dir: Path, ethos_output: Path, workdir: Path) -> Path:
    """Materialise an Ethos-MEDS-shape directory ``EQ_reprocess_ethos`` can consume.

    Layout:
        ethos_meds_shape/
        ├── data/
        │   ├── train/0.parquet     ← Ethos's `inject_time_intervals` stage output
        │   ├── tuning/0.parquet
        │   └── held_out/0.parquet
        └── metadata/               ← copied from the source MEDS dir
    """
    target = workdir / "ethos_meds_shape"
    if target.exists():
        shutil.rmtree(target)
    (target / "metadata").mkdir(parents=True)
    for split in ("train", "tuning", "held_out"):
        candidates = list((ethos_output / split).glob(INJECT_TIME_INTERVALS_GLOB))
        if not candidates:
            raise SystemExit(
                f"Could not find inject_time_intervals stage parquet under "
                f"{ethos_output / split}.  Available stages: "
                f"{[p.name for p in (ethos_output / split).iterdir() if p.is_dir()]}"
            )
        src = candidates[0]
        dst = target / "data" / split / "0.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("Copied %s → %s", src.relative_to(workdir), dst.relative_to(workdir))

    # Bring through metadata from the original MEDS dataset (codes / splits / dataset.json).
    for fname in ("dataset.json", "subject_splits.parquet", "codes.parquet"):
        src = meds_dir / "metadata" / fname
        if src.is_file():
            shutil.copy2(src, target / "metadata" / fname)
    logger.info("Ethos-MEDS-shape directory ready at %s", target)
    return target


def step_inspect_static_pickle(ethos_output: Path) -> None:
    """Walk pickle files and report their layout — confirmed schema is documented in
    ``every_query.paper_experiments.ethos_compat.reprocess._load_ethos_static_pickle``."""
    import pickle as _pickle

    pickles = list(ethos_output.rglob("*.pickle")) + list(ethos_output.rglob("*.pkl"))
    if not pickles:
        logger.warning("No pickle files under %s", ethos_output)
        return
    for p in pickles:
        logger.info("--- pickle: %s ---", p)
        try:
            with p.open("rb") as f:
                data = _pickle.load(f)
        except Exception as e:
            logger.warning("Could not load %s: %s", p, e)
            continue
        logger.info(
            "  type=%s; subjects=%d", type(data).__name__, len(data) if isinstance(data, dict) else -1
        )
        if isinstance(data, dict) and data:
            sample_sid = next(iter(data))
            sample_val = data[sample_sid]
            logger.info("  sample subject %s → %r", sample_sid, sample_val)


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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Workdir: %s", workdir)

    meds_dir = step_download_meds(workdir)
    ethos_bin = step_install_ethos(workdir)
    ethos_output = step_tokenize_each_split(meds_dir, ethos_bin, workdir)
    target = step_build_meds_shape_dir(meds_dir, ethos_output, workdir)
    step_inspect_static_pickle(ethos_output)

    static_pickle = ethos_output / "train" / "static_data.pickle"
    print()
    print("=" * 72)
    print("Done.  To run the integration test against this output:")
    print(f"  ETHOS_DEMO_DIR={target}")
    print(f"  ETHOS_DEMO_STATIC={static_pickle}")
    print("  pytest -m slow tests/test_ethos_compat.py::test_real_ethos_output_recombines_and_mtd_ingests")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
