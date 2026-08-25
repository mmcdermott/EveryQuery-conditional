"""Report which EQ path env vars are set and whether they resolve to real dirs.

Prints ONLY: var name, exists?, whether it looks like a MEDS event root, and shard counts.
It deliberately does NOT print the path values (they may point under /data) nor any rows.
"""

import os
import sys
from pathlib import Path

VARS = [
    "DATA_DIR",
    "COHORT_ROOT",
    "TOKENIZED_EVENTS_DIR",
    "TENSORIZED_COHORT_DIR",
    "TRAINING_TASKS_DIR",
    "EVAL_TASKS_DIR",
    "TRAINING_OUTPUT_DIR",
    "ONTOLOGY_DIR",
    "EQ_EXP_ROOT",
    "EQ_LOG_DIR",
]


def classify(p: Path) -> str:
    """Describe a dir by SHAPE only — never by content."""
    if not p.exists():
        return "MISSING"
    bits = []
    for split in ("train", "tuning", "held_out"):
        d = p / "data" / split
        if d.is_dir():
            bits.append(f"data/{split}={len(list(d.glob('*.parquet')))}pq")
    if (p / "metadata" / "codes.parquet").exists():
        bits.append("has metadata/codes.parquet")
    if (p / "tokenization" / "event_seqs").is_dir():
        bits.append("has tokenization/event_seqs")
    for split in ("train", "tuning", "held_out"):
        d = p / split
        if d.is_dir():
            bits.append(f"{split}/={len(list(d.glob('*.parquet')))}pq")
    return "EXISTS " + (", ".join(bits) if bits else "(no recognised subdirs)")


def main() -> int:
    for v in VARS:
        val = os.environ.get(v)
        if val is None:
            print(f"{v:24s} UNSET")
            continue
        p = Path(val)
        # Report whether it is under /data without echoing the path.
        under_data = str(p).startswith("/data")
        print(f"{v:24s} set(under_/data={under_data})  {classify(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
