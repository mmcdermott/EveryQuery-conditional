"""Dry-validate the designed spec YAMLs against the real eval-sampler parser.

Runs `read_sequence_specs` + `validate_spec_codes(addressable_codes(...))` -- the same two calls
the CLI makes before it touches the cohort -- so a malformed triple or an unaddressable node
fails here in seconds instead of after the training run.

Prints counts and spec names only, never code strings.
"""

import os
import sys
from pathlib import Path

from every_query.generate_tasks.sample_evaluation_query_sequences import (
    addressable_codes,
    read_sequence_specs,
    validate_spec_codes,
)
from every_query.generate_tasks.sample_tasks import read_query_codes


def main() -> int:
    spec_dir = Path(sys.argv[1])
    tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["len1", "len3"]
    vocab = addressable_codes(
        read_query_codes(os.environ["TENSORIZED_COHORT_DIR"]),
        ontology_dir=os.environ["NF_ONTOLOGY_DIR"],
    )
    print(f"addressable query vocabulary: {len(vocab)} node(s)")

    rc = 0
    for tag in tags:
        specs = read_sequence_specs(spec_dir / f"designed_{tag}.yaml")
        lens = {len(s.queries) for s in specs}
        bounds = [b for s in specs for b in (s.bounds or (None,) * len(s.queries))]
        n_bound = sum(1 for b in bounds if b is not None)
        by_cat: dict[str, int] = {}
        for s in specs:
            by_cat[s.name.split("_")[0]] = by_cat.get(s.name.split("_")[0], 0) + 1
        print(f"\n{tag}: {len(specs)} spec(s), lengths={sorted(lens)}, bounded positions={n_bound}")
        print(f"  by category: {by_cat}")
        try:
            validate_spec_codes(specs, vocab)
            print("  validate_spec_codes: PASS")
        except Exception as e:  # noqa: BLE001
            print(f"  validate_spec_codes: **FAIL** {type(e).__name__}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
