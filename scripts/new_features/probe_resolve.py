"""Print the ICU concept resolution table (concept -> node), and nothing else."""

import os
import sys

from clinical_concepts import CONCEPTS, resolve_concepts


def main() -> int:
    try:
        r = resolve_concepts(os.environ["NF_ONTOLOGY_DIR"], os.environ["TENSORIZED_COHORT_DIR"])
    except LookupError as e:
        print(f"UNRESOLVED: {e}")
        # Re-run per concept so the table still shows what DID resolve.
        r = {}
        for c in CONCEPTS:
            try:
                r.update(resolve_concepts(os.environ["NF_ONTOLOGY_DIR"],
                                          os.environ["TENSORIZED_COHORT_DIR"], [c]))
            except LookupError:
                pass
    print(f"\n{'concept':<20}{'anc':>5}{'n_desc':>8}{'n_occ':>13}  node")
    for c in sorted(r):
        v = r[c]
        print(f"{c:<20}{('yes' if v.is_ancestor else 'no'):>5}{v.n_desc:>8}{v.n_occ:>13}  {v.node}")
    print(f"\nresolved {len(r)}/{len(CONCEPTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
