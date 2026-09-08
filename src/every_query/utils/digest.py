"""One digest for every "is this artifact the one I labeled under" comparison.

Torch-free on purpose: the samplers' worker processes fingerprint their inputs and must not pay
for importing :mod:`every_query.data.ontology` (which imports torch) to do so, while the model
side digests the same closure table and has to get the same string.
"""

import polars as pl


def frame_digest(df: pl.DataFrame) -> str:
    """Serialization-independent digest of a frame's logical rows: ``"{height}:{hash16}"``.

    Polars' vectorized ``hash_rows`` summed over rows, combined with the row count.  Summing is
    order-independent and counts duplicates, so the digest depends only on the *multiset* of rows
    - exactly what decides a label - and a rewritten-but-identical parquet digests the same.  Not
    collision-proof; a collision only ever means a stale artifact is reused, never a wrong label.

    The same construction as ``sample_tasks._index_fingerprint`` for Stage 4's index partitions.

    Examples:
        >>> a = pl.DataFrame({"x": [1, 2], "y": ["p", "q"]})
        >>> frame_digest(a) == frame_digest(a.reverse())
        True
        >>> frame_digest(a) != frame_digest(a.head(1))
        True
        >>> frame_digest(a.clear())
        '0:0000000000000000'
    """
    total = int(df.hash_rows(seed=0).sum()) if df.height else 0
    return f"{df.height}:{total & 0xFFFFFFFFFFFFFFFF:016x}"
