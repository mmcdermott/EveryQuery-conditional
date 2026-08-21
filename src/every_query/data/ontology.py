"""Code ontology: a DAG over the cohort vocabulary, and the embedding mix it induces.

MEDS codes are already hierarchical in their names — ``LAB//220645//mEq/L//value_[135,136)``
sits under ``LAB//220645``, which sits under ``LAB`` — and a cohort's ``codes.parquet`` may also
carry an explicit ``parent_codes`` column.  This module turns that structure into two things:

1. **An extended vocabulary.**  Every ``//``-prefix that is not itself a code becomes an
   *ancestor node* with its own index, appended above the highest leaf index so leaf indices are
   preserved exactly.  That makes an ancestor directly *addressable as a query*: you can ask
   about a whole drug class rather than one specific product.
2. **A mix matrix** ``A`` (row-normalised, sparse, ``V_ext x V_ext``).  A node's embedding
   becomes the weighted average of its own raw row and its ancestors', with weight
   ``decay ** distance``.  Substituted for ModernBERT's ``tok_embeddings``, this ties a rare
   leaf's representation to its better-estimated parents — the actual hypothesis under test.

Three artifacts are written by ``EQ_build_ontology`` into one directory:

- ``nodes.parquet``   ``(node, vocab_index, is_leaf)`` — the extended vocabulary; ``V_ext``.
- ``mix.parquet``     ``(node_index, component_index, weight)`` — unnormalised COO entries.
- ``closure.parquet`` ``(code, node)`` — every leaf paired with itself and each ancestor NAME,
  used to explode an event stream so ancestor queries can be labelled by ordinary occurrence.

The upstream experiment's own verdict is worth stating plainly, because it bears on whether to
turn this on: the **embedding** effect on leaf tasks did not replicate (a seed-2 run reversed it,
and the unbundling suite scored it null), while the **DAG structure** was worth about +0.039
AUROC on ancestor queries.  The value found so far is in being able to *ask* about an ancestor,
not in the mixing improving ordinary leaf queries.
"""

import logging
from pathlib import Path

import polars as pl
import torch

from every_query.data import query_vocab

logger = logging.getLogger(__name__)

SEP = "//"

NODES_FILE = "nodes.parquet"
MIX_FILE = "mix.parquet"
CLOSURE_FILE = "closure.parquet"


def string_ancestors(code: str) -> list[str]:
    """All proper ``//``-prefix ancestors of a code, nearest first.

    A single ``/`` is not a separator, so ``ICD10CM/A04.72`` has no ancestors — the separator is
    the two-character ``//`` that MEDS uses between hierarchy levels.

    Examples:
        >>> string_ancestors("LAB//220645//mEq/L//value_[135.0,136.0)")
        ['LAB//220645//mEq/L', 'LAB//220645', 'LAB']
        >>> string_ancestors("BMI")
        []
        >>> string_ancestors("ICD10CM/A04.72")
        []
    """
    parts = code.split(SEP)
    return [SEP.join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def build_ontology(codes_df: pl.DataFrame, decay: float = 0.5) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build ``(nodes_df, mix_df)`` from a ``codes.parquet``-shaped frame.

    Args:
        codes_df: Columns ``code``, ``code/vocab_index``, optionally ``parent_codes``.
        decay: Per-level weight decay.  An ancestor ``d`` levels up contributes ``decay ** d``
            before row normalisation, so ``0.0`` disables mixing entirely and ``1.0`` weights
            every ancestor as much as the node itself.

    Returns:
        ``nodes_df`` ``(node, vocab_index, is_leaf)`` — leaves keep their original indices,
        ancestors get fresh ones from ``max_leaf_index + 1`` upward, sorted by name — and
        ``mix_df`` ``(node_index, component_index, weight)``, unnormalised, with a
        weight-1 self-loop for every node.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": ["A//B//C", "A//B//D", "E"],
        ...     "code/vocab_index": [1, 2, 3],
        ...     "parent_codes": [["G/x"], None, None],
        ... })
        >>> nodes, mix = build_ontology(df)
        >>> sorted(nodes.filter(~pl.col("is_leaf"))["node"].to_list())
        ['A', 'A//B', 'G/x']

        Leaf indices are untouched, so an ontology can be dropped onto an existing cohort:

        >>> nodes.filter(pl.col("is_leaf")).sort("vocab_index")["vocab_index"].to_list()
        [1, 2, 3]

        ``A//B//C`` mixes itself (1.0), ``A//B`` (0.5), ``A`` (0.25) and its declared parent
        ``G/x`` (0.5, a grouper edge counts as distance 1):

        >>> m = {r["component_index"]: r["weight"]
        ...      for r in mix.filter(pl.col("node_index") == 1).iter_rows(named=True)}
        >>> idx = dict(zip(nodes["node"], nodes["vocab_index"]))
        >>> m[idx["A//B//C"]], m[idx["A//B"]], m[idx["A"]], m[idx["G/x"]]
        (1.0, 0.5, 0.25, 0.5)

        Every node reachable in the index has a mix row, so no node can end up with an
        all-zero embedding:

        >>> set(nodes["vocab_index"]) == set(mix["node_index"].unique())
        True
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}")

    has_parents = "parent_codes" in codes_df.columns
    leaves = codes_df.select(pl.col("code"), pl.col("code/vocab_index").cast(pl.Int64).alias("vocab_index"))
    leaf_names = set(leaves["code"].to_list())

    node_ancestors: dict[str, dict[str, int]] = {}
    ancestor_names: set[str] = set()

    for row in codes_df.iter_rows(named=True):
        code = row["code"]
        amap: dict[str, int] = {}
        for dist, anc in enumerate(string_ancestors(code), start=1):
            if anc != code:
                amap[anc] = min(amap.get(anc, 10**9), dist)
        if has_parents and row.get("parent_codes"):
            for pc in row["parent_codes"]:
                if not pc or pc == code:
                    continue
                # A declared grouper edge is one level up by definition, and the grouper's own
                # name prefixes hang above it.
                amap[pc] = min(amap.get(pc, 10**9), 1)
                for d2, anc2 in enumerate(string_ancestors(pc), start=2):
                    amap[anc2] = min(amap.get(anc2, 10**9), d2)
        node_ancestors[code] = amap
        ancestor_names.update(a for a in amap if a not in leaf_names)

    # Close the ancestor set to a fixed point.  `parent_codes` can introduce a grouper name whose
    # own prefixes are new, and those prefixes can have prefixes; the upstream implementation ran
    # exactly one extra pass and then re-widened the name set afterwards, which could hand a name
    # an index but no mix row — an all-zero, permanently dead embedding.  Iterating instead makes
    # "every indexed node has a mix row" true by construction rather than by luck.
    pending = sorted(ancestor_names)
    while pending:
        nxt: list[str] = []
        for anc in pending:
            if anc in node_ancestors:
                continue
            amap = {}
            for dist, higher in enumerate(string_ancestors(anc), start=1):
                if higher != anc:
                    amap[higher] = dist
                    if higher not in leaf_names and higher not in ancestor_names:
                        nxt.append(higher)
            node_ancestors[anc] = amap
        ancestor_names.update(nxt)
        pending = sorted(set(nxt))

    # An ancestor name carrying one of the query grammar's separators could not be round-tripped
    # through a query string, so it must never become addressable.  Neither upstream fork could
    # hit this: no fork generated ancestor names at all.
    reserved = {a for a in ancestor_names if query_vocab.has_reserved_chars(a)}
    if reserved:
        logger.warning(
            "Dropping %d ancestor node(s) whose names contain a character the query grammar "
            "reserves (%s): %s",
            len(reserved),
            "".join(sorted(query_vocab.RESERVED_CHARS)),
            sorted(reserved)[:5],
        )
        ancestor_names -= reserved
        for name in reserved:
            node_ancestors.pop(name, None)
        for amap in node_ancestors.values():
            for name in reserved:
                amap.pop(name, None)

    max_leaf = int(leaves["vocab_index"].max())
    anc_sorted = sorted(ancestor_names)
    anc_index = {a: max_leaf + 1 + i for i, a in enumerate(anc_sorted)}
    name_to_index = (
        dict(zip(leaves["code"].to_list(), leaves["vocab_index"].to_list(), strict=True)) | anc_index
    )

    assert set(ancestor_names) <= node_ancestors.keys(), (
        "every indexed ancestor must have a mix row; otherwise it embeds to the zero vector"
    )

    nodes_df = pl.concat(
        [
            leaves.with_columns(pl.lit(True).alias("is_leaf")).rename({"code": "node"}),
            pl.DataFrame(
                {
                    "node": anc_sorted,
                    "vocab_index": [anc_index[a] for a in anc_sorted],
                    "is_leaf": [False] * len(anc_sorted),
                }
            ),
        ],
        how="vertical_relaxed",
    )

    mix_rows: dict[str, list] = {"node_index": [], "component_index": [], "weight": []}
    for node, amap in node_ancestors.items():
        ni = name_to_index[node]
        mix_rows["node_index"].append(ni)
        mix_rows["component_index"].append(ni)
        mix_rows["weight"].append(1.0)
        for anc, dist in amap.items():
            mix_rows["node_index"].append(ni)
            mix_rows["component_index"].append(name_to_index[anc])
            mix_rows["weight"].append(decay**dist)

    return nodes_df, pl.DataFrame(mix_rows)


def build_closure(nodes_df: pl.DataFrame, mix_df: pl.DataFrame) -> pl.DataFrame:
    """``(code, node)`` rows: every leaf paired with itself and each of its ancestor names.

    This is what lets an ancestor query be answered by the ordinary occurrence labeler — explode
    the event stream through it and "did any descendant of X occur" becomes "did X occur".

    Examples:
        >>> nodes = pl.DataFrame({
        ...     "node": ["A//B", "A"], "vocab_index": [1, 2], "is_leaf": [True, False]})
        >>> mix = pl.DataFrame({
        ...     "node_index": [1, 1, 2], "component_index": [1, 2, 2], "weight": [1.0, 0.5, 1.0]})
        >>> build_closure(nodes, mix).sort("node")["node"].to_list()
        ['A', 'A//B']
    """
    index_to_name = dict(zip(nodes_df["vocab_index"], nodes_df["node"], strict=True))
    leaf_indices = set(nodes_df.filter(pl.col("is_leaf"))["vocab_index"].to_list())

    rows = mix_df.filter(pl.col("node_index").is_in(list(leaf_indices)))
    return pl.DataFrame(
        {
            "code": [index_to_name[i] for i in rows["node_index"].to_list()],
            "node": [index_to_name[i] for i in rows["component_index"].to_list()],
        }
    )


def load_nodes(ontology_dir: str | Path) -> pl.DataFrame:
    """Read ``nodes.parquet`` — the extended vocabulary."""
    return pl.read_parquet(Path(ontology_dir) / NODES_FILE)


def extended_vocab_size(ontology_dir: str | Path) -> int:
    """``V_ext``: one past the highest node index, i.e. the embedding table's required height."""
    return int(load_nodes(ontology_dir)["vocab_index"].max()) + 1


def extend_code_map(code_to_index: dict[str, int], ontology_dir: str | Path) -> dict[str, int]:
    """Add ancestor names to a cohort's ``code -> index`` map, making them queryable.

    ``setdefault`` semantics: when a name is both a real code and some other code's prefix, its
    canonical leaf index wins.

    Examples:
        >>> import tempfile, polars as pl
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = pl.DataFrame({"node": ["A//B", "A"], "vocab_index": [1, 7],
        ...                       "is_leaf": [True, False]}).write_parquet(Path(d) / NODES_FILE)
        ...     extend_code_map({"A//B": 1}, d)
        {'A//B': 1, 'A': 7}
    """
    extended = dict(code_to_index)
    nodes = load_nodes(ontology_dir)
    for node, idx in zip(nodes["node"], nodes["vocab_index"], strict=True):
        extended.setdefault(node, int(idx))
    return extended


def load_mix_matrix(ontology_dir: str | Path, normalize: bool = True) -> torch.Tensor:
    """Load the sparse ``(V_ext, V_ext)`` embedding-mix matrix; rows sum to 1 when normalised."""
    ontology_dir = Path(ontology_dir)
    mix = pl.read_parquet(ontology_dir / MIX_FILE)
    v_ext = extended_vocab_size(ontology_dir)

    idx = torch.tensor([mix["node_index"].to_list(), mix["component_index"].to_list()], dtype=torch.long)
    w = torch.tensor(mix["weight"].to_list(), dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, w, size=(v_ext, v_ext)).coalesce()  # noqa: N806 — `A` is the mix matrix throughout the docs

    if normalize:
        # clamp guards row 0 (PAD), which is never a node and therefore has no entries at all.
        row_sums = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-9)
        vals = A.values() / row_sums[A.indices()[0]]
        A = torch.sparse_coo_tensor(A.indices(), vals, size=(v_ext, v_ext)).coalesce()  # noqa: N806
    return A


def load_closure_map(ontology_dir: str | Path) -> pl.DataFrame:
    """Read ``closure.parquet`` — ``(code, node)`` rows for event explosion."""
    return pl.read_parquet(Path(ontology_dir) / CLOSURE_FILE)


def explode_events_to_closure(events_df: pl.DataFrame, closure_df: pl.DataFrame) -> pl.DataFrame:
    """Repeat each event under every ancestor node name, so ancestor queries label normally.

    Codes absent from ``closure_df`` would be **dropped** by the inner join, silently deleting
    events — which happens whenever the ontology was built from a different ``codes.parquet``
    than the cohort.  Those codes are passed through unchanged instead, and the mismatch is
    reported.

    Examples:
        >>> from datetime import datetime
        >>> ev = pl.DataFrame({"subject_id": [1], "time": [datetime(2024, 1, 1)], "code": ["A//B"]})
        >>> cl = pl.DataFrame({"code": ["A//B", "A//B"], "node": ["A//B", "A"]})
        >>> sorted(explode_events_to_closure(ev, cl)["code"].to_list())
        ['A', 'A//B']

        An event whose code the ontology does not know survives as itself:

        >>> ev2 = pl.DataFrame({"subject_id": [1, 1], "time": [datetime(2024, 1, 1)] * 2,
        ...                     "code": ["A//B", "UNKNOWN"]})
        >>> sorted(explode_events_to_closure(ev2, cl)["code"].to_list())
        ['A', 'A//B', 'UNKNOWN']
    """
    known = set(closure_df["code"].to_list())
    present = set(events_df["code"].to_list())
    missing = present - known
    if missing:
        logger.warning(
            "%d event code(s) are absent from the ontology closure and are kept unexploded "
            "(the ontology may have been built from a different codes.parquet): %s",
            len(missing),
            sorted(missing)[:5],
        )

    exploded = (
        events_df.join(closure_df, on="code", how="inner")
        .drop("code")
        .rename({"node": "code"})
        .select(events_df.columns)
    )
    if not missing:
        return exploded
    return pl.concat([exploded, events_df.filter(pl.col("code").is_in(list(missing)))], how="vertical")
