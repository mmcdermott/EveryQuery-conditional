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
   ``decay ** distance``.  Installed as the encoder's input-embedding module, this ties a rare
   leaf's representation to its better-estimated parents — the actual hypothesis under test.

Three artifacts are written by ``EQ_build_ontology`` into one directory:

- ``ontology_vocab.parquet``  ``(node_name, token_id, is_observed_code)`` — the extended vocabulary;
  ``V_ext``.  ``is_observed_code`` means "can appear directly in the event stream".
- ``embedding_mix.parquet``  ``(target_token_id, component_token_id, unnormalized_weight)`` — COO
  entries of ``A`` before row normalisation.
- ``event_to_query_nodes.parquet``  ``(event_code, query_node)`` — every observed code paired with
  itself and each query node it satisfies, used to explode an event stream so ancestor queries can
  be labelled by ordinary occurrence.

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

ONTOLOGY_VOCAB_FILE = "ontology_vocab.parquet"
EMBEDDING_MIX_FILE = "embedding_mix.parquet"
EVENT_TO_QUERY_NODES_FILE = "event_to_query_nodes.parquet"


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


def build_ontology(
    codes_df: pl.DataFrame, decay: float = 0.5, subtree_suffix: str | None = "ANY"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build ``(nodes_df, mix_df)`` from a ``codes.parquet``-shaped frame.

    Args:
        codes_df: Columns ``code``, ``code/vocab_index``, optionally ``parent_codes``.
        decay: Per-level weight decay.  An ancestor ``d`` levels up contributes ``decay ** d``
            before row normalisation, so ``0.0`` disables mixing entirely and ``1.0`` weights
            every ancestor as much as the node itself.
        subtree_suffix: Name suffix for the *subtree node* minted beside every name that is both
            a real code and someone's ancestor (see the dual-role section in the body).  ``None``
            skips minting them, leaving such names purely exact and their subtree meaning
            unaskable.  Changing this changes ``V_ext``, so an ontology and the encoder sized
            from it must be built with the same value.

    Returns:
        ``nodes_df`` ``(node_name, token_id, is_observed_code)`` — leaves keep their original indices,
        ancestors get fresh ones from ``max_leaf_index + 1`` upward, sorted by name — and
        ``mix_df`` ``(target_token_id, component_token_id, unnormalized_weight)``, unnormalised, with a
        weight-1 self-loop for every node.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": ["A//B//C", "A//B//D", "E"],
        ...     "code/vocab_index": [1, 2, 3],
        ...     "parent_codes": [["G/x"], None, None],
        ... })
        >>> nodes, mix = build_ontology(df)
        >>> sorted(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
        ['A', 'A//B', 'G/x']

        Leaf indices are untouched, so an ontology can be dropped onto an existing cohort:

        >>> nodes.filter(pl.col("is_observed_code")).sort("token_id")["token_id"].to_list()
        [1, 2, 3]

        ``A//B//C`` mixes itself (1.0), ``A//B`` (0.5), ``A`` (0.25) and its declared parent
        ``G/x`` (0.5, a grouper edge counts as distance 1):

        >>> m = {r["component_token_id"]: r["unnormalized_weight"]
        ...      for r in mix.filter(pl.col("target_token_id") == 1).iter_rows(named=True)}
        >>> idx = dict(zip(nodes["node_name"], nodes["token_id"]))
        >>> m[idx["A//B//C"]], m[idx["A//B"]], m[idx["A"]], m[idx["G/x"]]
        (1.0, 0.5, 0.25, 0.5)

        Every node reachable in the index has a mix row, so no node can end up with an
        all-zero embedding:

        >>> set(nodes["token_id"]) == set(mix["target_token_id"].unique())
        True

        A **dual-role** name -- both a real code and the prefix of another code -- keeps its
        exact meaning and gains a separate subtree node, so both questions stay askable:

        >>> df = pl.DataFrame({
        ...     "code": ["INF//220949", "INF//220949//value_lo", "INF//220949//value_hi"],
        ...     "code/vocab_index": [1, 2, 3],
        ... })
        >>> nodes, mix = build_ontology(df)
        >>> sorted(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
        ['INF', 'INF//220949//ANY']

        ``INF//220949`` stays a leaf (it names 365k real events in the real cohort), while
        ``INF//220949//ANY`` is the node that means "the drug, valued or not":

        >>> closure = build_closure(nodes, mix)
        >>> sorted(closure.filter(pl.col("query_node") == "INF//220949//ANY")["event_code"].to_list())
        ['INF//220949', 'INF//220949//value_hi', 'INF//220949//value_lo']

        Nothing rolls up to the leaf itself except the leaf, which is what keeps its ordinary
        query exact:

        >>> sorted(closure.filter(pl.col("query_node") == "INF//220949")["event_code"].to_list())
        ['INF//220949']

        With ``subtree_suffix=None`` the extra node is not minted at all:

        >>> nodes2, _ = build_ontology(df, subtree_suffix=None)
        >>> sorted(nodes2.filter(~pl.col("is_observed_code"))["node_name"].to_list())
        ['INF']
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}")

    has_parents = "parent_codes" in codes_df.columns
    leaves = codes_df.select(pl.col("code"), pl.col("code/vocab_index").cast(pl.Int64).alias("token_id"))
    leaf_names = set(leaves["code"].to_list())

    # Declared `parent_codes` edges, keyed by the code that declares them.
    declared: dict[str, list[str]] = {}
    if has_parents:
        for row in codes_df.iter_rows(named=True):
            code = row["code"]
            pcs = row.get("parent_codes")
            if pcs:
                kept = [pc for pc in pcs if pc and pc != code]
                if kept:
                    declared[code] = kept

    def direct_parents(name: str) -> list[str]:
        """One hop up: the immediate ``//``-prefix parent, plus any declared groupers.

        Both edge kinds cost exactly one hop, which is what reproduces the historical distance
        scale — ``A//B//C`` reaches ``A//B`` at 1 and ``A`` at 2, and a declared grouper sits at
        1 with its own prefixes at 2, 3, ...  Since ``decay ** dist`` sets the mix weights,
        changing this scale would silently reweight every embedding.
        """
        out: list[str] = []
        if SEP in name:
            out.append(name.rsplit(SEP, 1)[0])
        out.extend(declared.get(name, ()))
        return out

    def ancestors_with_distance(start: str) -> dict[str, int]:
        """Every strict ancestor of ``start`` and its minimum hop count.

        A breadth-first walk over the *union* of prefix and declared edges.  Walking both kinds
        with one traversal is the whole point: the previous implementation followed a declared
        edge exactly one hop and then only string prefixes, so a chain ``X -> P -> GRP//G``
        stopped at ``P`` and ``GRP//G`` never became an ancestor of ``X`` — precisely the
        multi-level DAG that ``parent_codes`` exists to express.

        Note the walk passes *through* a leaf when a leaf is someone's declared parent.  That is
        deliberate and is not in tension with :func:`build_closure` refusing to make that leaf
        addressable as a subtree query: traversal and addressability are different questions.

        ``parent_codes`` is caller-supplied data, so cycles are possible.  ``start`` is never
        admitted to its own ancestor set, and a name already seen at an equal-or-shorter
        distance is not re-expanded, which bounds the walk.
        """
        dist: dict[str, int] = {}
        frontier = [(p, 1) for p in direct_parents(start)]
        while frontier:
            name, d = frontier.pop()
            if name == start:
                continue
            if name in dist and dist[name] <= d:
                continue
            dist[name] = d
            frontier.extend((p, d + 1) for p in direct_parents(name))
        return dist

    node_ancestors: dict[str, dict[str, int]] = {}
    ancestor_names: set[str] = set()
    for code in leaves["code"].to_list():
        amap = ancestors_with_distance(code)
        node_ancestors[code] = amap
        ancestor_names.update(a for a in amap if a not in leaf_names)

    # Every *indexed* node also needs its own mix row, or it embeds to the zero vector.  The walk
    # above is already transitive, so one pass over the ancestor set normally suffices; the loop
    # is kept so "every indexed node has a mix row" holds by construction rather than by luck.
    pending = sorted(ancestor_names)
    while pending:
        nxt: list[str] = []
        for anc in pending:
            if anc in node_ancestors:
                continue
            amap = ancestors_with_distance(anc)
            node_ancestors[anc] = amap
            nxt.extend(a for a in amap if a not in leaf_names and a not in ancestor_names)
        ancestor_names.update(nxt)
        pending = sorted(set(nxt))

    # ---- Dual-role names get a distinct subtree node ----------------------------------------
    #
    # A name can be both a real code and something else's ancestor: `INFUSION_START//220949` is
    # 365,723 unvalued infusion events *and* the prefix of ten `//value_[lo,hi)` variants.  One
    # string cannot mean both "exactly this code" and "this code or any descendant" without one
    # of the two meanings becoming unaskable, and the labeler has to pick one.
    #
    # So mint a second name.  The leaf keeps its exact meaning; a fresh ancestor node
    # `<name>//<suffix>` means the subtree, and every reference to the leaf *as an ancestor* is
    # rewritten to it.  Both rungs of the ladder stay addressable:
    #
    #     INFUSION_START                        all 635 infusion-start codes
    #     INFUSION_START//220949//ANY           the drug, valued or not
    #     INFUSION_START//220949                the 365,723 unvalued events
    #     INFUSION_START//220949//value_[...]   one rate bin
    #
    # `subtree_suffix=None` disables this and leaves dual-role names purely exact -- the subtree
    # meaning is then simply not expressible, which is the cheaper option when a cohort has no
    # dual-role names to begin with.
    if subtree_suffix:
        # Both shapes of dual role are caught by asking which *leaves* turned up as an ancestor:
        # `//`-prefix collisions and a declared `parent_codes` edge pointing at a real code.
        dual_role = {a for amap in node_ancestors.values() for a in amap if a in leaf_names}
        subtree_name = {leaf: f"{leaf}{SEP}{subtree_suffix}" for leaf in dual_role}

        clashes = sorted(n for n in subtree_name.values() if n in leaf_names)
        if clashes:
            raise ValueError(
                f"{len(clashes)} subtree node name(s) collide with real codes (e.g. {clashes[:3]}). "
                f"Pass a different `subtree_suffix` than {subtree_suffix!r}."
            )

        # Rewrite every ancestor reference to a dual-role leaf into its subtree node.  Distances
        # are preserved: the subtree node sits exactly where the leaf used to sit, so the
        # `decay ** dist` mix weights are unchanged for every descendant.
        def rewrite(amap: dict[str, int]) -> dict[str, int]:
            out: dict[str, int] = {}
            for name, dist in amap.items():
                key = subtree_name.get(name, name)
                out[key] = min(out.get(key, 10**9), dist)
            return out

        node_ancestors = {node: rewrite(amap) for node, amap in node_ancestors.items()}

        for leaf, sub in subtree_name.items():
            # The subtree node generalises the leaf, so it inherits the leaf's own ancestors...
            node_ancestors[sub] = dict(node_ancestors[leaf])
            # ...and sits one hop above the leaf itself, which is a member of its own subtree.
            node_ancestors[leaf] = dict(node_ancestors[leaf]) | {sub: 1}

    ancestor_names = (
        set(node_ancestors) | {a for amap in node_ancestors.values() for a in amap}
    ) - leaf_names

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

    max_leaf = int(leaves["token_id"].max())
    anc_sorted = sorted(ancestor_names)
    anc_index = {a: max_leaf + 1 + i for i, a in enumerate(anc_sorted)}
    name_to_index = dict(zip(leaves["code"].to_list(), leaves["token_id"].to_list(), strict=True)) | anc_index

    assert set(ancestor_names) <= node_ancestors.keys(), (
        "every indexed ancestor must have a mix row; otherwise it embeds to the zero vector"
    )

    nodes_df = pl.concat(
        [
            leaves.with_columns(pl.lit(True).alias("is_observed_code")).rename({"code": "node_name"}),
            pl.DataFrame(
                {
                    "node_name": anc_sorted,
                    "token_id": [anc_index[a] for a in anc_sorted],
                    "is_observed_code": [False] * len(anc_sorted),
                }
            ),
        ],
        how="vertical_relaxed",
    )

    mix_rows: dict[str, list] = {"target_token_id": [], "component_token_id": [], "unnormalized_weight": []}
    for node, amap in node_ancestors.items():
        ni = name_to_index[node]
        mix_rows["target_token_id"].append(ni)
        mix_rows["component_token_id"].append(ni)
        mix_rows["unnormalized_weight"].append(1.0)
        for anc, dist in amap.items():
            mix_rows["target_token_id"].append(ni)
            mix_rows["component_token_id"].append(name_to_index[anc])
            mix_rows["unnormalized_weight"].append(decay**dist)

    return nodes_df, pl.DataFrame(mix_rows)


def build_closure(nodes_df: pl.DataFrame, mix_df: pl.DataFrame) -> pl.DataFrame:
    """``(event_code, query_node)`` rows: every leaf paired with itself and each ancestor *node* above it.

    This is what lets an ancestor query be answered by the ordinary occurrence labeler — explode
    the event stream through it and "did any descendant of X occur" becomes "did X occur".

    **Only non-leaf components survive, plus the self-pair.**  A name that is itself a real code
    is never addressable as a subtree query — ``build_query_universe`` draws ancestor slots from
    non-leaf nodes only — so emitting ``(A//B//C -> A//B)`` for a leaf ``A//B`` could not widen
    anything the sampler would ask; it could only corrupt the ordinary leaf query ``A//B``,
    silently changing its meaning from "this exact code occurred" to "this code **or any
    descendant** occurred".  That flipped labels False -> True with no crash and no warning.

    The self-pair must survive the filter: ``(event_code=A//B, query_node=A//B)`` is what makes a leaf query
    answerable at all.  Dropping rows by ``component_token_id == target_token_id`` instead would break
    every leaf query, so the two conditions are deliberately spelled out separately below.

    ``mix_df`` is left alone.  Embedding *sharing* between ``A//B//C`` and ``A//B`` is desirable
    and is not what was broken — only the labeling closure is narrowed here.

    Examples:
        >>> nodes = pl.DataFrame({
        ...     "node_name": ["A//B", "A"], "token_id": [1, 2], "is_observed_code": [True, False]})
        >>> mix = pl.DataFrame({
        ...     "target_token_id": [1, 1, 2], "component_token_id": [1, 2, 2],
        ...     "unnormalized_weight": [1.0, 0.5, 1.0]})
        >>> build_closure(nodes, mix).sort("query_node")["query_node"].to_list()
        ['A', 'A//B']

        A leaf that prefixes another leaf keeps its exact meaning — ``A//B`` is a real code here,
        so it is not emitted as a node above ``A//B//C``:

        >>> nodes = pl.DataFrame({
        ...     "node_name": ["A//B", "A//B//C", "A"], "token_id": [1, 2, 3],
        ...     "is_observed_code": [True, True, False]})
        >>> mix = pl.DataFrame({
        ...     "target_token_id": [1, 1, 2, 2, 2, 3],
        ...     "component_token_id": [1, 3, 2, 1, 3, 3],
        ...     "unnormalized_weight": [1.0, 0.5, 1.0, 0.5, 0.25, 1.0]})
        >>> cl = build_closure(nodes, mix).sort("event_code", "query_node")
        >>> list(zip(cl["event_code"], cl["query_node"]))
        [('A//B', 'A'), ('A//B', 'A//B'), ('A//B//C', 'A'), ('A//B//C', 'A//B//C')]
    """
    index_to_name = dict(zip(nodes_df["token_id"], nodes_df["node_name"], strict=True))
    leaf_indices = set(nodes_df.filter(pl.col("is_observed_code"))["token_id"].to_list())

    leaf_list = list(leaf_indices)
    rows = mix_df.filter(
        pl.col("target_token_id").is_in(leaf_list)
        & (
            (pl.col("component_token_id") == pl.col("target_token_id"))
            | ~pl.col("component_token_id").is_in(leaf_list)
        )
    )
    return pl.DataFrame(
        {
            "event_code": [index_to_name[i] for i in rows["target_token_id"].to_list()],
            "query_node": [index_to_name[i] for i in rows["component_token_id"].to_list()],
        }
    )


def load_nodes(ontology_dir: str | Path) -> pl.DataFrame:
    """Read ``ontology_vocab.parquet`` — the extended vocabulary."""
    return pl.read_parquet(Path(ontology_dir) / ONTOLOGY_VOCAB_FILE)


def extended_vocab_size(ontology_dir: str | Path) -> int:
    """``V_ext``: one past the highest node index, i.e. the embedding table's required height."""
    return int(load_nodes(ontology_dir)["token_id"].max()) + 1


def extend_code_map(code_to_index: dict[str, int], ontology_dir: str | Path) -> dict[str, int]:
    """Add ancestor names to a cohort's ``code -> index`` map, making them queryable.

    ``setdefault`` semantics: when a name is both a real code and some other code's prefix, its
    canonical leaf index wins.

    Examples:
        >>> import tempfile, polars as pl
        >>> with tempfile.TemporaryDirectory() as d:
        ...     nodes = pl.DataFrame({"node_name": ["A//B", "A"], "token_id": [1, 7],
        ...                           "is_observed_code": [True, False]})
        ...     _ = nodes.write_parquet(Path(d) / ONTOLOGY_VOCAB_FILE)
        ...     extend_code_map({"A//B": 1}, d)
        {'A//B': 1, 'A': 7}
    """
    extended = dict(code_to_index)
    nodes = load_nodes(ontology_dir)
    for node, idx in zip(nodes["node_name"], nodes["token_id"], strict=True):
        extended.setdefault(node, int(idx))
    return extended


def load_mix_matrix(ontology_dir: str | Path, normalize: bool = True) -> torch.Tensor:
    """Load the sparse ``(V_ext, V_ext)`` embedding-mix matrix; rows sum to 1 when normalised."""
    ontology_dir = Path(ontology_dir)
    mix = pl.read_parquet(ontology_dir / EMBEDDING_MIX_FILE)
    v_ext = extended_vocab_size(ontology_dir)

    idx = torch.tensor(
        [mix["target_token_id"].to_list(), mix["component_token_id"].to_list()], dtype=torch.long
    )
    w = torch.tensor(mix["unnormalized_weight"].to_list(), dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, w, size=(v_ext, v_ext)).coalesce()  # noqa: N806 — `A` is the mix matrix throughout the docs

    if normalize:
        # clamp guards row 0 (PAD), which is never a node and therefore has no entries at all.
        row_sums = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-9)
        vals = A.values() / row_sums[A.indices()[0]]
        A = torch.sparse_coo_tensor(A.indices(), vals, size=(v_ext, v_ext)).coalesce()  # noqa: N806
    return A


def load_closure_map(ontology_dir: str | Path) -> pl.DataFrame:
    """Read ``event_to_query_nodes.parquet`` — ``(event_code, query_node)`` rows for event explosion."""
    return pl.read_parquet(Path(ontology_dir) / EVENT_TO_QUERY_NODES_FILE)


def explode_events_to_closure(events_df: pl.DataFrame, closure_df: pl.DataFrame) -> pl.DataFrame:
    """Repeat each event under every ancestor node name, so ancestor queries label normally.

    Codes absent from ``closure_df`` would be **dropped** by the inner join, silently deleting
    events — which happens whenever the ontology was built from a different ``codes.parquet``
    than the cohort.  Those codes are passed through unchanged instead, and the mismatch is
    reported.

    Examples:
        >>> from datetime import datetime
        >>> ev = pl.DataFrame({"subject_id": [1], "time": [datetime(2024, 1, 1)], "code": ["A//B"]})
        >>> cl = pl.DataFrame({"event_code": ["A//B", "A//B"], "query_node": ["A//B", "A"]})
        >>> sorted(explode_events_to_closure(ev, cl)["code"].to_list())
        ['A', 'A//B']

        An event whose code the ontology does not know survives as itself:

        >>> ev2 = pl.DataFrame({"subject_id": [1, 1], "time": [datetime(2024, 1, 1)] * 2,
        ...                     "code": ["A//B", "UNKNOWN"]})
        >>> sorted(explode_events_to_closure(ev2, cl)["code"].to_list())
        ['A', 'A//B', 'UNKNOWN']
    """
    known = set(closure_df["event_code"].to_list())
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
        events_df.join(closure_df, left_on="code", right_on="event_code", how="inner")
        .drop("code")
        .rename({"query_node": "code"})
        .select(events_df.columns)
    )
    if not missing:
        return exploded
    return pl.concat([exploded, events_df.filter(pl.col("code").is_in(list(missing)))], how="vertical")
