"""One source of truth for what a query string means and which codes it mentions.

A "query" in a :class:`~every_query.data.schema.QuerySeqSchema` ``queries`` list is usually a
bare MEDS vocabulary code, but not always.  Aggregate queries are parseable *expressions* over
several component codes, and ontology queries name *ancestors* that are not cohort codes at all.
Everywhere a query string meets a vocabulary, the question being asked is really "which
vocabulary codes does this refer to?" — and answering it separately in each place is how the
upstream experiments ended up shipping two different partial fixes to the same bug (exp-5's
``fe47d19`` and exp-6's ``b376223``, each patching a different subset of the call sites).

The four sites are:

- :meth:`~every_query.data.seq_dataset.ConditionalQueryPytorchDataset.encode_query`,
- ``every_query.predict.predict._check_vocab``,
- ``every_query.generate_tasks.sample_evaluation_query_sequences.validate_spec_codes``,
- ``every_query.generate_tasks.sample_tasks.read_query_codes`` (the sampling universe).

Aggregate syntax (ops are added to the model by the aggregate-queries feature; the grammar
lives here because vocabulary validation needs it regardless):

- ``ANY(c1|c2|...)``       — any component occurs in the window;
- ``ALL(c1|c2|...)``       — every component occurs;
- ``SEQ(c1>c2|gap=G)`` / ``SEQ(c1>c2|gap=Gs:Ge)`` — ordered, with the gap in ``[Gs, Ge)`` days;
- ``CO(c1&c2)``            — both at the *same* timestamp;
- ``WITHIN(c1&c2|W=W)``    — unordered, within ``W`` days of each other;
- ``GE2(c1|c2|c3)``        — at least two of three occur;
- ``XOR(c1|c2)``           — exactly one occurs;
- anything else            — an atomic vocabulary code.

Because a bare code parses as an atom, every function here is a no-op on the query forms that
exist today; nothing changes until a generator starts emitting expressions.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# Characters carrying meaning in the aggregate grammar.  A component code containing one of
# these could not be round-tripped through a query string, so it must be excluded from
# component pools *and* from any generated ontology-ancestor names (upstream exp-6 hit the
# first half of this as a live bug and fixed it in 5fd5acb; the ancestor half was unreachable
# there because no fork ran ontology and aggregate together).
RESERVED_CHARS = frozenset("|>&()")

OP_ATOM, OP_ANY, OP_ALL, OP_SEQ = 0, 1, 2, 3
OP_CO, OP_WITHIN, OP_GE2, OP_XOR = 4, 5, 6, 7
N_OPS = 8

_AGG_RE = re.compile(r"^(ANY|ALL|SEQ|CO|WITHIN|GE2|XOR)\((.*)\)$")
_OPS_BY_NAME = {
    "ANY": OP_ANY,
    "ALL": OP_ALL,
    "SEQ": OP_SEQ,
    "CO": OP_CO,
    "WITHIN": OP_WITHIN,
    "GE2": OP_GE2,
    "XOR": OP_XOR,
}


@dataclass(frozen=True)
class ParsedQuery:
    """A query string decomposed into its operator, component codes and temporal bounds.

    Attributes:
        op: One of the ``OP_*`` constants; ``OP_ATOM`` for a bare vocabulary code.
        components: The vocabulary codes the query mentions, in written order.
        gap_days: Upper temporal bound in days (``SEQ``'s ``Ge``, ``WITHIN``'s ``W``); ``None``
            when the operator has no temporal bound.
        gap_lo: Lower temporal bound in days (``SEQ``'s ``Gs``); ``0.0`` when unbounded below.
    """

    op: int
    components: list[str] = field(default_factory=list)
    gap_days: float | None = None
    gap_lo: float = 0.0


def parse_query(query: str) -> ParsedQuery:
    """Decompose a query string.  A bare code parses as an atom naming itself.

    Examples:
        Atoms — the overwhelmingly common case, including codes with ``//`` separators:

        >>> parse_query("LAB//GLUCOSE").op == OP_ATOM
        True
        >>> parse_query("LAB//GLUCOSE").components
        ['LAB//GLUCOSE']

        Set operators split on ``|``:

        >>> p = parse_query("ANY(A|B//C)")
        >>> p.op == OP_ANY, p.components
        (True, ['A', 'B//C'])
        >>> parse_query("GE2(A|B|C)").components
        ['A', 'B', 'C']
        >>> parse_query("XOR(A|B)").op == OP_XOR
        True

        ``SEQ`` is ordered and carries a gap range; a bare gap means ``[0, G)``:

        >>> p = parse_query("SEQ(A>B|gap=3.5)")
        >>> p.components, p.gap_lo, p.gap_days
        (['A', 'B'], 0.0, 3.5)
        >>> p = parse_query("SEQ(A>B|gap=1.0:7.0)")
        >>> p.gap_lo, p.gap_days
        (1.0, 7.0)

        Co-timed and windowed operators split on ``&``:

        >>> parse_query("CO(A&B)").components
        ['A', 'B']
        >>> p = parse_query("WITHIN(A&B|W=2.0)")
        >>> p.components, p.gap_days
        (['A', 'B'], 2.0)

        An unrecognised head is an atom, not an error — vocabulary codes are free-form:

        >>> parse_query("SOMETHING(A|B)").op == OP_ATOM
        True
    """
    m = _AGG_RE.match(query)
    if not m:
        return ParsedQuery(op=OP_ATOM, components=[query])

    kind, body = m.groups()
    op = _OPS_BY_NAME[kind]

    if kind in ("ANY", "ALL", "GE2", "XOR"):
        return ParsedQuery(op=op, components=body.split("|"))
    if kind == "CO":
        return ParsedQuery(op=op, components=body.split("&"))

    parts = body.split("|")
    gap_days: float | None = None
    gap_lo = 0.0
    for part in parts[1:]:
        if part.startswith("W="):
            gap_days = float(part[2:])
        elif part.startswith("gap="):
            arg = part[4:]
            if ":" in arg:
                lo_s, hi_s = arg.split(":")
                gap_lo, gap_days = float(lo_s), float(hi_s)
            else:
                gap_days = float(arg)

    sep = "&" if kind == "WITHIN" else ">"
    return ParsedQuery(op=op, components=parts[0].split(sep), gap_days=gap_days, gap_lo=gap_lo)


def is_aggregate(query: str) -> bool:
    """True when the query is a composite expression rather than a bare vocabulary code.

    Examples:
        >>> is_aggregate("ANY(A|B)"), is_aggregate("LAB//GLUCOSE")
        (True, False)
    """
    return parse_query(query).op != OP_ATOM


def component_codes(query: str) -> list[str]:
    """The vocabulary codes a query refers to — itself, if it is a bare code.

    This is the function every vocabulary check should go through: validating an aggregate
    query as if it were a code would reject a perfectly good query, and skipping validation
    would let a typo'd component through to produce meaningless predictions.

    Examples:
        >>> component_codes("LAB//GLUCOSE")
        ['LAB//GLUCOSE']
        >>> component_codes("SEQ(A>B|gap=3)")
        ['A', 'B']
    """
    return parse_query(query).components


def has_reserved_chars(code: str) -> bool:
    """True when a code cannot safely appear as an aggregate component.

    Examples:
        >>> has_reserved_chars("LAB//GLUCOSE")
        False
        >>> has_reserved_chars("DIAGNOSIS//A&B")
        True
    """
    return any(ch in RESERVED_CHARS for ch in code)


def unknown_codes(queries: Iterable[str], vocab: Iterable[str] | set[str]) -> list[str]:
    """Component codes mentioned by ``queries`` that are absent from ``vocab``, sorted unique.

    Examples:
        >>> unknown_codes(["A", "ANY(B|C)"], {"A", "B", "C"})
        []
        >>> unknown_codes(["A", "ANY(B|ZZZ)"], {"A", "B"})
        ['ZZZ']

        The aggregate wrapper itself is never reported as missing — only its components:

        >>> unknown_codes(["ALL(A|B)"], {"A", "B"})
        []
    """
    known = vocab if isinstance(vocab, set) else set(vocab)
    missing = {c for q in queries for c in component_codes(q) if c not in known}
    return sorted(missing)
