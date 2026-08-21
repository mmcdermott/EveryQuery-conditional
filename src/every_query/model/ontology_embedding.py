"""Embedding table whose rows are ontology-mixed averages of a raw table's rows.

Substituted for ModernBERT's ``tok_embeddings``, this makes every lookup return

    ``(A @ W)[ids]``

where ``W`` is the ordinary learned table and ``A`` is the row-normalised ancestor-mix matrix
from :mod:`every_query.data.ontology`.  A rare leaf code's representation is therefore pulled
toward its better-estimated parents, and an ancestor node — which never appears in a patient
stream — still receives gradient through every descendant that does.

Substituting the module rather than changing call sites is what makes the feature compose: the
patient encoder, the query code slot and an event-bounded query's boundary code all reach the
same attribute, so each one inherits ontology structure for free.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class OntologyEmbedding(torch.nn.Module):
    """Wraps a raw ``nn.Embedding`` with a sparse ancestor-mix matrix.

    The mixed table is **cached per forward pass** rather than recomputed per lookup.  Each
    lookup materialises the full dense ``(V_ext, H)`` product — 25 MB at a 16k vocabulary and
    ``H=384``, plus as much again in backward — and the conditional model touches the embedding
    table two to three times per forward (patient encoder, query codes, boundary codes).
    Recomputing it each time is cheap in FLOPs and dominant in activation memory,
    so the owning model clears the cache once per forward via a pre-hook and every lookup in
    between shares one product (and one autograd node).

    Args:
        raw: The underlying learned table, sized ``(V_ext, H)``.
        mix: Sparse ``(V_ext, V_ext)`` row-normalised mix matrix.

    Examples:
        A two-node ontology where node 1's row is the average of itself and node 0:

        >>> raw = torch.nn.Embedding(2, 3)
        >>> _ = raw.weight.data.copy_(torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))
        >>> mix = torch.sparse_coo_tensor(
        ...     torch.tensor([[0, 1, 1], [0, 0, 1]]), torch.tensor([1.0, 0.5, 0.5]), (2, 2)
        ... ).coalesce()
        >>> emb = OntologyEmbedding(raw, mix)
        >>> emb(torch.tensor([0]))
        tensor([[1., 1., 1.]], grad_fn=<IndexBackward0>)
        >>> emb(torch.tensor([1]))  # (0.5 * 1 + 0.5 * 3) = 2
        tensor([[2., 2., 2.]], grad_fn=<IndexBackward0>)

        Multi-dimensional index tensors work, so a whole ``(B, L, K)`` block of ids can be
        looked up in one call:

        >>> emb(torch.zeros(2, 2, 2, dtype=torch.long)).shape
        torch.Size([2, 2, 2, 3])

        Gradient reaches the raw table through the mix, which is how an ancestor row gets
        trained by its descendants:

        >>> emb.clear_cache()
        >>> emb(torch.tensor([1])).sum().backward()
        >>> raw.weight.grad
        tensor([[0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000]])
    """

    def __init__(self, raw: torch.nn.Embedding, mix: torch.Tensor):
        super().__init__()
        self.tok = raw
        self.register_buffer("mix", mix, persistent=False)
        self._mixed: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        """The raw table's weight — kept for code that introspects ``.weight`` (e.g. tying)."""
        return self.tok.weight

    @property
    def num_embeddings(self) -> int:
        return self.tok.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.tok.embedding_dim

    def clear_cache(self, *_args) -> None:
        """Drop the cached mixed table.  Called once per forward by the owning model's pre-hook.

        Also drops the autograd graph the cached tensor holds, so a cached product never
        survives into a second backward pass.
        """
        self._mixed = None

    def mixed_weight(self) -> torch.Tensor:
        """The full mixed table ``A @ W``, computed at most once per forward."""
        if self._mixed is None:
            # Force the sparse mm in the raw table's dtype: under autocast a bf16 dense operand
            # would not match the fp32 sparse matrix.
            with torch.autocast(device_type=self.tok.weight.device.type, enabled=False):
                self._mixed = torch.sparse.mm(self.mix.to(self.tok.weight.dtype), self.tok.weight)
        return self._mixed

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.mixed_weight()[ids]


def wrap_tok_embeddings(model: torch.nn.Module, mix: torch.Tensor) -> OntologyEmbedding:
    """Replace ``model.HF_model.embeddings.tok_embeddings`` with an :class:`OntologyEmbedding`.

    Registers a forward pre-hook on ``model`` that clears the per-forward cache, and returns the
    wrapper.  Raises if the raw table is smaller than the mix matrix — that means the encoder was
    sized from the cohort vocabulary ``V`` rather than the ontology's extended ``V_ext``, and
    every ancestor index would be out of range.
    """
    raw = model.HF_model.embeddings.tok_embeddings
    v_ext = mix.shape[0]
    if raw.num_embeddings < v_ext:
        raise ValueError(
            f"Embedding table has {raw.num_embeddings} rows but the ontology needs {v_ext} "
            f"(V_ext).  Size the encoder from the ontology: train.py does this automatically "
            f"when `lightning_module.model.ontology_dir` is set."
        )

    wrapper = OntologyEmbedding(raw, mix)
    model.HF_model.embeddings.tok_embeddings = wrapper
    model.register_forward_pre_hook(wrapper.clear_cache)
    logger.info("Ontology embeddings active: %d nodes, %d mix entries.", v_ext, mix._nnz())
    return wrapper
