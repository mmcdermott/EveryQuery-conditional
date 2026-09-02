"""Conditional query-sequence model: bidirectional patient encoder + block-autoregressive query decoder.

This module holds the **encoder-decoder** architecture
(:class:`ConditionalQueryEncoderDecoderModel`, aliased as ``ConditionalQueryModel`` for backward
compatibility) plus the pieces both conditional architectures share: the answer/token-type
constants, :class:`ConditionalQueryOutput`, :func:`masked_bce` and
:func:`validate_rope_time_pair`.  The alternative **decoder-only** architecture — one Llama
backbone jointly attending over patient history and query stream — lives in
:mod:`every_query.model.conditional_ar_model`.

Where :class:`~every_query.model.EveryQueryModel` answers a *single* ``(code, duration)`` query by
prepending the query to the patient sequence, this model answers a *sequence* of queries
``[Q1][A1][Q2][A2]...[QL][AL]`` against one patient context:

- A **bidirectional encoder** (ModernBERT) embeds the patient's event history up to the
  prediction time.  No query tokens are mixed into the patient sequence.
- A **block-autoregressive decoder** runs over the query/answer token stream and cross-attends to
  the encoded patient state.  Each query block ``j`` is three tokens: ``(code_j, duration_j,
  answer_j)``.  The self-attention mask enforces:

    * tokens of block ``j`` see *all* tokens of blocks ``< j`` (queries AND teacher-forced answers);
    * the ``code``/``duration`` tokens of block ``j`` see each other but **not** ``answer_j``;
    * the ``answer_j`` input token sees its own block's ``code``/``duration`` and itself (it is
      never used to predict ``A_j`` — only to condition blocks ``> j``);
    * nothing sees blocks ``> j``.

  The prediction for ``A_j`` is read from the decoder output at the ``duration_j`` token, so it is
  conditioned on the patient state, all prior queries and their (ground-truth, teacher-forced)
  answers, and ``Q_j`` itself — but never on ``A_j``.

**Every answer is binary**: each query asks *"is ``code_j`` observed in ``(t, t + duration_j]``?"*
and the answer is simply YES/NO (an event we did not observe — because the record ends first — is
NO, not a special "censored" class).  Censoring is expressed *as a query* instead: the
end-of-timeline code ``TIMELINE//END`` is an ordinary vocabulary code, so a query
``(TIMELINE//END, d)`` answered YES means "the record ends within ``d``" (the ``d``-window is not
fully observed) and NO means "data continue past ``t + d``".  A later query that conditions on that
answer recovers — and generalizes — the original EveryQuery's implicit ``P(occurs | data exist
after d)``: with ``TIMELINE//END = NO`` you get exactly that quantity, with ``= YES`` you get
``P(occurs | record ends within d)`` (the actionable form for terminal events such as death), and
the unconditioned ``(code, d)`` query gives the marginal.  Nothing is masked from the loss except
padding; there is no separate censor head and no reserved sentinel index.
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import AutoConfig, ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

from every_query.model.model import MLP

logger = logging.getLogger(__name__)

# Answer-token vocabulary for teacher forcing.  Answers are binary: every query asks
# "is this code observed in (t, t+d)?" and the answer is YES/NO.  There is no separate
# "censored" answer class — censoring is expressed *as a query*, by asking about the
# end-of-timeline code (TIMELINE//END): "[TIMELINE//END, d]" answered YES means the record
# ends within d (the window is not fully observed), NO means data continue past t+d.  A
# downstream query conditions on that answer, which is strictly more expressive than the
# original EveryQuery's implicit "P(occurs | data exist after d)".
ANSWER_NO = 0
ANSWER_YES = 1
N_ANSWER_CLASSES = 2

# Token-type indices within a query block.
TOKEN_CODE = 0
TOKEN_DURATION = 1
TOKEN_ANSWER = 2
TOKENS_PER_QUERY = 3


def build_block_causal_mask(n_queries: int, device: torch.device | None = None) -> torch.Tensor:
    """Build the ``(3L, 3L)`` boolean self-attention mask for ``L`` query blocks.

    Convention matches ``torch.nn.Transformer``: ``True`` means attention is **disallowed**.

    Token layout: position ``3j`` is ``code_j``, ``3j + 1`` is ``duration_j``, ``3j + 2`` is the
    teacher-forced ``answer_j`` input token.

    Allowed attention for a destination token at block ``b_i`` / type ``t_i`` onto a source token
    at block ``b_k`` / type ``t_k``:

      - ``b_k < b_i``  (everything in strictly earlier blocks, answers included), or
      - ``b_k == b_i`` and ``t_k < 2``  (own block's code/duration — never the own answer), or
      - ``b_k == b_i`` and ``t_i == t_k == 2``  (the answer input token attends to itself).

    Examples:
        One query block — code/duration see each other but not the answer; the answer sees all 3:

        >>> m = build_block_causal_mask(1)
        >>> (~m).long()
        tensor([[1, 1, 0],
                [1, 1, 0],
                [1, 1, 1]])

        Two blocks — block 1 tokens see *all* of block 0 (incl. its answer at index 2):

        >>> m = build_block_causal_mask(2)
        >>> (~m).long()[3:, :3]
        tensor([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])

        ...but block 0 tokens see nothing of block 1:

        >>> (~m).long()[:3, 3:]
        tensor([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])

        Within block 1 the sub-structure repeats (no token sees the in-block answer except the
        answer itself):

        >>> (~m).long()[3:, 3:]
        tensor([[1, 1, 0],
                [1, 1, 0],
                [1, 1, 1]])

        No row is fully masked (softmax never sees an all ``-inf`` row):

        >>> bool((~build_block_causal_mask(5)).any(dim=1).all())
        True
    """
    total = TOKENS_PER_QUERY * n_queries
    idx = torch.arange(total, device=device)
    block = idx // TOKENS_PER_QUERY
    tok = idx % TOKENS_PER_QUERY

    b_i, b_k = block.unsqueeze(1), block.unsqueeze(0)
    t_i, t_k = tok.unsqueeze(1), tok.unsqueeze(0)

    allowed = (b_k < b_i) | ((b_k == b_i) & (t_k < TOKEN_ANSWER))
    allowed |= (b_k == b_i) & (t_i == TOKEN_ANSWER) & (t_k == TOKEN_ANSWER)
    return ~allowed


def masked_bce(
    criterion: torch.nn.BCEWithLogitsLoss,
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """BCE-with-logits over ``mask``-selected positions; differentiable zero when empty.

    Shared by both conditional architectures: every real query position is an
    equally-weighted binary prediction point, and padding is the only exclusion.

    Examples:
        >>> crit = torch.nn.BCEWithLogitsLoss()
        >>> logits = torch.tensor([[0.0, 5.0]])
        >>> target = torch.tensor([[1.0, 0.0]])
        >>> only_first = torch.tensor([[True, False]])
        >>> torch.testing.assert_close(
        ...     masked_bce(crit, logits, target, only_first),
        ...     crit(torch.tensor([0.0]), torch.tensor([1.0])),
        ... )

        An all-padding batch yields a differentiable zero rather than NaN:

        >>> masked_bce(crit, logits, target, torch.zeros(1, 2, dtype=torch.bool))
        tensor(0.)
    """
    logits, target = logits[mask], target[mask].float()
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return criterion(logits, target)


def validate_rope_time_pair(use_rope_time: bool, time_pos: torch.Tensor | None) -> torch.Tensor | None:
    """Enforce that ``use_rope_time`` (model) and ``time_pos_ids`` (batch) arrive as a pair.

    Returns the validated ``time_pos`` tensor (``None`` when rope-time is off).  Both
    half-configurations are hard errors rather than silent fallbacks — either direction yields a
    model that trains, validates and checkpoints with normal-looking numbers while its backbone
    is missing the elapsed-time signal (see
    :meth:`ConditionalQueryEncoderDecoderModel._encoder_position_kwargs` for the full
    post-mortem this guard encodes).  Shared by both conditional architectures so the AR model
    cannot silently drift from the encoder-decoder model's contract.
    """
    if not use_rope_time:
        if time_pos is not None:
            raise ValueError(
                "use_rope_time=False but the batch carries time_pos_ids, which only "
                "ConditionalQueryPytorchDataset(..., strip_delta_tokens=True) emits — so the "
                "TIMELINE//DELTA* tokens have already been stripped from the model input, "
                "and token-index positions would leave the model with no elapsed-time "
                "information at all.  Either half of the pair fixes it: set "
                "`lightning_module.model.use_rope_time=true` to consume the positions (most "
                "likely what was meant, since the strip was switched on deliberately), or "
                "`datamodule.dataset_kwargs.strip_delta_tokens=false` to keep the delta "
                "tokens in the stream.  Refusing to silently discard elapsed time."
            )
        return None
    if time_pos is None:
        raise ValueError(
            "use_rope_time=True but the batch carries no time_pos_ids.  Build the dataset "
            "with ConditionalQueryPytorchDataset(..., strip_delta_tokens=True) — via "
            "`datamodule.dataset_kwargs.strip_delta_tokens=true` — so elapsed-hour positions "
            "are emitted.  Refusing to fall back to token-index positions silently."
        )
    return time_pos


def _init_aux_embeddings(std: float, *embeddings: torch.nn.Embedding) -> None:
    """Re-init embedding tables built outside the HF backbone to the backbone's scale.

    HF models initialize their own submodules in ``post_init()`` with
    ``config.initializer_range`` (0.02).  Tables constructed afterwards on the wrapper keep
    ``nn.Embedding``'s default ``N(0, 1)``, ~50x wider, so shared type/position vectors
    dominate the summed input at init.  Call this instead of ``self.apply(...)``, which would
    also reinitialize the already-initialized backbone.

        >>> emb = torch.nn.Embedding(1000, 64)
        >>> _init_aux_embeddings(0.02, emb)
        >>> bool(0.015 < emb.weight.std().item() < 0.025)
        True
    """
    for embedding in embeddings:
        torch.nn.init.normal_(embedding.weight, mean=0.0, std=std)


@dataclass
class ConditionalQueryOutput(BaseModelOutput):
    """Output container for both conditional query-sequence architectures.

    Attributes:
        answer_logits: ``(batch, n_queries)`` logits, one binary occurrence prediction per query
            block (probability that the query's code is observed in its window).
        valid_mask: ``(batch, n_queries)`` bool — True at real (non-padding) query positions, all
            of which carry loss.
    """

    answer_logits: torch.FloatTensor | None = None
    valid_mask: torch.BoolTensor | None = None

    @property
    def answer_probs(self) -> torch.Tensor | None:
        """Sigmoid probabilities for all answer logits, or ``None`` when logits are absent.

        Examples:
            >>> ConditionalQueryOutput(last_hidden_state=None).answer_probs is None
            True
            >>> out = ConditionalQueryOutput(last_hidden_state=None, answer_logits=torch.zeros(2, 3))
            >>> out.answer_probs
            tensor([[0.5000, 0.5000, 0.5000],
                    [0.5000, 0.5000, 0.5000]])

            bf16 logits are upcast before the sigmoid, so confident predictions stay
            distinguishable instead of all saturating to exactly 1.0:

            >>> bf16 = ConditionalQueryOutput(
            ...     last_hidden_state=None, answer_logits=torch.tensor([[8.0]], dtype=torch.bfloat16)
            ... )
            >>> bf16.answer_probs.dtype, bool(bf16.answer_probs < 1.0)
            (torch.float32, True)
        """
        if self.answer_logits is None:
            return None
        # Upcast before the sigmoid, exactly as EveryQueryOutput.logits_to_probs does.  Training
        # runs at bf16-mixed, where sigmoid rounds a logit of ~6 to exactly 1.0 — every confident
        # prediction collapses into a tie, which flattens ranking metrics like AUROC.  These
        # probabilities flow straight into the predictions parquet via predict_sequences, so the
        # damage would show up as a quietly depressed score rather than an error.
        return torch.sigmoid(self.answer_logits.float())


class ConditionalQueryEncoderDecoderModel(torch.nn.Module):
    """Encoder/decoder model for conditional query sequences.

    This is the original conditional architecture (previously named ``ConditionalQueryModel``;
    that name survives as a module-level alias so existing imports, Hydra configs and
    checkpoints keep working).  For the decoder-only alternative see
    :class:`~every_query.model.conditional_ar_model.ConditionalQueryARModel`.

    Args:
        precision: Lightning precision string; sets initial weight dtype like ``EveryQueryModel``.
        mlp_dropout: Dropout used in the answer head MLP.
        model_name: HF model name whose config seeds the encoder.
        num_hidden_layers: Encoder layer-count override.
        config_overrides: Raw overrides applied to the encoder's ``ModernBertConfig``.
        decoder_layers: Number of ``nn.TransformerDecoderLayer`` s.
        decoder_heads: Attention heads in the decoder.
        decoder_ffn_mult: Decoder feed-forward width as a multiple of ``hidden_size``.
        max_queries: Maximum query blocks per sequence (sizes the block-position embedding).
        ontology_dir: Directory of ontology artifacts (``nodes``/``mix``/``closure`` parquets).
            When set, the encoder's input-embedding module is replaced — through
            ``set_input_embeddings`` — so every code embedding becomes the ancestor-mixed
            average ``(A @ W)[ids]``; see :mod:`every_query.model.ontology_embedding`.  The
            encoder must be sized to the ontology's ``V_ext``, which ``train.py`` does
            automatically.  Because the wrapper substitutes the shared table, both the query
            codes and an event-bounded query's boundary codes inherit the structure too.
        use_rope_time: Drive the encoder's rotary positions from ``batch.time_pos_ids``
            (elapsed integer hours) instead of token index.  Pair with
            ``ConditionalQueryPytorchDataset(strip_delta_tokens=True)``, which removes the
            quantized ``TIMELINE//DELTA*`` tokens and emits those positions.  Attention then
            sees continuous relative time rather than token distance.  This flag and the
            dataset's ``strip_delta_tokens`` must be set together: a mismatch in *either*
            direction is a hard error at the first forward pass, because either one silently
            costs the encoder its elapsed-time signal.  See :meth:`_encoder_position_kwargs`.
    """

    PRECISION_TO_MODEL_WEIGHTS_DTYPE: ClassVar[dict[str, torch.dtype]] = {
        "32-true": torch.float32,
        "16-true": torch.float16,
        "16-mixed": torch.float32,
        "bf16-true": torch.bfloat16,
        "bf16-mixed": torch.float32,
        "transformer-engine": torch.bfloat16,
    }

    def __init__(
        self,
        precision: str = "32-true",
        mlp_dropout: float = 0.1,
        model_name: str = "answerdotai/ModernBERT-base",
        num_hidden_layers: int | None = None,
        config_overrides: dict[str, Any] | None = None,
        decoder_layers: int = 4,
        decoder_heads: int | None = None,
        decoder_ffn_mult: int = 4,
        max_queries: int = 8,
        use_rope_time: bool = False,
        ontology_dir: str | None = None,
    ):
        super().__init__()

        self.HF_model_config: ModernBertConfig = AutoConfig.from_pretrained(model_name)
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.HF_model_config, key, value)
        if num_hidden_layers is not None:
            self.HF_model_config.num_hidden_layers = num_hidden_layers

        # ModernBERT's `reference_compile` auto-detection triggers torch.compile, whose
        # Triton/aarch64 path is broken on some devices (observed: GB10).  The encoder is
        # always run uncompiled here; callers can torch.compile() the whole module themselves.
        self.HF_model_config.reference_compile = False
        self.HF_model_config.output_hidden_states = False
        self.HF_model_config.output_attentions = False
        self.HF_model_config.use_cache = False

        extra_kwargs = {"torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}
        self.HF_model = ModernBertModel._from_config(self.HF_model_config, **extra_kwargs)

        H = self.HF_model_config.hidden_size
        n_heads = decoder_heads if decoder_heads is not None else self.HF_model_config.num_attention_heads

        def make_decoder_layer() -> torch.nn.TransformerDecoderLayer:
            return torch.nn.TransformerDecoderLayer(
                d_model=H,
                nhead=n_heads,
                dim_feedforward=decoder_ffn_mult * H,
                dropout=mlp_dropout,
                batch_first=True,
                norm_first=True,
            )

        self.decoder = torch.nn.TransformerDecoder(
            make_decoder_layer(), num_layers=decoder_layers, norm=torch.nn.LayerNorm(H)
        )
        # nn.TransformerDecoder deep-copies its template layer, so every layer would start from
        # byte-identical weights (PyTorch's own docs recommend re-initializing).  Build each layer
        # freshly instead of picking a blanket re-init policy: each submodule keeps its native
        # initializer, just with independent draws.
        self.decoder.layers = torch.nn.ModuleList(make_decoder_layer() for _ in range(decoder_layers))

        self.duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        self.answer_embed = torch.nn.Embedding(N_ANSWER_CLASSES, H)
        self.token_type_embed = torch.nn.Embedding(TOKENS_PER_QUERY, H)
        self.block_pos_embed = torch.nn.Embedding(max_queries, H)
        # Marks a duration slot as event-bounded.  Added to the boundary code's token embedding
        # so the model can tell "window ends at the next X" from "is X observed" — both use the
        # same embedding table.  Always allocated (not gated on a flag) so the parameter set does
        # not depend on the data, which would make checkpoints silently incompatible.
        self.bound_marker = torch.nn.Parameter(torch.randn(H) * self.HF_model_config.initializer_range)
        self.answer_mlp = MLP(layers=[H, 128, 1], dropout_prob=mlp_dropout)

        # See _init_aux_embeddings: built outside the backbone, so ModernBertModel's own init
        # never reached them and they kept nn.Embedding's default N(0, 1).
        _init_aux_embeddings(
            self.HF_model_config.initializer_range,
            self.answer_embed,
            self.token_type_embed,
            self.block_pos_embed,
        )

        self.max_queries = max_queries
        self.use_rope_time = use_rope_time
        self.ontology_dir = ontology_dir
        if ontology_dir is not None:
            # Must run after HF_model exists and before any lookup.  Substituting the module
            # (rather than patching call sites) is what lets query codes and boundary codes
            # alike inherit ontology structure without further changes.
            from every_query.data.ontology import load_mix_matrix
            from every_query.model.ontology_embedding import wrap_tok_embeddings

            wrap_tok_embeddings(self, load_mix_matrix(ontology_dir))
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.hparams = {
            "precision": precision,
            "mlp_dropout": mlp_dropout,
            "model_name": model_name,
            "num_hidden_layers": self.HF_model_config.num_hidden_layers,
            "config_overrides": dict(config_overrides) if config_overrides else None,
            "decoder_layers": decoder_layers,
            "decoder_heads": n_heads,
            "decoder_ffn_mult": decoder_ffn_mult,
            "max_queries": max_queries,
            "use_rope_time": use_rope_time,
            "ontology_dir": ontology_dir,
        }

    @property
    def max_seq_len(self) -> int:
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        return self.HF_model_config.vocab_size

    def _masked_bce(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """BCE-with-logits over ``mask``-selected positions; differentiable zero when empty."""
        return masked_bce(self.criterion, logits, target, mask)

    def _encoder_position_kwargs(self, batch) -> dict[str, torch.Tensor]:
        """Extra ``HF_model`` kwargs carrying the encoder's rotary positions.

        Empty when ``use_rope_time`` is off *and* the batch carries no ``time_pos_ids``, so
        ModernBERT uses token-index positions and elapsed time reaches the encoder the ordinary
        way — as the quantized ``TIMELINE//DELTA*`` tokens sitting in the stream.

        ``use_rope_time`` (model) and ``strip_delta_tokens`` (dataset) are two halves of one
        setting, and **either** half-configuration is a hard error rather than a silent
        fallback.  Both directions yield a model that trains, validates and checkpoints with
        entirely normal-looking numbers while its encoder is missing the time signal:

        * ``use_rope_time=True`` with no ``time_pos_ids`` — the model would quietly train or
          score on token-index positions, indistinguishable from success until the numbers are
          already published.  The upstream experiment hit exactly this and had to issue an
          erratum after a full eval grid had been computed against a model that never received
          its time positions.  The misconfiguration is a real one: pairing a RoPE checkpoint
          with a dataset built without ``strip_delta_tokens=True``.
        * ``use_rope_time=False`` with ``time_pos_ids`` present — worse, and the reason this
          direction is checked at all.  ``time_pos_ids`` is emitted by exactly one code path,
          ``ConditionalQueryPytorchDataset(strip_delta_tokens=True)``, so its presence on the
          batch means the strip was **asked for**, and normally that the ``TIMELINE//DELTA*``
          tokens have already been deleted from ``batch.code``.  Ignoring the positions then
          leaves the encoder with **no elapsed-time information at all**: the delta tokens are
          gone from the stream and the elapsed hours that replaced them are dropped on the
          floor.  Nothing downstream recovers them — ``time_delta_days`` survives on the batch
          but never reaches the encoder, which reads only ``code``, the attention mask and
          these positions.

          One case emits the positions without deleting anything: a cohort whose vocabulary has
          no ``TIMELINE//DELTA*`` codes at all, where the dataset warns and carries on
          (:mod:`every_query.data.seq_dataset`).  Refusing is still correct there — the strip
          was requested deliberately, nothing is consuming the positions, and the run is
          misconfigured in a way the user needs told about rather than smoothed over.

        Rotary frequencies are computed from ``position_ids`` on the fly rather than looked up
        in a table, so elapsed-hour values far exceeding ``max_position_embeddings`` are
        well-defined.  Only the *local*-attention sliding-window mask is token-index based, and
        it is deliberately left that way — windows are over neighbouring events, not hours.
        """
        time_pos = validate_rope_time_pair(self.use_rope_time, getattr(batch, "time_pos_ids", None))
        if time_pos is None:
            return {}
        return {"position_ids": time_pos.to(batch.code.device)}

    def _query_code_embeds(self, batch) -> torch.Tensor:
        """``(B, L, H)`` content of each query block's **code** slot.

        A seam: a feature that changes *what is being asked about* replaces this, while the
        block layout, the mask and the answer readout stride stay untouched.
        """
        return self.HF_model.get_input_embeddings()(batch.q_codes)

    def _query_duration_embeds(self, batch) -> torch.Tensor:
        """``(B, L, H)`` content of each query block's **duration** slot.

        A seam: features that change *what bounds the window* (an event boundary rather than a
        scalar horizon) replace this.  Note the answer for block ``j`` is read from this slot's
        output position, so whatever goes here must stay a per-query vector.

        For an **event-bounded** query the window runs to the next occurrence of a boundary
        code rather than for a fixed horizon, so the slot carries that code's token embedding
        plus a learned marker distinguishing "this is a boundary" from "this code is being
        asked about" (the same embedding table feeds the code slot).  The scalar-duration MLP
        is bypassed entirely for those positions — its input there is only the sentinel.

        With no bounds present this is exactly the scalar path, so a bound-free dataset trains
        bit-identically to a model without the feature.
        """
        dur_emb = self.duration_embed((batch.q_durations / 365.0).unsqueeze(-1))

        q_bounds = getattr(batch, "q_bound_codes", None)
        if q_bounds is None:
            return dur_emb

        bound_emb = self.HF_model.get_input_embeddings()(q_bounds).to(dur_emb.dtype)
        bound_emb = bound_emb + self.bound_marker.to(dur_emb.dtype)
        return torch.where((q_bounds > 0).unsqueeze(-1), bound_emb, dur_emb)

    def _decoder_tokens(self, batch) -> torch.Tensor:
        """Build the interleaved ``(B, 3L, H)`` decoder input from a ConditionalQueryBatch.

        Exactly three tokens per query block.  Nothing here may add a fourth: ``TOKENS_PER_QUERY``
        is baked into :func:`build_block_causal_mask`, the ``TOKEN_DURATION::TOKENS_PER_QUERY``
        answer readout stride, and the mask's doctests.
        """
        B, L = batch.q_codes.shape

        code_emb = self._query_code_embeds(batch)
        dur_emb = self._query_duration_embeds(batch)
        ans_emb = self.answer_embed(batch.q_answers)

        tokens = torch.stack([code_emb, dur_emb, ans_emb], dim=2)  # (B, L, 3, H)

        device = tokens.device
        block_idx = torch.arange(L, device=device).clamp(max=self.max_queries - 1)
        tokens = tokens + self.block_pos_embed(block_idx).view(1, L, 1, -1)
        tokens = tokens + self.token_type_embed(torch.arange(TOKENS_PER_QUERY, device=device)).view(
            1, 1, TOKENS_PER_QUERY, -1
        )

        return tokens.reshape(B, L * TOKENS_PER_QUERY, -1)

    def forward(self, batch) -> tuple[torch.FloatTensor, ConditionalQueryOutput]:
        """Run encoder + block-autoregressive decoder; return ``(loss, outputs)``.

        Expects a ``ConditionalQueryBatch`` with ``code`` (patient tokens), ``q_codes``,
        ``q_durations``, ``q_answers`` and ``q_mask``.
        """
        _, L = batch.q_codes.shape

        enc_attention_mask = batch.code != batch.PAD_INDEX
        memory = self.HF_model(
            input_ids=batch.code, attention_mask=enc_attention_mask, **self._encoder_position_kwargs(batch)
        ).last_hidden_state

        tgt = self._decoder_tokens(batch)
        tgt_mask = build_block_causal_mask(L, device=tgt.device)
        tgt_key_padding_mask = (~batch.q_mask).repeat_interleave(TOKENS_PER_QUERY, dim=1)

        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=~enc_attention_mask,
        )

        # The answer for block j is predicted from the duration token at position 3j + 1 —
        # conditioned on the patient state, prior blocks (incl. answers) and Q_j, never A_j.
        answer_hidden = dec_out[:, TOKEN_DURATION::TOKENS_PER_QUERY, :]  # (B, L, H)
        answer_logits = self.answer_mlp(answer_hidden).squeeze(-1)  # (B, L)

        targets = (batch.q_answers == ANSWER_YES).float()
        # Every real (non-padding) query position is a simultaneous, equally-weighted binary
        # prediction point.  Padding positions are the only exclusion; there is no censored
        # answer class to mask — censoring is just the TIMELINE//END query, answered like any
        # other code.
        valid = batch.q_mask
        loss = self._masked_bce(answer_logits, targets, valid)

        outputs = ConditionalQueryOutput(
            last_hidden_state=None,
            answer_logits=answer_logits,
            valid_mask=valid,
        )
        return loss, outputs


# Backward-compatible alias: the class was named ``ConditionalQueryModel`` before the
# decoder-only architecture existed (issue #14).  Existing imports, Hydra ``_target_`` strings
# and checkpoint-restore paths that reference the old name resolve to the same class — same
# parameters, same state-dict keys — so pre-rename checkpoints load unchanged.
ConditionalQueryModel = ConditionalQueryEncoderDecoderModel
