"""Conditional query-sequence model: bidirectional patient encoder + block-autoregressive query decoder.

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

**Censoring** is handled by convention rather than a separate head: the first query of every
sequence is the special *censor query* ``(__CENSOR__, duration)`` asking "will any data be present
after prediction_time + duration?".  Its answer is always observed.  Subsequent query answers may
be unobserved (censored); these contribute no loss, and their teacher-forced input token uses a
dedicated ``CENSORED`` answer-vocabulary entry so later queries know the prior answer was unknown.
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import AutoConfig, ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

from every_query.model.model import MLP

logger = logging.getLogger(__name__)

# Answer-token vocabulary for teacher forcing.
ANSWER_NO = 0
ANSWER_YES = 1
ANSWER_CENSORED = 2
N_ANSWER_CLASSES = 3

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


@dataclass
class ConditionalQueryOutput(BaseModelOutput):
    """Output container for :class:`ConditionalQueryModel`.

    Attributes:
        answer_logits: ``(batch, n_queries)`` logits, one per query block.  Position 0 is the
            censor query; positions ``>= 1`` are occurrence queries.
        censor_loss: BCE loss over position-0 answers (always observed).
        occurs_loss: BCE loss over positions ``>= 1``, masked to observed (non-censored,
            non-padding) answers.
        valid_mask: ``(batch, n_queries)`` bool — True where the answer is observed (loss applied).
    """

    answer_logits: torch.FloatTensor | None = None
    censor_loss: torch.FloatTensor | None = None
    occurs_loss: torch.FloatTensor | None = None
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
        """
        if self.answer_logits is None:
            return None
        return torch.sigmoid(self.answer_logits)


class ConditionalQueryModel(torch.nn.Module):
    """Encoder/decoder model for conditional query sequences.

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
        occurs_loss_weight: Total loss is ``w * occurs_loss + (1 - w) * censor_loss``.
        censor_query_index: Vocab index reserved for the censor-query code token.  Defaults to
            ``vocab_size - 1`` (the dataset reserves the top index).  Must be < vocab_size.
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
        occurs_loss_weight: float = 0.5,
        censor_query_index: int | None = None,
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

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=H,
            nhead=n_heads,
            dim_feedforward=decoder_ffn_mult * H,
            dropout=mlp_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layers, norm=torch.nn.LayerNorm(H)
        )

        self.duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        self.answer_embed = torch.nn.Embedding(N_ANSWER_CLASSES, H)
        self.token_type_embed = torch.nn.Embedding(TOKENS_PER_QUERY, H)
        self.block_pos_embed = torch.nn.Embedding(max_queries, H)
        self.answer_mlp = MLP(layers=[H, 128, 1], dropout_prob=mlp_dropout)

        self.max_queries = max_queries
        self.occurs_loss_weight = occurs_loss_weight
        self.censor_query_index = (
            censor_query_index if censor_query_index is not None else self.HF_model_config.vocab_size - 1
        )
        if self.censor_query_index >= self.HF_model_config.vocab_size:
            raise ValueError(
                f"censor_query_index ({self.censor_query_index}) must be < vocab_size "
                f"({self.HF_model_config.vocab_size})."
            )
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
            "occurs_loss_weight": occurs_loss_weight,
            "censor_query_index": self.censor_query_index,
        }

    @property
    def max_seq_len(self) -> int:
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        return self.HF_model_config.vocab_size

    def _masked_bce(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """BCE-with-logits over ``mask``-selected positions; differentiable zero when empty."""
        logits, target = logits[mask], target[mask].float()
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return self.criterion(logits, target)

    def _decoder_tokens(self, batch) -> torch.Tensor:
        """Build the interleaved ``(B, 3L, H)`` decoder input from a ConditionalQueryBatch."""
        B, L = batch.q_codes.shape

        code_emb = self.HF_model.embeddings.tok_embeddings(batch.q_codes)
        dur_emb = self.duration_embed((batch.q_durations / 365.0).unsqueeze(-1))
        ans_emb = self.answer_embed(batch.q_answers)

        tokens = torch.stack([code_emb, dur_emb, ans_emb], dim=2)  # (B, L, 3, H)

        device = tokens.device
        block_idx = torch.arange(L, device=device).clamp(max=self.max_queries - 1)
        tokens = tokens + self.block_pos_embed(block_idx).view(1, L, 1, -1)
        tokens = tokens + self.token_type_embed(
            torch.arange(TOKENS_PER_QUERY, device=device)
        ).view(1, 1, TOKENS_PER_QUERY, -1)

        return tokens.reshape(B, L * TOKENS_PER_QUERY, -1)

    def forward(self, batch) -> tuple[torch.FloatTensor, ConditionalQueryOutput]:
        """Run encoder + block-autoregressive decoder; return ``(loss, outputs)``.

        Expects a ``ConditionalQueryBatch`` with ``code`` (patient tokens), ``q_codes``,
        ``q_durations``, ``q_answers`` and ``q_mask``.
        """
        B, L = batch.q_codes.shape

        enc_attention_mask = batch.code != batch.PAD_INDEX
        memory = self.HF_model(input_ids=batch.code, attention_mask=enc_attention_mask).last_hidden_state

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
        observed = batch.q_mask & (batch.q_answers != ANSWER_CENSORED)

        censor_mask = torch.zeros_like(observed)
        censor_mask[:, 0] = observed[:, 0]
        occurs_mask = observed.clone()
        occurs_mask[:, 0] = False

        censor_loss = self._masked_bce(answer_logits, targets, censor_mask)
        occurs_loss = self._masked_bce(answer_logits, targets, occurs_mask)
        loss = self.occurs_loss_weight * occurs_loss + (1 - self.occurs_loss_weight) * censor_loss

        outputs = ConditionalQueryOutput(
            last_hidden_state=None,
            answer_logits=answer_logits,
            censor_loss=censor_loss,
            occurs_loss=occurs_loss,
            valid_mask=observed,
        )
        return loss, outputs
