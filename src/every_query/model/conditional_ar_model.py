"""Decoder-only conditional query-sequence model: one Llama backbone over patient + queries.

Where :class:`~every_query.model.conditional_model.ConditionalQueryEncoderDecoderModel` splits
the work between a bidirectional ModernBERT patient encoder and a cross-attending
``nn.TransformerDecoder`` over query blocks, this model processes **one combined sequence**
with a single Hugging Face ``LlamaModel``:

.. code-block:: text

    [p₁, …, pₘ, c₁, d₁, a₁, c₂, d₂, a₂, …, c_L, d_L, a_L]

- ``p₁..pₘ`` are the patient's event tokens (right-padded rows are re-packed so the query
  stream starts **immediately after each row's last real event** — no padding gap).
- Each query block ``i`` is the same three tokens as the encoder-decoder model: the queried
  code ``c_i``, the duration/event-bound slot ``d_i``, and the teacher-forced binary answer
  ``a_i``.  The trailing ``a_L`` carries no supervision and conditions nothing (no later
  query exists), but keeping the uniform 3-token stride keeps every index computable in
  closed form.

**Attention is plain token-level causality** — no ported block mask.  The one behavioral
difference from ``build_block_causal_mask`` is that ``c_i`` can no longer peek *forward* at
``d_i``; the prediction is read from ``d_i``, which still sees ``c_i`` and itself, so the
complete current query is available at every prediction point.  The required invariant

.. code-block:: text

    d_i sees:        all patient events, every earlier (c, d, a) block, c_i, d_i
    d_i never sees:  a_i, anything later

falls straight out of the token ordering under Llama's internal causal mask; the model only
supplies a two-dimensional padding mask (1 = real token, 0 = padding).

The logit read from ``d_i`` therefore estimates
``P(a_i = 1 | patient, c₁, d₁, a₁, …, c_i, d_i)`` — earlier answers are caller-supplied
conditioning values (teacher-forced in training, user-chosen at inference), and all valid
query logits come out of one forward pass with no label leakage.  Sampling unknown answers
autoregressively and feeding predictions back in is deliberately out of scope here.

Everything about *what the tokens mean* is inherited from the encoder-decoder model: the
shared code-embedding table (patient events, query codes and event-bound boundary codes all
go through ``get_input_embeddings()``, so ontology mixing composes unchanged), the scalar
duration MLP, the boundary-code + learned-marker event-bound representation, the answer
embedding, binary answers with censoring-as-a-``TIMELINE//END``-query, and the
``(answer_logits, valid_mask)`` output contract.  New here are learned **token-type**
embeddings (patient / code / duration / answer) so the flat stream stays role-aware.

The four positional mechanisms have disjoint jobs:

- **Clinical-time RoPE** (``use_rope_time``): actual elapsed hours.  Every query token repeats
  the *final real patient event's* hour — that event is the prediction time and all queries are
  asked at it, so no clinical time passes across ``c_i, d_i, a_i`` or between blocks.
- **Block-position embedding**: query order, added identically to all three tokens of a block.
- **Token-type embedding**: the patient / code / duration / answer role.
- **Causal mask**: what each token may see — derived from physical token order, not from RoPE,
  so repeated rotary positions are safe.
"""

import logging
from typing import Any, ClassVar

import torch
from transformers import LlamaConfig, LlamaModel

from every_query.model.conditional_model import (
    ANSWER_YES,
    N_ANSWER_CLASSES,
    TOKEN_DURATION,
    TOKENS_PER_QUERY,
    ConditionalQueryOutput,
    _init_aux_embeddings,
    masked_bce,
    validate_rope_time_pair,
)
from every_query.model.model import MLP

logger = logging.getLogger(__name__)

# Token-type indices for the combined stream.  The three query-slot types deliberately sit at
# ``1 + TOKEN_*`` so the per-block layout constant (TOKENS_PER_QUERY, TOKEN_DURATION readout
# offset) stays shared with the encoder-decoder model.
TYPE_PATIENT = 0
TYPE_QUERY_CODE = 1
TYPE_QUERY_DURATION = 2
TYPE_QUERY_ANSWER = 3
N_TOKEN_TYPES = 4


class ConditionalQueryARModel(torch.nn.Module):
    """Decoder-only (autoregressive) conditional query-sequence model on a Llama backbone.

    Args:
        precision: Lightning precision string; sets initial weight dtype like the other models.
        mlp_dropout: Dropout used in the answer head MLP.
        config_overrides: Overrides applied to a fresh ``LlamaConfig`` (the backbone is always
            trained from scratch — no pretrained weights, no ``from_pretrained``).  Size the
            architecture here: ``hidden_size``, ``intermediate_size``, ``num_hidden_layers``,
            ``num_attention_heads``, ``num_key_value_heads``, ``max_position_embeddings``,
            ``attention_dropout``, ``vocab_size``, ``pad_token_id``.  ``use_cache`` is forced
            off — training never decodes incrementally.
        max_queries: Maximum query blocks per sequence.  Sizes the learned block-position
            table (query order), the per-position metrics in the Lightning module and the
            ``max_position_embeddings`` budget that ``train.py`` computes
            (``max_seq_len + 3 * max_queries``).
        use_rope_time: Drive rotary positions from ``batch.time_pos_ids`` (elapsed integer
            hours) instead of token index, exactly as the encoder-decoder model does for its
            encoder.  Every query token shares the last patient event's hour — the prediction
            time — since no clinical time passes while the queries are asked.
            Must be paired with ``ConditionalQueryPytorchDataset(strip_delta_tokens=True)``;
            a mismatch in either direction is a hard error (see
            :func:`~every_query.model.conditional_model.validate_rope_time_pair`).
        ontology_dir: Directory of ontology artifacts.  When set, the backbone's
            input-embedding module is replaced through ``set_input_embeddings`` so every code
            lookup — patient events, query codes and event-bound boundary codes alike —
            returns the ancestor-mixed average; see
            :mod:`every_query.model.ontology_embedding`.
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
        config_overrides: dict[str, Any] | None = None,
        max_queries: int = 8,
        use_rope_time: bool = False,
        ontology_dir: str | None = None,
    ):
        super().__init__()

        # Pass overrides through the constructor: ``head_dim`` is derived there from
        # ``hidden_size // num_attention_heads``, so setattr-after-the-fact would leave Llama's
        # default head_dim=128 (2x wider attention than intended).
        self.HF_model_config = LlamaConfig(**(config_overrides or {}))

        self.HF_model_config.use_cache = False
        self.HF_model_config.output_hidden_states = False
        self.HF_model_config.output_attentions = False

        extra_kwargs = {"torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}
        self.HF_model = LlamaModel._from_config(self.HF_model_config, **extra_kwargs)

        H = self.HF_model_config.hidden_size

        self.duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        self.answer_embed = torch.nn.Embedding(N_ANSWER_CLASSES, H)
        self.token_type_embed = torch.nn.Embedding(N_TOKEN_TYPES, H)
        # Query *order* only.  RoPE carries clinical time (all query tokens share the
        # prediction-time hour), so ordering needs its own learned table.  Patient tokens get none.
        self.block_pos_embed = torch.nn.Embedding(max_queries, H)
        # Same role as the encoder-decoder model's marker: distinguishes "window ends at the
        # next X" from "is X observed" while both read the shared code-embedding table.  Always
        # allocated so the parameter set does not depend on the data.
        self.bound_marker = torch.nn.Parameter(torch.randn(H) * self.HF_model_config.initializer_range)
        self.answer_mlp = MLP(layers=[H, 128, 1], dropout_prob=mlp_dropout)

        # These tables live outside the backbone, so LlamaModel.post_init() never touched them
        # and they kept nn.Embedding's default N(0, 1) -- ~50x the code embeddings' std, which
        # lets the shared type/position vectors swamp code identity in the summed input.
        # Re-init here, not via self.apply(), which would reset the already-initialized backbone.
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
            # Must run after HF_model exists and before any lookup; substituting the module is
            # what lets patient, query and boundary codes all inherit ontology structure.
            from every_query.data.ontology import load_mix_matrix
            from every_query.model.ontology_embedding import wrap_tok_embeddings

            wrap_tok_embeddings(self, load_mix_matrix(ontology_dir))
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.hparams = {
            # Restore-time dispatch key for ConditionalQueryLightningModule.load_from_checkpoint;
            # popped before the constructor sees the dict.  Checkpoints without it (all
            # encoder-decoder checkpoints, old and new) default to that architecture.
            "architecture": "autoregressive",
            "precision": precision,
            "mlp_dropout": mlp_dropout,
            "config_overrides": dict(config_overrides) if config_overrides else None,
            "max_queries": max_queries,
            "use_rope_time": use_rope_time,
            "ontology_dir": ontology_dir,
        }

    @property
    def max_seq_len(self) -> int:
        """Positions budget for the *combined* stream: patient tokens + 3 per query block."""
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        return self.HF_model_config.vocab_size

    def _query_code_embeds(self, batch) -> torch.Tensor:
        """``(B, L, H)`` content of each query block's **code** slot (shared code table)."""
        return self.HF_model.get_input_embeddings()(batch.q_codes)

    def _query_duration_embeds(self, batch) -> torch.Tensor:
        """``(B, L, H)`` content of each query block's **duration** slot.

        Identical semantics to the encoder-decoder model's slot: scalar horizons go through
        the duration MLP; an event-bounded query instead carries its boundary code's token
        embedding plus the learned ``bound_marker``.  With no bounds present this is exactly
        the scalar path.
        """
        dur_emb = self.duration_embed((batch.q_durations / 365.0).unsqueeze(-1))

        q_bounds = getattr(batch, "q_bound_codes", None)
        if q_bounds is None:
            return dur_emb

        bound_emb = self.HF_model.get_input_embeddings()(q_bounds).to(dur_emb.dtype)
        bound_emb = bound_emb + self.bound_marker.to(dur_emb.dtype)
        return torch.where((q_bounds > 0).unsqueeze(-1), bound_emb, dur_emb)

    def _query_tokens(self, batch) -> torch.Tensor:
        """Interleaved ``(B, 3L, H)`` query-stream embeddings ``[c₁, d₁, a₁, c₂, …]``.

        Each slot carries its content embedding + token-type embedding + the block-position
        embedding of its query block (the same vector for ``c_i``, ``d_i`` and ``a_i``), which
        is what encodes query order — clinical-time RoPE cannot, since every query token sits
        at the prediction time.
        """
        B, L = batch.q_codes.shape
        tt = self.token_type_embed.weight

        code_emb = self._query_code_embeds(batch)
        dur_emb = self._query_duration_embeds(batch).to(code_emb.dtype)
        ans_emb = self.answer_embed(batch.q_answers).to(code_emb.dtype)

        tokens = torch.stack(
            [
                code_emb + tt[TYPE_QUERY_CODE].to(code_emb.dtype),
                dur_emb + tt[TYPE_QUERY_DURATION].to(code_emb.dtype),
                ans_emb + tt[TYPE_QUERY_ANSWER].to(code_emb.dtype),
            ],
            dim=2,
        )  # (B, L, 3, H)
        block_idx = torch.arange(L, device=tokens.device).clamp(max=self.max_queries - 1)
        tokens = tokens + self.block_pos_embed(block_idx).view(1, L, 1, -1).to(tokens.dtype)
        return tokens.reshape(B, L * TOKENS_PER_QUERY, -1)

    def _position_ids(
        self,
        batch,
        n_patient: torch.Tensor,
        query_positions: torch.Tensor,
        total_len: int,
    ) -> torch.Tensor | None:
        """``(B, T)`` rotary position ids, or ``None`` for Llama's default token-index arange.

        With ``use_rope_time`` the patient part carries the dataset's elapsed-hour
        ``time_pos_ids`` and **every** query token repeats the last real event's hour: the
        final event *is* the prediction time and all queries are asked at that instant, so no
        clinical time passes across ``c_i, d_i, a_i`` or between blocks.  Repeated positions
        are harmless — causality comes from the physical token order under Llama's causal
        mask, and query order is carried by ``block_pos_embed``.  Padding keeps position 0; it
        is attention-masked, so its rotary angle never matters.

        Without rope time the combined stream is left-packed (patient prefix, then queries),
        so Llama's own ``arange`` positions are exactly right and nothing is passed.
        """
        time_pos = validate_rope_time_pair(self.use_rope_time, getattr(batch, "time_pos_ids", None))
        if time_pos is None:
            return None

        device = batch.code.device
        time_pos = time_pos.to(device)
        B, S = time_pos.shape

        last_idx = (n_patient - 1).clamp(min=0)
        last_hour = time_pos.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        last_hour = torch.where(n_patient > 0, last_hour, torch.zeros_like(last_hour))

        n_query_tokens = query_positions.shape[1]
        query_hours = last_hour.unsqueeze(1).expand(-1, n_query_tokens)

        position_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
        position_ids[:, :S] = time_pos
        position_ids.scatter_(1, query_positions, query_hours)
        return position_ids

    def forward(self, batch) -> tuple[torch.FloatTensor, ConditionalQueryOutput]:
        """One causal pass over ``[patient, queries]``; return ``(loss, outputs)``.

        Expects a ``ConditionalQueryBatch`` with ``code`` (patient tokens), ``q_codes``,
        ``q_durations``, ``q_answers`` and ``q_mask`` (plus optional ``q_bound_codes`` /
        ``time_pos_ids``).  Output contract matches the encoder-decoder model:
        ``answer_logits`` and ``valid_mask`` are both ``(batch, n_queries)``.
        """
        B, L = batch.q_codes.shape
        S = batch.code.shape[1]
        device = batch.code.device
        pad = batch.PAD_INDEX

        patient_mask = batch.code != pad  # (B, S); MEDS batches are right-padded
        n_patient = patient_mask.sum(dim=1)  # (B,)

        n_query_tokens = TOKENS_PER_QUERY * L
        total_len = S + n_query_tokens
        if total_len > self.max_seq_len:
            raise ValueError(
                f"Combined sequence needs {total_len} positions ({S} patient + {n_query_tokens} query "
                f"tokens) but max_position_embeddings={self.max_seq_len}.  The budget must "
                f"cover max_patient_tokens + {TOKENS_PER_QUERY} * max_queries; train.py sizes "
                f"it that way from the datamodule config and `max_queries`."
            )

        tt = self.token_type_embed.weight
        patient_emb = self.HF_model.get_input_embeddings()(batch.code)
        patient_emb = patient_emb + tt[TYPE_PATIENT].to(patient_emb.dtype)
        # Zero the padding rows so the scatter below writes onto a clean slate and untouched
        # padding carries no stale content (it is attention-masked regardless).
        patient_emb = patient_emb * patient_mask.unsqueeze(-1).to(patient_emb.dtype)

        query_tokens = self._query_tokens(batch).to(patient_emb.dtype)  # (B, 3L, H)
        H = patient_emb.shape[-1]

        # Row i's query stream starts at its own n_i — immediately after the last real event,
        # never after the batch-wide padded width.
        query_positions = n_patient.unsqueeze(1) + torch.arange(n_query_tokens, device=device).unsqueeze(0)

        inputs_embeds = torch.zeros(B, total_len, H, dtype=patient_emb.dtype, device=device)
        inputs_embeds[:, :S] = patient_emb
        inputs_embeds.scatter_(1, query_positions.unsqueeze(-1).expand(-1, -1, H), query_tokens)

        # 2D padding mask (1 = real, 0 = pad); Llama builds the causal component internally.
        attention_mask = torch.zeros(B, total_len, dtype=torch.long, device=device)
        attention_mask[:, :S] = patient_mask.long()
        query_attn = batch.q_mask.repeat_interleave(TOKENS_PER_QUERY, dim=1).long()  # (B, 3L)
        attention_mask.scatter_(1, query_positions, query_attn)

        position_ids = self._position_ids(batch, n_patient, query_positions, total_len)

        hidden = self.HF_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state  # (B, T, H)

        # The answer for block i is read from d_i (offset 3i + 1 into the query stream) —
        # causally conditioned on the patient, all earlier blocks incl. teacher-forced
        # answers, and (c_i, d_i) itself; never on a_i or anything later.
        d_positions = (
            n_patient.unsqueeze(1)
            + TOKENS_PER_QUERY * torch.arange(L, device=device).unsqueeze(0)
            + TOKEN_DURATION
        )  # (B, L)
        answer_hidden = hidden.gather(1, d_positions.unsqueeze(-1).expand(-1, -1, H))
        answer_logits = self.answer_mlp(answer_hidden).squeeze(-1)  # (B, L)

        targets = (batch.q_answers == ANSWER_YES).float()
        valid = batch.q_mask
        loss = masked_bce(self.criterion, answer_logits, targets, valid)

        outputs = ConditionalQueryOutput(
            last_hidden_state=None,
            answer_logits=answer_logits,
            valid_mask=valid,
        )
        return loss, outputs
