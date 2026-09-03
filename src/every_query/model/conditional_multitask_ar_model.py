"""Decoder-only all-vocabulary model for ordered multitask windows.

The combined causal stream is::

    [patient events, W0, C0, A0, ..., W(K-2), C(K-2), A(K-2), W(K-1)]

``W_i`` describes the start and end of window ``i``.  ``C_i`` names a code from
that window and ``A_i`` supplies its teacher-forced answer, so the answer can
condition later windows without leaking into the prediction for its own window.
Each ``W_i`` hidden state is projected onto the backbone's input-embedding table,
giving one logit for every vocabulary code without a separate output matrix.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import LlamaConfig, LlamaModel
from transformers.modeling_outputs import BaseModelOutput

from every_query.model.conditional_model import (
    N_ANSWER_CLASSES,
    _init_aux_embeddings,
    validate_rope_time_pair,
)
from every_query.model.model import MLP

TOKENS_PER_WINDOW = 3

TYPE_PATIENT = 0
TYPE_WINDOW = 1
TYPE_CONDITION_CODE = 2
TYPE_CONDITION_ANSWER = 3
N_TOKEN_TYPES = 4


@dataclass
class ConditionalMultitaskOutput(BaseModelOutput):
    """All-vocabulary predictions and the elements that participate in loss.

    Attributes:
        logits: ``(B, K, V)`` float32 logits.
        valid_mask: ``(B, K, V)`` boolean mask.  It combines ``q_mask`` with
            exclusion of the PAD vocabulary row.
    """

    logits: torch.FloatTensor | None = None
    valid_mask: torch.BoolTensor | None = None

    @property
    def probs(self) -> torch.Tensor | None:
        """Float32 sigmoid probabilities with the same shape as ``logits``."""
        if self.logits is None:
            return None
        return torch.sigmoid(self.logits.float())


class ConditionalMultitaskARModel(torch.nn.Module):
    """One Llama backbone over a patient prefix and ``3K-2`` query tokens.

    Args:
        precision: Lightning precision string used to choose the initial backbone dtype.
        config_overrides: Keyword overrides for a fresh :class:`LlamaConfig`.
        max_windows: Maximum supported ``K``; sizes the learned block positions.
        use_rope_time: If true, consume ``batch.time_pos_ids`` as elapsed-hour RoPE positions.
        ontology_dir: Reserved for future ontology support.  Any non-null value is rejected.
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
        config_overrides: dict[str, Any] | None = None,
        max_windows: int = 5,
        use_rope_time: bool = False,
        ontology_dir: str | None = None,
    ):
        super().__init__()
        if ontology_dir is not None:
            raise NotImplementedError("ConditionalMultitaskARModel does not yet support ontology_dir")
        if max_windows < 1:
            raise ValueError(f"max_windows must be at least 1, got {max_windows}")

        self.HF_model_config = LlamaConfig(**(config_overrides or {}))
        self.HF_model_config.use_cache = False
        self.HF_model_config.output_hidden_states = False
        self.HF_model_config.output_attentions = False
        extra_kwargs = {"torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}
        self.HF_model = LlamaModel._from_config(self.HF_model_config, **extra_kwargs)

        H = self.HF_model_config.hidden_size
        self.start_duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        self.end_duration_embed = MLP(layers=[1, 64, H], dropout_prob=0)
        self.start_marker = torch.nn.Parameter(torch.randn(H) * self.HF_model_config.initializer_range)
        self.bound_marker = torch.nn.Parameter(torch.randn(H) * self.HF_model_config.initializer_range)
        self.answer_embed = torch.nn.Embedding(N_ANSWER_CLASSES, H)
        self.token_type_embed = torch.nn.Embedding(N_TOKEN_TYPES, H)
        self.block_pos_embed = torch.nn.Embedding(max_windows, H)
        # The optimizer's existing ``bias`` rule puts this parameter in the no-decay group.
        self.code_bias = torch.nn.Parameter(torch.full((self.HF_model_config.vocab_size,), -3.0))

        _init_aux_embeddings(
            self.HF_model_config.initializer_range,
            self.answer_embed,
            self.token_type_embed,
            self.block_pos_embed,
        )

        self.max_windows = max_windows
        self.use_rope_time = use_rope_time
        self.ontology_dir = ontology_dir
        self.hparams = {
            "architecture": "conditional_multitask_ar",
            "precision": precision,
            "config_overrides": dict(config_overrides) if config_overrides else None,
            "max_windows": max_windows,
            "use_rope_time": use_rope_time,
            "ontology_dir": ontology_dir,
        }

    @property
    def max_seq_len(self) -> int:
        """Maximum length of the combined patient and query stream."""
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        return self.HF_model_config.vocab_size

    def _start_fields(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return explicit start tensors, filling the paired legacy absence with zeros."""
        durations = getattr(batch, "q_start_durations", None)
        codes = getattr(batch, "q_start_codes", None)
        if (durations is None) != (codes is None):
            missing = "q_start_durations" if durations is None else "q_start_codes"
            raise ValueError(
                f"q_start_durations and q_start_codes must be given together (got {missing}=None)"
            )
        if durations is None:
            durations = torch.zeros_like(batch.q_durations)
            codes = torch.zeros_like(batch.q_bound_codes)
        return durations, codes

    def _window_embeds(self, batch) -> torch.Tensor:
        """Build the ``(B, K, H)`` window tokens from role-distinct start/end specs."""
        start_durations, start_codes = self._start_fields(batch)
        code_embeddings = self.HF_model.get_input_embeddings()

        start_duration = self.start_duration_embed((start_durations / 365.0).unsqueeze(-1))
        start_event = code_embeddings(start_codes).to(start_duration.dtype)
        start_event = start_event + self.start_marker.to(start_duration.dtype)
        start_spec = torch.where((start_codes > 0).unsqueeze(-1), start_event, start_duration)

        end_duration = self.end_duration_embed((batch.q_durations / 365.0).unsqueeze(-1))
        end_event = code_embeddings(batch.q_bound_codes).to(end_duration.dtype)
        end_event = end_event + self.bound_marker.to(end_duration.dtype)
        end_spec = torch.where((batch.q_bound_codes > 0).unsqueeze(-1), end_event, end_duration)

        n_windows = batch.q_durations.shape[1]
        block_idx = torch.arange(n_windows, device=batch.q_durations.device)
        window_type = self.token_type_embed.weight[TYPE_WINDOW].to(start_spec.dtype)
        block_pos = self.block_pos_embed(block_idx).to(start_spec.dtype)
        return start_spec + end_spec + window_type + block_pos.unsqueeze(0)

    def _query_tokens(self, batch) -> torch.Tensor:
        """Return the exact ``[W0,C0,A0,...,W(K-1)]`` token stream."""
        B, n_windows = batch.q_durations.shape
        H = self.HF_model_config.hidden_size
        windows = self._window_embeds(batch)
        stream = torch.empty(
            B,
            TOKENS_PER_WINDOW * n_windows - 2,
            H,
            dtype=windows.dtype,
            device=windows.device,
        )
        stream[:, 0::TOKENS_PER_WINDOW] = windows

        if n_windows > 1:
            code_embeddings = self.HF_model.get_input_embeddings()
            condition_codes = code_embeddings(batch.condition_codes).to(windows.dtype)
            condition_answers = self.answer_embed(batch.condition_answers.long()).to(windows.dtype)
            tt = self.token_type_embed.weight
            block_idx = torch.arange(n_windows - 1, device=windows.device)
            block_pos = self.block_pos_embed(block_idx).to(windows.dtype).unsqueeze(0)
            stream[:, 1::TOKENS_PER_WINDOW] = (
                condition_codes + tt[TYPE_CONDITION_CODE].to(windows.dtype) + block_pos
            )
            stream[:, 2::TOKENS_PER_WINDOW] = (
                condition_answers + tt[TYPE_CONDITION_ANSWER].to(windows.dtype) + block_pos
            )
        return stream

    def _position_ids(
        self,
        batch,
        n_patient: torch.Tensor,
        query_positions: torch.Tensor,
        total_len: int,
    ) -> torch.Tensor | None:
        """Construct clinical-time positions; query starts never advance clinical time."""
        time_pos = validate_rope_time_pair(self.use_rope_time, getattr(batch, "time_pos_ids", None))
        if time_pos is None:
            return None

        device = batch.code.device
        time_pos = time_pos.to(device)
        B, S = time_pos.shape
        last_idx = (n_patient - 1).clamp(min=0)
        last_hour = time_pos.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        last_hour = torch.where(n_patient > 0, last_hour, torch.zeros_like(last_hour))

        position_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
        position_ids[:, :S] = time_pos
        query_hours = last_hour.unsqueeze(1).expand(-1, query_positions.shape[1])
        position_ids.scatter_(1, query_positions, query_hours)
        return position_ids

    def forward(self, batch) -> tuple[torch.FloatTensor, ConditionalMultitaskOutput]:
        """Run one causal pass and return masked BCE loss plus all-vocabulary logits."""
        B, n_windows = batch.q_durations.shape
        if n_windows < 1:
            raise ValueError("ConditionalMultitaskARModel requires at least one window")
        if self.max_windows < n_windows:
            raise ValueError(f"Batch has K={n_windows} windows but max_windows={self.max_windows}")
        # Validate the paired legacy rule even before token construction.
        self._start_fields(batch)

        S = batch.code.shape[1]
        n_query_tokens = TOKENS_PER_WINDOW * n_windows - 2
        total_len = S + n_query_tokens
        if total_len > self.max_seq_len:
            raise ValueError(
                f"Combined sequence needs {total_len} positions ({S} patient + "
                f"{n_query_tokens} query tokens) but max_position_embeddings={self.max_seq_len}. "
                "The configured budget must cover max_seq_len + 3 * max_windows."
            )

        device = batch.code.device
        pad = batch.PAD_INDEX
        patient_mask = batch.code != pad
        n_patient = patient_mask.sum(dim=1)

        patient_emb = self.HF_model.get_input_embeddings()(batch.code)
        patient_emb = patient_emb + self.token_type_embed.weight[TYPE_PATIENT].to(patient_emb.dtype)
        patient_emb = patient_emb * patient_mask.unsqueeze(-1).to(patient_emb.dtype)
        query_tokens = self._query_tokens(batch).to(patient_emb.dtype)
        H = patient_emb.shape[-1]

        query_positions = n_patient.unsqueeze(1) + torch.arange(n_query_tokens, device=device).unsqueeze(0)
        inputs_embeds = torch.zeros(B, total_len, H, dtype=patient_emb.dtype, device=device)
        inputs_embeds[:, :S] = patient_emb
        inputs_embeds.scatter_(1, query_positions.unsqueeze(-1).expand(-1, -1, H), query_tokens)

        attention_mask = torch.zeros(B, total_len, dtype=torch.long, device=device)
        attention_mask[:, :S] = patient_mask.long()
        query_attn = batch.q_mask.repeat_interleave(TOKENS_PER_WINDOW, dim=1)[:, :n_query_tokens]
        attention_mask.scatter_(1, query_positions, query_attn.long())

        position_ids = self._position_ids(batch, n_patient, query_positions, total_len)
        hidden = self.HF_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        window_positions = n_patient.unsqueeze(1) + TOKENS_PER_WINDOW * torch.arange(
            n_windows, device=device
        ).unsqueeze(0)
        window_hidden = hidden.gather(1, window_positions.unsqueeze(-1).expand(-1, -1, H))

        embedding_weight = self.HF_model.get_input_embeddings().weight
        if embedding_weight.shape[0] != batch.targets.shape[-1]:
            raise ValueError(
                f"Target vocabulary width V={batch.targets.shape[-1]} does not match the tied "
                f"embedding table width V={embedding_weight.shape[0]}"
            )
        # Explicitly leave autocast: `.float()` alone is still downcast by bf16 autocast.
        with torch.autocast(device_type=window_hidden.device.type, enabled=False):
            logits = window_hidden.float() @ embedding_weight.float().T
            logits = logits + self.code_bias.float()

        vocab_not_pad = torch.arange(logits.shape[-1], device=device) != pad
        valid_mask = batch.q_mask.unsqueeze(-1) & vocab_not_pad.view(1, 1, -1)
        valid_mask = valid_mask.expand(B, n_windows, -1)
        per_element = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch.targets.float(), reduction="none"
        )
        loss = (per_element * valid_mask).sum() / valid_mask.sum().clamp_min(1)

        return loss, ConditionalMultitaskOutput(
            last_hidden_state=None,
            logits=logits,
            valid_mask=valid_mask,
        )
