# `model/`

The EveryQuery model itself: the raw `nn.Module` architecture and the Lightning wrapper that
drives training / validation / prediction loops. Pure architecture concerns — no data-layer
shape, no Hydra entry points, no configs.

## What lives here

- **`model.py`** — `EveryQueryModel` (the ModernBERT-style encoder `nn.Module`) and
    `EveryQueryOutput` (the forward-pass output dataclass). The core architecture.
- **`lightning_module.py`** — `EveryQueryLightningModule`. Wraps `EveryQueryModel` for
    PyTorch Lightning with `training_step` / `validation_step` / `predict_step`. Shared between
    training and inference — the same LightningModule's `predict_step` is what `predict/` will
    use at inference time.
- **`conditional_model.py`** — `ConditionalQueryEncoderDecoderModel` (alias
    `ConditionalQueryModel`): the conditional query-sequence architecture with a bidirectional
    ModernBERT patient encoder, a cross-attending `nn.TransformerDecoder` over
    `[code, duration, answer]` query blocks and the custom `build_block_causal_mask`. Also home
    to the pieces both conditional architectures share (`ConditionalQueryOutput`, answer
    constants, `masked_bce`, `validate_rope_time_pair`).
- **`conditional_ar_model.py`** — `ConditionalQueryARModel`: the decoder-only conditional
    architecture. One Hugging Face `LlamaModel` (trained from scratch) jointly attends over
    `[patient events, c₁, d₁, a₁, …]` under a plain token-level causal mask; predictions are
    read from each block's duration token. See `docs/CONDITIONAL_QUERIES.md` §1 for how the two
    architectures' attention behaviors differ.
- **`conditional_lightning.py`** — `ConditionalQueryLightningModule`. One Lightning wrapper for
    both conditional architectures; checkpoints record which one they hold via the model's
    `architecture` hparam (absent = encoder–decoder, so pre-rename checkpoints load unchanged).
- **`ontology_embedding.py`** — `OntologyEmbedding` + `wrap_tok_embeddings`: ancestor-mixed
    code embeddings installed through `get_input_embeddings()`/`set_input_embeddings()`, shared
    by every architecture's patient, query-code and boundary-code lookups.

Call through the package so stage submodules don't need to know the file layout:

```python
from every_query.model import EveryQueryModel, EveryQueryLightningModule
```

Hydra `_target_` strings in configs use the fully-qualified module path
(`every_query.model.lightning_module.EveryQueryLightningModule`, etc.) for explicitness —
a config reader should see exactly which file the class lives in.

## Relationship to `data/`

The data-layer contract (dataset, batch, query types) lives in
[`every_query.data`](../data/). `model/` has no dependency on any stage submodule and no
dependency on the upstream `generate_tasks/` output layout — it only knows the shape of the
batch it receives, which is defined by `data/`.

This split mirrors MEICAR's `model/` (pure architecture) + MTD's dataset (shared dataset
plumbing). EQ has its own data layer because `EveryQueryBatch` carries query-specific fields
upstream MTD's batch doesn't.

## Pipeline position

```
data/   ─┐
         ├──►  model/  ─────►  predictions / loss
train/  ─┘          ▲
(or predict/)       │
                    Hydra-instantiated via train/configs/*.yaml
```

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 1 submodule restructure: [#79](https://github.com/payalchandak/EveryQuery/issues/79) (this PR)
