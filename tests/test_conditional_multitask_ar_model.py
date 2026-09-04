"""Contract tests for the decoder-only all-vocabulary multitask model."""

from functools import partial
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.model.conditional_multitask_ar_model import (
    TYPE_CONDITION_ANSWER,
    TYPE_CONDITION_CODE,
    TYPE_WINDOW,
    ConditionalMultitaskARModel,
)


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    yield


VOCAB = 31
MODEL_CFG = {
    "hidden_size": 24,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 48,
    "max_position_embeddings": 64,
    "vocab_size": VOCAB,
    "pad_token_id": 0,
    "attention_dropout": 0.0,
}


def tiny_model(**kwargs) -> ConditionalMultitaskARModel:
    torch.manual_seed(7)
    overrides = dict(MODEL_CFG, **kwargs.pop("config_overrides", {}))
    model = ConditionalMultitaskARModel(config_overrides=overrides, max_windows=5, **kwargs)
    model.eval()
    return model


def make_batch(
    *,
    n_windows: int = 3,
    code: list[list[int]] | None = None,
    q_mask: list[list[bool]] | None = None,
    starts: bool = True,
    time_pos_ids: torch.Tensor | None = None,
) -> MultitaskBoundaryBatch:
    code = code or [[2, 3, 4, 5], [7, 8, 0, 0]]
    B, S = len(code), len(code[0])
    start_durations = torch.tensor([[0.0, 2.0, -1.0, 4.0, 0.0][:n_windows]] * B)
    start_codes = torch.tensor([[0, 0, 9, 0, 0][:n_windows]] * B)
    durations = torch.tensor([[7.0, -1.0, 30.0, 4.0, 2.0][:n_windows]] * B)
    bounds = torch.tensor([[0, 10, 0, 0, 0][:n_windows]] * B)
    mask = torch.tensor(q_mask or [[True] * n_windows] * B)
    targets = torch.zeros(B, n_windows, VOCAB, dtype=torch.bool)
    if n_windows:
        targets[:, :, 2] = True
        targets[:, 0, 0] = True  # PAD is intentionally true to prove loss masking.
    conditions = torch.tensor([[11, 12, 13, 14][: n_windows - 1]] * B, dtype=torch.long)
    answers = torch.tensor([[True, False, True, False][: n_windows - 1]] * B, dtype=torch.bool)
    kwargs = {}
    if starts:
        kwargs.update(q_start_durations=start_durations, q_start_codes=start_codes)
    return MultitaskBoundaryBatch(
        code=torch.tensor(code),
        numeric_value=torch.zeros(B, S),
        numeric_value_mask=torch.zeros(B, S, dtype=torch.bool),
        time_delta_days=torch.zeros(B, S),
        q_durations=durations,
        q_bound_codes=bounds,
        q_mask=mask,
        targets=targets,
        condition_codes=conditions,
        condition_answers=answers,
        time_pos_ids=time_pos_ids,
        **kwargs,
    )


def test_forward_shape_dtype_mask_and_bias():
    model = tiny_model()
    batch = make_batch(q_mask=[[True, False, True]] * 2)
    # The tied projection must explicitly escape autocast; `.float()` operands alone are
    # downcast again by PyTorch's matmul autocast policy.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, out = model(batch)
    assert loss.isfinite()
    assert out.logits.shape == out.valid_mask.shape == (2, 3, VOCAB)
    assert out.logits.dtype == torch.float32 and out.probs.dtype == torch.float32
    assert torch.equal(model.code_bias, torch.full((VOCAB,), -3.0))
    assert not out.valid_mask[:, :, 0].any()
    assert not out.valid_mask[:, 1].any()
    assert out.valid_mask[:, (0, 2), 1:].all()


def test_exact_stream_order_and_token_count():
    model = tiny_model()
    batch = make_batch()
    tokens = model._query_tokens(batch)
    windows = model._window_embeds(batch)
    assert tokens.shape == (2, 3 * 3 - 2, MODEL_CFG["hidden_size"])
    torch.testing.assert_close(tokens[:, 0::3], windows)

    embedding = model.HF_model.get_input_embeddings()
    bp = model.block_pos_embed(torch.arange(2)).unsqueeze(0)
    tt = model.token_type_embed.weight
    expected_c = embedding(batch.condition_codes) + tt[TYPE_CONDITION_CODE] + bp
    expected_a = model.answer_embed(batch.condition_answers.long()) + tt[TYPE_CONDITION_ANSWER] + bp
    torch.testing.assert_close(tokens[:, 1::3], expected_c)
    torch.testing.assert_close(tokens[:, 2::3], expected_a)


def test_window_specs_use_matching_rows_and_distinct_roles():
    model = tiny_model()
    batch = make_batch()
    got = model._window_embeds(batch)
    starts, start_codes = model._start_fields(batch)
    embedding = model.HF_model.get_input_embeddings()
    start_d = model.start_duration_embed((starts / 365).unsqueeze(-1))
    start_e = embedding(start_codes) + model.start_marker
    end_d = model.end_duration_embed((batch.q_durations / 365).unsqueeze(-1))
    end_e = embedding(batch.q_bound_codes) + model.bound_marker
    expected = (
        torch.where((start_codes > 0).unsqueeze(-1), start_e, start_d)
        + torch.where((batch.q_bound_codes > 0).unsqueeze(-1), end_e, end_d)
        + model.token_type_embed.weight[TYPE_WINDOW]
        + model.block_pos_embed(torch.arange(3)).unsqueeze(0)
    )
    torch.testing.assert_close(got, expected)
    assert model.start_duration_embed is not model.end_duration_embed
    assert model.start_marker is not model.bound_marker


@pytest.mark.parametrize(
    ("start_duration", "start_code", "duration", "bound_code"),
    [
        (0.0, 0, 7.0, 0),
        (0.0, 0, -1.0, 10),
        (2.0, 0, 7.0, 0),
        (2.0, 0, -1.0, 10),
        (-1.0, 9, 7.0, 0),
        (-1.0, 9, -1.0, 10),
    ],
)
def test_all_six_start_end_combinations_are_finite(start_duration, start_code, duration, bound_code):
    model = tiny_model()
    batch = make_batch(n_windows=1)
    batch.q_start_durations.fill_(start_duration)
    batch.q_start_codes.fill_(start_code)
    batch.q_durations.fill_(duration)
    batch.q_bound_codes.fill_(bound_code)
    loss, out = model(batch)
    assert loss.isfinite() and out.logits.isfinite().all()


def test_current_answer_and_future_conditions_cannot_change_earlier_windows():
    model = tiny_model()
    base = make_batch()
    changed_answer = make_batch()
    changed_answer.condition_answers[:, 1] = ~changed_answer.condition_answers[:, 1]
    changed_future = make_batch()
    changed_future.condition_codes[:, 1] = 20
    _, a = model(base)
    _, b = model(changed_answer)
    _, c = model(changed_future)
    # A1 and C1 physically follow W1, so neither can change W0 or W1.
    torch.testing.assert_close(a.logits[:, :2], b.logits[:, :2])
    torch.testing.assert_close(a.logits[:, :2], c.logits[:, :2])
    assert not torch.equal(a.logits[:, 2], b.logits[:, 2])
    assert not torch.equal(a.logits[:, 2], c.logits[:, 2])


def test_earlier_answer_can_influence_later_window():
    model = tiny_model()
    a = make_batch()
    b = make_batch()
    b.condition_answers[:, 0] = ~b.condition_answers[:, 0]
    _, out_a = model(a)
    _, out_b = model(b)
    torch.testing.assert_close(out_a.logits[:, 0], out_b.logits[:, 0])
    assert not torch.equal(out_a.logits[:, 1:], out_b.logits[:, 1:])


def test_patient_padding_is_invisible_and_query_mask_covers_whole_block():
    model = tiny_model()
    narrow = make_batch(code=[[2, 3, 4, 5], [7, 8, 0, 0]], q_mask=[[True, False, True]] * 2)
    wide = make_batch(code=[[2, 3, 4, 5, 0, 0], [7, 8, 0, 0, 0, 0]], q_mask=[[True, False, True]] * 2)
    captured = {}

    def hook(_module, _args, kwargs):
        captured["mask"] = kwargs["attention_mask"].detach().clone()

    handle = model.HF_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        _, out_wide = model(wide)
    finally:
        handle.remove()
    _, out_narrow = model(narrow)
    torch.testing.assert_close(out_narrow.logits[:, (0, 2)], out_wide.logits[:, (0, 2)])
    n_patient = (wide.code != 0).sum(1)
    for row, n in enumerate(n_patient.tolist()):
        assert captured["mask"][row, n : n + 7].tolist() == [1, 1, 1, 0, 0, 0, 1]


def test_left_or_interior_padding_is_rejected():
    """The stream layout assumes right padding; anything else must raise, not corrupt silently."""
    model = tiny_model()
    right = make_batch(code=[[2, 3, 4, 5], [7, 8, 0, 0]])
    loss, _ = model(right)
    assert loss.isfinite()
    left = make_batch(code=[[2, 3, 4, 5], [0, 0, 7, 8]])
    with pytest.raises(ValueError, match="prefix mask"):
        model(left)
    interior = make_batch(code=[[2, 3, 4, 5], [7, 0, 8, 0]])
    with pytest.raises(ValueError, match="prefix mask"):
        model(interior)


def test_duration_embedding_is_on_the_code_embedding_scale():
    """A duration-bounded window token must not be dominated by the duration MLP.

    Only the final Linear of each MLP is rescaled (weight std = initializer_range, bias 0); the
    hidden ReLU layer keeps its default init, so the output norm measures ~3-5x an embedding row
    (7d / 365d, over seeds) instead of the ~12-19x of the default final-layer init.
    """
    torch.manual_seed(0)
    H = 384
    model = ConditionalMultitaskARModel(
        config_overrides=dict(MODEL_CFG, hidden_size=H, intermediate_size=4 * H), max_windows=5
    )
    code_row_norm = model.HF_model.get_input_embeddings().weight.norm(dim=1).mean().item()
    for mlp in (model.start_duration_embed, model.end_duration_embed):
        final = mlp.model[-1]
        assert isinstance(final, torch.nn.Linear)
        assert torch.equal(final.bias, torch.zeros(H))
        for days in (7.0, 365.0):
            emb = mlp(torch.tensor([[days / 365.0]]))
            ratio = emb.norm().item() / code_row_norm
            assert ratio < 6.0, f"duration embedding for {days}d is {ratio:.1f}x a code row"


def test_tied_readout_identity_and_gradient():
    model = tiny_model()
    assert "lm_head" not in dict(model.named_modules())
    weight = model.HF_model.get_input_embeddings().weight
    loss, _ = model(make_batch())
    loss.backward()
    assert weight is model.HF_model.get_input_embeddings().weight
    assert weight.grad is not None and weight.grad.isfinite().all() and weight.grad.abs().sum() > 0


def test_pad_target_is_excluded_from_loss():
    model = tiny_model()
    a = make_batch()
    b = make_batch()
    b.targets[:, :, 0] = ~b.targets[:, :, 0]
    loss_a, _ = model(a)
    loss_b, _ = model(b)
    torch.testing.assert_close(loss_a, loss_b)


def test_no_valid_elements_has_differentiable_zero_loss():
    model = tiny_model()
    batch = make_batch(q_mask=[[False] * 3] * 2)
    loss, out = model(batch)
    assert loss.item() == 0 and not out.valid_mask.any()
    loss.backward()


def test_k_one_and_legacy_starts():
    model = tiny_model()
    explicit = make_batch(n_windows=1)
    legacy = make_batch(n_windows=1, starts=False)
    assert explicit.condition_codes.shape == explicit.condition_answers.shape == (2, 0)
    assert model._query_tokens(explicit).shape[1] == 1
    _, a = model(explicit)
    _, b = model(legacy)
    torch.testing.assert_close(a.logits, b.logits)


def test_exactly_one_start_field_absent_raises():
    model = tiny_model()
    batch = make_batch()
    batch.q_start_codes = None
    with pytest.raises(ValueError, match="given together"):
        model(batch)


def test_validation_errors_without_dataclass_reconstruction():
    model = tiny_model()
    too_many = make_batch(n_windows=5)
    # Extend every K-shaped tensor consistently after construction.
    too_many.q_durations = torch.zeros(2, 6)
    too_many.q_bound_codes = torch.zeros(2, 6, dtype=torch.long)
    too_many.q_start_durations = torch.zeros(2, 6)
    too_many.q_start_codes = torch.zeros(2, 6, dtype=torch.long)
    too_many.q_mask = torch.ones(2, 6, dtype=torch.bool)
    too_many.targets = torch.zeros(2, 6, VOCAB, dtype=torch.bool)
    too_many.condition_codes = torch.ones(2, 5, dtype=torch.long)
    too_many.condition_answers = torch.zeros(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="max_windows"):
        model(too_many)

    wrong_vocab = make_batch()
    wrong_vocab.targets = torch.zeros(2, 3, VOCAB - 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="vocabulary width"):
        model(wrong_vocab)

    short_positions = tiny_model(config_overrides={"max_position_embeddings": 8})
    with pytest.raises(ValueError, match="max_position_embeddings"):
        short_positions(make_batch())

    with pytest.raises(NotImplementedError, match="ontology"):
        tiny_model(ontology_dir="unused")


def test_rope_time_pair_and_future_starts_do_not_advance_time():
    model = tiny_model(use_rope_time=True)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(make_batch())
    batch = make_batch(time_pos_ids=torch.tensor([[0, 12, 24, 36], [0, 5, 0, 0]]))
    n_patient = (batch.code != 0).sum(1)
    query_positions = n_patient[:, None] + torch.arange(7)[None, :]
    pos = model._position_ids(batch, n_patient, query_positions, 11)
    assert pos[0, 4:11].tolist() == [36] * 7
    assert pos[1, 2:9].tolist() == [5] * 7

    off = tiny_model(use_rope_time=False)
    with pytest.raises(ValueError, match="strip_delta_tokens"):
        off(batch)


def _save_checkpoint(module, path):
    import lightning as L

    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )


def test_lightning_predict_and_checkpoint_round_trip(tmp_path):
    from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule

    model = tiny_model()
    module = ConditionalMultitaskLightningModule(model=model, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    assert all(not metrics for metrics in module.metrics.values())
    prediction = module.predict_step(make_batch())
    assert set(prediction) == {
        "probs",
        "q_mask",
        "q_start_durations",
        "q_start_codes",
        "q_durations",
        "q_bound_codes",
        "targets",
        "condition_codes",
        "condition_answers",
    }
    ckpt = tmp_path / "model.ckpt"
    _save_checkpoint(module, ckpt)
    loaded = ConditionalMultitaskLightningModule.load_from_checkpoint(str(ckpt))
    assert isinstance(loaded.model, ConditionalMultitaskARModel)
    torch.testing.assert_close(loaded.model.code_bias, model.code_bias)

    bad = torch.load(ckpt, weights_only=False)
    bad["hyper_parameters"]["model"] = dict(bad["hyper_parameters"]["model"])
    bad["hyper_parameters"]["model"].pop("architecture")
    bad_ckpt = tmp_path / "missing-architecture.ckpt"
    torch.save(bad, bad_ckpt)
    with pytest.raises(KeyError, match="architecture"):
        ConditionalMultitaskLightningModule.load_from_checkpoint(str(bad_ckpt))


def test_configs_and_position_budget():
    from every_query.train.train import CONFIGS, required_position_embeddings

    for name in (
        "conditional_multitask_ar_config.yaml",
        "_demo_train_conditional_multitask_ar.yaml",
    ):
        cfg = yaml.safe_load((Path(CONFIGS) / name).read_text())
        assert cfg["datamodule"]["data_class"].endswith("MultitaskBoundaryPytorchDataset")
        assert "ontology_dir" not in cfg["datamodule"]["dataset_kwargs"]
        assert cfg["datamodule"]["dataset_kwargs"]["expected_vocab_size"].endswith(
            "config_overrides.vocab_size}"
        )
        assert cfg["lightning_module"]["model"]["max_windows"] == 5
    model_cfg = OmegaConf.create(
        {
            "_target_": "every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel",
            "max_windows": 5,
        }
    )
    assert required_position_embeddings(model_cfg, 256) == 271


def test_lazy_package_exports_do_not_cycle():
    from every_query.model import ConditionalMultitaskARModel as Exported

    assert Exported is ConditionalMultitaskARModel
