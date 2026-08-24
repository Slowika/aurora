"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the AuroraV1p5 forward pass.
"""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch

from ._helpers import _ATMOS_VARS, _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import Swin3DBlockAdapter, Swin3DResidualAdapter
from aurora.insolation import insolation
from aurora.model.film import AdaptiveLayerNorm
from aurora.model.swin3d import Swin3DTransformerBlock


def _make_backbone_adapters(
    model,
    value: float = 0.0,
    requires_grad: bool = False,
) -> list[Swin3DBlockAdapter]:
    def make_residual(dim: int) -> Swin3DResidualAdapter:
        return Swin3DResidualAdapter(
            scale=torch.full((1, dim), value, requires_grad=requires_grad),
            shift=torch.full((1, dim), value, requires_grad=requires_grad),
            gate=torch.full((1, dim), value, requires_grad=requires_grad),
        )

    return [
        Swin3DBlockAdapter(attention=make_residual(dim), mlp=make_residual(dim))
        for dim in model.backbone.adapter_dims
    ]


def test_forward_produces_all_vars():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    # Model should predict all `surf_vars` including output-only ones.
    for v in _SURF_VARS:
        assert v in pred.surf_vars, f"Missing surface variable: {v}"
    for v in _ATMOS_VARS:
        assert v in pred.atmos_vars, f"Missing atmospheric variable: {v}"


def test_forward_advances_time():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    expected_time = tuple(t + timedelta(hours=6) for t in batch.metadata.time)
    assert pred.metadata.time == expected_time
    assert pred.metadata.rollout_step == 1


def test_variable_lead_time_changes_output_time():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    with torch.inference_mode():
        pred3 = model.forward(batch, lead_times=torch.tensor([3.0]))
        pred6 = model.forward(batch, lead_times=torch.tensor([6.0]))

    expected3 = tuple(t + timedelta(hours=3) for t in batch.metadata.time)
    expected6 = tuple(t + timedelta(hours=6) for t in batch.metadata.time)
    assert pred3.metadata.time == expected3
    assert pred6.metadata.time == expected6


def test_missing_lead_times_raises():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with pytest.raises(ValueError, match="lead_times"):
        model.forward(batch)


def test_backbone_adapters_condition_frozen_model():
    model = _make_small_v1p5()
    model.eval()
    for module in model.backbone.modules():
        if isinstance(module, AdaptiveLayerNorm):
            torch.nn.init.normal_(module.ln_modulation[-1].weight, std=0.02)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    lead_times = torch.tensor([6.0])
    assert model.backbone.adapter_dims == (64, 64, 128, 128, 128, 128, 64, 64)
    identity_adapters = _make_backbone_adapters(model)

    with torch.inference_mode():
        pred_without_adapters = model.forward(batch, lead_times=lead_times)
        pred_with_identity_adapters = model.forward(
            batch,
            lead_times=lead_times,
            backbone_adapters=identity_adapters,
        )

    without_adapters = tuple(pred_without_adapters.surf_vars.values()) + tuple(
        pred_without_adapters.atmos_vars.values()
    )
    with_identity_adapters = tuple(pred_with_identity_adapters.surf_vars.values()) + tuple(
        pred_with_identity_adapters.atmos_vars.values()
    )
    for actual, expected in zip(with_identity_adapters, without_adapters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    backbone_adapters = _make_backbone_adapters(model, value=0.1, requires_grad=True)
    pred_with_adapters = model.forward(
        batch,
        lead_times=lead_times,
        backbone_adapters=backbone_adapters,
    )
    with_adapters = tuple(pred_with_adapters.surf_vars.values()) + tuple(
        pred_with_adapters.atmos_vars.values()
    )
    assert any(
        not torch.equal(actual, expected)
        for actual, expected in zip(with_adapters, without_adapters)
    )

    loss = sum(value.square().mean() for value in with_adapters)
    loss.backward()
    for adapter in backbone_adapters:
        for residual in adapter:
            for value in residual:
                assert value.grad is not None
                assert torch.count_nonzero(value.grad)


def test_backbone_adapters_validate_count_and_shape():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    backbone_adapters = _make_backbone_adapters(model)

    with pytest.raises(ValueError, match="8 block adapters"):
        model.forward(
            batch,
            lead_times=torch.tensor([6.0]),
            backbone_adapters=backbone_adapters[:-1],
        )

    first_adapter = backbone_adapters[0]
    backbone_adapters[0] = Swin3DBlockAdapter(
        attention=first_adapter.attention._replace(scale=torch.zeros(1, 65)),
        mlp=first_adapter.mlp,
    )

    with pytest.raises(ValueError, match="attention.scale"):
        model.forward(
            batch,
            lead_times=torch.tensor([6.0]),
            backbone_adapters=backbone_adapters,
        )


def test_backbone_adapter_modulates_sublayer_inputs():
    block = Swin3DTransformerBlock(
        dim=4,
        num_heads=1,
        time_dim=4,
        window_size=(1, 2, 2),
    )
    block.eval()
    x = torch.randn(1, 4, 4)
    context = torch.zeros(1, 4)
    captured: dict[str, list[torch.Tensor]] = {"attention": [], "mlp": []}

    def capture(name: str):
        def hook(module, inputs):
            captured[name].append(inputs[0].clone())

        return hook

    block.attn.register_forward_pre_hook(capture("attention"))
    block.mlp.register_forward_pre_hook(capture("mlp"))

    residual = Swin3DResidualAdapter(
        scale=torch.ones(1, 4),
        shift=torch.full((1, 4), 0.25),
        gate=torch.zeros(1, 4),
    )
    adapter = Swin3DBlockAdapter(attention=residual, mlp=residual)

    with torch.inference_mode():
        block(x, context, res=(1, 2, 2), rollout_step=0)
        block(x, context, res=(1, 2, 2), rollout_step=0, adapter=adapter)

    for name in ("attention", "mlp"):
        expected = captured[name][0] * 2 + 0.25
        torch.testing.assert_close(captured[name][1], expected)


def test_insolation_is_recomputed():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    with torch.inference_mode(), patch("aurora.model.aurora.insolation", wraps=insolation) as mock:
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    mock.assert_called()
    assert torch.isfinite(pred.surf_vars["insolation"]).all()


def test_log_transformed_vars_are_nonnegative():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    # Make the input positive for log-transformed vars.
    for k in batch.surf_vars:
        if k.startswith("scaled_"):
            batch.surf_vars[k] = batch.surf_vars[k].abs()

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    # `log_unscale(x) = eps * (exp(x) - 1)` with `eps = 1e-3`, so the theoretical minimum is `-eps`\
    # (when x -> -inf). Allow that margin.
    for k in pred.surf_vars:
        if k.startswith("scaled_"):
            message = f"Log-transformed var `{k}` has unexpected negative values."
            assert (pred.surf_vars[k] >= -1e-3).all(), message
