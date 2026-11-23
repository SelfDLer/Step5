"""Check actor-side Step5MoE.route against CPU results for the 16-expert setup."""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def production_router():
    root = Path(os.environ["STEP5_CHECK_FSDP_SRC"])
    expected = root / "fsdp_turbo/models/step5/modeling_step5.py"
    assert root.is_absolute() and expected.is_file(), f"Invalid FSDPTurbo root: {root}"
    impl = importlib.import_module("fsdp_turbo.models.step5.modeling_step5")
    assert Path(impl.__file__).resolve() == expected.resolve(), (
        f"Loaded {impl.__file__}, expected {expected}"
    )
    # Only the router is exercised. The unused expert MLPs can be small.
    config = SimpleNamespace(hidden_size=4096, moe_num_experts=16, moe_top_k=8,
                             moe_router_scaling_factor=3.0, norm_expert_weight=True,
                             swiglu_limits=[7.0], use_triton_mc2_permute=False,
                             use_triton_swiglu=True, moe_intermediate_size=2)
    return impl.Step5MoE(config, layer_idx=0)


def compare_routing(device, dtype, replay):
    router = production_router().to(device)
    # Keep the trained gate weight and activations in their test storage dtype.
    # Move only the gate to that dtype: the selection bias must stay FP32.
    router.gate.to(dtype=dtype)
    rng = torch.Generator().manual_seed(547)
    inputs = torch.randn(2, 3, 4096, generator=rng) * 0.2
    weights = torch.randn(16, 4096, generator=rng) * 1e-4
    weights[:, :16] = torch.eye(16)
    logits_pattern = torch.linspace(-1.8, 1.8, 16)
    for row, token in enumerate(inputs.reshape(-1, 4096)):
        token[:16] = logits_pattern.roll(row * 3)
    # Distinct scores avoid relying on cross-device top-k tie ordering.
    # Expert 0's selection-only bias moves it into the top k even on low logits.
    bias = torch.zeros(16)
    bias[0] = 1.0
    inputs, weights = inputs.to(dtype), weights.to(dtype)
    with torch.no_grad():
        router.gate.weight.copy_(weights)
        router.router_bias.copy_(bias)
    hidden = inputs.to(device).requires_grad_()

    cpu_hidden = inputs.double().requires_grad_()
    cpu_gate = weights.double().requires_grad_()
    scores = (cpu_hidden @ cpu_gate.T).sigmoid()
    ordering = (scores.detach() + bias.double()).argsort(dim=-1, descending=True)
    selected = ordering[..., :8]
    recorded = None
    mask = None
    if replay != "off":
        # Every recorded expert is valid and outside this token's normal top k.
        recorded = ordering[..., 8:].contiguous()
        mask = torch.ones((2, 3), dtype=torch.bool)
        if replay == "mixed":
            mask[:, 1] = False
            mask[1, 2] = False
        selected = torch.where(mask.unsqueeze(-1), recorded, selected)
    values = scores.gather(-1, selected)
    expected_weights = 3 * values / values.sum(dim=-1, keepdim=True)
    upstream = torch.randn(expected_weights.shape, generator=rng)
    expected_gradients = torch.autograd.grad(expected_weights, (cpu_hidden, cpu_gate), upstream.double())

    actual_weights, actual_ids = router.route(
        hidden, recorded.to(device) if recorded is not None else None,
        mask.to(device) if mask is not None else None,
    )
    actual_gradients = torch.autograd.grad(actual_weights, (hidden, router.gate.weight), upstream.to(device))
    assert actual_weights.dtype == torch.float32 and actual_ids.dtype == torch.int64
    torch.testing.assert_close(actual_ids.cpu(), selected, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights.cpu(), expected_weights.float(), atol=3e-5, rtol=2e-3)
    tolerances = dict(atol=2e-4, rtol=1e-2) if dtype == torch.bfloat16 else dict(atol=2e-5, rtol=2e-3)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert actual.dtype == dtype
        torch.testing.assert_close(actual.float().cpu(), expected.to(dtype).float(), **tolerances)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("replay", ["off", "all", "mixed"])
def test_step5_actor_routing_on_npu(dtype, replay):
    import torch_npu  # noqa: F401 — register the NPU device with PyTorch

    assert torch.npu.is_available(), "Step5 routing validation requires an available NPU"
    compare_routing(torch.device("npu:0"), dtype, replay)
    torch.npu.synchronize()
