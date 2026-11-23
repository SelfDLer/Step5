"""Check the A3 Step5 16-expert rollout router's device op against its NumPy oracle."""
import os
from pathlib import Path
import runpy

import pytest
import torch


@pytest.fixture(scope="module")
def gate_and_reference():
    import torch_npu  # noqa: F401 -- registers the NPU backend
    from vllm_ascend.device import device_op

    src = Path(os.environ["STEP5_CHECK_ASCEND_SRC"]).resolve(strict=True)
    if Path(device_op.__file__).resolve() != src / "vllm_ascend/device/device_op.py":
        pytest.fail("active DeviceOperator and STEP5_CHECK_ASCEND_SRC use different checkouts")
    if device_op.DeviceOperator is not device_op.BaseDeviceAdaptor:
        pytest.fail(f"expected the A3 standard device adaptor, got {device_op.DeviceOperator.__name__}")
    # This source registers the extension and provides a separate CPU NumPy oracle.
    tests = runpy.run_path(str(src / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_moe_gating_top_k.py"))
    return device_op.DeviceOperator.moe_gating_top_k, tests["moe_gating_top_k_numpy_ref"]


@pytest.mark.parametrize("tokens", [1, 8, 33, 128, 1152, 3072])
@torch.inference_mode()
def test_step5_rollout_moe_gate(tokens, gate_and_reference):
    gate, oracle = gate_and_reference
    experts, topk, scale = 16, 8, 3.0
    expert = torch.arange(experts)
    row = torch.arange(tokens).unsqueeze(1)
    logits = (((expert + 7 * row) % experts).float() - 7.5) / 4
    bias = (expert.float() - 7.5) * 0.017
    options = dict(k=topk, k_group=1, group_count=1, group_select_mode=1, renorm=1,
                   norm_type=1, routed_scaling_factor=scale, eps=1e-20)
    weights_ref, ids_ref, _ = oracle(logits, bias=bias, y2_flag=False, **options)
    weights, ids, _ = gate(logits.to("npu"), bias_opt=bias.to("npu"), out_flag=False, **options)
    torch.npu.synchronize()
    torch.testing.assert_close(ids.cpu(), torch.from_numpy(ids_ref), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), weights_ref, rtol=1e-5, atol=1e-8)
