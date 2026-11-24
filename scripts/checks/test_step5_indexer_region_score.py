"""Check rollout IndexerRegionScore against independent CPU scores and top-k membership."""
from importlib import import_module
import os
from pathlib import Path
import runpy

import pytest
import torch


@pytest.fixture(scope="module")
def upstream():
    src = Path(os.environ["STEP5_CHECK_ASCEND_SRC"]).resolve(strict=True)
    tests = runpy.run_path(str(src / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_indexer_region_score.py"))
    adapter = import_module("vllm_ascend.models.step5.indexer_region_score")
    assert Path(adapter.__file__).resolve() == src / "vllm_ascend/models/step5/indexer_region_score.py"
    assert tests["indexer_score_topk"] is adapter.indexer_score_topk
    assert callable(adapter.indexer_region_score), "IndexerRegionScore provider is unavailable"
    return tests


@pytest.mark.parametrize(("width", "lengths", "query_lengths"), [
    pytest.param(384, [1, 1025, 3072], [1, 1, 1], id="decode-3072-limit"),
    pytest.param(0, [3, 8], [3, 8], id="prefill-no-completed-region"),
    pytest.param(600, [9, 4801], [1, 4], id="prefill-pruning-and-repeatability"),
])
@torch.inference_mode()
def test_context_scores_and_ids(upstream, width, lengths, query_lengths):
    # Even when no previous region is visible, active requests own real KV pages.
    backing = max(upstream["SUMMARIES_PER_PAGE"], width)
    inputs = upstream["_build_inputs"](lengths, query_lengths, backing, seed=47)
    tokens, topk = sum(query_lengths), upstream["TOPK"]
    if width > topk:
        # One pruned row has an exact tie at the cutoff; the other rows retain random scores.
        inputs[0][query_lengths[0]].zero_()
    scores, ids = upstream["_run_fused"](*inputs, width)
    torch.npu.synchronize()
    assert scores.device.type == ids.device.type == "npu"
    assert scores.dtype == torch.float32 and ids.dtype == torch.int32
    assert scores.ndim == 2 and scores.shape[0] == tokens and scores.shape[1] >= width
    assert ids.shape == (tokens, topk)
    ids = ids.cpu()
    if width > topk:
        repeated_scores, repeated_ids = upstream["_run_fused"](*inputs, width)
        torch.npu.synchronize()
        torch.testing.assert_close(repeated_ids.cpu(), ids, rtol=0, atol=0)
    q, weights, cache, table, _, _ = inputs
    # Reproduce the two FP8 operands and the signed FP32 head reduction on CPU.
    q = q.cpu().to(torch.float8_e4m3fn).float()
    weights, cache, table = weights.cpu(), cache.cpu().float(), table.cpu().long()
    start = 0
    for req, (length, count) in enumerate(zip(lengths, query_lengths, strict=True)):
        logical_keys = cache[table[req], :, 0, :].reshape(-1, q.shape[-1])
        for offset in range(count):
            row = start + offset
            visible = (length - count + offset) // upstream["REGION_SIZE"]
            assert 0 <= visible <= width
            kept = min(visible, topk)
            selected = ids[row, :kept].long()
            assert (ids[row, kept:] == -1).all()
            assert ((selected >= 0) & (selected < visible)).all()
            assert (selected[1:] > selected[:-1]).all()
            if visible <= topk:
                torch.testing.assert_close(selected, torch.arange(visible), rtol=0, atol=0)
            if visible:
                dots = q[row, 0] @ logical_keys[:visible].T
                expected = (dots.relu() * weights[row, 0, :, None]).sum(dim=0)
                actual = scores[row, :visible].cpu()
                torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.1)
                if width > topk:
                    torch.testing.assert_close(repeated_scores[row, :visible].cpu(), actual, rtol=0, atol=0)
                if visible > topk:
                    omitted = torch.ones(visible, dtype=torch.bool)
                    omitted[selected] = False
                    # Equal scores may select different IDs. Check the cutoff on both sides.
                    assert actual[selected].min() >= actual[omitted].max()
                    cutoff = expected.topk(topk).values[-1]
                    assert expected[selected].min() >= cutoff - 1e-3
                    assert expected[omitted].max() <= cutoff + 1e-3
        start += count
