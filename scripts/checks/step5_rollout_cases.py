"""Run reference tests from the installed checkout against the Step5 NPU implementation."""

import ast
from pathlib import Path


_SINGLE = Path("tests/e2e/nightly/single_node/ops/singlecard_ops/triton")
_PREFILL = _SINGLE / "test_step4_csa_prefill_update.py"
_OLD = "vllm_ascend.models.step4."
_NEW = "vllm_ascend.models.step5."

# Only select numerical/structural checks. The long-context performance
# benchmarks in these files are separate workloads.
ROLLOUT_CASES = {
    "main_qk_norm_rope": (
        Path("tests/ut/ops/test_step4_fused_qknorm_rope.py"), "fused_qknorm_rope",
        ("test_fused_qknorm_rope_accuracy", "test_fused_qknorm_rope_rope_only_region",
         "test_step5_main_rope_position_dtype"),
    ),
    "paged_indexer_logits": (
        _SINGLE / "test_step4_indexer_logits_paged.py", "dsa_kernels",
        ("test_decode_identity_map", "test_spec_decode_verify_repeated_map",
         "test_prefill_packed_batch", "test_padded_mean_cache_page_stride"),
    ),
    "csa_prefill_default": (
        _PREFILL, "summary_kernels",
        ("test_prefill_update_matches_reference", "test_chunked_prefill_prefix_continuation",
         "test_mixed_decode_and_prefill_update"),
    ),
    "summary_layout_reset": (
        _SINGLE / "test_step4_dsa_summary_cache.py", "summary_kernels",
        ("test_layout_builder_matches_torch", "test_reset_blocks_clears_payload_and_slots"),
    ),
    "csa_prefill_planned": (
        _PREFILL, "summary_kernels", ("test_step5_planned_prefill_and_handoff",),
    ),
    "csa_prefill_handoff": (
        _PREFILL, "summary_kernels", ("test_step5_handoff_existing_and_occupied_slots",),
    ),
}

_MAIN_ROPE = '''
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_step5_main_rope_position_dtype(position_dtype, _import_kernel):
    # Packed positions include repeated request-local indices and a prefill tail.
    positions = torch.tensor([7, 7, 1, 2, 3, 4, 5, 6, 7], device="npu", dtype=position_dtype)
    q_heads, kv_heads, rotary_dim = 8, 1, 128
    torch.manual_seed(19)
    qkv = torch.randn(len(positions), (q_heads + 2 * kv_heads) * HEAD_DIM,
                      device="npu", dtype=torch.bfloat16)
    qw, kw = torch.randn(2, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    # Step5 rebuilds the RoPE coefficients in FP32 and passes split cache views.
    freqs = torch.outer(torch.arange(64, device="npu", dtype=torch.float32),
                        THETA_DSA ** (-torch.arange(0, rotary_dim, 2, device="npu",
                                                   dtype=torch.float32) / rotary_dim))
    cos, sin = torch.cat([freqs.cos(), freqs.sin()], -1).chunk(2, dim=-1)
    expected = _reference_qknorm_rope(qkv, qw, kw, cos, sin, positions, num_q_heads=q_heads,
                                      num_kv_heads=kv_heads, rotary_dim=rotary_dim,
                                      eps=EPS, norm_weight_bias=1.0)
    actual = _import_kernel(qkv, qw, kw, cos, sin, positions, head_dim=HEAD_DIM,
                            num_q_heads=q_heads, num_kv_heads=kv_heads,
                            rotary_pairs=rotary_dim // 2, eps=EPS, norm_weight_bias=1.0)
    torch.npu.synchronize()
    for ref, result in zip(expected[:2], actual[:2], strict=True):
        torch.testing.assert_close(result, ref, atol=0.01, rtol=0.01)
    assert torch.equal(expected[2], actual[2])
'''

_PLANNED = '''
def test_step5_planned_prefill_and_handoff():
    from vllm_ascend.worker.step5_csa_slot_manager import CSASlotManager
    from vllm_ascend.models.step5.summary_kernels import csa_compact_prefill_update, csa_prefill_handoff

    torch.manual_seed(23)
    prefill, decode = _alloc_state(), _alloc_state()
    manager = CSASlotManager(max_num_seqs=1, device=torch.device(DEVICE), pin_memory=False)
    # Two chunks span multiple regions. The first retains an incomplete
    # region for a continuation. The second transfers its tail to decode.
    total = 3 * REGION_SIZE + 4
    table = torch.tensor([3, 5], dtype=torch.int32, device=DEVICE)
    k, z = torch.randn(2, total, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    cut = REGION_SIZE + 4
    for start, end in ((0, cut), (cut, total)):
        rows = end - start
        qsl = torch.tensor([0, rows], dtype=torch.int32, device=DEVICE)
        lens = torch.tensor([end], dtype=torch.int32, device=DEVICE)
        plan = manager.plan(["request"], [start], [end], REGION_SIZE, [total])
        assert plan.needs_handoff == (end == total)
        flat, _, pos, reset, valid = _layout(_slot_mapping(table, range(start, end)), qsl, lens, rows)
        csa_compact_prefill_update(
            prefill["mean_cache"], prefill["active_region_ids"], prefill["active_slot_by_region"],
            prefill["allocation_success"], prefill["numerator"], prefill["denominator"],
            prefill["max_logits"], prefill["staging_valid"], flat, reset, valid, pos,
            k[start:end].view(rows, 1, PROXY_DIM), z[start:end].view(rows, 1, PROXY_DIM),
            query_start_loc=qsl, seq_lens=lens, live_rows=rows, num_pages=NUM_PAGES,
            summaries_per_page=SUMMARIES_PER_PAGE, proxy_dim=PROXY_DIM,
            region_block_size=REGION_SIZE, slot_plan=plan,
        )
        torch.npu.synchronize()
        assert prefill["allocation_success"].item() == 1
        _check_means(prefill, table, _ref_means(k[:end].float(), z[:end].float()))
    source = plan.target_slot.item()
    assert 0 <= source < ACTIVE_CAPACITY
    region = prefill["active_region_ids"][source].item()
    last = (total - 1) // REGION_SIZE
    assert region == table[last // SUMMARIES_PER_PAGE].item() * SUMMARIES_PER_PAGE + last % SUMMARIES_PER_PAGE
    expected_tail = (k[last * REGION_SIZE:].float() * z[last * REGION_SIZE:].float().softmax(dim=0)).sum(0)
    csa_prefill_handoff(
        prefill["active_region_ids"], prefill["active_slot_by_region"], prefill["numerator"],
        prefill["denominator"], prefill["max_logits"], decode["active_region_ids"],
        decode["active_slot_by_region"], decode["numerator"], decode["denominator"],
        decode["max_logits"], decode["staging_valid"], decode["allocation_success"],
        plan, proxy_dim=PROXY_DIM, region_size=REGION_SIZE,
    )
    torch.npu.synchronize()
    target = decode["active_slot_by_region"][region].item()
    assert 0 <= target < ACTIVE_CAPACITY and decode["allocation_success"].item() == 1
    assert decode["active_region_ids"][target].item() == region
    assert prefill["active_region_ids"][source].item() == -1
    assert prefill["active_slot_by_region"][region].item() == -1
    result = (decode["numerator"][target] / decode["denominator"][target]).reshape(-1)
    torch.testing.assert_close(result, expected_tail, rtol=2e-4, atol=2e-4)
'''

_HANDOFF = '''
def test_step5_handoff_existing_and_occupied_slots():
    from vllm_ascend.worker.step5_csa_slot_manager import CSASlotManager
    from vllm_ascend.models.step5.summary_kernels import csa_prefill_handoff

    prefill, decode = _alloc_state(), _alloc_state()
    manager = CSASlotManager(max_num_seqs=2, device=torch.device(DEVICE), pin_memory=False)
    metadata = manager.plan(["a", "b"], [0, 0], [4, 6], REGION_SIZE, [4, 6])
    assert metadata.needs_handoff and metadata.handoff.tolist() == [1, 1]
    # Both requests have incomplete summaries. The first region already has a
    # decode slot; the second region's first candidate belongs to another request.
    regions = (2, 3)
    sources = metadata.target_slot.tolist()
    assert len(set(sources)) == len(regions) and all(0 <= s < ACTIVE_CAPACITY for s in sources)
    decode["active_region_ids"][0] = 2
    decode["active_slot_by_region"][2] = 0
    decode["active_region_ids"][3] = 13
    decode["active_slot_by_region"][13] = 3
    decode["staging_valid"].fill_(1)
    torch.manual_seed(31)
    for source, region in zip(sources, regions, strict=True):
        prefill["active_region_ids"][source] = region
        prefill["active_slot_by_region"][region] = source
        prefill["numerator"][source].normal_()
        prefill["denominator"][source].uniform_(1, 2)
        prefill["max_logits"][source] = float(region)
    expected = {key: prefill[key].clone() for key in ("numerator", "denominator", "max_logits")}
    csa_prefill_handoff(prefill["active_region_ids"], prefill["active_slot_by_region"],
                        prefill["numerator"], prefill["denominator"], prefill["max_logits"],
                        decode["active_region_ids"], decode["active_slot_by_region"],
                        decode["numerator"], decode["denominator"], decode["max_logits"],
                        decode["staging_valid"], decode["allocation_success"], metadata,
                        proxy_dim=PROXY_DIM, region_size=REGION_SIZE)
    torch.npu.synchronize()
    assert decode["allocation_success"].item() == 1
    assert decode["active_slot_by_region"][2].item() == 0
    assert decode["active_region_ids"][3].item() == 13
    for source, region in zip(sources, regions, strict=True):
        target = decode["active_slot_by_region"][region].item()
        assert 0 <= target < ACTIVE_CAPACITY and decode["active_region_ids"][target].item() == region
        for key, values in expected.items():
            torch.testing.assert_close(decode[key][target], values[source], rtol=0, atol=0)
        assert not decode["staging_valid"][target].any().item()
        assert prefill["active_region_ids"][source].item() == -1
        assert prefill["active_slot_by_region"][region].item() == -1
'''


def prepare_rollout_case(name, ascend_src, out):
    """Materialize upstream test logic with the Step5 implementation under test."""
    relpath, module, nodes = ROLLOUT_CASES[name]
    ascend_src, out = Path(ascend_src).resolve(), Path(out).resolve()
    source = ascend_src / relpath
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == _OLD + module]
    if len(imports) != 1:
        raise ValueError(f"{source}: expected one reference import from {_OLD + module}")
    node = imports[0]
    aliases = []
    for item in node.names:
        target = item.name
        if item.name == "csa_compact_prefill_update":
            # The old reference helper takes three scratch tensors. Step5 has
            # preserved this path under _default, alongside the planned API.
            target += "_default"
        alias = item.asname or item.name
        aliases.append(target if target == alias else f"{target} as {alias}")
    lines = text.splitlines(keepends=True)
    indent = " " * node.col_offset
    lines[node.lineno - 1:node.end_lineno] = [f"{indent}from {_NEW + module} import {', '.join(aliases)}\n"]
    adapted = "".join(lines)
    if name == "main_qk_norm_rope":
        adapted += "\n" + _MAIN_ROPE
    if name == "csa_prefill_planned":
        adapted += "\n" + _PLANNED
    if name == "csa_prefill_handoff":
        adapted += "\n" + _HANDOFF
    functions = {n.name for n in ast.parse(adapted).body if isinstance(n, ast.FunctionDef)}
    if missing := set(nodes) - functions:
        raise ValueError(f"{source}: missing tests {sorted(missing)}")
    # A second editable install on sys.path would otherwise exercise the wrong
    # fork. Check imports inside the pytest process at collection time.
    modules = [_NEW + module]
    if name in {"csa_prefill_planned", "csa_prefill_handoff"}:
        modules.append("vllm_ascend.worker.step5_csa_slot_manager")
    adapted += "\nfrom importlib import import_module as _step5_import\nfrom pathlib import Path as _Step5Path\n"
    for mod in modules:
        expected = (ascend_src / (mod.replace(".", "/") + ".py")).resolve(strict=True)
        adapted += (f"_loaded = _Step5Path(_step5_import({mod!r}).__file__).resolve()\n"
                    f"assert _loaded == _Step5Path({str(expected)!r}), "
                    f"f'Expected {mod} from the selected checkout, loaded {{_loaded}}'\n")
    adapted += """
import pytest as _step5_pytest
@_step5_pytest.fixture(autouse=True)
def _step5_check_device():
    assert torch.npu.is_available(), "NPU unavailable"
    yield
    torch.npu.synchronize()
"""
    out.mkdir(parents=True, exist_ok=True)
    output = out / f"test_step5_{name}.py"
    output.write_text(adapted, encoding="utf-8")
    return ["-m", "pytest", "-q", "-rA", *[f"{output}::{test}" for test in nodes]]
