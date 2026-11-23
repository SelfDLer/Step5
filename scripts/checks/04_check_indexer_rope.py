"""Run the pinned rollout indexer kernel on one NPU and compare with the reference."""
import argparse
from importlib import import_module
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-src", type=Path, required=True, help="absolute vllm-ascend checkout path")
    parser.add_argument("--positions", choices=("int64", "int32"), default="int64")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 33, 129])
    args = parser.parse_args()
    if not args.ascend_src.is_absolute() or min(args.lengths) < 1:
        parser.error("use an absolute checkout path and positive sequence lengths")

    import torch
    import torch_npu
    kernel_module = import_module("vllm_ascend.models.step5.fused_qknorm_rope")
    src = args.ascend_src.resolve()
    if Path(kernel_module.__file__).resolve() != src / "vllm_ascend/models/step5/fused_qknorm_rope.py":
        raise RuntimeError("active vllm-ascend and --ascend-src point to different checkouts")
    if not torch.npu.is_available():
        raise RuntimeError("a visible, idle NPU is required")
    torch.npu.set_device(0)
    tests = runpy.run_path(str(src / "tests/ut/ops/test_step4_fused_indexer_norm_rope.py"))
    dim, rotary, eps = tests["HEAD_DIM"], tests["ROTARY_DIM"], tests["EPS"]
    torch.manual_seed(42)
    def sample(*shape):
        return torch.randn(*shape, device="npu:0", dtype=torch.bfloat16)
    qw, kw, kb = sample(dim), sample(dim), sample(dim)
    cos, sin = tests["_build_cos_sin_cache"](4096, rotary // 2, tests["THETA"], "npu:0")
    # 16 query heads in 4 provider groups at TP8 -> 4 local Q heads, 1 local K head.
    opts = dict(head_dim=dim, rotary_dim=rotary, num_q_heads=4, num_k_heads=1,
                eps=eps, q_norm_weight_bias=1.0)
    for n in args.lengths:
        q, k, z = sample(n, 6 * dim).split((4 * dim, dim, dim), dim=-1)
        pos = torch.randint(4096, (n,), device="npu:0", dtype=getattr(torch, args.positions))
        print(f"RUN fused indexer positions={pos.dtype} tokens={n} Q={tuple(q.shape)} K={tuple(k.shape)}", flush=True)
        expected = tests["_reference_indexer_norm_rope"](q, k, z, qw, kw, kb, cos, sin, pos, **opts)
        actual = kernel_module.fused_indexer_qkz_norm_rope(q, k, z, qw, kw, cos, sin, pos,
                                                           k_norm_bias=kb, **opts)
        torch.npu.synchronize()
        for got, want in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
        print(f"PASS positions={pos.dtype} tokens={n}", flush=True)


if __name__ == "__main__":
    main()
