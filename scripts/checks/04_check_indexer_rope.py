"""Check Step5 rollout indexer RoPE with the model's projected tensor layout."""
import argparse
from contextlib import nullcontext
from importlib import import_module
import json
import os
from pathlib import Path
import runpy
import tempfile
import traceback
from unittest.mock import patch


def projection_sections(config, tp):
    """Per-rank column widths from the pinned Step5FusedQKVIndexerLinear."""
    sparse = config["sparse_config"]
    main_heads = config["num_attention_heads"]
    kv_heads = config["num_attention_groups"]
    main_dim = config["head_dim"]
    groups = sparse.get("index_tp_size", sparse.get("num_provider_groups", 4))
    if groups <= 0:
        groups = tp
    q_heads = sparse["sparse_indexer_num_heads"]
    k_heads = sparse["sparse_indexer_num_k_heads"]
    for width in (kv_heads, k_heads, groups):
        if max(tp, width) % min(tp, width):
            raise ValueError(f"TP={tp} and head/group count={width} cannot shard or replicate evenly")
    if main_heads % tp or q_heads % groups:
        raise ValueError(f"incompatible TP={tp}, attention heads={main_heads}, indexer groups={groups}")
    dim = sparse["proxy_dim"]
    main_width = (main_heads // tp + 2 * max(1, kv_heads // tp)) * main_dim
    gate_width = main_heads // tp if config.get("use_head_wise_attn_gate", False) else 0
    q_width = q_heads // min(tp, groups) * dim
    k_width = max(1, k_heads // tp) * dim
    return main_width, q_width, k_width, k_width, gate_width


class CaptureLaunch:
    """Record exactly what the registered custom op passes to Triton."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.invocation = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.invocation = grid, args, kwargs
            return self.kernel[grid](*args, **kwargs)
        return launch


def compare(torch, outputs, expected):
    torch.npu.synchronize()
    for got, want in zip(outputs[:2], expected[:2], strict=True):
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(outputs[2], expected[2], atol=0, rtol=0)


def exercise_tiles(torch, capture, expected):
    """Try each registered Triton tile, even if the autotuner prefers a different one."""
    if capture.invocation is None:
        raise RuntimeError("real registered op did not issue the expected Triton launch")
    grid, args, kwargs = capture.invocation
    kernel = capture.kernel
    if not kernel.configs:
        raise RuntimeError("indexer autotuner exposes no candidate tiles")
    outputs = args[3:6]
    failures = []
    for config in kernel.configs:
        print(f"RUN tile={config.kwargs}", flush=True)
        try:
            # Capture uses the pinned kernel's x_q/x_k/x_z, out_q/out_k/out_z ABI.
            # Poison previous results to expose candidates that omit any stores.
            for output in outputs:
                output.fill_(float("nan"))
            kernel.fn[grid](*args, **kwargs, **config.all_kwargs())
            compare(torch, outputs, expected)
        except Exception:
            failures.append(config.kwargs)
            traceback.print_exc()
        else:
            print(f"PASS tile={config.kwargs}", flush=True)
    if failures:
        raise RuntimeError(f"failed candidate tiles: {failures}")


def run(args):
    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend

    src = args.ascend_src.resolve()
    mod = import_module("vllm_ascend.models.step5.fused_qknorm_rope")
    if Path(mod.__file__).resolve() != src / "vllm_ascend/models/step5/fused_qknorm_rope.py":
        raise RuntimeError("active vllm-ascend and --ascend-src point to different checkouts")
    if not torch.npu.is_available():
        raise RuntimeError("a visible, idle NPU is required")
    torch.npu.set_device(0)
    tests = runpy.run_path(str(src / "tests/ut/ops/test_step4_fused_indexer_norm_rope.py"))
    model = json.loads(args.config.read_text(encoding="utf-8"))
    config = model.get("text_config", model)
    sparse = config["sparse_config"]
    dim = sparse["proxy_dim"]
    rotary = sparse["sparse_indexer_rope_dim"]
    sections = projection_sections(config, args.tp)
    opts = dict(head_dim=dim, rotary_dim=rotary, num_q_heads=sections[1] // dim,
                num_k_heads=sections[2] // dim, eps=config["rms_norm_eps"],
                q_norm_weight_bias=float(config.get("zero_centered", True)))
    print(f"config={args.config} TP={args.tp} projected_sections={sections} opts={opts} "
          f"TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}", flush=True)
    torch.manual_seed(42)

    def sample(*shape):
        return torch.randn(*shape, device="npu:0", dtype=torch.bfloat16)

    qw, kw, kb = sample(dim), sample(dim), sample(dim)
    cache_rows = max(4096, max(args.lengths))
    cos, sin = tests["_build_cos_sin_cache"](cache_rows, rotary // 2, tests["THETA"], "npu:0")
    lengths = list(dict.fromkeys(args.lengths))
    tile_length = 33 if 33 in lengths else lengths[0]
    failures = []
    for n in lengths:
        # The actual forward slices the QKV prefix and gate from this single GEMM.
        combined = sample(n, sum(sections))
        _, q, k, z, _ = combined.split(sections, dim=-1)
        pos = torch.randint(cache_rows, (n,), device="npu:0", dtype=getattr(torch, args.positions))
        print(f"RUN positions={pos.dtype} tokens={n} shape={tuple(q.shape)} "
              f"row_stride={q.stride(0)} offsets={[t.storage_offset() for t in (q, k, z)]}", flush=True)
        expected = tests["_reference_indexer_norm_rope"](q, k, z, qw, kw, kb, cos, sin, pos, **opts)
        capture = CaptureLaunch(mod._indexer_qkz_norm_rope_kernel)
        hook = patch.object(mod, "_indexer_qkz_norm_rope_kernel", capture) if n == tile_length else nullcontext()
        try:
            with hook:
                actual = mod.fused_indexer_qkz_norm_rope(q, k, z, qw, kw, cos, sin, pos, k_norm_bias=kb, **opts)
            compare(torch, actual, expected)
        except Exception:
            failures.append((n, "registered op"))
            traceback.print_exc()
        else:
            print(f"PASS registered op positions={pos.dtype} tokens={n}", flush=True)
        if n == tile_length:
            try:
                exercise_tiles(torch, capture, expected)
            except Exception:
                failures.append((n, "tile sweep"))
                traceback.print_exc()
    if failures:
        raise RuntimeError(f"indexer failures for {args.positions}: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "rl/step5_16e_config.json")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--positions", choices=("int64", "int32"), default="int64")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 8, 32, 33, 129, 1024, 1152, 3072])
    args = parser.parse_args()
    if not args.ascend_src.is_absolute() or not args.config.is_absolute() or min(args.lengths) < 1 or args.tp < 1:
        parser.error("use absolute paths, positive TP and positive sequence lengths")
    # The suite supplies an empty per-case cache for inspection after execution.
    # A standalone invocation without an explicit cache gets a fresh one, too.
    cache = os.environ.get("TRITON_CACHE_DIR")
    with (nullcontext(cache) if cache else tempfile.TemporaryDirectory(prefix="step5-indexer-")) as cache_dir:
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        run(args)


if __name__ == "__main__":
    main()
