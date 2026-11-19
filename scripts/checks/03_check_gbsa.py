"""Check the Step5 GBSA API and BF16 TND forward/backward against CPU FP32."""
import argparse
import inspect
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-prefix', required=True)
    parser.add_argument('--api-only', action='store_true', help='Import/signature check only; no NPU execution')
    parser.add_argument('--device', type=int, default=0, help='Logical device after visibility filtering')
    parser.add_argument('--lengths', type=int, nargs='+', default=[16, 129])
    parser.add_argument('--kv-heads', type=int, choices=[1, 2, 4], default=4)
    parser.add_argument('--threshold', type=float, default=0.02, help='Smoke-test relative L2 bound, not a model acceptance target')
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path(args.expected_prefix).resolve():
        raise RuntimeError(f'Wrong Python prefix: {sys.prefix}')
    if any(n < 1 or n > 4104 for n in args.lengths):
        parser.error('The all-selected-prefix test supports lengths 1..4104 only')
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error('threshold must be positive and finite')

    import torch
    import torch_npu
    import cann_ops_transformer as ops
    print('Python:', sys.executable)
    print('torch:', torch.__version__, torch.__file__)
    print('torch_npu:', torch_npu.__version__, torch_npu.__file__)
    print('cann_ops_transformer:', ops.__file__, flush=True)
    for name in ('generic_block_sparse_attention_metadata', 'generic_block_sparse_attention', 'generic_block_sparse_attention_grad'):
        fn = getattr(ops, name, None)
        if not callable(fn):
            raise RuntimeError(f'Missing callable: cann_ops_transformer.{name}')
        try:
            signature = str(inspect.signature(fn))
        except (TypeError, ValueError):
            signature = '<signature unavailable>'
        print(name, signature, flush=True)
    print('GBSA_API PASS (callable interfaces only)', flush=True)
    if args.api_only:
        return

    if not torch.npu.is_available():
        raise RuntimeError('NPU unavailable')
    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = f'npu:{args.device}'
    kv_heads = args.kv_heads
    q_heads = kv_heads * 16

    def relative_l2(actual, reference):
        a = actual.detach().cpu().float()
        b = reference.detach().cpu().float()
        if a.shape != b.shape:
            raise RuntimeError(f'Shape mismatch: {a.shape} vs {b.shape}')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('NaN/Inf detected')
        return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

    for length in args.lengths:
        q, k, v = [torch.randn(length, h, 192).to(torch.bfloat16).to(device) for h in (q_heads, kv_heads, kv_heads)]
        cu = torch.tensor([0, length], device=device, dtype=torch.int64)
        # Complete blocks are explicit. residual_block_mode=1 supplies the partial current block.
        counts_cpu = (torch.arange(length) + 1) // 8
        blocks = torch.arange(513, dtype=torch.int32)
        ids_cpu = blocks[None, :].expand(length, -1).clone()
        ids_cpu.masked_fill_(blocks[None, :] >= counts_cpu[:, None], -1)
        ids = ids_cpu.unsqueeze(0).expand(kv_heads, -1, -1).contiguous().to(device)
        counts = counts_cpu.to(torch.int32)[None, :].expand(kv_heads, -1).contiguous().to(device)
        scale = 1 / math.sqrt(192)
        common = dict(cu_seqlens_q=cu, cu_seqlens_kv=cu, layout_q='TND', layout_kv='TND',
                      layout_sparse_pattern=4, mask_mode=1, quant_mode=0, softmax_precision=0,
                      residual_block_mode=1, is_consistent_topk=False)
        metadata = ops.generic_block_sparse_attention_metadata(
            ids, counts, q_heads, kv_heads, 192, [1, 8], max_seqlen_q=length, max_seqlen_kv=length, **common)
        y, lse = ops.generic_block_sparse_attention(
            q, k, v, ids, counts, [1, 8], metadata=metadata, softmax_scale=scale, return_softmax_lse=True, **common)
        dy = torch.randn_like(y)
        dq, dk, dv = ops.generic_block_sparse_attention_grad(
            q, k, v, dy.contiguous(), y, lse, ids.unsqueeze(0), counts.unsqueeze(0), [1, 8],
            metadata=None, attn_mask=None, layout_sparse_pattern=0, softmax_precision=0,
            residual_block_mode=1, is_consistent_topk=False, cu_seqlens_q=cu, cu_seqlens_kv=cu,
            layout_q='TND', layout_kv='TND', softmax_scale=scale, mask_mode=1)
        torch.npu.synchronize()
        qr, kr, vr = [x.cpu().float().requires_grad_() for x in (q, k, v)]
        ref = torch.nn.functional.scaled_dot_product_attention(
            qr.transpose(0, 1), kr.repeat_interleave(16, dim=1).transpose(0, 1),
            vr.repeat_interleave(16, dim=1).transpose(0, 1), is_causal=True).transpose(0, 1)
        ref.backward(dy.cpu().float())
        errors = {name: relative_l2(a, b) for name, a, b in
                  [('output', y, ref), ('dq', dq, qr.grad), ('dk', dk, kr.grad), ('dv', dv, vr.grad)]}
        print(json.dumps({'length': length, 'q_heads': q_heads, 'kv_heads': kv_heads,
                          'head_dim': 192, 'relative_l2': errors}), flush=True)
        if max(errors.values()) >= args.threshold:
            raise RuntimeError(f'GBSA numerical smoke check failed: {errors}')
    maps = Path('/proc/self/maps')
    if maps.is_file():
        libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                            if any(x in line for x in ('libcust_opapi', 'libopapi', 'libnnopbase'))})
        print('Loaded operator libraries:', json.dumps(libraries), flush=True)
    print('GBSA_SMALL_CASE PASS (not long sparse top-k, packed batches or full-model acceptance)')


if __name__ == '__main__':
    main()
