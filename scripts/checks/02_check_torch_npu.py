"""Check the selected Python environment and tiny FP32/BF16 NPU gradients."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-prefix', required=True)
    parser.add_argument('--device', type=int, default=0, help='Logical device after visibility filtering')
    args = parser.parse_args()
    print('Python:', sys.executable, flush=True)
    print('Prefix:', sys.prefix, flush=True)
    if Path(sys.prefix).resolve() != Path(args.expected_prefix).resolve():
        raise RuntimeError('Wrong Python environment: activate the expected Conda prefix first')

    import torch
    import torch_npu
    print('torch:', torch.__version__, torch.__file__)
    print('torch_npu:', torch_npu.__version__, torch_npu.__file__)
    if not torch.npu.is_available():
        raise RuntimeError('NPU unavailable')
    print('Visible NPUs:', torch.npu.device_count(), flush=True)
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.ones((16, 16), device=device, dtype=dtype, requires_grad=True)
        w = torch.ones((16, 16), device=device, dtype=dtype, requires_grad=True)
        y = x @ w
        y.sum().backward()
        torch.npu.synchronize()
        for name, value in (('output', y), ('dx', x.grad), ('dw', w.grad)):
            if value is None:
                raise RuntimeError(f'Missing {name}')
            torch.testing.assert_close(value.detach().cpu().float(), torch.full((16, 16), 16.0), rtol=0, atol=0)
        print(dtype, 'forward/backward PASS', flush=True)
    print('TORCH_NPU_BASIC PASS (single device only; HCCL and GBSA not tested)')


if __name__ == '__main__':
    main()
