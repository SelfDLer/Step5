"""Replay the three captured Step5 routing tables. Launch with torchrun on 2 x 8 NPUs."""
import argparse
import json
import os
from pathlib import Path
import zipfile

import torch
import torch.distributed as dist
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--iterations", default=1000, type=int)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 16
    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"
    comm = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    with zipfile.ZipFile(args.evidence) as z:
        routes = [json.loads(z.read(f"evidence/routes-rank-{r}.json"))["layers"] for r in range(16)]
    x = torch.empty((16, 4096), device="npu", dtype=torch.bfloat16)
    weights = torch.full((16, 8), 0.125, device="npu")
    mask = torch.ones(16, dtype=torch.bool, device="npu")
    base = torch.arange(16, device="npu").add(rank * 16).to(x.dtype)
    # Each BF16 value 0..255 represents exactly one original rank and token.
    x.copy_(base[:, None])
    cases = []
    for layer in ("32", "33", "34"):
        ids = torch.tensor(routes[rank][layer]["ids"], device="npu", dtype=torch.int32)
        expected = []
        for expert in (2 * rank, 2 * rank + 1):
            for src in range(16):
                for token, row in enumerate(routes[src][layer]["ids"]):
                    expected.extend((src, token, slot) for slot, dest in enumerate(row) if dest == expert)
        cases.append((ids, expected))
    kwargs = dict(group_ep=comm, ep_world_size=16, ep_rank_id=rank, moe_expert_num=32,
                  group_tp=comm, tp_world_size=1, tp_rank_id=0, global_bs=0,
                  expert_shard_type=0, shared_expert_rank_num=0, x_active_mask=mask,
                  comm_alg="")

    def forward():
        checks = []
        for ids, expected in cases:
            out = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x, expert_ids=ids, quant_mode=0, expert_token_nums_type=0,
                expert_scales=weights, **kwargs)
            expanded, _, mapping, groups, counts, tp_counts, scales = out[:7]
            n = len(expected)
            checks.append((mapping.reshape(-1)[:3 * n].clone(), expanded[:n, 0].clone(), groups.clone()))
            torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expanded, expert_ids=ids, expert_scales=weights,
                assist_info_for_combine=mapping, ep_send_counts=counts, tp_send_counts=tp_counts,
                expand_scales=scales, comm_quant_mode=0, **kwargs)
        return checks

    def verify(outputs, iteration):
        for index, ((_, expected), (mapping, values, groups)) in enumerate(zip(cases, outputs)):
            actual = list(zip(*[iter(mapping.tolist())] * 3))
            assert groups.tolist()[-1] == len(expected)
            for pos, (want, got, value) in enumerate(zip(expected, actual, values.tolist())):
                if got != want or value != want[0] * 16 + want[1]:
                    raise RuntimeError(f"rank={rank} iter={iteration} layer={35+index} row={pos} "
                                       f"expected={want} got={got} data_tag={value}")

    verify(forward(), "warmup")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = forward()
    for iteration in range(args.iterations):
        graph.replay()
        verify(outputs, iteration)
    print(f"PASS rank={rank} three-layer replays={args.iterations}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
