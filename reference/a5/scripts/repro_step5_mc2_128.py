"""Run one copy on each of the two A5 nodes after sourcing scripts/activate.sh."""

import argparse
import faulthandler
import os
from pathlib import Path
import random
import threading
import time


def watch_replays(worker):
    """Observe existing captured graphs without inserting any device operations."""
    import torch

    graph_type = torch.npu.NPUGraph
    original = graph_type.replay
    count, last = 0, None

    def replay(graph):
        nonlocal count, last
        result = original(graph)
        count += 1
        last = time.monotonic()
        return result

    def report():
        reported = 0
        while True:
            time.sleep(5)
            if last is not None and count != reported and time.monotonic() - last > 60:
                reported = count
                print(f"REPLAY_STALL pid={os.getpid()} submitted={count} idle_s={time.monotonic()-last:.0f}", flush=True)
                faulthandler.dump_traceback()

    graph_type.replay = replay
    threading.Thread(target=report, daemon=True).start()


class ReplayObserver:
    def install_replay_observer(self):
        watch_replays(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank", type=int, choices=(0, 1), help="0 on A5-01-02, 1 on A5-03-08")
    parser.add_argument("--model", type=Path, help="Override the original 32-expert model directory")
    args = parser.parse_args()
    host = ("197.13.22.1", "197.13.44.1")[args.rank]
    os.environ.update(
        VLLM_DP_RANK=str(args.rank), VLLM_DP_RANK_LOCAL="0", VLLM_DP_SIZE="2",
        VLLM_DP_MASTER_IP="197.13.22.1", VLLM_DP_MASTER_PORT="29674",
        VLLM_HOST_IP=host, HCCL_IF_IP=host, HCCL_SOCKET_IFNAME="enp39s0f0",
        GLOO_SOCKET_IFNAME="enp39s0f0", TP_SOCKET_IFNAME="enp39s0f0",
        HCCL_OP_EXPANSION_MODE="AIV", STEP5_ROLLOUT_HCCL_MODE="AIV", HCCL_HOST_SOCKET_PORT_RANGE="auto",
        ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY="1",
    )
    from vllm import LLM, SamplingParams

    model = args.model or (Path(os.environ["STEP5_R3_ROOT"]) / "codex_clip/run200/Step-5-Preview-BF16-Geo3K200-top32e")
    llm = LLM(
        model=str(model), tensor_parallel_size=8, enable_expert_parallel=True, distributed_executor_backend="mp",
        dtype="bfloat16", seed=42, max_model_len=6144, max_num_batched_tokens=6144, max_num_seqs=128,
        gpu_memory_utilization=0.86, enforce_eager=False, enable_sleep_mode=True, enable_return_routed_experts=True,
        enable_chunked_prefill=True, enable_prefix_caching=False, disable_custom_all_reduce=True,
        mm_processor_cache_gb=0, compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        additional_config={"weight_nz_mode": 0},
        worker_extension_cls="repro_step5_mc2_128.ReplayObserver",
    )
    llm.collective_rpc("install_replay_observer", timeout=60)
    rng = random.Random(42 + args.rank)
    prompts = [{"prompt_token_ids": [rng.randrange(1000, 16000) for _ in range(rng.randrange(96, 640))]}
               for _ in range(256)]
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512, ignore_eos=True, detokenize=False)
    print(f"GENERATE dp={args.rank} requests={len(prompts)}", flush=True)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    lengths = [len(out.outputs[0].token_ids) for out in outputs]
    assert lengths == [512] * 256, "Incomplete generation"
    print(f"PASS dp={args.rank} tokens={sum(lengths)}", flush=True)


if __name__ == "__main__":
    main()
