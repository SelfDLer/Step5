#!/usr/bin/env bash
# Geo3K GRPO, A5-01-01 + A5-03-08. See start_rl_ray_2node.sh for Ray startup.
# bash run_rl_rmpad_2node.sh {64|80} [train|config|prepare]
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
clip_dir="$script_dir/../codex_clip"
experts=${1:-}
action=${2:-train}
case "$experts:$action" in
    64:train|64:config|64:prepare|80:train|80:config|80:prepare) ;;
    *) echo "Usage: bash $0 {64|80} [train|config|prepare]" >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || { echo "Unexpected arguments: ${*:3}" >&2; exit 2; }
source "$script_dir/activate.sh"
set -u

# ---- Model and output ----
model_name="Step-5-Preview-BF16-Geo3K200-top${experts}e"
export MODEL="$clip_dir/run200/$model_name"
if [[ $action == prepare ]]; then
    cd -- "$clip_dir"
    exec 9>"run200/.prepare-top${experts}e.lock"
    flock 9
    if [[ ! -d $MODEL ]]; then
        export CLIP_KEEP="$experts" CLIP_EXAMPLES=200
        python -c 'import sys; from probe_experts import select; select(int(sys.argv[1]))' "$experts"
        python -u export.py
    fi
fi
if [[ $action != config ]]; then
    if [[ ! -s "$MODEL/model.safetensors.index.json" ]]; then
        echo "Missing checkpoint. Run: bash $0 $experts prepare" >&2
        exit 1
    fi
    python - "$MODEL" "$experts" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
count = json.loads((model / "config.json").read_text())["text_config"]["moe_num_experts"]
if count != int(sys.argv[2]):
    sys.exit(f"{model}: expected {sys.argv[2]} experts, found {count}")
index = json.loads((model / "model.safetensors.index.json").read_text())
for shard in set(index["weight_map"].values()):
    if not (model / shard).is_file():
        sys.exit(f"Missing checkpoint shard: {shard}")
PY
    if [[ $action == prepare ]]; then echo "Checkpoint ready: $MODEL"; exit 0; fi
fi

# ---- Batch and memory ----
# 32 prompts x 8 responses per GRPO update. Preserve the 4096-token response budget.
export TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8
export ROLLOUT_TP_SIZE=8
export ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-6144}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-6144}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export TOTAL_STEPS=${TOTAL_STEPS:-30} SAVE_FREQ=${SAVE_FREQ:--1}
export RUN_DIR=${RUN_DIR:-"$script_dir/outputs/grpo-${experts}e-2n-bs32-${action}-$(date +%Y%m%d-%H%M%S)-$$"}

# ---- Topology and Ray ----
head_ip=197.13.21.1
worker_ip=197.13.44.1
export RAY_ADDRESS="$head_ip:${STEP5_RAY_PORT:-26379}"
if [[ $action == config ]]; then
    export STEP5_RL_CONFIG_ONLY=1
else
    export STEP5_RL_CONFIG_ONLY=0
    python - "$head_ip" "$worker_ip" <<'PY'
import os
import sys
import ray

ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False)
nodes = {n["NodeManagerAddress"]: n["Resources"].get("NPU", 0) for n in ray.nodes() if n["Alive"]}
if nodes != {ip: 8 for ip in sys.argv[1:]}:
    sys.exit(f"Expected both A5 nodes with 8 NPUs each, got {nodes}. See start_rl_ray_2node.sh")
ray.shutdown()
PY
fi

turbo=actor_rollout_ref.actor.fsdp_config.turbo_config.distributed
exec bash "$script_dir/run_rl_rmpad.sh" \
    trainer.nnodes=2 trainer.n_gpus_per_node=8 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=16 \
    "++$turbo.fully_shard_parallel_size=16" \
    "++$turbo.expert_parallel_size=8" \
    "++$turbo.expert_fully_shard_parallel_size=2" \
    "++ray_kwargs.ray_init.address=$RAY_ADDRESS" \
    ray_kwargs.ray_init.num_cpus=null \
    ++ray_kwargs.ray_init.object_store_memory=null
