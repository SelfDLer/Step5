#!/usr/bin/env bash
# Start Ray on the head first, then run "start" on each worker. Run "train" on the head.
set -eo pipefail
action=${1:-config}
case "$action" in
    start|config|train) ;;
    *) echo "Usage: bash $0 {start|config|train}" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Unexpected arguments: ${*:2}" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
source "$script_dir/activate.sh"
set -u
export MAX_NUM_SEQS=64 ENFORCE_EAGER=false
: "${RUN_DIR:?set a shared output directory for this run}"
export RUN_DIR
export STEP5_PROFILE_ACTOR_MICROBATCH=0 STEP5_ROLLOUT_HCCL_MODE=AIV

head_ip=197.13.21.1
interface=enp39s0f0
port=${STEP5_RAY_PORT:-29652}
export RAY_ADDRESS="$head_ip:$port"
# Bound reverse lookup delays when the TCPStore accepts 128 workers.
export RES_OPTIONS=${RES_OPTIONS:-"timeout:1 attempts:1"}
export VLLM_WORKER_MULTIPROC_METHOD=spawn STEP5_EFSDP_CAST_FORWARD_INPUTS=true
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export PYTHONPATH="$script_dir/python:${PYTHONPATH:-}"
export MODEL=/mnt/weight/Step-5-Preview-BF16
data_dir=/mnt/share/z00988734/projects/step5-re/data/geo3k-rl-handoff-20261029
export TRAIN_FILES="$data_dir/train.parquet" VAL_FILES="$data_dir/validation.parquet"

if [[ $action != config ]]; then
    local_ip=$(ip -o -4 addr show dev "$interface" | awk '{split($4,a,"/"); print a[1]}' |
        awk -F. '$1==197 && $2==13 && ($3>=21 && $3<=36) && $4==1')
    [[ -n $local_ip && $local_ip != *$'\n'* ]] || { echo "Cannot identify A5 training address on $interface" >&2; exit 1; }
    export HCCL_IF_IP="$local_ip" VLLM_HOST_IP="$local_ip"
    export HCCL_SOCKET_IFNAME="$interface" GLOO_SOCKET_IFNAME="$interface" TP_SOCKET_IFNAME="$interface"
    export HCCL_OP_EXPANSION_MODE=AIV ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
if [[ $action == start ]]; then
    if ps -C raylet -o stat= | awk '$1 !~ /^Z/ {live=1} END {exit !live}'; then
        echo "A Ray node is already live on $local_ip; check ray status before starting another." >&2
        exit 1
    fi
    export RAY_TMPDIR="/tmp/s5rl-$port"
    if [[ $local_ip == "$head_ip" ]]; then
        role=(--head "--port=$port" --include-dashboard=false "--temp-dir=$RAY_TMPDIR")
    else
        role=("--address=$RAY_ADDRESS")
    fi
    if [[ ${STEP5_RAY_BLOCK:-0} == 1 ]]; then role+=(--block); fi
    unset RAY_ADDRESS
    exec ray start --min-worker-port=32000 --max-worker-port=32999 "--node-ip-address=$local_ip" --num-cpus=32 '--resources={"NPU":8}' \
        --object-store-memory=8589934592 --disable-usage-stats "${role[@]}"
fi

export TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8 LEARNING_RATE=1e-6
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=2048
export TOTAL_STEPS=200 TEST_FREQ=-1 SAVE_FREQ=-1
export MICRO_BATCH_SIZE=1 LOG_PROB_MICRO_BATCH_SIZE=1
export ACTOR_MAX_TOKEN_LEN_PER_GPU=6144 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=6144
export ROLLOUT_SLEEP_LEVEL=2 STEP5_STREAM_EXPERT_WEIGHTS=0
export RL_INSIGHT_SERVER_URL=${RL_INSIGHT_SERVER_URL:-http://197.13.21.1:18080}
export PYTHONPATH="${RL_INSIGHT_PYTHON_ROOT:-/mnt/share/t00886549/step5/script/rl/rl-insight-py}:$PYTHONPATH"
export ROLLOUT_TP_SIZE=8 ROLLOUT_MEMORY=${ROLLOUT_MEMORY:-0.86}

python - "$MODEL" "$TRAIN_FILES" "$VAL_FILES" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

model, train, val = map(Path, sys.argv[1:])
experts = json.loads((model / "config.json").read_text())["text_config"]["moe_num_experts"]
if experts != 352:
    raise SystemExit(f"Expected the full 352-expert model, found {experts}")
shards = set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values())
missing = [name for name in shards if not (model / name).is_file()]
if missing:
    raise SystemExit(f"Missing model shards: {missing}")
for path, expected in ((train, 2101), (val, 300)):
    count = pq.ParquetFile(path).metadata.num_rows
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} rows, found {count}")
PY

if [[ $action == config ]]; then
    export STEP5_RL_CONFIG_ONLY=1
else
    export STEP5_RL_CONFIG_ONLY=0
    [[ $local_ip == "$head_ip" ]] || { echo "Run train on A5-01-01." >&2; exit 1; }
    python - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False)
try:
    expected = {f"197.13.{i}.1": 8 for i in range(21, 37)}
    live = [n for n in ray.nodes() if n["Alive"]]
    found = {n["NodeManagerAddress"]: n["Resources"].get("NPU", 0) for n in live}
    if found != expected or len(live) != len(expected):
        raise SystemExit(f"Expected all 16 A5 nodes with eight NPUs each, found {found}")
finally:
    ray.shutdown()
PY
fi

# The shared checkout supplies pageable NPU Adam transfers, rank-0 state loading,
# the CSA slot fix, and the base launcher's ByteLevel tokenizer adapter.
distributed=actor_rollout_ref.actor.fsdp_config.turbo_config.distributed
profile_args=('global_profiler.steps=[]' actor_rollout_ref.actor.profiler.enable=false actor_rollout_ref.rollout.profiler.enable=false)
exec bash "$script_dir/run_rl_rmpad.sh" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE="AIV"' \
    trainer.nnodes=16 trainer.n_gpus_per_node=8 \
    data.val_batch_size=32 data.train_max_samples=-1 data.val_max_samples=-1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=128 \
    "++$distributed.fully_shard_parallel_size=128" \
    "++$distributed.expert_parallel_size=8" \
    "++$distributed.expert_fully_shard_parallel_size=16" \
    "++ray_kwargs.ray_init.address=$RAY_ADDRESS" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.RES_OPTIONS='$RES_OPTIONS'" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY="1"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.STEP5_RANK0_EXPERT_INIT="1"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.STEP5_STREAM_EXPERT_WEIGHTS="0"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.STEP5_ROLLOUT_HCCL_MODE="AIV"' \
    ray_kwargs.ray_init.num_cpus=null ++ray_kwargs.ray_init.object_store_memory=null \
    actor_rollout_ref.rollout.data_parallel_size=4 \
    actor_rollout_ref.rollout.expert_parallel_size=32 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    'trainer.logger=[console,rl_insight]' \
    "++ray_kwargs.ray_init.runtime_env.env_vars.RL_INSIGHT_SERVER_URL='$RL_INSIGHT_SERVER_URL'" \
    "${profile_args[@]}"
