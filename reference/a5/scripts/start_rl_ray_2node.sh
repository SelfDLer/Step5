#!/usr/bin/env bash
# Run "bash start_rl_ray_2node.sh head" on A5-01-01, then "worker" on A5-03-08.
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
source "$script_dir/activate.sh"
set -u
head_ip=197.13.21.1
worker_ip=197.13.44.1
port=${STEP5_RAY_PORT:-26379}
case "${1:-}" in
    head) local_ip=$head_ip; role_args=(--head "--port=$port" --include-dashboard=false "--temp-dir=/tmp/s5rl-$port") ;;
    worker) local_ip=$worker_ip; role_args=("--address=$head_ip:$port") ;;
    *) echo "Usage: bash $0 {head|worker}" >&2; exit 2 ;;
esac
interface=enp39s0f0
if ! ip -o -4 addr show dev "$interface" | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$local_ip"; then
    echo "Expected $local_ip on $interface for role $1." >&2
    exit 1
fi
# Exited Ray children may remain zombies until the container init reaps them.
if ps -C raylet -o stat= | awk '$1 !~ /^Z/ {live=1} END {exit !live}'; then
    echo "A Ray node is already running here. Check its cluster before starting another." >&2
    exit 1
fi
unset RAY_ADDRESS
export HCCL_IF_IP=$local_ip VLLM_HOST_IP=$local_ip
export HCCL_SOCKET_IFNAME=$interface GLOO_SOCKET_IFNAME=$interface TP_SOCKET_IFNAME=$interface
export HCCL_OP_EXPANSION_MODE=AIV ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn STEP5_EFSDP_CAST_FORWARD_INPUTS=true
export RAY_TMPDIR="/tmp/s5rl-$port"
exec ray start "--node-ip-address=$local_ip" --num-cpus=32 '--resources={"NPU":8}' \
    --object-store-memory=8589934592 --disable-usage-stats "${role_args[@]}"
