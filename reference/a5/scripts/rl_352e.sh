#!/usr/bin/env bash
# Run locally on each participating host. All hosts use the same shared output directory.
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd -P)
project=$(dirname -- "$scripts")
action=${1:-}
[[ $# == 2 && $2 == /* && $2 != *[[:space:]]* ]] || {
    echo "Usage: bash $0 {config|verify|start|train|single-config|single-train|stop} /absolute/shared/run_dir" >&2
    exit 2
}
export RUN_DIR
RUN_DIR=$(realpath -m -- "$2")
unit=step5-352e-$(printf '%s' "$RUN_DIR" | sha256sum | cut -c1-16)
native=${STEP5_352E_NATIVE_DIR:-$project/outputs/352e-native}
base="$scripts/run_rl_352e_16node.sh"
case "$action" in
    config) exec bash "$base" config ;;
    single-config) exec bash "$scripts/run_rl_sft100_1node.sh" config ;;
    sft100-driver) exec bash "$scripts/run_rl_sft100_1node.sh" train >> "$RUN_DIR/driver.log" 2>&1 ;;
    stop)
        for role in driver ray; do
            if systemctl is-active --quiet "$unit-$role"; then systemctl stop "$unit-$role"; fi
        done
        exit 0 ;;
    driver) exec bash "$base" train >> "$RUN_DIR/driver.log" 2>&1 ;;
    ray) export STEP5_RAY_BLOCK=1; exec bash "$base" start >> "$RUN_DIR/ray-$(hostname).log" 2>&1 ;;
    verify|start|train|single-train) ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac

set +u
source "$scripts/activate.sh"
set -u
site=$("$STEP5_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
ep="$site/deep_ep_experimental/ep_gmm_fused"
ext=_C$("$STEP5_PYTHON" -S -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
kernel_dir="$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/moe_distribute_dispatch_v3_apt"
kernel=MoeDistributeDispatchV3_e92e3b86d5d6432a166d18733d3c3613
sources=("$native/deepep/$ext" "$project/src/DeepEP/experiments/ep_gmm_fused/python/deep_ep_experimental/ep_gmm_fused/ops.py"
         "$native/dispatch/$kernel.o" "$native/dispatch/$kernel.json")
targets=("$ep/$ext" "$ep/ops.py" "$kernel_dir/$kernel.o" "$kernel_dir/$kernel.json")
opts=(--collect -p KillMode=control-group -p LimitNOFILE=65536)
checks=()
for index in "${!sources[@]}"; do
    src=${sources[index]}
    target=${targets[index]}
    [[ -s $src && -s $target ]] || { echo "Missing $src or $target" >&2; exit 1; }
    opts+=(-p "BindReadOnlyPaths=$src:$target")
    checks+=("$src" "$target")
done
if [[ -e /etc/hccl_rootinfo.json ]] && ! python3 -m json.tool /etc/hccl_rootinfo.json >/dev/null 2>&1; then
    python3 -c 'import json; assert json.load(open("/etc/hccl_rootinfo.json_bk"))["rank_count"]==8'
    opts+=(-p BindReadOnlyPaths=/etc/hccl_rootinfo.json_bk:/etc/hccl_rootinfo.json)
fi
case "$action" in
    verify)
        exec systemd-run --wait --pipe "${opts[@]}" /bin/bash -ec \
            'while (( $# )); do cmp -- "$1" "$2"; shift 2; done; echo "Four worker files verified, fd limit $(ulimit -Sn)"' _ "${checks[@]}" ;;
    start)
        role=ray ;;
    train|single-train)
        [[ ! -e "$RUN_DIR/train.log" && ! -e "$RUN_DIR/driver.log" ]] || {
            echo "Run already has training output: $RUN_DIR" >&2; exit 1;
        }
        role=driver ;;
esac
mkdir -p -- "$RUN_DIR"
for key in STEP5_RAY_PORT RL_INSIGHT_PYTHON_ROOT RL_INSIGHT_SERVER_URL ROLLOUT_MEMORY; do
    if [[ -v $key ]]; then opts+=(--setenv="$key=${!key}"); fi
done
entry=$role
if [[ $action == single-train ]]; then entry=sft100-driver; fi
exec systemd-run "${opts[@]}" --unit="$unit-$role" /bin/bash "$scripts/rl_352e.sh" "$entry" "$RUN_DIR"
