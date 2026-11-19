#!/usr/bin/env bash
# Activate scripts/activate.sh before launching, as for run_rl_rmpad.sh.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MODEL=${MODEL:-/mnt/share/j00949742/step5/outputs/eval-ckpt200-hf}
export RUN_DIR=${RUN_DIR:-$script_dir/outputs/rl-geo3k-sft200-lora-$(date +%Y%m%d-%H%M%S)-$$}

# Step5 expert LoRA supports the native eager EP dispatchers.
export STEP5_EP_DISPATCHER=${STEP5_EP_DISPATCHER:-custom_native_eager_fused_forward}

exec bash "$script_dir/run_rl_rmpad.sh" \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    'actor_rollout_ref.model.target_modules=".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|moe\.experts\.(gate_up_proj|down_proj))"' \
    ++actor_rollout_ref.model.lora.merge=true \
    "$@"
