#!/usr/bin/env bash
set -eo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd -P)
source "$scripts/activate.sh"
set -u
: "${RUN_DIR:?set a new shared output directory}"
case "${1:-}" in
    config) export STEP5_RL_CONFIG_ONLY=1 ;;
    train) export STEP5_RL_CONFIG_ONLY=0 ;;
    *) exit 2 ;;
esac

export MODEL=/mnt/share/j00949742/step5/outputs/eval-ckpt100-hf
data=/mnt/share/z00988734/projects/step5-re/data/geo3k-rl-sft200-seed42-20261029
export TRAIN_FILES="$data/train.parquet" VAL_FILES="$data/validation.parquet"
export TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=2048
export ACTOR_MAX_TOKEN_LEN_PER_GPU=3072 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=3072
export MICRO_BATCH_SIZE=1 LOG_PROB_MICRO_BATCH_SIZE=1
export TOTAL_STEPS=100 TEST_FREQ=10 SAVE_FREQ=-1 LEARNING_RATE=1e-6
export ROLLOUT_TP_SIZE=4 MAX_NUM_SEQS=128 ROLLOUT_SLEEP_LEVEL=2 ROLLOUT_MEMORY=0.65
export RAY_ADDRESS=local VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_OP_EXPANSION_MODE=AIV STEP5_ROLLOUT_HCCL_MODE=AIV
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 STEP5_RANK0_EXPERT_INIT=1 STEP5_STREAM_EXPERT_WEIGHTS=0
export RL_INSIGHT_SERVER_URL=${RL_INSIGHT_SERVER_URL:-http://197.13.21.1:18080}
export PYTHONPATH="${RL_INSIGHT_PYTHON_ROOT:-/mnt/share/t00886549/step5/script/rl/rl-insight-py}:$PYTHONPATH"
if [[ $1 == train ]]; then
    npu-smi info | python3 -S -c 'import sys; assert sys.stdin.read().count("No running processes found")==8, "Expected eight idle NPUs"'
    curl --noproxy '*' --fail --silent --show-error --max-time 10 "$RL_INSIGHT_SERVER_URL/healthz"
fi

exec bash "$scripts/run_rl_rmpad.sh" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    '++actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false' \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.expert_parallel_size=1 \
    'trainer.logger=[console,rl_insight]' data.val_batch_size=32 \
    'global_profiler.steps=[]' actor_rollout_ref.actor.profiler.enable=false actor_rollout_ref.rollout.profiler.enable=false \
    "++trainer.actor_microbatch_profile_dir=$RUN_DIR/actor-profile-step3" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.RL_INSIGHT_SERVER_URL='$RL_INSIGHT_SERVER_URL'" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE="AIV"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.STEP5_ROLLOUT_HCCL_MODE="AIV"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY="1"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.STEP5_RANK0_EXPERT_INIT="1"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.STEP5_STREAM_EXPERT_WEIGHTS="0"'
