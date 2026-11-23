#!/usr/bin/env bash
# Step5 GRPO, single A3 node (16 x 64 GB). Run with the verified Conda/CANN environment.
# Set the absolute paths listed in the RL README before invoking this script.
# Uses the A5 GRPO reward manager (Geo3K uses mathruler==0.1.0).
# For performance comparisons use the same model, samples, steps, reward and logger.
set -euo pipefail
case ${1:-} in smoke|config|train) mode=$1; shift ;; *) echo 'Usage: bash "$RL_SCRIPT" {smoke|config|train} [Hydra overrides ...]' >&2; exit 2 ;; esac
paths=(MODEL TRAIN_FILES VAL_FILES RUN_DIR BYTELEVEL_HELPER REWARD_MODULE)
if [[ $mode != config ]]; then paths+=(RAY_TMPDIR); fi
for name in "${paths[@]}"; do
    [[ ${!name:-} == /* ]] || { echo "Set $name to an absolute path." >&2; exit 2; }
done
for file in "$BYTELEVEL_HELPER" "$REWARD_MODULE"; do
    [[ -f $file && -r $file ]] || { echo "Cannot read $file" >&2; exit 2; }
done
MODEL=$(realpath -- "$MODEL")
TRAIN_FILES=$(realpath -- "$TRAIN_FILES")
VAL_FILES=$(realpath -- "$VAL_FILES")
[[ -r "$MODEL/config.json" && -f "$TRAIN_FILES" && -r "$TRAIN_FILES" && -f "$VAL_FILES" && -r "$VAL_FILES" ]] || { echo 'Check the HF config and both parquet paths.' >&2; exit 1; }

# ---- A3 topology ----
# A5: 8 ranks, actor FSDP8/EP8, rollout 2 x TP4. A3: 16 ranks,
# actor FSDP16/EP16, rollout 2 x TP8 to reduce per-rank weight memory.
N_GPUS=16
EP_SIZE=${EP_SIZE:-16}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-8}
for size in "$EP_SIZE" "$ROLLOUT_TP_SIZE"; do
    [[ $size =~ ^(1|2|4|8|16)$ ]] || { echo 'EP_SIZE and ROLLOUT_TP_SIZE must divide 16.' >&2; exit 2; }
done
EXPERT_FSDP_SIZE=$((N_GPUS / EP_SIZE))
export STEP5_EP_DISPATCHER=custom_native_eager_fused_forward
export STEP5_USE_TRITON_MC2_PERMUTE=false
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
# Match the working single-node runtime: Ray workers inherit the node's NPU list.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# The baseline uses normal Ascend attention backend selection on rollout.
export VLLM_STEP4_ENABLE_FA3=0
python - "$MODEL/config.json" "$EP_SIZE" "$ROLLOUT_TP_SIZE" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    model = json.load(stream)
cfg = model.get('text_config', model)
ep, tp = map(int, sys.argv[2:])
for key, divisor in [('moe_num_experts', ep), ('num_attention_heads', tp), ('num_sliding_attention_heads', tp)]:
    n = cfg.get(key)
    if isinstance(n, int) and n % divisor:
        raise SystemExit(f'{key}={n} is incompatible with parallel size {divisor}')
PY
if [[ $mode != config ]]; then
    python -P - <<'PY'
import os
from verl.utils.device import is_npu_available
if not is_npu_available:
    raise SystemExit("verl cannot see an NPU in the RL driver. Check the active CANN/torch_npu "
                     f"environment, TORCH_DEVICE_BACKEND_AUTOLOAD={os.getenv('TORCH_DEVICE_BACKEND_AUTOLOAD')!r}, "
                     f"ASCEND_RT_VISIBLE_DEVICES={os.getenv('ASCEND_RT_VISIBLE_DEVICES')!r}.")
import torch
n = torch.npu.device_count()
if n != 16:
    raise SystemExit(f"expected 16 visible NPUs, got {n}")
PY
fi

# ---- A5 sft100 FP32 run settings ----
if [[ $mode == smoke ]]; then
    TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=2
    MICRO_BATCH_SIZE=1 LOG_PROB_MICRO_BATCH_SIZE=1
    MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=128
    ACTOR_MAX_TOKEN_LEN_PER_GPU=1152 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=1152
    TOTAL_STEPS=1 TEST_FREQ=-1 SAVE_FREQ=-1
fi
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-1}
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-3072}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-3072}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_MEMORY=${ROLLOUT_MEMORY:-0.65}
ROLLOUT_SLEEP_LEVEL=${ROLLOUT_SLEEP_LEVEL:-2}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}
FREEZE_VISION_PROJECTOR=${FREEZE_VISION_PROJECTOR:-true}
TOTAL_STEPS=${TOTAL_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}
IGNORE_EOS=${IGNORE_EOS:-false}
TRAINER_LOGGERS='[console]'
if [[ -n ${RL_INSIGHT_SERVER_URL:-} ]]; then TRAINER_LOGGERS='[console,rl_insight]'; fi

# ---- Worker environment and output ----
export RAY_ADDRESS=local PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYDANTIC_DISABLE_PLUGINS=${PYDANTIC_DISABLE_PLUGINS:-__all__}
export STEP5_EFSDP_CAST_FORWARD_INPUTS=true VLLM_BATCH_INVARIANT=0 VERL_FULL_DETERMINISM=0
export HCCL_OP_EXPANSION_MODE=AIV STEP5_ROLLOUT_HCCL_MODE=AIV
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 STEP5_RANK0_EXPERT_INIT=1 STEP5_STREAM_EXPERT_WEIGHTS=0
mkdir -p -- "$RUN_DIR"
RUN_DIR=$(realpath -- "$RUN_DIR")
if [[ $mode != config && -e "$RUN_DIR/train.log" ]]; then echo "Choose a fresh RUN_DIR; train.log exists in $RUN_DIR" >&2; exit 2; fi
MODEL=$(python -S "$BYTELEVEL_HELPER" "$MODEL" "$RUN_DIR")
if [[ $mode != config ]]; then mkdir -p -- "$RAY_TMPDIR"; export RAY_TMPDIR; fi
# Quote string values for Hydra. Runtime env_vars must remain strings (e.g. "0").
hq() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}
worker_args=()
for name in PYTHONPATH RL_INSIGHT_SERVER_URL RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES VLLM_WORKER_MULTIPROC_METHOD VLLM_STEP4_ENABLE_FA3 HCCL_OP_EXPANSION_MODE \
    STEP5_ROLLOUT_HCCL_MODE VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY STEP5_RANK0_EXPERT_INIT \
    STEP5_STREAM_EXPERT_WEIGHTS STEP5_EFSDP_CAST_FORWARD_INPUTS VERL_FULL_DETERMINISM VLLM_BATCH_INVARIANT; do
    if [[ -n ${!name:-} ]]; then worker_args+=("++ray_kwargs.ray_init.runtime_env.env_vars.$name=$(hq "${!name}")"); fi
done

# ---- FSDPTurbo settings: inline Hydra overrides, no generated config file ----
turbo=actor_rollout_ref.actor.fsdp_config.turbo_config
turbo_args=(
    "++$turbo.distributed.data_parallel_size=1"
    "++$turbo.distributed.fully_shard_parallel_size=$N_GPUS"
    "++$turbo.distributed.tensor_parallel_size=1"
    "++$turbo.distributed.kv_allgather_parallel_size=1"
    "++$turbo.distributed.kcp_parallel_size=1"
    "++$turbo.distributed.ulysses_parallel_size=1"
    "++$turbo.distributed.expert_parallel_size=$EP_SIZE"
    "++$turbo.distributed.expert_fully_shard_parallel_size=$EXPERT_FSDP_SIZE"
    "++$turbo.distributed.expert_data_parallel_size=1"
    "++$turbo.distributed.fsdp_plan.ignored_modules=[]"
    # Escape literal braces in module patterns for Hydra's override parser.
    "++$turbo.distributed.fsdp_plan.apply_modules="'{model.embed_tokens:{},model.layers.\{*\}:{},model.layers.\{*\}.self_attn.q_norm:{},model.layers.\{*\}.self_attn.k_norm:{},model.layers.\{*\}.self_attn.sparse_indexer_q_norm:{},model.layers.\{*\}.self_attn.sparse_indexer_k_norm:{},model.layers.\{*\}.self_attn.sparse_indexer_w:{},model.layers.\{*\}.input_layernorm:{},model.layers.\{*\}.post_attention_layernorm:{},model.norm:{},lm_head:{},vision_model.transformer.resblocks.\{*\}:{},vit_large_projector:{}}'
    "++$turbo.distributed.fsdp_plan.cpu_offload=false"
    "++$turbo.distributed.fsdp_plan.param_dtype=bf16"
    "++$turbo.distributed.fsdp_plan.reduce_dtype=fp32"
    "++$turbo.distributed.fsdp_plan.output_dtype=bf16"
    "++$turbo.distributed.fsdp_plan.cast_forward_inputs=true"
    "++$turbo.distributed.fsdp_plan.num_to_forward_prefetch=0"
    "++$turbo.distributed.fsdp_plan.num_to_backward_prefetch=0"
    "++$turbo.distributed.fsdp_plan.hook_modules="'[model.layers.\{*\}]'
    "++$turbo.distributed.fsdp_plan.fsdp_implementation=native"
    "++$turbo.distributed.tp_plan={colwise_parallel:[],rowwise_parallel:[],sequence_parallel:[]}"
    "++$turbo.distributed.ep_plan.apply_modules="'[model.layers.\{*\}.moe.experts]'
    "++$turbo.distributed.ep_plan.apply_efsdp_modules="'[model.layers.\{*\}.moe.experts]'
    "++$turbo.distributed.ep_plan.dispatcher=$STEP5_EP_DISPATCHER"
    "++$turbo.distributed.ep_plan.expert_num_to_forward_prefetch=0"
    "++$turbo.distributed.ep_plan.expert_num_to_backward_prefetch=0"
    "++$turbo.distributed.cp_plan={ulysses_function_patches:[],kcp_function_patches:[],kv_allgather_function_patches:[],loss_function_patches:[]}"
    "++$turbo.memory.recompute=true"
    "++$turbo.memory.recompute_plan="'[model.layers.\{*\},vision_model.transformer.resblocks.\{*\}]'
    "++$turbo.memory.chunk_loss=false"
    "++$turbo.memory.chunk_loss_size=512"
)

# ---- Hydra overrides grouped by component; command-line overrides go last ----
data_args=(
    "data.train_files=$(hq "$TRAIN_FILES")"
    "data.val_files=$(hq "$VAL_FILES")"
    data.val_batch_size=32
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.shuffle=true
    data.seed=42
    data.filter_overlong_prompts=true
)

actor_args=(
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false
    "actor_rollout_ref.model.path=$(hq "$MODEL")"
    actor_rollout_ref.model.external_lib=verl.models.transformers.step5
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.model.enable_gradient_checkpointing=false
    "++actor_rollout_ref.model.override_config.freeze_vision_encoder=$FREEZE_VISION_ENCODER"
    "++actor_rollout_ref.model.override_config.freeze_vision_projector=$FREEZE_VISION_PROJECTOR"
    ++actor_rollout_ref.model.override_config.attn_implementation=eager
    ++actor_rollout_ref.model.override_config.text_config.use_fused_attention=true
    ++actor_rollout_ref.model.override_config.text_config.use_dsa_all_select_fast_path=true
    ++actor_rollout_ref.model.override_config.text_config.use_dsa_vectorized_aux_loss=true
    ++actor_rollout_ref.model.override_config.text_config.enable_dsa_aux_loss=false
    ++actor_rollout_ref.model.override_config.text_config.use_dsa_hybrid_prefix_fast_path=true
    ++actor_rollout_ref.model.override_config.text_config.use_gbsa_full_sequence=true
    ++actor_rollout_ref.model.override_config.text_config.use_gbsa_long_tail=false
    ++actor_rollout_ref.model.override_config.text_config.use_triton_qk_norm_rope=true
    ++actor_rollout_ref.model.override_config.text_config.use_triton_indexer_norm_rope=true
    "++actor_rollout_ref.model.override_config.text_config.use_triton_mc2_permute=${STEP5_USE_TRITON_MC2_PERMUTE:-false}"
    "++actor_rollout_ref.model.override_config.text_config.use_triton_swiglu=${STEP5_USE_TRITON_SWIGLU:-true}"
    '++actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false'
    actor_rollout_ref.actor.profiler.enable=false
    actor_rollout_ref.actor.strategy=fsdp_turbo
    ++actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz=true
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=false
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=false
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=$N_GPUS"
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    actor_rollout_ref.actor.fsdp_config.offload_policy=false
    actor_rollout_ref.actor.fsdp_config.full_determinism=false
    actor_rollout_ref.actor.fsdp_config.seed=42
)

rollout_args=(
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.disable_log_stats=false
    actor_rollout_ref.rollout.profiler.enable=false
    actor_rollout_ref.rollout.full_determinism=false
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.vllm_sleep_level=${ROLLOUT_SLEEP_LEVEL:-null}"
    actor_rollout_ref.rollout.enable_rollout_routing_replay=true
    ++actor_rollout_ref.rollout.allow_routing_replay_version_override=true
    actor_rollout_ref.rollout.calculate_log_probs=true
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.9
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEMORY"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    actor_rollout_ref.rollout.enable_prefix_caching=false
    actor_rollout_ref.rollout.load_format=auto
    "actor_rollout_ref.rollout.ignore_eos=$IGNORE_EOS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0
)

trainer_args=(
    reward.reward_manager.source=importlib
    reward.reward_manager.name=Step5Geo3KRewardManager
    "reward.reward_manager.module.path=$(hq "$REWARD_MODULE")"
    algorithm.adv_estimator=grpo
    algorithm.filter_groups.enable=false
    "trainer.logger=$TRAINER_LOGGERS"
    'global_profiler.steps=[]'
    "++trainer.actor_microbatch_profile_dir=$(hq "$RUN_DIR/actor-profile-step3")"
    "trainer.n_gpus_per_node=$N_GPUS"
    trainer.nnodes=1
    trainer.device=npu
    trainer.use_v1=true
    trainer.val_before_train=false
    trainer.log_val_generations=3
    "trainer.rollout_data_dir=$(hq "$RUN_DIR/rollouts")"
    "trainer.validation_data_dir=$(hq "$RUN_DIR/validation")"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$TOTAL_STEPS"
    trainer.balance_batch=false
    trainer.resume_mode=disable
    "trainer.default_local_dir=$(hq "$RUN_DIR/checkpoints")"
    trainer.project_name=step5-rl
    "trainer.experiment_name=$(basename "$RUN_DIR")"
    "${worker_args[@]}"
    ++ray_kwargs.ray_init.object_store_memory=1073741824
    transfer_queue.backend.SimpleStorage.total_storage_size=1024
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
)

cd -- "$RUN_DIR"
command=(python -P -m verl.trainer.main_ppo --config-name ppo_trainer
    "${turbo_args[@]}" "${data_args[@]}" "${actor_args[@]}" "${rollout_args[@]}" "${trainer_args[@]}" "$@")
printf 'A3: %s NPUs, actor FSDP=%s EP=%s expert-FSDP=%s, rollout TP=%s (%s replicas), dispatcher=%s\n' \
    "$N_GPUS" "$N_GPUS" "$EP_SIZE" "$EXPERT_FSDP_SIZE" "$ROLLOUT_TP_SIZE" "$((N_GPUS / ROLLOUT_TP_SIZE))" "$STEP5_EP_DISPATCHER"
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
echo "Artifacts: $RUN_DIR"
if [[ $mode != train ]]; then
    "${command[@]}" --cfg job --resolve > "$RUN_DIR/config.txt" 2> "$RUN_DIR/config.log" || { cat "$RUN_DIR/config.log" >&2; exit 1; }
fi
if [[ $mode != config ]]; then
    "${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
fi
