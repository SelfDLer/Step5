#!/usr/bin/env bash
# Step5 GRPO: SFT200 / seed42 train100 evaluation, 30 updates.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# Activate the project environment in the calling shell before running this script.
# Select the four project checkouts through scripts/python/sitecustomize.py.
export PYTHONPATH="$script_dir/python${PYTHONPATH:+:$PYTHONPATH}"
# The pinned runtime has no Pydantic plugins. Avoid scanning the shared venv in each worker.
export PYDANTIC_DISABLE_PLUGINS=${PYDANTIC_DISABLE_PLUGINS-__all__}
export RAY_ADDRESS=${RAY_ADDRESS:-local} PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export STEP5_EFSDP_CAST_FORWARD_INPUTS=${STEP5_EFSDP_CAST_FORWARD_INPUTS:-true}

# ---- Frequently changed parameters (also accept environment overrides) ----
ROOT=${ROOT:-/mnt/share/z00988734/projects/step5-re}
MODEL=${MODEL:-/mnt/share/j00949742/step5/outputs/eval-ckpt100-hf}
TRAIN_FILES=${TRAIN_FILES:-$ROOT/data/geo3k-rl-sft200-seed42-20261029/train.parquet}
VAL_FILES=${VAL_FILES:-$ROOT/data/geo3k-rl-sft200-seed42-20261029/validation.parquet}
RUN_DIR=${RUN_DIR:-$script_dir/outputs/rl-geo3k-sft200-ckpt100-$(date +%Y%m%d-%H%M%S)-$$}

STEP5_EP_DISPATCHER=${STEP5_EP_DISPATCHER:-custom_native_deepep_mc2_forward}
if [[ $STEP5_EP_DISPATCHER == custom_native_deepep_mc2_forward ]]; then
    STEP5_USE_TRITON_MC2_PERMUTE=${STEP5_USE_TRITON_MC2_PERMUTE:-true}
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
# Dynamic batches are packed by the token budgets below; their sequence counts vary.
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-8}
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-${MAX_TOKEN_LEN_PER_GPU:-8192}}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_MEMORY=${ROLLOUT_MEMORY:-0.65}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-4}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}
FREEZE_VISION_PROJECTOR=${FREEZE_VISION_PROJECTOR:-true}
TOTAL_STEPS=${TOTAL_STEPS:-30}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
IGNORE_EOS=${IGNORE_EOS:-false}
CONFIG_ONLY=${STEP5_RL_CONFIG_ONLY:-0}

# ---- Topology and output ----
N_GPUS=8
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
mkdir -p "$RUN_DIR"
MODEL=$(python -S "$script_dir/rl_bytelevel_model.py" "$MODEL" "$RUN_DIR")
RUN_DIR=$(realpath "$RUN_DIR")
RAY_TMPDIR=$(mktemp -d /tmp/s5rl-XXXXXX)
export RAY_TMPDIR

# ---- FSDPTurbo settings: inline Hydra overrides, no generated config file ----
turbo=actor_rollout_ref.actor.fsdp_config.turbo_config
turbo_args=(
    "++$turbo.distributed.data_parallel_size=1"
    "++$turbo.distributed.fully_shard_parallel_size=$N_GPUS"
    "++$turbo.distributed.tensor_parallel_size=1"
    "++$turbo.distributed.kv_allgather_parallel_size=1"
    "++$turbo.distributed.kcp_parallel_size=1"
    "++$turbo.distributed.ulysses_parallel_size=1"
    "++$turbo.distributed.expert_parallel_size=8"
    "++$turbo.distributed.expert_fully_shard_parallel_size=1"
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
    "data.train_files=$TRAIN_FILES"
    "data.val_files=$VAL_FILES"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.shuffle=true
    data.seed=42
    data.filter_overlong_prompts=true
)

actor_args=(
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false
    "actor_rollout_ref.model.path=$MODEL"
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=$N_GPUS"
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    actor_rollout_ref.actor.fsdp_config.offload_policy=false
    actor_rollout_ref.actor.fsdp_config.full_determinism=false
    actor_rollout_ref.actor.fsdp_config.seed=42
)

rollout_args=(
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
    "reward.reward_manager.module.path=$script_dir/step5_geo3k_reward.py"
    algorithm.adv_estimator=grpo
    algorithm.filter_groups.enable=false
    'trainer.logger=[console]'
    "trainer.n_gpus_per_node=$N_GPUS"
    trainer.nnodes=1
    trainer.device=npu
    trainer.use_v1=true
    trainer.val_before_train=false
    trainer.log_val_generations=3
    "trainer.rollout_data_dir=$RUN_DIR/rollouts"
    "trainer.validation_data_dir=$RUN_DIR/validation"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$TOTAL_STEPS"
    trainer.balance_batch=false
    trainer.resume_mode=disable
    "trainer.default_local_dir=$RUN_DIR/checkpoints"
    trainer.project_name=step5-rl
    "trainer.experiment_name=$(basename "$RUN_DIR")"
    ray_kwargs.ray_init.num_cpus=32
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$PYTHONPATH'"
    ++ray_kwargs.ray_init.object_store_memory=1073741824
    transfer_queue.backend.SimpleStorage.total_storage_size=1024
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
)

# Ray workers may import packages from the driver working directory.
cd -- "${script_dir%/*}"
command=(python -P -m verl.trainer.main_ppo
    --config-name ppo_trainer
    "${turbo_args[@]}"
    "${data_args[@]}"
    "${actor_args[@]}"
    "${rollout_args[@]}"
    "${trainer_args[@]}"
    "$@")
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
echo "Artifacts: $RUN_DIR"
if [[ "$CONFIG_ONLY" == 1 ]]; then
    "${command[@]}" --cfg job --resolve > "$RUN_DIR/config.txt" 2> "$RUN_DIR/config.log"
    echo 'Configuration exported; training was not started.'
    exit 0
fi
"${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"

