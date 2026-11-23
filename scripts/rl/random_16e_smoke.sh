#!/usr/bin/env bash
# bash scripts/rl/random_16e_smoke.sh a5 run
# bash scripts/rl/random_16e_smoke.sh a3 {plan|generate|config|train|run}
# A3: activate the verified environment; set ASSETS, TRAIN_FILES and VAL_FILES.
# MODEL (new output checkpoint), SEED and RUN_DIR can be overridden on either host.
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=$(cd "$here/../.." && pwd -P)
platform=${1:-}
action=${2:-}
case $action in plan|generate|config|train|run) ;; *) echo 'Choose plan, generate, config, train or run.' >&2; exit 2 ;; esac
case $platform in
    a5)
        project=${A5_PROJECT:-/mnt/share/z00988734/projects/step5-minimal-r3-deepep-20261110}
        source "$project/scripts/activate.sh"
        ASSETS=${ASSETS:-/mnt/share/j00949742/step5/outputs/eval-ckpt100-hf}
        data=/mnt/share/z00988734/projects/step5-re/data/geo3k-rl-sft200-seed42-20261029
        TRAIN_FILES=${TRAIN_FILES:-$data/train.parquet}
        VAL_FILES=${VAL_FILES:-$data/validation.parquet}
        ;;
    a3) ;;
    *) echo 'Choose a5 (8 cards) or a3 (16 cards).' >&2; exit 2 ;;
esac
set -u
seed=${SEED:-42}
export MODEL=${MODEL:-$root/.local/random-16e-bf16-seed$seed}
if [[ $action == plan || $action == generate || ( $action == run && ! -e $MODEL ) ]]; then
    generator=(python "$here/make_random_step5_16e.py" --assets "${ASSETS:?Set ASSETS to the baseline tokenizer/config directory}" --output "$MODEL" --seed "$seed")
    if [[ $action == plan ]]; then exec "${generator[@]}" --plan; fi
    "${generator[@]}"
    if [[ $action == generate ]]; then exit 0; fi
fi
python - "$MODEL" "$seed" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
marker = root / 'random-init.json'
if not marker.exists():
    sys.exit(f'{root}: expected a generated random checkpoint; choose a fresh MODEL and run generate')
meta = json.loads(marker.read_text())
if meta.get('checkpoint_dtype') != 'BF16' or meta['seed'] != int(sys.argv[2]):
    sys.exit('Checkpoint dtype or seed differs. Choose a fresh MODEL directory.')
PY
export TRAIN_FILES=${TRAIN_FILES:?Set TRAIN_FILES to a local training parquet}
export VAL_FILES=${VAL_FILES:?Set VAL_FILES to a local validation parquet}
export RUN_DIR=${RUN_DIR:-$root/outputs/random-16e-$platform-$(date +%Y%m%d-%H%M%S)-$$}
export STEP5_EP_DISPATCHER=custom_native_eager_fused_forward STEP5_USE_TRITON_MC2_PERMUTE=false
export TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=2
export MICRO_BATCH_SIZE=1 LOG_PROB_MICRO_BATCH_SIZE=1
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=128
export ACTOR_MAX_TOKEN_LEN_PER_GPU=1152 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=1152
export TOTAL_STEPS=1 TEST_FREQ=-1 SAVE_FREQ=-1 ROLLOUT_SLEEP_LEVEL=2
launch() {
    if [[ $platform == a3 ]]; then
        EP_SIZE=16 ROLLOUT_TP_SIZE=8 bash "$here/run_rl_rmpad_a3_fp32.sh" "$1"
    else
        local config_only=0
        if [[ $1 == config ]]; then config_only=1; fi
        STEP5_RL_CONFIG_ONLY=$config_only RAY_ADDRESS=local ROLLOUT_TP_SIZE=4 \
            bash "$project/scripts/run_rl_rmpad.sh" \
                actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
                actor_rollout_ref.rollout.dtype=bfloat16
    fi
}
if [[ $action == run ]]; then
    launch config
    launch train
else
    launch "$action"
fi
