#!/usr/bin/env bash
# A3, 1 node x 16 NPUs. Run from the verified Step5 environment.
# bash scripts/rl/random_16e_smoke.sh {init|plan|generate|config|train|run}
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=$(cd "$here/../.." && pwd -P)
if [[ ${1:-} == a3 ]]; then shift; fi
action=${1:-}
settings="$root/.local/random-16e-a3.env"
case $action in
    init)
        mkdir -p "$root/.local"
        if [[ ! -e $settings ]]; then
            (set -C; cat > "$settings" <<'EOF'
# Set these paths to files on this A3 host.
ASSETS=''       # Directory containing the matching config.json and tokenizer files
TRAIN_FILES=''  # Training Parquet
VAL_FILES=''    # Validation Parquet
# MODEL=''      # Optional output directory for the generated BF16 checkpoint
EOF
            )
        fi
        echo "Edit $settings, then use plan or run."
        exit 0 ;;
    plan|generate|config|train|run) ;;
    *) echo 'Choose init, plan, generate, config, train or run.' >&2; exit 2 ;;
esac
if [[ -f $settings ]]; then source "$settings"; fi
if [[ $action == run || $action == train || $action == config ]]; then
    export TRAIN_FILES=${TRAIN_FILES:?Set TRAIN_FILES to a local training parquet}
    export VAL_FILES=${VAL_FILES:?Set VAL_FILES to a local validation parquet}
    [[ -f $TRAIN_FILES && -r $TRAIN_FILES && -f $VAL_FILES && -r $VAL_FILES ]] || { echo 'Check both Parquet paths.' >&2; exit 2; }
fi
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
export RUN_DIR=${RUN_DIR:-$root/outputs/random-16e-a3-$(date +%Y%m%d-%H%M%S)-$$}
export STEP5_EP_DISPATCHER=custom_native_eager_fused_forward STEP5_USE_TRITON_MC2_PERMUTE=false
export TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=2
export MICRO_BATCH_SIZE=1 LOG_PROB_MICRO_BATCH_SIZE=1
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=128
export ACTOR_MAX_TOKEN_LEN_PER_GPU=1152 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=1152
export TOTAL_STEPS=1 TEST_FREQ=-1 SAVE_FREQ=-1 ROLLOUT_SLEEP_LEVEL=2
launch() {
    EP_SIZE=16 ROLLOUT_TP_SIZE=8 bash "$here/run_rl_rmpad_a3_fp32.sh" "$1"
}
if [[ $action == run ]]; then
    launch config
    launch train
else
    launch "$action"
fi
