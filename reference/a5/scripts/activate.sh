# Source this file from Bash or zsh.
if [ -n "${ZSH_VERSION:-}" ]; then
    _step5_activation_file=${(%):-%x}
else
    _step5_activation_file=${BASH_SOURCE[0]}
fi
_step5_activate_project() {
    local project_dir runtime restore_nounset=0 status_code=0
    project_dir=$(CDPATH= cd -- "$(dirname -- "$1")/.." && pwd -P) || return 1
    runtime=/mnt/share/z00988734/environments/step5-cann920-post7-rl-521/activate.sh
    case $- in *u*) restore_nounset=1; set +u ;; esac
    # The shared runtime derives compiler caches from TMPDIR. Use node-local storage.
    export TMPDIR=/tmp
    . "$runtime" || status_code=$?
    if [ "$restore_nounset" = 1 ]; then set -u; fi
    [ "$status_code" = 0 ] || return "$status_code"
    export STEP5_R3_ROOT="$project_dir"
    export PYTHONPATH="$project_dir/scripts/python:$project_dir/src/FSDPTurbo:$project_dir/src/verl:$project_dir/src/vllm:$project_dir/src/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"
}
if _step5_activate_project "$_step5_activation_file"; then
    unset _step5_activation_file
    unset -f _step5_activate_project
else
    unset _step5_activation_file
    unset -f _step5_activate_project
    return 1
fi
