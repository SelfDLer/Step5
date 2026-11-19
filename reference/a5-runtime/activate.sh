export STEP5_HOST_ENV=/mnt/share/z00988734/environments/step5-cann920-post7-rl-521
export CANN_ENV_SCRIPT=/mnt/share/cann_install_path/cann920/cann-9.2.0/set_env.sh
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$STEP5_HOST_ENV/.venv" ]; then
    echo 'Deactivate the other Python venv first.' >&2; return 1
fi
# Allow a second activation from this installation's own runtime view.
case "${ASCEND_HOME_PATH:-}" in
    "$STEP5_HOST_ENV"/runtime-config/hybm/cann-*)
        if [ "$(readlink -f -- "$ASCEND_HOME_PATH/lib64")" = "$(readlink -f -- "${CANN_ENV_SCRIPT%/*}/lib64")" ]; then
            export ASCEND_HOME_PATH="${CANN_ENV_SCRIPT%/*}"
        fi ;;
esac
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ "$ASCEND_HOME_PATH/set_env.sh" != "$CANN_ENV_SCRIPT" ]; then
    echo 'Use a fresh shell without another CANN runtime.' >&2; return 1
fi
unset PYTHONPATH ASCEND_CUSTOM_OPP_PATH
. "$CANN_ENV_SCRIPT" || return 1
. "$STEP5_HOST_ENV/.venv/bin/activate" || return 1
export STEP5_PYTHON="$VIRTUAL_ENV/bin/python"
export STEP5_RE_ROOT="$STEP5_HOST_ENV/project"
_gbsa="$STEP5_RE_ROOT/runtime/gbsa-eab6333"
_fla="$VIRTUAL_ENV/lib/python3.12/site-packages/fla_npu/opp"
_plugin="$STEP5_RE_ROOT/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export PYTHONPATH="$_gbsa/python:${PYTHONPATH%:}"
export ASCEND_CUSTOM_OPP_PATH="$_plugin:$_fla/vendors/fla_npu_transformer:$_fla:$_gbsa/opp/vendors/step5_gbsa_eab6333_transformer"
export LD_LIBRARY_PATH="$STEP5_HOST_ENV/python-libs:$_gbsa/opp/vendors/step5_gbsa_eab6333_transformer/op_api/lib:${LD_LIBRARY_PATH}"
export PATH="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
_cache="${TMPDIR:-/tmp}/step5-post7-$(id -u)-$(printf %s "$STEP5_HOST_ENV" | cksum | cut -d' ' -f1)"
export XDG_CACHE_HOME="$_cache" TORCH_EXTENSIONS_DIR="$_cache/torch_extensions" TRITON_CACHE_DIR="$_cache/triton" VLLM_CACHE_ROOT="$_cache/vllm" RAY_TMPDIR="$_cache/ray"
export PYTHONNOUSERSITE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_STEP4_ENABLE_FA3=0
export HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
unset _cache _gbsa _fla _plugin

# step5-post7-rl-runtime-v1
if [ -f "$STEP5_HOST_ENV/project/runtime/batch-invariant-2.0.0/activate.sh" ]; then
    . "$STEP5_HOST_ENV/project/runtime/batch-invariant-2.0.0/activate.sh" || return 1
fi
. "$STEP5_HOST_ENV/runtime-config/hybm/activate-cann-view.sh" || return 1

# DeepEP MC2 runtime (ascend-deepep)
export SHMEM_ROOT=/mnt/share/z00988734/projects/step5-re/src/DeepEP/third-party/shmem/install/shmem
case ":${LD_LIBRARY_PATH:-}:" in
    *":$SHMEM_ROOT/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
