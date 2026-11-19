# Source from the Step5 repository root in Bash or zsh:
#   source scripts/install/network.sh
# Mirrors/proxy policy adapted from:
# https://github.com/SelfDLer/server-transfer/blob/main/files/install_nextqa_a3_npu_manual.sh
# Changes affect this shell only. No installation or persistent config writes.

export CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export ASCEND_PIP_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
export HUAWEI_PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
export PYTORCH_CPU_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
export MOONCAKE_INDEX_URL=https://mirrors.aliyun.com/pypi/web/simple
export PIP_TRUSTED_HOST="mirror.sjtu.edu.cn s3.jcloud.sjtu.edu.cn download.pytorch.org download-r2.pytorch.org mirrors.huaweicloud.com mirrors.tuna.tsinghua.edu.cn mirrors.aliyun.com ascend.devcloud.huaweicloud.com"

# Resolve the currently selected Conda at invocation time. A subshell keeps the
# temporary variable and Python's context override out of the caller's shell.
conda_with_shell_proxy() (
    if ! command -v conda >/dev/null 2>&1; then
        printf '%s\n' 'Conda is unavailable; load your intended Conda first.' >&2
        return 1
    fi
    step5_conda_base=$(conda info --base) || return
    if [ ! -x "$step5_conda_base/bin/python" ]; then
        printf '%s\n' 'The selected Conda has no executable bin/python.' >&2
        return 1
    fi
    "$step5_conda_base/bin/python" -c '
import os
import sys

if __name__ == "__main__":
    proxy_names = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
    use_shell_proxy = any(name in os.environ or name.upper() in os.environ for name in proxy_names)
    if use_shell_proxy:
        for name in proxy_names:
            # Lowercase wins, including an explicitly empty value.
            value = os.environ.get(name, os.environ.get(name.upper()))
            if value is not None:
                os.environ[name] = os.environ[name.upper()] = value
        from conda.base.context import context
        context.proxy_servers = {}

    from conda.cli.main import main
    sys.exit(main(*sys.argv[1:]))
' "$@"
)
