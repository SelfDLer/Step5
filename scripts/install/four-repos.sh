#!/usr/bin/env bash
# Run one stage at a time with bash from a Step5 checkout.
set -euo pipefail
case ${1:-} in
    tools|deps|triton|arctic|vllm|ascend|fsdp|verl|verify) stage=$1 ;;
    *) echo 'Usage: bash scripts/install/four-repos.sh {tools|deps|triton|arctic|vllm|ascend|fsdp|verl|verify}' >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$root"
work="$root/.local/a3-install"
pins="$work/verified-base.constraints.txt"
: "${ASCEND_HOME_PATH:?Source your verified CANN environment first}"
: "${PIP_INDEX_URL:?Source scripts/install/network.sh first}"
: "${ASCEND_PIP_INDEX_URL:?Source scripts/install/network.sh first}"
: "${PYTORCH_CPU_INDEX_URL:?Source scripts/install/network.sh first}"
check_base() {
    python - "$pins" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path
expected = Path('/mnt/share/z00988734/.conda/step5').resolve()
if Path(sys.prefix).resolve() != expected or sys.version_info[:2] != (3, 12):
    sys.exit(f'Activate {expected} first (current Python: {sys.executable})')
pins = dict(line.split('==', 1) for line in Path(sys.argv[1]).read_text().splitlines())
for name in ('torch', 'torch-npu', 'numpy'):
    actual = version(name)
    if actual != pins[name]:
        sys.exit(f'{name}: recorded {pins[name]}, now {actual}')
if pins['torch'].split('+')[0] != '2.10.0':
    sys.exit('The pinned vllm-ascend source requires torch==2.10.0')
PY
}
check_base
for repo in vllm vllm-ascend FSDPTurbo verl; do
    expected=$(git rev-parse "HEAD:src/$repo")
    actual=$(git -C "src/$repo" rev-parse HEAD)
    [[ $actual == "$expected" ]] || { echo "Unexpected checkout: src/$repo" >&2; exit 1; }
done
exec > >(tee "$work/$stage-$(date +%Y%m%d-%H%M%S).log") 2>&1
pip_install=(python -m pip install -c "$pins")
editable=("${pip_install[@]}" --no-build-isolation --no-deps -v -e)
export MAX_JOBS=${MAX_JOBS:-4}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}
case $stage in
tools)
    "${pip_install[@]}" 'cmake>=3.26.1,<4' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81' \
        'setuptools-scm>=8' 'setuptools-rust>=1.9' wheel jinja2 pybind11 cffi pathlib2
    for tool in gcc g++ make cmake ninja pkg-config perl; do command -v "$tool"; done
    (cd src/vllm && rustc --version && cargo --version)
    ;;
deps)
    # Resolve the two CPU packages against the CPU index first.
    "${pip_install[@]}" --index-url "$PYTORCH_CPU_INDEX_URL" --extra-index-url "$PIP_INDEX_URL" \
        'torchvision==0.25.0+cpu' 'torchaudio==2.10.0+cpu'
    # Arctic has its own build stage. Use the exact remaining Ascend requirements.
    sed '/^[[:space:]]*arctic-inference==/d' src/vllm-ascend/requirements.txt > "$work/ascend-runtime.txt"
    "${pip_install[@]}" --extra-index-url "$ASCEND_PIP_INDEX_URL" \
        -r src/vllm/requirements/common.txt -r "$work/ascend-runtime.txt" \
        'opencv-python-headless==4.11.0.86' 'transformers==5.14.1' \
        accelerate codetiming datasets dill hydra-core peft 'pyarrow>=19' pylatexenc \
        'ray[default]==2.48.0' torchdata 'tensordict==0.10.0' wandb tensorboard \
        'TransferQueue==0.1.11' concurrent-log-handler attrs googleapis-common-protos \
        pyelftools pytest-xdist expecttest
    ;;
triton)
    # Overlay the Ascend implementation after resolving its dependencies. Match
    # the wheel to this host and keep any version already supplied on A3.
    spec=${STEP5_TRITON_ASCEND_SPEC:-}
    if [[ -z $spec ]]; then
        spec=$(python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
try:
    print('triton-ascend==' + version('triton-ascend'))
except PackageNotFoundError:
    print('triton-ascend==3.6.0.post2')
PY
)
    fi
    dir=$(mktemp -d "$work/triton.XXXXXX")
    python -m pip download --no-deps --only-binary=:all: --extra-index-url "$ASCEND_PIP_INDEX_URL" -d "$dir" "$spec"
    wheels=("$dir"/triton_ascend-*.whl)
    [[ -f ${wheels[0]} && ${#wheels[@]} -eq 1 ]]
    # Resolve dependencies first, then restore ownership of overlapping files.
    python - "${wheels[0]}" "$work/triton-runtime.txt" <<'PY'
from email.parser import BytesParser
from importlib.metadata import version
from pathlib import Path
import sys
from zipfile import ZipFile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
with ZipFile(sys.argv[1]) as whl:
    metadata, = (p for p in whl.namelist() if p.endswith('.dist-info/METADATA'))
    headers = BytesParser().parsebytes(whl.read(metadata))
requirements = []
for line in headers.get_all('Requires-Dist', []):
    req = Requirement(line)
    if canonicalize_name(req.name) == 'numpy':
        if version('numpy') not in req.specifier:
            print(f'Keeping verified numpy {version("numpy")}; wheel declares {req}', flush=True)
    else:
        requirements.append(line)
Path(sys.argv[2]).write_text('\n'.join(requirements) + '\n')
PY
    "${pip_install[@]}" --extra-index-url "$ASCEND_PIP_INDEX_URL" -r "$work/triton-runtime.txt"
    "${pip_install[@]}" --no-deps --force-reinstall "${wheels[0]}"
    python -c 'import triton, triton.backends.ascend; print(triton.__file__, triton.backends.ascend.__path__)'
    ;;
arctic)
    overlay=$(mktemp -d "$work/arctic-build.XXXXXX")
    python -m pip install --target "$overlay" 'nanobind==2.9.2' 'protobuf==5.29.5' 'grpcio-tools<1.72' 'setuptools==80.10.2'
    PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" "${pip_install[@]}" --no-build-isolation --no-deps 'arctic-inference==0.1.1'
    ;;
vllm)
    unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST
    VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1 "${editable[@]}" "$root/src/vllm"
    python -c 'import vllm._rust_tool_parser as p; print(p.__file__)'
    ;;
ascend)
    if [[ -n ${SOC_VERSION:-} && $SOC_VERSION != ascend910_93* ]]; then
        echo "Check SOC_VERSION for this A3: $SOC_VERSION" >&2; exit 1
    fi
    git -C src/vllm-ascend submodule update --init csrc/third_party/catlass
    gitlink=$(git -C src/vllm-ascend rev-parse HEAD:csrc/third_party/catlass)
    [[ $(git -C src/vllm-ascend/csrc/third_party/catlass rev-parse HEAD) == "$gitlink" ]]
    # Redirect the upstream build script's git config --global write.
    touch "$work/build.gitconfig"
    build_path=${PYTHONPATH:-}
    if python -c 'import numpy; raise SystemExit(int(numpy.__version__.split(".")[0]) < 2)'; then
        overlay=$(mktemp -d "$work/ascend-numpy.XXXXXX")
        python -m pip install --no-deps --target "$overlay" 'numpy==1.26.4'
        build_path="$overlay${PYTHONPATH:+:$PYTHONPATH}"
    fi
    PYTHONPATH="$build_path" COMPILE_CUSTOM_KERNELS=1 GIT_CONFIG_GLOBAL="$work/build.gitconfig" \
        "${editable[@]}" "$root/src/vllm-ascend"
    python -c 'import torch_npu, vllm_ascend.vllm_ascend_C as ext; from vllm_ascend._build_info import __device_type__ as soc; print(ext.__file__, soc); assert soc == "A3", soc'
    ;;
fsdp) "${editable[@]}" "$root/src/FSDPTurbo" ;;
verl) "${editable[@]}" "$root/src/verl" ;;
verify)
    export STEP5_EP_DISPATCHER=custom_native_eager_fused_forward
    python - <<'PY'
import importlib
from pathlib import Path
import torch_npu
from vllm.platforms import current_platform
if current_platform.device_type != 'npu':
    raise RuntimeError(f'vLLM selected {current_platform.device_type}')
for repo, module in [('vllm','vllm._rust_tool_parser'),
                     ('vllm-ascend','vllm_ascend.vllm_ascend_C'),
                     ('FSDPTurbo','fsdp_turbo.models.step5.modeling_step5'),
                     ('verl','verl.trainer.main_ppo')]:
    path = Path(importlib.import_module(module).__file__).resolve()
    print(module, path)
    if not path.is_relative_to((Path('src') / repo).resolve()):
        raise RuntimeError(f'Unexpected module path: {path}')
PY
    python scripts/checks/02_check_torch_npu.py --expected-prefix "$CONDA_PREFIX" --device 0
    python scripts/checks/03_check_gbsa.py --expected-prefix "$CONDA_PREFIX" --device 0 --lengths 16 129 --kv-heads 4
    ;;
esac
check_base
printf '%s completed\n' "$stage"
