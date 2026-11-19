# 基础检查通过后：网络与四仓安装准备

适用于 `a5toa3` 的旧 Step5 四仓迁移。CANN 9.3.0 及基础 NPU、GBSA/GBSAG 检查通过后，保留已验证环境，逐段执行下面命令。
这一步准备网络、保存版本和获取源码，不安装四仓或 DeepEP。完整模型和分布式精度仍需后续验收。

## 1. 更新资料并加载网络配置

在已经克隆的 Step5 根目录执行；若有本地修改，先保留自己的修改再更新。

```sh
git switch a5toa3
git pull --ff-only
source scripts/install/network.sh
```

网络设置来自用户指定的 [install_nextqa_a3_npu_manual.sh](https://github.com/SelfDLer/server-transfer/blob/main/files/install_nextqa_a3_npu_manual.sh)，只提取镜像地址及 Conda 的 shell 代理优先逻辑。

| 用途 | 变量 | 地址 |
|---|---|---|
| Conda conda-forge | `CONDA_CHANNEL` | `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge` |
| pip 默认源 | `PIP_INDEX_URL` | `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` |
| Ascend 专用包 | `ASCEND_PIP_INDEX_URL` | `https://mirrors.huaweicloud.com/ascend/repos/pypi` |
| 华为通用备选源 | `HUAWEI_PIP_INDEX_URL` | `https://mirrors.huaweicloud.com/repository/pypi/simple` |
| PyTorch CPU wheel | `PYTORCH_CPU_INDEX_URL` | `https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/` |
| 原脚本 Mooncake 源，备用 | `MOONCAKE_INDEX_URL` | `https://mirrors.aliyun.com/pypi/web/simple` |

同时沿用原脚本的 `PIP_TRUSTED_HOST` 列表。只有 `PIP_INDEX_URL` 和 `PIP_TRUSTED_HOST` 会被 pip 自动识别；其他源变量在需要时通过 `--index-url` / `--extra-index-url` 显式使用。不会把所有源同时加入 pip，也不会清除你已有的 `PIP_EXTRA_INDEX_URL`。

该文件可重复 source，兼容 Bash/zsh，不更改 PATH、Python/CANN 激活、`.bashrc`、`.zshrc`、`.condarc` 或 pip 配置文件。新终端需重新加载。没有复制 NextQA 的包版本、安装过程或持久化环境配置。

代理沿用 A3 当前终端已有的 `http_proxy` / `https_proxy` / `all_proxy` / `no_proxy`（或大写变量）。不假定 A3 存在某个代理端口，也不写入 A5 或 Windows 的代理地址。
`conda_with_shell_proxy` 只在单次 Conda 子进程中统一代理变量大小写，小写优先、空值也保留，并覆盖该进程的 Conda 代理配置；未设置任何代理变量时保留 Conda 原有配置。
普通 `conda` 命令不会自动经过这个函数。pip/Git 保持自身的代理处理规则。

## 2. 激活现有环境，检查包源

```sh
conda activate /mnt/share/z00988734/.conda/step5
export CANN_ENV_SCRIPT=/mnt/share/q30068470/step-1026/ops_1105/ascend-toolkit/set_env.sh
source "$CANN_ENV_SCRIPT"
source scripts/install/network.sh
python -c 'import sys; print(sys.executable); print(sys.prefix)'
python -m pip index versions setuptools
```

若 Conda 尚未初始化，先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。CANN 路径以你已经通过检查时实际使用的路径为准。
`pip index versions` 查询清华源，不安装或升级任何包。清华访问失败时可单独测试备选源：

```sh
python -m pip index versions setuptools --index-url "$HUAWEI_PIP_INDEX_URL"
```

成功后若需要本终端改用华为默认源，可显式执行 `export PIP_INDEX_URL="$HUAWEI_PIP_INDEX_URL"`；再次 source 网络脚本会恢复清华默认源。

Conda 源的只读查询（环境已经存在，不再 create）：

```sh
conda_with_shell_proxy search --override-channels -c "$CONDA_CHANNEL" 'python=3.12'
```

Conda/pip 镜像不代理 GitHub 或 GitCode。四仓源码仍需要能访问其托管地址；FSDPTurbo 按指定地址使用 GitCode。

## 3. 保存已验证的基础版本

下面仅在当前 Conda 环境内读取版本，在仓库的忽略目录 `.local/a3-install/` 保存本地记录。不要把记录加入 Git。

```sh
python - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys

expected = Path('/mnt/share/z00988734/.conda/step5').resolve()
assert Path(sys.prefix).resolve() == expected, (sys.prefix, str(expected))
assert sys.version_info[:2] == (3, 12), sys.version
pins = {name: md.version(name) for name in ('torch', 'torch-npu', 'numpy')}
folder = Path('.local/a3-install')
folder.mkdir(parents=True, exist_ok=True)
target = folder / 'verified-base.constraints.txt'
content = ''.join(f'{name}=={version}\n' for name, version in pins.items())
if target.exists() and target.read_text() != content:
    raise SystemExit('基础版本已改变，停止；请先核对原记录，勿覆盖。')
target.write_text(content)
snapshot = folder / 'packages-before-four-repos.json'
if not snapshot.exists():
    packages = sorted(
        ({'name': d.metadata['Name'], 'version': d.version} for d in md.distributions()),
        key=lambda d: (d['name'] or '').lower(),
    )
    snapshot.write_text(json.dumps(packages, indent=2) + '\n')
print(content, end='')
print('基础版本已记录：', target.resolve())
PY
```

后续普通 pip 依赖安装应显式加 `-c "$PWD/.local/a3-install/verified-base.constraints.txt"`（在仓库根目录执行）。约束冲突应停下核对；约束文件本身不会安装任何包，也不会自动约束所有未来命令。
构建隔离可能另建环境，因此源码构建还需要准备构建依赖并明确使用 `--no-build-isolation`。

## 4. 拉取四仓固定源码

```sh
git submodule update --init src/verl src/FSDPTurbo src/vllm src/vllm-ascend
git submodule status
```

SHA 应分别以 `db76ca6`、`14bffd5`、`144a421`、`bd4a6ce` 开头（按仓库名匹配，不依赖输出顺序）。正常状态前面是空格；`-` 表示未初始化，`+` 表示当前检出与固定提交不符，`U` 表示冲突。
不使用 `--remote`，不切换到最新 main，不拉取 DeepEP。

## 后续源码安装的依赖边界

下一阶段顺序为：确认构建工具和运行依赖 → vLLM（`VLLM_TARGET_DEVICE=empty`）→ vLLM Ascend（在 A3 本机编译原生扩展）→ FSDPTurbo → verl → 重跑基础检查、模块导入和模型冒烟。
现阶段不要直接对四仓逐个执行普通 `pip install -e .`，以下是固定源码中已核对的冲突：

- vLLM Ascend 的构建依赖仍含 `torch_npu==2.10.0.post4` 等旧 pin。不能让默认隔离构建替换已验证的版本；准备构建依赖后需用 `--no-build-isolation`。
- verl 要求 `numpy>=2.0.0`、`transformers>=5.5.3,!=5.6.0,<5.13`；原 A5 记录为 NumPy 1.26.4、Transformers 5.14.1，且固定 vLLM Ascend 要求 Transformers 5.14.1。因此这组原始元数据不能同时由普通 pip 求解满足。
- FSDPTurbo 同时声明 `triton` 和 `triton_ascend`，需要明确 Ascend 实现及实际导入来源，避免安装过程中覆盖同名模块。

## 5. 安装构建工具和运行依赖

前四步完成后，在同一个已激活的 A3 终端进入 Step5 根目录，执行 `git pull --ff-only`。如果换过终端，先按第 2 节恢复 Conda、CANN、网络配置。若机器另行提供与 CANN 配套的 ATB，按其实际位置加载 ATB 的 `set_env.sh`。

下面每次运行一行，上一步显示对应的 `completed` 后再继续。日志保存在 `.local/a3-install/`。脚本会检查当前 Python、四个源码提交，以及准备阶段保存的 torch、torch-npu、NumPy 版本。

```sh
bash scripts/install/four-repos.sh tools
bash scripts/install/four-repos.sh deps
bash scripts/install/four-repos.sh triton
bash scripts/install/four-repos.sh arctic
```

- `tools` 安装 Python 构建工具，并检查系统编译工具及 Rust。vLLM 的 `rust-toolchain.toml` 指定 Rust 1.95。已有 rustup 时，可运行 `rustup toolchain install 1.95 --profile minimal`。若机器用 Conda 提供 Rust，可在当前环境运行 `conda_with_shell_proxy install -p "$CONDA_PREFIX" --freeze-installed --override-channels -c "$CONDA_CHANNEL" 'rust=1.95.*'`，查看事务后再确认。C/C++ 编译器按目标 CANN 工具链配套。
- `deps` 先装配套的 CPU torchvision/torchaudio，再按四仓依赖安装普通 Python 包。沿用旧环境的 Transformers 5.14.1、Ray 2.48.0、TransferQueue 0.1.11 和 TensorDict 0.10.0。其余间接依赖由 A3 的 pip 解析。镜像若缺指定包，保留报错中的包名和版本，使用提供方对应的 A3 包源或 wheel。
- `triton` 优先复用当前环境记录的 triton-ascend 版本。若尚未安装，采用 3.6.0.post2 作为候选，供 FSDPTurbo 的 Triton 3.6 路径测试；这个组合仍需在 CANN 9.3.0 上验证。有配套 wheel 时可先执行 `export STEP5_TRITON_ASCEND_SPEC=/绝对路径/triton_ascend-实际文件名.whl`。脚本按 wheel 声明安装依赖，保留已验证的 NumPy，最后由 Ascend wheel 提供 Triton 模块。如果 wheel 声明的 NumPy 与现有版本不同，脚本打印差异，后续应检查真实模型上的 Triton kernel。
- `arctic` 用临时目录准备 Arctic 0.1.1 的构建依赖，在当前 torch 上构建，避免更改运行时 protobuf。

### deps 报 numpy==2.5.5 与 mistral-common 冲突

Python 3.12 上，本次解析到的 mistral-common 要求 NumPy 小于 2.4，OpenCV 4.11 要求至少 1.26.0。此时可选用原 A5 环境的 1.26.4。保持已验证的 Conda、CANN 和网络环境，选好之前测试用的空闲卡，在仓库根目录逐行执行。

```sh
bash scripts/install/recheck-numpy.sh
bash scripts/install/four-repos.sh deps
```

第一条会备份原约束，仅安装 NumPy 的二进制 wheel，在逻辑卡 0 复跑基础 NPU 和 GBSA/GBSAG 检查。两项都通过后才更新 NumPy 约束并打印 `NumPy 1.26.4 verified`。如果检查失败，当前环境里可能已装入 1.26.4，约束文件仍保留原值。可处理报错后重跑第一条，或用 `python -m pip install --only-binary=:all: --no-deps 'numpy==2.5.5'` 恢复本次修改前的版本。

## 6. 逐仓安装

```sh
bash scripts/install/four-repos.sh vllm
bash scripts/install/four-repos.sh ascend
bash scripts/install/four-repos.sh fsdp
bash scripts/install/four-repos.sh verl
```

vLLM 使用 `VLLM_TARGET_DEVICE=empty` 并构建 Rust 解析器。vLLM Ascend 从 GitCode 拉取固定 catlass，用本机 A3 和 CANN 编译插件扩展及仓库自带的 ACLNN 算子。其产物安装在当前 vLLM Ascend 源码树中。保留之前验证过的 GBSA/GBSAG wrapper 和算子环境。若芯片自动识别失败，请按本机板卡确认 `SOC_VERSION` 后重跑 `ascend`。默认构建并发为 4，内存不足时可执行 `MAX_JOBS=2 bash scripts/install/four-repos.sh ascend`。

四仓都按源码 editable 安装，使用 `--no-deps --no-build-isolation`。需要从目标 A3 下载依赖并本机编译。镜像可达性、工具链和这组版本仍以目标机执行结果为准。

## 7. 验证安装结果

保持已验证的空闲卡可见，逻辑设备号为 0。在当前终端执行：

```sh
export STEP5_EP_DISPATCHER=custom_native_eager_fused_forward
bash scripts/install/four-repos.sh verify
python -m pip check
```

`verify` 检查 vLLM 选中 NPU、四仓模块实际来源、Rust 和 Ascend 原生扩展、FSDPTurbo Step5 模型及 verl PPO 入口，并复跑基础 NPU 和 GBSA/GBSAG 数值检查。看到 `verify completed` 后再进入带权重的单批精度验证。`pip check` 会列出元数据冲突，核对前述 verl/Transformers 及 NumPy 例外；其他报错需逐条定位。某阶段失败可修复后只重跑该阶段。完整模型使用的 FLA、batch-invariant 等可选实现应按实际启用的后端在 A3 验收。
