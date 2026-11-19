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

`--no-deps` 只能跳过依赖解析，不能补齐运行依赖或证明兼容。完整四仓安装命令需要把这些例外落实到明确的依赖清单与导入检查；不能把保存基础版本或取完源码当作四仓已经安装成功。
这些是旧快照复现的限制，不应套用 `step5-up` 新六仓安装脚本。

你现在可以先完成本页四步；后续安装继续使用这里的网络配置和实际保存的基础版本。
